import argparse
import ast
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from cuebridge_eval_utils import (
    CUEBRIDGE_ARTIFACT_STEPS,
    add_cuebridge_compare_args,
    build_cuebridge_artifacts,
    cuebridge_compare_mode,
    print_cuebridge_delta_summary,
)

# Suppress NumPy/Pandas compatibility warnings and general python warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

RESULTS_DIR = str(Path(__file__).resolve().parents[1] / "results")
EVIDENCE_MATCHER = "sentence_or_chunk_overlap_v1"
MIN_EVIDENCE_CHARS = 40
MIN_EVIDENCE_TOKENS = 6
try:
    from cuebridge_sdk import CueBridgeClient
except Exception:
    sdk_path = Path(__file__).resolve().parents[1] / "cuebridge" / "sdk" / "python"
    if str(sdk_path) not in sys.path:
        sys.path.insert(0, str(sdk_path))
    try:
        from cuebridge_sdk import CueBridgeClient
    except Exception:
        CueBridgeClient = None


def clean_output(text: str) -> str:
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def normalize_text(text: str) -> str:
    text = text.replace("assistant: ", "").replace("user: ", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def approx_token_count(text: str) -> int:
    """Cheap model-agnostic estimate for the retrieved context footprint."""
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def context_token_count(contents: list[str], k: int | None = None) -> int:
    selected = contents if k is None else contents[:k]
    return sum(approx_token_count(content) for content in selected)


def percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    sorted_values = sorted(values)
    idx = math.ceil((p / 100.0) * len(sorted_values)) - 1
    idx = max(0, min(idx, len(sorted_values) - 1))
    return sorted_values[idx]


def extract_json_object(text: str) -> dict:
    text = clean_output(text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in status output: {text!r}")
    return json.loads(text[start : end + 1])


def resolve_cuemap_command(cmd: list[str]) -> list[str]:
    """Use the launcher-selected release binary for CLI fallbacks."""
    binary = os.environ.get("CUEMAP_RUST_BIN")
    if binary and cmd and Path(cmd[0]).name == "cuemap":
        return [binary, *cmd[1:]]
    return cmd


def run_cmd(cmd: list[str], *, check: bool = False) -> subprocess.CompletedProcess:
    resolved_cmd = resolve_cuemap_command(cmd)
    result = subprocess.run(resolved_cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(resolved_cmd)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def make_cuebridge_client(args):
    if not getattr(args, "cuebridge_observe", False):
        return None
    if CueBridgeClient is None:
        print("WARNING: CueBridge observation disabled; could not import cuebridge_sdk.")
        args.cuebridge_observe = False
        return None
    return CueBridgeClient(
        project_id=args.cuebridge_observe_project or f"beam_{args.context}_eval",
        endpoint=args.cuebridge_agent_url,
        key=args.cuebridge_api_key,
        batch_size=args.cuebridge_observe_batch_size,
        flush_interval=args.cuebridge_observe_flush_interval,
    )


def post_json(url: str, endpoint: str, project_id: str, payload: dict, *, timeout: int = 30) -> dict:
    last_error = None
    body = json.dumps(payload).encode("utf-8")
    for base_url in status_urls(url):
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}{endpoint}",
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Project-ID": project_id,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"POST {request.full_url} failed with HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            last_error = f"POST {request.full_url} failed: {exc}"
    raise RuntimeError(last_error or f"POST {url.rstrip()}{endpoint} failed")


def cuebridge_text_result_id(content: str) -> str:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return f"text_sha256:{digest}"


def cuebridge_recall_response(recalled_contents: list[str]) -> dict:
    return {"results": [{"content": content} for content in recalled_contents]}


def cuebridge_eval_labels(
    *,
    recalled_contents: list[str],
    expected_contexts: list[str],
    hit_rank: int,
    project_id: str,
    q_type: str,
    name: str,
) -> dict:
    target_result_id = None
    if hit_rank > 0 and hit_rank <= len(recalled_contents):
        target_result_id = cuebridge_text_result_id(recalled_contents[hit_rank - 1])
    elif expected_contexts:
        target_result_id = cuebridge_text_result_id(expected_contexts[0])
    return {
        "target_result_id": target_result_id,
        "top_result_id": cuebridge_text_result_id(recalled_contents[0]) if recalled_contents else None,
        "source_eval": "beam",
        "cuemap_project_id": project_id,
        "question_category": q_type,
        "recall_attempt_name": name,
    }


def cuebridge_send_ingest(args, *, project_id: str, record_idx: int, turn_idx: int, content: str, metadata: dict) -> None:
    if getattr(args, "cuebridge_client", None) is None or not getattr(args, "cuebridge_observe_ingest", True):
        return
    args.cuebridge_client.send_ingest(
        {
            "id": cuebridge_text_result_id(content),
            "content": content,
            "metadata": metadata,
        },
        source_eval="beam",
        cuemap_project_id=project_id,
        record_idx=record_idx,
        turn_idx=turn_idx,
    )


def cue_value(value) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower()).strip("_")
    return normalized or "unknown"


def status_urls(url: str) -> list[str]:
    parsed = urllib.parse.urlparse(url)
    urls = [url.rstrip("/")]
    if parsed.hostname == "localhost":
        port = f":{parsed.port}" if parsed.port else ""
        replacement = urllib.parse.urlunparse((
            parsed.scheme,
            f"127.0.0.1{port}",
            parsed.path.rstrip("/"),
            "",
            "",
            "",
        ))
        urls.append(replacement.rstrip("/"))
    return list(dict.fromkeys(urls))


def job_status_via_http(project_id: str, url: str) -> dict:
    last_error = None
    for base_url in status_urls(url):
        endpoint = f"{base_url}/jobs/status"
        request = urllib.request.Request(endpoint, headers={"X-Project-ID": project_id})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GET {endpoint} failed with HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            last_error = f"GET {endpoint} failed: {exc}"
    raise RuntimeError(last_error or f"GET {url.rstrip('/')}/jobs/status failed")


def job_status_via_cli(project_id: str, url: str) -> dict:
    commands = [
        ["cuemap", "status", "--jobs", "-p", project_id, "--url", url, "--json"],
        ["cuemap", "status", "--jobs", "-p", project_id, "--url", url],
    ]
    errors = []
    for cmd in commands:
        res = run_cmd(cmd)
        if res.returncode != 0:
            errors.append(f"{' '.join(cmd)} exited {res.returncode}: {res.stderr.strip()}")
            continue
        try:
            return extract_json_object(res.stdout)
        except ValueError as exc:
            errors.append(str(exc))
    raise RuntimeError("; ".join(errors))


def job_status(project_id: str, url: str) -> dict:
    try:
        return job_status_via_http(project_id, url)
    except RuntimeError as http_error:
        try:
            return job_status_via_cli(project_id, url)
        except RuntimeError as cli_error:
            raise RuntimeError(f"{http_error}; CLI fallback failed: {cli_error}") from cli_error


def intent_jobs_done(status: dict) -> bool:
    if status.get("intent_ready") is not True:
        return False
    memory_total = int(status.get("intent_memory_total", 0))
    annotated = int(status.get("intent_annotated", 0))
    missing = int(status.get("intent_missing", 0))
    intent_total = int(status.get("intent_total", 0))
    intent_completed = int(status.get("intent_completed", 0))
    intent_failed = int(status.get("intent_failed", 0))
    return (
        missing == 0
        and annotated >= memory_total
        and intent_failed == 0
        and (intent_total == 0 or intent_completed >= intent_total)
    )


def jobs_done(status: dict) -> bool:
    pairs = [
        ("writes_completed", "writes_total"),
    ]
    return all(
        int(status.get(done, 0)) >= int(status.get(total, 0)) for done, total in pairs
    ) and intent_jobs_done(status)


def progress_counts(status: dict) -> str:
    return (
        f"w {status.get('writes_completed', 0)}/{status.get('writes_total', 0)} "
        f"j {status.get('intent_completed', 0)}/{status.get('intent_total', 0)} "
        f"c {status.get('intent_annotated', 0)}/{status.get('intent_memory_total', 0)}"
    )


def wait_for_bg_jobs(project_id: str, url: str, timeout_seconds: int, poll_seconds: float) -> dict:
    print("\nWaiting for background jobs to settle", end="", flush=True)
    start = time.time()
    last_status = {}
    while time.time() - start < timeout_seconds:
        try:
            last_status = job_status(project_id, url)
        except RuntimeError as exc:
            if time.time() - start < min(timeout_seconds, 30):
                print(f"\rWaiting for background jobs to settle status unavailable, retrying: {exc}", end="", flush=True)
                time.sleep(poll_seconds)
                continue
            raise
        counts = progress_counts(last_status)
        if jobs_done(last_status):
            print(
                f"\r\033[KWaiting for background jobs to settle "
                f"phase={last_status.get('phase', 'unknown')} {counts} Done!"
            )
            return last_status
        phase = last_status.get("phase", "unknown")
        print(f"\r\033[KWaiting for background jobs to settle phase={phase} {counts}", end="", flush=True)
        time.sleep(poll_seconds)
    raise TimeoutError(f"Timed out waiting for background jobs on {project_id}: {last_status}")


def evidence_units(text: str) -> list[str]:
    """Return meaningful sentence/line units for expanded-context matching."""
    return _evidence_units_from_text(text)


def _evidence_units_from_text(text: str) -> list[str]:
    units = []
    seen = set()
    for raw_unit in re.split(r"\r?\n+|(?<=[.!?])\s+", text):
        unit = normalize_text(raw_unit).strip(" -*#\t")
        token_count = len(re.findall(r"[a-z0-9]+", unit))
        if (
            len(unit) >= MIN_EVIDENCE_CHARS
            and token_count >= MIN_EVIDENCE_TOKENS
            and unit not in seen
        ):
            units.append(unit)
            seen.add(unit)
    return units


def evidence_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", normalize_text(text)))


@dataclass(frozen=True)
class EvidenceUnitProfile:
    text: str
    tokens: frozenset[str]


@dataclass(frozen=True)
class EvidenceProfile:
    normalized: str
    tokens: frozenset[str]
    units: tuple[EvidenceUnitProfile, ...]


def _evidence_profile(text: str) -> EvidenceProfile:
    normalized = normalize_text(text)
    tokens = frozenset(re.findall(r"[a-z0-9]+", normalized))
    units = tuple(
        EvidenceUnitProfile(
            text=unit,
            tokens=frozenset(re.findall(r"[a-z0-9]+", unit)),
        )
        for unit in _evidence_units_from_text(text)
    )
    return EvidenceProfile(normalized=normalized, tokens=tokens, units=units)


def _profiles_match(expected: EvidenceProfile, returned: EvidenceProfile) -> bool:
    """Match whole memories and expanded neighboring evidence monotonically.

    A logical-block result may be one sentence while the BEAM gold context is
    the complete source message. Conversely, expansion may concatenate several
    neighboring chunks, making the returned text larger than the gold message.
    Whole-text substring matching handles only one of those cases. Sentence/
    line-unit matching lets both representations receive credit without
    accepting tiny generic fragments.
    """
    expected_norm = expected.normalized
    returned_norm = returned.normalized
    if not expected_norm or not returned_norm:
        return False

    if expected_norm in returned_norm:
        return True

    returned_token_count = len(returned.tokens)
    if (
        returned_norm in expected_norm
        and len(returned_norm) >= MIN_EVIDENCE_CHARS
        and returned_token_count >= MIN_EVIDENCE_TOKENS
    ):
        return True

    expected_units = expected.units
    if not expected_units:
        return False

    for unit in expected_units:
        if unit.text in returned_norm:
            return True

    # Handle small punctuation/whitespace differences at a sentence boundary.
    # Compare against returned units using containment of the shorter unit.
    returned_units = returned.units
    for expected_unit in expected_units:
        expected_unit_tokens = expected_unit.tokens
        if len(expected_unit_tokens) < MIN_EVIDENCE_TOKENS:
            continue
        for returned_unit in returned_units:
            returned_unit_tokens = returned_unit.tokens
            if len(returned_unit_tokens) < MIN_EVIDENCE_TOKENS:
                continue
            smaller = min(len(expected_unit_tokens), len(returned_unit_tokens))
            overlap = len(expected_unit_tokens & returned_unit_tokens) / smaller
            if overlap >= 0.82:
                return True

    return False


def matches_expected(expected: str, returned: str) -> bool:
    """Match one expected evidence against one returned result."""
    return _profiles_match(_evidence_profile(expected), _evidence_profile(returned))


def build_evidence_match_matrix(
    expected_contexts: list[str], recalled_contents: list[str]
) -> list[set[int]]:
    """Compute expected-evidence matches once for every returned result.

    The benchmark derives several metrics from the same ranked results. Keeping
    this matrix avoids repeatedly normalizing, tokenizing, and splitting the
    same expanded result for Hit@K, Recall@K, and NDCG calculations.
    """
    expected_profiles = [_evidence_profile(expected) for expected in expected_contexts]
    returned_profiles = []
    profile_cache: dict[str, EvidenceProfile] = {}
    for content in recalled_contents:
        profile = profile_cache.get(content)
        if profile is None:
            profile = _evidence_profile(content)
            profile_cache[content] = profile
        returned_profiles.append(profile)

    return [
        {
            expected_idx
            for expected_idx, expected_profile in enumerate(expected_profiles)
            if _profiles_match(expected_profile, returned_profile)
        }
        for returned_profile in returned_profiles
    ]


def calc_match(
    expected_contexts: list[str],
    recalled_contents: list[str],
    match_matrix: list[set[int]] | None = None,
) -> tuple[int, list[int]]:
    if match_matrix is None:
        match_matrix = build_evidence_match_matrix(expected_contexts, recalled_contents)

    hit_rank = -1
    rel_array = [0] * len(recalled_contents)
    matched_expected = set()

    for rank, matched_at_rank in enumerate(match_matrix):
        # A repeated chunk or neighboring result may match the same gold
        # context again. Count each gold context only at its first rank so
        # duplicate results cannot inflate NDCG above 1.0.
        new_matches = matched_at_rank - matched_expected
        if new_matches:
            rel_array[rank] = 1
            matched_expected.update(new_matches)
            if hit_rank == -1:
                hit_rank = rank + 1

    return hit_rank, rel_array


def calc_recall_for_contents(
    expected_contexts: list[str],
    recalled_contents: list[str],
    k: int,
    match_matrix: list[set[int]] | None = None,
) -> tuple[float, float]:
    if match_matrix is None:
        match_matrix = build_evidence_match_matrix(expected_contexts, recalled_contents)
    found = set().union(*(matches for matches in match_matrix[:k])) if k else set()
    total_expected = max(1, len(expected_contexts))
    frac = len(found) / total_expected
    all_found = 1.0 if len(found) == len(expected_contexts) else 0.0
    return frac, all_found


def calc_ndcg_for_rel(rel_array: list[int], expected_count: int, k: int) -> float:
    dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(rel_array[:k]))
    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(min(k, expected_count)))
    return dcg / idcg if idcg > 0 else 0.0


def flatten_source_chat_ids(value) -> list[int]:
    ids = []
    if value is None:
        return ids
    if isinstance(value, dict):
        for child in value.values():
            ids.extend(flatten_source_chat_ids(child))
        return ids
    if isinstance(value, (list, tuple, set)):
        for child in value:
            ids.extend(flatten_source_chat_ids(child))
        return ids
    if isinstance(value, int):
        ids.append(value)
        return ids
    if isinstance(value, str):
        clean = value.strip()
        if re.fullmatch(r"\d+(?:\s*,\s*\d+)*", clean):
            ids.extend(int(match) for match in re.findall(r"\d+", clean))
        return ids
    return ids


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def score_recall_attempt(attempt: dict, expected_count: int) -> tuple:
    hit_rank = attempt["hit_rank"]
    recalled_contents = attempt["recalled_contents"]
    rel_array = attempt["rel_array"]
    match_matrix = attempt.get("match_matrix")
    if match_matrix is None:
        match_matrix = build_evidence_match_matrix(
            attempt["expected_contexts"], recalled_contents
        )
    r5_frac, r5_all = calc_recall_for_contents(
        attempt["expected_contexts"], recalled_contents, 5, match_matrix
    )
    r10_frac, r10_all = calc_recall_for_contents(
        attempt["expected_contexts"], recalled_contents, 10, match_matrix
    )
    r20_frac, r20_all = calc_recall_for_contents(
        attempt["expected_contexts"], recalled_contents, 20, match_matrix
    )
    n5 = calc_ndcg_for_rel(rel_array, expected_count, 5)
    n10 = calc_ndcg_for_rel(rel_array, expected_count, 10)
    n20 = calc_ndcg_for_rel(rel_array, expected_count, 20)

    return (
        hit_rank > 0,
        -(hit_rank if hit_rank > 0 else 1_000_000),
        r5_all,
        r5_frac,
        n5,
        r10_all,
        r10_frac,
        n10,
        r20_all,
        r20_frac,
        n20,
    )


def default_output_path(context: str) -> str:
    return f"{RESULTS_DIR}/beam_results_{context}.json"


RESUME_STEPS = ("ingest", "raw-recall", *CUEBRIDGE_ARTIFACT_STEPS)


def discover_resume_project_id(args: argparse.Namespace, record_idx: int) -> str:
    run_root = Path(args.cuebridge_run_root).expanduser()
    prefix = f"beam_{args.context}_{record_idx}_"
    candidates = [path for path in run_root.glob(f"{prefix}*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(
            f"No CueBridge run directory found for record {record_idx} under {run_root}. "
            "Pass --project-id explicitly or start from ingest."
        )
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    return latest.name[len(prefix):]


def cuebridge_record_run_dir(args: argparse.Namespace, record_idx: int, project_id: str) -> Path:
    return Path(args.cuebridge_run_root).expanduser() / f"beam_{args.context}_{record_idx}_{project_id}"


def raw_baseline_path(run_dir: Path) -> Path:
    return run_dir / "cuebridge_raw_baseline.json"


def save_raw_baseline(path: Path, *, record_idx: int, project_id: str, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "artifact_type": "beam_raw_recall_baseline",
        "record_idx": record_idx,
        "project_id": project_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "probing_results": results,
    }
    path.write_text(json.dumps(payload, indent=2))


def load_raw_baseline(path: Path, record_questions: list[dict]) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot resume past raw-recall: raw baseline checkpoint is missing: {path}. "
            "Run from --start-step raw-recall to intentionally recompute it."
        )
    payload = json.loads(path.read_text())
    results = payload.get("probing_results")
    if not isinstance(results, list):
        raise ValueError(f"Raw baseline checkpoint has no probing_results list: {path}")
    if len(results) != len(record_questions):
        raise ValueError(
            f"Raw baseline checkpoint question count mismatch: {len(results)} saved, "
            f"{len(record_questions)} expected: {path}"
        )
    for idx, (saved, question) in enumerate(zip(results, record_questions), start=1):
        if saved.get("question") != question.get("question"):
            raise ValueError(f"Raw baseline checkpoint question mismatch at #{idx}: {path}")
    return results


def save_results(output_path: str, results: list[dict]) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    file_count = 0
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(rel)
        digest.update(b"\0")
        with open(child, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
        file_count += 1
    return digest.hexdigest(), file_count


def cuebridge_artifact_metadata(paths: list[str]) -> list[dict]:
    artifacts = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        entry = {"name": path.name, "path": str(path)}
        if not path.exists():
            entry.update({"exists": False})
        elif path.is_dir():
            digest, file_count = sha256_directory(path)
            entry.update({
                "exists": True,
                "kind": "directory",
                "sha256": digest,
                "file_count": file_count,
            })
        elif path.is_file():
            entry.update({
                "exists": True,
                "kind": "file",
                "sha256": sha256_file(path),
                "file_count": 1,
            })
        else:
            entry.update({"exists": True, "kind": "other"})
        artifacts.append(entry)
    return artifacts


def delete_project(project_id: str, url: str) -> None:
    errors = []
    for base_url in status_urls(url):
        endpoint = f"{base_url}/projects/{urllib.parse.quote(project_id, safe='')}"
        request = urllib.request.Request(endpoint, method="DELETE")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                if 200 <= response.status < 300 or response.status == 404:
                    return
                errors.append(f"DELETE {endpoint} returned HTTP {response.status}")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return
            body = exc.read().decode("utf-8", errors="replace")
            errors.append(f"DELETE {endpoint} failed with HTTP {exc.code}: {body}")
        except urllib.error.URLError as exc:
            errors.append(f"DELETE {endpoint} failed: {exc}")

    raise RuntimeError("; ".join(errors) or f"DELETE project {project_id} failed")


def delete_project_files(project_id: str, snapshots_dir: str, contents_dir: str) -> None:
    snapshots = Path(snapshots_dir)
    if snapshots.exists():
        for suffix in (".bin", "_aliases.bin"):
            path = snapshots / f"{project_id}{suffix}"
            if path.exists():
                path.unlink()

    contents = Path(contents_dir) / project_id
    if contents.exists():
        shutil.rmtree(contents)


def build_recall_payload(
    args,
    question: str,
    query_time: str | None = None,
    *,
    cuebridge_artifacts_enabled: bool | None = None,
) -> dict:
    use_cuebridge_artifacts = (
        args.enable_cuebridge_artifacts
        if cuebridge_artifacts_enabled is None
        else cuebridge_artifacts_enabled
    )
    payload = {
        "query_text": question,
        "cues": [],
        "semantic_mode": os.environ.get("CUEMAP_SEMANTIC_MODE", "hybrid"),
        "limit": args.limit,
        "auto_reinforce": False,
        "depth": 1,
        "disable_salience_bias": False,
        "disable_alias_expansion": not args.enable_alias_expansion,
        "disable_cuebridge_artifacts": not use_cuebridge_artifacts,
        "parent_fusion": args.parent_fusion,
        "parent_fusion_limit": args.parent_fusion_limit,
        "parent_fusion_min_chunks": args.parent_fusion_min_chunks,
        "ordered_reconstruction": args.ordered_reconstruction,
        "ordered_reconstruction_limit": args.ordered_reconstruction_limit,
        "ordered_session_scan_limit": args.ordered_session_scan_limit,
        "ordered_max_sessions": args.ordered_max_sessions,
        "evidence_coverage": args.evidence_coverage,
        "evidence_coverage_limit": args.evidence_coverage_limit,
        "evidence_coverage_session_scan_limit": args.evidence_coverage_session_scan_limit,
        "evidence_coverage_max_sessions": args.evidence_coverage_max_sessions,
    }
    if args.expansion_depth is not None:
        payload["expansion_depth"] = args.expansion_depth
    if query_time:
        payload["query_time"] = query_time
    return payload


def run_recall_attempt(
    args,
    project_id: str,
    q_type: str,
    question: str,
    query_time: str | None,
    expected_contexts: list[str],
    *,
    name: str,
    cuebridge_artifacts_enabled: bool | None = None,
) -> dict:
    recall_started = time.perf_counter()
    recall_response = post_json(
        args.url,
        "/recall",
        project_id,
        build_recall_payload(
            args,
            question,
            query_time,
            cuebridge_artifacts_enabled=cuebridge_artifacts_enabled,
        ),
        timeout=120,
    )
    latency_ms = int((time.perf_counter() - recall_started) * 1000)
    recalled_results = recall_response.get("results")
    if not isinstance(recalled_results, list):
        raise RuntimeError(f"POST /recall returned no results array: {recall_response!r}")
    recalled_results = [result for result in recalled_results if isinstance(result, dict)]
    recalled_contents = [str(result.get("content", "")) for result in recalled_results]
    recalled_memory_ids = [result.get("memory_id") for result in recalled_results]
    recalled_scores = [result.get("score") for result in recalled_results]
    ctx_tokens_20 = context_token_count(recalled_contents, 20)
    ctx_tokens_returned = context_token_count(recalled_contents)
    match_matrix = build_evidence_match_matrix(expected_contexts, recalled_contents)
    hit_rank, rel_array = calc_match(expected_contexts, recalled_contents, match_matrix)
    if getattr(args, "cuebridge_client", None) is not None:
        args.cuebridge_client.send_event(
            {"query_text": question, "query_time": query_time, "limit": args.limit},
            cuebridge_recall_response(recalled_contents),
            latency_ms=latency_ms,
            **cuebridge_eval_labels(
                recalled_contents=recalled_contents,
                expected_contexts=expected_contexts,
                hit_rank=hit_rank,
                project_id=project_id,
                q_type=q_type,
                name=name,
            ),
        )
    return {
        "name": name,
        "hit_rank": hit_rank,
        "rel_array": rel_array,
        "recalled_contents": recalled_contents,
        "ctx_tokens": ctx_tokens_20,
        "ctx_tokens_returned": ctx_tokens_returned,
        "recalled_memory_ids": recalled_memory_ids,
        "recalled_scores": recalled_scores,
        "expected_contexts": expected_contexts,
        "match_matrix": match_matrix,
    }


def recall_once(
    args,
    project_id: str,
    q_type: str,
    question: str,
    query_time: str | None,
    expected_contexts: list[str],
    *,
    name: str = "base",
    cuebridge_artifacts_enabled: bool | None = None,
) -> tuple[dict, list[dict]]:
    attempt = run_recall_attempt(
        args,
        project_id,
        q_type,
        question,
        query_time,
        expected_contexts,
        name=name,
        cuebridge_artifacts_enabled=cuebridge_artifacts_enabled,
    )
    return attempt, [attempt]


def get_expected_contexts(q: dict, flat_chat: list) -> list[str]:
    expected = []
    # Try looking up by source_chat_ids in flat_chat
    source_ids = flatten_source_chat_ids(q.get("source_chat_ids", []))
    if source_ids:
        source_set = set(source_ids)
        for msg in flat_chat:
            msg_global_id = msg.get("_beam_global_id")
            try:
                msg_id = int(msg["id"]) if msg.get("id") is not None else None
            except (TypeError, ValueError):
                msg_id = None
            if (
                (msg_id is not None and msg_id in source_set)
                or (msg_global_id is not None and msg_global_id in source_set)
            ):
                content = msg.get("content", "").strip()
                if content:
                    expected.append(content)

    # Fallback to conversation_reference or conversation_references if expected is empty
    refs = q.get("conversation_reference") or q.get("conversation_references")
    if not expected and refs:
        if isinstance(refs, list):
            for r in refs:
                if isinstance(r, str):
                    clean_r = re.sub(r"^(?:plan-\d+,\s*)?Turn\s+\d+:\s*", "", r, flags=re.IGNORECASE).strip()
                    if clean_r:
                        expected.append(clean_r)
        elif isinstance(refs, str):
            clean_ref = re.sub(r"^(?:plan-\d+,\s*)?Turn\s+\d+:\s*", "", refs, flags=re.IGNORECASE).strip()
            if clean_ref:
                expected.append(clean_ref)

    # Fallback to key_facts_tested or answer as final resorts if still empty
    if not expected:
        ans = q.get("answer") or q.get("ideal_response")
        if ans:
            if isinstance(ans, str):
                expected.append(ans.strip())
            elif isinstance(ans, list):
                expected.extend([x.strip() for x in ans if isinstance(x, str)])

    return dedupe_preserve_order([x for x in expected if x])


def score_beam_question(
    args,
    project_id: str,
    flat_chat: list,
    q_idx: int,
    q: dict,
    *,
    phase_name: str,
    cuebridge_artifacts_enabled: bool | None,
) -> dict | None:
    question = q.get("question", "")
    q_type = q.get("category", "Unknown")
    if not question:
        return None

    expected_context_lines = get_expected_contexts(q, flat_chat)
    if not expected_context_lines:
        print(f"WARNING: no expected contexts found for question '{question}', skipping scoring.")
        return None

    best_recall, recall_attempts = recall_once(
        args,
        project_id,
        q_type,
        question,
        q.get("time_anchor"),
        expected_context_lines,
        name=phase_name,
        cuebridge_artifacts_enabled=cuebridge_artifacts_enabled,
    )
    recalled_contents = best_recall["recalled_contents"]
    hit_rank = best_recall["hit_rank"]
    rel_array = best_recall["rel_array"]
    match_matrix = best_recall["match_matrix"]
    ctx_tokens = best_recall["ctx_tokens"]
    ctx_tokens_returned = best_recall["ctx_tokens_returned"]

    r5_frac, r5_all = calc_recall_for_contents(
        expected_context_lines, recalled_contents, 5, match_matrix
    )
    r10_frac, r10_all = calc_recall_for_contents(
        expected_context_lines, recalled_contents, 10, match_matrix
    )
    r20_frac, r20_all = calc_recall_for_contents(
        expected_context_lines, recalled_contents, 20, match_matrix
    )
    r50_frac, r50_all = calc_recall_for_contents(
        expected_context_lines, recalled_contents, 50, match_matrix
    )
    r100_frac, r100_all = calc_recall_for_contents(
        expected_context_lines, recalled_contents, 100, match_matrix
    )
    n5 = calc_ndcg_for_rel(rel_array, len(expected_context_lines), 5)
    n10 = calc_ndcg_for_rel(rel_array, len(expected_context_lines), 10)
    n20 = calc_ndcg_for_rel(rel_array, len(expected_context_lines), 20)
    n50 = calc_ndcg_for_rel(rel_array, len(expected_context_lines), 50)
    n100 = calc_ndcg_for_rel(rel_array, len(expected_context_lines), 100)

    print(
        f"  [{q_idx + 1}] {phase_name} | Type: {q_type} | "
        f"Hit Rank: {hit_rank if hit_rank > 0 else 'MISS'} | "
        f"NDCG@10: {n10:.2f} | Recall_Frac@10: {r10_frac:.2f} | "
        f"CtxTokens@20: {ctx_tokens}"
    )

    return {
        "question": question,
        "category": q_type,
        "hit_rank": hit_rank,
        "selected_recall_attempt": best_recall["name"],
        "recall_attempts": [
            {
                "name": attempt["name"],
                "hit_rank": attempt["hit_rank"],
                "hit_at_20": 0 < attempt["hit_rank"] <= 20,
                "ctx_tokens": attempt["ctx_tokens"],
                "ctx_tokens_returned": attempt["ctx_tokens_returned"],
            }
            for attempt in recall_attempts
        ],
        "expected_contexts": expected_context_lines,
        "recalled_contents": recalled_contents,
        "ctx_tokens": ctx_tokens,
        "ctx_tokens_returned": ctx_tokens_returned,
        "recalled_memory_ids": best_recall["recalled_memory_ids"],
        "recalled_scores": best_recall["recalled_scores"],
        "rel_array": rel_array,
        "recall_frac_5": r5_frac,
        "recall_frac_10": r10_frac,
        "recall_frac_20": r20_frac,
        "recall_frac_50": r50_frac,
        "recall_frac_100": r100_frac,
        "recall_all_5": r5_all,
        "recall_all_10": r10_all,
        "recall_all_20": r20_all,
        "recall_all_50": r50_all,
        "recall_all_100": r100_all,
        "ndcg_5": n5,
        "ndcg_10": n10,
        "ndcg_20": n20,
        "ndcg_50": n50,
        "ndcg_100": n100,
    }


def evaluate():
    parser = argparse.ArgumentParser(
        description="Settled/background-aware CueMap BEAM Memory Benchmark harness."
    )
    parser.add_argument("--context", choices=["128k", "500k", "1m", "10m"], default="128k",
                        help="Context/conversation length slice to evaluate.")
    parser.add_argument("--url", default="http://127.0.0.1:8735")
    parser.add_argument("--recall-only", action="store_true")
    parser.add_argument("--no-wait-bg", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--expansion-depth",
        type=int,
        default=None,
        help=(
            "Pass through cuemap recall --expansion-depth. A value of 4 includes "
            "up to three neighboring memories before and after each hit."
        ),
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--start-step",
        choices=RESUME_STEPS,
        default="ingest",
        help=(
            "Resume the first selected record from this step. Use ingest for a full run, "
            "raw-recall to reuse an existing project but rerun raw recall, or a CueBridge "
            "artifact step such as score-questions to reuse earlier artifact outputs."
        ),
    )
    parser.add_argument(
        "--project-id",
        default=None,
        help=(
            "Existing project id for the first resumed record. If omitted with --start-step "
            "after ingest, the newest matching CueBridge run directory is used."
        ),
    )
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--ingest-long-form",
        action="store_true",
        help="Use /ingest/content for each message so long records are chunked before indexing.",
    )
    parser.add_argument(
        "--source-order-metadata",
        action="store_true",
        help=(
            "Attach source_session_id/source_turn_index metadata to message-level ingestion "
            "so recall expansion can retrieve neighboring full-message memories."
        ),
    )
    parser.add_argument(
        "--long-form-segmenter",
        choices=("sentence_window", "logical_block"),
        default="logical_block",
        help="Segmenter used with --ingest-long-form.",
    )
    parser.add_argument("--segment-window-size", type=int, default=8)
    parser.add_argument("--segment-overlap", type=int, default=0)
    parser.add_argument("--segment-min-chunk-chars", type=int, default=120)
    parser.add_argument("--segment-max-chunk-chars", type=int, default=4000)
    parser.add_argument("--enable-alias-expansion", action="store_true")
    parser.add_argument(
        "--enable-cuebridge-artifacts",
        action="store_true",
        help="Allow installed CueBridge artifacts during recall. Disabled by default for core evals.",
    )
    parser.add_argument("--no-auto-reinforce", action="store_true")
    parser.add_argument("--parent-fusion", choices=["off", "auto", "force"], default="off")
    parser.add_argument("--parent-fusion-limit", type=int, default=80)
    parser.add_argument("--parent-fusion-min-chunks", type=int, default=2)
    parser.add_argument("--ordered-reconstruction", choices=["off", "auto", "force"], default="off")
    parser.add_argument("--ordered-reconstruction-limit", type=int, default=80)
    parser.add_argument("--ordered-session-scan-limit", type=int, default=4096)
    parser.add_argument("--ordered-max-sessions", type=int, default=3)
    parser.add_argument("--evidence-coverage", choices=["off", "auto", "force"], default="off")
    parser.add_argument("--evidence-coverage-limit", type=int, default=100)
    parser.add_argument("--evidence-coverage-session-scan-limit", type=int, default=4096)
    parser.add_argument("--evidence-coverage-max-sessions", type=int, default=3)
    parser.add_argument(
        "--cuebridge-observe",
        action="store_true",
        default=env_bool("CUEBRIDGE_OBSERVE", False),
        help=(
            "Observe recall attempts with the CueBridge Python SDK. Raw query text is sent "
            "only to the local agent; returned/target IDs are content hashes."
        ),
    )
    parser.add_argument(
        "--cuebridge-agent-url",
        default=os.environ.get("CUEBRIDGE_AGENT_URL", "http://127.0.0.1:8099"),
        help="Local CueBridge agent URL used with --cuebridge-observe.",
    )
    parser.add_argument(
        "--cuebridge-api-key",
        default=os.environ.get("CUEBRIDGE_API_KEY"),
        help="Optional CueBridge API key for cloud or authenticated endpoints.",
    )
    parser.add_argument(
        "--cuebridge-observe-project",
        default=os.environ.get("CUEBRIDGE_OBSERVE_PROJECT"),
        help=(
            "Optional CueBridge project to aggregate all observed eval traffic under. "
            "Defaults to beam_<context>_eval."
        ),
    )
    parser.add_argument(
        "--cuebridge-observe-batch-size",
        type=int,
        default=int(os.environ.get("CUEBRIDGE_OBSERVE_BATCH_SIZE", "25")),
        help="CueBridge SDK batch size for observed recall events.",
    )
    parser.add_argument(
        "--cuebridge-observe-flush-interval",
        type=float,
        default=float(os.environ.get("CUEBRIDGE_OBSERVE_FLUSH_INTERVAL", "2.0")),
        help="CueBridge SDK background flush interval in seconds.",
    )
    parser.add_argument(
        "--no-cuebridge-observe-ingest",
        dest="cuebridge_observe_ingest",
        action="store_false",
        default=env_bool("CUEBRIDGE_OBSERVE_INGEST", True),
        help="Disable CueBridge ingest observations; recall observations remain enabled.",
    )
    parser.add_argument(
        "--delete-project-after-record",
        action="store_true",
        help="Delete each non-recall-only eval project from the running server after scoring it.",
    )
    parser.add_argument(
        "--delete-project-files-after-record",
        action="store_true",
        help="Also delete that eval project's snapshot files and disk-backed content directory after scoring it.",
    )
    parser.add_argument("--snapshots-dir", default=str(Path.home() / ".cuemap" / "data" / "snapshots"))
    parser.add_argument("--contents-dir", default=str(Path.home() / ".cuemap" / "data" / "contents"))
    parser.add_argument(
        "--cuebridge-artifact",
        action="append",
        default=[],
        help="Active CueBridge artifact file or directory; path and sha256 are recorded in result metadata.",
    )
    parser.add_argument("--output", default=None)
    add_cuebridge_compare_args(parser)
    args = parser.parse_args()
    cuebridge_mode = cuebridge_compare_mode(args)
    cuebridge_enabled = cuebridge_mode != "off"
    if args.recall_only and args.start_step != "ingest":
        raise ValueError("--start-step is for non-recall-only runs.")
    if args.start_step in CUEBRIDGE_ARTIFACT_STEPS and not cuebridge_enabled:
        raise ValueError(f"--start-step {args.start_step} requires a CueBridge compare mode.")

    output_path = args.output or default_output_path(args.context)
    active_artifacts = cuebridge_artifact_metadata(args.cuebridge_artifact)
    print(f"Context / Config size: {args.context}")
    print("Server management: external (assumes running server on url)")
    print(f"Output path: {output_path}")
    if active_artifacts:
        print(f"Active CueBridge artifacts: {len(active_artifacts)}")
    if cuebridge_enabled:
        print(f"CueBridge compare mode: {cuebridge_mode}")
    if args.cuebridge_observe:
        observe_target = args.cuebridge_observe_project or f"beam_{args.context}_eval"
        ingest_mode = "on" if args.cuebridge_observe_ingest else "off"
        print(
            f"CueBridge observation: enabled | agent={args.cuebridge_agent_url} | "
            f"project={observe_target} | ingest={ingest_mode}"
        )
    args.cuebridge_client = make_cuebridge_client(args)

    from datasets import load_dataset

    # Load dataset using Hugging Face datasets library
    print(f"Fetching dataset 'Mohammadta/BEAM' for config...")
    if args.context == "128k":
        ds = load_dataset("Mohammadta/BEAM")
        records = ds["100K"]
    elif args.context == "500k":
        ds = load_dataset("Mohammadta/BEAM")
        records = ds["500K"]
    elif args.context == "1m":
        ds = load_dataset("Mohammadta/BEAM")
        records = ds["1M"]
    elif args.context == "10m":
        print(f"Fetching dataset 'Mohammadta/BEAM-10M'...")
        ds = load_dataset("Mohammadta/BEAM-10M")
        records = ds["10M"]
    else:
        raise ValueError(f"Unknown context value: {args.context}")

    end_index = len(records) if args.max_records is None else min(len(records), args.start_index + args.max_records)
    records_to_test = list(enumerate(records))[args.start_index:end_index]

    project_mapping = {}
    if args.recall_only:
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"{output_path} not found; recall-only needs an existing output file")
        with open(output_path, "r") as f:
            old_results = json.load(f)
        project_mapping = {r["record_idx"]: r["project_id"] for r in old_results}

    results = []
    hit_at_1 = hit_at_5 = hit_at_10 = hit_at_20 = hit_at_50 = hit_at_100 = 0
    sum_recall_frac_5 = sum_recall_frac_10 = sum_recall_frac_20 = sum_recall_frac_50 = sum_recall_frac_100 = 0.0
    sum_recall_all_5 = sum_recall_all_10 = sum_recall_all_20 = sum_recall_all_50 = sum_recall_all_100 = 0.0
    sum_ndcg_5 = sum_ndcg_10 = sum_ndcg_20 = sum_ndcg_50 = sum_ndcg_100 = 0.0
    total_questions = 0
    context_tokens_by_question = []
    stats_by_type = defaultdict(lambda: {
        "total": 0, "hit1": 0, "hit5": 0, "hit10": 0, "hit20": 0, "hit50": 0, "hit100": 0,
        "sum_recall_frac_5": 0.0, "sum_recall_frac_10": 0.0, "sum_recall_frac_20": 0.0, "sum_recall_frac_50": 0.0, "sum_recall_frac_100": 0.0,
        "sum_recall_all_5": 0.0, "sum_recall_all_10": 0.0, "sum_recall_all_20": 0.0, "sum_recall_all_50": 0.0, "sum_recall_all_100": 0.0,
        "sum_ndcg_5": 0.0, "sum_ndcg_10": 0.0, "sum_ndcg_20": 0.0, "sum_ndcg_50": 0.0, "sum_ndcg_100": 0.0,
        "context_tokens": [],
        "hit20_base": 0, "final_miss": 0,
    })
    hit20_by_attempt = defaultdict(int)
    final_misses = 0

    for record_idx, record in records_to_test:
        resume_first_record = (
            not args.recall_only
            and args.start_step != "ingest"
            and record_idx == args.start_index
        )
        if args.recall_only:
            project_id = project_mapping.get(record_idx)
            if not project_id:
                continue
        elif resume_first_record:
            if args.project_id:
                project_id = args.project_id
            else:
                project_id = discover_resume_project_id(args, record_idx)
                print(f"Auto-discovered resume project id for record {record_idx}: {project_id}")
        else:
            project_id = f"eval_beam_{args.context}_{record_idx}_{int(time.time())}"

        if args.recall_only:
            mode = "RECALL-ONLY"
        elif resume_first_record:
            mode = f"RESUME:{args.start_step}"
        else:
            mode = "FULL"
        print(f"\n--- Testing Record {record_idx + 1}/{len(records)} | Project: {project_id} | Mode: {mode} | Context: {args.context} ---")

        # Parse chat data recursively to be extremely robust.
        # Preserve source-provided plan grouping when the dataset exposes it.
        flat_chat = []
        if args.context == "10m" and "plans" in record:
            plans = record.get("plans", [])
            for plan_idx, plan in enumerate(plans):
                plan_chat = plan.get("chat", [])
                if isinstance(plan_chat, list):
                    for item in plan_chat:
                        if isinstance(item, list):
                            for sub_item in item:
                                if isinstance(sub_item, dict):
                                    msg = dict(sub_item)
                                    msg.setdefault("plan_idx", plan_idx)
                                    msg["_beam_global_id"] = len(flat_chat)
                                    flat_chat.append(msg)
                        elif isinstance(item, dict):
                            msg = dict(item)
                            msg.setdefault("plan_idx", plan_idx)
                            msg["_beam_global_id"] = len(flat_chat)
                            flat_chat.append(msg)
        else:
            chat = record.get("chat", [])
            if isinstance(chat, list):
                for item in chat:
                    if isinstance(item, list):
                        for sub_item in item:
                            if isinstance(sub_item, dict):
                                msg = dict(sub_item)
                                msg["_beam_global_id"] = len(flat_chat)
                                flat_chat.append(msg)
                    elif isinstance(item, dict):
                        msg = dict(item)
                        msg["_beam_global_id"] = len(flat_chat)
                        flat_chat.append(msg)

        settle_status = None
        if resume_first_record:
            print(f"Skipping ingestion for resume; using existing project {project_id}.")
        if not args.recall_only and not resume_first_record:
            total_messages = len(flat_chat)
            print(f"Total messages to ingest for this record: {total_messages}")
            ingested_count = 0

            for msg in flat_chat:
                role = msg.get("role", "")
                content = msg.get("content", "")
                text_to_add = f"{role}: {content}"
                metadata = {"source_role": role}
                time_anchor = msg.get("time_anchor")
                if time_anchor:
                    metadata["source_date"] = time_anchor
                if msg.get("plan_idx") is not None:
                    metadata["source_plan_idx"] = msg["plan_idx"]
                if args.source_order_metadata:
                    metadata.update({
                        "source_session_id": f"beam:{args.context}:record:{record_idx}",
                        "source_turn_index": ingested_count,
                        "source_chat_id": msg.get("id", ingested_count),
                        "source_beam_global_id": msg.get("_beam_global_id", ingested_count),
                    })

                if args.ingest_long_form:
                    msg_id = msg.get("id", ingested_count)
                    source_key_id = msg.get("_beam_global_id", msg_id)
                    source_key = f"beam:{args.context}:record:{record_idx}:message:{source_key_id}"
                    structural_cues = ["source_type:chat_message"]
                    if role:
                        structural_cues.append(f"source_role:{cue_value(role)}")
                    if time_anchor:
                        structural_cues.append(f"source_date:{cue_value(time_anchor)}")
                    if msg.get("plan_idx") is not None:
                        structural_cues.append(f"source_plan:{cue_value(msg['plan_idx'])}")

                    post_json(
                        args.url,
                        "/ingest/content",
                        project_id,
                        {
                            "content": text_to_add,
                            "filename": f"beam_record_{record_idx}_message_{msg_id}.txt",
                            "source_key": source_key,
                            "metadata": {
                                **metadata,
                                "source_session_id": f"beam:{args.context}:record:{record_idx}",
                                "source_turn_index": ingested_count,
                                "source_chat_id": msg_id,
                                "source_beam_global_id": source_key_id,
                            },
                            "structural_cues": structural_cues,
                            "segmenter": args.long_form_segmenter,
                            "segment_window_size": args.segment_window_size,
                            "segment_overlap": args.segment_overlap,
                            "segment_min_chunk_chars": args.segment_min_chunk_chars,
                            "segment_max_chunk_chars": args.segment_max_chunk_chars,
                        },
                        timeout=60,
                    )
                else:
                    run_cmd(
                        [
                            "cuemap",
                            "add",
                            "-p",
                            project_id,
                            "--url",
                            args.url,
                            "--metadata",
                            json.dumps(metadata, separators=(",", ":")),
                            text_to_add,
                        ],
                        check=True,
                    )

                cuebridge_send_ingest(
                    args,
                    project_id=project_id,
                    record_idx=record_idx,
                    turn_idx=ingested_count,
                    content=text_to_add,
                    metadata=metadata,
                )
                ingested_count += 1
                if ingested_count % 50 == 0:
                    print(f"Ingested {ingested_count}/{total_messages}...", end="\r", flush=True)

            print(f"\nIngested {total_messages} messages.")
            if not args.no_wait_bg:
                settle_status = wait_for_bg_jobs(project_id, args.url, args.timeout_seconds, args.poll_seconds)

        # Parse probing questions
        probing_questions_raw = record.get("probing_questions", "")
        probing_questions = {}
        if isinstance(probing_questions_raw, str):
            try:
                probing_questions = ast.literal_eval(probing_questions_raw)
            except Exception as exc:
                print(f"WARNING: failed to parse probing questions using ast: {exc}")
        elif isinstance(probing_questions_raw, dict):
            probing_questions = probing_questions_raw

        record_questions = []
        for category, q_list in probing_questions.items():
            if category == "abstention":
                # We skip abstentions as they evaluate withholding answers, which does not map to a ground truth context
                continue
            for q_dict in q_list:
                if isinstance(q_dict, dict):
                    q_dict["category"] = category
                    record_questions.append(q_dict)

        print(f"Running {len(record_questions)} probing questions...")

        baseline_results = []
        for q_idx, q in enumerate(record_questions):
            scored = score_beam_question(
                args,
                project_id,
                flat_chat,
                q_idx,
                q,
                phase_name="raw" if cuebridge_enabled else "base",
                cuebridge_artifacts_enabled=False if cuebridge_enabled else None,
            )
            if scored is not None:
                baseline_results.append(scored)

        artifact_metadata = None
        record_results = baseline_results
        if cuebridge_enabled:
            target_texts = []
            target_questions = []
            if cuebridge_mode in {"oracle", "question_oracle"}:
                for raw_result in baseline_results:
                    raw_rank = raw_result["hit_rank"]
                    if raw_rank <= 0 or raw_rank > args.cuebridge_target_rank_threshold:
                        if cuebridge_mode == "oracle":
                            target_texts.extend(raw_result["expected_contexts"])
                        else:
                            target_questions.append(
                                {
                                    "id": f"beam_{record_idx}_{len(target_questions) + 1:04d}",
                                    "question": raw_result["question"],
                                    "category": raw_result["category"],
                                    "target_texts": raw_result["expected_contexts"],
                                }
                            )

            if cuebridge_mode == "product" or target_texts or target_questions:
                run_dir = Path(args.cuebridge_run_root).expanduser() / f"beam_{args.context}_{record_idx}_{project_id}"
                if cuebridge_mode == "oracle":
                    print(
                        f"Building CueBridge artifacts for project {project_id} "
                        f"from {len(target_texts)} gold evidence targets..."
                    )
                elif cuebridge_mode == "question_oracle":
                    print(
                        f"Building CueBridge artifacts for project {project_id} "
                        f"from {len(target_questions)} actual eval questions..."
                    )
                else:
                    print(f"Building CueBridge artifacts for project {project_id} in product mode...")
                artifact_start_step = (
                    args.start_step
                    if resume_first_record and args.start_step in CUEBRIDGE_ARTIFACT_STEPS
                    else "analyze-project"
                )
                artifact_metadata = build_cuebridge_artifacts(
                    args,
                    project_id,
                    run_dir,
                    target_texts=target_texts if cuebridge_mode == "oracle" else None,
                    target_questions=target_questions if cuebridge_mode == "question_oracle" else None,
                    start_step=artifact_start_step,
                )
                if not artifact_metadata.get("skipped"):
                    enhanced_results = []
                    for q_idx, q in enumerate(record_questions):
                        scored = score_beam_question(
                            args,
                            project_id,
                            flat_chat,
                            q_idx,
                            q,
                            phase_name="cuebridge",
                            cuebridge_artifacts_enabled=True,
                        )
                        if scored is not None:
                            enhanced_results.append(scored)
                    record_results = []
                    for raw_result, enhanced_result in zip(baseline_results, enhanced_results):
                        enhanced_result["raw_result"] = {
                            "hit_rank": raw_result["hit_rank"],
                            "recalled_contents": raw_result["recalled_contents"],
                            "ctx_tokens": raw_result["ctx_tokens"],
                            "ctx_tokens_returned": raw_result["ctx_tokens_returned"],
                            "recall_frac_20": raw_result["recall_frac_20"],
                            "recall_all_20": raw_result["recall_all_20"],
                            "ndcg_20": raw_result["ndcg_20"],
                        }
                        enhanced_result["cuebridge_delta"] = {
                            "raw_rank": raw_result["hit_rank"],
                            "enhanced_rank": enhanced_result["hit_rank"],
                            "rescued_at_20": not (0 < raw_result["hit_rank"] <= 20)
                            and 0 < enhanced_result["hit_rank"] <= 20,
                            "regressed_from_20": 0 < raw_result["hit_rank"] <= 20
                            and not (0 < enhanced_result["hit_rank"] <= 20),
                        }
                        record_results.append(enhanced_result)
            else:
                artifact_metadata = {
                    "skipped": "all_raw_ranks_within_target_threshold",
                    "target_rank_threshold": args.cuebridge_target_rank_threshold,
                }
                record_results = []
                for raw_result in baseline_results:
                    raw_result["raw_result"] = {
                        "hit_rank": raw_result["hit_rank"],
                        "recalled_contents": raw_result["recalled_contents"],
                        "ctx_tokens": raw_result["ctx_tokens"],
                        "ctx_tokens_returned": raw_result["ctx_tokens_returned"],
                        "recall_frac_20": raw_result["recall_frac_20"],
                        "recall_all_20": raw_result["recall_all_20"],
                        "ndcg_20": raw_result["ndcg_20"],
                    }
                    raw_result["cuebridge_delta"] = {
                        "raw_rank": raw_result["hit_rank"],
                        "enhanced_rank": raw_result["hit_rank"],
                        "rescued_at_20": False,
                        "regressed_from_20": False,
                    }
                    record_results.append(raw_result)

        for scored_result in record_results:
            q_type = scored_result["category"]
            hit_rank = scored_result["hit_rank"]
            recalled_contents = scored_result["recalled_contents"]
            rel_array = scored_result["rel_array"]
            r5_frac = scored_result["recall_frac_5"]
            r10_frac = scored_result["recall_frac_10"]
            r20_frac = scored_result["recall_frac_20"]
            r50_frac = scored_result["recall_frac_50"]
            r100_frac = scored_result["recall_frac_100"]
            r5_all = scored_result["recall_all_5"]
            r10_all = scored_result["recall_all_10"]
            r20_all = scored_result["recall_all_20"]
            r50_all = scored_result["recall_all_50"]
            r100_all = scored_result["recall_all_100"]
            n5 = scored_result["ndcg_5"]
            n10 = scored_result["ndcg_10"]
            n20 = scored_result["ndcg_20"]
            n50 = scored_result["ndcg_50"]
            n100 = scored_result["ndcg_100"]
            ctx_tokens = scored_result["ctx_tokens"]

            if hit_rank == 1:
                hit_at_1 += 1
                stats_by_type[q_type]["hit1"] += 1
            if 0 < hit_rank <= 5:
                hit_at_5 += 1
                stats_by_type[q_type]["hit5"] += 1
            if 0 < hit_rank <= 10:
                hit_at_10 += 1
                stats_by_type[q_type]["hit10"] += 1
            if 0 < hit_rank <= 20:
                hit_at_20 += 1
                stats_by_type[q_type]["hit20"] += 1
                attempt_name = scored_result["selected_recall_attempt"]
                hit20_by_attempt[attempt_name] += 1
                if attempt_name in {"base", "raw"}:
                    stats_by_type[q_type]["hit20_base"] += 1
            
            if 0 < hit_rank <= 50:
                hit_at_50 += 1
                stats_by_type[q_type]["hit50"] += 1
            if 0 < hit_rank <= 100:
                hit_at_100 += 1
                stats_by_type[q_type]["hit100"] += 1
            
            if hit_rank == -1 or hit_rank > 100:
                final_misses += 1
                stats_by_type[q_type]["final_miss"] += 1

            sum_recall_frac_5 += r5_frac
            sum_recall_frac_10 += r10_frac
            sum_recall_frac_20 += r20_frac
            sum_recall_frac_50 += r50_frac
            sum_recall_frac_100 += r100_frac
            
            sum_recall_all_5 += r5_all
            sum_recall_all_10 += r10_all
            sum_recall_all_20 += r20_all
            sum_recall_all_50 += r50_all
            sum_recall_all_100 += r100_all
            
            sum_ndcg_5 += n5
            sum_ndcg_10 += n10
            sum_ndcg_20 += n20
            sum_ndcg_50 += n50
            sum_ndcg_100 += n100

            stats_by_type[q_type]["sum_recall_frac_5"] += r5_frac
            stats_by_type[q_type]["sum_recall_frac_10"] += r10_frac
            stats_by_type[q_type]["sum_recall_frac_20"] += r20_frac
            stats_by_type[q_type]["sum_recall_frac_50"] += r50_frac
            stats_by_type[q_type]["sum_recall_frac_100"] += r100_frac
            
            stats_by_type[q_type]["sum_recall_all_5"] += r5_all
            stats_by_type[q_type]["sum_recall_all_10"] += r10_all
            stats_by_type[q_type]["sum_recall_all_20"] += r20_all
            stats_by_type[q_type]["sum_recall_all_50"] += r50_all
            stats_by_type[q_type]["sum_recall_all_100"] += r100_all
            
            stats_by_type[q_type]["sum_ndcg_5"] += n5
            stats_by_type[q_type]["sum_ndcg_10"] += n10
            stats_by_type[q_type]["sum_ndcg_20"] += n20
            stats_by_type[q_type]["sum_ndcg_50"] += n50
            stats_by_type[q_type]["sum_ndcg_100"] += n100
            context_tokens_by_question.append(ctx_tokens)
            stats_by_type[q_type]["context_tokens"].append(ctx_tokens)
            stats_by_type[q_type]["total"] += 1
            total_questions += 1

        results.append({
            "record_idx": record_idx,
            "project_id": project_id,
            "context": args.context,
            "start_step": args.start_step if resume_first_record else "ingest",
            "ingest_long_form": args.ingest_long_form,
            "source_order_metadata": args.source_order_metadata,
            "long_form_segmenter": args.long_form_segmenter,
            "segment_window_size": args.segment_window_size,
            "segment_overlap": args.segment_overlap,
            "segment_min_chunk_chars": args.segment_min_chunk_chars,
            "segment_max_chunk_chars": args.segment_max_chunk_chars,
            "expansion_depth": args.expansion_depth,
            "evidence_matcher": EVIDENCE_MATCHER,
            "ordered_reconstruction": args.ordered_reconstruction,
            "ordered_reconstruction_limit": args.ordered_reconstruction_limit,
            "ordered_session_scan_limit": args.ordered_session_scan_limit,
            "ordered_max_sessions": args.ordered_max_sessions,
            "evidence_coverage": args.evidence_coverage,
            "evidence_coverage_limit": args.evidence_coverage_limit,
            "evidence_coverage_session_scan_limit": args.evidence_coverage_session_scan_limit,
            "evidence_coverage_max_sessions": args.evidence_coverage_max_sessions,
            "cuebridge_artifacts": active_artifacts,
            "cuebridge_artifacts_enabled": args.enable_cuebridge_artifacts or cuebridge_enabled,
            "cuebridge_compare": cuebridge_enabled,
            "cuebridge_compare_mode": cuebridge_mode,
            "cuebridge_built_artifacts": artifact_metadata,
            "settle_status": settle_status,
            "probing_results": record_results,
        })
        save_results(output_path, results)
        if getattr(args, "cuebridge_client", None) is not None:
            try:
                args.cuebridge_client.flush()
            except Exception as exc:
                print(f"WARNING: CueBridge observer flush failed: {exc}")

        if args.delete_project_after_record and not args.recall_only:
            try:
                delete_project(project_id, args.url)
                if args.delete_project_files_after_record:
                    delete_project_files(project_id, args.snapshots_dir, args.contents_dir)
                print(f"Deleted eval project: {project_id}")
            except Exception as exc:
                print(f"WARNING: Failed to delete eval project {project_id}: {exc}")

    print("\n============== SUMMARY ==============")
    print(f"\nTotal Probing Questions Evaluated: {total_questions}")
    if total_questions > 0:
        print("\n[ Recall_Any (Hit@K) ] - At least one relevant fact retrieved")
        print(f"Recall_Any@1:   {hit_at_1}/{total_questions} ({(hit_at_1 / total_questions * 100):.1f}%)")
        print(f"Recall_Any@5:   {hit_at_5}/{total_questions} ({(hit_at_5 / total_questions * 100):.1f}%)")
        print(f"Recall_Any@10:  {hit_at_10}/{total_questions} ({(hit_at_10 / total_questions * 100):.1f}%)")
        print(f"Recall_Any@20:  {hit_at_20}/{total_questions} ({(hit_at_20 / total_questions * 100):.1f}%)")
        print(f"Recall_Any@50:  {hit_at_50}/{total_questions} ({(hit_at_50 / total_questions * 100):.1f}%)")
        print(f"Recall_Any@100: {hit_at_100}/{total_questions} ({(hit_at_100 / total_questions * 100):.1f}%)")

        print("\n[ Recall_All ] - All relevant facts for the query retrieved")
        print(f"Recall_All@5:   {(sum_recall_all_5 / total_questions * 100):.1f}%")
        print(f"Recall_All@10:  {(sum_recall_all_10 / total_questions * 100):.1f}%")
        print(f"Recall_All@20:  {(sum_recall_all_20 / total_questions * 100):.1f}%")
        print(f"Recall_All@50:  {(sum_recall_all_50 / total_questions * 100):.1f}%")
        print(f"Recall_All@100: {(sum_recall_all_100 / total_questions * 100):.1f}%")

        print("\n[ Recall_Frac ] - Average fraction of relevant facts retrieved")
        print(f"Recall_Frac@5:   {(sum_recall_frac_5 / total_questions * 100):.1f}%")
        print(f"Recall_Frac@10:  {(sum_recall_frac_10 / total_questions * 100):.1f}%")
        print(f"Recall_Frac@20:  {(sum_recall_frac_20 / total_questions * 100):.1f}%")
        print(f"Recall_Frac@50:  {(sum_recall_frac_50 / total_questions * 100):.1f}%")
        print(f"Recall_Frac@100: {(sum_recall_frac_100 / total_questions * 100):.1f}%")

        print("\n[ NDCG ] - Relevance ranked scoring")
        print(f"NDCG@5:   {(sum_ndcg_5 / total_questions * 100):.1f}%")
        print(f"NDCG@10:  {(sum_ndcg_10 / total_questions * 100):.1f}%")
        print(f"NDCG@20:  {(sum_ndcg_20 / total_questions * 100):.1f}%")
        print(f"NDCG@50:  {(sum_ndcg_50 / total_questions * 100):.1f}%")
        print(f"NDCG@100: {(sum_ndcg_100 / total_questions * 100):.1f}%")

        avg_ctx_tokens = sum(context_tokens_by_question) / total_questions
        print("\n[ Retrieved Context Tokens @20 ] - Approx tokens from top-20 recalled memory text")
        print(f"CtxTokens@20 Avg: {avg_ctx_tokens:.0f}")
        print(f"CtxTokens@20 P50: {percentile(context_tokens_by_question, 50)}")
        print(f"CtxTokens@20 P95: {percentile(context_tokens_by_question, 95)}")
        print(f"CtxTokens@20 P99: {percentile(context_tokens_by_question, 99)}")
        print(f"CtxTokens@20 Max: {max(context_tokens_by_question) if context_tokens_by_question else 0}")

        print("\n[ Recall Attempt Attribution @20 ]")
        attempt_label = "Selected pass" if cuebridge_enabled else "Base pass"
        selected_hits = hit_at_20 if cuebridge_enabled else hit20_by_attempt.get("base", 0)
        print(f"{attempt_label} Hit@20: {selected_hits}/{total_questions}")
        print(f"Final misses (limit {args.limit}): {final_misses}/{total_questions}")

        print("\n============== BY QUESTION TYPE ==============")
        for q_type, s in stats_by_type.items():
            total = s["total"]
            if total == 0:
                continue
            print(f"{q_type} (Total: {total}):")
            print(f"  Recall_Any@1:   {s['hit1']}/{total} ({(s['hit1'] / total * 100):.1f}%)")
            print(f"  Recall_Any@5:   {s['hit5']}/{total} ({(s['hit5'] / total * 100):.1f}%)")
            print(f"  Recall_Any@10:  {s['hit10']}/{total} ({(s['hit10'] / total * 100):.1f}%)")
            print(f"  Recall_Any@50:  {s['hit50']}/{total} ({(s['hit50'] / total * 100):.1f}%)")
            print(f"  Recall_Any@100: {s['hit100']}/{total} ({(s['hit100'] / total * 100):.1f}%)")
            print(f"  Recall_All@5:   {(s['sum_recall_all_5'] / total * 100):.1f}%")
            print(f"  Recall_All@10:  {(s['sum_recall_all_10'] / total * 100):.1f}%")
            print(f"  Recall_All@50:  {(s['sum_recall_all_50'] / total * 100):.1f}%")
            print(f"  Recall_All@100: {(s['sum_recall_all_100'] / total * 100):.1f}%")
            print(f"  Recall_Frac@5:   {(s['sum_recall_frac_5'] / total * 100):.1f}%")
            print(f"  Recall_Frac@10:  {(s['sum_recall_frac_10'] / total * 100):.1f}%")
            print(f"  Recall_Frac@50:  {(s['sum_recall_frac_50'] / total * 100):.1f}%")
            print(f"  Recall_Frac@100: {(s['sum_recall_frac_100'] / total * 100):.1f}%")
            print(f"  NDCG@5:   {(s['sum_ndcg_5'] / total * 100):.1f}%")
            print(f"  NDCG@10:  {(s['sum_ndcg_10'] / total * 100):.1f}%")
            print(f"  NDCG@50:  {(s['sum_ndcg_50'] / total * 100):.1f}%")
            print(f"  NDCG@100: {(s['sum_ndcg_100'] / total * 100):.1f}%")
            type_ctx_tokens = s["context_tokens"]
            print(
                "  CtxTokens@20: "
                f"avg={(sum(type_ctx_tokens) / total):.0f}, "
                f"p95={percentile(type_ctx_tokens, 95)}"
            )
            print(
                "  Hit@20 source: "
                f"base={s['hit20_base']}, "
                f"final_miss={s['final_miss']}"
            )

    print(f"\nSaved details to {output_path}")
    if cuebridge_enabled:
        print_cuebridge_delta_summary(results, limit=20)
    if getattr(args, "cuebridge_client", None) is not None:
        try:
            args.cuebridge_client.shutdown()
        except Exception as exc:
            print(f"WARNING: CueBridge observer shutdown failed: {exc}")


if __name__ == "__main__":
    evaluate()
