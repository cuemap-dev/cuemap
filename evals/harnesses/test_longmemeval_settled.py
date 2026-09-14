import argparse
import json
import math
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

from cuebridge_eval_utils import (
    add_cuebridge_compare_args,
    build_cuebridge_artifacts,
    cuebridge_compare_mode,
    print_cuebridge_delta_summary,
)


DATASET_PATH = str(Path(__file__).resolve().parents[1] / "data" / "longmemeval_s_cleaned.json")
RESULTS_DIR = str(Path(__file__).resolve().parents[1] / "results")

VARIANTS = ("core",)


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
    """Use the launcher-selected release binary for every CLI call."""
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
    pairs = [("writes_completed", "writes_total")]
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


def parse_recall_contents(stdout: str) -> list[str]:
    lines = clean_output(stdout).split("\n")
    recalled_contents = []
    parsing_results = False
    current_item_lines = []

    for line in lines:
        if line.startswith("--- RECALL RESULTS"):
            parsing_results = True
            continue

        if parsing_results:
            if line.startswith("- ["):
                if current_item_lines:
                    recalled_contents.append("\n".join(current_item_lines))
                parts = line.split("] ", 2)
                current_item_lines = [parts[-1].strip()] if len(parts) >= 2 else []
            elif line.strip() and current_item_lines:
                current_item_lines.append(line)

    if current_item_lines:
        recalled_contents.append("\n".join(current_item_lines))
    return recalled_contents


def calc_match(expected_contexts: list[str], recalled_contents: list[str]) -> tuple[int, list[int]]:
    hit_rank = -1
    rel_array = [0] * len(recalled_contents)

    for rank, r_content in enumerate(recalled_contents):
        rc_norm = normalize_text(r_content)
        for expected in expected_contexts:
            expected_norm = normalize_text(expected)
            if expected_norm in rc_norm or rc_norm in expected_norm:
                rel_array[rank] = 1
                if hit_rank == -1:
                    hit_rank = rank + 1
                break

    return hit_rank, rel_array


def calc_recall_for_contents(expected_contexts: list[str], recalled_contents: list[str], k: int) -> tuple[float, float]:
    found = set()
    for r_content in recalled_contents[:k]:
        rc_norm = normalize_text(r_content)
        for idx, expected in enumerate(expected_contexts):
            expected_norm = normalize_text(expected)
            if expected_norm in rc_norm or rc_norm in expected_norm:
                found.add(idx)
    total_expected = max(1, len(expected_contexts))
    frac = len(found) / total_expected
    all_found = 1.0 if len(found) == len(expected_contexts) else 0.0
    return frac, all_found


def calc_ndcg_for_rel(rel_array: list[int], expected_count: int, k: int) -> float:
    dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(rel_array[:k]))
    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(min(k, expected_count)))
    return dcg / idcg if idcg > 0 else 0.0


def score_recall_attempt(attempt: dict, expected_count: int) -> tuple:
    hit_rank = attempt["hit_rank"]
    recalled_contents = attempt["recalled_contents"]
    rel_array = attempt["rel_array"]
    r5_frac, r5_all = calc_recall_for_contents(attempt["expected_contexts"], recalled_contents, 5)
    r10_frac, r10_all = calc_recall_for_contents(attempt["expected_contexts"], recalled_contents, 10)
    r20_frac, r20_all = calc_recall_for_contents(attempt["expected_contexts"], recalled_contents, 20)
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


def default_output_path(variant: str) -> str:
    return f"{RESULTS_DIR}/longmemeval_s_results_{variant.replace('-', '_')}.json"


def save_results(output_path: str, results: list[dict]) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


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


def build_recall_cmd(
    args,
    project_id: str,
    q_type: str,
    question: str,
    query_time: str | None = None,
    *,
    cuebridge_artifacts_enabled: bool | None = None,
) -> list[str]:
    cmd = ["cuemap", "recall", "-p", project_id, "-l", str(args.limit), "--url", args.url]
    cmd.extend(["--semantic-mode", os.environ.get("CUEMAP_SEMANTIC_MODE", "hybrid")])
    if args.enable_alias_expansion:
        cmd.append("--enable-alias-expansion")
    use_cuebridge_artifacts = (
        args.enable_cuebridge_artifacts
        if cuebridge_artifacts_enabled is None
        else cuebridge_artifacts_enabled
    )
    if not use_cuebridge_artifacts:
        cmd.append("--disable-cuebridge-artifacts")
    cmd.append("--no-auto-reinforce")
    if args.disable_default_cuepacks:
        cmd.append("--disable-default-cuepacks")
    if args.cuepacks:
        cmd.extend(["--cuepacks", args.cuepacks])
    if query_time:
        cmd.extend(["--query-time", query_time])
    cmd.append(question)
    return cmd


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
    recall_res = run_cmd(
        build_recall_cmd(
            args,
            project_id,
            q_type,
            question,
            query_time,
            cuebridge_artifacts_enabled=cuebridge_artifacts_enabled,
        ),
        check=True,
    )
    recalled_contents = parse_recall_contents(recall_res.stdout)
    hit_rank, rel_array = calc_match(expected_contexts, recalled_contents)
    return {
        "name": name,
        "hit_rank": hit_rank,
        "rel_array": rel_array,
        "recalled_contents": recalled_contents,
        "ctx_tokens": context_token_count(recalled_contents),
        "expected_contexts": expected_contexts,
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
    attempts = [
        run_recall_attempt(
            args,
            project_id,
            q_type,
            question,
            query_time,
            expected_contexts,
            name=name,
            cuebridge_artifacts_enabled=cuebridge_artifacts_enabled,
        )
    ]

    return attempts[0], attempts


def evaluate():
    parser = argparse.ArgumentParser(
        description="Settled/background-aware CueMap LongMemEval harness. Assumes the CueMap server is already running."
    )
    parser.add_argument("--variant", choices=VARIANTS, default="core")
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--output", default=None)
    parser.add_argument("--url", default="http://127.0.0.1:8735")
    parser.add_argument("--recall-only", action="store_true")
    parser.add_argument("--no-wait-bg", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--enable-alias-expansion", action="store_true")
    parser.add_argument(
        "--enable-cuebridge-artifacts",
        action="store_true",
        help="Allow installed CueBridge artifacts during recall. Disabled by default for core evals.",
    )
    parser.add_argument("--no-auto-reinforce", action="store_true")
    parser.add_argument(
        "--cuepacks",
        default=None,
        help="Comma-separated CuePacks to pass to cuemap recall, e.g. default,memory-general or off.",
    )
    parser.add_argument(
        "--disable-default-cuepacks",
        action="store_true",
        help="Pass --disable-default-cuepacks to cuemap recall.",
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
    add_cuebridge_compare_args(parser)
    args = parser.parse_args()
    cuebridge_mode = cuebridge_compare_mode(args)
    cuebridge_enabled = cuebridge_mode != "off"

    output_path = args.output or default_output_path(args.variant)
    print(f"Variant: {args.variant}")
    cuepack_label = "off" if args.disable_default_cuepacks else (args.cuepacks or "default")
    print(f"CuePacks: {cuepack_label}")
    print("Server management: external (this script does not start or stop CueMap)")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {output_path}")
    if cuebridge_enabled:
        print(f"CueBridge compare mode: {cuebridge_mode}")

    with open(args.dataset, "r") as f:
        data = json.load(f)

    end_index = len(data) if args.max_records is None else min(len(data), args.start_index + args.max_records)
    records_to_test = list(enumerate(data))[args.start_index:end_index]

    project_mapping = {}
    if args.recall_only:
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"{output_path} not found; recall-only needs an existing output file")
        with open(output_path, "r") as f:
            old_results = json.load(f)
        project_mapping = {r["record_idx"]: r["project_id"] for r in old_results}

    results = []
    hit_at_1 = hit_at_5 = hit_at_10 = hit_at_20 = 0
    sum_recall_frac_5 = sum_recall_frac_10 = sum_recall_frac_20 = 0.0
    sum_recall_all_5 = sum_recall_all_10 = sum_recall_all_20 = 0.0
    sum_ndcg_5 = sum_ndcg_10 = sum_ndcg_20 = 0.0
    total_questions = 0
    context_tokens_by_question = []
    abstention_cases = 0
    stats_by_type = defaultdict(lambda: {
        "total": 0, "hit1": 0, "hit5": 0, "hit10": 0, "hit20": 0,
        "sum_recall_frac_5": 0.0, "sum_recall_frac_10": 0.0, "sum_recall_frac_20": 0.0,
        "sum_recall_all_5": 0.0, "sum_recall_all_10": 0.0, "sum_recall_all_20": 0.0,
        "sum_ndcg_5": 0.0, "sum_ndcg_10": 0.0, "sum_ndcg_20": 0.0,
        "context_tokens": [],
        "hit20_base": 0, "final_miss": 0,
    })
    abstention_by_type = defaultdict(int)
    final_misses = 0

    for record_idx, record in records_to_test:
        q_type = record.get("question_type", "Unknown")
        question = record.get("question", "")
        is_abstention = record.get("question_id", "").endswith("_abs")

        if is_abstention:
            abstention_cases += 1
            abstention_by_type[q_type] += 1
            print(f"\n--- Skipping Record {record_idx + 1}/{len(data)} | Type: {q_type} | Abstention ---")
            results.append({
                "record_idx": record_idx,
                "project_id": None,
                "variant": args.variant,
                "settle_status": None,
                "question_type": q_type,
                "question": question,
                "hit_rank": -1,
                "selected_recall_attempt": "skipped_abstention",
                "cuepacks": cuepack_label,
                "cuebridge_artifacts_enabled": args.enable_cuebridge_artifacts,
                "recall_attempts": [],
                "expected_contexts": [],
                "recalled_contents": [],
                "ctx_tokens": 0,
                "skipped": "abstention",
            })
            save_results(output_path, results)
            continue

        if args.recall_only:
            project_id = project_mapping.get(record_idx)
            if not project_id:
                continue
        else:
            project_id = f"eval_small_{args.variant.replace('-', '_')}_{record_idx}_{int(time.time())}"

        mode = "RECALL-ONLY" if args.recall_only else "FULL"
        print(f"\n--- Testing Record {record_idx + 1}/{len(data)} | Project: {project_id} | Mode: {mode} | Variant: {args.variant} ---")

        expected_context_lines = []
        sessions = record.get("haystack_sessions", [])
        for turn in sessions:
            for message in turn:
                if message.get("has_answer", False):
                    expected_context_lines.append(message.get("content", ""))

        settle_status = None
        if not args.recall_only:
            total_messages = sum(len(turn) for turn in sessions)
            print(f"Total messages to ingest for this record: {total_messages}")
            ingested_count = 0

            haystack_dates = record.get("haystack_dates", [])
            haystack_session_ids = record.get("haystack_session_ids", [])
            for turn_idx, turn in enumerate(sessions):
                source_date = haystack_dates[turn_idx] if turn_idx < len(haystack_dates) else None
                source_session_id = (
                    haystack_session_ids[turn_idx]
                    if turn_idx < len(haystack_session_ids)
                    else None
                )
                for message in turn:
                    role = message.get("role", "")
                    content = message.get("content", "")
                    text_to_add = f"{role}: {content}"
                    metadata = {"source_role": role}
                    if source_date:
                        metadata["source_date"] = source_date
                    if source_session_id:
                        metadata["source_session_id"] = source_session_id
                    add_cmd = [
                        "cuemap",
                        "add",
                        "-p",
                        project_id,
                        "--url",
                        args.url,
                        "--metadata",
                        json.dumps(metadata, separators=(",", ":")),
                    ]
                    if args.disable_default_cuepacks:
                        add_cmd.append("--disable-default-cuepacks")
                    if args.cuepacks:
                        add_cmd.extend(["--cuepacks", args.cuepacks])
                    add_cmd.append(text_to_add)
                    run_cmd(add_cmd, check=True)

                    ingested_count += 1
                    if ingested_count % 50 == 0:
                        print(f"Ingested {ingested_count}/{total_messages}...", end="\r", flush=True)

            print(f"\nIngested {total_messages} messages.")
            if not args.no_wait_bg:
                settle_status = wait_for_bg_jobs(project_id, args.url, args.timeout_seconds, args.poll_seconds)
            else:
                settle_status = job_status(project_id, args.url)

        if not question:
            continue

        artifact_metadata = None
        best_recall, recall_attempts = recall_once(
            args,
            project_id,
            q_type,
            question,
            record.get("question_date"),
            expected_context_lines,
            name="raw" if cuebridge_enabled else "base",
            cuebridge_artifacts_enabled=False if cuebridge_enabled else None,
        )
        raw_recall = best_recall
        should_build_cuebridge = (
            cuebridge_mode == "product"
            or (
                cuebridge_mode in {"oracle", "question_oracle"}
                and (
                    raw_recall["hit_rank"] <= 0
                    or raw_recall["hit_rank"] > args.cuebridge_target_rank_threshold
                )
            )
        )
        if should_build_cuebridge:
            run_dir = Path(args.cuebridge_run_root).expanduser() / f"longmemeval_{record_idx}_{project_id}"
            print(f"Building CueBridge artifacts for project {project_id} in {cuebridge_mode} mode...")
            artifact_metadata = build_cuebridge_artifacts(
                args,
                project_id,
                run_dir,
                target_texts=expected_context_lines if cuebridge_mode == "oracle" else None,
                target_questions=[
                    {
                        "id": f"longmemeval_{record_idx}",
                        "question": question,
                        "category": q_type,
                        "target_texts": expected_context_lines,
                    }
                ]
                if cuebridge_mode == "question_oracle"
                else None,
            )
            if not artifact_metadata.get("skipped"):
                best_recall, enhanced_attempts = recall_once(
                    args,
                    project_id,
                    q_type,
                    question,
                    record.get("question_date"),
                    expected_context_lines,
                    name="cuebridge",
                    cuebridge_artifacts_enabled=True,
                )
                recall_attempts.extend(enhanced_attempts)
        elif cuebridge_enabled:
            artifact_metadata = {
                "skipped": "raw_rank_within_target_threshold",
                "raw_rank": raw_recall["hit_rank"],
                "target_rank_threshold": args.cuebridge_target_rank_threshold,
            }

        recalled_contents = best_recall["recalled_contents"]
        hit_rank = best_recall["hit_rank"]
        rel_array = best_recall["rel_array"]
        ctx_tokens = best_recall["ctx_tokens"]
        raw_ctx_tokens = raw_recall["ctx_tokens"]
        recall_attempt_summaries = [
            {
                "name": attempt["name"],
                "hit_rank": attempt["hit_rank"],
                "hit_at_20": 0 < attempt["hit_rank"] <= 20,
                "ctx_tokens": attempt["ctx_tokens"],
            }
            for attempt in recall_attempts
        ]

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
            stats_by_type[q_type]["hit20_base"] += 1
        else:
            final_misses += 1
            stats_by_type[q_type]["final_miss"] += 1

        r5_frac, r5_all = calc_recall_for_contents(expected_context_lines, recalled_contents, 5)
        r10_frac, r10_all = calc_recall_for_contents(expected_context_lines, recalled_contents, 10)
        r20_frac, r20_all = calc_recall_for_contents(expected_context_lines, recalled_contents, 20)
        n5 = calc_ndcg_for_rel(rel_array, len(expected_context_lines), 5)
        n10 = calc_ndcg_for_rel(rel_array, len(expected_context_lines), 10)
        n20 = calc_ndcg_for_rel(rel_array, len(expected_context_lines), 20)
        raw_r5_frac, raw_r5_all = calc_recall_for_contents(expected_context_lines, raw_recall["recalled_contents"], 5)
        raw_r10_frac, raw_r10_all = calc_recall_for_contents(expected_context_lines, raw_recall["recalled_contents"], 10)
        raw_r20_frac, raw_r20_all = calc_recall_for_contents(expected_context_lines, raw_recall["recalled_contents"], 20)
        raw_n5 = calc_ndcg_for_rel(raw_recall["rel_array"], len(expected_context_lines), 5)
        raw_n10 = calc_ndcg_for_rel(raw_recall["rel_array"], len(expected_context_lines), 10)
        raw_n20 = calc_ndcg_for_rel(raw_recall["rel_array"], len(expected_context_lines), 20)

        sum_recall_frac_5 += r5_frac
        sum_recall_frac_10 += r10_frac
        sum_recall_frac_20 += r20_frac
        sum_recall_all_5 += r5_all
        sum_recall_all_10 += r10_all
        sum_recall_all_20 += r20_all
        sum_ndcg_5 += n5
        sum_ndcg_10 += n10
        sum_ndcg_20 += n20
        context_tokens_by_question.append(ctx_tokens)

        stats_by_type[q_type]["sum_recall_frac_5"] += r5_frac
        stats_by_type[q_type]["sum_recall_frac_10"] += r10_frac
        stats_by_type[q_type]["sum_recall_frac_20"] += r20_frac
        stats_by_type[q_type]["sum_recall_all_5"] += r5_all
        stats_by_type[q_type]["sum_recall_all_10"] += r10_all
        stats_by_type[q_type]["sum_recall_all_20"] += r20_all
        stats_by_type[q_type]["sum_ndcg_5"] += n5
        stats_by_type[q_type]["sum_ndcg_10"] += n10
        stats_by_type[q_type]["sum_ndcg_20"] += n20
        stats_by_type[q_type]["context_tokens"].append(ctx_tokens)
        stats_by_type[q_type]["total"] += 1
        total_questions += 1

        print(f"Q: {question}")
        if cuebridge_enabled:
            raw_rank = raw_recall["hit_rank"] if raw_recall["hit_rank"] > 0 else "MISS"
            enhanced_rank = best_recall["hit_rank"] if best_recall["hit_rank"] > 0 else "MISS"
            print(f"CueBridge compare: raw_rank={raw_rank} -> enhanced_rank={enhanced_rank}")
        if best_recall["name"] != "base":
            print(f"Selected recall attempt: {best_recall['name']}")
        print(
            f"Type: {q_type} | Hit Rank: {hit_rank if hit_rank > 0 else 'MISS'} | "
            f"NDCG@10: {n10:.2f} | Recall_Frac@10: {r10_frac:.2f} | "
            f"Recall_All@10: {r10_all:.2f} | CtxTokens: {ctx_tokens}"
        )

        results.append({
            "record_idx": record_idx,
            "project_id": project_id,
            "variant": args.variant,
            "settle_status": settle_status,
            "question_type": q_type,
            "question": question,
            "hit_rank": hit_rank,
            "selected_recall_attempt": best_recall["name"],
            "cuepacks": cuepack_label,
            "cuebridge_artifacts_enabled": args.enable_cuebridge_artifacts or cuebridge_enabled,
            "cuebridge_compare": cuebridge_enabled,
            "cuebridge_compare_mode": cuebridge_mode,
            "cuebridge_artifacts": artifact_metadata,
            "raw_hit_rank": raw_recall["hit_rank"],
            "raw_metrics": {
                "recall_frac_5": raw_r5_frac,
                "recall_frac_10": raw_r10_frac,
                "recall_frac_20": raw_r20_frac,
                "recall_all_5": raw_r5_all,
                "recall_all_10": raw_r10_all,
                "recall_all_20": raw_r20_all,
                "ndcg_5": raw_n5,
                "ndcg_10": raw_n10,
                "ndcg_20": raw_n20,
                "ctx_tokens": raw_ctx_tokens,
            },
            "enhanced_metrics": {
                "recall_frac_5": r5_frac,
                "recall_frac_10": r10_frac,
                "recall_frac_20": r20_frac,
                "recall_all_5": r5_all,
                "recall_all_10": r10_all,
                "recall_all_20": r20_all,
                "ndcg_5": n5,
                "ndcg_10": n10,
                "ndcg_20": n20,
                "ctx_tokens": ctx_tokens,
            },
            "raw_recalled_contents": raw_recall["recalled_contents"],
            "raw_ctx_tokens": raw_ctx_tokens,
            "cuebridge_delta": {
                "raw_rank": raw_recall["hit_rank"],
                "enhanced_rank": hit_rank,
                "rescued_at_20": not (0 < raw_recall["hit_rank"] <= 20) and 0 < hit_rank <= 20,
                "regressed_from_20": 0 < raw_recall["hit_rank"] <= 20 and not (0 < hit_rank <= 20),
            },
            "recall_attempts": recall_attempt_summaries,
            "expected_contexts": expected_context_lines,
            "recalled_contents": recalled_contents,
            "ctx_tokens": ctx_tokens,
        })
        save_results(output_path, results)

        if args.delete_project_after_record and not args.recall_only:
            try:
                delete_project(project_id, args.url)
                if args.delete_project_files_after_record:
                    delete_project_files(project_id, args.snapshots_dir, args.contents_dir)
                print(f"Deleted eval project: {project_id}")
            except Exception as exc:
                print(f"WARNING: Failed to delete eval project {project_id}: {exc}")

    print("\n============== SUMMARY ==============")
    print(f"\nTotal Base Questions: {total_questions} (Excluded {abstention_cases} Abstention Cases)")
    print("\n[ Recall_Any (Hit@K) ] - At least one relevant fact retrieved")
    print(f"Recall_Any@1:  {hit_at_1}/{total_questions} ({(hit_at_1 / total_questions * 100) if total_questions else 0:.1f}%)")
    print(f"Recall_Any@5:  {hit_at_5}/{total_questions} ({(hit_at_5 / total_questions * 100) if total_questions else 0:.1f}%)")
    print(f"Recall_Any@10: {hit_at_10}/{total_questions} ({(hit_at_10 / total_questions * 100) if total_questions else 0:.1f}%)")
    print(f"Recall_Any@20: {hit_at_20}/{total_questions} ({(hit_at_20 / total_questions * 100) if total_questions else 0:.1f}%)")

    print("\n[ Recall_All ] - All relevant facts for the query retrieved")
    print(f"Recall_All@5:  {(sum_recall_all_5 / total_questions * 100) if total_questions else 0:.1f}%")
    print(f"Recall_All@10: {(sum_recall_all_10 / total_questions * 100) if total_questions else 0:.1f}%")
    print(f"Recall_All@20: {(sum_recall_all_20 / total_questions * 100) if total_questions else 0:.1f}%")

    print("\n[ Recall_Frac ] - Average fraction of relevant facts retrieved")
    print(f"Recall_Frac@5:  {(sum_recall_frac_5 / total_questions * 100) if total_questions else 0:.1f}%")
    print(f"Recall_Frac@10: {(sum_recall_frac_10 / total_questions * 100) if total_questions else 0:.1f}%")
    print(f"Recall_Frac@20: {(sum_recall_frac_20 / total_questions * 100) if total_questions else 0:.1f}%")

    print("\n[ NDCG ] - Relevance ranked scoring")
    print(f"NDCG@5:  {(sum_ndcg_5 / total_questions * 100) if total_questions else 0:.1f}%")
    print(f"NDCG@10: {(sum_ndcg_10 / total_questions * 100) if total_questions else 0:.1f}%")
    print(f"NDCG@20: {(sum_ndcg_20 / total_questions * 100) if total_questions else 0:.1f}%")

    avg_ctx_tokens = sum(context_tokens_by_question) / total_questions if total_questions else 0
    print("\n[ Retrieved Context Tokens ] - Approx tokens from recalled memory text")
    print(f"CtxTokens Avg: {avg_ctx_tokens:.0f}")
    print(f"CtxTokens P50: {percentile(context_tokens_by_question, 50)}")
    print(f"CtxTokens P95: {percentile(context_tokens_by_question, 95)}")
    print(f"CtxTokens P99: {percentile(context_tokens_by_question, 99)}")
    print(f"CtxTokens Max: {max(context_tokens_by_question) if context_tokens_by_question else 0}")

    print("\n[ Recall Attempt Attribution @20 ]")
    attempt_label = "Selected pass" if cuebridge_enabled else "Base pass"
    print(f"{attempt_label} Hit@20: {hit_at_20}/{total_questions}")
    print(f"Final misses: {final_misses}/{total_questions}")

    print("\n============== BY QUESTION TYPE ==============")
    for q_type, s in stats_by_type.items():
        total = s["total"]
        if total == 0:
            continue
        print(f"{q_type} (Total: {total}):")
        print(f"  Recall_Any@1:  {s['hit1']}/{total} ({(s['hit1'] / total * 100):.1f}%)")
        print(f"  Recall_Any@5:  {s['hit5']}/{total} ({(s['hit5'] / total * 100):.1f}%)")
        print(f"  Recall_Any@10: {s['hit10']}/{total} ({(s['hit10'] / total * 100):.1f}%)")
        print(f"  Recall_All@5:  {(s['sum_recall_all_5'] / total * 100):.1f}%")
        print(f"  Recall_All@10: {(s['sum_recall_all_10'] / total * 100):.1f}%")
        print(f"  Recall_Frac@5:  {(s['sum_recall_frac_5'] / total * 100):.1f}%")
        print(f"  Recall_Frac@10: {(s['sum_recall_frac_10'] / total * 100):.1f}%")
        print(f"  NDCG@5:  {(s['sum_ndcg_5'] / total * 100):.1f}%")
        print(f"  NDCG@10: {(s['sum_ndcg_10'] / total * 100):.1f}%")
        type_ctx_tokens = s["context_tokens"]
        print(
            "  CtxTokens: "
            f"avg={(sum(type_ctx_tokens) / total):.0f}, "
            f"p95={percentile(type_ctx_tokens, 95)}"
        )
        print(
            "  Hit@20 source: "
            f"base={s['hit20_base']}, "
            f"final_miss={s['final_miss']}"
        )
        if abstention_by_type[q_type] > 0:
            print(f"  [Abstention Cases Excluded: {abstention_by_type[q_type]}]")

    print(f"\nSaved details to {output_path}")
    if cuebridge_enabled:
        print_cuebridge_delta_summary(results, limit=20)
    save_results(output_path, results)


if __name__ == "__main__":
    evaluate()
