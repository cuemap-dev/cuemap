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
import warnings
from collections import defaultdict
from pathlib import Path

from cuebridge_eval_utils import (
    add_cuebridge_compare_args,
    build_cuebridge_artifacts,
    cuebridge_compare_mode,
    print_cuebridge_delta_summary,
)

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

DATASET_PATH = str(Path(__file__).resolve().parents[1] / "data" / "locomo10.json")
RESULTS_DIR = str(Path(__file__).resolve().parents[1] / "results")


MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
}


def parse_locomo_date(date_str: str) -> str | None:
    if not date_str:
        return None
    match = re.search(r"on\s+(?P<day>\d{1,2})\s+(?P<month>[a-zA-Z]+),\s+(?P<year>\d{4})", date_str)
    if match:
        day = int(match.group("day"))
        month_name = match.group("month").lower()
        year = int(match.group("year"))
        month = MONTHS.get(month_name)
        if month:
            return f"{year:04d}-{month:02d}-{day:02d}"
    return None


def clean_output(text: str) -> str:
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def normalize_text(text: str) -> str:
    text = text.replace("assistant: ", "").replace("user: ", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def approx_token_count(text: str) -> int:
    # Cheap model-agnostic estimate for retrieved context budget reporting.
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


def calc_match(expected_context_groups: list[list[str]], recalled_contents: list[str]) -> tuple[int, list[int]]:
    hit_rank = -1
    rel_array = [0] * len(recalled_contents)
    matched_groups = set()

    for rank, r_content in enumerate(recalled_contents):
        rc_norm = normalize_text(r_content)
        for group_idx, group in enumerate(expected_context_groups):
            if group_idx in matched_groups:
                continue
            matched_group = False
            for expected in group:
                if " [Image caption: " in expected:
                    dialogue_part, caption_part = expected.split(" [Image caption: ", 1)
                    caption_part = caption_part.rstrip("]")
                    dialogue_norm = normalize_text(dialogue_part)
                    caption_norm = normalize_text(caption_part)
                    if (dialogue_norm in rc_norm or rc_norm in dialogue_norm) and (caption_norm in rc_norm):
                        matched_group = True
                        break
                else:
                    expected_norm = normalize_text(expected)
                    if expected_norm in rc_norm or rc_norm in expected_norm:
                        matched_group = True
                        break
            if matched_group:
                matched_groups.add(group_idx)
                rel_array[rank] = 1
                if hit_rank == -1:
                    hit_rank = rank + 1
                break

    return hit_rank, rel_array


def calc_recall_for_contents(expected_context_groups: list[list[str]], recalled_contents: list[str], k: int) -> tuple[float, float]:
    found_groups = set()
    for r_content in recalled_contents[:k]:
        rc_norm = normalize_text(r_content)
        for group_idx, group in enumerate(expected_context_groups):
            if group_idx in found_groups:
                continue
            matched_group = False
            for expected in group:
                if " [Image caption: " in expected:
                    dialogue_part, caption_part = expected.split(" [Image caption: ", 1)
                    caption_part = caption_part.rstrip("]")
                    dialogue_norm = normalize_text(dialogue_part)
                    caption_norm = normalize_text(caption_part)
                    if (dialogue_norm in rc_norm or rc_norm in dialogue_norm) and (caption_norm in rc_norm):
                        matched_group = True
                        break
                else:
                    expected_norm = normalize_text(expected)
                    if expected_norm in rc_norm or rc_norm in expected_norm:
                        matched_group = True
                        break
            if matched_group:
                found_groups.add(group_idx)
                
    total_expected = max(1, len(expected_context_groups))
    frac = len(found_groups) / total_expected
    all_found = 1.0 if len(found_groups) == len(expected_context_groups) else 0.0
    return frac, all_found


def calc_ndcg_for_rel(rel_array: list[int], expected_count: int, k: int) -> float:
    dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(rel_array[:k]))
    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(min(k, expected_count)))
    return dcg / idcg if idcg > 0 else 0.0


def score_recall_attempt(attempt: dict, expected_count: int) -> tuple:
    hit_rank = attempt["hit_rank"]
    recalled_contents = attempt["recalled_contents"]
    rel_array = attempt["rel_array"]
    r5_frac, r5_all = calc_recall_for_contents(attempt["expected_context_groups"], recalled_contents, 5)
    r10_frac, r10_all = calc_recall_for_contents(attempt["expected_context_groups"], recalled_contents, 10)
    r20_frac, r20_all = calc_recall_for_contents(attempt["expected_context_groups"], recalled_contents, 20)
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


def default_output_path() -> str:
    return f"{RESULTS_DIR}/locomo_results.json"


def save_results(output_path: str, results: list[dict]) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def explain_matching(expected_context_groups: list[list[str]], recalled_contents: list[str]) -> list[dict]:
    explanations = []
    for rank, r_content in enumerate(recalled_contents):
        rc_norm = normalize_text(r_content)
        matched = False
        details = ""
        matched_expected = None
        
        for exp_idx, group in enumerate(expected_context_groups):
            matched_group = False
            for expected in group:
                if " [Image caption: " in expected:
                    dialogue_part, caption_part = expected.split(" [Image caption: ", 1)
                    caption_part = caption_part.rstrip("]")
                    dialogue_norm = normalize_text(dialogue_part)
                    caption_norm = normalize_text(caption_part)
                    
                    dia_in_rc = dialogue_norm in rc_norm
                    rc_in_dia = rc_norm in dialogue_norm
                    cap_in_rc = caption_norm in rc_norm
                    
                    if (dia_in_rc or rc_in_dia) and cap_in_rc:
                        matched_group = True
                        matched_expected = expected
                        details = (
                            f"Multimodal match with expected group[{exp_idx}] item:\n"
                            f"  - Dialogue part normalized: '{dialogue_norm}'\n"
                            f"  - Caption part normalized: '{caption_norm}'\n"
                            f"  - Dialogue match: {'dialogue_norm in rc_norm' if dia_in_rc else 'rc_norm in dialogue_norm'}\n"
                            f"  - Caption match: caption_norm in rc_norm"
                        )
                        break
                else:
                    expected_norm = normalize_text(expected)
                    exp_in_rc = expected_norm in rc_norm
                    rc_in_exp = rc_norm in expected_norm
                    
                    if exp_in_rc or rc_in_exp:
                        matched_group = True
                        matched_expected = expected
                        details = (
                            f"Text match with expected group[{exp_idx}] item:\n"
                            f"  - Expected normalized: '{expected_norm}'\n"
                            f"  - Match type: {'expected_norm in rc_norm' if exp_in_rc else 'rc_norm in expected_norm'}"
                        )
                        break
            if matched_group:
                matched = True
                break
        
        explanations.append({
            "rank": rank + 1,
            "raw_content": r_content,
            "norm_content": rc_norm,
            "matched": matched,
            "matched_expected": matched_expected,
            "details": details
        })
    return explanations


def generate_comparison_report(results: list[dict], report_path: str) -> None:
    missed_questions = []
    total_questions = 0
    hit_at_1 = hit_at_5 = hit_at_10 = hit_at_20 = 0
    
    for r in results:
        record_idx = r["record_idx"]
        sample_id = r["sample_id"]
        for q_idx, q_res in enumerate(r["probing_results"]):
            total_questions += 1
            hit_rank = q_res["hit_rank"]
            if hit_rank == 1:
                hit_at_1 += 1
            if 0 < hit_rank <= 5:
                hit_at_5 += 1
            if 0 < hit_rank <= 10:
                hit_at_10 += 1
            if 0 < hit_rank <= 20:
                hit_at_20 += 1
            
            if hit_rank == -1 or hit_rank > 20:
                missed_questions.append({
                    "record_idx": record_idx,
                    "sample_id": sample_id,
                    "q_idx": q_idx + 1,
                    "question": q_res["question"],
                    "category": q_res["category"]
                })

    md = []
    md.append("# CueMap LoCoMo Retrieval Comparison Report\n")
    md.append("## Summary Table\n")
    md.append("| Metric | Count / Total | Percentage |")
    md.append("|---|---|---|")
    if total_questions > 0:
        md.append(f"| **Recall_Any@1** | {hit_at_1}/{total_questions} | {(hit_at_1 / total_questions * 100):.1f}% |")
        md.append(f"| **Recall_Any@5** | {hit_at_5}/{total_questions} | {(hit_at_5 / total_questions * 100):.1f}% |")
        md.append(f"| **Recall_Any@10** | {hit_at_10}/{total_questions} | {(hit_at_10 / total_questions * 100):.1f}% |")
        md.append(f"| **Recall_Any@20** | {hit_at_20}/{total_questions} | {(hit_at_20 / total_questions * 100):.1f}% |")
    else:
        md.append("| No questions evaluated | - | - |")
    md.append("\n---\n")

    md.append("## Missed Questions List\n")
    if missed_questions:
        md.append(f"CueMap completely missed **{len(missed_questions)}** out of {total_questions} questions (Hit@20 Miss).\n")
        for m in missed_questions:
            md.append(f"- **Record {m['record_idx'] + 1}** (ID: {m['sample_id']}), Q{m['q_idx']}: \"{m['question']}\" (Category: `{m['category']}`)")
    else:
        md.append("Incredible! No missed questions (100% Hit@20)!\n")
    md.append("\n---\n")

    md.append("## Detailed Probing Matches & Explanations\n")
    for r in results:
        record_idx = r["record_idx"]
        sample_id = r["sample_id"]
        project_id = r["project_id"]
        
        md.append(f"### Record {record_idx + 1} (ID: {sample_id})")
        md.append(f"CueMap Project: `{project_id}`\n")
        
        for q_idx, q_res in enumerate(r["probing_results"]):
            question = q_res["question"]
            category = q_res["category"]
            hit_rank = q_res["hit_rank"]
            selected_attempt = q_res.get("selected_recall_attempt", "unknown")
            expected_context_groups = q_res["expected_context_groups"]
            recalled_contents = q_res["recalled_contents"]
            
            status_str = f"HIT (Rank {hit_rank})" if hit_rank > 0 else "MISS"
            status_emoji = "✅" if hit_rank > 0 else "❌"
            
            md.append(f"#### Q{q_idx + 1}: \"{question}\"")
            md.append(f"- **Category**: `{category}`")
            md.append(f"- **Status**: {status_emoji} **{status_str}**")
            md.append(f"- **Selected Retrieval Mode**: `{selected_attempt}`\n")
            
            md.append("**Expected Context Groups (Alternatives):**")
            for idx, group in enumerate(expected_context_groups):
                md.append(f"{idx + 1}. **Group {idx + 1}**:")
                for alt_idx, expected in enumerate(group):
                    if " [Image caption: " in expected:
                        dialogue_part, caption_part = expected.split(" [Image caption: ", 1)
                        caption_part = caption_part.rstrip("]")
                        dialogue_norm = normalize_text(dialogue_part)
                        caption_norm = normalize_text(caption_part)
                        md.append(f"   - Alt {alt_idx + 1}: `{expected}`")
                        md.append(f"     - *Dialogue Norm*: `'{dialogue_norm}'`")
                        md.append(f"     - *Caption Norm*: `'{caption_norm}'`")
                    else:
                        expected_norm = normalize_text(expected)
                        md.append(f"   - Alt {alt_idx + 1}: `{expected}`")
                        md.append(f"     - *Text Norm*: `'{expected_norm}'`")
            md.append("")
            
            md.append("**Recalled Contents & Comparison Details (Top 20):**")
            explanations = explain_matching(expected_context_groups, recalled_contents)
            for exp in explanations:
                rank = exp["rank"]
                raw = exp["raw_content"].replace("\n", " ")
                norm = exp["norm_content"]
                matched_emoji = "🎯" if exp["matched"] else "⚪"
                
                md.append(f"- **Rank {rank}** {matched_emoji}")
                md.append(f"  - **Raw**: `{raw}`")
                md.append(f"  - **Normalized**: `{norm}`")
                if exp["matched"]:
                    md.append("  - **Match Explanation**:")
                    indented_details = "\n".join("    " + line for line in exp["details"].split("\n"))
                    md.append(indented_details)
            md.append("\n---\n")

    with open(report_path, "w") as f:
        f.write("\n".join(md))
    print(f"Generated comparison report at {report_path}")


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
    if args.depth is not None:
        cmd.extend(["--depth", str(args.depth)])
    if args.expansion_depth is not None:
        cmd.extend(["--expansion-depth", str(args.expansion_depth)])
    if args.ordered_reconstruction is not None:
        cmd.extend(["--ordered-reconstruction", args.ordered_reconstruction])
    if args.evidence_coverage is not None:
        cmd.extend(["--evidence-coverage", args.evidence_coverage])
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
    expected_context_groups: list[list[str]],
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
    hit_rank, rel_array = calc_match(expected_context_groups, recalled_contents)
    return {
        "name": name,
        "hit_rank": hit_rank,
        "rel_array": rel_array,
        "recalled_contents": recalled_contents,
        "expected_context_groups": expected_context_groups,
    }


def recall_once(
    args,
    project_id: str,
    q_type: str,
    question: str,
    query_time: str | None,
    expected_context_groups: list[list[str]],
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
            expected_context_groups,
            name=name,
            cuebridge_artifacts_enabled=cuebridge_artifacts_enabled,
        )
    ]

    return attempts[0], attempts


def get_flat_chat(conversation: dict) -> list[dict]:
    flat_chat = []
    # Identify all sessions
    session_keys = []
    for k in conversation.keys():
        if k.startswith("session_") and not k.endswith("_date_time") and not k.endswith("_observation") and not k.endswith("_summary"):
            try:
                num = int(k.split("_")[1])
                session_keys.append((num, k))
            except ValueError:
                pass
    # Sort chronologically
    session_keys.sort()
    for num, k in session_keys:
        session_turns = conversation.get(k, [])
        session_date = conversation.get(f"{k}_date_time", "")
        for session_turn_idx, turn in enumerate(session_turns):
            if isinstance(turn, dict):
                turn["session_num"] = num
                turn["session_turn_index"] = session_turn_idx
                turn["session_date"] = session_date
                flat_chat.append(turn)
    return flat_chat


def make_expected_string(turn: dict) -> str:
    speaker = turn.get("speaker", "")
    text = turn.get("text", "").strip()
    text_to_add = f"{speaker}: {text}" if speaker and text else text
    blip_caption = turn.get("blip_caption")
    if blip_caption:
        text_to_add += f" [Image caption: {blip_caption}]"
    return text_to_add


def get_expected_context_groups(q: dict, flat_chat: list, window: int = 2) -> list[list[str]]:
    raw_evidence_ids = q.get("evidence", [])
    if isinstance(raw_evidence_ids, str):
        raw_evidence_ids = [raw_evidence_ids]
    
    evidence_ids = []
    for x in raw_evidence_ids:
        if isinstance(x, str):
            parts = re.split(r"[;,\s]+", x)
            evidence_ids.extend(parts)
        else:
            evidence_ids.append(x)

    # Map dia_id -> index in flat_chat
    dia_to_idx = {}
    for idx, turn in enumerate(flat_chat):
        dia_id = turn.get("dia_id")
        if dia_id is not None:
            dia_to_idx[str(dia_id).strip()] = idx

    # Group evidence ID indices by their session
    session_groups = defaultdict(list)
    for ev_id in evidence_ids:
        ev_id_clean = str(ev_id).strip()
        if ev_id_clean in dia_to_idx:
            idx = dia_to_idx[ev_id_clean]
            target_turn = flat_chat[idx]
            target_session = target_turn.get("session_num")
            
            # Sibling indices in the same session within +/- window
            start_i = max(0, idx - window)
            end_i = min(len(flat_chat) - 1, idx + window)
            
            turn_indices = set()
            for i in range(start_i, end_i + 1):
                if flat_chat[i].get("session_num") == target_session:
                    turn_indices.add(i)
            
            if turn_indices:
                session_groups[target_session].append(turn_indices)

    groups = []

    # For each session, merge overlapping index sets
    for session_num, index_sets in session_groups.items():
        merged = []
        for s in index_sets:
            # Find any existing sets in merged that overlap with s
            overlapping_indices = []
            for idx, existing in enumerate(merged):
                if not existing.isdisjoint(s):
                    overlapping_indices.append(idx)
            
            if not overlapping_indices:
                merged.append(s)
            else:
                # Merge all overlapping sets and s into a single set
                new_set = s.copy()
                for idx in sorted(overlapping_indices, reverse=True):
                    new_set.update(merged.pop(idx))
                merged.append(new_set)
        
        # Convert merged index sets back to expected context strings
        for index_set in merged:
            sorted_indices = sorted(list(index_set))
            group_strings = []
            for i in sorted_indices:
                s_str = make_expected_string(flat_chat[i])
                if s_str and s_str not in group_strings:
                    group_strings.append(s_str)
            if group_strings:
                groups.append(group_strings)

    # Fallback to answer if groups is empty
    if not groups and q.get("answer"):
        ans = q["answer"]
        ans_strings = []
        if isinstance(ans, str):
            ans_strings.append(ans.strip())
        elif isinstance(ans, list):
            ans_strings.extend([str(x).strip() for x in ans])
        ans_strings = [x for x in ans_strings if x]
        if ans_strings:
            groups.append(ans_strings)

    return groups


def score_locomo_question(
    args,
    project_id: str,
    flat_chat: list,
    q_idx: int,
    q: dict,
    last_date: str | None,
    *,
    phase_name: str,
    cuebridge_artifacts_enabled: bool | None,
) -> dict | None:
    question = q.get("question", "")
    category_mapping = {
        1: "single-hop",
        2: "multi-hop",
        3: "temporal-reasoning",
        4: "common-sense",
        5: "adversarial",
    }
    cat_val = q.get("category", "Unknown")
    q_type = category_mapping.get(cat_val, str(cat_val))

    if not question:
        return None

    expected_context_groups = get_expected_context_groups(q, flat_chat, window=args.evidence_window)
    if not expected_context_groups:
        print(f"WARNING: no expected contexts found for question '{question}', skipping scoring.")
        return None

    best_recall, recall_attempts = recall_once(
        args,
        project_id,
        q_type,
        question,
        last_date,
        expected_context_groups,
        name=phase_name,
        cuebridge_artifacts_enabled=cuebridge_artifacts_enabled,
    )
    recalled_contents = best_recall["recalled_contents"]
    hit_rank = best_recall["hit_rank"]
    rel_array = best_recall["rel_array"]
    ctx_tokens = context_token_count(recalled_contents)
    ctx_chars = sum(len(content) for content in recalled_contents)
    r5_frac, r5_all = calc_recall_for_contents(expected_context_groups, recalled_contents, 5)
    r10_frac, r10_all = calc_recall_for_contents(expected_context_groups, recalled_contents, 10)
    r20_frac, r20_all = calc_recall_for_contents(expected_context_groups, recalled_contents, 20)
    n5 = calc_ndcg_for_rel(rel_array, len(expected_context_groups), 5)
    n10 = calc_ndcg_for_rel(rel_array, len(expected_context_groups), 10)
    n20 = calc_ndcg_for_rel(rel_array, len(expected_context_groups), 20)

    print(
        f"  [{q_idx + 1}] {phase_name} | Type: {q_type} | "
        f"Hit Rank: {hit_rank if hit_rank > 0 else 'MISS'} | "
        f"NDCG@10: {n10:.2f} | Recall_Frac@10: {r10_frac:.2f} | CtxTokens: {ctx_tokens}"
    )
    use_cuebridge_artifacts = (
        args.enable_cuebridge_artifacts
        if cuebridge_artifacts_enabled is None
        else cuebridge_artifacts_enabled
    )

    return {
        "question": question,
        "category": q_type,
        "hit_rank": hit_rank,
        "selected_recall_attempt": best_recall["name"],
        "cuebridge_artifacts_enabled": bool(use_cuebridge_artifacts),
        "recall_attempts": [
            {
                "name": attempt["name"],
                "hit_rank": attempt["hit_rank"],
                "hit_at_20": 0 < attempt["hit_rank"] <= 20,
            }
            for attempt in recall_attempts
        ],
        "expected_context_groups": expected_context_groups,
        "recalled_contents": recalled_contents,
        "rel_array": rel_array,
        "ctx_tokens": ctx_tokens,
        "ctx_chars": ctx_chars,
        "ctx_items": len(recalled_contents),
        "recall_frac_5": r5_frac,
        "recall_frac_10": r10_frac,
        "recall_frac_20": r20_frac,
        "recall_all_5": r5_all,
        "recall_all_10": r10_all,
        "recall_all_20": r20_all,
        "ndcg_5": n5,
        "ndcg_10": n10,
        "ndcg_20": n20,
    }


def evaluate():
    parser = argparse.ArgumentParser(
        description="Settled/background-aware CueMap LoCoMo Memory Benchmark harness."
    )
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--url", default="http://127.0.0.1:8735")
    parser.add_argument("--recall-only", action="store_true")
    parser.add_argument("--no-wait-bg", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Pass through cuemap recall --depth for multi-hop recall. Omitted by default to use the engine default.",
    )
    parser.add_argument(
        "--expansion-depth",
        type=int,
        default=None,
        help="Pass through cuemap recall --expansion-depth to include neighboring chunk/context memories around each hit.",
    )
    parser.add_argument(
        "--ordered-reconstruction",
        choices=["off", "auto", "force"],
        default=None,
        help="Pass through cuemap recall --ordered-reconstruction. Omitted by default to use the engine default.",
    )
    parser.add_argument(
        "--evidence-coverage",
        choices=["off", "auto", "force"],
        default=None,
        help="Pass through cuemap recall --evidence-coverage. Omitted by default to use the engine default.",
    )
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
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--evidence-window",
        type=int,
        default=2,
        help="Number of sibling turns in either direction in the same session to allow as alternative matches for a gold turn.",
    )
    add_cuebridge_compare_args(parser)
    args = parser.parse_args()
    cuebridge_mode = cuebridge_compare_mode(args)
    cuebridge_enabled = cuebridge_mode != "off"

    output_path = args.output or default_output_path()
    print("Dataset: LoCoMo benchmark (locomo10.json)")
    print("Server management: external (assumes running server on url)")
    print(f"Output path: {output_path}")
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
    stats_by_type = defaultdict(lambda: {
        "total": 0, "hit1": 0, "hit5": 0, "hit10": 0, "hit20": 0,
        "sum_recall_frac_5": 0.0, "sum_recall_frac_10": 0.0, "sum_recall_frac_20": 0.0,
        "sum_recall_all_5": 0.0, "sum_recall_all_10": 0.0, "sum_recall_all_20": 0.0,
        "sum_ndcg_5": 0.0, "sum_ndcg_10": 0.0, "sum_ndcg_20": 0.0,
        "context_tokens": [],
        "hit20_base": 0, "final_miss": 0,
    })
    final_misses = 0

    for record_idx, record in records_to_test:
        sample_id = record.get("sample_id", f"conv_{record_idx}")
        if args.recall_only:
            project_id = project_mapping.get(record_idx)
            if not project_id:
                continue
        else:
            project_id = f"eval_locomo_{sample_id}_{int(time.time())}"

        mode = "RECALL-ONLY" if args.recall_only else "FULL"
        print(f"\n--- Testing Record {record_idx + 1}/{len(data)} | Project: {project_id} | Mode: {mode} | ID: {sample_id} ---")

        conversation = record.get("conversation", {})
        flat_chat = get_flat_chat(conversation)

        settle_status = None
        if not args.recall_only:
            total_messages = len(flat_chat)
            print(f"Total messages to ingest for this record: {total_messages}")
            ingested_count = 0

            for turn_idx, turn in enumerate(flat_chat):
                speaker = turn.get("speaker", "")
                text = turn.get("text", "")
                dia_id = turn.get("dia_id", "")
                
                role = speaker
                # Format turn as text
                text_to_add = f"{speaker}: {text}"
                metadata = {
                    "source_role": role,
                    "source_session_id": f"{sample_id}:session_{turn.get('session_num')}",
                    "source_turn_index": turn.get("session_turn_index", turn_idx),
                    "source_chat_id": sample_id,
                    "dia_id": dia_id,
                    "session_num": turn.get("session_num")
                }
                
                # Retrieve session timestamp if available
                session_date = turn.get("session_date")
                if session_date:
                    parsed_date = parse_locomo_date(session_date)
                    if parsed_date:
                        metadata["source_date"] = parsed_date
                
                # Append BLIP image caption if turn has a multimodal caption
                blip_caption = turn.get("blip_caption")
                if blip_caption:
                    text_to_add += f" [Image caption: {blip_caption}]"
                    metadata["has_image"] = True

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

                ingested_count += 1
                if ingested_count % 50 == 0:
                    print(f"Ingested {ingested_count}/{total_messages}...", end="\r", flush=True)

            print(f"\nIngested {total_messages} messages.")
            if not args.no_wait_bg:
                settle_status = wait_for_bg_jobs(project_id, args.url, args.timeout_seconds, args.poll_seconds)

        # Resolve reference query time from the last session date
        last_date = None
        for turn in reversed(flat_chat):
            session_date = turn.get("session_date")
            if session_date:
                parsed = parse_locomo_date(session_date)
                if parsed:
                    last_date = parsed
                    break

        # Process LoCoMo QA questions
        qa_list = record.get("qa", [])
        print(f"Running {len(qa_list)} probing questions...")

        baseline_results = []
        for q_idx, q in enumerate(qa_list):
            scored = score_locomo_question(
                args,
                project_id,
                flat_chat,
                q_idx,
                q,
                last_date,
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
                        groups = raw_result["expected_context_groups"]
                        if cuebridge_mode == "oracle":
                            for group in groups:
                                target_texts.extend(group)
                        else:
                            target_questions.append(
                                {
                                    "id": f"locomo_{record_idx}_{len(target_questions) + 1:04d}",
                                    "question": raw_result["question"],
                                    "category": raw_result["category"],
                                    "target_texts": [text for group in groups for text in group],
                                }
                            )

            if cuebridge_mode == "product" or target_texts or target_questions:
                run_dir = Path(args.cuebridge_run_root).expanduser() / f"locomo_{record_idx}_{project_id}"
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
                artifact_metadata = build_cuebridge_artifacts(
                    args,
                    project_id,
                    run_dir,
                    target_texts=target_texts if cuebridge_mode == "oracle" else None,
                    target_questions=target_questions if cuebridge_mode == "question_oracle" else None,
                )
                if not artifact_metadata.get("skipped"):
                    enhanced_results = []
                    for q_idx, q in enumerate(qa_list):
                        scored = score_locomo_question(
                            args,
                            project_id,
                            flat_chat,
                            q_idx,
                            q,
                            last_date,
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
            ctx_tokens = scored_result["ctx_tokens"]
            ctx_chars = scored_result["ctx_chars"]
            r5_frac = scored_result["recall_frac_5"]
            r10_frac = scored_result["recall_frac_10"]
            r20_frac = scored_result["recall_frac_20"]
            r5_all = scored_result["recall_all_5"]
            r10_all = scored_result["recall_all_10"]
            r20_all = scored_result["recall_all_20"]
            n5 = scored_result["ndcg_5"]
            n10 = scored_result["ndcg_10"]
            n20 = scored_result["ndcg_20"]

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

        results.append({
            "record_idx": record_idx,
            "project_id": project_id,
            "sample_id": sample_id,
            "settle_status": settle_status,
            "cuebridge_compare": cuebridge_enabled,
            "cuebridge_compare_mode": cuebridge_mode,
            "cuebridge_artifacts": artifact_metadata,
            "probing_results": record_results,
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
    print(f"\nTotal Probing Questions Evaluated: {total_questions}")
    if total_questions > 0:
        print("\n[ Recall_Any (Hit@K) ] - At least one relevant fact retrieved")
        print(f"Recall_Any@1:  {hit_at_1}/{total_questions} ({(hit_at_1 / total_questions * 100):.1f}%)")
        print(f"Recall_Any@5:  {hit_at_5}/{total_questions} ({(hit_at_5 / total_questions * 100):.1f}%)")
        print(f"Recall_Any@10: {hit_at_10}/{total_questions} ({(hit_at_10 / total_questions * 100):.1f}%)")
        print(f"Recall_Any@20: {hit_at_20}/{total_questions} ({(hit_at_20 / total_questions * 100):.1f}%)")

        print("\n[ Recall_All ] - All relevant facts for the query retrieved")
        print(f"Recall_All@5:  {(sum_recall_all_5 / total_questions * 100):.1f}%")
        print(f"Recall_All@10: {(sum_recall_all_10 / total_questions * 100):.1f}%")
        print(f"Recall_All@20: {(sum_recall_all_20 / total_questions * 100):.1f}%")

        print("\n[ Recall_Frac ] - Average fraction of relevant facts retrieved")
        print(f"Recall_Frac@5:  {(sum_recall_frac_5 / total_questions * 100):.1f}%")
        print(f"Recall_Frac@10: {(sum_recall_frac_10 / total_questions * 100):.1f}%")
        print(f"Recall_Frac@20: {(sum_recall_frac_20 / total_questions * 100):.1f}%")

        print("\n[ NDCG ] - Relevance ranked scoring")
        print(f"NDCG@5:  {(sum_ndcg_5 / total_questions * 100):.1f}%")
        print(f"NDCG@10: {(sum_ndcg_10 / total_questions * 100):.1f}%")
        print(f"NDCG@20: {(sum_ndcg_20 / total_questions * 100):.1f}%")

        avg_ctx_tokens = sum(context_tokens_by_question) / total_questions
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

    print(f"\nSaved details to {output_path}")
    if cuebridge_enabled:
        print_cuebridge_delta_summary(results, limit=20)
    
    # Generate verbose comparison report
    report_path = str(Path(output_path).with_name("locomo_comparison_report.md"))
    generate_comparison_report(results, report_path)


if __name__ == "__main__":
    evaluate()
