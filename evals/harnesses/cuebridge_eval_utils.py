import argparse
import hashlib
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
from collections import Counter
from pathlib import Path
from typing import Any


CUEMAP_ROOT = Path(__file__).resolve().parents[2]
CUEBRIDGE = Path(os.environ.get("CUEBRIDGE_CLI", str(CUEMAP_ROOT / "cuebridge" / "cuebridge.py")))
DEFAULT_CUEBRIDGE_RUN_ROOT = CUEMAP_ROOT / "evals" / "cuebridge_runs"
CUEBRIDGE_ARTIFACT_STEPS = (
    "analyze-project",
    "question-generation",
    "score-questions",
    "propose-fixes",
    "validate-proposals",
    "compile",
    "validate-artifact",
    "install",
    "enhanced-recall",
)


def cuebridge_step_index(step: str) -> int:
    try:
        return CUEBRIDGE_ARTIFACT_STEPS.index(step)
    except ValueError as exc:
        raise ValueError(f"Unknown CueBridge start step: {step}") from exc


def cuebridge_should_run(start_step: str, step: str) -> bool:
    return cuebridge_step_index(step) >= cuebridge_step_index(start_step)


def require_resume_file(path: Path, *, step: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Cannot resume at {step}: required file is missing: {path}")


def run_pipeline_cmd(cmd: list[str], *, timeout: int | None = None) -> subprocess.CompletedProcess:
    print("+ " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    if result.stdout.strip():
        print(result.stdout.rstrip())
    if result.stderr.strip():
        print(result.stderr.rstrip())
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_cuebridge_recall_option_args(cmd: list[str], args: argparse.Namespace) -> None:
    for attr, flag in (
        ("parent_fusion", "--parent-fusion"),
        ("parent_fusion_limit", "--parent-fusion-limit"),
        ("parent_fusion_min_chunks", "--parent-fusion-min-chunks"),
        ("ordered_reconstruction", "--ordered-reconstruction"),
        ("ordered_reconstruction_limit", "--ordered-reconstruction-limit"),
        ("ordered_session_scan_limit", "--ordered-session-scan-limit"),
        ("ordered_max_sessions", "--ordered-max-sessions"),
        ("evidence_coverage", "--evidence-coverage"),
        ("evidence_coverage_limit", "--evidence-coverage-limit"),
        ("evidence_coverage_session_scan_limit", "--evidence-coverage-session-scan-limit"),
        ("evidence_coverage_max_sessions", "--evidence-coverage-max-sessions"),
    ):
        if hasattr(args, attr):
            cmd.extend([flag, str(getattr(args, attr))])


def add_cuebridge_compare_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("CueBridge compare")
    group.add_argument(
        "--compare-cuebridge",
        action="store_true",
        help=(
            "Legacy alias for --compare-cuebridge-oracle. Runs raw recall first, "
            "builds artifacts from failed benchmark gold evidence, then reruns recall."
        ),
    )
    group.add_argument(
        "--compare-cuebridge-product",
        action="store_true",
        help=(
            "Run raw recall first, build/install CueBridge artifacts from project memories only, "
            "then rerun recall. This mode does not use benchmark gold evidence."
        ),
    )
    group.add_argument(
        "--compare-cuebridge-oracle",
        action="store_true",
        help=(
            "Diagnostic mode: run raw recall first, build/install CueBridge artifacts from "
            "gold evidence for benchmark questions ranked worse than --cuebridge-target-rank-threshold, "
            "then rerun recall. Do not publish this as product-mode accuracy."
        ),
    )
    group.add_argument(
        "--compare-cuebridge-question-oracle",
        action="store_true",
        help=(
            "Diagnostic mode: use the actual benchmark question text plus matched gold memories "
            "as targeted weak questions, then build/install CueBridge artifacts and rerun recall. "
            "This directly tests whether CueBridge can bridge known eval questions; do not publish "
            "this as product-mode accuracy."
        ),
    )
    group.add_argument("--cuebridge-run-root", default=str(DEFAULT_CUEBRIDGE_RUN_ROOT))
    group.add_argument("--cuebridge-python", default="python")
    group.add_argument(
        "--cuebridge-provider",
        choices=["auto", "llama-cpp", "llama-server", "openai-compatible"],
        default="openai-compatible",
    )
    group.add_argument("--llama-binary", default="llama-cli")
    group.add_argument("--llama-model", default="")
    group.add_argument("--llama-n-predict", type=int, default=1024)
    group.add_argument("--llama-temp", type=float, default=0.0)
    group.add_argument("--llama-top-p", type=float, default=1.0)
    group.add_argument("--llama-seed", type=int, default=42)
    group.add_argument("--llama-ctx-size", type=int, default=8192)
    group.add_argument("--llama-timeout-seconds", type=int, default=300)
    group.add_argument("--llama-display-prompt", action="store_true")
    group.add_argument("--llama-extra-arg", action="append", default=[])
    group.add_argument("--llama-server-binary", default="llama-server")
    group.add_argument("--llama-server-host", default="127.0.0.1")
    group.add_argument("--llama-server-port", type=int, default=8088)
    group.add_argument("--llama-server-start-timeout-seconds", type=int, default=180)
    group.add_argument("--llama-server-extra-arg", action="append", default=[])
    group.add_argument("--openai-base-url", default="http://127.0.0.1:1234/v1")
    group.add_argument("--openai-model", default="qwen3-4b-cuebridge")
    group.add_argument("--openai-api-key", default="local")
    group.add_argument(
        "--openai-extra-param",
        action="append",
        default=[],
        help="Repeatable KEY=JSON_VALUE forwarded to CueBridge OpenAI-compatible calls, e.g. reasoning={\"enabled\":false}.",
    )
    group.add_argument("--continue-on-model-error", action="store_true", default=True)
    group.add_argument("--cuebridge-max-samples", type=int, default=500)
    group.add_argument("--cuebridge-max-jobs", type=int, default=200)
    group.add_argument("--cuebridge-job-offset", type=int, default=0)
    group.add_argument("--cuebridge-max-fix-cases", type=int, default=1000)
    group.add_argument("--cuebridge-case-offset", type=int, default=0)
    group.add_argument("--cuebridge-question-concurrency", type=int, default=1)
    group.add_argument("--cuebridge-question-batch-size", type=int, default=1)
    group.add_argument("--cuebridge-question-underfill-retries", type=int, default=2)
    group.add_argument("--cuebridge-fix-concurrency", type=int, default=1)
    group.add_argument("--cuebridge-fix-batch-size", type=int, default=1)
    group.add_argument("--cuebridge-progress-every", type=int, default=1)
    group.add_argument("--cuebridge-page-size", type=int, default=1000)
    group.add_argument("--cuebridge-salient-cue-limit", type=int, default=24)
    group.add_argument("--cuebridge-available-cue-limit", type=int, default=128)
    group.add_argument("--cuebridge-excerpt-chars", type=int, default=900)
    group.add_argument("--cuebridge-include-raw", action="store_true", default=True)
    group.add_argument("--max-questions-per-memory", type=int, default=3)
    group.add_argument("--cuebridge-weak-rank-threshold", type=int, default=20)
    group.add_argument("--cuebridge-accept-rank-threshold", type=int, default=20)
    group.add_argument("--cuebridge-min-rank-improvement", type=int, default=1)
    group.add_argument("--cuebridge-collateral-policy", choices=("off", "tier", "rank"), default="tier")
    group.add_argument(
        "--cuebridge-target-rank-threshold",
        type=int,
        default=10,
        help=(
            "In oracle compare modes, only raw benchmark questions ranked worse than this "
            "become gold-memory/question targets."
        ),
    )
    group.add_argument("--include-rank-6-20-fixes", action="store_true")
    group.add_argument("--score-with-artifacts", action="store_true")
    group.add_argument("--min-gap-confidence", type=float, default=0.60)
    group.add_argument("--min-alias-confidence", type=float, default=0.80)
    group.add_argument("--max-fanout", type=int, default=3)
    group.add_argument("--cuebridge-command-timeout-seconds", type=int, default=7200)


def cuebridge_compare_mode(args: argparse.Namespace) -> str:
    product = bool(getattr(args, "compare_cuebridge_product", False))
    oracle = bool(getattr(args, "compare_cuebridge_oracle", False))
    question_oracle = bool(getattr(args, "compare_cuebridge_question_oracle", False))
    legacy_oracle = bool(getattr(args, "compare_cuebridge", False))
    selected_count = sum(1 for enabled in (product, oracle, question_oracle, legacy_oracle) if enabled)
    if selected_count > 1:
        raise ValueError(
            "Choose only one CueBridge compare mode: --compare-cuebridge-product, "
            "--compare-cuebridge-oracle, or --compare-cuebridge-question-oracle."
        )
    if product:
        return "product"
    if question_oracle:
        return "question_oracle"
    if oracle or legacy_oracle:
        return "oracle"
    return "off"


def cuebridge_compare_enabled(args: argparse.Namespace) -> bool:
    return cuebridge_compare_mode(args) != "off"


def cuebridge_product_enabled(args: argparse.Namespace) -> bool:
    return cuebridge_compare_mode(args) == "product"


def cuebridge_oracle_enabled(args: argparse.Namespace) -> bool:
    return cuebridge_compare_mode(args) == "oracle"


def _resolve_provider(args: argparse.Namespace) -> str:
    provider = args.cuebridge_provider
    if provider == "auto":
        llama_cli_available = (
            "/" in args.llama_binary and Path(args.llama_binary).expanduser().exists()
        ) or shutil.which(args.llama_binary)
        provider = "llama-cpp" if llama_cli_available else "llama-server"
    return provider


def _append_provider_args(cmd: list[str], args: argparse.Namespace, provider: str) -> None:
    if provider == "llama-cpp":
        cmd.extend(["--llama-binary", args.llama_binary, "--llama-model", args.llama_model])
        if args.llama_display_prompt:
            cmd.append("--llama-display-prompt")
        for value in args.llama_extra_arg:
            cmd.append(f"--llama-extra-arg={value}")
    elif provider == "llama-server":
        cmd.extend(
            [
                "--llama-server-binary",
                args.llama_server_binary,
                "--llama-model",
                args.llama_model,
                "--llama-server-host",
                args.llama_server_host,
                "--llama-server-port",
                str(args.llama_server_port),
                "--llama-server-start-timeout-seconds",
                str(args.llama_server_start_timeout_seconds),
                "--openai-model",
                args.openai_model,
            ]
        )
        for value in args.llama_server_extra_arg:
            cmd.append(f"--llama-server-extra-arg={value}")
    elif provider == "openai-compatible":
        cmd.extend(
            [
                "--openai-base-url",
                args.openai_base_url,
                "--openai-model",
                args.openai_model,
                "--openai-api-key",
                args.openai_api_key,
            ]
        )
        for value in args.openai_extra_param:
            cmd.append(f"--openai-extra-param={value}")


def _proposal_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    metrics = payload.get("metrics") if isinstance(payload, dict) else None
    if isinstance(metrics, dict):
        return metrics
    if not isinstance(payload, dict):
        return {}
    return {
        "generated_questions": len(payload.get("generated_questions", []) or []),
        "gap_proposals": len(payload.get("gap_proposals", []) or []),
        "alias_proposals": len(payload.get("alias_proposals", []) or []),
        "rejected": len(payload.get("rejected", []) or []),
        "failures": len(payload.get("failures", []) or []),
    }


def _status_urls(url: str) -> list[str]:
    parsed = urllib.parse.urlparse(url)
    urls = [url.rstrip("/")]
    if parsed.hostname == "localhost":
        port = f":{parsed.port}" if parsed.port else ""
        replacement = urllib.parse.urlunparse(
            (parsed.scheme, f"127.0.0.1{port}", parsed.path.rstrip("/"), "", "", "")
        )
        urls.append(replacement.rstrip("/"))
    return list(dict.fromkeys(urls))


def _request_json(
    method: str,
    url: str,
    path: str,
    project_id: str,
    *,
    timeout: int = 120,
) -> dict[str, Any]:
    last_error = None
    body = None
    for base_url in _status_urls(url):
        endpoint = f"{base_url.rstrip('/')}{path}"
        request = urllib.request.Request(
            endpoint,
            data=body,
            method=method,
            headers={"Content-Type": "application/json", "X-Project-ID": project_id},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {endpoint} failed with HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            last_error = f"{method} {endpoint} failed: {exc}"
    raise RuntimeError(last_error or f"{method} {url}{path} failed")


def _export_project_memories(
    *,
    url: str,
    project_id: str,
    page_size: int,
    include_content: bool,
) -> list[dict[str, Any]]:
    memories = []
    cursor = None
    while True:
        query = {
            "limit": str(page_size),
            "include_content": "true" if include_content else "false",
            "include_cues": "true",
            "include_metadata": "true",
        }
        if cursor is not None:
            query["cursor"] = str(cursor)
        encoded = urllib.parse.urlencode(query)
        path = f"/projects/{urllib.parse.quote(project_id, safe='')}/export?{encoded}"
        page = _request_json("GET", url, path, project_id)
        batch = page.get("memories", [])
        if not isinstance(batch, list):
            raise RuntimeError("Project export response did not contain a memories array")
        memories.extend(batch)
        if not page.get("has_more"):
            return memories
        cursor = page.get("next_cursor")
        if cursor is None:
            return memories


def _normalize_cue(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _normalize_match_text(value: Any) -> str:
    text = str(value or "").replace("assistant: ", "").replace("user: ", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _text_matches_target(memory_content: str, target_text: str) -> bool:
    memory_norm = _normalize_match_text(memory_content)
    target_norm = _normalize_match_text(target_text)
    if not memory_norm or not target_norm:
        return False
    if len(target_norm) >= 24 and target_norm in memory_norm:
        return True
    if len(memory_norm) >= 24 and memory_norm in target_norm:
        return True
    memory_tokens = set(re.findall(r"[a-z0-9_]+", memory_norm))
    target_tokens = set(re.findall(r"[a-z0-9_]+", target_norm))
    if len(target_tokens) < 4:
        return False
    overlap = len(memory_tokens & target_tokens)
    return overlap >= max(4, int(len(target_tokens) * 0.75))


def _memory_excerpt(content: str, max_chars: int) -> str:
    content = re.sub(r"\s+", " ", content).strip()
    if len(content) <= max_chars:
        return content
    return content[: max(0, max_chars - 1)].rstrip() + "..."


def _memory_sample(memory: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    content = str(memory.get("content", ""))
    cues = [_normalize_cue(cue) for cue in memory.get("cues", []) if _normalize_cue(cue)]
    metadata = memory.get("metadata", {}) if isinstance(memory.get("metadata"), dict) else {}
    sample = {
        "memory_id": memory.get("id"),
        "source_key": memory.get("source_key"),
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "created_at": memory.get("created_at"),
        "metadata": metadata,
        "salient_cues": cues[: args.cuebridge_salient_cue_limit],
        "available_cues": cues[: args.cuebridge_available_cue_limit],
        "content_char_count": len(content),
    }
    if args.cuebridge_include_raw:
        sample["content"] = content
    else:
        sample["content_excerpt"] = _memory_excerpt(content, args.cuebridge_excerpt_chars)
    return sample


_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "between", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "how", "i", "in",
    "is", "it", "me", "my", "of", "on", "or", "our", "should", "the", "their",
    "them", "then", "there", "these", "they", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "with", "would", "you",
    "your",
}


def _query_signature_from_question(question: str, limit: int = 16) -> dict[str, list[str]]:
    cues: list[str] = []
    seen = set()
    for raw in re.findall(r"[A-Za-z0-9_][A-Za-z0-9_+.#/-]*", question.lower()):
        cue = raw.strip("-_/")
        if len(cue) < 3 or cue in _QUERY_STOPWORDS or cue in seen:
            continue
        seen.add(cue)
        cues.append(cue)
        if len(cues) >= limit:
            break
    return {"required_any": cues}


def _select_memories_for_target_texts(
    memories: list[dict[str, Any]],
    target_texts: list[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_ids = set()
    for memory in memories:
        content = str(memory.get("content", ""))
        if any(_text_matches_target(content, target) for target in target_texts):
            memory_id = str(memory.get("id"))
            if memory_id not in seen_ids:
                selected.append(_memory_sample(memory, args))
                seen_ids.add(memory_id)
    return selected


def _write_targeted_analysis(
    args: argparse.Namespace,
    project_id: str,
    analysis_path: Path,
    target_texts: list[str],
) -> dict[str, Any]:
    memories = _export_project_memories(
        url=args.url,
        project_id=project_id,
        page_size=args.cuebridge_page_size,
        include_content=True,
    )
    cue_freq = Counter()
    target_texts = [text for text in target_texts if str(text).strip()]
    for memory in memories:
        cues = [_normalize_cue(cue) for cue in memory.get("cues", []) if _normalize_cue(cue)]
        cue_freq.update(cues)
    selected = _select_memories_for_target_texts(memories, target_texts, args)

    hub_cues = [
        {"cue": cue, "count": count}
        for cue, count in cue_freq.most_common(100)
        if count > 1
    ]
    analysis = {
        "schema_version": 1,
        "artifact_type": "cuebridge_project_analysis",
        "project_id": project_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "compiler_version": "eval-targeted",
        "metrics": {
            "exported_memories": len(memories),
            "target_text_count": len(target_texts),
            "sample_count": len(selected),
            "unique_cues": len(cue_freq),
            "targeting": "benchmark_raw_rank_gt_threshold_gold_memory",
        },
        "hub_cues": hub_cues,
        "question_generation_jobs": [
            {
                "id": f"qgen_target_{idx + 1:06d}",
                "task": "generate recall questions and lexical-gap proposal candidates for this memory",
                "memory": sample,
                "instructions": {
                    "do_not_answer": True,
                    "generate_questions": True,
                    "focus": "minimal lexical overlap between question and memory while preserving answerability",
                    "proposal_goal": "surface only safe lexical-gap bridges for CueMap validation",
                },
            }
            for idx, sample in enumerate(selected)
        ],
    }
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True))
    print(
        f"Wrote targeted CueBridge analysis with {len(selected)} gold memories "
        f"from {len(target_texts)} target texts to {analysis_path}"
    )
    return {
        "target_text_count": len(target_texts),
        "target_memory_count": len(selected),
        "exported_memories": len(memories),
    }


def _write_question_oracle_analysis(
    args: argparse.Namespace,
    project_id: str,
    analysis_path: Path,
    questions_path: Path,
    target_questions: list[dict[str, Any]],
) -> dict[str, Any]:
    memories = _export_project_memories(
        url=args.url,
        project_id=project_id,
        page_size=args.cuebridge_page_size,
        include_content=True,
    )
    cue_freq = Counter()
    for memory in memories:
        cues = [_normalize_cue(cue) for cue in memory.get("cues", []) if _normalize_cue(cue)]
        cue_freq.update(cues)

    jobs: list[dict[str, Any]] = []
    generated_questions: list[dict[str, Any]] = []
    matched_target_count = 0
    matched_memory_ids = set()

    for case_idx, case in enumerate(target_questions, start=1):
        question = str(case.get("question", "")).strip()
        target_texts = [text for text in case.get("target_texts", []) if str(text).strip()]
        if not question or not target_texts:
            continue
        selected = _select_memories_for_target_texts(memories, target_texts, args)
        if not selected:
            continue
        matched_target_count += 1
        for mem_idx, sample in enumerate(selected, start=1):
            memory_id = sample.get("memory_id")
            matched_memory_ids.add(str(memory_id))
            job_id = f"qoracle_{case_idx:06d}_{mem_idx:03d}"
            question_id = f"{job_id}_question"
            jobs.append(
                {
                    "id": job_id,
                    "task": "generate targeted lexical-gap proposal candidates for this known weak question",
                    "memory": sample,
                    "instructions": {
                        "do_not_answer": True,
                        "generate_questions": False,
                        "focus": "bridge the actual question wording to this expected memory",
                        "proposal_goal": "surface only safe lexical-gap bridges for CueMap validation",
                    },
                    "eval_question": {
                        "id": case.get("id") or f"eval_question_{case_idx:06d}",
                        "question": question,
                        "category": case.get("category"),
                    },
                }
            )
            generated_questions.append(
                {
                    "id": question_id,
                    "job_id": job_id,
                    "question": question,
                    **({"query_time": case.get("query_time")} if case.get("query_time") else {}),
                    "expected_memory_id": memory_id,
                    "query_signature": _query_signature_from_question(question),
                    "expected_expansion_cues": sample.get("salient_cues", []),
                    "gap_pairs": [],
                    "provenance": {
                        "source": "benchmark_question_oracle",
                        "eval_question_id": case.get("id") or f"eval_question_{case_idx:06d}",
                        "category": case.get("category"),
                    },
                }
            )

    hub_cues = [
        {"cue": cue, "count": count}
        for cue, count in cue_freq.most_common(100)
        if count > 1
    ]
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    analysis = {
        "schema_version": 1,
        "artifact_type": "cuebridge_project_analysis",
        "project_id": project_id,
        "created_at": created_at,
        "compiler_version": "eval-question-oracle",
        "metrics": {
            "exported_memories": len(memories),
            "target_question_count": len(target_questions),
            "matched_target_question_count": matched_target_count,
            "target_memory_count": len(matched_memory_ids),
            "question_memory_target_count": len(generated_questions),
            "generated_question_count": len(generated_questions),
            "unique_cues": len(cue_freq),
            "targeting": "benchmark_raw_rank_gt_threshold_actual_question",
        },
        "hub_cues": hub_cues,
        "question_generation_jobs": jobs,
    }
    proposal = {
        "schema_version": 1,
        "artifact_type": "cuebridge_question_proposals",
        "project_id": project_id,
        "created_at": created_at,
        "compiler_version": "eval-question-oracle",
        "source_analysis": str(analysis_path),
        "generated_questions": generated_questions,
        "gap_proposals": [],
        "alias_proposals": [],
        "rejected": [],
        "metrics": {
            "generated_questions": len(generated_questions),
            "target_question_count": len(target_questions),
            "matched_target_question_count": matched_target_count,
            "target_memory_count": len(matched_memory_ids),
            "question_memory_target_count": len(generated_questions),
        },
    }
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True))
    questions_path.write_text(json.dumps(proposal, indent=2, sort_keys=True))
    print(
        f"Wrote question-oracle CueBridge analysis with {len(generated_questions)} "
        f"question-memory targets from {matched_target_count}/{len(target_questions)} eval questions to {analysis_path}"
    )
    return {
        "target_question_count": len(target_questions),
        "matched_target_question_count": matched_target_count,
        "target_memory_count": len(matched_memory_ids),
        "generated_question_count": len(generated_questions),
        "exported_memories": len(memories),
    }


def build_cuebridge_artifacts(
    args: argparse.Namespace,
    project_id: str,
    run_dir: Path,
    *,
    target_texts: list[str] | None = None,
    target_questions: list[dict[str, Any]] | None = None,
    start_step: str = "analyze-project",
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    analysis = run_dir / "cuebridge_analysis.json"
    questions = run_dir / "cuebridge_questions.json"
    scored = run_dir / "cuebridge_scored_questions.json"
    targeted = run_dir / "cuebridge_targeted_proposals.json"
    validated = run_dir / "cuebridge_validated_proposals.json"
    artifact_dir = run_dir / "artifact"
    validation = run_dir / "artifact_validation.json"
    raw_response_dir = run_dir / "raw_model_responses"
    raw_fix_response_dir = run_dir / "raw_targeted_responses"
    timeout = args.cuebridge_command_timeout_seconds
    start_step = start_step or "analyze-project"
    cuebridge_step_index(start_step)

    common_python = [args.cuebridge_python, str(CUEBRIDGE)]
    targeted_metrics = None
    if target_questions is not None:
        if cuebridge_should_run(start_step, "analyze-project"):
            targeted_metrics = _write_question_oracle_analysis(
                args,
                project_id,
                analysis,
                questions,
                target_questions,
            )
            if targeted_metrics["generated_question_count"] == 0:
                return {
                    "analysis": str(analysis),
                    "questions": str(questions),
                    "provider": None,
                    "skipped": "no_gold_question_memories_matched",
                    "targeting": targeted_metrics,
                    "artifact_files": [],
                    "start_step": start_step,
                }
        else:
            require_resume_file(analysis, step=start_step)
            require_resume_file(questions, step=start_step)
    elif target_texts is not None:
        if cuebridge_should_run(start_step, "analyze-project"):
            targeted_metrics = _write_targeted_analysis(args, project_id, analysis, target_texts)
            if targeted_metrics["target_memory_count"] == 0:
                return {
                    "analysis": str(analysis),
                    "provider": None,
                    "skipped": "no_gold_memories_matched",
                    "targeting": targeted_metrics,
                    "artifact_files": [],
                    "start_step": start_step,
                }
        else:
            require_resume_file(analysis, step=start_step)
            require_resume_file(questions, step=start_step)
    else:
        if cuebridge_should_run(start_step, "analyze-project"):
            run_pipeline_cmd(
                common_python
                + [
                    "analyze-project",
                    "--project",
                    project_id,
                    "--url",
                    args.url,
                    "--out",
                    str(analysis),
                    "--max-samples",
                    str(args.cuebridge_max_samples),
                    "--page-size",
                    str(args.cuebridge_page_size),
                    "--salient-cue-limit",
                    str(args.cuebridge_salient_cue_limit),
                    "--available-cue-limit",
                    str(args.cuebridge_available_cue_limit),
                    "--excerpt-chars",
                    str(args.cuebridge_excerpt_chars),
                ]
                + (["--include-raw"] if args.cuebridge_include_raw else []),
                timeout=timeout,
            )
        else:
            require_resume_file(analysis, step=start_step)

    provider = _resolve_provider(args)
    if target_questions is None and cuebridge_should_run(start_step, "question-generation"):
        propose_cmd = common_python + [
            "propose",
            "--analysis",
            str(analysis),
            "--provider",
            provider,
            "--questions-only",
            "--llama-n-predict",
            str(args.llama_n_predict),
            "--llama-temp",
            str(args.llama_temp),
            "--llama-top-p",
            str(args.llama_top_p),
            "--llama-seed",
            str(args.llama_seed),
            "--llama-ctx-size",
            str(args.llama_ctx_size),
            "--max-questions-per-memory",
            str(args.max_questions_per_memory),
            "--timeout-seconds",
            str(args.llama_timeout_seconds),
            "--raw-response-dir",
            str(raw_response_dir),
            "--progress-every",
            str(args.cuebridge_progress_every),
            "--concurrency",
            str(args.cuebridge_question_concurrency),
            "--batch-size",
            str(args.cuebridge_question_batch_size),
            "--question-underfill-retries",
            str(args.cuebridge_question_underfill_retries),
            "--out",
            str(questions),
        ]
        if args.cuebridge_max_jobs is not None:
            propose_cmd.extend(["--max-jobs", str(args.cuebridge_max_jobs)])
        if args.cuebridge_job_offset:
            propose_cmd.extend(["--job-offset", str(args.cuebridge_job_offset)])
        if args.continue_on_model_error:
            propose_cmd.append("--continue-on-error")
        _append_provider_args(propose_cmd, args, provider)
        run_pipeline_cmd(propose_cmd, timeout=timeout)
    else:
        require_resume_file(questions, step=start_step)

    if cuebridge_should_run(start_step, "score-questions"):
        score_cmd = common_python + [
            "score-questions",
            "--proposal",
            str(questions),
            "--project",
            project_id,
            "--url",
            args.url,
            "--limit",
            str(args.limit),
            "--weak-rank-threshold",
            str(args.cuebridge_weak_rank_threshold),
            "--out",
            str(scored),
            "--progress-every",
            str(args.cuebridge_progress_every),
        ]
        if args.score_with_artifacts:
            score_cmd.append("--with-artifacts")
        append_cuebridge_recall_option_args(score_cmd, args)
        run_pipeline_cmd(score_cmd, timeout=timeout)
    else:
        require_resume_file(scored, step=start_step)

    if cuebridge_should_run(start_step, "propose-fixes"):
        fix_cmd = common_python + [
            "propose-fixes",
            "--scored",
            str(scored),
            "--analysis",
            str(analysis),
            "--provider",
            provider,
            "--llama-n-predict",
            str(args.llama_n_predict),
            "--llama-temp",
            str(args.llama_temp),
            "--llama-top-p",
            str(args.llama_top_p),
            "--llama-seed",
            str(args.llama_seed),
            "--llama-ctx-size",
            str(args.llama_ctx_size),
            "--timeout-seconds",
            str(args.llama_timeout_seconds),
            "--raw-response-dir",
            str(raw_fix_response_dir),
            "--progress-every",
            str(args.cuebridge_progress_every),
            "--out",
            str(targeted),
            "--concurrency",
            str(args.cuebridge_fix_concurrency),
            "--batch-size",
            str(args.cuebridge_fix_batch_size),
        ]
        if args.cuebridge_max_fix_cases is not None:
            fix_cmd.extend(["--max-cases", str(args.cuebridge_max_fix_cases)])
        if args.cuebridge_case_offset:
            fix_cmd.extend(["--case-offset", str(args.cuebridge_case_offset)])
        if args.include_rank_6_20_fixes or target_questions is not None:
            fix_cmd.append("--include-rank-6-20")
        if args.continue_on_model_error:
            fix_cmd.append("--continue-on-error")
        _append_provider_args(fix_cmd, args, provider)
        run_pipeline_cmd(fix_cmd, timeout=timeout)
    else:
        require_resume_file(targeted, step=start_step)

    if cuebridge_should_run(start_step, "validate-proposals"):
        validate_cmd = common_python + [
            "validate-proposals",
            "--proposal",
            str(targeted),
            "--project",
            project_id,
            "--url",
            args.url,
            "--limit",
            str(args.limit),
            "--accept-rank-threshold",
            str(args.cuebridge_accept_rank_threshold),
            "--min-rank-improvement",
            str(args.cuebridge_min_rank_improvement),
            "--collateral-policy",
            args.cuebridge_collateral_policy,
            "--min-gap-confidence",
            str(args.min_gap_confidence),
            "--min-alias-confidence",
            str(args.min_alias_confidence),
            "--max-fanout",
            str(args.max_fanout),
            "--progress-every",
            str(args.cuebridge_progress_every),
            "--out",
            str(validated),
        ]
        append_cuebridge_recall_option_args(validate_cmd, args)
        run_pipeline_cmd(validate_cmd, timeout=timeout)
    else:
        require_resume_file(validated, step=start_step)

    if cuebridge_should_run(start_step, "compile"):
        run_pipeline_cmd(
            common_python
            + [
                "compile",
                "--project",
                project_id,
                "--analysis",
                str(analysis),
                "--proposal",
                str(validated),
                "--out",
                str(artifact_dir),
                "--min-gap-confidence",
                str(args.min_gap_confidence),
                "--min-alias-confidence",
                str(args.min_alias_confidence),
                "--max-fanout",
                str(args.max_fanout),
            ],
            timeout=timeout,
        )
    elif not artifact_dir.exists():
        raise FileNotFoundError(f"Cannot resume at {start_step}: artifact directory is missing: {artifact_dir}")
    if cuebridge_should_run(start_step, "validate-artifact"):
        run_pipeline_cmd(
            common_python + ["validate", "--artifact", str(artifact_dir), "--out", str(validation)],
            timeout=timeout,
        )
    else:
        require_resume_file(validation, step=start_step)
    if cuebridge_should_run(start_step, "install"):
        run_pipeline_cmd(
            common_python
            + [
                "install",
                "--project",
                project_id,
                "--artifact",
                str(artifact_dir),
                "--replace",
                "--url",
                args.url,
            ],
            timeout=timeout,
        )

    artifact_files = []
    for path in sorted(artifact_dir.glob("*.json")):
        artifact_files.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )

    return {
        "analysis": str(analysis),
        "questions": str(questions),
        "scored_questions": str(scored),
        "targeted_proposals": str(targeted),
        "validated_proposals": str(validated),
        "artifact_dir": str(artifact_dir),
        "validation": str(validation),
        "provider": provider,
        "start_step": start_step,
        "raw_response_dir": str(raw_response_dir),
        "raw_fix_response_dir": str(raw_fix_response_dir),
        "proposal_metrics": {
            "questions": _proposal_metrics(questions),
            "targeted": _proposal_metrics(targeted),
            "validated": _proposal_metrics(validated),
        },
        "artifact_files": artifact_files,
        "targeting": targeted_metrics,
    }


def _rank_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _rank_improved(raw_rank: int, enhanced_rank: int) -> bool:
    raw_effective = raw_rank if raw_rank > 0 else 10**9
    enhanced_effective = enhanced_rank if enhanced_rank > 0 else 10**9
    return enhanced_effective < raw_effective


def _rank_worsened(raw_rank: int, enhanced_rank: int) -> bool:
    raw_effective = raw_rank if raw_rank > 0 else 10**9
    enhanced_effective = enhanced_rank if enhanced_rank > 0 else 10**9
    return enhanced_effective > raw_effective


def _normalize_recall_text(text: str) -> str:
    text = text.replace("assistant: ", "").replace("user: ", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _expected_groups(record: dict[str, Any]) -> list[list[str]]:
    groups = record.get("expected_context_groups")
    if isinstance(groups, list):
        normalized_groups = []
        for group in groups:
            if isinstance(group, list):
                values = [str(item) for item in group if item]
            elif group:
                values = [str(group)]
            else:
                values = []
            if values:
                normalized_groups.append(values)
        if normalized_groups:
            return normalized_groups

    contexts = record.get("expected_contexts")
    if isinstance(contexts, list):
        return [[str(context)] for context in contexts if context]
    return []


def _matches_expected_group(content: str, group: list[str]) -> bool:
    content_norm = _normalize_recall_text(content)
    for expected in group:
        expected_norm = _normalize_recall_text(expected)
        if expected_norm and (expected_norm in content_norm or content_norm in expected_norm):
            return True
    return False


def _recall_metrics_from_contents(
    expected_groups: list[list[str]],
    recalled_contents: list[Any],
    *,
    ks: list[int],
) -> dict[str, float]:
    contents = [str(item) for item in recalled_contents if item is not None]
    expected_total = max(1, len(expected_groups))
    rel_array = []
    matched_by_rank: list[int | None] = []
    for content in contents:
        matched_idx = None
        for idx, group in enumerate(expected_groups):
            if _matches_expected_group(content, group):
                matched_idx = idx
                break
        rel_array.append(1 if matched_idx is not None else 0)
        matched_by_rank.append(matched_idx)

    metrics: dict[str, float] = {}
    for k in ks:
        found = {idx for idx in matched_by_rank[:k] if idx is not None}
        metrics[f"recall_frac_{k}"] = len(found) / expected_total
        metrics[f"recall_all_{k}"] = 1.0 if len(found) == len(expected_groups) and expected_groups else 0.0
        dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(rel_array[:k]))
        idcg = sum(1.0 / math.log2(idx + 2) for idx in range(min(k, len(expected_groups))))
        metrics[f"ndcg_{k}"] = dcg / idcg if idcg > 0 else 0.0
    return metrics


def _side_metric(
    item: dict[str, Any],
    *,
    side: str,
    metric: str,
    k: int,
    ks: list[int],
) -> float | None:
    key = f"{metric}_{k}"
    if side == "enhanced" and isinstance(item.get(key), (int, float)):
        return float(item[key])
    if side == "raw":
        raw_result = item.get("raw_result")
        if isinstance(raw_result, dict) and isinstance(raw_result.get(key), (int, float)):
            return float(raw_result[key])
        raw_metrics = item.get("raw_metrics")
        if isinstance(raw_metrics, dict) and isinstance(raw_metrics.get(key), (int, float)):
            return float(raw_metrics[key])
    else:
        enhanced_metrics = item.get("enhanced_metrics")
        if isinstance(enhanced_metrics, dict) and isinstance(enhanced_metrics.get(key), (int, float)):
            return float(enhanced_metrics[key])

    expected = _expected_groups(item)
    if not expected:
        return None
    if side == "raw":
        raw_result = item.get("raw_result")
        if isinstance(raw_result, dict):
            contents = raw_result.get("recalled_contents")
        else:
            contents = item.get("raw_recalled_contents")
    else:
        contents = item.get("recalled_contents")
    if not isinstance(contents, list):
        return None
    return _recall_metrics_from_contents(expected, contents, ks=ks).get(key)


def _cuebridge_items(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for record in results:
        if not isinstance(record, dict):
            continue
        probing = record.get("probing_results")
        if isinstance(probing, list):
            for item in probing:
                if isinstance(item, dict) and isinstance(item.get("raw_result"), dict):
                    items.append(item)
        elif "raw_hit_rank" in record:
            raw_result = {
                "hit_rank": record.get("raw_hit_rank"),
                "recalled_contents": record.get("raw_recalled_contents"),
            }
            item = {
                **record,
                "category": record.get("question_type") or record.get("category") or "unknown",
                "raw_result": raw_result,
            }
            items.append(item)
    return items


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def print_cuebridge_delta_summary(results: list[dict[str, Any]], *, limit: int = 20) -> None:
    items = _cuebridge_items(results)
    if not items:
        return

    pairs = [
        (_rank_value(item.get("raw_result", {}).get("hit_rank")), _rank_value(item.get("hit_rank")))
        for item in items
    ]
    ks = sorted({1, 5, 10, 20, limit})
    metric_ks = [k for k in (5, 10, 20) if k <= max(ks)]
    improved = sum(1 for raw, enh in pairs if _rank_improved(raw, enh))
    worsened = sum(1 for raw, enh in pairs if _rank_worsened(raw, enh))
    unchanged = len(pairs) - improved - worsened

    print("\n============== CUEBRIDGE DELTA ==============")
    print(f"Compared questions: {len(items)}")
    print("\n[ Hit@K ]")
    for k in ks:
        raw_hits = sum(1 for raw, _enh in pairs if 0 < raw <= k)
        enhanced_hits = sum(1 for _raw, enh in pairs if 0 < enh <= k)
        rescues = sum(1 for raw, enh in pairs if not (0 < raw <= k) and 0 < enh <= k)
        regressions = sum(1 for raw, enh in pairs if 0 < raw <= k and not (0 < enh <= k))
        delta = enhanced_hits - raw_hits
        print(
            f"Hit@{k}: raw={raw_hits}/{len(items)} ({raw_hits / len(items) * 100:.1f}%) "
            f"enhanced={enhanced_hits}/{len(items)} ({enhanced_hits / len(items) * 100:.1f}%) "
            f"delta={delta:+d} rescues={rescues} regressions={regressions}"
        )

    print("\n[ Rank Movement ]")
    print(f"Improved rank: {improved}")
    print(f"Worsened rank: {worsened}")
    print(f"Unchanged rank: {unchanged}")

    print("\n[ Quality Metrics ]")
    for metric, label in (
        ("recall_all", "Recall_All"),
        ("recall_frac", "Recall_Frac"),
        ("ndcg", "NDCG"),
    ):
        for k in metric_ks:
            raw_values = [
                value
                for item in items
                if (value := _side_metric(item, side="raw", metric=metric, k=k, ks=metric_ks)) is not None
            ]
            enhanced_values = [
                value
                for item in items
                if (value := _side_metric(item, side="enhanced", metric=metric, k=k, ks=metric_ks)) is not None
            ]
            if not raw_values or not enhanced_values:
                continue
            raw_avg = _avg(raw_values)
            enhanced_avg = _avg(enhanced_values)
            print(f"{label}@{k}: raw={_fmt_pct(raw_avg)} enhanced={_fmt_pct(enhanced_avg)} delta={(enhanced_avg - raw_avg) * 100:+.1f}pp")

    by_type: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        q_type = str(item.get("category") or item.get("question_type") or "unknown")
        by_type.setdefault(q_type, []).append(item)
    if len(by_type) > 1:
        print("\n[ By Question Type ]")
        for q_type, typed_items in sorted(by_type.items()):
            typed_pairs = [
                (_rank_value(item.get("raw_result", {}).get("hit_rank")), _rank_value(item.get("hit_rank")))
                for item in typed_items
            ]
            raw_hit5 = sum(1 for raw, _enh in typed_pairs if 0 < raw <= 5)
            enhanced_hit5 = sum(1 for _raw, enh in typed_pairs if 0 < enh <= 5)
            raw_hit20 = sum(1 for raw, _enh in typed_pairs if 0 < raw <= 20)
            enhanced_hit20 = sum(1 for _raw, enh in typed_pairs if 0 < enh <= 20)
            raw_ndcg10 = [
                value
                for item in typed_items
                if (value := _side_metric(item, side="raw", metric="ndcg", k=10, ks=metric_ks)) is not None
            ]
            enhanced_ndcg10 = [
                value
                for item in typed_items
                if (value := _side_metric(item, side="enhanced", metric="ndcg", k=10, ks=metric_ks)) is not None
            ]
            raw_frac10 = [
                value
                for item in typed_items
                if (value := _side_metric(item, side="raw", metric="recall_frac", k=10, ks=metric_ks)) is not None
            ]
            enhanced_frac10 = [
                value
                for item in typed_items
                if (value := _side_metric(item, side="enhanced", metric="recall_frac", k=10, ks=metric_ks)) is not None
            ]
            print(f"{q_type} (n={len(typed_items)}):")
            print(
                f"  Hit@5 raw={raw_hit5}/{len(typed_items)} enhanced={enhanced_hit5}/{len(typed_items)} "
                f"delta={enhanced_hit5 - raw_hit5:+d}"
            )
            print(
                f"  Hit@20 raw={raw_hit20}/{len(typed_items)} enhanced={enhanced_hit20}/{len(typed_items)} "
                f"delta={enhanced_hit20 - raw_hit20:+d}"
            )
            if raw_ndcg10 and enhanced_ndcg10:
                raw_avg = _avg(raw_ndcg10)
                enhanced_avg = _avg(enhanced_ndcg10)
                print(f"  NDCG@10 raw={_fmt_pct(raw_avg)} enhanced={_fmt_pct(enhanced_avg)} delta={(enhanced_avg - raw_avg) * 100:+.1f}pp")
            if raw_frac10 and enhanced_frac10:
                raw_avg = _avg(raw_frac10)
                enhanced_avg = _avg(enhanced_frac10)
                print(f"  Recall_Frac@10 raw={_fmt_pct(raw_avg)} enhanced={_fmt_pct(enhanced_avg)} delta={(enhanced_avg - raw_avg) * 100:+.1f}pp")
