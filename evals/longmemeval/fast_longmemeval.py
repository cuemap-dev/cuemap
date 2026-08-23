#!/usr/bin/env python3
"""Run the canonical LongMemEval harness with BEAM-style ingestion.

The canonical harness uses ``cuemap add`` once per message. This adapter keeps
its recall and scoring logic intact while replacing only those client calls
with direct HTTP POSTs to ``/ingest/content``, matching the BEAM harness. The
server queues the write and the canonical harness still waits for the project
to settle before recall.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


_HARNESS_ENV = "CUEMAP_LONGMEMEVAL_HARNESS"
_TIMING_MARKER = "--- TIMING ---"


def _load_harness():
    harness_path = os.environ.get(_HARNESS_ENV)
    if not harness_path:
        raise RuntimeError(f"{_HARNESS_ENV} must point to test_longmemeval_settled.py")

    path = Path(harness_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LongMemEval harness not found: {path}")

    # The canonical harness imports sibling evaluation utilities. When it is
    # executed directly, Python adds its parent directory automatically; an
    # adapter loaded from rust_engine needs to reproduce that behavior.
    harness_parent = str(path.parent)
    if harness_parent not in sys.path:
        sys.path.insert(0, harness_parent)

    spec = importlib.util.spec_from_file_location("cuemap_longmemeval_harness", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load LongMemEval harness: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_add_command(cmd: list[str]) -> tuple[str, str, dict[str, Any], str, bool] | None:
    if len(cmd) < 3 or Path(cmd[0]).name != "cuemap" or cmd[1] != "add":
        return None

    try:
        project_idx = next(i for i, value in enumerate(cmd) if value in {"-p", "--project"})
        url_idx = cmd.index("--url")
        metadata_idx = cmd.index("--metadata")
        project_id = cmd[project_idx + 1]
        url = cmd[url_idx + 1]
        metadata = json.loads(cmd[metadata_idx + 1])
        content = cmd[-1]
    except (StopIteration, IndexError, ValueError, json.JSONDecodeError):
        return None

    if not isinstance(metadata, dict):
        metadata = {}

    return (
        project_id,
        url,
        metadata,
        content,
        "--disable-temporal-chunking" in cmd,
    )


def _cue_value(value: Any) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower()).strip("_")
    return normalized or "unknown"


def _post_ingest(
    *,
    url: str,
    project_id: str,
    message_index: int,
    metadata: dict[str, Any],
    content: str,
    disable_temporal_chunking: bool,
    timeout: int = 60,
) -> dict[str, Any]:
    source_key = f"longmemeval:{project_id}:message:{message_index}"
    structural_cues = ["source_type:chat_message"]
    if metadata.get("source_role"):
        structural_cues.append(f"source_role:{_cue_value(metadata['source_role'])}")
    if metadata.get("source_date"):
        structural_cues.append(f"source_date:{_cue_value(metadata['source_date'])}")

    # Use a window larger than the message's sentence count and a max size
    # larger than its content so each LongMemEval message remains one memory,
    # while still taking the same /ingest/content route as BEAM.
    sentence_window = max(1, len(content) + 1)
    max_chunk_chars = max(2000, len(content) + 1)
    payload = {
        "content": content,
        "filename": f"longmemeval_{project_id}_{message_index}.txt",
        "source_key": source_key,
        "metadata": metadata,
        "structural_cues": structural_cues,
        "segmenter": "sentence_window",
        "segment_window_size": sentence_window,
        "segment_overlap": 0,
        "segment_min_chunk_chars": 1,
        "segment_max_chunk_chars": max_chunk_chars,
    }
    # The ingest/content endpoint does not currently expose
    # disable_temporal_chunking. The canonical LongMemEval harness does not
    # set the flag in normal runs; keep parsing it only for command shape
    # compatibility without changing user metadata.
    _ = disable_temporal_chunking

    endpoint = f"{url.rstrip('/')}/ingest/content"
    request = Request(
        endpoint,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Project-ID": project_id,
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(
                    f"POST {endpoint} failed with HTTP {response.status}: {body}"
                )
            return json.loads(body) if body else {}
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {endpoint} failed with HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"POST {endpoint} failed: {exc}") from exc


def _timing_enabled() -> bool:
    return os.environ.get("CUEMAP_TRACE_TIMING") == "1" and bool(
        os.environ.get("CUEMAP_TIMING_FILE")
    )


def _append_timing_record(stdout: str) -> None:
    marker_index = stdout.find(_TIMING_MARKER)
    if marker_index < 0:
        return

    timing_text = stdout[marker_index + len(_TIMING_MARKER) :].strip()
    try:
        timing = json.loads(timing_text)
    except json.JSONDecodeError:
        return
    if not isinstance(timing, dict):
        return

    timing_path = Path(os.environ["CUEMAP_TIMING_FILE"]).expanduser()
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    with timing_path.open("a", encoding="utf-8") as handle:
        json.dump(timing, handle, separators=(",", ":"))
        handle.write("\n")


def main() -> None:
    harness = _load_harness()
    original_run_cmd = harness.run_cmd
    message_indices: dict[tuple[str, str], int] = {}

    def fast_run_cmd(cmd: list[str], *, check: bool = False) -> subprocess.CompletedProcess:
        parsed = _parse_add_command(cmd)
        is_recall = len(cmd) >= 2 and Path(cmd[0]).name == "cuemap" and cmd[1] == "recall"
        if parsed is None and not is_recall:
            return original_run_cmd(cmd, check=check)

        if parsed is None:
            recall_cmd = list(cmd)
            if "--semantic-mode" not in recall_cmd:
                recall_cmd.extend([
                    "--semantic-mode",
                    os.environ.get("CUEMAP_SEMANTIC_MODE", "hybrid"),
                ])
            if _timing_enabled() and "--trace-timing" not in recall_cmd:
                recall_cmd.append("--trace-timing")
            result = original_run_cmd(recall_cmd, check=check)
            if _timing_enabled() and result.returncode == 0:
                _append_timing_record(result.stdout)
            return result

        project_id, url, metadata, content, disable_temporal_chunking = parsed
        key = (url, project_id)
        message_index = message_indices.get(key, 0)
        message_indices[key] = message_index + 1
        _post_ingest(
            url=url,
            project_id=project_id,
            message_index=message_index,
            metadata=metadata,
            content=content,
            disable_temporal_chunking=disable_temporal_chunking,
        )
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="✓ Memory queued\n",
            stderr="",
        )

    harness.run_cmd = fast_run_cmd
    harness.evaluate()


if __name__ == "__main__":
    main()
