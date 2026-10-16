#!/usr/bin/env bash
# v0.7.5 release evals: the numbers for the website and README, in three recall settings.
#
#   lean      expansion_depth 1: matched memories only (no conversation rounds)
#   default   the engine defaults: conversation rounds plus the reply to the top result
#   context5  conversation rounds plus replies for the first 5 results (more context)
#   window10  LoCoMo only: expansion_depth 10, the setting behind earlier LoCoMo numbers
#
# Every suite ingests once with conversation metadata (session, turn and role), keeps its
# projects, then runs the three settings as recall-only arms on the same projects, so the
# settings differ only in recall. Engine settings are the shipped defaults: no CUEMAP_*
# overrides except CUEMAP_ROUND_REPLY_TOP=5 for the context5 arm.
# Resumable: finished outputs are skipped. A summary and comparisons are written to
# evals/release_0.7.5_summary.md at the end.
#
# Suites: locomo, lme, beam128k, beam1m, beam10m. The default runs all but beam10m, which
# takes many hours and runs one record at a time to fit in memory (see run_beam10m); add it
# explicitly. Example:
#   bash rust_engine/evals/run_release_evals.sh                  # locomo lme beam128k beam1m
#   bash rust_engine/evals/run_release_evals.sh beam10m          # just BEAM 10M
set -euo pipefail

ENGINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ENGINE_ROOT"
TAG=release_0.7.5
PORT="${CUEMAP_PORT:-8735}"; BIN="$ENGINE_ROOT/target/release/cuemap"; URL="http://127.0.0.1:$PORT"
LOG="evals/${TAG}.log"
exec > >(tee -a "$LOG") 2>&1
# Shipped defaults only.
unset CUEMAP_ROUND_REPLY_TOP CUEMAP_SOURCE_ROLE_PENALTY CUEMAP_MEMORY_TYPE_HEAD CUEMAP_MEMORY_TYPE_HEAD_PATH \
  CUEMAP_MEMORY_TYPE_MODEL_PATH CUEMAP_MEMORY_TYPE_TOKENIZER_PATH CUEMAP_SEMANTIC_INTENT_NO_RECALL_PENALTY \
  CUEMAP_SEMANTIC_FUSION CUEMAP_SEMANTIC_COREML_ENABLED EXPANSION_DEPTH

ENGINE_PID=""
start_engine() {  # $@ = extra start flags; CUEMAP_ROUND_REPLY_TOP may be set by the caller
  if curl -fsS "$URL/healthz" >/dev/null 2>&1; then
    echo "Port $PORT already has an engine. Stop it first." >&2; exit 1
  fi
  echo "[$(date +%T)] start engine $* (CUEMAP_ROUND_REPLY_TOP=${CUEMAP_ROUND_REPLY_TOP:-default})"
  "$BIN" start --port "$PORT" "$@" >>"evals/${TAG}_engine.log" 2>&1 &
  ENGINE_PID=$!
  # Startup loads every kept project; with many projects this takes a while.
  for _ in $(seq 1 3600); do
    curl -fsS "$URL/healthz" >/dev/null 2>&1 && return 0
    kill -0 "$ENGINE_PID" 2>/dev/null || { echo "engine exited; see evals/${TAG}_engine.log" >&2; exit 1; }
    sleep 1
  done
  echo "engine did not become healthy" >&2; exit 1
}
stop_engine() {
  [[ -n "$ENGINE_PID" ]] || return 0
  echo "[$(date +%T)] stop engine (pid $ENGINE_PID)"
  kill -TERM "$ENGINE_PID" 2>/dev/null || true
  wait "$ENGINE_PID" 2>/dev/null || true
  ENGINE_PID=""
}
trap stop_engine EXIT

# LongMemEval and LoCoMo recall-only runs read project ids from OUTPUT and overwrite it, so
# each arm starts from a copy of the ingest output and is renamed when complete.
recall_arm() {  # $1 suite script, $2 ingest output, $3 arm output, rest = env assignments
  local script="$1" ingest="$2" out="$3"; shift 3
  if [[ -s "$out" ]]; then echo "skip $out: exists"; return 0; fi
  cp "$ingest" "$out.partial"
  echo "[$(date +%T)] recall arm $out ($*)"
  env CUEMAP_URL="$URL" RECALL_ONLY=1 OUTPUT="$out.partial" "$@" bash "$script"
  mv "$out.partial" "$out"
}

# Runs the three settings for one suite with its arm function ($1), called as
# "<arm function> <setting name> [env assignments...]".
run_settings() {
  local arm_fn="$1"
  start_engine --disable-bg-jobs
  "$arm_fn" lean EXPANSION_DEPTH=1
  "$arm_fn" default
  # LoCoMo only: the +/-9-message window that earlier website numbers used (about 10k tokens).
  if [[ "$arm_fn" == locomo_arm ]]; then "$arm_fn" window10 EXPANSION_DEPTH=10; fi
  stop_engine
  export CUEMAP_ROUND_REPLY_TOP=5
  start_engine --disable-bg-jobs
  "$arm_fn" context5 EXPANSION_DEPTH=conversation_round
  stop_engine
  unset CUEMAP_ROUND_REPLY_TOP
}

LOCOMO_OUT="evals/locomo/results/$TAG"
locomo_arm() { local name="$1"; shift
  recall_arm ./evals/locomo/run_locomo.sh "$LOCOMO_OUT/locomo_ingest.json" "$LOCOMO_OUT/locomo_$name.json" "$@"; }
run_locomo() {
  mkdir -p "$LOCOMO_OUT"
  local ingest="$LOCOMO_OUT/locomo_ingest.json"
  if [[ ! -s "$ingest" ]]; then
    start_engine
    env CUEMAP_URL="$URL" EXPANSION_DEPTH=1 DELETE_PROJECTS=0 OUTPUT="$ingest.partial" bash ./evals/locomo/run_locomo.sh
    mv "$ingest.partial" "$ingest"
    stop_engine
  fi
  run_settings locomo_arm
}

LME_OUT="evals/longmemeval/results/$TAG"
# TRACE_TIMING stays off: with it on, the harness appends timing JSON to the last result.
lme_arm() { local name="$1"; shift
  recall_arm ./evals/longmemeval/run_longmemeval.sh "$LME_OUT/longmemeval_ingest.json" "$LME_OUT/longmemeval_$name.json" TRACE_TIMING=0 "$@"; }
run_lme() {
  mkdir -p "$LME_OUT"
  local ingest="$LME_OUT/longmemeval_ingest.json"
  if [[ ! -s "$ingest" ]]; then
    start_engine
    env CUEMAP_URL="$URL" TRACE_TIMING=0 SOURCE_ORDER_METADATA=1 EXPANSION_DEPTH=1 DELETE_PROJECTS=0 \
      OUTPUT="$ingest.partial" bash ./evals/longmemeval/run_longmemeval.sh
    mv "$ingest.partial" "$ingest"
    stop_engine
  fi
  run_settings lme_arm
}

BEAM_OUT="evals/beam/results/$TAG"
BEAM_CONTEXT=""
beam_arm() { local name="$1"; shift
  local result="$BEAM_OUT/beam_${BEAM_CONTEXT}_$name.json"
  if [[ -s "$result" ]]; then echo "skip $result: exists"; return 0; fi
  echo "[$(date +%T)] recall arm $result ($*)"
  env CUEMAP_URL="$URL" CONTEXT="$BEAM_CONTEXT" SOURCE_ORDER_METADATA=1 TRACE_TIMING=1 RECALL_ONLY=1 \
    REUSE_PROJECTS_FROM="$BEAM_OUT/beam_${BEAM_CONTEXT}_ingest.json" "$@" \
    OUTPUT="$result.partial" TIMING_FILE="$BEAM_OUT/beam_${BEAM_CONTEXT}_${name}_timing.jsonl" bash ./evals/beam/run_beam.sh
  mv "$result.partial" "$result"
}
run_beam() {  # $1 = context: 128k, 1m or 10m
  BEAM_CONTEXT="$1"; mkdir -p "$BEAM_OUT"
  local ingest="$BEAM_OUT/beam_${BEAM_CONTEXT}_ingest.json"
  if [[ ! -s "$ingest" ]]; then
    # Ingestion needs background jobs, so this engine runs without --disable-bg-jobs.
    start_engine
    env CUEMAP_URL="$URL" CONTEXT="$BEAM_CONTEXT" SOURCE_ORDER_METADATA=1 TRACE_TIMING=1 EXPANSION_DEPTH=1 \
      OUTPUT="$ingest.partial" TIMING_FILE="$BEAM_OUT/beam_${BEAM_CONTEXT}_ingest_timing.jsonl" bash ./evals/beam/run_beam.sh
    mv "$ingest.partial" "$ingest"
    stop_engine
  fi
  run_settings beam_arm
}

# BEAM 10M projects are too large to keep all ten in memory, so this suite runs one record
# at a time: ingest it, run the three settings, then move that project's snapshot out of the
# engine's snapshot folder (to $PARK_DIR; nothing is deleted) before the next record. The
# per-record outputs are merged into beam_10m_<setting>.json at the end. Before starting, the
# other suites' eval projects (eval_small_*, eval_locomo_*, eval_beam_128k_*, eval_beam_1m_*)
# and any 10M project not used by an unfinished record are parked the same way, so engine
# startups only load what this record needs.
SNAPSHOTS="${CUEMAP_SNAPSHOTS_DIR:-$HOME/.cuemap/data/snapshots}"
CONTENTS="${CUEMAP_CONTENTS_DIR:-$HOME/.cuemap/data/contents}"
PARK_DIR="${EVAL_PARK_DIR:-$HOME/.cuemap/data/eval_parked}"
park_project() {  # $1 = project id
  local pid="$1" f
  mkdir -p "$PARK_DIR"
  for f in "$SNAPSHOTS/$pid.bin" "$SNAPSHOTS/${pid}_aliases.bin" "$SNAPSHOTS/${pid}_lexicon.bin" "$CONTENTS/$pid"; do
    if [[ -e "$f" ]]; then mv -n "$f" "$PARK_DIR/"; fi
  done
  echo "[$(date +%T)] parked $pid in $PARK_DIR"
}
beam10m_arm() {  # $1 record, $2 setting name, rest = env assignments
  local i="$1" name="$2"; shift 2
  local dir="$BEAM_OUT/10m_records"
  local result="$dir/beam_10m_r${i}_$name.json"
  if [[ -s "$result" ]]; then echo "skip $result: exists"; return 0; fi
  echo "[$(date +%T)] recall arm $result ($*)"
  env CUEMAP_URL="$URL" CONTEXT=10m START_INDEX="$i" MAX_RECORDS=1 SOURCE_ORDER_METADATA=1 TRACE_TIMING=1 RECALL_ONLY=1 \
    REUSE_PROJECTS_FROM="$dir/beam_10m_r${i}_ingest.json" "$@" \
    OUTPUT="$result.partial" TIMING_FILE="$dir/beam_10m_r${i}_${name}_timing.jsonl" bash ./evals/beam/run_beam.sh
  mv "$result.partial" "$result"
}
run_beam10m() {
  BEAM_CONTEXT=10m
  local dir="$BEAM_OUT/10m_records" records="${BEAM10M_RECORDS:-10}" i pid
  mkdir -p "$dir"
  # Records finished by an earlier all-at-once run keep their projects: split them out.
  if [[ -s "$BEAM_OUT/beam_10m_ingest.json.partial" ]]; then
    python3 - "$BEAM_OUT/beam_10m_ingest.json.partial" "$dir" <<'PY'
import json, os, sys
partial, out_dir = sys.argv[1], sys.argv[2]
for row in json.load(open(partial)):
    path = os.path.join(out_dir, f"beam_10m_r{row['record_idx']}_ingest.json")
    if not os.path.exists(path):
        json.dump([row], open(path, "w"))
        print("reusing ingested 10M record", row["record_idx"], row["project_id"])
PY
    mv "$BEAM_OUT/beam_10m_ingest.json.partial" "$BEAM_OUT/beam_10m_ingest.json.partial.superseded"
  fi
  # Park every eval project this suite does not need right now.
  local keep
  keep="$(python3 - "$dir" "$records" <<'PY'
import json, os, sys
out_dir, records = sys.argv[1], int(sys.argv[2])
for i in range(records):
    path = os.path.join(out_dir, f"beam_10m_r{i}_ingest.json")
    if os.path.exists(path) and not os.path.exists(os.path.join(out_dir, f"r{i}.done")):
        print(json.load(open(path))[0]["project_id"])
PY
)"
  for f in "$SNAPSHOTS"/eval_*.bin; do
    [[ -e "$f" ]] || continue
    case "$f" in *_aliases.bin|*_lexicon.bin) continue ;; esac
    pid="$(basename "$f" .bin)"
    if ! grep -qx "$pid" <<<"$keep"; then park_project "$pid"; fi
  done
  for ((i = 0; i < records; i++)); do
    if [[ -e "$dir/r${i}.done" ]]; then echo "skip 10M record $i: done"; continue; fi
    local ingest="$dir/beam_10m_r${i}_ingest.json"
    if [[ ! -s "$ingest" ]]; then
      start_engine
      env CUEMAP_URL="$URL" CONTEXT=10m START_INDEX="$i" MAX_RECORDS=1 SOURCE_ORDER_METADATA=1 TRACE_TIMING=1 \
        EXPANSION_DEPTH=1 OUTPUT="$ingest.partial" TIMING_FILE="$dir/beam_10m_r${i}_ingest_timing.jsonl" \
        bash ./evals/beam/run_beam.sh
      mv "$ingest.partial" "$ingest"
      stop_engine
    fi
    start_engine --disable-bg-jobs
    beam10m_arm "$i" lean EXPANSION_DEPTH=1
    beam10m_arm "$i" default
    stop_engine
    export CUEMAP_ROUND_REPLY_TOP=5
    start_engine --disable-bg-jobs
    beam10m_arm "$i" context5 EXPANSION_DEPTH=conversation_round
    stop_engine
    unset CUEMAP_ROUND_REPLY_TOP
    pid="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))[0]["project_id"])' "$ingest")"
    park_project "$pid"
    touch "$dir/r${i}.done"
  done
  local arm
  for arm in lean default context5; do
    python3 - "$BEAM_OUT/beam_10m_$arm.json" "$dir" "$arm" "$records" <<'PY'
import json, os, sys
out, out_dir, arm, records = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
rows = []
for i in range(records):
    path = os.path.join(out_dir, f"beam_10m_r{i}_{arm}.json")
    if os.path.exists(path):
        rows += json.load(open(path))
json.dump(rows, open(out, "w"))
print(f"{out}: {len(rows)} records")
PY
    cat "$dir"/beam_10m_r*_"${arm}"_timing.jsonl > "$BEAM_OUT/beam_10m_${arm}_timing.jsonl" 2>/dev/null || true
  done
}

summary() {
  local md="evals/${TAG}_summary.md"
  echo "[$(date +%T)] summary -> $md"
  {
    echo "# CueMap v0.7.5 release evals ($(date +%F))"
    echo
    echo "Settings: lean = expansion_depth 1; default = engine defaults (conversation rounds + reply to the top result); context5 = conversation rounds + replies for the first 5 results; window10 (LoCoMo only) = expansion_depth 10, the setting behind earlier LoCoMo website numbers. LoCoMo runs keep the harness's ordered_reconstruction=auto."
    for pair in "locomo:$LOCOMO_OUT/locomo" "lme:$LME_OUT/longmemeval" \
                "beam128k:$BEAM_OUT/beam_128k" "beam1m:$BEAM_OUT/beam_1m" "beam10m:$BEAM_OUT/beam_10m"; do
      local name="${pair%%:*}" prefix="${pair#*:}"
      [[ -s "${prefix}_default.json" ]] || continue
      echo; echo "## $name"; echo
      local files=() arm
      for arm in lean default context5 window10; do [[ -s "${prefix}_$arm.json" ]] && files+=("${prefix}_$arm.json"); done
      python3 evals/harnesses/scorecard.py summary "${files[@]}" || true
      for cmp in "lean default" "default context5"; do
        set -- $cmp
        [[ -s "${prefix}_$1.json" && -s "${prefix}_$2.json" ]] || continue
        local timing=()
        [[ -s "${prefix}_$1_timing.jsonl" && -s "${prefix}_$2_timing.jsonl" ]] && \
          timing=(--baseline-timing "${prefix}_$1_timing.jsonl" --candidate-timing "${prefix}_$2_timing.jsonl")
        echo; python3 evals/harnesses/scorecard.py compare "${prefix}_$1.json" "${prefix}_$2.json" \
          --budgets 2000,4000,8000 ${timing[@]+"${timing[@]}"} || true
      done
    done
  } > "$md" 2>&1
}

echo "[$(date +%T)] cargo build --release"
cargo build --release
"$BIN" --version || true

SUITES=("$@"); [[ ${#SUITES[@]} -gt 0 ]] || SUITES=(locomo lme beam128k beam1m)
for suite in "${SUITES[@]}"; do
  case "$suite" in
    locomo) run_locomo ;;
    lme) run_lme ;;
    beam128k) run_beam 128k ;;
    beam1m) run_beam 1m ;;
    beam10m) run_beam10m ;;
    *) echo "unknown suite $suite (locomo, lme, beam128k, beam1m, beam10m)" >&2; exit 1 ;;
  esac
done
summary
echo "[$(date +%T)] done"
