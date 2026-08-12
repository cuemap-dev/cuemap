#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUEMAP_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CUEMAP_EVALS_DIR="${CUEMAP_EVALS_DIR:-$CUEMAP_ROOT/evals}"
HARNESS="$CUEMAP_EVALS_DIR/test_beam_settled.py"
export PATH="$SCRIPT_DIR/bin:$PATH"

if [[ ! -f "$HARNESS" ]]; then
  echo "Missing BEAM harness: $HARNESS" >&2
  echo "Set CUEMAP_EVALS_DIR to the directory containing test_beam_settled.py." >&2
  exit 1
fi

CONTEXT="${CONTEXT:-128k}"
CUEMAP_URL="${CUEMAP_URL:-http://127.0.0.1:8080}"
LIMIT="${LIMIT:-100}"
MODE="${MODE:-raw}"
SEMANTIC_MODE="${SEMANTIC_MODE:-hybrid}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/results}"
OUTPUT="${OUTPUT:-$OUT_DIR/beam_${CONTEXT}_${MODE}.json}"
TRACE_TIMING="${TRACE_TIMING:-0}"
TIMING_FILE="${TIMING_FILE:-$OUT_DIR/beam_${CONTEXT}_${MODE}_${SEMANTIC_MODE}_timing_$(date +%Y%m%d_%H%M%S).jsonl}"

case "$SEMANTIC_MODE" in
  lexical|semantic|hybrid)
    export CUEMAP_SEMANTIC_MODE="$SEMANTIC_MODE"
    ;;
  *)
    echo "Unknown SEMANTIC_MODE=$SEMANTIC_MODE. Use lexical, semantic, or hybrid." >&2
    exit 1
    ;;
esac

mkdir -p "$OUT_DIR"

if [[ "$TRACE_TIMING" == "1" ]]; then
  : >"$TIMING_FILE"
  export CUEMAP_TRACE_TIMING=1
  export CUEMAP_TIMING_FILE="$TIMING_FILE"
fi

args=(
  python "$HARNESS"
  --context "$CONTEXT"
  --url "$CUEMAP_URL"
  --limit "$LIMIT"
  --ingest-long-form
  --ordered-reconstruction "${ORDERED_RECONSTRUCTION:-auto}"
  --evidence-coverage "${EVIDENCE_COVERAGE:-off}"
  --no-auto-reinforce
  --output "$OUTPUT"
)

if [[ "${DELETE_PROJECTS:-1}" == "1" ]]; then
  args+=(--delete-project-after-record)
fi

if [[ -n "${START_INDEX:-}" ]]; then
  args+=(--start-index "$START_INDEX")
fi

if [[ -n "${MAX_RECORDS:-}" ]]; then
  args+=(--max-records "$MAX_RECORDS")
fi

case "$MODE" in
  raw)
    ;;
  product-cuebridge)
    args+=(
      --compare-cuebridge-product
      --cuebridge-provider openai-compatible
      --openai-base-url "${CUEBRIDGE_BASE_URL:-http://127.0.0.1:1234/v1}"
      --openai-model "${CUEBRIDGE_MODEL:-qwen3-4b-cuebridge}"
      --openai-api-key "${CUEBRIDGE_API_KEY:-local}"
      --llama-n-predict "${CUEBRIDGE_N_PREDICT:-1024}"
      --cuebridge-target-rank-threshold "${CUEBRIDGE_TARGET_RANK_THRESHOLD:-10}"
      --cuebridge-accept-rank-threshold "${CUEBRIDGE_ACCEPT_RANK_THRESHOLD:-20}"
      --cuebridge-min-rank-improvement "${CUEBRIDGE_MIN_RANK_IMPROVEMENT:-1}"
      --cuebridge-collateral-policy "${CUEBRIDGE_COLLATERAL_POLICY:-tier}"
      --cuebridge-max-samples "${CUEBRIDGE_MAX_SAMPLES:-500}"
      --cuebridge-max-jobs "${CUEBRIDGE_MAX_JOBS:-500}"
      --max-questions-per-memory "${MAX_QUESTIONS_PER_MEMORY:-8}"
      --cuebridge-question-concurrency "${CUEBRIDGE_QUESTION_CONCURRENCY:-1}"
      --cuebridge-question-batch-size "${CUEBRIDGE_QUESTION_BATCH_SIZE:-1}"
      --cuebridge-fix-concurrency "${CUEBRIDGE_FIX_CONCURRENCY:-1}"
      --cuebridge-fix-batch-size "${CUEBRIDGE_FIX_BATCH_SIZE:-1}"
    )
    ;;
  question-oracle)
    args+=(
      --compare-cuebridge-question-oracle
      --cuebridge-provider openai-compatible
      --openai-base-url "${CUEBRIDGE_BASE_URL:-http://127.0.0.1:1234/v1}"
      --openai-model "${CUEBRIDGE_MODEL:-qwen3-4b-cuebridge}"
      --openai-api-key "${CUEBRIDGE_API_KEY:-local}"
      --llama-n-predict "${CUEBRIDGE_N_PREDICT:-1024}"
      --cuebridge-target-rank-threshold "${CUEBRIDGE_TARGET_RANK_THRESHOLD:-10}"
      --cuebridge-accept-rank-threshold "${CUEBRIDGE_ACCEPT_RANK_THRESHOLD:-20}"
      --cuebridge-min-rank-improvement "${CUEBRIDGE_MIN_RANK_IMPROVEMENT:-1}"
      --cuebridge-collateral-policy "${CUEBRIDGE_COLLATERAL_POLICY:-tier}"
      --cuebridge-fix-concurrency "${CUEBRIDGE_FIX_CONCURRENCY:-1}"
      --cuebridge-fix-batch-size "${CUEBRIDGE_FIX_BATCH_SIZE:-1}"
    )
    ;;
  *)
    echo "Unknown MODE=$MODE. Use raw, product-cuebridge, or question-oracle." >&2
    exit 1
    ;;
esac

if [[ "$MODE" != "raw" ]]; then
  args+=(--cuebridge-max-fix-cases "${CUEBRIDGE_MAX_FIX_CASES:-1000}")
fi

echo "Running BEAM $CONTEXT in $MODE mode with $SEMANTIC_MODE retrieval"
echo "Output: $OUTPUT"

run_status=0
if "${args[@]}"; then
  run_status=0
else
  run_status=$?
fi

if [[ "$TRACE_TIMING" == "1" && -s "$TIMING_FILE" ]]; then
  python "$SCRIPT_DIR/report_timing.py" --input "$TIMING_FILE"
  echo "Timing samples: $TIMING_FILE"
fi

exit "$run_status"
