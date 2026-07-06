#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUEMAP_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CUEMAP_EVALS_DIR="${CUEMAP_EVALS_DIR:-$CUEMAP_ROOT/evals}"
HARNESS="$CUEMAP_EVALS_DIR/test_beam_settled.py"

if [[ ! -f "$HARNESS" ]]; then
  echo "Missing BEAM harness: $HARNESS" >&2
  echo "Set CUEMAP_EVALS_DIR to the directory containing test_beam_settled.py." >&2
  exit 1
fi

CONTEXT="${CONTEXT:-128k}"
CUEMAP_URL="${CUEMAP_URL:-http://127.0.0.1:8080}"
LIMIT="${LIMIT:-100}"
MODE="${MODE:-raw}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/results}"
OUTPUT="${OUTPUT:-$OUT_DIR/beam_${CONTEXT}_${MODE}.json}"

mkdir -p "$OUT_DIR"

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
      --cuebridge-target-rank-threshold "${CUEBRIDGE_TARGET_RANK_THRESHOLD:-10}"
      --cuebridge-max-samples "${CUEBRIDGE_MAX_SAMPLES:-500}"
      --cuebridge-max-jobs "${CUEBRIDGE_MAX_JOBS:-500}"
      --max-questions-per-memory "${MAX_QUESTIONS_PER_MEMORY:-8}"
      --cuebridge-max-fix-cases "${CUEBRIDGE_MAX_FIX_CASES:-200}"
    )
    ;;
  question-oracle)
    args+=(
      --compare-cuebridge-question-oracle
      --cuebridge-provider openai-compatible
      --openai-base-url "${CUEBRIDGE_BASE_URL:-http://127.0.0.1:1234/v1}"
      --openai-model "${CUEBRIDGE_MODEL:-qwen3-4b-cuebridge}"
      --openai-api-key "${CUEBRIDGE_API_KEY:-local}"
      --cuebridge-target-rank-threshold "${CUEBRIDGE_TARGET_RANK_THRESHOLD:-10}"
      --cuebridge-max-fix-cases "${CUEBRIDGE_MAX_FIX_CASES:-200}"
    )
    ;;
  *)
    echo "Unknown MODE=$MODE. Use raw, product-cuebridge, or question-oracle." >&2
    exit 1
    ;;
esac

echo "Running BEAM $CONTEXT in $MODE mode"
echo "Output: $OUTPUT"
exec "${args[@]}"

