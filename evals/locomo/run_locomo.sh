#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUEMAP_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CUEMAP_EVALS_DIR="${CUEMAP_EVALS_DIR:-$CUEMAP_ROOT/evals}"
HARNESS="$CUEMAP_EVALS_DIR/test_locomo_settled.py"

if [[ ! -f "$HARNESS" ]]; then
  echo "Missing LoCoMo harness: $HARNESS" >&2
  echo "Set CUEMAP_EVALS_DIR to the directory containing test_locomo_settled.py." >&2
  exit 1
fi

CUEMAP_URL="${CUEMAP_URL:-http://127.0.0.1:8080}"
LIMIT="${LIMIT:-20}"
EXPANSION_DEPTH="${EXPANSION_DEPTH:-10}"
MODE="${MODE:-raw}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/results}"
OUTPUT="${OUTPUT:-$OUT_DIR/locomo_${MODE}_depth${EXPANSION_DEPTH}.json}"

mkdir -p "$OUT_DIR"

args=(
  python "$HARNESS"
  --url "$CUEMAP_URL"
  --limit "$LIMIT"
  --expansion-depth "$EXPANSION_DEPTH"
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
      --cuebridge-target-rank-threshold "${CUEBRIDGE_TARGET_RANK_THRESHOLD:-20}"
      --cuebridge-max-samples "${CUEBRIDGE_MAX_SAMPLES:-500}"
      --cuebridge-max-jobs "${CUEBRIDGE_MAX_JOBS:-500}"
      --max-questions-per-memory "${MAX_QUESTIONS_PER_MEMORY:-8}"
      --cuebridge-max-fix-cases "${CUEBRIDGE_MAX_FIX_CASES:-200}"
    )
    ;;
  *)
    echo "Unknown MODE=$MODE. Use raw or product-cuebridge." >&2
    exit 1
    ;;
esac

echo "Running LoCoMo in $MODE mode"
echo "Expansion depth: $EXPANSION_DEPTH"
echo "Output: $OUTPUT"
exec "${args[@]}"
