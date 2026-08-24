#!/usr/bin/env bash

set -euo pipefail

ENGINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_URL="${CUEMAP_URL:-http://127.0.0.1:8080}"
PROJECT_ID="${CUEMAP_PROJECT_ID:-repository-recall-demo-v072}"
EXPECTED_MEMORY_COUNT=1932

for command in curl jq; do
  if ! command -v "$command" >/dev/null 2>&1; then
    printf 'Required command not found: %s\n' "$command" >&2
    exit 1
  fi
done

if ! curl -fsS "$BASE_URL/" >/dev/null 2>&1; then
  printf 'CueMap is not responding at %s. Start the engine first or set CUEMAP_URL.\n' \
    "$BASE_URL" >&2
  exit 1
fi

printf 'Server:  %s\n' "$BASE_URL"
printf 'Project: %s\n' "$PROJECT_ID"
printf 'Configuring the repository ingestion agent...\n'

ingest_payload="$(jq -n \
  --arg watch_dir "$ENGINE_ROOT" \
  '{
    watch_dir: $watch_dir,
    included_paths: [
      "src",
      "README.md",
      "CHANGELOG.md",
      "evals/README.md",
      "evals/beam/report.md",
      "evals/locomo/report.md",
      "evals/longmemeval/report.md"
    ],
    ignored_patterns: [],
    ignored_extensions: []
  }')"

curl -fsS "$BASE_URL/projects/$PROJECT_ID/watch-dir" \
  -H 'content-type: application/json' \
  --data "$ingest_payload" \
  | jq '{status, project_id, watch_dir, included_paths}'

printf 'Waiting for repository ingestion to begin...\n'
ingestion_started=0
status='{}'
for _ in $(seq 1 60); do
  status="$(curl -fsS "$BASE_URL/jobs/status" -H "X-Project-ID: $PROJECT_ID")"
  if [[ "$(jq -r '.writes_total // 0' <<<"$status")" -gt 0 ]]; then
    ingestion_started=1
    break
  fi
  sleep 1
done

if [[ "$ingestion_started" != "1" ]]; then
  printf 'Repository ingestion did not begin. Last status:\n%s\n' "$status" >&2
  exit 1
fi

printf 'Waiting for repository ingestion and local intent annotations...\n'
intent_ready=0
for _ in $(seq 1 300); do
  status="$(curl -fsS "$BASE_URL/jobs/status" -H "X-Project-ID: $PROJECT_ID")"
  if [[ "$(jq -r '.phase // ""' <<<"$status")" == "done" \
    && "$(jq -r '.intent_ready // false' <<<"$status")" == "true" ]]; then
    intent_ready=1
    break
  fi
  sleep 1
done

if [[ "$intent_ready" != "1" ]]; then
  printf 'Intent annotations did not finish. Last status:\n%s\n' "$status" >&2
  exit 1
fi
stored_count="$(jq -r '.intent_memory_total // 0' <<<"$status")"
printf 'Stored memories: %s\n' "$stored_count"
printf 'Intent annotations: %s/%s\n' \
  "$(jq -r '.intent_annotated' <<<"$status")" \
  "$stored_count"
if [[ "$stored_count" != "$EXPECTED_MEMORY_COUNT" ]]; then
  printf 'Published trace count: %s; this checkout produced: %s.\n' \
    "$EXPECTED_MEMORY_COUNT" "$stored_count" >&2
  printf 'Continuing so source or chunker drift remains visible.\n' >&2
fi

run_trace() {
  local title="$1"
  local query="$2"
  local semantic_mode="$3"
  local expansion_depth="$4"
  local payload
  local response

  payload="$(jq -n \
    --arg query_text "$query" \
    --arg semantic_mode "$semantic_mode" \
    --argjson expansion_depth "$expansion_depth" \
    '{
      query_text: $query_text,
      semantic_mode: $semantic_mode,
      limit: 10,
      min_intersection: 1,
      auto_reinforce: false,
      explain: true,
      trace_timing: false,
      depth: 1,
      expansion_depth: $expansion_depth,
      disable_alias_expansion: true,
      parent_fusion: "off",
      ordered_reconstruction: "off",
      evidence_coverage: "off"
    }')"

  response="$(curl -fsS "$BASE_URL/recall" \
    -H 'content-type: application/json' \
    -H "X-Project-ID: $PROJECT_ID" \
    --data "$payload")"

  jq \
    --arg title "$title" \
    --arg query "$query" \
    --arg semantic_mode "$semantic_mode" \
    --argjson expansion_depth "$expansion_depth" \
    '
      def relative_source:
        (.metadata.source_path // "unknown")
        | split("/rust_engine/")
        | if length > 1 then .[-1] else .[0] end;
      def excerpt:
        gsub("[\\n\\r\\t ]+"; " ")
        | if length > 280 then .[0:277] + "..." else . end;
      {
        example: $title,
        query: $query,
        semantic_mode: $semantic_mode,
        recall_knobs: {
          limit: 10,
          min_intersection: 1,
          expansion_depth: $expansion_depth,
          parent_fusion: "off",
          ordered_reconstruction: "off",
          evidence_coverage: "off"
        },
        query_cues_complete: .explain.query_cues,
        added_weighted_facets: .explain.query_plan.weighted_cues,
        query_labels_complete: .explain.query_plan.labels,
        expanded_cues_complete: .explain.expanded_cues,
        results: [
          .results
          | to_entries[]
          | {
              rank: (.key + 1),
              memory_id: .value.memory_id,
              source: (.value | relative_source),
              score: .value.score,
              intersection_count: .value.intersection_count,
              excerpt: (.value.content | excerpt)
            }
        ]
      }
    ' <<<"$response"
}

printf '\n=== Coding-agent example ===\n'
run_trace \
  "Coding agent: inspect temporal-facet extraction" \
  "How are temporal facets extracted?" \
  "lexical" \
  1
run_trace \
  "Coding agent: inspect temporal-facet extraction" \
  "How are temporal facets extracted?" \
  "hybrid" \
  1

printf '\n=== Product/release example ===\n'
run_trace \
  "Product question: corroborate a release claim" \
  "Was bundled semantic reranking local in the August 4 release?" \
  "lexical" \
  2
run_trace \
  "Product question: corroborate a release claim" \
  "Was bundled semantic reranking local in the August 4 release?" \
  "hybrid" \
  2
