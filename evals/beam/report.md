# BEAM: Scaling Deterministic Recall To 10M Tokens

`CueMap v0.7.2` `BEAM 128K / 1M / 10M` `CueBridge lift` `embedding-free`

BEAM is the stress test for scale. CueMap uses deterministic lexical/facet recall instead of an embedding service or vector database, so this benchmark asks the central question: can that architecture stay competitive at 1M and 10M tokens?

The answer in v0.7.2 is yes. The latest raw 128K, 1M, and 10M runs reach 84.2%, 80.3%, and 67.0% Hit@20, with Hit@100 at 96.3%, 93.1%, and 83.5% respectively. All three runs use the wrapper's default **hybrid recall mode** (`SEMANTIC_MODE=hybrid`), message-level turn ingestion, evidence coverage disabled, and ordered reconstruction disabled.

## Headline

| Context tier | Questions | Hit@1 | Hit@5 | Hit@10 | Hit@20 | Hit@50 | Hit@100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128K | 355 | 49.6% | 71.8% | 77.5% | 84.2% | 90.4% | 96.3% |
| 1M | 625 | 38.4% | 63.4% | 74.6% | 80.3% | 90.4% | 93.1% |
| 10M | 176 | 33.5% | 50.0% | 58.5% | 67.0% | 77.8% | 83.5% |

The 10M tier crosses the 50% Hit@20 target while keeping Hit@100 above 80%. That matters because it shows CueMap handles the scale jump and often puts the right memory into the candidate set with embedding-free recall.

## Depth Metrics

| Context tier | Recall_All@20 | Recall_Frac@20 | NDCG@20 |
|---|---:|---:|---:|
| 128K | 54.9% | 69.2% | 54.7% |
| 1M | 34.9% | 54.4% | 41.6% |
| 10M | 23.9% | 38.1% | 29.7% |

This is where the next upside is visible. Hit@100 remains much higher than Hit@20, especially at 10M. CueMap is often finding relevant candidates, and a stronger deterministic reranker can move more of them into the top-20 answer context.

## 128K Category Breakdown

| Type | Hit@1 | Hit@5 | Hit@10 | Hit@20 | Hit@100 | Read |
|---|---:|---:|---:|---:|---:|---|
| Contradiction resolution | 85.0% | 100.0% | 100.0% | 100.0% | 100.0% | Excellent. |
| Event ordering | 17.5% | 57.5% | 67.5% | 85.0% | 97.5% | Candidate discovery is high; ordering is the next lift. |
| Information extraction | 50.0% | 72.5% | 80.0% | 90.0% | 95.0% | Strong base with top-rank upside. |
| Instruction following | 17.5% | 42.5% | 47.5% | 57.5% | 82.5% | Good candidate base for instruction-aware ranking. |
| Knowledge update | 87.5% | 97.5% | 97.5% | 97.5% | 97.5% | Strong. |
| Multi-session reasoning | 57.5% | 87.5% | 97.5% | 100.0% | 100.0% | Good candidate discovery. |
| Preference following | 23.1% | 46.2% | 51.3% | 61.5% | 94.9% | Strong target for semantic preference bridges. |
| Summarization | 13.9% | 38.9% | 52.8% | 63.9% | 100.0% | Candidate discovery high, with set-coverage upside. |
| Temporal reasoning | 90.0% | 100.0% | 100.0% | 100.0% | 100.0% | Strong. |

## 10M Category Breakdown

| Type | Hit@20 | Hit@100 | Read |
|---|---:|---:|---|
| Contradiction resolution | 95.0% | 100.0% | Robust at 10M. |
| Knowledge update | 95.0% | 100.0% | Strong. |
| Multi-session reasoning | 85.0% | 100.0% | Good candidate discovery with full-evidence upside. |
| Information extraction | 80.0% | 90.0% | Strong base with top-rank upside. |
| Temporal reasoning | 65.0% | 80.0% | Candidate set exists; rank lift is available. |
| Instruction following | 25.0% | 50.0% | Clear target for intent-aware scoring. |
| Event ordering | 50.0% | 85.0% | Clear target for order-aware reranking. |
| Summarization | 50.0% | 68.8% | Clear target for set-coverage reranking. |
| Preference following | 55.0% | 75.0% | Clear target for semantic preference bridges. |

## Historical CueBridge Diagnostic At 128K

This separate question-oracle calibration predates the latest raw 128K run. It
improved Hit@20 from `287/355` to `297/355`; do not compare its raw column
directly with the latest 128K baseline above.

| Metric | Raw | CueBridge diagnostic | Delta |
|---|---:|---:|---:|
| Hit@1 | 131/355 | 132/355 | +1 |
| Hit@5 | 217/355 | 219/355 | +2 |
| Hit@10 | 253/355 | 258/355 | +5 |
| Hit@20 | 287/355 | 297/355 | +10 |
| Recall_Frac@20 | 60.6% | 62.1% | +1.5 pp |
| NDCG@20 | 44.7% | 45.2% | +0.5 pp |

This is the CueBridge result worth highlighting for v0.7. It is a diagnostic question-oracle run: benchmark questions are used to test whether generated artifacts can promote rank-constrained evidence. The result matters because it shows the artifact path can improve ranking on a benchmark where the raw engine has real headroom, and it gives a concrete direction for product-mode CueBridge packaging.

## Product Read

BEAM is where CueMap's next release upside is most visible.

What works now:

| Strength | Evidence |
|---|---|
| Candidate discovery | 128K Hit@100 is 96.3%; 1M Hit@100 is 93.1%; 10M Hit@100 is 83.5%. |
| Contradictions and updates | These categories stay strong even at 10M. |
| Embedding-free architecture | The engine preserves strong 10M candidate discovery with deterministic indexing. |

Next lift areas:

| Area | Opportunity |
|---|---|
| Event ordering | Add order-aware reranking for cases where relevant memories are already in the candidate set. |
| Summarization | Promote broader evidence-set coverage into top-20 answer context. |
| Preference following | Use CueBridge-style semantic bridges with tighter artifact gates. |
| Instruction following | Add intent-aware scoring for instruction-shaped memories. |

The next practical upgrade is a deterministic reranker over the top-100 candidate set: evidence-set coverage, recency/order features, intent-specific scoring, and conservative CueBridge artifacts.

## Retrieved Context Footprint

Each BEAM JSON result now records `ctx_tokens` for the top-20 memory text for
every scored question, matching the benchmark's primary Hit@20 metric. This is
an approximate, model-agnostic token count (using the same word/punctuation
estimator as the LoCoMo harness), not the model tokenizer. The evaluator also
prints aggregate `Avg`, `P50`, `P95`, `P99`, and `Max` values, plus average/P95
values by question type. `ctx_tokens_returned` retains the count for the full
returned limit (normally top-100) for diagnostic comparison. When CueBridge
comparison is enabled, the raw baseline counts are retained under
`raw_result.ctx_tokens` and `raw_result.ctx_tokens_returned`.

| Run | Avg | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| Latest raw 128K, top-20 | 4,702 | 2,816 | 14,889 | 20,954 | 34,393 |
| Latest raw 1M, top-20 | 2,934 | 1,354 | 11,554 | 20,731 | 29,243 |
| Latest raw 10M, top-20 | 1,749 | 1,220 | 4,170 | 16,802 | 19,733 |

## Reproduce

Start CueMap first:

```bash
cuemap start
```

Raw 128K:

```bash
CONTEXT=128k bash evals/beam/run_beam.sh
```

The wrapper defaults to message-level ingestion (`cuemap add` / `/memories`),
which preserves one indexed memory per BEAM turn. To reproduce the older
segmented long-form path explicitly, set `INGEST_MODE=long-form`.

Raw 1M:

```bash
CONTEXT=1m bash evals/beam/run_beam.sh
```

Raw 10M:

```bash
CONTEXT=10m bash evals/beam/run_beam.sh
```

128K CueBridge diagnostic:

```bash
CONTEXT=128k \
MODE=question-oracle \
CUEBRIDGE_BASE_URL=http://127.0.0.1:1234/v1 \
CUEBRIDGE_MODEL=qwen3-4b-cuebridge \
CUEBRIDGE_API_KEY=local \
CUEBRIDGE_TARGET_RANK_THRESHOLD=10 \
bash evals/beam/run_beam.sh
```

Useful knobs:

| Variable | Default | Purpose |
|---|---:|---|
| `CONTEXT` | `128k` | BEAM tier: `128k`, `500k`, `1m`, or `10m`. |
| `LIMIT` | `100` | Recall limit; BEAM reports include @50 and @100. |
| `SEMANTIC_MODE` | `hybrid` | Retrieval mode: `lexical`, `semantic`, or `hybrid`. |
| `MODE` | `raw` | `raw`, `product-cuebridge`, or `question-oracle`. |
| `DELETE_PROJECTS` | `1` | Delete temporary eval projects after each record. |

The wrapper writes fresh output under `evals/beam/results/` by default. The figures above come from the latest v0.7.2 raw runs for all three tiers.
