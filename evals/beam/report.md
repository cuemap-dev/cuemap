# BEAM: Scaling Deterministic Recall To 10M Tokens

`CueMap v0.7` `BEAM 128K / 1M / 10M` `CueBridge lift` `embedding-free`

BEAM is the stress test for scale. CueMap uses deterministic lexical/facet recall instead of an embedding service or vector database, so this benchmark asks the central question: can that architecture stay competitive at 1M and 10M tokens?

The answer in v0.7 is yes. CueMap reaches 69.9% Hit@20 at BEAM 1M and 51.7% Hit@20 at BEAM 10M, which sits inside the reported leading ranges of roughly 64-72% for 1M and 48-64% for 10M. BEAM 128K is also where the current CueBridge diagnostic run produced the most meaningful lift.

## Headline

| Context tier | Questions | Hit@1 | Hit@5 | Hit@10 | Hit@20 | Hit@50 | Hit@100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128K | 355 | 36.9% | 61.4% | 70.7% | 80.3% | 89.6% | 94.4% |
| 1M | 625 | 28.6% | 50.1% | 61.8% | 69.9% | 80.3% | 86.2% |
| 10M | 176 | 23.3% | 38.1% | 44.3% | 51.7% | 61.9% | 71.6% |

The 10M tier crosses the 50% Hit@20 target while keeping Hit@100 above 70%. That matters because it shows CueMap handles the scale jump and often puts the right memory into the candidate set with embedding-free recall.

## Depth Metrics

| Context tier | Recall_All@20 | Recall_Frac@20 | NDCG@20 |
|---|---:|---:|---:|
| 128K | 46.5% | 60.0% | 44.5% |
| 1M | 25.0% | 42.7% | 33.2% |
| 10M | 17.0% | 28.4% | 23.3% |

This is where the next upside is visible. Hit@100 remains much higher than Hit@20, especially at 10M. CueMap is often finding relevant candidates, and a stronger deterministic reranker can move more of them into the top-20 answer context.

## 128K Category Breakdown

| Type | Hit@1 | Hit@5 | Hit@10 | Hit@20 | Hit@100 | Read |
|---|---:|---:|---:|---:|---:|---|
| Contradiction resolution | 72.5% | 90.0% | 92.5% | 100.0% | 100.0% | Excellent. |
| Event ordering | 10.0% | 35.0% | 57.5% | 70.0% | 100.0% | Candidate discovery is high; ordering is the next lift. |
| Information extraction | 35.0% | 62.5% | 67.5% | 75.0% | 92.5% | Strong base with top-rank upside. |
| Instruction following | 35.0% | 45.0% | 45.0% | 47.5% | 72.5% | Good candidate base for instruction-aware ranking. |
| Knowledge update | 47.5% | 72.5% | 80.0% | 95.0% | 97.5% | Strong. |
| Multi-session reasoning | 42.5% | 65.0% | 82.5% | 92.5% | 100.0% | Good candidate discovery. |
| Preference following | 25.6% | 59.0% | 64.1% | 74.4% | 87.2% | Strong target for semantic preference bridges. |
| Summarization | 11.1% | 33.3% | 52.8% | 69.4% | 100.0% | Candidate discovery high, with set-coverage upside. |
| Temporal reasoning | 50.0% | 87.5% | 92.5% | 97.5% | 100.0% | Strong. |

## 10M Category Breakdown

| Type | Hit@20 | Hit@100 | Read |
|---|---:|---:|---|
| Contradiction resolution | 90.0% | 100.0% | Robust at 10M. |
| Knowledge update | 80.0% | 90.0% | Strong. |
| Multi-session reasoning | 70.0% | 85.0% | Good candidate discovery with full-evidence upside. |
| Information extraction | 65.0% | 75.0% | Strong base with top-rank upside. |
| Temporal reasoning | 40.0% | 80.0% | Candidate set exists; rank lift is available. |
| Instruction following | 40.0% | 50.0% | Clear target for intent-aware scoring. |
| Event ordering | 30.0% | 65.0% | Clear target for order-aware reranking. |
| Summarization | 31.2% | 43.8% | Clear target for set-coverage reranking. |
| Preference following | 15.0% | 50.0% | Clear target for semantic preference bridges. |

## CueBridge Diagnostic At 128K

Question-oracle CueBridge mode improved BEAM 128K Hit@20 from `287/355` to `297/355`.

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
| Candidate discovery | 128K Hit@100 is 94.4%; 10M Hit@100 is 71.6%. |
| Contradictions and updates | These categories stay strong even at 10M. |
| Embedding-free architecture | The engine stays in the reported leading 10M band with deterministic indexing. |

Next lift areas:

| Area | Opportunity |
|---|---|
| Event ordering | Add order-aware reranking for cases where relevant memories are already in the candidate set. |
| Summarization | Promote broader evidence-set coverage into top-20 answer context. |
| Preference following | Use CueBridge-style semantic bridges with tighter artifact gates. |
| Instruction following | Add intent-aware scoring for instruction-shaped memories. |

The next practical upgrade is a deterministic reranker over the top-100 candidate set: evidence-set coverage, recency/order features, intent-specific scoring, and conservative CueBridge artifacts.

## Reproduce

Start CueMap first:

```bash
cuemap start
```

Raw 128K:

```bash
CONTEXT=128k bash evals/beam/run_beam.sh
```

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
| `MODE` | `raw` | `raw`, `product-cuebridge`, or `question-oracle`. |
| `DELETE_PROJECTS` | `1` | Delete temporary eval projects after each record. |

The wrapper writes fresh output under `evals/beam/results/` by default. The metrics above came from the v0.7 release calibration runs for `128k`, `1m`, and `10m`.
