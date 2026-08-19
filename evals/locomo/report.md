# LoCoMo: Long Conversation Recall With Low Context

`CueMap v0.7.2` `conversation memory` `context expansion`

LoCoMo stresses long conversations where the answer evidence is often near the recalled turn rather than exactly inside it. CueMap handles this best with engine-level expansion depth, which is a real product feature: return the matching memory plus adjacent conversation turns so an answer model receives the local neighborhood.

## Headline

| Setting | Questions | Hit@1 | Hit@5 | Hit@10 | Hit@20 | Avg ctx tokens |
|---|---:|---:|---:|---:|---:|---:|
| Lean context, expansion-depth 3 | 1,986 | 62.9% | 85.3% | 90.4% | 94.2% | 3,184 |
| Strong context, expansion-depth 10 | 1,986 | 66.1% | 87.7% | 92.5% | 96.1% | 10,044 |

The strong setting reaches 96.1% Hit@20 while still using far less context than many leaderboard-style RAG reports. As a reference point, public Agent Memory Benchmark views show some LoCoMo systems in the 14.7K to 36.2K context-token range. CueMap's lean run averages 3.2K tokens, and the stronger run averages 10.0K.

Both reported runs use the wrapper's default **hybrid recall mode**
(`SEMANTIC_MODE=hybrid`), combining lexical and semantic retrieval signals.

## Strong Setting Metrics

Settings: `limit=20`, `expansion-depth=10`.

| Metric | Score |
|---|---:|
| Recall_Any@1 | 66.1% |
| Recall_Any@5 | 87.7% |
| Recall_Any@10 | 92.5% |
| Recall_Any@20 | 96.1% |
| Recall_All@5 | 77.8% |
| Recall_All@10 | 83.6% |
| Recall_All@20 | 88.9% |
| Recall_Frac@5 | 82.5% |
| Recall_Frac@10 | 88.1% |
| Recall_Frac@20 | 92.7% |
| NDCG@5 | 74.6% |
| NDCG@10 | 76.6% |
| NDCG@20 | 78.0% |

## Lean Setting Metrics

Settings: `limit=20`, `expansion-depth=3`.

| Metric | Score |
|---|---:|
| Recall_Any@1 | 62.9% |
| Recall_Any@5 | 85.3% |
| Recall_Any@10 | 90.4% |
| Recall_Any@20 | 94.2% |
| Recall_All@5 | 76.0% |
| Recall_All@10 | 81.6% |
| Recall_All@20 | 86.3% |
| Recall_Frac@5 | 80.3% |
| Recall_Frac@10 | 86.0% |
| Recall_Frac@20 | 90.4% |
| NDCG@5 | 72.0% |
| NDCG@10 | 74.0% |
| NDCG@20 | 75.3% |

## Context Footprint

| Setting | Avg | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| Expansion-depth 3 | 3,184 | 3,242 | 3,950 | 4,244 | 5,689 |
| Expansion-depth 10 | 10,044 | 10,136 | 12,093 | 12,789 | 13,749 |

This is the main CueMap story on LoCoMo: competitive recall using compact retrieved neighborhoods instead of full-conversation context.

## By Question Type

| Type | Count | Hit@1 | Hit@5 | Hit@10 | Hit@20 | Read |
|---|---:|---:|---:|---:|---:|---|
| Multi-hop | 321 | 65.1% | 85.7% | 90.3% | 95.0% | Solid with local context. |
| Temporal reasoning | 96 | 29.2% | 54.2% | 64.6% | 80.2% | Strong candidate base for time-aware ranking lift. |
| Single-hop | 282 | 49.6% | 81.2% | 90.1% | 96.8% | Good Hit@20 with top-rank upside. |
| Common-sense | 841 | 73.2% | 91.7% | 95.2% | 97.1% | Strong. |
| Adversarial | 446 | 71.7% | 92.8% | 96.4% | 98.0% | Strong retrieval that gives the answer model evidence for abstention. |

## Product Read

LoCoMo is partly a neighboring-turn retrieval task. CueMap handles this directly with `expansion-depth`, returning compact local neighborhoods around matched memories.

Next lift areas:

| Area | Opportunity |
|---|---|
| Temporal reasoning | Add stronger ordering and recency signals for date and relative-time questions. |
| Single-hop top rank | Promote exact evidence turns from the already-retrieved local neighborhood. |
| CueBridge packaging | Use narrower, more conversation-aware artifact gates before enabling broad LoCoMo product mode. |

## Reproduce

Start CueMap first:

```bash
cuemap start
```

Strong LoCoMo run:

```bash
EXPANSION_DEPTH=10 \
LIMIT=20 \
bash evals/locomo/run_locomo.sh
```

Lean context run:

```bash
EXPANSION_DEPTH=3 \
LIMIT=20 \
bash evals/locomo/run_locomo.sh
```

Useful knobs:

| Variable | Default | Purpose |
|---|---:|---|
| `EXPANSION_DEPTH` | `10` | How many adjacent conversation turns to attach around recalled memories. |
| `LIMIT` | `20` | Recall limit. |
| `SEMANTIC_MODE` | `hybrid` | Retrieval mode: `lexical`, `semantic`, or `hybrid`. |
| `DELETE_PROJECTS` | `1` | Delete temporary eval projects after each record. |

The wrapper writes fresh output under `evals/locomo/results/` by default. The metrics above came from the full 1,986-question v0.7.2 rerun on 2026-08-15.
