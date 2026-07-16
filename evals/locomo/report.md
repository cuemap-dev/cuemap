# LoCoMo: Long Conversation Recall With Low Context

`CueMap v0.7` `conversation memory` `context expansion`

LoCoMo stresses long conversations where the answer evidence is often near the recalled turn rather than exactly inside it. CueMap handles this best with engine-level expansion depth, which is a real product feature: return the matching memory plus adjacent conversation turns so an answer model receives the local neighborhood.

## Headline

| Setting | Questions | Hit@1 | Hit@5 | Hit@10 | Hit@20 | Avg ctx tokens |
|---|---:|---:|---:|---:|---:|---:|
| Lean context, expansion-depth 3 | 1,986 | 50.3% | 75.3% | 83.1% | 89.2% | 3,362 |
| Strong context, expansion-depth 10 | 1,986 | 56.8% | 80.5% | 88.2% | 93.2% | 10,279 |

The strong setting crosses 93% Hit@20 while still using far less context than many leaderboard-style RAG reports. As a reference point, public Agent Memory Benchmark views show some LoCoMo systems in the 14.7K to 36.2K context-token range. CueMap's lean run was 3.4K average tokens, and the stronger run was 10.3K.

## Strong Setting Metrics

Settings: `limit=20`, `expansion-depth=10`.

| Metric | Score |
|---|---:|
| Recall_Any@1 | 56.8% |
| Recall_Any@5 | 80.5% |
| Recall_Any@10 | 88.2% |
| Recall_Any@20 | 93.2% |
| Recall_All@5 | 71.7% |
| Recall_All@10 | 79.4% |
| Recall_All@20 | 85.5% |
| Recall_Frac@5 | 75.6% |
| Recall_Frac@10 | 83.8% |
| Recall_Frac@20 | 89.5% |
| NDCG@5 | 66.2% |
| NDCG@10 | 69.2% |
| NDCG@20 | 70.9% |

## Context Footprint

| Setting | Avg | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| Expansion-depth 3 | 3,362 | 3,395 | 4,047 | 4,272 | 4,926 |
| Expansion-depth 10 | 10,279 | 10,269 | 12,248 | 12,981 | 13,656 |

This is the main CueMap story on LoCoMo: competitive recall using compact retrieved neighborhoods instead of full-conversation context.

## By Question Type

| Type | Count | Hit@1 | Hit@5 | Hit@10 | Hit@20 | Read |
|---|---:|---:|---:|---:|---:|---|
| Multi-hop | 321 | 52.0% | 74.8% | 83.8% | 90.7% | Solid with local context. |
| Temporal reasoning | 96 | 30.2% | 56.2% | 63.5% | 78.1% | Strong candidate base for time-aware ranking lift. |
| Single-hop | 282 | 39.7% | 69.9% | 83.3% | 89.7% | Good Hit@20 with top-rank upside. |
| Common-sense | 841 | 63.5% | 85.3% | 92.2% | 96.0% | Strong. |
| Adversarial | 446 | 64.3% | 87.4% | 92.4% | 95.3% | Strong retrieval that gives the answer model evidence for abstention. |

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
| `DELETE_PROJECTS` | `1` | Delete temporary eval projects after each record. |

The wrapper writes fresh output under `evals/locomo/results/` by default. The metrics above came from the v0.7 release calibration run.
