# LongMemEval: Compact Long-Memory Recall

`CueMap v0.7` `raw lexical recall` `near-saturated Hit@20`

LongMemEval is the cleanest showcase for CueMap's raw engine: compact user memories, compact retrieval depth, and a strong deterministic recall path. CueMap reaches near-saturated Hit@20 while keeping the recall path embedding-free and LLM-free.

## Headline

| Run | Questions | Hit@1 | Hit@5 | Hit@10 | Hit@20 | NDCG@20 |
|---|---:|---:|---:|---:|---:|---:|
| Raw CueMap | 470 | 55.1% | 92.1% | 98.1% | 99.1% | 74.2% |

The important signal is that the raw engine is already close to saturated at Hit@20. LongMemEval is the raw-recall showcase; BEAM 128K is the stronger CueBridge-lift showcase.

## Quality Metrics

| Metric | Raw |
|---|---:|
| Recall_All@5 | 65.3% |
| Recall_All@10 | 83.0% |
| Recall_All@20 | 90.4% |
| Recall_Frac@5 | 78.5% |
| Recall_Frac@10 | 91.2% |
| Recall_Frac@20 | 95.7% |
| NDCG@5 | 67.4% |
| NDCG@10 | 72.6% |
| NDCG@20 | 74.2% |

## Product Read

LongMemEval is already close to saturated at Hit@20, so the next product lift is top-rank quality and higher Recall_All for multi-fact questions.

Next lift areas:

| Area | Opportunity |
|---|---|
| Multi-session questions | Bring split evidence sets together more consistently. |
| Preference questions | Add more semantic paraphrase bridges through CueBridge. |
| Top-1 ranking | Convert more of the excellent Hit@20 performance into first-result precision. |

## Reproduce

Start CueMap first:

```bash
cuemap start
```

Raw run:

```bash
bash evals/longmemeval/run_longmemeval.sh
```

Useful knobs:

| Variable | Default | Purpose |
|---|---:|---|
| `LIMIT` | `20` | Recall limit used for scoring. |
| `VARIANT` | `core` | LongMemEval variant. |
| `DELETE_PROJECTS` | `1` | Delete temporary eval projects after each record. |
| `MODE` | `raw` | `raw`, `question-oracle`, or `product-cuebridge`; BEAM is the recommended CueBridge showcase. |

The wrapper writes fresh output under `evals/longmemeval/results/` by default. The metrics above came from the v0.7 release calibration run.
