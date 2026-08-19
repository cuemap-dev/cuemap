# LongMemEval: Compact Long-Memory Recall

`CueMap v0.7.2` `raw hybrid recall` `near-saturated Hit@20`

LongMemEval is the cleanest showcase for CueMap's raw engine: compact user memories, compact retrieval depth, and a strong deterministic recall path. CueMap reaches near-saturated Hit@20 while keeping the recall path embedding-free and LLM-free. The latest run uses the wrapper default `SEMANTIC_MODE=hybrid`, combining lexical and semantic retrieval signals.

## Headline

| Run | Questions | Hit@1 | Hit@5 | Hit@10 | Hit@20 | NDCG@20 |
|---|---:|---:|---:|---:|---:|---:|
| Raw CueMap · hybrid | 470 | 57.9% | 90.2% | 94.9% | 96.2% | 75.1% |

The important signal is that the raw engine is already close to saturated at Hit@20. LongMemEval is the raw-recall showcase; BEAM 128K is the stronger CueBridge-lift showcase.

## Quality Metrics

| Metric | Raw hybrid |
|---|---:|
| Recall_All@5 | 73.0% |
| Recall_All@10 | 82.1% |
| Recall_All@20 | 88.1% |
| Recall_Frac@5 | 81.8% |
| Recall_Frac@10 | 88.8% |
| Recall_Frac@20 | 92.9% |
| NDCG@5 | 70.8% |
| NDCG@10 | 73.7% |
| NDCG@20 | 75.1% |

## Question-Type Breakdown

The run excluded 30 abstention cases before scoring. The remaining 470
questions break down as follows; context figures are approximate top-20 token
footprints for each type.

| Question type | Questions | Hit@1 | Hit@5 | Hit@10 | Hit@20 | Avg ctx | P95 ctx |
|---|---:|---:|---:|---:|---:|---:|---:|
| Single-session user | 64 | 75.0% | 98.4% | 98.4% | 98.4% | 4,338 | 6,965 |
| Multi-session | 121 | 62.8% | 90.1% | 95.9% | 95.9% | 3,969 | 6,027 |
| Single-session preference | 30 | 30.0% | 66.7% | 86.7% | 86.7% | 3,543 | 5,338 |
| Temporal reasoning | 127 | 59.1% | 88.2% | 91.3% | 91.3% | 3,799 | 5,560 |
| Knowledge update | 72 | 61.1% | 98.6% | 100.0% | 100.0% | 4,266 | 6,645 |
| Single-session assistant | 56 | 35.7% | 87.5% | 94.6% | 94.6% | 3,318 | 5,941 |

## Product Read

LongMemEval is already close to saturated at Hit@20, so the next product lift is top-rank quality and higher Recall_All for multi-fact questions.

Next lift areas:

| Area | Opportunity |
|---|---|
| Multi-session questions | Bring split evidence sets together more consistently. |
| Preference questions | Add more semantic paraphrase bridges through CueBridge. |
| Top-1 ranking | Convert more of the excellent Hit@20 performance into first-result precision. |

## Retrieved Context Footprint

Each LongMemEval JSON result now records `ctx_tokens` for the selected recall
attempt and `raw_ctx_tokens` for the raw baseline. These are approximate,
model-agnostic counts of the returned memory text, using the same
word/punctuation estimator as LoCoMo. The evaluator prints aggregate `Avg`,
`P50`, `P95`, `P99`, and `Max` values, plus average/P95 values by question type.

| Run | Avg | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| Latest raw hybrid, top-20 | 3,914 | 3,771 | 6,263 | 7,250 | 10,812 |

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
| `SEMANTIC_MODE` | `hybrid` | Retrieval mode: `lexical`, `semantic`, or `hybrid`. |
| `DELETE_PROJECTS` | `1` | Delete temporary eval projects after each record. |
| `MODE` | `raw` | `raw`, `question-oracle`, or `product-cuebridge`; BEAM is the recommended CueBridge showcase. |

The wrapper writes fresh output under `evals/longmemeval/results/` by default. The metrics above came from the latest v0.7.2 raw hybrid run: 470 scored questions with 30 abstention cases excluded.
