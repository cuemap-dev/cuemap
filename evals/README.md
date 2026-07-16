# CueMap v0.7 Evaluation Pack

CueMap is a deterministic memory engine with an embedding-free, vector-database-free, LLM-free recall hot path. These reports document the v0.7 retrieval runs we used to calibrate release readiness across LongMemEval, LoCoMo, and BEAM.

The short version: CueMap is already very strong on compact long-memory retrieval, competitive on LoCoMo when adjacent context expansion is enabled, and competitive with reported leading BEAM bands at both 1M and 10M scale. CueBridge's most meaningful v0.7 diagnostic lift showed up on BEAM 128K, where question-oracle artifacts improved Hit@20 by 10 questions while preserving top-20 wins. The next lift is ranking dense multi-evidence sets, event ordering, summarization, and preference-following at larger BEAM scales.

## Results At A Glance

| Benchmark | Mode | Scale | Recall_Any@20 | Recall_Any@100 | Context tokens | Main read |
|---|---:|---:|---:|---:|---:|---|
| LongMemEval | Raw | 470 non-abstention Qs | 99.1% | n/a | Compact top-20 recall | Very strong baseline |
| LoCoMo | Raw, expansion-depth 10 | 1,986 Qs | 93.2% | n/a | 10,279 avg | Strong recall with explicit context expansion |
| LoCoMo | Raw, expansion-depth 3 | 1,986 Qs | 89.2% | n/a | 3,362 avg | Lean context setting |
| BEAM | Raw | 128K | 80.3% | 94.4% | Top-100 retrieval | Strong candidate discovery |
| BEAM | CueBridge diagnostic | 128K | 83.7% | 94.6% | Top-100 retrieval | +10 Hit@20 with preserved top-20 wins |
| BEAM | Raw | 1M | 69.9% | 86.2% | Top-100 retrieval | Inside the reported leading 64-72% band |
| BEAM | Raw | 10M | 51.7% | 71.6% | Top-100 retrieval | Inside the reported leading 48-64% band |

## Hot-Path Latency Context

These retrieval benchmarks focus on accuracy. For hot-path performance, the release latency runs measured:

| Dataset size | Operation | Throughput | Avg | P50 | P99 |
|---:|---|---:|---:|---:|---:|
| 100K memories | Write | 320 ops/s | 3.13 ms | 2.56 ms | 10.98 ms |
| 100K memories | Read, NL full | 574 ops/s | 1.73 ms | 1.65 ms | 3.67 ms |
| 1M memories | Write | 351 ops/s | 2.85 ms | 2.39 ms | 11.23 ms |
| 1M memories | Read, NL lean | 369 ops/s | 2.70 ms | 2.06 ms | 5.10 ms |

The benchmark harnesses add HTTP, ingest, scoring, and optional CueBridge generation overhead. The runtime point is that raw recall remains a deterministic, millisecond-class path.

## Reports

- [LongMemEval report](longmemeval/report.md)
- [LoCoMo report](locomo/report.md)
- [BEAM report](beam/report.md)

## Reproducing

Each benchmark has a shell wrapper in its directory:

```bash
bash evals/longmemeval/run_longmemeval.sh
bash evals/locomo/run_locomo.sh
bash evals/beam/run_beam.sh
```

The wrappers call the canonical Python harnesses under `<cuemap-root>/evals`. If your checkout is laid out differently, set `CUEMAP_EVALS_DIR`:

```bash
CUEMAP_EVALS_DIR=/path/to/cuemap/evals bash evals/beam/run_beam.sh
```

The default URL is `http://127.0.0.1:8080`. Start the Rust engine before running:

```bash
cuemap start
```

For disposable benchmark projects, keep `DELETE_PROJECTS=1` so temporary `eval_*` projects are deleted after each record.

## Reading The Numbers Correctly

These are raw retrieval metrics rather than LLM-as-judge answer accuracy. This is deliberate: raw retrieval shows what the memory engine actually found before answer-model interpretation.

CueBridge numbers in these reports are labeled carefully. The highlighted CueBridge lift is the BEAM 128K diagnostic question-oracle run, which uses benchmark questions to probe whether artifacts can improve ranking. It demonstrates mechanism and upside; product-mode CueBridge is the next packaging step.
