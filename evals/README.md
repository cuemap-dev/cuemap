# CueMap v0.7.3 Evaluation Pack

CueMap is a deterministic memory engine with an embedding-free, vector-database-free, LLM-free recall hot path. These reports document the v0.7.3 retrieval runs used to calibrate release readiness across LongMemEval, LoCoMo, and BEAM.

The short version: CueMap is already very strong on compact long-memory retrieval, competitive on LoCoMo when adjacent context expansion is enabled, and shows strong candidate discovery at BEAM 10M scale. The latest raw BEAM 128K, 1M, and 10M runs reach 84.2%, 80.3%, and 67.0% Hit@20, with 4,702, 2,934, and 1,749 average top-20 context tokens respectively. The separate historical CueBridge question-oracle run improved Hit@20 by 10 questions; it is reported independently from the latest raw baselines.

The latest LoCoMo, LongMemEval, and BEAM runs use the wrappers' default `SEMANTIC_MODE=hybrid`, so their reported retrieval combines lexical and semantic signals. Set `SEMANTIC_MODE=lexical` or `SEMANTIC_MODE=semantic` when intentionally comparing a single retrieval mode.

## Results At A Glance

| Benchmark | Mode | Scale | Recall_Any@20 | Recall_Any@100 | Context tokens | Main read |
|---|---:|---:|---:|---:|---:|---|
| LongMemEval | Raw · hybrid | 470 non-abstention Qs | 96.2% | n/a | 3,914 avg top-20 | Very strong baseline |
| LoCoMo | Raw, expansion-depth 10 | 1,986 Qs | 96.1% | n/a | 10,044 avg | Strong recall with explicit context expansion |
| LoCoMo | Raw, expansion-depth 3 | 1,986 Qs | 94.2% | n/a | 3,184 avg | Lean context setting |
| BEAM | Raw | 128K | 84.2% | 96.3% | 4,702 avg top-20 | Strong candidate discovery |
| BEAM | Historical CueBridge diagnostic | 128K | 83.7% | 94.6% | Previous calibration | +10 Hit@20 in a separate run |
| BEAM | Raw | 1M | 80.3% | 93.1% | 2,934 avg top-20 | Latest raw run |
| BEAM | Raw | 10M | 67.0% | 83.5% | 1,749 avg top-20 | Latest raw run |

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

Each wrapper pins CLI fallbacks to `rust_engine/target/release/cuemap` and
prints the resolved binary and version before running. Override it explicitly
with `CUEMAP_RUST_BIN=/path/to/cuemap` when comparing another build.

BEAM defaults to message-level turn ingestion. Use `INGEST_MODE=long-form` only
when intentionally evaluating the segmented `/ingest/content` path.

The LoCoMo, LongMemEval, and BEAM harnesses report an approximate retrieved
context-token footprint. Each scored question stores `ctx_tokens` in its JSON
record, and the run summary prints Avg/P50/P95/P99/Max plus question-type
breakdowns. For BEAM, `ctx_tokens` is explicitly the top-20 footprint; the full
returned-limit count is available as `ctx_tokens_returned`. The estimate is
tokenizer-independent (word and punctuation counting), so use it for relative
context-budget comparisons rather than exact model billing.

The default URL is `http://127.0.0.1:8080`. Start the Rust engine before running:

```bash
rust_engine/target/release/cuemap start
```

For disposable benchmark projects, keep `DELETE_PROJECTS=1` so temporary `eval_*` projects are deleted after each record.

## Reading The Numbers Correctly

These are raw retrieval metrics rather than LLM-as-judge answer accuracy. This is deliberate: raw retrieval shows what the memory engine actually found before answer-model interpretation.

CueBridge numbers in these reports are labeled carefully. The highlighted CueBridge lift is the BEAM 128K diagnostic question-oracle run, which uses benchmark questions to probe whether artifacts can improve ranking. It demonstrates mechanism and upside; product-mode CueBridge is the next packaging step.
