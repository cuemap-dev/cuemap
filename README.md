# CueMap Rust Engine

[![CI](https://github.com/cuemap-dev/cuemap/actions/workflows/coverage.yml/badge.svg?branch=v0.7.2)](https://github.com/cuemap-dev/cuemap/actions/workflows/coverage.yml)
[![Coverage](https://codecov.io/github/cuemap-dev/cuemap/branch/v0.7.2/graph/badge.svg?flag=rust-engine)](https://app.codecov.io/github/cuemap-dev/cuemap)

**High-performance temporal-associative memory store** designed for dynamic contextual retrieval.

## Overview

CueMap implements a **Continuous Gradient Algorithm** optimized for associative data structures:

1.  **Intersection (Context Filter)**: Triangulates relevant memories by overlapping cues
2.  **Structural Extraction**: Emits deterministic cues for observable evidence such as dates, numbers, lists, source metadata, and surface entities.
3.  **Recency & Salience (Signal Dynamics)**: Balances fresh data with salient, high-signal events prioritized by an adaptive impact scoring module.
4.  **Reinforcement (Access-based Learning)**: Frequently accessed memories gain signal strength, remaining highly accessible even as they age.
5.  **Sparse Recall**: Uses normalized lexical cues, structural facets, recency, salience, and bounded deterministic reranking.

As of v0.7.2, CueMap's default core path is deterministic and ontology-free. GloVe/Ollama cue generation, WordNet/POS expansion, semantic bridges, pattern completion, external lexicon graphs, context expansion/speculation endpoints, and autonomous consolidation have been removed from the default engine path. v0.7.2 bundles a qint8 `all-MiniLM-L3-v2` vector layer for semantic reranking, intent classification, and query embeddings; the `edge` profile selects a q4 build of the same model. The encoder can still be disabled for constrained builds or deployments.

v0.7.2 also uses numeric per-project memory IDs everywhere. If callers need deterministic upsert/dedupe identity, pass `source_key`; memory IDs remain compact runtime addresses.

Built with Rust for maximum performance and reliability.

## Quick Start

### Build & Run

```bash
# Production (optimized)
cargo build --release
./target/release/cuemap start --port 8080

# Development
cargo run -- start
```

CueMap treats the nlprule tokenizer as a runtime asset, not a build artifact. Set `TOKENIZER_PATH` to a compiled tokenizer `.bin` file, or place `en_tokenizer.bin` under `~/.cuemap/data`. The production Docker image downloads the same checksum-pinned tokenizer used by the install script and packages it at `/app/assets/en_tokenizer.bin`.

### Docker

```bash
docker build -t cuemap/engine:0.7.2 .
docker run -p 8080:8080 -v "$(pwd)/local_snapshot_dir:/app/data" cuemap/engine:0.7.2
```

The container runs as the unprivileged `cuemap` user. Ensure a bind-mounted data directory is writable by UID/GID `10001`, or use a Docker-managed volume. Runtime defaults can be overridden with `CUEMAP_PORT`, `CUEMAP_DATA_DIR`, `CUEMAP_SNAPSHOT_INTERVAL_SECONDS`, `TOKENIZER_PATH`, and `RUST_LOG`.

### Native npm packages

Build the Darwin ARM64, Darwin x64, and Linux x64 native packages without publishing them:

```bash
./scripts/build-npm-native-packages.sh
./scripts/verify-npm-native-packages.sh
```

The packager builds Linux on Debian Bookworm for an older glibc baseline, bundles the checksum-pinned tokenizer, and writes package tarballs plus `SHA256SUMS` under `dist/npm-native/tarballs`.

### CLI Commands

CueMap provides a unified CLI for server management, ingestion, and interaction.

Install CLI:

```bash
# Install CLI
cargo install --path . --locked
```

```bash
cuemap <COMMAND> [OPTIONS]
```

#### Core Commands
- **`start`**: Start the CueMap server.
- **`stop`**: Stop the background server instance.
- **`status`**: Check server health, metrics, and background jobs.
- **`logs`**: View or tail server logs.

#### Interaction
- **`add`**: Add a memory via natural language.
- **`recall`**: Search memories (supports Grounded Recall and Web Recall).
- **`ingest`**: Ingest data from files or URLs.
- **`projects`**: Create and list projects.
- **`set-project`**: Set the default project for the current session.
- **`set-watch-dir`**: Set a watch directory for a project (enables agent).

#### Deterministic Semantics
- **`lexicon`**: Inspect lexicon entries and wire/unwire cues.
- **`alias`**: Manage explicit deterministic aliases.

Hint: Use `cuemap --help` to see available commands and options.

## Configuration

CueMap uses a layered configuration system that prioritizes settings in the following order:
**CLI Args** > **Env Vars** > **`server_config.toml`** > **Defaults**.

This system allows you to:
1.  **Centralize Settings**: Manage server options, security keys, and engine tuning in `~/.cuemap/server_config.toml`.
2.  **Fine-Tune Performance**: Adjust critical engine parameters like scoring weights, scan limits, and bounded fallback thresholds via the `[tuning]` section without recompiling.
3.  **Manage Project Context**: Use `set-watch-dir` to persist project-specific settings (like agent watch paths) in `.meta.json` files alongside your data.

## Self-Learning Agent (Zero-Friction Ingestion)

CueMap includes a **Self-Learning Agent** that automatically watches local directories, extracts structured "facts", and ingests them into your memory store.

### Automated Bootstrapping

On startup, if `--agent-dir` is provided, CueMap initializes the **Self-Learning Agent**.

### Example

```bash
# Point CueMap at your project
./target/release/cuemap start --agent-dir ~/projects/my-app

# The agent will automatically:
# 1. Supercharged Structural Ingestion (Rust, Python, Go, JS/TS, PHP, Java).
#    - Native tree-sitter queries capture definitions, calls, and imports as grounded cues.
# 2. Document & Data Parsing (PDF, Word, Excel, JSON, CSV, YAML, XML).
#    - Extracts headers, keys, and metadata as structural metadata.
# 3. Source-aware chunking: related chunks share stable parent/session/order metadata for optional bounded reconstruction during recall.
```

## AI Agent Integration (MCP Server)

CueMap provides a native Model Context Protocol (MCP) server, allowing AI coding assistants (like Claude Desktop, Cursor, and Windsurf) to instantly recall codebase context using the engine.

### Setup

We provide a zero-config NPM package that automatically downloads and manages the CueMap background engine.

Add the MCP server to your AI agent's configuration (e.g., Claude Desktop, Cursor, or Windsurf):
```json
{
  "mcpServers": {
    "cuemap": {
      "command": "npx",
      "args": [
        "-y",
        "cuemap-mcp"
      ],
      "env": {
        "CUEMAP_PORT": "8080"
      }
    }
  }
}
```

Once configured, the AI Agent can use the `cuemap_init` and `cuemap_recall` tools to query your codebase memories natively. There is no need to manually start the engine.

## Project Management & Persistence

CueMap provides complete project isolation with automatic persistence:

### Features

- **Project Isolation**: Each project has its own memory space, identified by `X-Project-ID` header.
- **Auto-Save on Shutdown**: All projects are saved on graceful shutdown when persistence is enabled.
- **Auto-Load on Startup**: Snapshots are restored from the configured data directory when persistence is enabled.
- **Zero Configuration**: Works out of the box

### Usage

CueMap runs in multi-tenant mode by default. Simply specify a project ID in your requests.

```bash
# Start server
./target/release/cuemap start --port 8080
```

### Example

```bash
# Add memory to project
curl -X POST http://localhost:8080/memories \
  -H "X-Project-ID: my-project" \
  -H "Content-Type: application/json" \
  -d '{"content": "Important data", "cues": ["test"]}'

# Stop server (Ctrl+C) - saves all projects when persistence is enabled
# Restart server - loads persisted snapshots
# Data persists across restarts unless snapshots are disabled.
```

### Snapshot Management

Snapshots are automatically managed:
- **Created**: Periodically and on graceful shutdown (SIGINT/Ctrl+C) when persistence is enabled.
- **Loaded**: On server startup
- **Disabled**: `--disable-snapshots` turns off periodic and shutdown snapshot saves.
- **Location**: `~/.cuemap/data/snapshots/` by default, or `<--data-dir>/snapshots` when `--data-dir` is set. Older installs may also be discovered under the legacy sibling `snapshots/` directory.
- **Format**: zstd-compressed JSON inside `.bin` files. This preserves arbitrary metadata reliably while keeping snapshots compact; older uncompressed bincode snapshots remain readable when their metadata can be decoded.
- **Migration note**: Some pre-v0.7.2 bincode snapshots that contain dynamic JSON metadata cannot be decoded by bincode's `deserialize_any` limitation. Those projects are reported at startup and must be reingested or exported from a compatible older binary before upgrading.
- **Files**: `{project-id}.bin`, `{project-id}_lexicon.bin`, `{project-id}_aliases.bin`

### Cloud Backup

CueMap supports secure offsite backups to AWS S3, Google Cloud Storage, and Azure Blob Storage.

**Configuration**:
Enable cloud backup via CLI flags or `~/.cuemap/server_config.toml`.

```bash
# S3 Example
./target/release/cuemap start \
  --cloud-backup s3 \
  --cloud-bucket my-backup-bucket \
  --cloud-region us-east-1
```

**Supported Providers**:
- `s3`: AWS S3 or compatible (MinIO, DigitalOcean Spaces)
- `gcs`: Google Cloud Storage
- `azure`: Azure Blob Storage
- `local`: Local path (for testing/replication)

**Management**:
Backups can be triggered manually via API (`/backup/upload`, `/backup/download`).


## Authentication

Secure your CueMap instance with API key authentication.

### Enable Authentication

Set an API key via environment variable:

```bash
# Single API key
CUEMAP_API_KEY=your-secret-key ./target/release/cuemap start --port 8080

# Multiple API keys (comma-separated)
CUEMAP_API_KEYS=key1,key2,key3 ./target/release/cuemap start --port 8080
```

Or configure keys in `~/.cuemap/server_config.toml`:

```toml
[security]
api_keys = ["your-secret-key"]
```

### Using Authentication

Include the API key in the `X-API-Key` header:

```bash
# Without auth (fails if enabled)
curl http://localhost:8080/stats
# Response: Missing X-API-Key header

# With correct key
curl -H "X-API-Key: your-secret-key" -H "X-Project-ID: default" http://localhost:8080/stats
# Response: {"total_memories": 1000, ...}

# With wrong key
curl -H "X-API-Key: wrong-key" -H "X-Project-ID: default" http://localhost:8080/stats
# Response: Invalid API key
```

### SDK Usage

#### Standard SDKs

Python:
```python
from cuemap import CueMap

# With authentication
client = CueMap(
    url="http://localhost:8080",
    api_key="your-secret-key"
)

client.add("Memory", cues=["test"])
```

TypeScript:
```typescript
import CueMap from 'cuemap';

const client = new CueMap({
  url: 'http://localhost:8080',
  apiKey: 'your-secret-key'
});

await client.add('Memory', ['test']);
```

### Docker with Authentication

```bash
docker run -p 8080:8080 -v "$(pwd)/local_snapshot_dir:/app/data" \
  -e CUEMAP_API_KEY=your-secret-key \
  cuemap/engine
```

### Security Notes

- Authentication is **disabled by default** (no keys = no auth required, unless `security.require_auth=true` is set in config)
- Keys can be loaded from `security.api_keys` in `~/.cuemap/server_config.toml` or from `CUEMAP_API_KEY` / `CUEMAP_API_KEYS`.
- Use strong, randomly generated keys in production
- Rotate keys regularly
- Use HTTPS in production to protect keys in transit

## Encryption

CueMap supports **encryption-at-rest** for all memory content using modern authenticated encryption.

- **Algorithm**: ChaCha20-Poly1305 (IETF) (via `chacha20poly1305` crate).
- **Master Key**: Uses a 256-bit (32-byte) key for encryption.
- **Key Derivation (PBKDF2)**: Supports deriving the master key from a human-readable passphrase using PBKDF2-HMAC-SHA256 with 100,000 iterations and a persistent installation-unique salt.
- **Nonce**: A random 12-byte nonce is generated for every memory encryption operation and stored alongside the ciphertext.
- **Key Handling**: The engine does not write the master key into CueMap data files. Encryption-at-rest is opt-in and the key must be provided at startup via environment variables (`CUEMAP_MASTER_KEY` or `CUEMAP_MASTER_PASSWORD`) or `security.master_key` in the configuration file. If you store the key in `server_config.toml`, protect that file like any other secret.

## Compression

To optimize storage efficiency, especially for large textual memories, CueMap employs transparent compression.

- **Algorithm**: Zstandard (Zstd), configured for a balanced compression level (3).
- **Strategy**: Content is compressed *before* encryption. This ensures maximum entropy reduction before the data is scrambled, often resulting in 40-60% storage savings for English text.
- **Performance**: Zstd provides extremely fast decompression speeds, ensuring that the "hot path" for reading memories remains sub-millisecond even with compression enabled.

## Performance

### Benchmark Results (v0.7.2)

Tests performed on **Real-World Data** (Wikipedia Articles), processing full natural language sentences with the complete NLP pipeline.

**Hardware:** MacBook Pro M-series, 64GB RAM, single node. The v0.7.2 release table below records completed lexical and hybrid runs at 10K, 100K, and 1M memories. Lexical runs isolate the sparse core with the semantic encoder disabled; hybrid runs include the bundled local encoder. P95 is the release headline percentile, while P99 remains available in the JSON diagnostics.

#### Benchmark Methodology

The NL benchmark script lives at `benchmarks/benchmark_nl.py`.

Benchmark setup:
- Uses the public [Wikipedia Plaintext (2023-07-01) Kaggle dataset](https://www.kaggle.com/datasets/jjinho/wikipedia-20230701) as the release corpus. The benchmark script downloads it automatically when `--wikipedia-path` is omitted, caches it under `~/.cache/cuemap/benchmarks/wikipedia-20230701`, and samples parquet files with a `text` column. Install the downloader first with `python -m pip install kagglehub`; configure Kaggle access if Kaggle prompts for authentication. To avoid the download or use another corpus, pass `--wikipedia-path /path/to/parquet-or-directory`.
- Deduplicates sampled snippets and consumes them without replacement, so 100K and 1M write runs do not reuse the same text.
- Writes use HTTP `POST /memories` with `minimal_response=true` and no explicit cues, forcing CueMap to run deterministic cue/facet extraction and indexing.
- Reads generate keyword-style natural-language queries from retained ingested snippets.
- Recall numbers use the script's lean recall mode: `semantic_mode=lexical`, `auto_reinforce=false`, salience disabled, alias expansion disabled, CueBridge artifacts disabled, `depth=1`, `expansion_depth=1`, and parent/order/evidence reconstruction disabled. This isolates the core sparse recall path from the bundled semantic encoder and reranker.
- Each requested size runs in its own run-scoped project, so a 1M pass is not layered on top of a previous 100K pass or stale state from an earlier invocation.
- `--trace-timing` records engine timing breakdowns but is not required for throughput measurements.

Example run with the checked-in release fixture:

```bash
CUEMAP_SEMANTIC_ENCODER_ENABLED=false cuemap start --disable-snapshots --disable-bg-jobs

python benchmarks/benchmark_nl.py \
  --sizes 10000,100000,1000000 \
  --project-id nl_test \
  --semantic-mode lexical \
  --wiki-reservoir-size 20000 \
  --query-sample-size 5000 \
  --payload-buffer-size 500
```

Restart the engine without `CUEMAP_SEMANTIC_ENCODER_ENABLED=false`, then run the
same command with `--semantic-mode hybrid` to produce the hybrid comparison. The
dataset is downloaded only once and reused from the local cache on subsequent
runs.

#### v0.7.2 latency comparison

The lexical release rerun now covers 10K, 100K, and 1M writes plus lean recall
queries. The compact comparison below records the 10K, 100K, and 1M hybrid
runs as well.

| Mode | Write avg | Write P50 | Write P95 | Write throughput | Read avg | Read P50 | Read P95 | Read throughput |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Lexical | 2.13 ms | 1.88 ms | 4.13 ms | 470 ops/s | 1.05 ms | 1.01 ms | 1.60 ms | 939 ops/s |
| Hybrid | 11.52 ms | 10.38 ms | 17.45 ms | 87 ops/s | 6.54 ms | 6.80 ms | 8.41 ms | 152 ops/s |

The script still stores p99 in the JSON result for diagnostics, but p95 is the
headline percentile used by the console output and release chart.

| Hybrid scale | Write avg | Write P50 | Write P95 | Write throughput | Read avg | Read P50 | Read P95 | Read throughput |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| **10,000** | 11.52 ms | 10.38 ms | 17.45 ms | 87 ops/s | 6.54 ms | 6.80 ms | 8.41 ms | 152 ops/s |
| **100,000** | 11.21 ms | 10.27 ms | 16.43 ms | 89 ops/s | 6.94 ms | 7.14 ms | 8.83 ms | 144 ops/s |
| **1,000,000** | 11.28 ms | 10.38 ms | 16.81 ms | 89 ops/s | 8.36 ms | 8.12 ms | 10.67 ms | 119 ops/s |

#### 1. Ingestion (Write) Performance — lexical
*Measures HTTP ingestion, deterministic cue/facet extraction, memory allocation, and indexing.*

| Dataset Scale | Avg Latency | P50 | P95 | Throughput |
|:---|:---|:---|:---|:---|
| **10,000** | 2.13 ms | 1.88 ms | 4.13 ms | 470 ops/s |
| **100,000** | 2.92 ms | 2.41 ms | 5.70 ms | 343 ops/s |
| **1,000,000** | 3.33 ms | 2.74 ms | 6.18 ms | 301 ops/s |

Write latency remains mostly flat with project size; the dominant cost is per-memory extraction/indexing rather than corpus scan time.

#### 2. Recall (Read) Performance — lexical
*Measures the time to parse a query, resolve deterministic cues, and score sparse candidate intersections.*

| Dataset Scale | Avg Latency | P50 | P95 | Throughput |
|:---|:---|:---|:---|---:|
| **10,000** | 1.05 ms | 1.01 ms | 1.60 ms | 939 ops/s |
| **100,000** | 1.86 ms | 1.71 ms | 3.16 ms | 535 ops/s |
| **1,000,000** | 2.63 ms | 2.06 ms | 3.72 ms | 378 ops/s |

**Key Metrics**:
- **Low-latency recall:** The lexical v0.7.2 1M run measured 2.63ms average with 3.72ms p95; hybrid measurements remain separate because they include bundled encoder work.
- **Numeric ID memory reduction:** 1M in-memory footprint dropped from about 5.25GB to about 1.93GB after the v0.7 numeric memory-ID refactor.
- **Controlled hot path:** the release benchmark disables the local semantic encoder, LLMs, network services, and disk scans; normal v0.7.2 hybrid recall can use the bundled local encoder for bounded reranking.

## Architecture

### Core Components

- **Axum**: Minimal overhead async web framework
- **DashMap + aHash**: Lock-free concurrent hash map with high-speed hashing
- **IndexSet**: O(1) move-to-front operations
- **Bincode**: Fast binary serialization for persistence
- **Zstd**: High-ratio, real-time compression for storage
- **ChaCha20-Poly1305**: Authenticated encryption at rest

### Optimizations

- **Zero-copy**: Efficient memory management with Arc
- **Pre-allocated collections**: Capacity hints eliminate reallocation
- **Unstable sorting**: 2-3x faster than stable sort
- **Iterative deepening**: Early termination on hot paths

## API

### Deterministic Cue Extraction

CueMap extracts cues synchronously from content and metadata using deterministic tokenization, normalization, structural facets, and explicit aliases. The lexical candidate path is deterministic; the default v0.7.2 hybrid path may then use the bundled local qint8 MiniLM-L3 encoder for bounded semantic and intent reranking. No LLM or external API is required.

Structural facets describe observable form—such as dates, times, quantities, ranges, identifiers, lists, URLs, email addresses, quoted spans, code/document markers, scripts, file names, and file paths—and do not claim to understand domain meaning. Query planning adds a small bounded set of English grammatical/query-shape heuristics for perspective, answer shape, collection, summary, ordering, and reference-time resolution. Locale-specific implementations can be added later under separate language packs rather than expanding the core with domain ontology rules.

```bash
# 1. Start CueMap
./target/release/cuemap start

# 2. Add memory in natural language
curl -X POST http://localhost:8080/memories \
  -H "X-Project-ID: default" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "The payments service is down due to a timeout."
  }'
# Deterministic extraction adds normalized lexical cues plus structural/facet cues.
```

### Local Semantic Retrieval

The engine also accepts caller-provided vectors at ingestion and query time. This is useful when an application already owns an embedding provider or wants to avoid automatic encoding:

```json
{
  "content": "The payments service is down due to a timeout.",
  "embedding": [0.12, -0.04, 0.88]
}
```

```json
{
  "query_text": "Why did payments fail?",
  "query_embedding": [0.10, -0.02, 0.91],
  "limit": 8
}
```

Enable or tune it with the `[semantic]` server configuration section. The default build selects the bundled qint8 `all-MiniLM-L3-v2` weights and enables local text embeddings. Set `encoder_enabled = false`, `profile = "off"`, or `CUEMAP_SEMANTIC_ENCODER_ENABLED=false` to disable automatic encoding. The encoder-free build remains available with `cargo build --no-default-features`. The `edge` profile selects the bundled q4 L3 model while lowering ANN fanout and the vector memory budget:

```toml
[semantic]
profile = "edge"       # off, edge, balanced, or quality
dimensions = 384        # both bundled MiniLM variants emit 384 dimensions
storage = "int8"        # auto, f32, f16, or int8
index = "auto"           # auto, exact, or ann
max_memory_mb = 32
model_id = "all-MiniLM-L3-v2"
model_version = "bundled-q4-minilm-l3"
```

Profiles choose device-oriented defaults for dimensions, compact storage, ANN fanout, and a conservative memory budget. Any explicitly set dimension/storage/index/budget overrides the profile. CueMap never downloads a model at runtime. The default build embeds the MiniLM ONNX graph and tokenizer; empty model paths use those bundled assets, while explicit paths provide local overrides:

```toml
[semantic]
profile = "quality"
dimensions = 384
storage = "int8"
index = "auto"
encoder_enabled = true
model_id = "all-MiniLM-L3-v2"
model_version = "bundled-qint8-minilm-l3"
model_path = ""           # empty = bundled model
tokenizer_path = ""       # empty = bundled tokenizer
max_tokens = 128          # bundled L3 default
encoder_threads = 0        # runtime/platform default
coreml_enabled = true
semantic_rerank_weight = 0.60  # hybrid only; 0 = lexical, 1 = semantic ordering
semantic_rerank_candidate_limit = 200  # lexical candidates scored semantically in hybrid
query_embedding_cache_capacity = 256  # repeated query embeddings retained in memory
intent_rerank_enabled = true
intent_rerank_weight = 0.65
intent_no_recall_penalty = 0.20
intent_rerank_max_delta = 64
```

Build the default path with `cargo build --release`; `--no-default-features` produces an encoder-free build. No model download or network call is made by CueMap while serving. Both the qint8 default and q4 edge MiniLM-L3 bundles emit 384-dimensional vectors with a 128-word-piece window. An explicit local model path can override either bundled asset. On Apple targets, `coreml_enabled = true` requests the CoreML execution provider from the CoreML-enabled ONNX Runtime; set it to false to use CPU execution. The index uses in-memory random-hyperplane ANN buckets and falls back to exact candidate discovery for small indexes. A query can use lexical, semantic, or hybrid signals via `semantic_mode`; hybrid performs lexical discovery, keeps the configured lexical rerank window, and applies local semantic and intent reranking before the final result limit, while semantic mode uses vector candidate discovery without lexical cues. Repeated query text embeddings use a bounded in-memory cache when enabled. `/ingest/content` accepts an `embeddings` array with one vector per produced chunk.

For launch scripts, the same encoder fields can be supplied through `CUEMAP_SEMANTIC_*` environment variables, including `CUEMAP_SEMANTIC_PROFILE`, `CUEMAP_SEMANTIC_ENCODER_ENABLED`, `CUEMAP_SEMANTIC_MODEL_PATH`, `CUEMAP_SEMANTIC_TOKENIZER_PATH`, `CUEMAP_SEMANTIC_DIMENSIONS`, `CUEMAP_SEMANTIC_STORAGE`, `CUEMAP_SEMANTIC_INDEX`, `CUEMAP_SEMANTIC_ENCODER_THREADS`, `CUEMAP_SEMANTIC_COREML_ENABLED`, `CUEMAP_SEMANTIC_RERANK_WEIGHT`, `CUEMAP_SEMANTIC_RERANK_CANDIDATE_LIMIT`, `CUEMAP_SEMANTIC_QUERY_CACHE_CAPACITY`, `CUEMAP_SEMANTIC_INTENT_RERANK_ENABLED`, `CUEMAP_SEMANTIC_INTENT_RERANK_WEIGHT`, `CUEMAP_SEMANTIC_INTENT_NO_RECALL_PENALTY`, and `CUEMAP_SEMANTIC_INTENT_RERANK_MAX_DELTA`. Set the encoder flag to `false` to disable automatic text embedding, or set the cache capacity to `0` to disable query embedding caching.

### Local Intent Classification

`POST /intent/classify` uses the configured local encoder to classify either query intent or durable-memory intent. A model-specific 8-class linear head maps the frozen MiniLM embedding directly to relative ranking scores; production classification contains no semantic word lists, phrase matching, or category overrides. Recall eligibility follows the learned category, with a syntax-only fallback for low-margin question-shaped or short incomplete queries; durable-memory eligibility always follows the learned category. The qint8 and q4 heads are trained separately because quantization changes the embedding space. A custom semantic model version has no intent classifier unless a compatible head is bundled, preventing weights trained for one encoder from silently being applied to another. The response includes the primary and top intents, confidence weight, `recall_eligible`, `recall_action`, `memory_eligible`, and model/taxonomy versions:

```bash
curl -X POST http://localhost:8080/intent/classify \
  -H "X-Project-ID: default" \
  -H "Content-Type: application/json" \
  -d '{"text":"What did we decide about retries?","target":"query"}'
```

Newly ingested memories are classified in the background when background jobs and the encoder are enabled. `/jobs/status` reports `intent_completed`, `intent_total`, `intent_failed`, coverage counts, and `intent_ready`; consumers should not treat ingestion as fully ready until the intent work is terminal and `intent_ready` is true.

## API Reference

### Add Memory

```bash
# Basic manual cues
curl -X POST http://localhost:8080/memories \
  -H "X-Project-ID: default" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "API Rate Limit Policy: 1000/min",
    "cues": ["api", "rate_limit", "policy"],
    "source_key": "doc:api-rate-limit-policy",
    "event_time": 1704067200.25
  }'

# Deterministic cues are extracted from content when `cues` is omitted or empty
curl -X POST http://localhost:8080/memories \
  -H "X-Project-ID: default" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "The payments service is down due to a timeout."
  }'
```

`source_key` makes ingestion idempotent: adding the same source again updates its existing memory. `event_time` is the original event time as Unix seconds and defaults to ingestion time. Importers may instead provide `metadata.source_timestamp` as Unix seconds or RFC 3339; an explicit `event_time` takes precedence. `embedding` accepts a precomputed vector for one memory; raw-content ingestion uses `embeddings` with exactly one vector per produced chunk.

### Recall Memories

#### Explicit Cues
```bash
curl -X POST http://localhost:8080/recall \
  -H "X-Project-ID: default" \
  -H "Content-Type: application/json" \
  -d '{
    "cues": ["api", "rate_limit"],
    "limit": 10
  }'
```

#### Natural Language Search (Symbol-First Intent Routing)
```bash
curl -X POST http://localhost:8080/recall \
  -H "X-Project-ID: default" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "where is process_data used?",
    "limit": 10,
    "expansion_depth": 2
  }'
```
Returns surgical code recall. The engine uses a deterministic **Symbol-First Router** with sparse BM25-style scoring to convert fuzzy queries into structural cues (e.g., `calls_function:process_data`). Set `expansion_depth` above 1 to include nearby source-order chunks when session/order metadata is available.

```json
{
  "explain": {
    "query_cues": ["payments"],
    "expanded_cues": [
      ["payments", 1.0],
      ["service:payments", 0.85]
    ]
  },
  "results": [
    {
      "content": "...",
      "score": 145.2,
      "explain": {
        "intersection_weighted": 1.85,
        "recency_component": 0.5
      }
    }
  ]
}
```

### Reinforce Memory

```bash
curl -X PATCH http://localhost:8080/memories/{id}/reinforce \
  -H "X-Project-ID: default" \
  -H "Content-Type: application/json" \
  -d '{
    "cues": ["important", "urgent"]
  }'
```
Reinforcement is used to boost the relevance of a memory. It is a way to tell CueMap that a memory is important and should be recalled more often. Standard `POST /recall` requests do not auto-reinforce by default; enable `auto_reinforce` explicitly or reinforce a memory manually through the API.

### Get Memory

```bash
curl -H "X-Project-ID: default" http://localhost:8080/memories/{id}
```

### Get Stats
```bash
curl -H "X-Project-ID: default" http://localhost:8080/stats
```

### Alias Management

Manage synonyms and semantic mappings deterministically.

#### Add Alias
```bash
curl -X POST http://localhost:8080/aliases \
  -H "X-Project-ID: default" \
  -H "Content-Type: application/json" \
  -d '{
    "from": "pay",
    "to": "service:payment",
    "weight": 0.9
  }'
```

#### Merge Aliases (Bulk)
```bash
curl -X POST http://localhost:8080/aliases/merge \
  -H "X-Project-ID: default" \
  -H "Content-Type: application/json" \
  -d '{
    "cues": ["bill", "invoice", "statement"],
    "to": "service:billing"
  }'
```

#### Get Aliases
```bash
# Reverse lookup: Find all aliases for "service:payment"
curl -H "X-Project-ID: default" "http://localhost:8080/aliases?cue=service:payment"
```

### Project Management

#### Create Project
```bash
curl -X POST http://localhost:8080/projects \
  -H "Content-Type: application/json" \
  -d '{"project_id": "my-project"}'
```

#### List Projects
```bash
curl http://localhost:8080/projects
```

#### Delete Project
```bash
curl -X DELETE "http://localhost:8080/projects/default"
```

### Lexicon Management

#### Inspect Cue
View incoming tokens and manually wired canonical cue mappings.
```bash
curl "http://localhost:8080/lexicon/inspect/service:payment"
```

#### Wire Token (Manual Connection)
Manually connect a token to a canonical cue.
```bash
curl -X POST http://localhost:8080/lexicon/wire \
  -H "Content-Type: application/json" \
  -d '{
    "token": "stripe",
    "canonical": "service:payment"
  }'
```

#### Unwire/Delete Entry
Remove a specific token from the lexicon.
```bash
curl -X DELETE "http://localhost:8080/lexicon/entry/cue:stripe"
```

### CueBridge Artifacts

CueBridge artifacts are offline-compiled lexical-gap packages. CueMap loads them into memory and uses them deterministically during recall:

- **AliasPack**: safe lexical variants applied during query cue resolution.
- **GapPack**: gated expansion cues applied only when exact recall is weak.

Install artifacts into the project artifact directory, then reload them:

```bash
curl -X POST http://localhost:8080/projects/my-project/artifacts
```

Inspect active artifacts:

```bash
curl http://localhost:8080/projects/my-project/artifacts
```

Recall can disable installed artifacts for baseline checks:

```bash
cuemap recall -p my_project --disable-cuebridge-artifacts "what foundation did we choose?"
```

### Cloud Backup Management

#### Upload Snapshot
```bash
curl -X POST http://localhost:8080/backup/upload \
  -H "Content-Type: application/json" \
  -d '{"project_id": "default"}'
```

#### Download Snapshot
```bash
curl -X POST http://localhost:8080/backup/download \
  -H "Content-Type: application/json" \
  -d '{"project_id": "default"}'
```

#### List Backups
```bash
curl http://localhost:8080/backup/list
```

### Monitoring

#### Prometheus Metrics
Exposes internal system metrics for scraping (Prometheus format).

```bash
curl http://localhost:8080/metrics
# Output:
# cuemap_ingestion_rate 120.0
# cuemap_recall_latency_p99 0.8
# cuemap_memory_usage_bytes 1024
# ...
```

### Ingestion

#### Ingest URL
Extract content from a web page and ingest it.
```bash
curl -X POST http://localhost:8080/ingest/url \
  -H "X-Project-ID: default" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com"
  }'
```

#### Ingest Raw Content
Ingest text directly, simulating a file.
```bash
curl -X POST http://localhost:8080/ingest/content \
  -H "X-Project-ID: default" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "The quick brown fox jumps over the lazy dog.",
    "filename": "fox.txt"
  }'
```

#### Ingest File (Multipart)
Upload a file for processing by the Agent (supports Text, PDF, JSON, etc. if Agent is configured).
```bash
curl -X POST http://localhost:8080/ingest/file \
  -H "X-Project-ID: default" \
  --form "file=@/path/to/document.pdf"
```

#### Grounded Recall (Budgeted)

```bash
curl -X POST http://localhost:8080/recall/grounded \
  -H "X-Project-ID: default" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "Why is the server down?",
    "token_budget": 500,
    "limit": 10
  }'
```

Grounded recall deterministically fills a token budget with the highest-scoring memories and returns a context block designed to be passed to an LLM alongside a structured proof.

Grounded recall enables `auto_reinforce` by default; set `"auto_reinforce": false` for read-only evaluation or benchmark runs.

**Response** (example with Ed25519 signing configured):
```json
{
  "verified_context": "[VERIFIED CONTEXT] (1) Fact... Rules:...",
  "proof": {
    "trace_id": "966579b1-...",
    "selected": [...],
    "excluded_top": [...]
  },
  "signature_alg": "ed25519",
  "signature": "9b2d...",
  "public_key": "ed25519:4f8c...",
  "engine_latency_ms": 0.83
}
```

### Signed Context (Immutable RAG)

CueMap can sign grounded recall context so clients can verify that the `verified_context` block was produced by the configured CueMap server and was not modified in transit before it reaches an LLM.

Preferred setup uses Ed25519 asymmetric signatures. Generate one 32-byte private seed, store it securely, and reuse it across restarts:

```bash
openssl rand -hex 32 > ~/.cuemap/signing_ed25519_seed.hex
CUEMAP_SIGNING_PRIVATE_KEY="$(cat ~/.cuemap/signing_ed25519_seed.hex)" cuemap start
```

Or configure it in `~/.cuemap/server_config.toml`:

```toml
[security]
signing_private_key = "ed25519:<32-byte-hex-seed>"
```

Grounded recall responses include the signature algorithm and public key:

```json
{
  "verified_context": "...",
  "signature_alg": "ed25519",
  "signature": "9b2d...",
  "public_key": "ed25519:4f8c..."
}
```

Clients verify `signature` over the exact UTF-8 bytes of `verified_context` using the pinned Ed25519 public key. The signature is a lowercase hex-encoded 64-byte Ed25519 signature. The public key is `ed25519:` plus the lowercase hex-encoded raw 32-byte Ed25519 public key.

Treat the response `public_key` as discovery metadata; production clients should pin the expected public key from deployment config rather than trusting a key delivered by the same response they are verifying.

For compatibility, `CUEMAP_SECRET_KEY` still enables legacy `hmac-sha256` signatures. HMAC verification requires sharing the same secret with verifiers, so Ed25519 is recommended for client-side or third-party verification.

## System Architecture

### 1. High-Level Overview

```mermaid
graph TB
    subgraph "Clients"
        SDK[Python/TS SDKs]
        CURL[HTTP Clients]
    end
    
    subgraph "API Layer"
        AXUM[Axum HTTP Server]
        AUTH[Auth Middleware]
    end
    
    subgraph "Multi-Tenant Core"
        MT[MultiTenantEngine]
        MAIN[CueMap Engine<br/>DashMap + aHash]
        LEX[Lexicon Engine<br/>Token → Cue]
        ALIAS[Alias Engine<br/>Synonyms]
    end
    
    subgraph "Background Processing"
        QUEUE[Job Queue<br/>Reinforcement + Agent Jobs]
        SESSION[Session Manager<br/>Ingest Progress]
    end
    
    subgraph "Intelligence"
        NL[NL Tokenizer<br/>Lemmatization + RAKE]
        STRUCT[Structural Facets<br/>Evidence + Metadata]
    end
    
    subgraph "Persistence"
        PERSIST[Snapshots<br/>Zstd + ChaCha20]
    end
    
    SDK --> AXUM
    CURL --> AXUM
    AXUM --> AUTH --> MT
    
    MT --> MAIN
    MT --> LEX
    MT --> ALIAS
    
    AXUM --> QUEUE
    AXUM --> SESSION
    
    QUEUE --> LEX
    
    MAIN <-.-> PERSIST
    LEX <-.-> PERSIST
    
    style MAIN fill:#4CAF50
    style LEX fill:#2196F3
    style ALIAS fill:#FF9800
    style QUEUE fill:#9C27B0
```

### 2. Write Flow (POST /memories)

```mermaid
sequenceDiagram
    participant C as Client
    participant API as API Handler
    participant NL as NL Tokenizer
    participant Norm as Normalizer
    participant Tax as Taxonomy
    participant Main as CueMap Engine
    
    C->>API: POST /memories<br/>{content, cues[]}
    
    alt cues[] is empty
        API->>NL: tokenize_to_cues(content)
        NL-->>API: ["payment", "timeout", ...]
    end
    
    API->>Norm: normalize_cue(each)
    Norm-->>API: normalized cues
    
    API->>Tax: validate_cues(cues)
    Tax-->>API: {accepted[], rejected[]}
    
    API->>Main: add_memory(content, accepted)
    Main-->>API: memory_id
    
    API-->>C: 200 {id, cues, latency_ms}
    Note over C,API: ✅ Synchronous ~2ms
    
    Note over API,Main: Cue extraction and indexing happen synchronously
```

### 3. Read Flow (POST /recall)

```mermaid
sequenceDiagram
    participant C as Client
    participant API as API Handler
    participant Lex as Lexicon
    participant Alias as Alias Engine
    participant Art as CueBridge Artifacts
    participant Main as CueMap Engine
    participant Q as Job Queue
    
    C->>API: POST /recall<br/>{query_text?, cues[], limit}
    
    alt query_text provided
        API->>Lex: resolve_cues_from_text(query)
        Lex-->>API: resolved_cues[]
    end
    
    API->>API: Merge & Normalize cues
    
    opt explicit aliases enabled
        API->>Alias: apply_aliases(cues)
        Alias-->>API: weighted_cues[(cue, weight)]
    end

    opt exact recall is weak and artifacts are enabled
        API->>Art: lookup GapPack(query_signature)
        Art-->>API: capped expansion cues
    end
    
    API->>Main: recall_weighted(cues, limit, options)
    Main->>Main: Salience Bias
    Main->>Main: Score & Rank
    
    Main-->>API: RecallResult[]
    
    opt auto_reinforce = true
        API->>Q: Enqueue ReinforceMemories
        API->>Q: Enqueue ReinforceLexicon
    end
    
    API-->>C: {results, explain?, latency_ms}
```

### 4. Background Job Pipeline

```mermaid
graph TB
    subgraph "Job Sources"
        INGEST[Ingest API]
        RECALL[POST /recall]
        AGENT[Self-Learning Agent]
        TIMER[60s Heatmap Tick]
    end
    
    subgraph "Job Types"
        J4[ReinforceMemories]
        J5[ReinforceLexicon]
        J7[ExtractAndIngest]
        J8[VerifyFile]
        J10[DeleteMemory]
        J9[UpdateMarketHeatmap]
    end
    
    subgraph "Processing"
        SESSION[Session Manager<br/>Tracks write completion]
        QUEUE[MPSC Queue<br/>Async Worker]
    end
    
    subgraph "Side Effects"
        E1[Memories Reinforced]
        E2[Lexicon Reinforced]
        E4[Content Extracted]
        E5[Stale File Memories Deleted]
        E6[Market Heatmap Updated]
    end
    
    RECALL --> J4 & J5
    INGEST --> J7
    AGENT --> J7 & J8 & J10
    TIMER --> J9
    
    J7 --> SESSION
    J4 & J5 --> QUEUE
    J7 & J8 & J10 --> QUEUE
    J9 --> QUEUE
    
    QUEUE --> E1 & E2 & E4 & E5 & E6
    
    style QUEUE fill:#9C27B0
    style SESSION fill:#673AB7
    style E1 fill:#2196F3
    style E2 fill:#4CAF50
    style E5 fill:#F44336
```

## Advanced Capabilities

### 1. Self-Learning Ingestion Agent

The agent transforms your local filesystem into a deterministic structural knowledge base with zero manual effort.

*   **Universal Format Support**: Deeply integrates with dozens of formats:
    *   **Languages**: Rust, Python, TypeScript, Go, Java, PHP, HTML, CSS (via Tree-sitter).
    *   **Documents**: PDF (text extraction), Word (DOCX), Excel (XLSX).
    *   **Data**: CSV (row-aware), JSON (key-aware), YAML, XML.
*   **Tree-sitter Powered Chunking**: Smartly splits code into functions, classes, and modules while preserving context.
*   **Deterministic Knowledge Extraction**: Uses tree-sitter structure, document parsers, metadata facets, and token normalization; no runtime model call is required.
*   **Idempotent Updates**: Uses content-aware hashing (`file:<path>:<hash>`) to prevent memory duplication and ensure stale memories are pruned.
*   **Background Verification Loop**: Continuously verifies that memories in the engine still exist on disk, pruning stale references automatically.

### 2. Deterministic Natural Language Engine

CueMap bridges unstructured text to sparse deterministic recall without vector search, runtime models, or background semantic expansion by default. Optional vector retrieval can add externally computed semantic candidates without changing the structural extraction path.

#### How It Works

At add-time, CueMap extracts cues synchronously from real structure:

- normalized lexical cues
- surface entity, quote, model-like, and structural evidence cues
- evidence facets such as numbers, money, dates, durations, and lists
- source facets from metadata such as role, channel, session, and order

At query-time, CueMap uses the same deterministic normalization path, then applies only bounded in-memory expansions:

- explicit aliases when enabled
- installed CueBridge AliasPack entries during query cue resolution
- installed CueBridge GapPack entries only when exact recall is weak
- optional ordered/evidence reconstruction passes when explicitly requested

#### Semantic Boundary

CueMap Core does not try to infer broad semantic relationships from local co-occurrence or ontology rules. That keeps the default recall fast, deterministic, and inspectable. Semantic gap closure can come from externally precomputed vectors or explicit artifacts:

- **Manual Lexicon Wiring**: explicit token-to-canonical cue connections for project owners.
- **CueBridge Artifacts**: offline-compiled GapPack/AliasPack files generated by CueBridge Local or Cloud and loaded into CueMap.

This split is intentional: CueMap Core stays lean and latency-stable, while CueBridge can use heavier local or cloud models offline to generate static lexical-gap artifacts.

### 3. Advanced Contextual Recall

CueMap keeps advanced recall behavior deterministic and inspectable:

#### Source-Order Context and Episodes
Long-form ingests, chat logs, files, and agent chunks preserve parent/session/order metadata. Recall can use `expansion_depth` to include nearby source-order chunks, and add-time temporal episode cues can be disabled on memory writes with `disable_temporal_chunking: true`.

#### Adaptive Salience Bias
Not all memories are created equal. The engine calculates a **Salience Multiplier** based on cue density, reinforcement frequency, and rare cue combinations. High-signal memories rank above routine events when other structural evidence is comparable. Can be disabled per-recall via `disable_salience_bias: true`.

#### Match Integrity
Every recall result now includes a **Match Integrity** score. This internal diagnostic combines intersection strength, reinforcement history, and context agreement to tell you how structurally reliable a specific recall result is.

#### Bounded Reconstruction
For long-form chat logs, tickets, transcripts, and benchmark records, recall can optionally run bounded reconstruction passes:

- `parent_fusion`: stitch related chunks that share source parent metadata.
- `ordered_reconstruction`: retrieve ordered evidence from a small number of matching sessions.
- `evidence_coverage`: diversify results across multiple evidence cues for summary-style queries.

These modes are off by default and are designed for diagnostics or workloads that explicitly trade a bounded second pass for higher evidence coverage.

## License

BSL-1.1 (Business Source License 1.1) converting to Apache 2.0 after 4 years.
See `LICENSE` for details.

This allows full use for development, testing, and self-hosting, while preventing the software from being offered as a competing managed Database Service.

For commercial licensing (closed-source SaaS or offering as a service), contact: hello@cuemap.dev
