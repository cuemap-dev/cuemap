# CueMap Rust Engine

**High-performance temporal-associative memory store** that mimics the brain's recall mechanism.

## Overview

CueMap implements a **Continuous Gradient Algorithm** inspired by biological memory:

1.  **Intersection (Context Filter)**: Triangulates relevant memories by overlapping cues (e.g., `service:payments` + `error:500`).
2.  **Recency (Signal Attenuation)**: Applies a continuous decay curve, prioritizing fresh data.
3.  **Reinforcement (Hebbian Learning)**: Frequently accessed memories gain signal strength, staying "front of mind" even as they age.
4.  **Self-Regulating Dynamics**: Uses a $\sigma = \sqrt{N}$ scaling factor to automatically balance Recency vs. Reinforcement as datasets grow from 10k to 10M+.

Built with Rust for maximum performance and reliability.

## Quick Start

### Build & Run

```bash
# Development
cargo run

# Production (optimized)
cargo build --release
./target/release/cuemap-rust --port 8080
```

### Docker

```bash
docker build -f Dockerfile.production -t cuemap/engine .
docker run -p 8080:8080 cuemap/engine
```

### CLI Options

```bash
./target/release/cuemap-rust --help

Options:
  -p, --port <PORT>                    Server port [default: 8080]
  -d, --data-dir <DATA_DIR>            Data directory [default: ./data]
  -s, --snapshot-interval <SECONDS>    Snapshot interval [default: 60]
  -m, --multi-tenant                   Enable multi-tenancy
```

## Multi-Tenant Mode with Persistence

Multi-tenant mode provides complete project isolation with automatic persistence:

### Features

- **Project Isolation**: Each project has its own memory space
- **Auto-Save on Shutdown**: All projects saved when server stops (Ctrl+C)
- **Auto-Load on Startup**: All snapshots restored when server starts
- **Zero Configuration**: Works out of the box

### Usage

```bash
# Start in multi-tenant mode
./target/release/cuemap-rust --port 8080 --multi-tenant
```

### Example

```bash
# Add memory to project
curl -X POST http://localhost:8080/memories \
  -H "X-Project-ID: my-project" \
  -H "Content-Type: application/json" \
  -d '{"content": "Important data", "cues": ["test"]}'

# Stop server (Ctrl+C) - auto-saves all projects
# Restart server - auto-loads all projects

# Data persists across restarts!
```

### Snapshot Management

Snapshots are automatically managed:
- **Created**: On graceful shutdown (SIGINT/Ctrl+C)
- **Loaded**: On server startup
- **Location**: `./data/snapshots/` (configurable via `--data-dir`)
- **Format**: Bincode binary (same as single-tenant mode)
- **Files**: `{project-id}.bin` (one file per project)

Test persistence:
```bash
./test_persistence.sh
```

## Authentication

Secure your CueMap instance with API key authentication.

### Enable Authentication

Set an API key via environment variable:

```bash
# Single API key
CUEMAP_API_KEY=your-secret-key ./target/release/cuemap-rust --port 8080

# Multiple API keys (comma-separated)
CUEMAP_API_KEYS=key1,key2,key3 ./target/release/cuemap-rust --port 8080
```

### Using Authentication

Include the API key in the `X-API-Key` header:

```bash
# Without auth (fails if enabled)
curl http://localhost:8080/stats
# Response: Missing X-API-Key header

# With correct key
curl -H "X-API-Key: your-secret-key" http://localhost:8080/stats
# Response: {"total_memories": 1000, ...}

# With wrong key
curl -H "X-API-Key: wrong-key" http://localhost:8080/stats
# Response: Invalid API key
```

### SDK Usage

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
docker run -p 8080:8080 \
  -e CUEMAP_API_KEY=your-secret-key \
  cuemap/engine
```

### Security Notes

- Authentication is **disabled by default** (no keys = no auth required)
- Keys are loaded from environment variables only
- Use strong, randomly generated keys in production
- Rotate keys regularly
- Use HTTPS in production to protect keys in transit

## Performance

### Benchmark Results

Tested on realistic workloads with Zipfian distribution (80% of operations hit 20% of cues):

#### Write Performance

| Dataset | Avg Latency | P99 Latency | Throughput |
|---------|-------------|-------------|------------|
| 100K    | 0.19ms      | 0.30ms      | 3,067 ops/s |
| 1M      | 0.20ms      | 0.33ms      | 2,926 ops/s |
| 10M     | 0.25ms      | 0.32ms      | 2,567 ops/s |

#### Read Performance

| Dataset | Avg Latency | P50 Latency | P99 Latency | Throughput |
|---------|-------------|-------------|-------------|------------|
| 100K    | 0.23ms      | 0.22ms      | 0.35ms      | 2,782 ops/s |
| 1M      | 0.23ms      | 0.22ms      | 0.37ms      | 2,763 ops/s |
| 10M     | 0.23ms      | 0.28ms      | 0.38ms      | 2,800 ops/s |

**Key Metrics**:
- ✅ **Sub-millisecond P99 latency** at 1M scale
- ✅ **0.38ms P99 latency** at 10M scale (production-tested)
- ✅ **Consistent performance** across dataset sizes
- ✅ **2,700+ ops/sec** sustained throughput (10M)

**Memory Efficiency**:
- ✅ **~500 bytes per memory** (content + cues + indexes)
- ✅ **5 GB RAM for 10M memories** (production-tested)
- ✅ **Linear scaling** with dataset size

### Correctness Tests

Validated on 120+ test scenarios:
- ✅ **Recency**: 30/30 (100%) - Recent memories prioritized
- ✅ **Intersection**: 30/30 (100%) - Multi-cue matching works
- ✅ **Reinforcement**: 20/20 (100%) - Move-to-front operation
- ✅ **Multi-Cue**: 20/20 (100%) - Complex queries
- ✅ **Noise Filtering**: 20/20 (100%) - Irrelevant memories filtered

### Concurrent Performance

Stress tested with 400+ parallel operations:
- ✅ **100% success rate** under concurrent load
- ✅ **100% recall accuracy** with parallel reads/writes
- ✅ **Lock-free operations** with DashMap

## Architecture

### Core Components

- **Axum**: Minimal overhead async web framework
- **DashMap**: Lock-free concurrent hash map (32 shards)
- **IndexSet**: O(1) move-to-front operations
- **Bincode**: Fast binary serialization for persistence

### Optimizations

- **Zero-copy**: Efficient memory management with Arc
- **Pre-allocated collections**: Capacity hints eliminate reallocation
- **Unstable sorting**: 2-3x faster than stable sort
- **Iterative deepening**: Early termination on hot paths

## API

### Add Memory

```bash
curl -X POST http://localhost:8080/memories \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Server password is abc123",
    "cues": ["server", "password", "credentials"]
  }'
```

### Recall Memories

```bash
curl -X POST http://localhost:8080/recall \
  -H "Content-Type: application/json" \
  -d '{
    "cues": ["server", "password"],
    "limit": 10,
    "auto_reinforce": false
  }'
# Note: auto_reinforce=true will increment the reinforcement count for returned memories
```

### Reinforce Memory

```bash
curl -X PATCH http://localhost:8080/memories/{id}/reinforce \
  -H "Content-Type: application/json" \
  -d '{
    "cues": ["important", "urgent"]
  }'
```

### Get Memory

```bash
curl http://localhost:8080/memories/{id}
```

### Get Stats

```bash
curl http://localhost:8080/stats
```

## Production Features

### Persistence

- **Bincode snapshots**: 10x faster than JSON
- **Background saves**: Every 60s (configurable)
- **Atomic writes**: Temp file + rename pattern
- **Graceful shutdown**: SIGINT/SIGTERM handlers

### Authentication

```bash
export CUEMAP_API_KEY="your-secret-key"
./target/release/cuemap-rust
```

### Multi-Tenancy

```bash
./target/release/cuemap-rust --multi-tenant

# Use project-specific endpoints
curl -X POST http://localhost:8080/v1/my-project/memories ...
```

## Monitoring

### Health Check

```bash
curl http://localhost:8080/
```

### Statistics

```bash
curl http://localhost:8080/stats | jq .
```

Returns:
```json
{
  "total_memories": 1000000,
  "total_cues": 1418,
  "cues": ["user", "system", "data", ...]
}
```

## License

AGPLv3 - See LICENSE for details

For commercial licensing (closed-source SaaS), contact: kaandemirel@yahoo.com
