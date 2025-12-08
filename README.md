# CueMap Rust Engine

High-performance Rust implementation of CueMap using Axum and DashMap.

## Build & Run

```bash
# Development
cargo run

# Production (optimized)
cargo build --release
./target/release/cuemap-rust
```

The server will listen on `http://localhost:8080`

## Docker

```bash
docker build -t cuemap-rust .
docker run -p 8080:8080 cuemap-rust
```

## Performance Features

- **Axum**: Minimal overhead async web framework
- **DashMap**: Lock-free concurrent hash map for thread-safe operations
- **Zero-copy**: Efficient memory management with Arc and references
- **Optimized builds**: LTO and single codegen unit for maximum performance

## API Compatibility

The Rust engine implements the exact same API as the Python version:

- `POST /memories` - Add memory
- `POST /recall` - Recall memories
- `PATCH /memories/{id}/reinforce` - Reinforce memory
- `GET /memories/{id}` - Get memory
- `GET /stats` - Get statistics
