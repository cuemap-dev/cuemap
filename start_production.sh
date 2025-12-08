#!/bin/bash

# Start CueMap Rust Engine in Production Mode

echo "🚀 Starting CueMap Rust Engine - Production Mode"
echo "================================================"

# Set API key if not already set
if [ -z "$CUEMAP_API_KEY" ]; then
    export CUEMAP_API_KEY="cuemap-dev-key-$(date +%s)"
    echo "📝 Generated API key: $CUEMAP_API_KEY"
fi

# Configuration
PORT="${PORT:-8080}"
DATA_DIR="${DATA_DIR:-./data}"
SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-60}"
MULTI_TENANT="${MULTI_TENANT:-false}"

echo ""
echo "Configuration:"
echo "  Port: $PORT"
echo "  Data Directory: $DATA_DIR"
echo "  Snapshot Interval: ${SNAPSHOT_INTERVAL}s"
echo "  Multi-Tenant: $MULTI_TENANT"
echo "  API Key: $CUEMAP_API_KEY"
echo ""

# Build if needed
if [ ! -f "./target/release/cuemap-rust" ]; then
    echo "Building release binary..."
    cargo build --release
fi

# Start server
echo "Starting server..."
echo ""

if [ "$MULTI_TENANT" = "true" ]; then
    ./target/release/cuemap-rust \
        --port $PORT \
        --data-dir $DATA_DIR \
        --snapshot-interval $SNAPSHOT_INTERVAL \
        --multi-tenant
else
    ./target/release/cuemap-rust \
        --port $PORT \
        --data-dir $DATA_DIR \
        --snapshot-interval $SNAPSHOT_INTERVAL
fi
