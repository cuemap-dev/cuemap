#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting CueMap Engine CI Check${NC}"

# 1. Check Prerequisites (Rust & Ollama)
echo "Checking prerequisites..."
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}Rust/Cargo not found.${NC}"
    exit 1
fi

if ! command -v ollama &> /dev/null; then
    echo -e "${RED}Ollama not found. Please install Ollama.${NC}"
    exit 1
fi

# 2. Ensure Ollama is running
echo "Ensuring Ollama is running..."
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama in background..."
    ollama serve & > /dev/null 2>&1
    sleep 5
fi

# Check model availability
if ! ollama list | grep -q "mistral"; then
    echo "Pulling mistral model..."
    ollama pull mistral
fi

# 3. Code Generation & Build
echo "Building project..."
cargo build --verbose

# 4. Unit Tests
echo -e "${GREEN}Running Unit Tests...${NC}"
cargo test --lib

# 5. Integration Tests (Fixtures)
echo -e "${GREEN}Running Fixture Tests...${NC}"
# Ensure fixture exists
if [ ! -f "data/snapshots/rust_engine.bin" ]; then
    echo "Generating fixture..."
    cargo run --bin create_fixture src data/snapshots/rust_engine.bin
fi
cargo test --test fixture_test

# 6. Real Evals (Deterministic)
echo -e "${GREEN}Running Real Evals...${NC}"
cargo test --test real_evals

# 7. Live System Tests (Sequential)
echo -e "${GREEN}Running Live System Tests (Sequential)...${NC}"
cargo test --test live_system_tests -- --test-threads=1 --nocapture

echo -e "${GREEN}All Checks Passed!${NC}"
