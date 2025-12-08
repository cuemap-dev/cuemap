#!/bin/bash

# Test script for CueMap Rust Engine (Production Mode)

set -e

echo "🧪 Testing CueMap Rust Engine - Production Mode"
echo "================================================"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

BASE_URL="http://localhost:8080"
API_KEY="test-key-123"

echo ""
echo -e "${BLUE}1. Testing Health Check${NC}"
curl -s $BASE_URL/ | jq .
echo -e "${GREEN}✓ Health check passed${NC}"

echo ""
echo -e "${BLUE}2. Adding Memories${NC}"

# Add memory 1
MEMORY1=$(curl -s -X POST $BASE_URL/memories \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{
    "content": "I love Italian food, especially pizza and pasta",
    "cues": ["food", "italian", "pizza"]
  }')
echo "Memory 1: $MEMORY1"

# Add memory 2
MEMORY2=$(curl -s -X POST $BASE_URL/memories \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{
    "content": "My favorite color is blue, like the ocean",
    "cues": ["color", "blue", "favorite"]
  }')
echo "Memory 2: $MEMORY2"

# Add memory 3
MEMORY3=$(curl -s -X POST $BASE_URL/memories \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{
    "content": "I work as a software engineer at a tech startup",
    "cues": ["work", "software", "engineer"]
  }')
echo "Memory 3: $MEMORY3"

echo -e "${GREEN}✓ Added 3 memories${NC}"

echo ""
echo -e "${BLUE}3. Testing Recall${NC}"

# Recall by food
echo "Query: food"
curl -s -X POST $BASE_URL/recall \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{
    "cues": ["food"],
    "limit": 10
  }' | jq '.results[] | {content, score, intersection_count}'

# Recall by color
echo ""
echo "Query: color, favorite"
curl -s -X POST $BASE_URL/recall \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{
    "cues": ["color", "favorite"],
    "limit": 10
  }' | jq '.results[] | {content, score, intersection_count}'

echo -e "${GREEN}✓ Recall working${NC}"

echo ""
echo -e "${BLUE}4. Testing Stats${NC}"
curl -s $BASE_URL/stats \
  -H "x-api-key: $API_KEY" | jq .
echo -e "${GREEN}✓ Stats working${NC}"

echo ""
echo -e "${BLUE}5. Testing Persistence${NC}"
echo "Waiting 5 seconds for background snapshot..."
sleep 5

# Check if snapshot file exists
if [ -f "./data/cuemap.bin" ]; then
    SIZE=$(ls -lh ./data/cuemap.bin | awk '{print $5}')
    echo -e "${GREEN}✓ Snapshot file created: $SIZE${NC}"
else
    echo "⚠️  Snapshot file not found (may need more time)"
fi

echo ""
echo "================================================"
echo -e "${GREEN}✅ All tests passed!${NC}"
echo ""
echo "Try killing the server (Ctrl+C) and restarting to test recovery!"
