#!/bin/bash
# test-api.sh — End-to-end test for ApiLLM backend
#
# Set environment variables before running:
#   export QMD_API_BASE_URL="http://your-api-endpoint/openai/v1"
#   export QMD_API_KEY="your-api-key"
#   export QMD_EMBED_MODEL="text-embedding-3-small"      # optional
#   export QMD_GENERATE_MODEL="gpt-4.1-mini"             # optional
#
# Usage: ./test-api.sh

set -e

if [ -z "$QMD_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
  echo "Error: QMD_API_KEY or OPENAI_API_KEY must be set"
  echo ""
  echo "  export QMD_API_BASE_URL=\"http://your-endpoint/openai/v1\""
  echo "  export QMD_API_KEY=\"your-key\""
  echo "  ./test-api.sh"
  exit 1
fi

cd "$(dirname "$0")"
DB="/tmp/qmd-api-test.sqlite"
rm -f "$DB"

echo "==============================================="
echo "  QMD ApiLLM End-to-End Test"
echo "==============================================="
echo ""
echo "API:   ${QMD_API_BASE_URL:-https://api.openai.com/v1}"
echo "Embed: ${QMD_EMBED_MODEL:-text-embedding-3-small}"
echo "Chat:  ${QMD_GENERATE_MODEL:-gpt-4.1-mini}"
echo "DB:    $DB"
echo ""

# Step 1: Create collection and index files
echo "-- Step 1: Create collection (index test/eval-docs/) --"
INDEX_PATH="$DB" npx tsx src/cli/qmd.ts collection add ./test/eval-docs --name test-docs
echo ""

# Step 2: Generate embeddings via API
echo "-- Step 2: Embed --"
INDEX_PATH="$DB" npx tsx src/cli/qmd.ts embed
echo ""

# Step 3: BM25 keyword search (no model needed)
echo "-- Step 3: BM25 search 'API design' --"
INDEX_PATH="$DB" npx tsx src/cli/qmd.ts search "API design"
echo ""

# Step 4: Vector similarity search (requires embeddings)
echo "-- Step 4: Vector search 'distributed systems consistency' --"
INDEX_PATH="$DB" npx tsx src/cli/qmd.ts vsearch "distributed systems consistency"
echo ""

# Step 5: Full query (expand + search + vector + rerank)
echo "-- Step 5: Query 'how to handle API versioning' --"
INDEX_PATH="$DB" npx tsx src/cli/qmd.ts query "how to handle API versioning"
echo ""

echo "==============================================="
echo "  All steps completed successfully!"
echo "==============================================="
