#!/bin/bash
# Start both RAG application UIs
# Syncs rag/ to agents/doc_agent/ for ADK Web

set -euo pipefail

echo "Starting RAG Applications..."

# Free ports if already in use
echo "Ensuring ports 8501 and 8001 are free..."
lsof -ti:8501,8001 | xargs kill -9 2>/dev/null || true

# Sync rag folder to agents for ADK web
echo "Syncing agent code..."
rm -rf agents
mkdir -p agents/doc_agent
cp -r rag/* agents/doc_agent/
rm -rf agents/doc_agent/.adk agents/.adk

# Start Streamlit in background with logging
echo "Starting Streamlit Web App..."
uv run streamlit run app.py > streamlit.log 2>&1 &
STREAMLIT_PID=$!

# Start ADK Web in background with logging
echo "Starting ADK Web UI..."
uv run adk web agents --port 8001 > adk_web.log 2>&1 &
ADK_PID=$!

# Wait for services to be ready
sleep 4

echo ""
echo "✓ Applications started successfully!"
echo "  Streamlit Web App: http://localhost:8501"
echo "  ADK Web UI:        http://localhost:8001 (select 'doc_agent')"
echo ""
echo "Logs: streamlit.log, adk_web.log"
echo "To stop both: lsof -ti:8501,8001 | xargs kill"
echo "Note: Edit code in rag/ folder. Run this script again to sync changes to ADK web."
