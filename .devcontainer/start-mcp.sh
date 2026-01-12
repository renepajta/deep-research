#!/bin/bash
# Auto-start MCP server if .env file exists

ENV_FILE="/workspaces/deep-research/.env"

if [ -f "$ENV_FILE" ]; then
    echo "📦 Starting Deep Research MCP server..."
    cd /workspaces/deep-research
    nohup python -m deep_research.server --http > /tmp/mcp-server.log 2>&1 &
    sleep 2
    if curl -s http://localhost:8000/mcp > /dev/null 2>&1; then
        echo "✅ MCP server running at http://localhost:8000/mcp"
    else
        echo "⚠️  MCP server may still be starting. Check: tail -f /tmp/mcp-server.log"
    fi
else
    echo "⏭️  Skipping MCP server (no .env file found)"
    echo "   Create .env from .env.example and restart the container"
fi
