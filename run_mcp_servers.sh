#!/bin/bash

# Run both MCP servers for hackathon compliance
# FlowMap MCP: http://localhost:3001 (code intelligence)
# Slack MCP: http://localhost:13080 (Slack data)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Source .env for tokens
source "$SCRIPT_DIR/.env" 2>/dev/null || true

echo -e "${GREEN}Starting MCP Servers...${NC}"

# Start Slack MCP Server on port 13080
echo -e "${BLUE}Starting Slack MCP Server on port 13080...${NC}"
SLACK_MCP_XOXB_TOKEN=$SLACK_BOT_TOKEN slack-mcp-server -t sse &
SLACK_PID=$!

# Wait for Slack MCP to start
sleep 3

# Start FlowMap MCP Server on port 3001
echo -e "${BLUE}Starting FlowMap MCP Server on port 3001...${NC}"
cd "$SCRIPT_DIR"
python -m packages.agent.src.agent.mcp_transport --transport sse --port 3001 &
FLOWMAP_PID=$!

sleep 2

echo -e "${GREEN}Both MCP servers started!${NC}"
echo -e "  Slack MCP: http://localhost:13080/sse"
echo -e "  FlowMap MCP: http://localhost:3001/sse"
echo ""
echo "Press Ctrl+C to stop"

# Trap cleanup
trap "kill $SLACK_PID $FLOWMAP_PID 2>/dev/null; exit" INT TERM

# Wait
wait
