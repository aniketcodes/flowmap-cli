"""FlowMap Agent - Unified server with Slack + MCP endpoints."""

import os
import sys
import logging
import threading
import asyncio
import httpx
from pathlib import Path

# Add parent path for flowmap imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flask import Flask, request, jsonify
from slack_bolt.adapter.flask import SlackRequestHandler

from agent.bot import SlackBot
from agent.agent import Agent
from agent.server import FlowMapMCPServer
from agent.llm import LLMClient
from agent.security import verify_slack_signature, check_rate_limit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Lazy-initialized components
_bot = None
_handler = None
_mcp = None
_llm = None
_agent = None


def _get_components():
    """Initialize components on first request."""
    global _bot, _handler, _mcp, _llm, _agent
    if _bot is None:
        _mcp = FlowMapMCPServer()
        _llm = LLMClient(provider=os.getenv("LLM_PROVIDER", "ollama"))
        _agent = Agent(mcp=_mcp, llm=_llm)
        _bot = SlackBot(agent=_agent)
        _handler = SlackRequestHandler(_bot.slack_app)
        logger.info("components_initialized llm=%s tools=%s", _llm.provider, _mcp.list_tools())
    return _bot, _handler, _mcp, _llm


# ── Slack Endpoints ────────────────────────────────────────────────────

@app.route("/slack/events", methods=["POST"])
def slack_events():
    """Handle Slack events with signature verification and rate limiting."""
    bot, handler, mcp, llm = _get_components()
    client_ip = request.remote_addr or "unknown"

    if not check_rate_limit(client_ip):
        logger.warning("rate_limited ip=%s", client_ip)
        return jsonify({"error": "rate_limited"}), 429

    if request.json and request.json.get("type") == "url_verification":
        return jsonify({"challenge": request.json.get("challenge")})

    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")
    body = request.get_data()

    if not verify_slack_signature(bot.signing_secret, timestamp, signature, body):
        logger.warning("invalid_signature ip=%s", client_ip)
        return jsonify({"error": "invalid_signature"}), 403

    return handler.handle(request)


# ── MCP Tools API ──────────────────────────────────────────────────────

@app.route("/mcp/tools", methods=["GET"])
def mcp_list_tools():
    """List available MCP tools (for Slack agent discovery)."""
    _, _, mcp, _ = _get_components()
    
    tools = [
        {
            "name": "flowmap_search",
            "description": "Search code across all indexed repositories",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "mode": {"type": "string", "enum": ["auto", "semantic", "keyword", "symbol"], "default": "auto"},
                    "repo": {"type": "string", "description": "Filter to specific repo"},
                    "limit": {"type": "integer", "default": 10}
                },
                "required": ["query"]
            }
        },
        {
            "name": "flowmap_cat",
            "description": "Read a file from any indexed repository",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Repository name"},
                    "file": {"type": "string", "description": "File path"},
                    "lines": {"type": "string", "description": "Line range like 100-150"},
                    "symbol": {"type": "string", "description": "Symbol name to find"}
                },
                "required": ["repo", "file"]
            }
        },
        {
            "name": "flowmap_history",
            "description": "Get commit history related to a query",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "repo": {"type": "string", "description": "Filter to specific repo"},
                    "limit": {"type": "integer", "default": 10}
                },
                "required": ["query"]
            }
        },
        {
            "name": "flowmap_repos",
            "description": "List all indexed repositories",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "flowmap_symbols",
            "description": "List symbols in a repository",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Repository name"},
                    "limit": {"type": "integer", "default": 30}
                },
                "required": ["repo"]
            }
        },
        {
            "name": "flowmap_map",
            "description": "Show repository structure",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Specific repo to show"}
                }
            }
        }
    ]
    
    return jsonify({"tools": tools})


@app.route("/mcp/call", methods=["POST"])
def mcp_call_tool():
    """Call an MCP tool (for Slack agent)."""
    _, _, mcp, _ = _get_components()
    
    data = request.json
    tool_name = data.get("tool")
    args = data.get("args", {})
    
    try:
        result = mcp.call_tool(tool_name, args)
        return jsonify({"result": str(result)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── Health Check ────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    bot, handler, mcp, llm = _get_components()

    ollama_ok = False
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{llm.base_url}/api/tags")
            ollama_ok = resp.status_code == 200
    except Exception:
        ollama_ok = False

    status = "ok" if ollama_ok else "degraded"
    return jsonify({
        "status": status,
        "llm": llm.provider,
        "ollama_reachable": ollama_ok,
        "mcp_tools": mcp.list_tools(),
        "mcp_endpoints": {
            "list_tools": "/mcp/tools",
            "call_tool": "/mcp/call",
        },
    }), 200 if ollama_ok else 503


if __name__ == "__main__":
    logger.info("FlowMap Agent starting...")
    logger.info("Slack endpoint: http://0.0.0.0:3000/slack/events")
    logger.info("MCP tools: http://0.0.0.0:3000/mcp/tools")
    logger.info("MCP call: http://0.0.0.0:3000/mcp/call")
    app.run(host="0.0.0.0", port=3000)
