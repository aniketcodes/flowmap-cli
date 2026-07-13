"""Grafana Adapter — Wraps mcp-grafana subprocess via stdio JSON-RPC.

Architecture:
- mcp-grafana runs as a subprocess (stdio transport)
- Communication: newline-delimited JSON-RPC 2.0 over stdin/stdout
- No external MCP library needed — direct subprocess + JSON parsing
- No anyio/HTTP complexity — stdio is connectionless

The adapter spawns the subprocess lazily on first connect() and reuses it
for subsequent calls. This is simpler than running mcp-grafana as a
long-lived container.
"""

import json
import logging
import subprocess
import threading
from typing import Any

from .adapter import MCPAdapter

logger = logging.getLogger(__name__)


class GrafanaAdapter(MCPAdapter):
    """Adapter for the official grafana/mcp-grafana MCP server.

    Wraps mcp-grafana via stdio JSON-RPC. Tool names are prefixed with
    'grafana_' to disambiguate from FlowMap/Slack tools.
    """

    def __init__(
        self,
        command: list[str] = None,
        grafana_url: str = "http://localhost:3000",
        service_account_token: str = "",
    ):
        # Default command: docker exec into the running mcp-grafana container
        self._command = command or [
            "docker", "exec", "-i", "demo-repos-grafana-mcp-1",
            "/app/mcp-grafana", "-t", "stdio", "-disable-proxied",
        ]
        self._grafana_url = grafana_url
        self._service_account_token = service_account_token
        self._proc = None  # subprocess.Popen instance
        self._connected = False
        self._tools_cache: list[dict] | None = None
        self._request_id = 0
        self._lock = threading.Lock()  # Serialize calls to subprocess

    def name(self) -> str:
        return "grafana"

    def connect(self) -> bool:
        """Spawn the mcp-grafana subprocess and perform the MCP initialize handshake."""
        if self._connected:
            return True

        try:
            env = {}
            if self._grafana_url:
                env["GRAFANA_URL"] = self._grafana_url
            if self._service_account_token:
                env["GRAFANA_SERVICE_ACCOUNT_TOKEN"] = self._service_account_token

            self._proc = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # line-buffered
                env={**__import__("os").environ, **env} if env else None,
            )
        except Exception as e:
            logger.error("Failed to spawn mcp-grafana subprocess: %s", e)
            return False

        # Send initialize request
        try:
            init_response = self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "flowmap-agent", "version": "1.0"},
            })
            if "error" in init_response:
                logger.error("mcp-grafana initialize failed: %s", init_response["error"])
                self._proc.terminate()
                self._proc = None
                return False

            # Send notifications/initialized (no response expected)
            self._send_notification("notifications/initialized")
            self._connected = True
            logger.info("Connected to mcp-grafana: %s", init_response.get("result", {}).get("serverInfo", {}))
            return True
        except Exception as e:
            logger.error("mcp-grafana handshake failed: %s", e)
            self._proc.terminate()
            self._proc = None
            return False

    def disconnect(self):
        """Terminate the mcp-grafana subprocess."""
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
            self._connected = False

    def _send_request(self, method: str, params: dict = None) -> dict:
        """Send a JSON-RPC request and read the response (one line)."""
        with self._lock:
            self._request_id += 1
            req = {
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
            }
            if params is not None:
                req["params"] = params

            line = json.dumps(req) + "\n"
            self._proc.stdin.write(line)
            self._proc.stdin.flush()

            # Read until we get a response with matching id
            while True:
                response_line = self._proc.stdout.readline()
                if not response_line:
                    raise RuntimeError("mcp-grafana subprocess closed unexpectedly")
                response = json.loads(response_line)
                if response.get("id") == self._request_id:
                    return response

    def _send_notification(self, method: str, params: dict = None):
        """Send a JSON-RPC notification (no response expected)."""
        with self._lock:
            notif = {
                "jsonrpc": "2.0",
                "method": method,
            }
            if params is not None:
                notif["params"] = params
            line = json.dumps(notif) + "\n"
            self._proc.stdin.write(line)
            self._proc.stdin.flush()

    def list_tools(self) -> list[dict]:
        """Return the 4 Grafana tools we expose, prefixed with 'grafana_'.

        These definitions are hardcoded because:
        - We want to expose only 4 of the 65 tools mcp-grafana provides
        - The schemas are stable (documented in the mcp-grafana README)
        - Hardcoding prevents silent breakage if mcp-grafana renames tools
        """
        return [
            {
                "name": "grafana_query_loki",
                "description": (
                    "Query Loki logs using LogQL. Use line filter (|=) not label filter "
                    "({request_id=\"...\"}) because the label index is updated lazily (~10 min delay). "
                    "Example: '{job=\"demo-services\"} |= \"<request_id>\"' returns log lines from all 3 services."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "datasourceUid": {"type": "string", "default": "loki"},
                        "logql": {"type": "string", "description": "LogQL query"},
                        "limit": {"type": "integer", "default": 50},
                    },
                    "required": ["logql"],
                },
            },
            {
                "name": "grafana_query_prometheus",
                "description": (
                    "Query Prometheus metrics using PromQL. Example: 'rate(ledger_transactions_not_found_total[5m])'"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "datasourceUid": {"type": "string", "default": "prometheus"},
                        "query": {"type": "string", "description": "PromQL query"},
                        "queryType": {"type": "string", "enum": ["instant", "range"], "default": "instant"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "grafana_list_datasources",
                "description": "List all configured Grafana datasources (Loki, Prometheus, Tempo, etc.)",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "grafana_search_dashboards",
                "description": "Search for Grafana dashboards by title or tag",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query (e.g., 'transaction')"},
                    },
                },
            },
        ]

    def call_tool(self, name: str, args: dict[str, Any]) -> str:
        """Route a tool call to mcp-grafana via JSON-RPC.

        Strips the 'grafana_' prefix before sending to mcp-grafana.
        Returns the tool result as a string.
        """
        if not self._connected:
            if not self.connect():
                return f"Error: not connected to mcp-grafana"

        # Strip prefix: grafana_query_loki -> query_loki_logs
        mcp_name = name.removeprefix("grafana_")
        # Map our tool name to mcp-grafana's tool name
        mcp_name = self._map_tool_name(mcp_name)

        # Apply correct datasourceUid (LLM often hallucinates UIDs like "loki-uid")
        if mcp_name == "query_loki_logs":
            args["datasourceUid"] = "loki"
        elif mcp_name == "query_prometheus":
            args["datasourceUid"] = "prometheus"

        try:
            response = self._send_request("tools/call", {
                "name": mcp_name,
                "arguments": args,
            })

            if "error" in response:
                return f"Error: {response['error']}"

            result = response.get("result", {})
            content_list = result.get("content", [])
            if not content_list:
                return "No results"

            # Concatenate text content
            texts = []
            for item in content_list:
                if item.get("type") == "text":
                    texts.append(item.get("text", ""))
            return "\n".join(texts)
        except Exception as e:
            logger.error("mcp-grafana call_tool error: %s", e, exc_info=True)
            return f"Error: {e}"

    def _map_tool_name(self, name: str) -> str:
        """Map our short names to mcp-grafana's actual tool names.

        Our prefix:       mcp-grafana:
        query_loki          -> query_loki_logs
        query_prometheus    -> query_prometheus
        list_datasources    -> list_datasources
        search_dashboards   -> search_dashboards
        """
        mapping = {
            "query_loki": "query_loki_logs",
            "query_prometheus": "query_prometheus",
            "list_datasources": "list_datasources",
            "search_dashboards": "search_dashboards",
        }
        return mapping.get(name, name)
