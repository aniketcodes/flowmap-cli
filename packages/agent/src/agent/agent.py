"""Agent Logic — LLM decides what tools to call, when to stop."""

import logging
import os
from typing import Callable, Optional

from .adapter import MCPAdapter
from .llm import ToolCall

logger = logging.getLogger(__name__)

# Minimum tool calls before accepting a text response (prevents premature answers)
MIN_TOOL_CALLS = 5

# Clean system prompt — rules only, no TOOL_CALL/FINAL_ANSWER format instructions
_SYSTEM_PROMPT = """You are a code intelligence agent. Use tools to search code, read files, check Git history, and query Grafana.

YOU MUST FOLLOW THESE RULES EXACTLY:
1. Call one tool at a time. Wait for results before the next call.
2. Use FlowMap and Grafana MCP tools extensively — search code, read files, check history, query logs and metrics.
3. ALWAYS use flowmap_cat to read files before concluding. Every claim needs file:line evidence.
4. If a tool returns ERROR or empty data — STOP. Do NOT retry. Switch to a different approach.
5. After getting tool results, READ them carefully. When you have enough evidence, PROVIDE YOUR TEXT ANSWER. Do NOT keep calling tools indefinitely.
6. Loki logs use job="demo-services" (not demo-order-service). Filter by service label or line filter (|=) to find specific service logs.
7. Large numeric IDs (Snowflake IDs, etc.) may be ROUNDED in logs due to IEEE 754 precision loss. If searching for an exact ID fails, search for the first 10-12 digits as a partial match.
8. ALWAYS search by request_id first — it's consistent across all services and shows the full transaction flow. When you find a request_id in logs, search Loki for ALL logs with that request_id to see what EVERY service did. Compare transaction IDs across services to detect modifications.
9. When tracing a transaction through multiple services, check EVERY service in the chain (payment → order → ledger). Do NOT skip any service.

ROOT CAUSE ANALYSIS — YOU MUST DO THIS FOR EVERY DIAGNOSIS:
10. "not found", "timeout", or "error" = SYMPTOM. You are NOT done. Find WHY.
11. Service A reports error → Search who SENDS data to Service A. Bug is in SENDER.
12. Search ACROSS all repos, not just one.
13. Name EXACT FILE AND LINE of root cause.

FORMAT for Slack:
- **Bold** headers, - bullet points, `file:line` references
- Start with one-line summary, then details"""


class Agent:
    """Agent that lets the LLM decide what to do — no hardcoded logic."""

    def __init__(
        self,
        adapters: list[MCPAdapter] = None,
        llm=None,
        slack_client=None,
        on_progress: Optional[Callable[[str], None]] = None,
        # Backward compatibility
        mcp=None,
        slack_mcp_client=None,
    ):
        self.adapters = adapters or []
        self.llm = llm
        self.slack_client = slack_client
        self.on_progress = on_progress or (lambda x: None)

        # Backward compatibility: wrap old mcp param as adapter
        if mcp and not self.adapters:
            from .flowmap_adapter import FlowMapAdapter
            self.adapters.append(FlowMapAdapter(server=mcp))

        if slack_mcp_client and not any(a.name() == "slack" for a in self.adapters):
            from .slack_adapter import SlackAdapter
            self.adapters.append(SlackAdapter(client=slack_mcp_client))

        # Auto-add GrafanaAdapter only if explicitly enabled via env var
        # (avoids breaking tests that pass their own adapter list)
        if not any(a.name() == "grafana" for a in self.adapters) and os.getenv("ENABLE_GRAFANA_ADAPTER", "1") == "1":
            try:
                from .grafana_adapter import GrafanaAdapter
                grafana_adapter = GrafanaAdapter(
                    grafana_url=os.getenv("GRAFANA_URL", "http://localhost:3000"),
                    service_account_token=os.getenv("GRAFANA_SERVICE_ACCOUNT_TOKEN", ""),
                )
                # Test connect — if it fails, skip adding the adapter
                if grafana_adapter.connect():
                    self.adapters.append(grafana_adapter)
                    logger.info("GrafanaAdapter connected to mcp-grafana")
                else:
                    logger.warning("GrafanaAdapter failed to connect, skipping")
            except Exception as e:
                logger.warning("GrafanaAdapter not available: %s", e)

        # Build routing table once — O(1) lookup per tool call
        self._tool_routes: dict[str, MCPAdapter] = {}
        for adapter in self.adapters:
            for tool in adapter.list_tools():
                self._tool_routes[tool["name"]] = adapter

    def _build_openai_tools(self) -> list[dict]:
        """Build OpenAI-format tool definitions for native tool calling."""
        tools = []
        for adapter in self.adapters:
            for tool in adapter.list_tools():
                schema = tool.get("inputSchema", {})
                openai_params = {
                    "type": "object",
                    "properties": {},
                    "required": schema.get("required", []),
                }
                for prop_name, prop_info in schema.get("properties", {}).items():
                    openai_params["properties"][prop_name] = {
                        "type": prop_info.get("type", "string"),
                        "description": prop_info.get("description", ""),
                    }

                tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": openai_params,
                    }
                })

        # Add Slack tools
        if self.slack_client:
            tools.extend([
                {
                    "type": "function",
                    "function": {
                        "name": "slack_history",
                        "description": "Get recent messages from a Slack channel.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "channel_id": {"type": "string", "description": "Slack channel ID"},
                                "limit": {"type": "integer", "description": "Number of messages to fetch"},
                            },
                            "required": ["channel_id"],
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "slack_search",
                        "description": "Search Slack messages for a query.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search query"},
                                "channel_id": {"type": "string", "description": "Optional channel ID to search in"},
                                "limit": {"type": "integer", "description": "Number of results"},
                            },
                            "required": ["query"],
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "slack_channels",
                        "description": "List all Slack channels the bot can access.",
                        "parameters": {"type": "object", "properties": {}, "required": []}
                    }
                },
            ])

        return tools

    def classify_intent(self, message: str) -> str:
        """Not used — kept for backward compatibility."""
        return "diagnose"

    def get_tools_for_intent(self, intent: str) -> list:
        """Not used — kept for backward compatibility."""
        return []

    def diagnose(self, query: str, on_progress: Callable[[str], None] = None,
                 max_steps: int = 15) -> str:
        """Let the LLM drive — it decides what tools to call."""
        progress = on_progress or self.on_progress

        if not self.llm:
            return f"Diagnosis for: {query}. No LLM available."

        if not self.adapters and not self.slack_client:
            return self.llm.chat(query).content

        progress("🤔 Analyzing your query...")

        tools = self._build_openai_tools()

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        total_tool_calls = 0
        tool_call_history = {}  # {tool_name: count} — prevent repeated calls

        for step in range(max_steps):
            progress(f"🧠 Thinking... (step {step + 1})")

            response = self.llm.chat(message="", messages=messages, tools=tools)
            text = response.content.strip() if response.content else ""

            logger.info("agent_step=%d tool_calls=%s args=%s content=%.200s",
                        step,
                        [tc.name for tc in response.tool_calls] if response.tool_calls else "none",
                        {tc.name: tc.arguments for tc in response.tool_calls} if response.tool_calls else {},
                        text)

            # If LLM returned tool calls, execute them
            if response.tool_calls:
                # Block tools called too many times
                blocked_tools = {name for name, count in tool_call_history.items() if count >= 3}
                filtered_calls = [tc for tc in response.tool_calls if tc.name not in blocked_tools]

                if not filtered_calls and response.tool_calls:
                    # All requested tools are blocked — force LLM to answer
                    blocked_list = ", ".join(blocked_tools)
                    messages.append({"role": "user", "content": (
                        f"You have called these tools too many times: {blocked_list}. "
                        "They are not returning useful data. STOP calling them. "
                        "Provide your best answer now based on what you have found so far."
                    )})
                    continue

                assistant_msg = {"role": "assistant", "content": text or None}
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments}
                    }
                    for tc in filtered_calls
                ]
                messages.append(assistant_msg)

                for tc in filtered_calls:
                    total_tool_calls += 1
                    tool_call_history[tc.name] = tool_call_history.get(tc.name, 0) + 1
                    progress(f"🔍 Calling {tc.name}...")
                    result = self._execute_tool_call(tc)
                    progress(f"✓ {tc.name} done")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

                continue

            # No tool calls — LLM is giving a text response
            if text:
                # Strip FINAL_ANSWER prefix if present (LLM may still generate it)
                if text.startswith("FINAL_ANSWER:"):
                    text = text[len("FINAL_ANSWER:"):].strip()

                # Accept answer after minimum tool calls
                if total_tool_calls >= MIN_TOOL_CALLS:
                    progress("✅ Found it!")
                    return text

                # Not enough tool calls yet — nudge
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": (
                    f"You have only made {total_tool_calls} tool call(s). "
                    "You must search and read code before answering. "
                    "Call flowmap_search or another tool to find the answer."
                )})
            else:
                messages.append({"role": "user", "content": (
                    "You must call a tool to find the answer."
                )})

        return "Reached max steps without a final answer."

    def _execute_tool_call(self, tc: ToolCall) -> str:
        """Execute a native tool call."""
        try:
            adapter = self._tool_routes.get(tc.name)
            if adapter:
                return adapter.call_tool(tc.name, tc.arguments)

            if tc.name.startswith("slack_") and self.slack_client:
                return self._execute_slack_tool(tc.name, tc.arguments)

            return f"Unknown tool: {tc.name}"
        except Exception as e:
            return f"Tool error: {e}"

    def _execute_slack_tool(self, tool_name: str, kwargs: dict) -> str:
        """Execute direct Slack API tools (not MCP)."""
        if tool_name == "slack_history" and self.slack_client:
            messages = self.slack_client.get_channel_history(
                kwargs.get("channel_id", ""),
                limit=int(kwargs.get("limit", 50))
            )
            return self._format_slack_messages(messages)
        elif tool_name == "slack_search" and self.slack_client:
            messages = self.slack_client.search_messages(
                kwargs.get("query", ""),
                channel_id=kwargs.get("channel_id"),
                limit=int(kwargs.get("limit", 10))
            )
            return self._format_slack_messages(messages)
        elif tool_name == "slack_channels" and self.slack_client:
            channels = self.slack_client.list_channels()
            return "\n".join(f"- #{ch.name} (id: {ch.id})" for ch in channels)
        return f"Unknown Slack tool: {tool_name}"

    def _format_slack_messages(self, messages: list) -> str:
        """Format Slack messages for display."""
        if not messages:
            return "No Slack messages found."
        formatted = []
        for msg in messages:
            formatted.append(f"[{msg.user}] {msg.text[:200]}")
        return "\n".join(formatted)
