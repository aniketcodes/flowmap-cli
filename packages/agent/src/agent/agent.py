"""Agent Logic — LLM decides what tools to call, when to stop."""

import json
import logging
from typing import Callable, Optional

from .adapter import MCPAdapter

logger = logging.getLogger(__name__)

# Instructions for the LLM (tool definitions are generated dynamically)
_TOOL_INSTRUCTIONS = """To call a tool, output exactly one line in this format:
TOOL_CALL: tool_name(arg1, arg2, key=value)

To give your final answer, output exactly one line:
FINAL_ANSWER: your answer here

Rules — you MUST follow these:
1. Call tools one at a time — wait for results before calling the next
2. Start with flowmap_search to find relevant code
3. Also search Slack for related conversations using slack_search or slack_history
4. NEVER guess or assume — always read the actual file with flowmap_cat before concluding
5. If a search result references a file, READ that file with flowmap_cat before using it in your answer
6. If the query asks about timing, also call flowmap_history
7. Trace the full flow — if you find a constant, find where it's used; if you find a function, find where it's called
8. Every claim must reference a file:line that you actually read
9. Combine code findings with Slack conversation context for a complete diagnosis
10. Stop calling tools only when you have read the code and have evidence for your answer
11. NEVER reference Slack channels or conversations you haven't explicitly searched with slack_search or slack_history. If you didn't call a Slack tool, don't mention Slack.
12. NEVER cite specific TTL values, durations, or config values unless you read them from the actual code with flowmap_cat

FORMAT YOUR FINAL_ANSWER for Slack readability:
- Use **bold** for section headers and key terms
- Use bullet points (-) for lists and enumerations
- Put file paths and code references in backticks: `src/file.ts:42`
- Break long answers into sections with headers
- Keep each bullet point to one line
- Start with a one-line summary, then details"""


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

        # Build routing table once — O(1) lookup per tool call
        self._tool_routes: dict[str, MCPAdapter] = {}
        for adapter in self.adapters:
            for tool in adapter.list_tools():
                self._tool_routes[tool["name"]] = adapter

    def _build_tool_prompt(self) -> str:
        """Generate tool definitions dynamically from adapters."""
        lines = ["You have access to these tools:\n"]

        for adapter in self.adapters:
            for tool in adapter.list_tools():
                # Build parameter signature from inputSchema
                schema = tool.get("inputSchema", {})
                props = schema.get("properties", {})
                required = schema.get("required", [])

                params = []
                for param_name, param_info in props.items():
                    default = "null" if param_name not in required else ""
                    params.append(f"{param_name}={default}" if default else param_name)

                lines.append(f"{tool['name']}({', '.join(params)})")
                lines.append(f"  {tool.get('description', '')}\n")

        # Add direct Slack tools if available
        if self.slack_client:
            lines.append("slack_history(channel_id, limit=50)")
            lines.append("  Get recent messages from a Slack channel.\n")
            lines.append("slack_search(query, channel_id=null, limit=10)")
            lines.append("  Search Slack messages for a query.\n")
            lines.append("slack_channels()")
            lines.append("  List all Slack channels.\n")

        lines.append(_TOOL_INSTRUCTIONS)
        return "\n".join(lines)

    def _get_tool_definitions(self) -> list[dict]:
        """Return list of tool definitions (kept for backward compatibility)."""
        tools = []
        for adapter in self.adapters:
            for tool in adapter.list_tools():
                tools.append({
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                })
        if self.slack_client:
            tools.extend([
                {"name": "slack_history", "description": "Get recent messages from a Slack channel"},
                {"name": "slack_search", "description": "Search Slack messages for a query"},
                {"name": "slack_channels", "description": "List all Slack channels"},
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

        # Generate tool prompt dynamically from adapters
        tool_prompt = self._build_tool_prompt()

        messages = [
            {"role": "system", "content": tool_prompt},
            {"role": "user", "content": query},
        ]

        cat_calls = 0  # Track how many files the LLM has actually read

        # Agentic loop — LLM calls tools until it gives FINAL_ANSWER
        for step in range(max_steps):
            progress(f"🧠 Thinking... (step {step + 1})")
            prompt = self._format_messages(messages)
            response = self.llm.chat(prompt)
            text = response.content.strip()

            logger.info("agent_step=%d response=%.200s", step, text)

            # Check for final answer
            if text.startswith("FINAL_ANSWER:"):
                # Don't accept FINAL_ANSWER until at least one file was read
                if cat_calls == 0:
                    messages.append({"role": "assistant", "content": text})
                    messages.append({"role": "user", "content": (
                        "You have not read any files yet. Before giving a final answer, "
                        "you MUST call flowmap_cat to read the relevant source code. "
                        "Do not guess — read the actual code first."
                    )})
                    continue
                progress("✅ Found it!")
                return text[len("FINAL_ANSWER:"):].strip()

            # Check for tool call
            if text.startswith("TOOL_CALL:"):
                tool_call = text[len("TOOL_CALL:"):].strip()
                if "flowmap_cat(" in tool_call:
                    cat_calls += 1

                # Extract tool name for progress
                tool_name = tool_call.split("(")[0] if "(" in tool_call else tool_call
                progress(f"🔍 Calling {tool_name}...")

                result = self._execute_tool(tool_call)
                progress(f"✓ {tool_name} done")
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": f"Tool result:\n{result}"})
                continue

            # LLM is reasoning — push it to use a tool or answer
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": (
                "You must respond with either TOOL_CALL: or FINAL_ANSWER:. "
                "Do not explain your reasoning — just call the tool or give the answer."
            )})
            continue

        return "Reached max steps without a final answer."

    def _execute_tool(self, tool_call: str) -> str:
        """Parse and execute a tool call."""
        try:
            # Parse: tool_name(arg1, arg2, key=value)
            paren_start = tool_call.index("(")
            paren_end = tool_call.rindex(")")
            tool_name = tool_call[:paren_start].strip()
            args_str = tool_call[paren_start + 1:paren_end]

            # Parse arguments
            kwargs = {}
            if args_str:
                for arg in self._parse_args(args_str):
                    if "=" in arg:
                        k, v = arg.split("=", 1)
                        kwargs[k.strip()] = self._coerce(v.strip())
                    else:
                        # Positional arg
                        kwargs["query" if "query" not in kwargs else "repo"] = self._coerce(arg.strip())

            # O(1) routing via pre-built table
            adapter = self._tool_routes.get(tool_name)
            if adapter:
                return adapter.call_tool(tool_name, kwargs)

            # Handle direct Slack tools
            if tool_name.startswith("slack_") and self.slack_client:
                return self._execute_slack_tool(tool_name, kwargs)

            return f"Unknown tool: {tool_name}"

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

    def _parse_args(self, args_str: str) -> list[str]:
        """Simple argument parser that handles quotes."""
        args = []
        current = ""
        in_quote = None
        for ch in args_str:
            if ch in ("'", '"') and in_quote is None:
                in_quote = ch
            elif ch == in_quote:
                in_quote = None
            elif ch == "," and in_quote is None:
                args.append(current.strip())
                current = ""
                continue
            current += ch
        if current.strip():
            args.append(current.strip())
        return args

    def _coerce(self, value: str):
        """Coerce string values to appropriate types."""
        if value == "null":
            return None
        if value == "true":
            return True
        if value == "false":
            return False
        # Strip quotes
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            return value[1:-1]
        try:
            return int(value)
        except ValueError:
            return value

    def _format_messages(self, messages: list[dict]) -> str:
        """Format message history into a single prompt."""
        parts = []
        for m in messages:
            parts.append(f"[{m['role'].upper()}]\n{m['content']}")
        return "\n\n".join(parts)

    def _format_results(self, results: list) -> str:
        if not results:
            return "No results found."
        formatted = []
        for i, r in enumerate(results):
            formatted.append(f"[{i+1}] {r.repo}/{r.file} (lines {r.start_line}-{r.end_line})")
            formatted.append(f"    {r.text[:300]}")
            if r.signature:
                formatted.append(f"    Signature: {r.signature}")
            formatted.append("")
        return "\n".join(formatted)

    def _format_commits(self, commits: list) -> str:
        if not commits:
            return "No commit history available."
        return "\n".join(
            f"- [{c.sha[:8]}] {c.date} by {c.author}: {c.message}"
            for c in commits
        )

    def _format_slack_messages(self, messages: list) -> str:
        """Format Slack messages for display."""
        if not messages:
            return "No Slack messages found."
        formatted = []
        for msg in messages:
            formatted.append(f"[{msg.user}] {msg.text[:200]}")
        return "\n".join(formatted)
