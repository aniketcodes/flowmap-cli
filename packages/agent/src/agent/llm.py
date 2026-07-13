"""LLM Integration - Simple chat interface. Agent decides what to do."""

import os
import time
import logging
import httpx
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BASE_DELAY = 1.0


@dataclass
class ToolCall:
    """A single tool call from the LLM."""

    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """Response from LLM."""

    content: str
    tool_calls: Optional[List[ToolCall]] = None


class LLMClient:
    """LLM client — just chats. No business logic."""

    def __init__(self, provider: str = "ollama"):
        self.provider = provider
        if provider == "ollama":
            self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        elif provider == "openrouter":
            self.base_url = "https://openrouter.ai/api/v1"
            self.api_key = os.getenv("OPENROUTER_API_KEY")
        else:
            self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def chat(self, message: str, tools: list = None, system: str = None,
             messages: list = None) -> LLMResponse:
        """Send a message to the LLM with retry and exponential backoff.

        Args:
            message: User message (used when messages=None)
            tools: List of OpenAI-format tool definitions
            system: System prompt
            messages: Full message list (overrides message/system)
        """
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if self.provider == "ollama":
                    return self._ollama_chat(message, tools, system, messages)
                elif self.provider == "openrouter":
                    return self._openrouter_chat(message, tools)
                else:
                    return LLMResponse(content=f"Unknown provider: {self.provider}")
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    delay = BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        "llm_retry attempt=%d/%d provider=%s delay=%.1f error=%s",
                        attempt, MAX_RETRIES, self.provider, delay, e,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "llm_failed attempt=%d provider=%s error=%s",
                        attempt, self.provider, e,
                    )

        return LLMResponse(content=f"LLM unavailable after {MAX_RETRIES} retries: {last_error}")

    def _ollama_chat(self, message: str, tools: list = None,
                     system: str = None, messages: list = None) -> LLMResponse:
        """Chat with Ollama API with native tool calling support."""
        model = os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")

        # Build messages list
        if messages:
            ollama_messages = messages
        else:
            ollama_messages = []
            if system:
                ollama_messages.append({"role": "system", "content": system})
            ollama_messages.append({"role": "user", "content": message})

        payload = {
            "model": model,
            "messages": ollama_messages,
            "stream": False
        }

        # Pass tools in OpenAI format if provided
        if tools:
            payload["tools"] = tools

        with httpx.Client(timeout=120.0) as client:
            response = client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

        msg = data.get("message", {})
        content = msg.get("content", "") or ""
        raw_tool_calls = msg.get("tool_calls", [])

        # Convert to ToolCall objects
        tool_calls = None
        if raw_tool_calls:
            tool_calls = []
            for tc in raw_tool_calls:
                func = tc.get("function", {})
                tool_calls.append(ToolCall(
                    id=tc.get("id", ""),
                    name=func.get("name", ""),
                    arguments=func.get("arguments", {}),
                ))

        return LLMResponse(content=content, tool_calls=tool_calls)

    def _openrouter_chat(self, message: str, tools: list = None) -> LLMResponse:
        """Chat with OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "anthropic/claude-3-haiku",
            "messages": [{"role": "user", "content": message}]
        }

        with httpx.Client(timeout=60.0) as client:
            response = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        content = data["choices"][0]["message"]["content"]
        return LLMResponse(content=content, tool_calls=None)

    def _claude_chat(self, message: str, tools: list = None,
                     system: str = None, messages: list = None) -> LLMResponse:
        """Chat via `claude -p` CLI. Embeds tools in prompt."""
        # Build the prompt with tool definitions
        prompt_parts = []

        if system:
            prompt_parts.append(f"SYSTEM: {system}")

        # Add tool definitions
        if tools:
            tool_desc = "\n".join([
                f"- {t['function']['name']}: {t['function'].get('description', '')}"
                for t in tools
            ])
            prompt_parts.append(f"AVAILABLE TOOLS:\n{tool_desc}")
            prompt_parts.append(
                "To call a tool, respond with exactly this format (no other text):\n"
                "TOOL_CALL: {\"name\": \"tool_name\", \"arguments\": {...}}\n"
                "If no tool needed, just respond normally."
            )

        # Add conversation history
        if messages:
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    prompt_parts.insert(0, f"SYSTEM: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"ASSISTANT: {content}")
                elif role == "user":
                    prompt_parts.append(f"USER: {content}")
        else:
            prompt_parts.append(f"USER: {message}")

        full_prompt = "\n\n".join(prompt_parts)

        # Call claude CLI
        result = subprocess.run(
            ["claude", "-p", full_prompt, "--output-format", "json",
             "--model", self._model, "--allowedTools", ""],
            capture_output=True, text=True, timeout=120
        )

        if result.returncode != 0:
            raise RuntimeError(f"claude CLI failed: {result.stderr[:200]}")

        # Parse JSON response
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return LLMResponse(content=result.stdout)

        content = data.get("result", "")

        # Extract tool calls from response
        tool_calls = None
        tool_match = re.search(r'TOOL_CALL:\s*(\{.*?\})', content, re.DOTALL)
        if tool_match:
            try:
                tc_json = json.loads(tool_match.group(1))
                tool_calls = [ToolCall(
                    id=f"call_{int(time.time()*1000)}",
                    name=tc_json["name"],
                    arguments=tc_json.get("arguments", {}),
                )]
                # Remove tool call from content
                content = content[:tool_match.start()].strip()
            except (json.JSONDecodeError, KeyError):
                pass

        return LLMResponse(content=content, tool_calls=tool_calls)
