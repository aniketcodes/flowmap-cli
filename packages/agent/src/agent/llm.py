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
class LLMResponse:
    """Response from LLM."""

    content: str
    tool_calls: Optional[List[str]] = None


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

    def chat(self, message: str, tools: list = None) -> LLMResponse:
        """Send a message to the LLM with retry and exponential backoff."""
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if self.provider == "ollama":
                    return self._ollama_chat(message, tools)
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

    def _ollama_chat(self, message: str, tools: list = None) -> LLMResponse:
        """Chat with Ollama API."""
        model = os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "stream": False
        }

        with httpx.Client(timeout=60.0) as client:
            response = client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

        content = data.get("message", {}).get("content", "")
        return LLMResponse(content=content, tool_calls=None)

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
