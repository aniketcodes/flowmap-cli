"""Tests for LLM Integration."""

import pytest
from unittest.mock import patch, Mock
import httpx


class TestLLM:
    """Test LLM client functionality."""

    def test_llm_responds_to_query(self):
        """LLM returns a response."""
        from agent.llm import LLMClient

        client = LLMClient(provider="ollama")
        response = client.chat("Say hello in one word")
        assert response is not None
        assert len(response.content) > 0

    @patch("agent.llm.httpx.Client")
    def test_llm_mock_http_call(self, MockClient):
        """LLM returns response from mocked HTTP call."""
        from agent.llm import LLMClient

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "message": {"content": "Mocked LLM response"}
        }
        MockClient.return_value.__enter__ = Mock(return_value=MockClient.return_value)
        MockClient.return_value.__exit__ = Mock(return_value=False)
        MockClient.return_value.post.return_value = mock_response

        client = LLMClient(provider="ollama")
        response = client.chat("Test message")

        assert response.content == "Mocked LLM response"
        assert response.tool_calls is None

    @patch("agent.llm.httpx.Client")
    def test_llm_retries_on_failure(self, MockClient):
        """LLM retries on transient errors."""
        from agent.llm import LLMClient

        fail_response = Mock()
        fail_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=Mock(), response=Mock(status_code=500)
        )

        ok_response = Mock()
        ok_response.raise_for_status = Mock()
        ok_response.json.return_value = {"message": {"content": "Recovered"}}

        MockClient.return_value.__enter__ = Mock(return_value=MockClient.return_value)
        MockClient.return_value.__exit__ = Mock(return_value=False)
        MockClient.return_value.post.side_effect = [fail_response, ok_response]

        client = LLMClient(provider="ollama")
        response = client.chat("test")

        assert response.content == "Recovered"

    @patch("agent.llm.httpx.Client")
    def test_llm_returns_error_after_max_retries(self, MockClient):
        """LLM returns error after exhausting retries."""
        from agent.llm import LLMClient, MAX_RETRIES

        fail_response = Mock()
        fail_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=Mock(), response=Mock(status_code=500)
        )

        MockClient.return_value.__enter__ = Mock(return_value=MockClient.return_value)
        MockClient.return_value.__exit__ = Mock(return_value=False)
        MockClient.return_value.post.return_value = fail_response

        client = LLMClient(provider="ollama")
        response = client.chat("test")

        assert "unavailable" in response.content.lower() or "error" in response.content.lower()
        assert MockClient.return_value.post.call_count == MAX_RETRIES

    def test_llm_unknown_provider_returns_error(self):
        """Unknown provider returns error response."""
        from agent.llm import LLMClient

        client = LLMClient(provider="nonexistent")
        response = client.chat("test")

        assert "unknown provider" in response.content.lower()
