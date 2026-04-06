"""Tests for embedding backends — retry logic, error handling."""

from unittest.mock import MagicMock, patch

import pytest
import requests


class TestOllamaRetry:
    """Verify OllamaBackend retries on transient errors."""

    def _make_backend(self):
        """Create an OllamaBackend with mocked availability check."""
        with patch("flowmap.embeddings._ollama_checked", {"http://localhost:11434|test-model": True}):
            from flowmap.embeddings import OllamaBackend
            backend = OllamaBackend(model="test-model", url="http://localhost:11434")
        return backend

    def test_retry_on_connection_error(self):
        """ConnectionError retries 3 times with backoff, then raises."""
        backend = self._make_backend()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}

        # Fail twice with ConnectionError, succeed on third attempt
        backend._session.post = MagicMock(side_effect=[
            requests.ConnectionError("connection refused"),
            requests.ConnectionError("connection refused"),
            mock_response,
        ])

        with patch("time.sleep"):  # skip actual sleep
            result = backend._embed_batch(["test text"])

        assert len(result) == 1
        assert result[0] == [0.1, 0.2, 0.3]
        assert backend._session.post.call_count == 3

    def test_retry_on_http_500(self):
        """HTTPError (5xx) retries instead of crashing."""
        backend = self._make_backend()

        # Create a real HTTPError
        bad_response = MagicMock()
        bad_response.status_code = 500
        bad_response.raise_for_status.side_effect = requests.HTTPError(
            "500 Server Error", response=bad_response
        )

        good_response = MagicMock()
        good_response.status_code = 200
        good_response.raise_for_status = MagicMock()
        good_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}

        # Fail once with 500, succeed on second attempt
        backend._session.post = MagicMock(side_effect=[bad_response, good_response])

        with patch("time.sleep"):
            result = backend._embed_batch(["test text"])

        assert len(result) == 1
        assert backend._session.post.call_count == 2

    def test_all_retries_exhausted_raises(self):
        """After 3 failed attempts, ConnectionError is raised."""
        backend = self._make_backend()

        backend._session.post = MagicMock(
            side_effect=requests.ConnectionError("connection refused")
        )

        with patch("time.sleep"), pytest.raises(ConnectionError, match="3 retries"):
            backend._embed_batch(["test text"])

        assert backend._session.post.call_count == 3

    def test_timeout_raises_immediately(self):
        """Timeout does NOT retry — raises immediately."""
        backend = self._make_backend()

        backend._session.post = MagicMock(side_effect=requests.Timeout("timed out"))

        with pytest.raises(TimeoutError, match="timed out"):
            backend._embed_batch(["test text"])

        assert backend._session.post.call_count == 1
