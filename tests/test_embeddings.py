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

    def test_connection_error_fails_fast_without_truncate_backstop(self):
        """A persistent ConnectionError (runner unreachable) fails fast after the
        retries — no truncate backstop attempt."""
        backend = self._make_backend()

        backend._session.post = MagicMock(
            side_effect=requests.ConnectionError("connection refused")
        )

        with patch("time.sleep"), pytest.raises(ConnectionError, match="unreachable"):
            backend._embed_batch(["test text"])

        # 3 transient retries only — NOT a 4th truncate-backstop call.
        assert backend._session.post.call_count == 3

    def test_http_error_single_input_tries_truncate_backstop_then_raises(self):
        """A single input failing every retry with an HTTP error gets one
        truncate-backstop attempt before raising (3 retries + 1 truncate)."""
        backend = self._make_backend()

        bad = MagicMock()
        bad.status_code = 400
        bad.text = '{"error":"...: EOF"}'
        bad.raise_for_status.side_effect = requests.HTTPError("400", response=bad)
        backend._session.post = MagicMock(return_value=bad)

        with patch("time.sleep"), pytest.raises(ConnectionError, match="even with truncation"):
            backend._embed_batch(["test text"])

        assert backend._session.post.call_count == 4


class TestOllamaBisection:
    """Verify a failing multi-item batch is bisected to isolate the bad input."""

    def _make_backend(self):
        with patch("flowmap.embeddings._ollama_checked", {"http://localhost:11434|test-model": True}):
            from flowmap.embeddings import OllamaBackend
            return OllamaBackend(model="test-model", url="http://localhost:11434")

    @staticmethod
    def _vec(text):
        """Deterministic, input-specific vector so a reorder/half-swap is detectable."""
        return [float(sum(ord(c) for c in text))]

    def test_bisection_recovers_good_inputs_around_a_poison_input(self):
        """A poison input that fails whenever it shares a batch is isolated by
        bisection; all inputs return in original order. Distinct per-input vectors
        catch a misalignment a count check would miss."""
        backend = self._make_backend()
        poison = "POISON"

        def fake_post(url, json=None, timeout=None):
            batch = json["input"]
            truncate = json["truncate"]
            resp = MagicMock()
            # The poison input crashes the runner unless it is alone AND truncated.
            if poison in batch and not (len(batch) == 1 and truncate):
                resp.status_code = 400
                resp.raise_for_status.side_effect = requests.HTTPError(
                    "do embedding request: ...: EOF", response=resp
                )
                resp.text = '{"error":"...: EOF"}'
                return resp
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {"embeddings": [self._vec(t) for t in batch]}
            return resp

        backend._session.post = MagicMock(side_effect=fake_post)
        inputs = ["a", "b", poison, "c"]

        with patch("time.sleep"):
            result = backend._embed_batch(inputs)

        # 1:1 alignment preserved — each input maps to ITS OWN vector, in order.
        assert result == [self._vec(t) for t in inputs]

    def test_connection_outage_fails_fast_without_bisecting(self):
        """A down runner (ConnectionError) on a multi-item batch fails fast — no
        bisection fan-out into a dead server."""
        backend = self._make_backend()
        backend._session.post = MagicMock(
            side_effect=requests.ConnectionError("connection refused")
        )

        with patch("time.sleep"), pytest.raises(ConnectionError, match="unreachable"):
            backend._embed_batch(["a", "b", "c", "d"])

        # Only the transient retries on the original batch — no bisection fan-out.
        assert backend._session.post.call_count == 3

    def test_clean_batch_makes_no_extra_calls(self):
        """A batch that succeeds first try must not bisect or retry."""
        backend = self._make_backend()

        def fake_post(url, json=None, timeout=None):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {"embeddings": [[0.1] for _ in json["input"]]}
            return resp

        backend._session.post = MagicMock(side_effect=fake_post)
        result = backend._embed_batch(["a", "b", "c"])

        assert len(result) == 3
        assert backend._session.post.call_count == 1

    def test_timeout_raises_immediately(self):
        """Timeout does NOT retry — raises immediately."""
        backend = self._make_backend()

        backend._session.post = MagicMock(side_effect=requests.Timeout("timed out"))

        with pytest.raises(TimeoutError, match="timed out"):
            backend._embed_batch(["test text"])

        assert backend._session.post.call_count == 1
