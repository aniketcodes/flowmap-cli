"""Tests for Security Features."""

import time
import pytest
from unittest.mock import patch


class TestTTLDedup:
    """Test TTL-based event deduplication."""

    def test_duplicate_event_within_ttl(self):
        """Duplicate events within TTL are ignored."""
        from agent.security import TTLCache

        cache = TTLCache(ttl=300)
        cache.add("EV1")
        assert cache.contains("EV1") is True

    def test_event_expires_after_ttl(self):
        """Events expire after TTL."""
        from agent.security import TTLCache

        cache = TTLCache(ttl=1)
        cache.add("EV1")
        time.sleep(1.1)
        assert cache.contains("EV1") is False

    def test_different_events_not_deduped(self):
        """Different event IDs are not confused."""
        from agent.security import TTLCache

        cache = TTLCache(ttl=300)
        cache.add("EV1")
        assert cache.contains("EV2") is False

    def test_len_returns_count(self):
        """len() returns number of active items."""
        from agent.security import TTLCache

        cache = TTLCache(ttl=300)
        cache.add("EV1")
        cache.add("EV2")
        assert len(cache) == 2

    def test_len_excludes_expired(self):
        """len() excludes expired items."""
        from agent.security import TTLCache

        cache = TTLCache(ttl=1)
        cache.add("EV1")
        time.sleep(1.1)
        assert len(cache) == 0


class TestTTLCacheAtomic:
    """Test TTLCache atomic operations for thread safety."""

    def test_contains_and_add_is_atomic(self):
        """contains_and_add returns False on first call, True on second."""
        from agent.security import TTLCache

        cache = TTLCache(ttl=60)
        assert cache.contains_and_add("evt1") is False  # First time: not seen
        assert cache.contains_and_add("evt1") is True   # Second time: seen
        assert cache.contains_and_add("evt2") is False  # Different event

    def test_contains_and_add_thread_safe(self):
        """Concurrent calls don't produce duplicate processing."""
        import threading
        from agent.security import TTLCache

        cache = TTLCache(ttl=60)
        results = []

        def try_add():
            result = cache.contains_and_add("shared_event")
            results.append(result)

        threads = [threading.Thread(target=try_add) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one thread should get False (first processor)
        assert results.count(False) == 1
        # Nine threads should get True (duplicate detected)
        assert results.count(True) == 9


class TestSlackSignatureVerification:
    """Test Slack signature verification."""

    def test_valid_signature_accepted(self):
        """Valid signature is accepted."""
        from agent.security import verify_slack_signature
        import hashlib
        import hmac

        secret = "test_secret"
        timestamp = str(int(time.time()))
        body = b'{"test": true}'
        sig_basestring = f"v0:{timestamp}:{body.decode()}"
        signature = "v0=" + hmac.new(secret.encode(), sig_basestring.encode(), hashlib.sha256).hexdigest()

        assert verify_slack_signature(secret, timestamp, signature, body) is True

    def test_invalid_signature_rejected(self):
        """Invalid signature is rejected."""
        from agent.security import verify_slack_signature

        assert verify_slack_signature("secret", "123", "v0=bad", b"body") is False

    def test_expired_timestamp_rejected(self):
        """Requests older than 5 minutes are rejected."""
        from agent.security import verify_slack_signature

        old_timestamp = str(int(time.time()) - 600)
        assert verify_slack_signature("secret", old_timestamp, "v0=bad", b"body") is False

    def test_no_secret_skips_verification(self):
        """Missing secret skips verification (warning only)."""
        from agent.security import verify_slack_signature

        assert verify_slack_signature(None, "123", "sig", b"body") is True

    def test_malformed_timestamp_rejected(self):
        """Non-numeric timestamp is rejected."""
        from agent.security import verify_slack_signature

        assert verify_slack_signature("secret", "not-a-number", "v0=bad", b"body") is False


class TestRateLimiting:
    """Test rate limiting."""

    def test_allows_requests_under_limit(self):
        """Requests under limit are allowed."""
        from agent.security import check_rate_limit, _rate_limits
        _rate_limits.clear()

        for _ in range(5):
            assert check_rate_limit("test_ip") is True

    def test_blocks_requests_over_limit(self):
        """Requests over limit are blocked."""
        from agent.security import check_rate_limit, _rate_limits, RATE_LIMIT_MAX
        _rate_limits.clear()

        for _ in range(RATE_LIMIT_MAX + 1):
            result = check_rate_limit("test_ip2")

        assert result is False

    def test_different_ips_independent(self):
        """Different IPs have independent limits."""
        from agent.security import check_rate_limit, _rate_limits, RATE_LIMIT_MAX
        _rate_limits.clear()

        for _ in range(RATE_LIMIT_MAX):
            check_rate_limit("ip_a")

        assert check_rate_limit("ip_b") is True

    def test_cleanup_removes_stale_entries(self):
        """cleanup_rate_limits removes expired entries."""
        from agent.security import check_rate_limit, cleanup_rate_limits, _rate_limits, RATE_LIMIT_MAX
        _rate_limits.clear()

        # Add entries for an IP
        for _ in range(RATE_LIMIT_MAX):
            check_rate_limit("cleanup_test_ip")

        # Force entries to be old by manipulating timestamps
        _rate_limits["cleanup_test_ip"] = [time.time() - 120] * RATE_LIMIT_MAX

        removed = cleanup_rate_limits()
        assert removed > 0
        assert "cleanup_test_ip" not in _rate_limits
