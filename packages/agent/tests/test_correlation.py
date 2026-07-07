"""Tests for Temporal Correlation."""

import pytest
from datetime import datetime


class TestCorrelation:
    """Test temporal correlation between commits and metric spikes."""

    def test_correlate_close_timestamps(self):
        """Commit 15 min before spike → high confidence."""
        from agent.correlation import correlate

        commit_ts = datetime(2024, 7, 1, 13, 45)
        metric_ts = datetime(2024, 7, 1, 14, 0)
        confidence = correlate(commit_ts, metric_ts)
        assert confidence > 0.8

    def test_correlate_distant_timestamps(self):
        """Commit 3 hours before spike → low confidence."""
        from agent.correlation import correlate

        commit_ts = datetime(2024, 7, 1, 11, 0)
        metric_ts = datetime(2024, 7, 1, 14, 0)
        confidence = correlate(commit_ts, metric_ts)
        assert confidence < 0.5

    def test_correlate_exact_match(self):
        """Same timestamp → 1.0 confidence."""
        from agent.correlation import correlate

        ts = datetime(2024, 7, 1, 14, 0)
        confidence = correlate(ts, ts)
        assert confidence == 1.0

    def test_correlate_outside_window(self):
        """Commit outside 4-hour window → 0.0 confidence."""
        from agent.correlation import correlate

        commit_ts = datetime(2024, 7, 1, 8, 0)
        metric_ts = datetime(2024, 7, 1, 14, 0)
        confidence = correlate(commit_ts, metric_ts)
        assert confidence == 0.0
