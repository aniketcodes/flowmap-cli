"""Pytest configuration — set env vars before any tests run."""

import os

# Disable GrafanaAdapter auto-connection during tests unless explicitly enabled
# (tests that need GrafanaAdapter set this to 1 themselves)
os.environ.setdefault("ENABLE_GRAFANA_ADAPTER", "0")
