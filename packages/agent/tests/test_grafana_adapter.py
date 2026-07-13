"""Tests for GrafanaAdapter — stdio subprocess wrapper for mcp-grafana."""

import os
import pytest


class TestGrafanaAdapterName:
    """Test 1: The adapter reports its name for tool prefixing."""

    def test_name_returns_grafana(self):
        """name() returns 'grafana' for tool prefix disambiguation."""
        from agent.grafana_adapter import GrafanaAdapter

        # Use a dummy command — we never spawn anything in this test
        adapter = GrafanaAdapter(
            command=["echo", "noop"],
            grafana_url="http://localhost:3000",
            service_account_token="dummy",
        )
        assert adapter.name() == "grafana"


class TestGrafanaAdapterListTools:
    """Test 2: list_tools() returns the 4 prefixed tools we expose."""

    def test_list_tools_returns_four_prefixed_tools(self):
        """list_tools() returns 4 tools, all prefixed with 'grafana_'."""
        from agent.grafana_adapter import GrafanaAdapter

        adapter = GrafanaAdapter(
            command=["echo", "noop"],
            grafana_url="http://localhost:3000",
            service_account_token="dummy",
        )

        tools = adapter.list_tools()

        names = {t["name"] for t in tools}
        assert names == {
            "grafana_query_loki",
            "grafana_query_prometheus",
            "grafana_list_datasources",
            "grafana_search_dashboards",
        }
        for tool in tools:
            assert "description" in tool
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"


# Container name for live integration tests
MCP_GRAFANA_CONTAINER = "demo-repos-grafana-mcp-1"


@pytest.mark.integration
class TestGrafanaAdapterQueryLoki:
    """Test 3: query_loki returns log content from all 3 services.

    This is an integration test — requires the mcp-grafana container running.
    """

    def test_query_loki_returns_log_content(self):
        """query_loki returns log lines containing the given request_id."""
        from agent.grafana_adapter import GrafanaAdapter

        adapter = GrafanaAdapter(
            command=[
                "docker", "exec", "-i", MCP_GRAFANA_CONTAINER,
                "/app/mcp-grafana", "-t", "stdio", "-disable-proxied",
            ],
        )

        # Use a known request_id from the test fixtures
        result = adapter.call_tool(
            "grafana_query_loki",
            {
                "datasourceUid": "loki",
                "logql": '{job="demo-services"} |= "661f961b-9101-4249-8ccc-b66ce7c0488b"',
                "limit": 10,
            },
        )

        assert isinstance(result, str)
        # The bug: payment has original (334374682734104576), order has corrupted (334374682734104600)
        assert "334374682734104576" in result, (
            f"Expected original txn_id in response, got: {result[:300]}"
        )

    def test_query_prometheus_returns_metric_value(self):
        """query_prometheus returns the actual metric value, not just 'no data'."""
        from agent.grafana_adapter import GrafanaAdapter

        adapter = GrafanaAdapter(
            command=[
                "docker", "exec", "-i", MCP_GRAFANA_CONTAINER,
                "/app/mcp-grafana", "-t", "stdio", "-disable-proxied",
            ],
        )

        result = adapter.call_tool(
            "grafana_query_prometheus",
            {
                "datasourceUid": "prometheus",
                "expr": "ledger_transactions_not_found_total",
                "queryType": "instant",
                "endTime": "now",
            },
        )

        assert isinstance(result, str)
        # Must contain a numeric value (proves Prometheus returned data)
        import re
        assert re.search(r'\d+', result), (
            f"Expected numeric value, got: {result[:300]}"
        )
        # Must mention the metric name
        assert "ledger" in result.lower() or "transaction" in result.lower()

    def test_list_datasources_returns_configured_datasources(self):
        """list_datasources returns at least 3 datasources (Loki, Prometheus, Tempo)."""
        from agent.grafana_adapter import GrafanaAdapter

        adapter = GrafanaAdapter(
            command=[
                "docker", "exec", "-i", MCP_GRAFANA_CONTAINER,
                "/app/mcp-grafana", "-t", "stdio", "-disable-proxied",
            ],
        )

        result = adapter.call_tool("grafana_list_datasources", {})

        assert isinstance(result, str)
        result_lower = result.lower()
        # Must mention all 3 configured datasources
        assert "loki" in result_lower, f"Expected Loki, got: {result[:300]}"
        assert "prometheus" in result_lower, f"Expected Prometheus, got: {result[:300]}"
        assert "tempo" in result_lower, f"Expected Tempo, got: {result[:300]}"

    def test_search_dashboards_returns_provisioned_dashboards(self):
        """search_dashboards returns the 2 provisioned demo dashboards."""
        from agent.grafana_adapter import GrafanaAdapter

        adapter = GrafanaAdapter(
            command=[
                "docker", "exec", "-i", MCP_GRAFANA_CONTAINER,
                "/app/mcp-grafana", "-t", "stdio", "-disable-proxied",
            ],
        )

        result = adapter.call_tool("grafana_search_dashboards", {"query": "transaction"})

        assert isinstance(result, str)
        result_lower = result.lower()
        # Must find at least one of the provisioned dashboards
        assert "transaction" in result_lower, (
            f"Expected transaction-flow dashboard, got: {result[:500]}"
        )

    def test_two_hop_query_reveals_corrupted_txn_id(self):
        """THE DEMO TEST: 2-hop query reveals the JSON precision bug.

        Simulates the demo flow: given an original transaction_id, find
        log lines that show the payment/order/ledger mismatch.

        Hop 1: query with original txn_id (only payment has this)
        Hop 2: query with the discovered request_id (all 3 services)
        Verify: both original and corrupted txn_id appear in Hop 2
        """
        from agent.grafana_adapter import GrafanaAdapter

        adapter = GrafanaAdapter(
            command=[
                "docker", "exec", "-i", MCP_GRAFANA_CONTAINER,
                "/app/mcp-grafana", "-t", "stdio", "-disable-proxied",
            ],
        )

        known_request_id = "661f961b-9101-4249-8ccc-b66ce7c0488b"
        original_txn_id = "334374682734104576"  # What Python generated
        corrupted_txn_id = "334374682734104600"  # What JavaScript received

        # HOP 2 (the main one — request_id is the join key across services)
        result = adapter.call_tool(
            "grafana_query_loki",
            {
                "datasourceUid": "loki",
                "logql": f'{{job="demo-services"}} |= "{known_request_id}"',
                "limit": 50,
            },
        )

        # The bug: both original and corrupted values should appear
        # (proving the bug is visible in the logs)
        assert original_txn_id in result, (
            f"Expected original txn_id in result, got: {result[:500]}"
        )
        assert corrupted_txn_id in result, (
            f"Expected corrupted txn_id in result, got: {result[:500]}"
        )
        # And they should be DIFFERENT (proving this is a real bug)
        assert original_txn_id != corrupted_txn_id, (
            "Test fixture error: original and corrupted txn_ids must differ"
        )
