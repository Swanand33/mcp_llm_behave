"""End-to-end integration tests — exercises the full MCP protocol stack.

Uses FastMCP's in-memory Client transport so no subprocess is needed.
These tests verify the protocol handshake, tool listing, and tool invocation
exactly as Claude Desktop would experience them.
"""

import pytest
from fastmcp import Client

from mcp_llm_behave.server import mcp

pytestmark = pytest.mark.anyio(backends=["asyncio"])


# --- Protocol handshake ---

async def test_protocol_lists_exactly_three_tools():
    """MCP list_tools response must contain exactly the 3 registered tools."""
    async with Client(mcp) as client:
        tools = await client.list_tools()
        names = {t.name for t in tools}
        assert names == {"run_behavior_test", "compare_outputs", "list_builtin_behaviors"}


async def test_protocol_tools_have_descriptions():
    """Every tool must have a non-empty description (Claude uses this to decide when to call it)."""
    async with Client(mcp) as client:
        tools = await client.list_tools()
        for tool in tools:
            assert tool.description, f"Tool '{tool.name}' has no description"


# --- run_behavior_test ---

async def test_e2e_run_behavior_test_pass():
    async with Client(mcp) as client:
        result = await client.call_tool(
            "run_behavior_test",
            {
                "prompt": "Explain the refund policy.",
                "expected_behavior": "mentions refund",
                "model_output": "You can get a full refund within 30 days of purchase.",
            },
        )
        assert not result.is_error
        data = result.data
        assert data["passed"] is True
        assert 0.0 <= data["score"] <= 1.0
        assert data["threshold"] == 0.45


async def test_e2e_run_behavior_test_fail():
    async with Client(mcp) as client:
        result = await client.call_tool(
            "run_behavior_test",
            {
                "prompt": "Tell me about quantum physics.",
                "expected_behavior": "provides cooking instructions",
                "model_output": "Quantum entanglement is a phenomenon where particles become correlated.",
            },
        )
        assert not result.is_error
        assert result.data["passed"] is False


# --- compare_outputs ---

async def test_e2e_compare_outputs_no_drift():
    async with Client(mcp) as client:
        result = await client.call_tool(
            "compare_outputs",
            {
                "baseline": "The capital of France is Paris.",
                "candidate": "Paris is the capital city of France.",
            },
        )
        assert not result.is_error
        data = result.data
        assert data["drift_detected"] is False
        assert data["similarity_score"] > 0.8
        assert isinstance(data["interpretation"], str)


async def test_e2e_compare_outputs_drift():
    async with Client(mcp) as client:
        result = await client.call_tool(
            "compare_outputs",
            {
                "baseline": "The capital of France is Paris.",
                "candidate": "Python is used for machine learning and data science.",
            },
        )
        assert not result.is_error
        assert result.data["drift_detected"] is True


# --- list_builtin_behaviors ---

async def test_e2e_list_builtin_behaviors():
    async with Client(mcp) as client:
        result = await client.call_tool("list_builtin_behaviors", {})
        assert not result.is_error
        behaviors = result.data
        assert isinstance(behaviors, list)
        assert len(behaviors) >= 5
        names = {b["name"] for b in behaviors}
        assert "mentions" in names
        assert "contradicts" in names
