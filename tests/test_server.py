"""Tests confirming the FastMCP server registers all 3 tools."""

from mcp_llm_behave.server import mcp


def test_server_has_three_tools():
    """FastMCP server must expose exactly 3 tools — no more, no less."""
    tool_names = {
        k.removeprefix("tool:").split("@")[0]
        for k in mcp.local_provider._components
        if k.startswith("tool:")
    }
    assert tool_names == {"run_behavior_test", "compare_outputs", "list_builtin_behaviors"}


def test_server_name():
    assert mcp.name == "mcp-llm-behave"
