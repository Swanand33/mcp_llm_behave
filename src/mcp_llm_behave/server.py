"""FastMCP app — tool registration and server entry point."""

from fastmcp import FastMCP

from mcp_llm_behave.tools import compare_outputs, list_builtin_behaviors, run_behavior_test

mcp = FastMCP(
    "mcp-llm-behave",
    instructions="Behavioral regression testing for LLM prompts using offline embedding similarity.",
)

mcp.tool()(run_behavior_test)
mcp.tool()(compare_outputs)
mcp.tool()(list_builtin_behaviors)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
