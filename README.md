# mcp-llm-behave

MCP server exposing [llm-behave](https://pypi.org/project/llm-behave/) behavioral regression testing as callable tools inside Claude Desktop, Claude Code, and any MCP-compatible client.

Runs **offline** — no API calls, no external services. Uses sentence-transformers for embedding-based similarity.

## Tools

| Tool | What it does |
|------|-------------|
| `run_behavior_test` | Assert that a model output matches an expected behavior |
| `compare_outputs` | Detect semantic drift between a baseline and new output |
| `list_builtin_behaviors` | Browse pre-defined behavioral checks from llm-behave |

## Install in Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mcp-llm-behave": {
      "command": "uvx",
      "args": ["mcp-llm-behave"]
    }
  }
}
```

See `examples/claude_desktop_config.json` for the full snippet.

## Install via pip

```bash
pip install mcp-llm-behave
```

## Requirements

- Python 3.10+
- No API keys needed

## License

MIT
