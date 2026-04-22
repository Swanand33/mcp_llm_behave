# mcp-llm-behave

MCP server exposing [llm-behave](https://pypi.org/project/llm-behave/) behavioral regression testing as callable tools inside Claude Desktop, Claude Code, and any MCP-compatible client.

Runs **offline** — no API calls, no external services. Uses sentence-transformers for embedding-based similarity.

---

## Tools

| Tool | What it does |
|------|-------------|
| `run_behavior_test` | Assert that a model output matches an expected behavior description |
| `compare_outputs` | Detect semantic drift between a baseline and a new LLM output |
| `list_builtin_behaviors` | Browse the built-in behavioral checks shipped with llm-behave |

---

## Quickstart — Claude Desktop

Add to your `claude_desktop_config.json` (no install needed, `uvx` handles it):

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

Config file location:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

Restart Claude Desktop after editing. The first run downloads the sentence-transformers model (~80 MB) once and caches it.

---

## Quickstart — Claude Code (CLI)

```bash
claude mcp add mcp-llm-behave uvx mcp-llm-behave
```

---

## Install via pip / uv

```bash
pip install mcp-llm-behave
# or
uv add mcp-llm-behave
```

Run the server directly:

```bash
mcp-llm-behave
```

---

## Tool reference

### `run_behavior_test`

Check whether a model output semantically satisfies an expected behavior.

**Arguments**

| Name | Type | Description |
|------|------|-------------|
| `prompt` | str | The original prompt sent to the LLM (used for context/logging) |
| `expected_behavior` | str | Plain-language description of what the output should do |
| `model_output` | str | The actual text returned by the LLM |

**Returns**

```json
{
  "score": 0.82,
  "passed": true,
  "threshold": 0.45
}
```

---

### `compare_outputs`

Detect semantic drift between a known-good baseline and a new output. Useful in CI after prompt or model changes.

**Arguments**

| Name | Type | Description |
|------|------|-------------|
| `baseline` | str | The reference / previous LLM output |
| `candidate` | str | The new LLM output to compare |

**Returns**

```json
{
  "similarity_score": 0.91,
  "drift_detected": false,
  "interpretation": "Outputs are nearly identical — no drift."
}
```

---

### `list_builtin_behaviors`

Returns the catalog of pre-defined behavioral checks available in llm-behave, with method signatures and descriptions.

**Returns** — list of objects with `name`, `method`, and `description` keys.

---

## Requirements

- Python 3.10+
- No API keys needed
- ~80 MB disk for the sentence-transformers model (downloaded once on first run)

---

## Development

```bash
git clone https://github.com/Swanand33/mcp_llm_behave
cd mcp-llm-behave
uv sync
uv run pytest
```

---

## License

MIT — see [LICENSE](LICENSE).
