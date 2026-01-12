# Deep Research MCP Server

An MCP server for deep research powered by Azure AI Foundry.

## Project Structure

```
deep-research/
├── src/deep_research/
│   ├── __init__.py      # Package exports
│   └── server.py        # MCP server with tools
├── tests/
│   └── test_server.py   # Integration tests
├── pyproject.toml       # Package configuration
├── README.md            # Documentation
├── LICENSE              # MIT license
└── .env.example         # Environment template
```

## Tools Exposed

| Tool | Description | Speed |
|------|-------------|-------|
| `ask` | Quick answers with real-time web search | Fast (~10s) |
| `web_research` | Iterative research crawling 10-25 sources | Medium (2-9 min) |
| `deep_research` | Exhaustive multi-agent research | Slow (5-30 min) |

## Running the Server

```bash
# Using uvx (recommended)
uvx --from git+https://github.com/renepajta/deep-research.git deep-research

# Local development
pip install -e .
deep-research
```

## Configuration

Required environment variables:
- `AI_FOUNDRY_ENDPOINT` - Azure AI Foundry endpoint
- `MODEL_DEPLOYMENT_NAME` - Model deployment name (e.g., gpt-5.2)
- `AI_FOUNDRY_DEEP_RESEARCH_ENDPOINT` - Deep research endpoint
- `DEEP_RESEARCH_MODEL_DEPLOYMENT_NAME` - Deep research model (e.g., o3-deep-research)

## Authentication

Supports three methods (in priority order):

1. **Service Principal**: Set `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`
2. **Managed Identity**: Automatic in Azure environments
3. **Azure CLI**: Run `az login` (default for local dev)
