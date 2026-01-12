# Deep Research MCP Server

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP](https://img.shields.io/badge/MCP-compatible-green.svg)](https://modelcontextprotocol.io)

An [MCP (Model Context Protocol)](https://modelcontextprotocol.io) server that brings deep research capabilities to AI assistants like GitHub Copilot and Claude. Powered by Azure AI Foundry.

## Overview

Deep Research extends AI assistants with the ability to perform thorough, multi-source research on any topic. It exposes three tools with different depth/speed tradeoffs:

| Tool | Use Case | Duration |
|------|----------|----------|
| **`ask`** | Quick factual questions with web grounding | ~10 seconds |
| **`web_research`** | In-depth research crawling 10-25 sources | 2-9 minutes |
| **`deep_research`** | Exhaustive analysis for high-stakes decisions | 5-30 minutes |

## Quick Start

### Prerequisites

- [uv](https://docs.astral.sh/uv/) package manager
- Azure AI Foundry endpoints configured
- Authentication: Azure CLI (`az login`) or Service Principal credentials

### Installation

No installation required! Use `uvx` to run directly:

```bash
uvx --from git+https://github.com/renepajta/deep-research.git deep-research
```

### VS Code / GitHub Copilot

Add to `.vscode/mcp.json` in your workspace:

#### Minimal (uses Azure CLI auth)

```json
{
  "servers": {
    "deep-research": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "git+https://github.com/renepajta/deep-research.git", "deep-research"],
      "env": {
        "PYTHONUNBUFFERED": "1",
        "AI_FOUNDRY_ENDPOINT": "https://your-resource.openai.azure.com/openai/v1/",
        "MODEL_DEPLOYMENT_NAME": "gpt-5.2",
        "AI_FOUNDRY_DEEP_RESEARCH_ENDPOINT": "https://your-research.openai.azure.com/openai/v1/",
        "DEEP_RESEARCH_MODEL_DEPLOYMENT_NAME": "o3-deep-research"
      }
    }
  }
}
```

#### With Service Principal (no az login required)

```json
{
  "servers": {
    "deep-research": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "git+https://github.com/renepajta/deep-research.git", "deep-research"],
      "env": {
        "PYTHONUNBUFFERED": "1",
        "AI_FOUNDRY_ENDPOINT": "https://your-resource.openai.azure.com/openai/v1/",
        "MODEL_DEPLOYMENT_NAME": "gpt-5.2",
        "AI_FOUNDRY_DEEP_RESEARCH_ENDPOINT": "https://your-research.openai.azure.com/openai/v1/",
        "DEEP_RESEARCH_MODEL_DEPLOYMENT_NAME": "o3-deep-research",
        "AZURE_TENANT_ID": "your-tenant-id",
        "AZURE_CLIENT_ID": "your-client-id",
        "AZURE_CLIENT_SECRET": "your-client-secret"
      }
    }
  }
}
```

For global availability across all workspaces, add to your VS Code **User Settings** instead.

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "deep-research": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/renepajta/deep-research.git", "deep-research"],
      "env": {
        "AI_FOUNDRY_ENDPOINT": "https://your-resource.openai.azure.com/openai/v1/",
        "MODEL_DEPLOYMENT_NAME": "gpt-5.2",
        "AI_FOUNDRY_DEEP_RESEARCH_ENDPOINT": "https://your-research.openai.azure.com/openai/v1/",
        "DEEP_RESEARCH_MODEL_DEPLOYMENT_NAME": "o3-deep-research"
      }
    }
  }
}
```

## Configuration

### Environment Variables

Create a `.env` file or set these environment variables:

```env
# Azure AI Foundry - Primary endpoint
AI_FOUNDRY_ENDPOINT=https://your-resource.openai.azure.com/openai/v1/
MODEL_DEPLOYMENT_NAME=gpt-5.2

# Azure AI Foundry - Deep research endpoint
AI_FOUNDRY_DEEP_RESEARCH_ENDPOINT=https://your-research-resource.openai.azure.com/openai/v1/
DEEP_RESEARCH_MODEL_DEPLOYMENT_NAME=o3-deep-research
```

### Authentication

The server supports multiple authentication methods (in priority order):

#### Option 1: Service Principal (Recommended for CI/CD)

Set these environment variables:

```env
AZURE_TENANT_ID=your-tenant-id
AZURE_CLIENT_ID=your-client-id
AZURE_CLIENT_SECRET=your-client-secret
```

**Create a Service Principal with required permissions:**

```bash
# Create SPN and assign Cognitive Services User role to your AI Foundry resources
az ad sp create-for-rbac --name "deep-research" --role "Cognitive Services User" \
  --scopes \
    /subscriptions/{sub}/resourceGroups/{rg}/providers/Microsoft.CognitiveServices/accounts/{primary-resource} \
    /subscriptions/{sub}/resourceGroups/{rg}/providers/Microsoft.CognitiveServices/accounts/{research-resource}
```

**Required Azure RBAC permissions:**

| Role | Scope | Purpose |
|------|-------|--------|
| `Cognitive Services User` | AI Foundry primary resource | Access to gpt-5.2 model |
| `Cognitive Services User` | AI Foundry research resource | Access to o3-deep-research model |

#### Option 2: Azure CLI (Default for local development)

```bash
az login
```

#### Option 3: Managed Identity (Azure hosted environments)

Automatically used when running in Azure Container Apps, Azure VMs, etc.

## Tools

### `ask`

Quick, conversational answers with real-time web search. Best for factual questions.

```json
{ "question": "What are the key features of Python 3.13?" }
```

### `web_research`

Iterative research that searches, reads, and refines across 10-25 sources. **Recommended default for research questions.**

```json
{ "topic": "Current state of quantum computing in 2026" }
```

### `deep_research`

Comprehensive research using the o3-deep-research model. Reserved for high-stakes decisions requiring exhaustive verification.

```json
{
  "topic": "Competitive analysis of enterprise AI platforms",
  "skip_synthesis": false
}
```

> ⚠️ **Note**: Deep research is slow (5-30 min) and expensive. Use `ask` or `web_research` for most queries.

## Development

```bash
# Clone the repository
git clone https://github.com/renepajta/deep-research.git
cd deep-research

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## Architecture

```
deep-research/
├── src/deep_research/
│   ├── __init__.py      # Package exports
│   └── server.py        # MCP server implementation
├── tests/               # Test suite
├── pyproject.toml       # Package configuration
└── README.md
```

## How It Works

1. **MCP Protocol**: The server communicates via stdio using the Model Context Protocol
2. **Azure AI Foundry**: Leverages Azure's AI infrastructure for reliable, enterprise-grade inference
3. **Tool Selection**: AI assistants automatically choose the right tool based on query complexity

## License

MIT
