"""Deep Research MCP Server.

An MCP server for deep research powered by Azure AI Foundry, providing:
- ask: Quick answers with real-time web search
- web_research: Iterative research crawling multiple sources
- deep_research: Comprehensive multi-agent research
"""

__version__ = "0.1.0"

from .server import main, mcp

__all__ = ["mcp", "main", "__version__"]
