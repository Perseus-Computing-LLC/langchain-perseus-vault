"""langchain-mimir — Mimir persistent memory for LangChain.

Mimir (github.com/Perseus-Computing-LLC/mimir) is an open-source (MIT),
local-first, encrypted persistent memory engine that speaks MCP JSON-RPC over
stdio. This package exposes it to LangChain via the modern ``langchain-core``
interfaces:

- :class:`MimirClient` — low-level stdio client for the ``mimir`` binary.
- :func:`create_mimir_tools` — ``StructuredTool``s (remember / recall) for agents.
- :class:`MimirRetriever` — a ``BaseRetriever`` for RAG chains.

Requirements:
    A ``mimir`` binary must be on ``$PATH`` or passed via ``mimir_binary=``.
    Download from https://github.com/Perseus-Computing-LLC/mimir/releases
"""

from .client import MimirClient, MimirError
from .integration import (
    MimirRetriever,
    create_mimir_tools,
    create_recall_tool,
    create_remember_tool,
)

__version__ = "0.1.0"

__all__ = [
    "MimirClient",
    "MimirError",
    "MimirRetriever",
    "create_mimir_tools",
    "create_remember_tool",
    "create_recall_tool",
]
