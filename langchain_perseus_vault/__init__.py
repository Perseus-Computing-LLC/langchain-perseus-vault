"""langchain-perseus-vault — Perseus Vault persistent memory for LangChain.

Perseus Vault (github.com/Perseus-Computing-LLC/perseus-vault) is an
open-source (MIT), local-first, encrypted persistent memory engine that speaks
MCP JSON-RPC over stdio. This package exposes it to LangChain via the modern
``langchain-core`` interfaces:

- :class:`PerseusVaultClient` — low-level stdio client for the ``perseus-vault`` binary.
- :func:`create_perseus_vault_tools` — ``StructuredTool``s (remember / recall) for agents.
- :class:`PerseusVaultRetriever` — a ``BaseRetriever`` for RAG chains.

Requirements:
    A ``perseus-vault`` binary must be on ``$PATH`` or passed via
    ``perseus_vault_binary=``.
    Download from https://github.com/Perseus-Computing-LLC/perseus-vault/releases
"""

from .client import PerseusVaultClient, PerseusVaultError
from .integration import (
    PerseusVaultRetriever,
    create_perseus_vault_tools,
    create_recall_tool,
    create_remember_tool,
)

__version__ = "0.1.0"

__all__ = [
    "PerseusVaultClient",
    "PerseusVaultError",
    "PerseusVaultRetriever",
    "create_perseus_vault_tools",
    "create_remember_tool",
    "create_recall_tool",
]
