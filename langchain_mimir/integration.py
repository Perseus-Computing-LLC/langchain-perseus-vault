"""LangChain integration surface for the Mimir memory engine.

Two complementary, current-recommended ``langchain-core`` surfaces are exposed:

1. **Tools** — :func:`create_mimir_tools` returns a pair of ``StructuredTool``s
   (``mimir_remember`` / ``mimir_recall``) that an agent can call to persist and
   retrieve long-term memory. Tool-calling is the modern LangChain pattern for
   giving an agent agency over its own memory (the legacy ``Memory`` /
   ``ConversationBufferMemory`` classes are deprecated).

2. **Retriever** — :class:`MimirRetriever` is a ``BaseRetriever`` that turns a
   query into ``Document`` objects, for drop-in use in RAG chains and anywhere a
   LangChain retriever is accepted (``.invoke(query)``).

Both are thin wrappers over :class:`langchain_mimir.client.MimirClient`.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .client import MimirClient

__all__ = [
    "MimirRetriever",
    "create_mimir_tools",
    "create_remember_tool",
    "create_recall_tool",
]


# ── helpers ──────────────────────────────────────────────────────────────────


def _item_to_text(item: dict) -> str:
    """Extracts the best human-readable text from a Mimir recall item."""
    text = item.get("text")
    if text:
        return text
    body = item.get("body_json")
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            return body
    if isinstance(body, dict):
        return body.get("text") or body.get("content") or json.dumps(body)
    return ""


def _item_to_document(item: dict) -> Document:
    """Converts a raw Mimir recall item into a LangChain ``Document``."""
    return Document(
        page_content=_item_to_text(item),
        metadata={
            "id": item.get("id"),
            "category": item.get("category"),
            "key": item.get("key"),
            "tags": item.get("tags", []),
            "decay_score": item.get("decay_score"),
            "created_at_unix_ms": item.get("created_at_unix_ms"),
        },
    )


# ── retriever ────────────────────────────────────────────────────────────────


class MimirRetriever(BaseRetriever):
    """Retriever backed by Mimir's FTS5 keyword search.

    Example::

        from langchain_mimir import MimirClient, MimirRetriever

        client = MimirClient(db_path="~/.langchain/mimir.db")
        client.remember("The capital of France is Paris.")

        retriever = MimirRetriever(client=client)
        docs = retriever.invoke("What is the capital of France?")
    """

    client: MimirClient
    k: int = 5
    category: str | None = None

    # MimirClient is an arbitrary (non-pydantic) type.
    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        items = self.client.recall(query, limit=self.k, category=self.category)
        return [_item_to_document(it) for it in items]


# ── tools ────────────────────────────────────────────────────────────────────


class _RememberInput(BaseModel):
    text: str = Field(description="The fact or memory to store for later recall.")
    tags: list[str] | None = Field(
        default=None, description="Optional tags to label this memory."
    )


class _RecallInput(BaseModel):
    query: str = Field(description="What to search the memory for.")
    limit: int = Field(default=5, description="Max number of memories to return.")


def create_remember_tool(
    client: MimirClient,
    *,
    category: str = "langchain-memory",
) -> StructuredTool:
    """Builds a ``StructuredTool`` that stores a memory in Mimir."""

    def _remember(text: str, tags: list[str] | None = None) -> str:
        result = client.remember(text, category=category, tags=tags)
        action = result.get("action", "stored")
        key = result.get("key", "")
        return f"Memory {action} (key={key})."

    return StructuredTool.from_function(
        func=_remember,
        name="mimir_remember",
        description=(
            "Store a fact or memory in long-term persistent memory so it can be "
            "recalled in future conversations. Use this whenever the user shares "
            "durable information worth remembering."
        ),
        args_schema=_RememberInput,
    )


def create_recall_tool(
    client: MimirClient,
    *,
    category: str | None = "langchain-memory",
) -> StructuredTool:
    """Builds a ``StructuredTool`` that searches Mimir's memory."""

    def _recall(query: str, limit: int = 5) -> str:
        items = client.recall(query, limit=limit, category=category)
        if not items:
            return "No relevant memories found."
        lines = [f"- {_item_to_text(it)}" for it in items if _item_to_text(it)]
        return "\n".join(lines) if lines else "No relevant memories found."

    return StructuredTool.from_function(
        func=_recall,
        name="mimir_recall",
        description=(
            "Search long-term persistent memory for facts relevant to a query. "
            "Use this to recall things the user told you in past conversations."
        ),
        args_schema=_RecallInput,
    )


def create_mimir_tools(
    client: MimirClient,
    *,
    category: str = "langchain-memory",
) -> list[StructuredTool]:
    """Returns ``[remember_tool, recall_tool]`` bound to ``client``.

    Pass the result to any LangChain agent / ``bind_tools`` call to give the
    model agency over its own persistent memory.

    Args:
        client: An initialized :class:`MimirClient`.
        category: The Mimir category (namespace) used for both tools.
    """
    return [
        create_remember_tool(client, category=category),
        create_recall_tool(client, category=category),
    ]
