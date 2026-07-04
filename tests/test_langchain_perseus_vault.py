"""Tests for langchain-perseus-vault.

The unit tests monkeypatch ``subprocess.Popen`` with an in-process fake that
speaks JSON-RPC 2.0 over fake stdin/stdout pipes and models the Perseus Vault
remember/recall behavior, so they run with no real ``perseus-vault`` binary.
They exercise the real RPC, threading, tool, and retriever code paths.

A final smoke test runs a real remember->recall round trip if (and only if) a
``perseus-vault`` binary is discoverable; otherwise it is skipped.
"""

from __future__ import annotations

import json
import queue
import shutil

import pytest

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import StructuredTool

import langchain_perseus_vault.client as client_mod
from langchain_perseus_vault import (
    PerseusVaultClient,
    PerseusVaultError,
    PerseusVaultRetriever,
    create_perseus_vault_tools,
)


# ── Fake Perseus Vault MCP stdio server ──────────────────────────────────────


class _FakeStdin:
    def __init__(self, on_line):
        self._on_line = on_line

    def write(self, s):
        for line in s.splitlines():
            if line.strip():
                self._on_line(line)

    def flush(self):
        pass

    def close(self):
        pass


class _FakeStdout:
    """Blocking, iterable line source fed by the fake server."""

    def __init__(self):
        self._q = queue.Queue()

    def put(self, line):
        self._q.put(line)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._q.get()
        if item is None:
            raise StopIteration
        return item

    def close(self):
        self._q.put(None)


class FakePerseusVault:
    """Minimal Popen-compatible fake of the Perseus Vault MCP stdio server.

    Models remember as an upsert into ``self.store`` and recall as a naive
    OR-of-terms substring match over stored text, returning Perseus-Vault-shaped
    items.
    """

    def __init__(self, *, answer_tools=True):
        self.store: dict[tuple, dict] = {}  # (category, key) -> item
        self._counter = 0
        self.stdout = _FakeStdout()
        self.stdin = _FakeStdin(self._handle)
        self._alive = True
        self._answer_tools = answer_tools

    # Popen-compatible surface -------------------------------------------------
    def terminate(self):
        self._alive = False
        self.stdout.close()

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self._alive = False
        self.stdout.close()

    # JSON-RPC handling --------------------------------------------------------
    def _reply(self, rid, result):
        self.stdout.put(json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}))

    def _handle(self, line):
        req = json.loads(line)
        rid = req.get("id")
        method = req.get("method")
        if rid is None:
            return  # notification, no response
        if method == "initialize":
            self._reply(rid, {"protocolVersion": "2024-11-05", "capabilities": {}})
            return
        if method == "tools/call":
            if not self._answer_tools:
                return  # simulate a hang -> RPC timeout
            self._handle_tool(rid, req["params"])
            return
        self._reply(rid, {})

    def _handle_tool(self, rid, params):
        name = params["name"]
        args = params["arguments"]
        if name == "perseus_vault_remember":
            self._counter += 1
            ckey = (args["category"], args["key"])
            existed = ckey in self.store
            body = args.get("body_json", "{}")
            try:
                text = json.loads(body).get("text", "")
            except json.JSONDecodeError:
                text = ""
            self.store[ckey] = {
                "id": f"mem-{self._counter}",
                "category": args["category"],
                "key": args["key"],
                "text": text,
                "body_json": body,
                "tags": args.get("tags", []),
                "decay_score": 0.5,
                "created_at_unix_ms": 1000 + self._counter,
            }
            sc = {
                "action": "updated" if existed else "created",
                "category": args["category"],
                "key": args["key"],
                "id": self.store[ckey]["id"],
            }
            self._mcp_reply(rid, sc)
        elif name == "perseus_vault_recall":
            query = args.get("query", "").lower()
            terms = [t for t in query.split() if t]
            cat = args.get("category")
            limit = args.get("limit", 5)
            items = []
            for (c, _k), item in self.store.items():
                if cat is not None and c != cat:
                    continue
                hay = item["text"].lower()
                if any(t in hay for t in terms):
                    items.append(item)
            items = items[:limit]
            self._mcp_reply(rid, {"items": items, "total": len(items)})
        else:
            self._mcp_reply(rid, {})

    def _mcp_reply(self, rid, structured):
        # Mirror real MCP tools/call result shape.
        self._reply(
            rid,
            {
                "content": [{"type": "text", "text": json.dumps(structured)}],
                "structuredContent": structured,
            },
        )


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_client(monkeypatch, tmp_path):
    """A PerseusVaultClient wired to an in-process FakePerseusVault (no binary)."""
    fake = FakePerseusVault()

    def fake_popen(argv, **kwargs):
        return fake

    monkeypatch.setattr(client_mod.subprocess, "Popen", fake_popen)
    # Make binary resolution succeed without a real executable.
    monkeypatch.setattr(client_mod.shutil, "which", lambda name: "/fake/perseus-vault")

    client = PerseusVaultClient(db_path=str(tmp_path / "mimir.db"))
    client._fake = fake  # for assertions
    yield client
    client.close()


# ── client tests ─────────────────────────────────────────────────────────────


def test_binary_not_found(monkeypatch, tmp_path):
    monkeypatch.setattr(client_mod.shutil, "which", lambda name: None)
    with pytest.raises(PerseusVaultError, match="perseus-vault binary not found"):
        PerseusVaultClient(
            db_path=str(tmp_path / "x.db"),
            perseus_vault_binary="definitely-missing",
        )


def test_remember_then_recall(fake_client):
    r = fake_client.remember("The capital of France is Paris.", key="k1")
    assert r["action"] == "created"

    items = fake_client.recall("capital France")
    assert len(items) == 1
    assert "Paris" in items[0]["text"]


def test_remember_is_idempotent_upsert(fake_client):
    fake_client.remember("first", key="dup")
    r2 = fake_client.remember("second", key="dup")
    assert r2["action"] == "updated"
    assert len(fake_client._fake.store) == 1


def test_recall_respects_limit(fake_client):
    for i in range(5):
        fake_client.remember(f"alpha memory number {i}", key=f"k{i}")
    items = fake_client.recall("alpha", limit=2)
    assert len(items) == 2


def test_recall_no_match_returns_empty(fake_client):
    fake_client.remember("something unrelated", key="k1")
    assert fake_client.recall("nonexistent zebra") == []


def test_rpc_timeout(monkeypatch, tmp_path):
    fake = FakePerseusVault(answer_tools=True)
    monkeypatch.setattr(client_mod.subprocess, "Popen", lambda *a, **k: fake)
    monkeypatch.setattr(client_mod.shutil, "which", lambda name: "/fake/perseus-vault")
    client = PerseusVaultClient(db_path=str(tmp_path / "m.db"), timeout_s=0.3)
    # Flip the fake to stop answering tool calls -> the next call must time out.
    fake._answer_tools = False
    with pytest.raises(PerseusVaultError, match="timed out"):
        client.recall("anything")
    client.close()


# ── tools tests ──────────────────────────────────────────────────────────────


def test_create_perseus_vault_tools_shape(fake_client):
    tools = create_perseus_vault_tools(fake_client)
    assert len(tools) == 2
    assert all(isinstance(t, StructuredTool) for t in tools)
    names = {t.name for t in tools}
    assert names == {"perseus_vault_remember", "perseus_vault_recall"}


def test_remember_tool_invoke(fake_client):
    remember, recall = create_perseus_vault_tools(fake_client)
    out = remember.invoke({"text": "I love Rust.", "tags": ["pref"]})
    assert "created" in out or "stored" in out
    # And it is recallable through the recall tool.
    recalled = recall.invoke({"query": "Rust"})
    assert "Rust" in recalled


def test_recall_tool_no_results(fake_client):
    _, recall = create_perseus_vault_tools(fake_client)
    out = recall.invoke({"query": "nothing here"})
    assert out == "No relevant memories found."


def test_tool_args_schema_present(fake_client):
    remember, recall = create_perseus_vault_tools(fake_client)
    assert "text" in remember.args
    assert "query" in recall.args


# ── retriever tests ──────────────────────────────────────────────────────────


def test_retriever_is_base_retriever(fake_client):
    r = PerseusVaultRetriever(client=fake_client)
    assert isinstance(r, BaseRetriever)


def test_retriever_returns_documents(fake_client):
    fake_client.remember("The capital of France is Paris.", key="k1")
    retriever = PerseusVaultRetriever(client=fake_client, k=3)
    docs = retriever.invoke("What is the capital of France?")
    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert "Paris" in docs[0].page_content
    assert docs[0].metadata["key"] == "k1"
    assert docs[0].metadata["category"] == "langchain-memory"


def test_retriever_empty(fake_client):
    retriever = PerseusVaultRetriever(client=fake_client)
    assert retriever.invoke("zebra unicorn") == []


def test_retriever_category_scoping(fake_client):
    fake_client.remember("scoped fact apple", category="catA", key="a")
    fake_client.remember("other fact apple", category="catB", key="b")
    retriever = PerseusVaultRetriever(client=fake_client, category="catA")
    docs = retriever.invoke("apple")
    assert len(docs) == 1
    assert docs[0].metadata["category"] == "catA"


# ── real binary smoke test (skipped if perseus-vault is unavailable) ─────────


def _find_perseus_vault():
    return (
        shutil.which("perseus-vault")
        or shutil.which("perseus-vault.exe")
        # `mimir` is only a compat symlink; accept it if present as a fallback.
        or shutil.which("mimir")
        or shutil.which("mimir.exe")
    )


@pytest.mark.skipif(
    _find_perseus_vault() is None, reason="no real perseus-vault binary on PATH"
)
def test_real_roundtrip(tmp_path):
    """Real remember -> recall against an actual perseus-vault subprocess."""
    binary = _find_perseus_vault()
    client = PerseusVaultClient(
        db_path=str(tmp_path / "real.db"), perseus_vault_binary=binary
    )
    try:
        client.remember(
            "The capital of France is Paris.", category="lc-smoke", key="smoke1"
        )
        items = client.recall("capital France", category="lc-smoke")
        assert any("Paris" in (it.get("text") or "") for it in items)

        retriever = PerseusVaultRetriever(client=client, category="lc-smoke")
        docs = retriever.invoke("capital of France")
        assert any("Paris" in d.page_content for d in docs)
    finally:
        client.close()
