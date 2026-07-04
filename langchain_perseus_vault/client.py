"""Mimir MCP stdio client.

Spawns a local ``mimir`` binary and speaks JSON-RPC 2.0 over its stdin/stdout
(MCP stdio transport).  This is the low-level transport reused by the LangChain
tools and retriever; it has no LangChain dependency of its own.

The subprocess/JSON-RPC machinery here is adapted from the proven
``adk-mimir-memory`` client (github.com/Perseus-Computing-LLC/adk-mimir-memory):
a background reader thread pumps stdout lines into a queue so RPC calls can wait
with a timeout and correlate responses by id, and a lock serializes
request/response exchanges so they never interleave.

Requirements:
    A ``mimir`` binary must be on ``$PATH`` or passed explicitly via
    ``mimir_binary``.  Download from:
    https://github.com/Perseus-Computing-LLC/perseus-vault/releases
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import shutil
import subprocess
import threading
import time

__all__ = ["MimirClient", "MimirError"]


class MimirError(RuntimeError):
    """Raised when the Mimir subprocess errors, crashes, or times out."""


class MimirClient:
    """Thread-safe JSON-RPC client for a local Mimir MCP stdio server.

    Starts ``mimir --db <db_path>`` as a subprocess, performs the MCP
    ``initialize`` handshake, and exposes :meth:`call_tool` for invoking any of
    Mimir's MCP tools (``mimir_remember``, ``mimir_recall``, ...).

    Attributes:
        db_path: Filesystem path to the Mimir SQLite database.
    """

    def __init__(
        self,
        db_path: str = "~/.langchain/mimir.db",
        mimir_binary: str = "mimir",
        timeout_s: float = 30.0,
        encryption_key: str | None = None,
    ) -> None:
        """Initializes and starts the Mimir client.

        Args:
            db_path: Path to the Mimir database file.  Defaults to
                ``~/.langchain/mimir.db``.
            mimir_binary: Name or absolute path of the ``mimir`` executable.
                Defaults to ``mimir`` (resolved from ``$PATH``).
            timeout_s: Maximum time to wait for any single RPC response.
            encryption_key: Optional path to an AES-256-GCM key file; if given,
                passed to the binary via ``--encryption-key``.

        Raises:
            MimirError: If the binary cannot be found or the subprocess fails to
                start or complete the MCP handshake.
        """
        self.db_path = os.path.expanduser(db_path)
        self._timeout_s = timeout_s

        # Resolve the mimir binary.
        if os.path.isabs(mimir_binary) and os.path.exists(mimir_binary):
            self._mimir_binary = mimir_binary
        else:
            resolved = shutil.which(mimir_binary)
            if resolved is None and os.path.exists(mimir_binary):
                resolved = mimir_binary
            if resolved is None:
                raise MimirError(
                    f"mimir binary not found (looked for '{mimir_binary}'). "
                    "Install Perseus Vault from "
                    "https://github.com/Perseus-Computing-LLC/perseus-vault/releases "
                    "or pass the absolute path via mimir_binary=."
                )
            self._mimir_binary = resolved

        # Ensure the database directory exists.
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)

        argv = [self._mimir_binary, "--db", self.db_path]
        if encryption_key:
            argv += ["--encryption-key", encryption_key]

        # Start the MCP stdio subprocess. stderr is discarded: nothing drains
        # it, so a chatty server filling the OS pipe buffer would block on its
        # stderr write while we wait on stdout (a two-pipe deadlock).
        try:
            self._proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError as e:
            raise MimirError(f"failed to start mimir subprocess: {e}") from e

        self._lock = threading.Lock()
        self._request_id = 0
        self._closed = False

        # Background reader: pump stdout lines into a queue so _rpc can wait with
        # a timeout and correlate responses by id, rather than blocking forever
        # on a bare readline().
        self._recv: queue.Queue = queue.Queue()
        proc_stdout = self._proc.stdout

        def _pump() -> None:
            try:
                for line in proc_stdout:
                    self._recv.put(line)
            except Exception:
                pass
            finally:
                self._recv.put(None)  # EOF sentinel

        self._reader = threading.Thread(target=_pump, daemon=True)
        self._reader.start()

        # MCP handshake: initialize, then the required initialized notification
        # before any tools/call.
        self._rpc(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "langchain-perseus-vault", "version": "0.1.0"},
            },
        )
        self._notify("notifications/initialized", {})

        atexit.register(self.close)

    # ── lifecycle ──────────────────────────────────────────────────────────

    def close(self) -> None:
        """Terminates the Mimir subprocess.  Safe to call multiple times."""
        if self._closed:
            return
        self._closed = True
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass

    def __enter__(self) -> "MimirClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── JSON-RPC core ──────────────────────────────────────────────────────

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _rpc(self, method: str, params: object) -> dict:
        """Sends a JSON-RPC request and returns the ``result`` dict.

        The lock is held for the whole exchange so request/response pairs never
        interleave.  Replies with a non-matching id (notifications, stale
        replies) are skipped.

        Raises:
            MimirError: On transport failure, RPC error, or timeout.
        """
        with self._lock:
            req_id = self._next_id()
            req = {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": method,
                "params": params,
            }
            payload = json.dumps(req, default=str)
            try:
                self._proc.stdin.write(payload + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                raise MimirError(
                    f"mimir communication failed: {e}. The process may have crashed."
                ) from e

            deadline = time.monotonic() + self._timeout_s
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MimirError(
                        f"mimir RPC '{method}' timed out after {self._timeout_s}s."
                    )
                try:
                    raw = self._recv.get(timeout=remaining)
                except queue.Empty:
                    raise MimirError(
                        f"mimir RPC '{method}' timed out after {self._timeout_s}s."
                    )
                if raw is None:
                    raise MimirError(
                        "mimir closed its output stream (it may have crashed)."
                    )
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    resp = json.loads(raw)
                except json.JSONDecodeError:
                    continue  # non-JSON noise on stdout
                if resp.get("id") != req_id:
                    continue  # notification or a stale/other reply

                if "error" in resp:
                    err = resp["error"]
                    raise MimirError(
                        f"mimir RPC error [{err.get('code')}]: {err.get('message')}"
                    )
                return resp.get("result", {})

    def _notify(self, method: str, params: object) -> None:
        """Sends a JSON-RPC notification (no id, no response expected)."""
        payload = json.dumps({"jsonrpc": "2.0", "method": method, "params": params})
        with self._lock:
            try:
                self._proc.stdin.write(payload + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass

    # ── public API ─────────────────────────────────────────────────────────

    def call_tool(self, name: str, arguments: dict) -> dict:
        """Calls a Mimir MCP tool and returns its structured result.

        Args:
            name: The Mimir tool name (e.g. ``mimir_remember``).
            arguments: The tool's arguments dict.

        Returns:
            The tool's ``structuredContent`` if present, otherwise the parsed
            text content, otherwise ``{}``.
        """
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        # MCP result: {content: [{type: "text", text: "..."}], structuredContent: {...}}
        sc = result.get("structuredContent")
        if sc is not None:
            return sc
        content = result.get("content", [])
        if content:
            try:
                return json.loads(content[0].get("text", "{}"))
            except (json.JSONDecodeError, IndexError, KeyError, AttributeError):
                pass
        return {}

    # convenience wrappers ---------------------------------------------------

    def remember(
        self,
        text: str,
        *,
        category: str = "langchain-memory",
        key: str | None = None,
        tags: list[str] | None = None,
        extra_body: dict | None = None,
    ) -> dict:
        """Stores a memory.  Returns the ``mimir_remember`` result.

        Args:
            text: The natural-language memory content.
            category: Mimir category (namespace) for the entity.
            key: Stable key within the category; autogenerated if omitted.
                Reusing a key updates that entity (idempotent upsert).
            tags: Optional tags stored on the entity.
            extra_body: Extra fields merged into the stored JSON body.
        """
        if key is None:
            key = f"mem-{int(time.time() * 1000)}-{self._next_id()}"
        body = {"text": text}
        if extra_body:
            body.update(extra_body)
        args = {
            "category": category,
            "key": key,
            "body_json": json.dumps(body),
        }
        if tags:
            args["tags"] = tags
        return self.call_tool("mimir_remember", args)

    def recall(
        self,
        query: str,
        *,
        limit: int = 5,
        category: str | None = None,
    ) -> list[dict]:
        """Searches memories.  Returns the list of raw Mimir items.

        Args:
            query: Natural-language / keyword query (FTS5; terms OR'd).
            limit: Maximum number of items to return.
            category: Optional category to scope the search.
        """
        args: dict = {"query": query, "limit": limit}
        if category is not None:
            args["category"] = category
        result = self.call_tool("mimir_recall", args)
        items = result.get("items")
        if items is None:
            items = result.get("results", [])
        return items or []
