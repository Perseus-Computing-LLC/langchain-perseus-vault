# langchain-perseus-vault

Persistent, local-first, encrypted memory for [LangChain](https://www.langchain.com/),
backed by [Perseus Vault](https://github.com/Perseus-Computing-LLC/perseus-vault) (formerly "Mimir"/"Mneme") — an open-source
(MIT) memory engine with FTS5 + dense hybrid search and optional AES-256-GCM
encryption, exposed over the Model Context Protocol (MCP) stdio transport.

It gives a LangChain agent durable memory that survives across runs and
processes, stored in a single local SQLite file you control — no external
service, no cloud.

## What you get

This package wraps Perseus Vault using the modern `langchain-core` interfaces:

- **`create_mimir_tools(client)`** — a pair of `StructuredTool`s
  (`mimir_remember` / `mimir_recall`) you give to an agent so it can manage its
  own long-term memory via tool calls. This is the current-recommended LangChain
  pattern (the legacy `Memory` / `ConversationBufferMemory` classes are
  deprecated).
- **`MimirRetriever`** — a `BaseRetriever` returning `Document`s, for drop-in use
  in RAG chains and anywhere LangChain accepts a retriever (`.invoke(query)`).
- **`MimirClient`** — the low-level MCP stdio client, if you want direct access.

## Prerequisite: the `mimir` binary

This package talks to a local `mimir` executable via JSON-RPC over stdio. You
must have it installed:

- Download a release from
  <https://github.com/Perseus-Computing-LLC/perseus-vault/releases>, or build from source
  (`cargo build --release`), and put `mimir` on your `$PATH`.
- Or pass an absolute path: `MimirClient(mimir_binary="/path/to/mimir")`.

On Windows the binary may be named `mimir.exe`; ensure its directory is on
`PATH`, or pass the full path.

## Install

```bash
pip install langchain-perseus-vault
```

## Usage

### As agent tools

```python
from langchain_perseus_vault import MimirClient, create_mimir_tools

client = MimirClient(db_path="~/.langchain/mimir.db")
tools = create_mimir_tools(client)  # [mimir_remember, mimir_recall]

# Bind to any tool-calling model / agent:
from langchain.chat_models import init_chat_model

llm = init_chat_model("anthropic:claude-sonnet-4-5")
llm_with_memory = llm.bind_tools(tools)

resp = llm_with_memory.invoke("Remember that my favorite language is Rust.")
# ... the model will call mimir_remember; execute the tool call as usual.
```

### As a retriever

```python
from langchain_perseus_vault import MimirClient, MimirRetriever

client = MimirClient(db_path="~/.langchain/mimir.db")
client.remember("The capital of France is Paris.")

retriever = MimirRetriever(client=client, k=5)
docs = retriever.invoke("What is the capital of France?")
print(docs[0].page_content)  # -> "The capital of France is Paris."
```

### Direct client

```python
from langchain_perseus_vault import MimirClient

client = MimirClient(db_path="~/.langchain/mimir.db")
client.remember("Project deadline is July 15.", tags=["project", "deadline"])
items = client.recall("when is the deadline")
print(items[0]["text"])
```

## How it works

`MimirClient` spawns `mimir --db <path>` as a subprocess and speaks JSON-RPC 2.0
(MCP) over its stdin/stdout. A background reader thread and a lock make calls
thread-safe and timeout-bounded. Memories are stored via `mimir_remember` and
retrieved via `mimir_recall`.

## License

MIT © 2026 Perseus Computing LLC
