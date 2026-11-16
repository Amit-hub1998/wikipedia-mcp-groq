# Wikipedia MCP server + Groq chat client

A small Model Context Protocol setup:

- `server.py` — MCP server exposing English Wikipedia as five tools, over Streamable HTTP at `http://127.0.0.1:8000/mcp`.
- `client.py` — terminal chat client. It connects to the server, gives its tools to a Groq model, and lets the model look things up to answer your questions.

## Tools

| Tool | Arguments | Returns |
|---|---|---|
| `search_wikipedia` | `query`, `limit=5` | Matching article titles with snippets |
| `get_summary` | `title` | The article's introduction and URL |
| `get_article` | `title`, `max_chars=6000` | Plain-text article, truncated to `max_chars` |
| `summarize_article` | `title`, `focus=""` | Bullet-point summary written by the client's LLM (sampling) |
| `save_article` | `title` | Saves the full article as Markdown into the client's shared folder (roots) |

## Sampling

`summarize_article` needs an LLM, but the server has no API key. With MCP sampling the server asks the client to run a prompt on the client's model:

1. The chat model calls `summarize_article(title, focus)`.
2. On the server, the resolver `_summary_request` fetches the article and returns a `Sample(...)` request holding the prompt.
3. The SDK sends that to the client as `sampling/createMessage`.
4. On the client, `handle_sampling` runs the prompt on Groq and returns a `CreateMessageResult`.
5. The SDK injects the result into the tool's `summary` parameter, and the tool returns the text.

See [SAMPLING_EXAMPLES.md](SAMPLING_EXAMPLES.md) for real output from a run with sampling and a run without.

Two pieces make this work:

- **Server:** a tool parameter annotated `Annotated[CreateMessageResult, Resolve(_summary_request)]`. It is hidden from the calling model.
- **Client:** `Client(url, sampling_callback=...)`. Passing the callback also advertises the `sampling` capability; a client without it gets an error from this tool.

## Roots

Roots are folders the client tells the server it may work in. The server never chooses a location on your disk itself; it asks the client and stays inside what it is given.

`save_article` uses this to save a full article as a Markdown file:

1. The chat model calls `save_article(title)`.
2. On the server, the resolver `_client_roots` returns `ListRoots()`, and the SDK sends `roots/list` to the client.
3. On the client, `handle_list_roots` answers with one root: the `saved_articles/` folder next to `client.py` (as a `file://` URI).
4. The server fetches the article and writes `<Title>.md` into that folder, after checking the final path is still inside the root.

Example — `Use save_article to save the Wikipedia article on the Eiffel Tower.`

```
  [tool] save_article({"title": "Eiffel Tower"})
  [roots] server asked for roots; shared ...\MCP_server_client\saved_articles
  [log:info] Client root: saved_articles -> ...\MCP_server_client\saved_articles
  [log:info] Wrote 45144 characters to Eiffel_Tower.md
```

How it is wired:

- **Server:** a tool parameter annotated `Annotated[ListRootsResult, Resolve(_client_roots)]`, hidden from the model.
- **Client:** `Client(..., list_roots_callback=handle_list_roots)`. Passing the callback also advertises the `roots` capability. To share a different folder, change `SAVE_DIR` in `client.py`.

Roots are advisory: the protocol does not stop a server from writing elsewhere, so the server has to enforce the boundary itself, as `save_article` does. This only works here because server and client run on the same machine; `saved_articles/` is in `.gitignore`.

## Logs and progress notifications

While a tool is running, the server pushes status updates to the client. They are shown to the person at the terminal and are not part of the tool result, so the LLM never sees them.

| Client output | Server call | MCP message |
|---|---|---|
| `[log:info]`, `[log:warning]`, `[log:error]` | `ctx.info(...)`, `ctx.warning(...)`, `ctx.error(...)` | `notifications/message` |
| `[progress] 1/3 ...` | `ctx.report_progress(done, total, message)` | `notifications/progress` |

Example — `Use get_article to read the Photosynthesis article and tell me what the light-dependent reactions produce.`

```
  [tool] get_article({"max_chars": 2000, "title": "Photosynthesis"})
  [progress] 0/3 Requesting article from Wikipedia
  [log:info] Fetching the full article 'Photosynthesis'
  [progress] 1/3 Downloaded 49740 characters
  [log:info] Article is 49740 characters; truncating to 2000
  [progress] 2/3 Text prepared
  [progress] 3/3 Done
```

Example — `Use get_summary on the article titled Mercury.` (a warning):

```
  [tool] get_summary({"title": "Mercury"})
  [log:info] Fetching the introduction of 'Mercury'
  [log:info] Got 196 characters from 'Mercury'
  [log:warning] 'Mercury' is a disambiguation page
```

How it is wired:

- **Server:** any tool with a `ctx: Context` parameter can call `ctx.info` / `ctx.report_progress`. The parameter is filled by the SDK and hidden from the model. All tools log; `get_article` also reports progress.
- **Client, logs:** `Client(..., logging_callback=handle_log, log_level="info")`. The `log_level` is required: the server sends log messages only to clients that opt in with a level. Change `LOG_LEVEL` in `client.py` to `"warning"` to see fewer.
- **Client, progress:** `call_tool(name, args, progress_callback=handle_progress)`, passed per tool call.

Note: the MCP spec marks the logging feature as deprecated from protocol version 2026-07-28 (it still works in this SDK). Progress notifications are not affected.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Dependencies install automatically on first run (Python 3.13, `mcp` 2.x, `groq`, `httpx`, `python-dotenv`).

The client reads `GROQ_API_KEY` from a `.env` file in this folder or, if there is none, from the parent folder (`Module_1_ops_dash/.env`).

Optional: set `GROQ_MODEL` in `.env` to use a different Groq model. The default is `openai/gpt-oss-120b`; the model must support tool calling.

## Run

Terminal 1 — start the server:

```bash
uv run server.py
```

Terminal 2 — start the chat:

```bash
uv run client.py
```

Type a question; type `quit` to exit. Each tool call the model makes is printed as a `[tool]` line.

## How it works

1. The client opens an MCP session to the server URL and calls `list_tools`.
2. Each MCP tool schema is converted to Groq's OpenAI-style `tools` format.
3. Your question goes to Groq. If the model replies with tool calls, the client runs them through the MCP session and sends the results back.
4. This repeats (up to 6 rounds per question) until the model gives a final answer.

## Notes

- This uses `mcp` 2.x, where `FastMCP` was renamed to `MCPServer` and the client is `mcp.Client`. Tutorials written for 1.x use different imports.
- `get_article` truncates text to keep requests inside Groq's token-per-minute limits. Raise `max_chars` (up to 20000) if your plan allows.
