"""Groq chat client for the Wikipedia MCP server.

Connects to the server over Streamable HTTP, hands its tools to a Groq model,
and runs a terminal chat loop. Start server.py first.

Run with:  uv run client.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import AsyncGroq, BadRequestError
from mcp import Client, types

SERVER_URL = "http://127.0.0.1:8000/mcp"
DEFAULT_MODEL = "openai/gpt-oss-120b"  # override with GROQ_MODEL in .env
MAX_TOOL_ROUNDS = 6  # stop a runaway tool loop
LOG_LEVEL = "info"  # lowest server log level to show: debug, info, warning, error
MAX_PARSE_RETRIES = 3 # attempts when Groq rejects a malformed tool call
REASONING_HEADROOM = 1500  # extra tokens for the model's hidden reasoning

SYSTEM_PROMPT = (
    "You are a research assistant that answers questions using English "
    "Wikipedia. Use the tools to look things up instead of answering from "
    "memory: search for the right article title first, then read the summary "
    "or the article. Keep answers concise and name the article you used."
)

# A .env next to this file wins; otherwise fall back to the one in the parent
# project folder (Module_1_ops_dash/.env).
HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")
load_dotenv(HERE.parent / ".env")
MODEL = os.getenv("GROQ_MODEL", DEFAULT_MODEL)

# The one folder this client shares with the server as a root.
SAVE_DIR = HERE / "saved_articles"


def to_groq_tools(mcp_tools) -> list[dict]:
    """Convert MCP tool definitions to Groq's OpenAI-style tool format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            },
        }
        for tool in mcp_tools
    ]


async def handle_log(params: types.LoggingMessageNotificationParams) -> None:
    """Print a log message the server pushed with ctx.info / warning / error.

    Arrives as a `notifications/message` notification while a tool is running.
    """
    print(f"  [log:{params.level}] {params.data}")


async def handle_progress(
    progress: float, total: float | None, message: str | None
) -> None:
    """Print a progress update the server pushed with ctx.report_progress.

    Arrives as a `notifications/progress` notification for the tool call that
    this callback was passed to.
    """
    amount = f"{progress:g}/{total:g}" if total else f"{progress:g}"
    print(f"  [progress] {amount} {message or ''}".rstrip())


async def handle_list_roots(context) -> types.ListRootsResult:
    """Tell the server which folders it may work in.

    The server sends `roots/list`; we answer with file:// URIs. Passing this
    callback to Client() also advertises the `roots` capability. Only
    SAVE_DIR is shared, so that is the only place the server will write.
    """
    print(f"  [roots] server asked for roots; shared {SAVE_DIR}")
    return types.ListRootsResult(
        roots=[types.Root(uri=SAVE_DIR.as_uri(), name="saved_articles")]
    )


def make_sampling_handler(groq: AsyncGroq):
    """Build the callback that answers the server's sampling requests.

    When a server tool needs an LLM, the server sends `sampling/createMessage`
    to this client. We run the prompt on Groq and return the completion, so the
    server never needs its own API key. Passing this callback to Client() is
    also what advertises the `sampling` capability to the server.
    """

    async def handle_sampling(context, params: types.CreateMessageRequestParams):
        messages = []
        if params.system_prompt:
            messages.append({"role": "system", "content": params.system_prompt})
        for message in params.messages:
            blocks = message.content if isinstance(message.content, list) else [message.content]
            text = "\n".join(b.text for b in blocks if getattr(b, "text", None))
            messages.append({"role": message.role, "content": text})

        size = sum(len(m["content"]) for m in messages)
        print(f"  [sampling] server asked {MODEL} to complete a {size}-char prompt")
        try:
            response = await groq.chat.completions.create(
                model=MODEL,
                messages=messages,
                # Groq counts the model's hidden reasoning against this limit,
                # so add headroom or the visible answer gets cut off.
                max_completion_tokens=params.max_tokens + REASONING_HEADROOM,
                temperature=params.temperature,
            )
        except Exception as exc:
            return types.ErrorData(code=-32603, message=f"Sampling failed: {exc}")

        choice = response.choices[0]
        return types.CreateMessageResult(
            role="assistant",
            content=types.TextContent(type="text", text=choice.message.content or ""),
            model=MODEL,
            stop_reason="maxTokens" if choice.finish_reason == "length" else "endTurn",
        )

    return handle_sampling


async def run_tool(mcp_client: Client, name: str, raw_args: str) -> str:
    """Run one tool call through the MCP server and return its text result."""
    try:
        args = json.loads(raw_args or "{}")
    except json.JSONDecodeError:
        return f"Error: arguments were not valid JSON: {raw_args}"

    print(f"  [tool] {name}({json.dumps(args, ensure_ascii=False)})")
    try:
        result = await mcp_client.call_tool(
            name, args, progress_callback=handle_progress
        )
    except Exception as exc:  # report the failure to the model, don't crash the chat
        return f"Error: tool call failed ({exc})."

    text = "\n".join(
        block.text for block in result.content if getattr(block, "text", None)
    )
    return text or "(The tool returned no text.)"


async def chat_with_tools(groq: AsyncGroq, tools, messages):
    """Ask Groq for the next step, retrying when it emits a malformed tool call.

    Groq returns a 400 (`output_parse_failed` / `tool_use_failed`) when the
    model's output cannot be parsed as a tool call. That is random per attempt,
    so the same request usually succeeds when sent again.
    """
    for attempt in range(1, MAX_PARSE_RETRIES + 1):
        try:
            return await groq.chat.completions.create(
                model=MODEL, messages=messages, tools=tools, tool_choice="auto"
            )
        except BadRequestError as exc:
            malformed = "output_parse_failed" in str(exc) or "tool_use_failed" in str(exc)
            if not malformed or attempt == MAX_PARSE_RETRIES:
                raise
            print(f"  [retry] model produced a malformed tool call, retrying ({attempt})")


async def answer(groq: AsyncGroq, mcp_client: Client, tools, messages) -> str:
    """Let the model call tools until it produces a final answer."""
    for _ in range(MAX_TOOL_ROUNDS):
        response = await chat_with_tools(groq, tools, messages)
        message = response.choices[0].message

        if not message.tool_calls:
            messages.append({"role": "assistant", "content": message.content})
            return message.content or ""

        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in message.tool_calls
                ],
            }
        )
        for call in message.tool_calls:
            output = await run_tool(
                mcp_client, call.function.name, call.function.arguments
            )
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": output}
            )

    # Out of rounds: ask for an answer from what has been gathered so far. The
    # model no longer has tools here, so tell it plainly not to try calling one
    # (Groq rejects the request with a 400 if it does).
    messages.append(
        {
            "role": "user",
            "content": (
                "Tool limit reached. Do not call any more tools. Answer my last "
                "question now using only the tool results above."
            ),
        }
    )
    try:
        response = await groq.chat.completions.create(model=MODEL, messages=messages)
        content = response.choices[0].message.content or ""
    except BadRequestError:
        content = ""
    content = content or (
        f"I could not finish within {MAX_TOOL_ROUNDS} rounds of tool calls. "
        "Try a narrower question."
    )
    messages.append({"role": "assistant", "content": content})
    return content


async def main() -> None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        sys.exit("GROQ_API_KEY not found. Add it to .env here or in the parent folder.")
    groq = AsyncGroq(api_key=api_key)
    SAVE_DIR.mkdir(exist_ok=True)

    connected = False
    try:
        async with Client(
            SERVER_URL,
            sampling_callback=make_sampling_handler(groq),
            list_roots_callback=handle_list_roots,
            logging_callback=handle_log,
            # Servers only send log messages to clients that opt in with a
            # level; the callback alone is not enough.
            log_level=LOG_LEVEL,
        ) as mcp_client:
            mcp_tools = (await mcp_client.list_tools()).tools
            connected = True
            tools = to_groq_tools(mcp_tools)
            print(f"Connected to {SERVER_URL}")
            print("Tools:", ", ".join(tool.name for tool in mcp_tools))
            print("Ask a question (type 'quit' to exit).\n")

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            while True:
                try:
                    question = (await asyncio.to_thread(input, "You: ")).strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if question.lower() in {"quit", "exit", "q"}:
                    break
                if not question:
                    continue

                turn_start = len(messages)
                messages.append({"role": "user", "content": question})
                try:
                    reply = await answer(groq, mcp_client, tools, messages)
                except Exception as exc:  # e.g. Groq rate limit; keep the chat alive
                    del messages[turn_start:]  # drop the half-finished turn
                    print(f"Error from Groq: {exc}\n")
                    continue
                print(f"\nAssistant: {reply}\n")
    except* Exception as group:
        if connected:
            raise  # a real bug after connecting: show the full traceback
        sys.exit(
            f"Could not reach the MCP server at {SERVER_URL}. "
            f"Start it first with: uv run server.py\n({group.exceptions[0]})"
        )


if __name__ == "__main__":
    asyncio.run(main())
