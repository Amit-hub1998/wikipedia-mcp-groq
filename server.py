"""Wikipedia MCP server.

Exposes five tools over Streamable HTTP at http://127.0.0.1:8000/mcp:
search_wikipedia, get_summary, get_article, summarize_article (which uses
sampling to have the client's LLM write the summary) and save_article (which
uses roots to save into a folder the client shares).

Run with:  uv run server.py
"""

import html
import re
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse
from urllib.request import url2pathname

import httpx
from mcp.server.mcpserver import Context, ListRoots, MCPServer, Resolve, Sample
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import (
    CreateMessageResult,
    ListRootsResult,
    SamplingMessage,
    TextContent,
)

HOST = "127.0.0.1"
PORT = 8000
WIKI_API = "https://en.wikipedia.org/w/api.php"
# Wikipedia asks automated clients to identify themselves.
USER_AGENT = "wikipedia-mcp-demo/0.1 (educational MCP example)"

mcp = MCPServer(
    "wikipedia",
    instructions=(
        "Tools for reading English Wikipedia. Use search_wikipedia to find the "
        "exact article title, then get_summary for a quick answer or "
        "get_article for the full text."
    ),
)


async def _wiki_query(**params) -> dict:
    """Call the MediaWiki action API and return the decoded JSON."""
    params = {"action": "query", "format": "json", "formatversion": "2", **params}
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT}, timeout=15
    ) as client:
        response = await client.get(WIKI_API, params=params)
        response.raise_for_status()
        return response.json()


async def _fetch_page(title: str, intro_only: bool) -> dict | str:
    """Fetch one page's plain-text extract.

    Returns the page dict, or a readable error string the model can act on.
    """
    params = {
        "prop": "extracts|info|pageprops",
        "titles": title,
        "explaintext": "1",
        "redirects": "1",
        "inprop": "url",
        "ppprop": "disambiguation",
    }
    if intro_only:
        params["exintro"] = "1"
    try:
        data = await _wiki_query(**params)
    except httpx.HTTPError as exc:
        return f"Error: could not reach Wikipedia ({exc})."

    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing") or pages[0].get("invalid"):
        return (
            f"Error: no Wikipedia article titled '{title}'. "
            "Call search_wikipedia to find the exact title."
        )
    return pages[0]


def _is_disambiguation(page: dict) -> bool:
    return "disambiguation" in page.get("pageprops", {})


# --- Logs and progress notifications -------------------------------------------
# A tool that takes a `ctx: Context` parameter can push messages to the client
# while it is still running. The parameter is filled by the SDK and hidden from
# the calling model.
#   ctx.info / ctx.warning / ctx.error  -> `notifications/message`  (log lines)
#   ctx.report_progress(done, total)    -> `notifications/progress` (progress)
# These go to the person running the client, not to the LLM: they do not become
# part of the tool result.


@mcp.tool()
async def search_wikipedia(query: str, ctx: Context, limit: int = 5) -> str:
    """Search Wikipedia and return matching article titles with short snippets.

    Args:
        query: What to search for.
        limit: Maximum number of results (1-10).
    """
    limit = max(1, min(limit, 10))
    await ctx.info(f"Searching Wikipedia for '{query}' (limit {limit})")
    try:
        data = await _wiki_query(list="search", srsearch=query, srlimit=str(limit))
    except httpx.HTTPError as exc:
        await ctx.error(f"Wikipedia request failed: {exc}")
        return f"Error: could not reach Wikipedia ({exc})."

    results = data.get("query", {}).get("search", [])
    if not results:
        await ctx.warning(f"No results for '{query}'")
        return f"No Wikipedia articles found for '{query}'."
    await ctx.info(f"Found {len(results)} result(s)")

    lines = []
    for i, item in enumerate(results, start=1):
        snippet = html.unescape(re.sub(r"<[^>]+>", "", item.get("snippet", "")))
        lines.append(f"{i}. {item['title']} - {snippet}")
    return "\n".join(lines)


@mcp.tool()
async def get_summary(title: str, ctx: Context) -> str:
    """Get the introduction (lead section) of a Wikipedia article.

    Args:
        title: Exact article title, e.g. from search_wikipedia.
    """
    await ctx.info(f"Fetching the introduction of '{title}'")
    page = await _fetch_page(title, intro_only=True)
    if isinstance(page, str):
        await ctx.warning(page)
        return page

    text = page.get("extract", "").strip() or "(This article has no summary text.)"
    await ctx.info(f"Got {len(text)} characters from '{page['title']}'")
    note = ""
    if _is_disambiguation(page):
        await ctx.warning(f"'{page['title']}' is a disambiguation page")
        note = (
            "Note: this is a disambiguation page. Call search_wikipedia to pick "
            "a more specific title.\n\n"
        )
    return f"{note}{page['title']}\n{page.get('fullurl', '')}\n\n{text}"


@mcp.tool()
async def get_article(title: str, ctx: Context, max_chars: int = 6000) -> str:
    """Get the plain text of a Wikipedia article, truncated to max_chars.

    Args:
        title: Exact article title, e.g. from search_wikipedia.
        max_chars: Maximum characters of article text to return (500-20000).
    """
    max_chars = max(500, min(max_chars, 20000))
    await ctx.report_progress(0, 3, "Requesting article from Wikipedia")
    await ctx.info(f"Fetching the full article '{title}'")
    page = await _fetch_page(title, intro_only=False)
    if isinstance(page, str):
        await ctx.warning(page)
        return page

    text = page.get("extract", "").strip() or "(This article has no text.)"
    total = len(text)
    await ctx.report_progress(1, 3, f"Downloaded {total} characters")
    if total > max_chars:
        await ctx.info(f"Article is {total} characters; truncating to {max_chars}")
        text = (
            text[:max_chars].rstrip()
            + f"\n\n[Truncated: showing {max_chars} of {total} characters.]"
        )
    await ctx.report_progress(2, 3, "Text prepared")
    note = ""
    if _is_disambiguation(page):
        await ctx.warning(f"'{page['title']}' is a disambiguation page")
        note = (
            "Note: this is a disambiguation page. Call search_wikipedia to pick "
            "a more specific title.\n\n"
        )
    await ctx.report_progress(3, 3, "Done")
    return f"{note}{page['title']}\n{page.get('fullurl', '')}\n\n{text}"


# --- Sampling -----------------------------------------------------------------
# The server has no LLM and no API key. With sampling it asks the *client* to run
# a prompt on the client's model and hand back the completion.
#
# In mcp 2.x a tool does this through a resolver: a function that runs before the
# tool body and returns a Sample(...) request. The SDK sends it to the client as
# `sampling/createMessage` and injects the reply into the tool parameter that is
# annotated with Resolve(...). That parameter is hidden from the calling model.

SUMMARY_SOURCE_CHARS = 12000  # how much article text to send for summarising


async def _summary_request(title: str, focus: str = "") -> Sample:
    """Resolver: fetch the article and build the prompt for the client's LLM.

    `title` and `focus` are filled from the tool call's arguments by name.
    """
    page = await _fetch_page(title, intro_only=False)
    if isinstance(page, str):
        raise ToolError(page)  # e.g. article not found; reported to the model
    if _is_disambiguation(page):
        raise ToolError(
            f"'{title}' is a disambiguation page. Call search_wikipedia to pick "
            "a more specific title."
        )

    text = page.get("extract", "").strip()[:SUMMARY_SOURCE_CHARS]
    instruction = "Summarise this Wikipedia article in at most 5 bullet points."
    if focus:
        instruction += f" Focus on: {focus}."
    prompt = f"{instruction}\n\nArticle: {page['title']}\n\n{text}"

    return Sample(
        [SamplingMessage(role="user", content=TextContent(type="text", text=prompt))],
        max_tokens=400,
        temperature=0.2,
        system_prompt="You write short, factual summaries using only the given text.",
    )


@mcp.tool()
async def summarize_article(
    title: str,
    summary: Annotated[CreateMessageResult, Resolve(_summary_request)],
    ctx: Context,
    focus: str = "",
) -> str:
    """Get a short bullet-point summary of a Wikipedia article.

    Cheaper than get_article when only the key points are needed.

    Args:
        title: Exact article title, e.g. from search_wikipedia.
        focus: Optional aspect to concentrate on, e.g. "construction history".
    """
    # `summary` is the client LLM's reply to the Sample request above.
    text = getattr(summary.content, "text", "") or "(The model returned no text.)"
    await ctx.info(
        f"Client model {summary.model} returned a {len(text)}-character summary"
    )
    note = ""
    if summary.stop_reason == "maxTokens":
        await ctx.warning("The summary hit its length limit")
        note = "\n\n[The summary hit its length limit and may end abruptly.]"
    return (
        f"Summary of '{title}' (written by {summary.model}). This is the complete "
        f"result; do not call this tool again for the same article.\n\n{text}{note}"
    )


# --- Roots --------------------------------------------------------------------
# Roots are folders the *client* says the server may work in. The server asks
# for them with `roots/list` and must stay inside them: it never picks a
# location on the user's disk by itself.
#
# Like sampling, the request is made by a resolver: returning ListRoots() makes
# the SDK ask the client and inject the ListRootsResult into the tool.


def _client_roots() -> ListRoots:
    """Resolver: ask the client which folders this server may use."""
    return ListRoots()


def _root_to_path(uri: str) -> Path:
    """Convert a file:// root URI to a local path."""
    return Path(url2pathname(urlparse(uri).path))


@mcp.tool()
async def save_article(
    title: str,
    roots: Annotated[ListRootsResult, Resolve(_client_roots)],
    ctx: Context,
) -> str:
    """Save the full text of a Wikipedia article as a Markdown file.

    The file goes into the folder the client has shared with this server.

    Args:
        title: Exact article title, e.g. from search_wikipedia.
    """
    if not roots.roots:
        await ctx.warning("The client shared no roots; nowhere to save")
        return "Error: the client has not shared a folder (root) to save into."
    root = roots.roots[0]
    folder = _root_to_path(str(root.uri)).resolve()
    await ctx.info(f"Client root: {root.name or 'unnamed'} -> {folder}")
    if not folder.is_dir():
        return f"Error: the shared folder {folder} does not exist."

    page = await _fetch_page(title, intro_only=False)
    if isinstance(page, str):
        await ctx.warning(page)
        return page

    # Build the filename from the title only, then confirm the final path is
    # still inside the root, so a crafted title cannot write anywhere else.
    name = re.sub(r"[^A-Za-z0-9 _-]+", "", page["title"]).strip().replace(" ", "_")
    target = (folder / f"{name or 'article'}.md").resolve()
    if not target.is_relative_to(folder):
        await ctx.error(f"Refused to write outside the root: {target}")
        return "Error: refusing to write outside the shared folder."

    text = page.get("extract", "").strip()
    target.write_text(
        f"# {page['title']}\n\nSource: {page.get('fullurl', '')}\n\n{text}\n",
        encoding="utf-8",
    )
    await ctx.info(f"Wrote {len(text)} characters to {target.name}")
    return f"Saved '{page['title']}' ({len(text)} characters) to {target}"


if __name__ == "__main__":
    print(f"Wikipedia MCP server listening on http://{HOST}:{PORT}/mcp")
    mcp.run(transport="streamable-http", host=HOST, port=PORT)
