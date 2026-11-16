# Sampling examples

Two real runs of `client.py` against `server.py`, showing one question that uses MCP sampling and one that does not.

How to read the output:

- `[tool]` — the chat model called a tool on the MCP server.
- `[sampling]` — the server sent a `sampling/createMessage` request back to the client, and the client ran it on its Groq model. This line appears only when sampling is involved.

Model used: `openai/gpt-oss-120b` on Groq.

These runs were captured before log and progress notifications were added, so a current run also shows `[log:...]` and `[progress]` lines between the ones below (see the README).

## Example 1 — with sampling

`summarize_article` needs an LLM to write the summary. The server has no API key, so it asks the client's model to do it.

**Prompt**

```
Use the summarize_article tool to give me a bullet summary of the Eiffel Tower article, focused on its construction.
```

**Output**

```
You: Use the summarize_article tool to give me a bullet summary of the Eiffel Tower article, focused on its construction.
  [tool] summarize_article({"focus": "construction", "title": "Eiffel Tower"})
  [sampling] server asked openai/gpt-oss-120b to complete a 12170-char prompt

Assistant: **Construction highlights from the Wikipedia article “Eiffel Tower”**

- Foundations began 28 Jan 1887; east/south legs on concrete slabs, west/north legs on deep compressed‑air caissons.
- Shoes attached to stone with 10 cm‑diameter, 7.5 m‑long bolts; foundations completed 30 Jun 1887, after which iron erection started.
- Design office produced 1,700 general + 3,629 detailed drawings for 18,038 iron parts; rivet‑hole locations to within 1 mm, angles to a second of arc.
- All components prefabricated at Levallois‑Perret, shipped by horse‑drawn carts; no on‑site drilling—mis‑fitting pieces returned to factory.
- Assembly used initial bolting, then riveted; final structure comprised 18,038 wrought‑iron pieces forming the tower’s lattice.
```

**What happened**

1. The chat model called `summarize_article` on the server.
2. The server fetched the article from Wikipedia and built a 12,170-character prompt (instructions plus article text).
3. The server sent that prompt to the client as a sampling request — the `[sampling]` line.
4. The client ran it on Groq and returned the summary to the server.
5. The server returned the summary as the tool result, and the chat model wrote the final answer.

## Example 2 — without sampling

`get_summary` only returns text from Wikipedia. No LLM is needed on the server side, so there is no `[sampling]` line.

**Prompt**

```
Use get_summary to tell me who designed the Eiffel Tower.
```

**Output**

```
You: Use get_summary to tell me who designed the Eiffel Tower.
  [tool] search_wikipedia({"limit": 5, "query": "Eiffel Tower"})
  [tool] get_summary({"title": "Eiffel Tower"})

Assistant: The Eiffel Tower was designed by the French engineer **Gustave Eiffel** (through his engineering company).【Eiffel Tower】
```

**What happened**

1. The chat model searched for the article title, then called `get_summary`.
2. The server returned the article's introduction directly.
3. The chat model answered from that text.

## Try it yourself

Start the server and the client (see [README.md](README.md)), then paste either prompt. Use a fresh client session for each test: if the question was already answered earlier in the same chat, the model may repeat that answer without calling a tool.
