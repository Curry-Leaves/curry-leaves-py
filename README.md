<p align="center">
  <img src="assets/logo.png" alt="Curry Leaves logo" width="128" height="128">
</p>

<h1 align="center">Curry Leaves Agent Loop</h1>

<p align="center">A small, provider-agnostic, multi-agent kernel for building AI agents of any kind — in clean, readable Python.</p>

<p align="center">
  <a href="https://pypi.org/project/curry-leaves/"><img src="https://img.shields.io/pypi/v/curry-leaves.svg" alt="PyPI version"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="license: MIT"></a>
  <a href="https://www.python.org"><img src="https://img.shields.io/badge/python-%3E%3D3.11-brightgreen.svg" alt="python: >=3.11"></a>
  <a href="./src/curry_leaves/py.typed"><img src="https://img.shields.io/badge/types-included-blue.svg" alt="types included"></a>
  <a href="https://github.com/ilayanambi-ponramu/curry-leaves-py"><img src="https://img.shields.io/badge/github-repo-181717.svg?logo=github" alt="GitHub repo"></a>
</p>

**Curry Leaves** is a general-purpose agent kernel small enough to read in an afternoon. At its core
is a streaming tool-use loop — call the model, run the tools it asks for, feed the results back,
repeat — with everything a real agent needs built around it: **sub-agents**, **skills**, **MCP
tools**, **permission gating**, **session recording**, automatic **context compaction**, and
per-turn **reasoning-effort sizing**. Point it at any tools and any task; the kernel is
domain-agnostic — it just happens to ship a batteries-included coding toolset and CLIs on top.

One engine, any provider, any UI. The loop knows nothing about a specific LLM wire format — each
provider translates at its own boundary — so Anthropic, OpenAI, and Ollama are a *config choice, not
branching logic*. Use it as a **library** (below), or launch either bundled CLI: a full-screen
terminal UI or a line REPL.

```python
from curry_leaves import Agent, Runner, coding_tools

agent  = Agent(model="claude-sonnet-4-5", tools=coding_tools())
result = await Runner(agent).run("Summarize README.md in three bullets.")
print(result.output_text)
```

---

## Table of contents

- [Features](#features)
- [Install](#install)
- [Quick start](#quick-start)
- [Configure a provider](#configure-a-provider)
- [Terminal UI & REPL](#terminal-ui--repl)
- [Library usage](#library-usage)
- [Environment variables](#environment-variables)
- [Architecture](#architecture)
- [Project layout](#project-layout)
- [Development](#development)
- [Contributing](#contributing)
- [Non-goals](#non-goals)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Features

- **Multi-provider.** Anthropic, OpenAI, and Ollama (any OpenAI-compatible gateway) implement one
  `Provider.stream()` interface; the loop never changes. Streaming SSE is assembled into a neutral
  `AssistantMessage` at the provider boundary and nowhere else.
- **Rich toolset.** `read`, `write`, `edit`, `find`, `search`, `bash`,
  `task_create`/`task_update`/`task_list`/`task_get`, `ask`, `current_time`, `web_fetch`,
  `web_search`. Tool args are [pydantic](https://docs.pydantic.dev) models → JSON Schema for free.
  Oversized output is offloaded to an artifact store the model can page through with `read`.
- **MCP tools.** Connect any [Model Context Protocol](https://modelcontextprotocol.io) server —
  stdio subprocess or HTTP — and pick specific tools by name with `mcp_tools()`; the result is a
  plain `list[Tool]` spliced into `Agent(tools=[...])` like any preset. Servers can also be declared
  in `settings.json`'s `mcpServers` key. MCP tools default to `risk="exec"`, so they always go
  through the permission gate.
- **Deferred tools + `search_tools`.** Keep the advertised list lean; the model discovers more
  tools by keyword and activates them for the next turn — works across providers.
- **Sub-agents.** Declare `subagents=[...]`; the parent gets a `task` tool (delegation that returns
  a result) and a `transfer` tool (one-way handoff). Bounded recursion depth.
- **Structured output.** Give an agent an `output_type` (a pydantic model); the Runner injects the
  schema, validates the final reply, and retries on mismatch. `result.output` is typed.
- **Auto-thinking.** `auto_thinking=True` sizes reasoning effort (Anthropic thinking budget /
  OpenAI reasoning effort) per turn with a cheap classifier.
- **Skills.** Drop a `SKILL.md` under `.curry-leaves/skills/<name>/`; its teaser goes into the prompt and
  the model pulls the full body via `read skill://<name>` only when relevant (progressive disclosure).
- **Permissions.** An opt-in gate authorizes each tool call (`allow` / `ask` / `deny`), with standing
  approvals and contained-change auto-approval. Off by default — headless runs never hang.
- **Sessions.** Each run can be recorded to `<home>/sessions/<id>/` (`meta.json` + `transcript.jsonl`).
- **Compaction.** Long conversations are summarized as they near the context window — automatic, or
  on demand via `Runner.compact()` / the `/compact` command.
- **Typed, async, dependency-light.** Python 3.11+, `mypy --strict`, ships `py.typed`. The library
  core rides on `pydantic` + `httpx`; the CLIs add [Textual](https://textual.textualize.io) + Rich,
  and MCP support uses the official `mcp` SDK.

## Install

```bash
pip install curry-leaves
```

Then launch a CLI:

```bash
curry-leaves        # full-screen terminal UI (alias: curry)
curry-leaves-repl   # line REPL (works with piped input)
```

Requires **Python 3.11+**.

## Quick start

```python
import asyncio
from curry_leaves import Agent, Runner, coding_tools

async def main() -> None:
    agent = Agent(
        model="claude-sonnet-4-5",
        instructions="You are a concise coding assistant.",
        tools=coding_tools(),
    )
    result = await Runner(agent).run("What does src/curry_leaves/runner.py do?")
    print(result.output_text)

asyncio.run(main())
```

`Agent` is a stateless **definition** (model, tools, instructions); `Runner` holds the live
conversation and drives the streaming loop. Set an API key first (see below).

## Configure a provider

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # or
export OPENAI_API_KEY=sk-...
```

The provider is inferred from the model id (`claude-*` → Anthropic, `gpt-*`/`o1-*` → OpenAI,
`gemma*`/`llama*`/`qwen*`/… → Ollama), or set `CURRY_LEAVES_PROVIDER` / pass an explicit `provider`
to the `Agent`.

### Local models via Ollama

Ollama speaks the OpenAI wire format, so any pulled tag works with no API key:

```bash
ollama pull qwen3
CURRY_LEAVES_MODEL=qwen3 curry-leaves-repl
```

```python
from curry_leaves import Agent, Runner, coding_tools
agent = Agent(model="qwen3", tools=coding_tools())   # provider → Ollama
```

Set `OLLAMA_HOST` to point at a non-default server (`http://host:port`). Tool use needs a model with
the `tools` capability (qwen3, llama3.x, gemma3, …). Reasoning-effort knobs are dropped for Ollama,
and usage cost is `$0` (local). You can point the same `OpenAIProvider` at any OpenAI-compatible
gateway via `OPENAI_BASE_URL` or explicit provider options.

## Terminal UI & REPL

A full-screen terminal UI (built on [Textual](https://textual.textualize.io)) ships with the
package — a header bar, a streaming transcript, live thinking blocks, spinner-tracked tool calls,
indented sub-agent activity, a status bar, and a persistent input box. Finished turns land in real
terminal scrollback; the active turn streams in place.

```bash
curry-leaves
```

```
╭──────────────────────────────────────────────────────────╮
│ curry-leaves · claude-sonnet-4-5 (anthropic)             │
│ /repo · 11 tools · subagents: explore, plan              │
╰──────────────────────────────────────────────────────────╯

you › what does src/curry_leaves/runner.py do?
ai ›
  → read({"path":"src/curry_leaves/runner.py"})
      1  """The Runner — holds one live conversation … """
The Runner composes an Agent with conversation state and drives the loop …

 ● ready                                    in 4213 · out 187 · $0.0155
╭──────────────────────────────────────────────────────────╮
│ › ask anything — /help for commands                      │
╰──────────────────────────────────────────────────────────╯
```

**Slash commands:** `/help`, `/reset`, `/tools`, `/skills`, `/model`, `/stats`, `/clear`,
`/compact [focus]`, `/auto` (toggle contained-change auto-approve), `/autonomous` (toggle self-drive
mode), `/exit`. The TUI needs an interactive terminal (a TTY).

The line-streaming REPL is available as `curry-leaves-repl` — handy for piped / non-TTY input.

## Library usage

### Stream events instead of awaiting

```python
async for event in runner.stream("Refactor the parser"):
    if event.type == "message_update" and event.delta and event.delta.kind == "text":
        print(event.delta.value, end="", flush=True)
```

### Structured output

```python
import pydantic
from curry_leaves import Agent, Runner

class Summary(pydantic.BaseModel):
    title: str
    bullets: list[str]

agent = Agent(model="claude-sonnet-4-5", output_type=Summary)

result = await Runner(agent).run("Summarize this repo.")
report = result.output   # validated Summary instance; the Runner retries on a schema mismatch
```

### Sub-agents

```python
from curry_leaves import Agent, Runner, coding_tools, explore_agent, plan_agent

agent = Agent(
    model="claude-sonnet-4-5",
    tools=coding_tools(),
    subagents=[explore_agent("claude-sonnet-4-5"), plan_agent("claude-sonnet-4-5")],
)
# The parent can `task` (delegate → result) or `transfer` (one-way handoff) to these.
```

### MCP tools

```python
from curry_leaves import Agent, Runner, coding_tools
from curry_leaves.mcp import McpServerStdio, mcp_tools

async with McpServerStdio(name="github", command="npx", args=["-y", "@modelcontextprotocol/server-github"]) as gh:
    agent = Agent(
        model="claude-sonnet-4-5",
        tools=[*coding_tools(), *await mcp_tools(gh, "search_issues")],
    )
    result = await Runner(agent).run("Find open issues about streaming.")
```

Or declare servers once in `.curry-leaves/settings.json` under `mcpServers` and load them with
`load_mcp_servers()` + `McpServerManager`. See
[`examples/07_mcp_tools.py`](examples/07_mcp_tools.py) for a complete, self-contained program.

See [`examples/01_basic.py`](examples/01_basic.py) for the smallest runnable program, and
[`examples/06_host_and_permissions.py`](examples/06_host_and_permissions.py) for a self-contained
illustration of the host / permission model.

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | Anthropic auth | — |
| `OPENAI_API_KEY` | OpenAI auth | — |
| `CURRY_LEAVES_MODEL` | Model id for the CLIs | auto-detected |
| `CURRY_LEAVES_PROVIDER` | Force a provider (`anthropic` \| `openai` \| `ollama`) | inferred from model id |
| `CURRY_LEAVES_HOME` | Base dir for settings / skills / sessions | `~/.curry-leaves` |
| `CURRY_LEAVES_NO_RECORD` | Disable session recording when set | recording on |
| `OPENAI_BASE_URL` | Point `OpenAIProvider` at any OpenAI-compatible gateway | `https://api.openai.com/v1` |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `NO_COLOR` | Disable ANSI color | color on |

## Architecture

The design is a strict layering — a **stateless definition** on a **stateful driver** on a **pure
engine** — with all I/O pushed to swappable seams (Provider, Host, Tool).

| Layer | File | Responsibility |
|---|---|---|
| **Message model** | `core/messages.py` | Provider-neutral `Message`/`Content` types — the one thing everything agrees on. |
| **Events** | `core/events.py` | What the loop yields; a small structural set + a streaming `delta` payload. |
| **Loop** | `core/loop.py` | The pure engine: `stream → run tools → stream`, while tools are called. Yields events. |
| **Tools** | `core/tools.py` | A registry of pydantic-typed tools + a concurrent executor with a universal large-result guard. |
| **Agent** | `core/agent.py` | A stateless definition (model, tools, instructions, sub-agents, `output_type`). |
| **Runner** | `runner.py` | Live conversation state; builds the `Context` each turn; wires sub-agents, handoff, permissions, compaction. |
| **Providers** | `providers/*` | The only place that knows a wire format. Anthropic / OpenAI / Ollama behind one `Provider`. |
| **MCP** | `mcp/*` | MCP client: stdio/HTTP server connections, a manager, a settings loader, and a tool adapter. |
| **Host** | `core/host.py` | The frontend seam: `emit(event)` + `request(req)`. Headless by default. |
| **Prompt** | `prompt.py` | Layered system prompt (identity → instructions → env → context → tools), cache-friendly. |
| **Permission** | `permission.py` | Per-call `allow` / `ask` / `deny` gate with standing approvals. |
| **Thinking** | `thinking.py` | A tiny classifier that sizes reasoning effort per task. |
| **Skills** | `skills.py` | Progressive-disclosure capability packages via `skill://` refs. |
| **Compaction** | `compaction.py` | Summarizes old history as the context window fills. |

The one decision that drives the whole loop:

```
runnable = stop_reason in ("tool_use", "stop") AND tool_calls exist
```

Tools were called → loop again. None → stop.

## Project layout

```
src/curry_leaves/
  core/        # messages, events, loop, tools, agent, host, blobs — the kernel
  providers/   # anthropic, openai/ollama, factory, sse, base
  tools/       # read, write, edit, find, search, bash, tasks, ask, web, …
  mcp/         # MCP client: servers, manager, config, tool adapter
  session/     # session store + recording
  cli/         # chat REPL + Textual TUI
  util/        # paths, retry, frontmatter, resources
  runner.py prompt.py permission.py thinking.py skills.py compaction.py catalog.py settings.py
  __init__.py  # public API surface
examples/      # runnable examples
tests/         # pytest suite (MCP subsystem)
```

## Development

```bash
git clone https://github.com/ilayanambi-ponramu/curry-leaves-py.git
cd curry-leaves-py
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

mypy src               # strict type check (the correctness gate)
pytest                 # run the test suite
curry-leaves           # launch the Textual TUI (needs a TTY)
curry-leaves-repl      # launch the REPL (works with piped input)
python3 examples/01_basic.py "What is this project?"
```

`mypy --strict` is the gate every change must pass; `pytest` covers the MCP subsystem.
Contributions extending the test suite are welcome.

## Contributing

Contributions are very welcome — bug reports, features, docs, and tests. In short:

1. **Open an issue first** for anything non-trivial, so we can agree on the approach.
2. **Fork & branch** off `main` (`git checkout -b feat/short-description`).
3. **Make the change**, keep the diff focused, and ensure `mypy src` and `pytest` pass.
4. **Open a pull request** describing the what and why.

See **[CONTRIBUTING.md](./CONTRIBUTING.md)** for the full guide — dev setup, code conventions, how to
add a tool / provider / frontend capability, commit style, and bug-reporting.

## Non-goals

curry-leaves is deliberately small. It does **not** include LSP integration, vector stores, or a
plugin marketplace. If you need those, they belong in a layer built on top of the kernel, not inside
it. (Unlike the TypeScript sibling, MCP *is* included here — as a thin client layer that feeds the
existing `Tool` seam, not a change to the kernel.)

## Acknowledgements

curry-leaves for Python is the sibling of
[curry-leaves-ts](https://github.com/ilayanambi-ponramu/curry-leaves-ts), a TypeScript kernel of the
same design; the two ports mirror each other module-for-module.

## License

[MIT](./LICENSE) © Ilayanambi Ponramu
