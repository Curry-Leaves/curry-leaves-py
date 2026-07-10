# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0] - 2026-07-09

### Added

- **Deferred-tool teasers in the system prompt** — when `search_tools` is
  advertised, the prompt now lists the names and one-line descriptions of
  tools that exist but aren't callable yet, and tells the model it must call
  `search_tools` to activate one before calling it. Previously an
  unadvertised tool referenced by name (e.g. in a skill or instruction) had
  no listing to match against; the provider's schema constraint would
  silently collapse the call onto the nearest advertised tool name instead,
  causing confused retries. `ToolRegistry.deferred_teasers()` builds the
  list; `Runner` passes it through `BuildPromptOptions.deferred_tools`.

## [1.3.1] - 2026-07-09

### Added

- **Session forking** — branch a new session off an existing conversation's
  recorded transcript instead of starting from scratch or only ever
  continuing linearly. `curry_leaves.session.fork_session(source_id, new_id,
  meta, upto_turn=...)` replays a source session's transcript (optionally
  truncated after the Nth user turn) into live `Message`s and opens a new,
  independent session store seeded with that copied history — edits from the
  fork point never touch the original session's transcript. `Runner` gained
  `RunConfig.initial_messages` to seed conversation state from a fork (or any
  other precomputed history). Wired up as `/fork [n]` in both the REPL and
  TUI, alongside `/reset`.
- **Tool-result elision** (opt-in) — reclaim context by stubbing out stale
  tool results before lossy compaction is needed. A result is eligible once
  it's stale (superseded by an identical later call, or untouched for
  `age_turns` user turns) and big enough to matter; eligible results are
  applied in batched, biggest-first sweeps (gated on occupancy and a minimum
  savings floor) so the provider prompt cache isn't invalidated turn by turn.
  Originals are preserved whole in the blob store — every stub carries a
  preview and an `artifact://<id>` pointer the model can `read` back.
  Deterministic, no model call. Enable via
  `RunConfig(elision=ElisionConfig(enabled=True))`; emits an `ElisionEvent`
  (recorded to session transcripts as `kind: "elision"`).

## [1.2.0] - 2026-07-09

### Added

- `UserMessage.origin` — tags a message as `"steering"` or `"follow_up"` when
  injected via `Runner.steer()` / `Runner.follow_up()`, `None` for a normal
  prompt. Lets consumers of the event stream tell barge-in interrupts apart
  from queued follow-ups without tracking `Runner`'s internal queues.
- `TaskStore` can now persist to disk: pass `path` to `TaskStore(path=...)` or
  `task_tools(store=...)` to keep the task list across runs (e.g. one file per
  chat session). Every mutation rewrites the file; a fully-completed list is
  reset on load so a new request starts from an empty checklist.

## [1.0.0] - 2026-07-08

### Added

- Initial release of Curry Leaves for Python — a provider-agnostic, multi-agent
  kernel for building AI agents: streaming tool-use loop, subagents, skills,
  thinking, permissions, sessions, and compaction. A faithful port of the
  [TypeScript kernel](https://github.com/ilayanambi-ponramu/curry-leaves-ts),
  module-for-module.
- Core engine (`agent_loop`), stateful driver (`Runner`), and stateless agent
  definitions (`Agent`), with agent-as-tool delegation (`task`) and one-way
  handoff (`transfer`).
- Providers for Anthropic, OpenAI, and Ollama (OpenAI-compatible), with
  automatic provider inference from model id.
- Tool system with pydantic schemas, risk-based permission fallback, deferred
  tool discovery via `search_tools`, and blob-backed result capping.
- **MCP (Model Context Protocol) client support** — connect stdio or HTTP MCP
  servers (`McpServerStdio` / `McpServerHttp` / `McpServerManager`), pick
  specific tools with `mcp_tools()`, or load server definitions from
  `settings.json`'s `mcpServers` key. Goes beyond the TypeScript sibling, which
  lists MCP as a non-goal.
- Permission engine with per-call resolution (deny > per-tool allow > standing
  approval > per-tool ask > default > risk fallback).
- Auto-thinking (reasoning-effort classifier) and auto-compaction near context
  limits.
- Skills system with progressive disclosure from `~/.curry-leaves/skills/` and
  `.curry-leaves/skills/`.
- Model catalog sourced from models.dev (context windows, pricing).
- Session recording to `<home>/sessions/<id>/`.
- Two bundled CLIs: a full-screen Textual TUI (`curry-leaves`, alias `curry`)
  and a line REPL (`curry-leaves-repl`).
- Example scripts covering basic usage, streaming, custom tools, structured
  output, subagents, host/permissions, and MCP tools.
- Test suite for the MCP subsystem (`tests/mcp`); strict `mypy` across `src/`.
