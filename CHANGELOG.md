# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
