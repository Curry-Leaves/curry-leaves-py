# Contributing to curry-leaves

Thanks for your interest in improving **curry-leaves** — bug reports, features, docs, and tests are all
welcome. This guide covers how to get set up, the conventions that keep the codebase coherent, and
how to extend the kernel.

By participating, you agree to keep interactions respectful and constructive.

## Table of contents

- [Ways to contribute](#ways-to-contribute)
- [Development setup](#development-setup)
- [Project layout](#project-layout)
- [Pull request workflow](#pull-request-workflow)
- [Code conventions](#code-conventions)
- [Extending the kernel](#extending-the-kernel)
- [Commit messages](#commit-messages)
- [Reporting bugs](#reporting-bugs)
- [License](#license)

## Ways to contribute

- **Report a bug** — open an issue with a minimal reproduction (see [Reporting bugs](#reporting-bugs)).
- **Propose a feature** — open an issue describing the use case *before* writing code, so we can
  agree on the approach.
- **Improve docs** — READMEs, code comments, and examples all count.
- **Add tests** — the MCP subsystem has a suite under `tests/mcp`; extending coverage to the core
  loop, providers, and tools is a high-value contribution.

## Development setup

Requires **Python 3.11+**.

```bash
git clone https://github.com/Curry-Leaves/curry-leaves-py.git
cd curry-leaves-py
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Common commands:

| Command | What it does |
|---|---|
| `mypy src` | Strict type check — **the correctness gate** |
| `pytest` | Run the test suite (`tests/`) |
| `curry-leaves` | Launch the full-screen Textual TUI (needs a TTY) |
| `curry-leaves-repl` | Launch the line REPL (works with piped input) |
| `python3 examples/01_basic.py "..."` | Run an example end-to-end |

To exercise a real turn you need a provider key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY=sk-...
curry-leaves
```

Or run fully local with [Ollama](https://ollama.com) — no key required:

```bash
ollama pull qwen3
CURRY_LEAVES_MODEL=qwen3 curry-leaves-repl
```

> **Testing status:** `mypy src` under `strict` is the gate every change must pass, plus `pytest`
> for the MCP suite. Providers are written so their request builders and stream parsers are plain
> module functions (not client methods) specifically to be unit-testable — contributions that add
> a test suite around them are especially welcome.

## Project layout

```
src/curry_leaves/
  core/        # messages, events, loop, tools, agent, host, blobs — the kernel
  providers/   # anthropic, openai/ollama, factory, sse, base
  tools/       # read, write, edit, find, search, bash, tasks, ask, web, …
  mcp/         # MCP client: stdio/HTTP servers, manager, settings loader, tool adapter
  session/     # session store + recording
  cli/         # chat REPL + Textual TUI
  util/        # paths, retry, frontmatter, resources
  runner.py prompt.py permission.py thinking.py skills.py compaction.py catalog.py settings.py
  __init__.py  # public API surface
examples/      # runnable examples
tests/         # pytest suite (MCP subsystem)
```

The architecture is a strict layering — a **stateless definition** (`Agent`) on a **stateful
driver** (`Runner`) on a **pure engine** (`agent_loop`) — with all I/O pushed to swappable seams
(`Provider`, `Host`, `Tool`). Keep that separation in mind: behavior is added at a seam, not by
branching inside the loop. See the [README architecture section](./README.md#architecture) for the
big picture.

## Pull request workflow

1. **Open an issue first** for anything non-trivial, so we can agree on scope and approach.
2. **Fork and branch** off `main`:
   ```bash
   git checkout -b feat/short-description   # or fix/… , docs/… , test/…
   ```
3. **Make the change.** Keep the diff focused on one concern.
4. **Verify it passes:**
   ```bash
   mypy src && pytest
   ```
   For a change with a runtime surface, also drive the affected flow (a TUI turn, an example, the
   REPL) and confirm it behaves as intended — don't rely on typecheck alone.
5. **Update docs** (`README.md`, code comments, `examples/`) when behavior or the public API changes.
6. **Open a pull request** with a clear description of *what* changed and *why*. Link the issue.

Keep PRs small and reviewable. A large PR is easier to land when split into focused commits or
separate PRs.

## Code conventions

- **`src/curry_leaves/__init__.py` is the public API.** When you add an exported symbol consumers
  should see, re-export it there and add it to `__all__`.
- **Keep new source under `src/curry_leaves/`** — the package uses a `src/` layout.
- **Match the surrounding style** — small, single-purpose modules; clear names; comments that explain
  *why*, not *what*. Prefer boring, direct solutions.
- **Type strictly.** `mypy --strict` is on and must stay green. No new `Any` where a real type fits;
  public functions carry full annotations.
- **Tools describe their `risk`** (`read` / `write` / `exec` / `network`) — it drives the permission
  fallback, so set it accurately.
- **Complete the change** — types, docs, and (where practical) a runnable check, not just the happy path.

## Extending the kernel

Common extension points, each done at a seam rather than by editing the loop:

- **Add a tool** — implement the `Tool` protocol (a pydantic args model + `run`), add a factory under
  `src/curry_leaves/tools/`, export it from `__init__.py`, and add it to a preset in `presets.py` if
  it belongs in the default kit. Set `risk` correctly.
- **Add a provider** — implement `Provider.stream` in `src/curry_leaves/providers/`, keep all
  wire-format translation at that edge (nowhere else), and register it in `providers/factory.py`.
- **Add a frontend capability** — add a `Request` kind in `src/curry_leaves/core/host.py` (with a
  default value), **not** a new `Host` method — so headless hosts keep working by returning the
  default.
- **Connect an MCP server** — usually no code change needed: construct `McpServerStdio`/`McpServerHttp`
  (or declare it in `settings.json`'s `mcpServers`) and pick tools with `mcp_tools()`. Framework-level
  MCP changes live under `src/curry_leaves/mcp/` and are covered by `tests/mcp`.

## Commit messages

Use short, imperative summaries. Conventional-commit prefixes are appreciated but not required:

```
feat: add web_search deferred tool
fix: retry only on transient provider errors
docs: clarify Ollama setup
test: cover the anthropic stream parser
```

## Reporting bugs

Open an issue that includes:

- What you did (a minimal code snippet or the exact CLI command).
- What you expected vs. what happened (include the full error / stack).
- Your environment: `python --version`, curry-leaves version, provider + model id, OS.

A minimal reproduction is the fastest path to a fix.

## License

By contributing, you agree that your contributions are licensed under the project's
[MIT License](./LICENSE).
