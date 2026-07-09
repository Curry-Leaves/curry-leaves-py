"""Prompt builder — LAYER the bundled templates into a system prompt.

The agent carries only its OWN instructions (its persona/task). The builder wraps
those in layers, ordered by CHANGE CADENCE — stable first, volatile last — so a stable
prefix can be prompt-cached and the most dynamic guidance stays freshest:

  1. identity     harness role + principles        STABLE (never changes)
  2. instructions the agent's own prompt           stable per agent
  3. environment  cwd / date / platform            session-ish
  4. context      project files (AGENTS.md, ...)   session
  4a. subagents / 4b. skills                       session
  5. tools        tool-use guidance + LIVE tool list   VOLATILE (per turn)
  6. output       required output schema

Templates are plain string builders (no template engine) — simple and dependency-free.
The Runner calls this per turn so the prompt reflects the live tool set / context.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field

from curry_leaves.util.paths import find_up, home, join

# The neutral, task-agnostic identity — the default harness role for ANY agent. Domain
# flavor belongs in the agent's own `instructions` (or a custom identity via
# RunConfig.identity / an identity.md override), NOT here.
DEFAULT_IDENTITY = """You are a capable AI agent operating in the curry-leaves harness.
Optimize for correctness first. Verify with tools instead of guessing, prefer simple and
direct solutions, and complete the task fully rather than stopping at a plausible-looking answer."""

# A coding-flavored identity — opt in for software tasks (used by the coding CLI/presets).
CODING_IDENTITY = """You are an AI coding agent operating in the curry-leaves harness.
Optimize for correctness first, then for the next maintainer. Prefer boring, clear
solutions; verify with tools instead of guessing, and complete the task fully rather
than stopping at a plausible-looking answer."""

_PROJECT_IDENTITY_REL = ".curry-leaves/identity.md"


def resolve_identity(cwd: str | None = None) -> str:
    """The identity layer text, by precedence: project > user (~/.curry-leaves) > bundled default."""
    project = find_up(_PROJECT_IDENTITY_REL, cwd)
    if project:
        with open(project, encoding="utf-8") as f:
            return f.read().strip()
    user = join(home(), "identity.md")
    if os.path.isfile(user):
        with open(user, encoding="utf-8") as f:
            return f.read().strip()
    return DEFAULT_IDENTITY


class ContextFile(BaseModel):
    source: str
    content: str


class Environment(BaseModel):
    cwd: str
    date: str | None = None
    platform: str | None = None


class BuildPromptOptions(BaseModel):
    identity: str | None = None
    tools: set[str] = Field(default_factory=set)
    context_files: list[ContextFile] = Field(default_factory=list)
    subagents: list[tuple[str, str]] = Field(default_factory=list)
    skills: list[tuple[str, str]] = Field(default_factory=list)
    environment: Environment | None = None
    output_schema: str | None = None


def _environment_layer(env: Environment) -> str:
    lines = ["# Environment", f"Working directory: {env.cwd}"]
    if env.date:
        lines.append(f"Today's date: {env.date}")
    if env.platform:
        lines.append(f"Platform: {env.platform}")
    return "\n".join(lines)


def _context_file_layer(f: ContextFile) -> str:
    return f'<project-context source="{f.source}">\n{f.content.strip()}\n</project-context>'


def _subagents_layer(subagents: list[tuple[str, str]]) -> str:
    lines = [
        "# Transfer",
        "You can hand off the WHOLE conversation to one of these agents when the request really belongs to",
        "it. Transfer is ONE-WAY: the chosen agent takes over and continues from here — you do not resume.",
        "Use it for routing/triage, not for a sub-task you want a result from (for that, call the agent's",
        "tool instead). Transfer only when the request clearly falls to another agent.",
        "",
        "Agents you can transfer to:",
    ]
    for name, desc in subagents:
        lines.append(f"- **{name}** — {desc}")
    return "\n".join(lines)


def _skills_layer(skills: list[tuple[str, str]]) -> str:
    lines = [
        "# Skills",
        "Skills are specialized, on-demand knowledge. Scan the descriptions below for your task.",
        "If one applies, you MUST `read skill://<name>` to load its full instructions BEFORE proceeding.",
        "A skill may bundle files (scripts, references, assets) — read them with `read skill://<name>/<path>`,",
        "and run bundled scripts with `bash`.",
    ]
    for name, desc in skills:
        lines.append(f"- **{name}** — {desc}")
    return "\n".join(lines)


def _tools_layer(tools: set[str]) -> str:
    lines = ["# Tools", "Use a tool whenever it improves correctness or grounding."]
    if "read" in tools:
        lines.append("- Read files with `read` (it returns numbered lines), not `cat`.")
    if "bash" in tools:
        lines.append("- Run shell commands with `bash` when the task needs the system.")
    if "current_time" in tools:
        lines.append("- Get the current date/time (optionally per timezone) with `current_time`.")
    if "search_tools" in tools:
        lines.append(
            "- Need a capability you don't see listed? Call `search_tools` to discover more tools, then call them."
        )
    lines.append("- You MUST complete the task using the tools available; don't stop at a plausible guess.")
    return "\n".join(lines)


TODO_GUIDANCE = """# Task tracking
When a request needs more than ~2 steps, or spans multiple files, MAINTAIN a live plan with
the task tools — don't wait to be asked:
- Before you start, call `task_create` once per step to lay out the full ordered plan.
- As you work, `task_update` each item by id: flip it to `in_progress` when you start it and
  `completed` when it's done. Keep exactly ONE item `in_progress` at a time.
- Use `task_list` to re-read the plan if you lose track; `task_get` to inspect one item.
- Skip all this only for trivial, single-step requests.
This keeps you (and the user) oriented on long tasks and stops you from dropping steps."""


def _output_layer(schema: str) -> str:
    return f"""# Output format
Your FINAL reply MUST be a single JSON object matching this schema exactly —
no prose, no explanation, no markdown code fences:

{schema}"""


def build_system_prompt(instructions: str = "", opts: BuildPromptOptions | None = None) -> list[str]:
    """Wrap the agent's own `instructions` in the standard layers. Empty layers are dropped."""
    opts = opts if opts is not None else BuildPromptOptions()
    tools = opts.tools
    layers: list[str] = []

    layers.append(opts.identity if opts.identity is not None else DEFAULT_IDENTITY)  # 1. identity (stable)
    if instructions.strip():
        layers.append(instructions.strip())  # 2. the agent's prompt
    if opts.environment:
        layers.append(_environment_layer(opts.environment))  # 3. environment
    for f in opts.context_files:
        layers.append(_context_file_layer(f))  # 4. project context
    if opts.subagents:
        layers.append(_subagents_layer(opts.subagents))  # 4a. transfer roster
    if opts.skills:
        layers.append(_skills_layer(opts.skills))  # 4b. skill teasers
    if tools:
        layers.append(_tools_layer(tools))  # 5. tools (volatile)
    if "task_create" in tools:
        layers.append(TODO_GUIDANCE)  # 5b. task tracking
    if opts.output_schema:
        layers.append(_output_layer(opts.output_schema))  # 6. output format

    return [layer for layer in layers if layer.strip()]
