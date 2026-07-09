"""Rendering — turn the loop's event stream into terminal output.

Every frontend's core job: CONSUME events and display them. Thinking is shown as a
framed block; subagent activity is indented by depth; tools/errors get their own lines.
`run_turn` streams one turn (subscribing for subagent activity), then prints a usage footer.
"""

from __future__ import annotations

import json
import re
import sys

from curry_leaves.core.events import AgentEvent
from curry_leaves.core.messages import TextBlock
from curry_leaves.runner import Runner

from .theme import CYAN, DIM, ITALIC, RED, RESET


def _w(s: str) -> None:
    sys.stdout.write(s)


# Whether we're mid-"thinking" stream, so reasoning is framed and closed before answer text.
_in_thinking = False
# Per-depth flags so each subagent frames its reasoning independently.
_sub_thinking: dict[int, bool] = {}


def _end_thinking() -> None:
    global _in_thinking
    if _in_thinking:
        _w(f"\n{DIM}╰─ end thinking ─────────────────{RESET}\n")
        _in_thinking = False


def _snippet(text: str, max_len: int) -> str:
    """Truncate a tool result to a single readable line."""
    one_line = re.sub(r"\s+", " ", text).strip()
    return f"{one_line[:max_len]}…" if len(one_line) > max_len else one_line


def render(e: AgentEvent) -> None:
    """Map one loop event to terminal output (the main agent's stream)."""
    global _in_thinking

    if e.type == "message_update" and e.delta is not None and e.delta.kind == "thinking":
        if not _in_thinking:
            _w(f"\n{DIM}╭─ 💭 thinking ──────────────────{RESET}\n")
            _in_thinking = True
        _w(f"{DIM}{ITALIC}{e.delta.value}{RESET}")
        return

    _end_thinking()  # any non-thinking event closes the reasoning block first

    if e.type == "message_update" and e.delta is not None and e.delta.kind == "text":
        _w(e.delta.value)
    elif e.type == "tool_start":
        _w(f"\n{DIM}  ⚙ {e.tool_name}({_compact_args(e.args)}){RESET}\n")
    elif e.type == "tool_end":
        content = e.result.content
        text = content[0].text if content and isinstance(content[0], TextBlock) else ""
        mark = "✗" if e.result.is_error else "→"
        _w(f"{DIM}  {mark} {_snippet(text, 120)}{RESET}\n")
    elif e.type == "thinking":
        _w(f"{DIM}  🧠 effort: {e.effort}{RESET}\n")
    elif e.type == "handoff":
        _w(f"\n{DIM}  ⇢ handoff: {e.from_agent} → {e.to_agent}{RESET}\n")
    elif e.type == "compaction":
        _w(f"\n{DIM}  ⛁ compacted history ({e.reason}): {e.messages_before} → {e.messages_after} messages{RESET}\n")
    elif e.type == "subagent_activity":
        render_subagent(e.event, e.depth, e.name)
    elif e.type == "error":
        _w(f"\n{RED}[error] {e.message}{RESET}\n")


def render_subagent(e: AgentEvent, depth: int, name: str) -> None:
    """Surface what a `task` subagent is doing, indented by nesting depth and tagged by name."""
    pad = "  " * depth
    if e.type == "message_update" and e.delta is not None and e.delta.kind == "thinking":
        if not _sub_thinking.get(depth):
            _w(f"\n{DIM}{pad}↳ [{name}] 💭 thinking ───────{RESET}\n")
            _sub_thinking[depth] = True
        indented = e.delta.value.replace("\n", f"\n{pad}    ")
        _w(f"{DIM}{ITALIC}{indented}{RESET}")
        return
    if _sub_thinking.get(depth):
        _sub_thinking[depth] = False
        _w(f"\n{DIM}{pad}↳ ────────────────────────{RESET}\n")
    if e.type == "thinking":
        _w(f"{DIM}{pad}↳ [{name}] 🧠 effort: {e.effort}{RESET}\n")
    elif e.type == "tool_start":
        _w(f"{DIM}{pad}↳ [{name}] ⚙ {e.tool_name}({_compact_args(e.args)}){RESET}\n")
    elif e.type == "tool_end":
        content = e.result.content
        text = content[0].text if content and isinstance(content[0], TextBlock) else ""
        _w(f"{DIM}{pad}↳ → {_snippet(text, 100)}{RESET}\n")
    elif e.type == "error":
        _w(f"\n{RED}{pad}↳ [{name}] error: {e.message}{RESET}\n")
    # Subagent final prose is intentionally not streamed here — the parent's `task` tool_end
    # carries its result. (message_update text deltas are skipped.)


def _compact_args(args: dict[str, object]) -> str:
    s = json.dumps(args)
    return f"{s[:100]}…" if len(s) > 100 else s


async def run_turn(runner: Runner, text: str) -> None:
    """Stream one turn, render it, and print the usage footer."""
    global _in_thinking
    _w(f"{CYAN}ai ›{RESET} ")
    _in_thinking = False

    # stream() now yields subagent activity too (merged in), so a single loop renders it all.
    try:
        async for e in runner.stream(text):
            render(e)
            if e.type == "agent_end":
                _end_thinking()
                _w("\n")
    except Exception as err:
        _end_thinking()
        name = type(err).__name__
        msg = str(err)
        _w(f"\n{RED}[failed] {name}: {msg}{RESET}\n")

    u = runner.usage
    cost = f" · ${u.cost.total:.4f}" if u.cost.total else ""
    cache_read = f" · cache_read {u.cache_read}" if u.cache_read else ""
    _w(f"{DIM}  tokens: in {u.input} · out {u.output}{cache_read}{cost}{RESET}\n\n")
