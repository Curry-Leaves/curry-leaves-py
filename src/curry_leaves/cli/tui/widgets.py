"""Presentational widgets for the TUI — styled after modern agent CLIs (Claude Code /
Hermes): mostly borderless, a `⏺` bullet per action, tool results hanging under a `⎿`
connector, dim-italic thinking, an animated `✻` status line, and a box only on the
input. Pure render + a couple of self-contained timers; no app state lives here.

Textual has no direct analogue of Ink's <Static> (render once, land in real
scrollback) / <Box> (flex layout) — instead: `TranscriptLog` is a `RichLog` that each
committed `Entry` is written to exactly once (so it behaves like <Static>: printed and
never re-rendered), `LiveTurn` is a `Static` re-rendered on every streamed event, and
layout is achieved by building one Rich renderable (a `Group`/`Table`) per widget
rather than nesting many small flex boxes.
"""

from __future__ import annotations

from typing import Optional

from rich.console import Group, RenderableType
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.reactive import reactive
from textual.widgets import RichLog, Static

from .figlet import big_text
from .logo_image import LOGO_PALETTE, LOGO_ROWS
from .state import (
    AssistantEntry,
    BannerData,
    BannerEntry,
    CompactionPart,
    EffortPart,
    Entry,
    ErrorPart,
    HandoffPart,
    NoticeEntry,
    Part,
    StatsPart,
    TextPart,
    ThinkingPart,
    ToolPart,
    UserEntry,
    compact_args,
)

# Curry-leaf green palette. Truecolor; degrades gracefully on 16-color terminals.
ACCENT = "#5fbf3a"  # leaf green — bullets, spinner, borders
GOLD = "#9bcc4a"  # lime — group labels / secondary
TITLE_ROWS = ["#c8f56a", "#a3e84f", "#77ce35", "#4faf25", "#2f8016"]  # top→bottom gradient
SPIN_FRAMES = ["✻", "✳", "✶", "✷", "✸", "✹"]

STATUS_WORD = {"idle": "Ready", "thinking": "Thinking", "working": "Working"}


# ── formatting helpers ────────────────────────────────────────────────────────


def _format_args(args: dict[str, object]) -> str:
    """Render tool args compactly: `value` for a single arg, else `k: v, …`."""
    entries = list(args.items())
    if not entries:
        return ""

    def show(v: object) -> str:
        if isinstance(v, str):
            return v
        import json

        return json.dumps(v)

    if len(entries) == 1:
        s = show(entries[0][1])
    else:
        s = ", ".join(f"{k}: {show(v)}" for k, v in entries)
    if len(s) > 72:
        return f"{s[:72]}…"
    return s or compact_args(args)


def fmt_k(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def meter(ratio: float, width: int = 8) -> str:
    """A compact fill meter: `[▓▓▓░░░░]`."""
    filled = max(0, min(width, round(ratio * width)))
    return f"[{'▓' * filled}{'░' * (width - filled)}]"


# ── parts of an assistant turn ────────────────────────────────────────────────


def _render_tool_part(part: ToolPart, frame: str) -> RenderableType:
    running = part.status == "running"
    nested = part.depth > 0
    glyph_color = ACCENT if running else ("red" if part.status == "error" else (GOLD if nested else ACCENT))
    glyph = frame if running else ("↳" if nested else "⏺")
    sub_indent = "  " * part.depth

    head = Text()
    head.append(f"{glyph} ", style=glyph_color)
    head.append(sub_indent)
    if part.agent:
        head.append(f"{part.agent} ", style=GOLD)
    head.append(part.name, style=f"bold{' red' if part.status == 'error' else ''}")
    head.append(f"({_format_args(part.args)})", style="dim")

    result_style = "red" if part.status == "error" else "dim"
    result_line = Text(f"  {sub_indent}⎿ {'…' if running else (part.result or 'done')}", style=result_style)
    return Group(head, result_line)


def _render_part(part: Part, frame: str) -> Optional[RenderableType]:
    if isinstance(part, TextPart):
        t = Text()
        t.append("⏺ ", style=ACCENT)
        t.append(part.text)
        return t
    if isinstance(part, ThinkingPart):
        nested = part.depth > 0
        t = Text("💭 ")
        if nested and part.agent:
            t.append(f"{part.agent} ", style=GOLD)
        t.append(part.text.strip(), style="dim italic")
        return t
    if isinstance(part, ToolPart):
        return _render_tool_part(part, frame)
    if isinstance(part, HandoffPart):
        t = Text()
        t.append("⏺ ", style=ACCENT)
        t.append("transfer → ", style="dim")
        t.append(part.to, style=ACCENT)
        return t
    if isinstance(part, CompactionPart):
        t = Text()
        t.append("⛁ ", style=GOLD)
        t.append(
            f"compacted history ({part.reason}) · {part.messages_before}→{part.messages_after} messages",
            style="dim",
        )
        return t
    if isinstance(part, ErrorPart):
        t = Text()
        t.append("⏺ ", style="red")
        t.append(part.message, style="red")
        return t
    if isinstance(part, StatsPart):
        cost = f" · ${part.cost:.4f}" if part.cost > 0 else ""
        t = Text(
            f"⏱ {part.seconds:.1f}s · ↑{fmt_k(part.in_tok)} in · ↓{fmt_k(part.out_tok)} out{cost}",
            style="dim",
        )
        return t
    if isinstance(part, EffortPart):
        # A visible "Thinking…" header so extended reasoning is unmistakable. `minimal` means
        # the classifier chose no extended thinking, so there's nothing to announce.
        if part.effort == "minimal":
            return None
        t = Text()
        t.append("✻ ", style=GOLD)
        t.append("Thinking… ", style=GOLD)
        t.append(f"({part.effort})", style="dim")
        return t
    return None


def _renders_nothing(p: Part) -> bool:
    """A part that renders to nothing — skipped so it doesn't leave a stray blank line."""
    return isinstance(p, EffortPart) and p.effort == "minimal"


def render_parts(parts: list[Part], frame: str) -> RenderableType:
    visible = [p for p in parts if not _renders_nothing(p)]
    rows: list[RenderableType] = []
    for i, p in enumerate(visible):
        # A blank line before each block, so tools / thinking / text are visually separated.
        # Exception: thinking text hugs its "Thinking…" header (prev effort) instead of a gap.
        gap = i > 0 and not isinstance(visible[i - 1], EffortPart)
        rendered = _render_part(p, frame)
        if rendered is None:
            continue
        rows.append(Padding(rendered, (1, 0, 0, 0)) if gap else rendered)
    return Group(*rows) if rows else Text("")


# ── the startup splash (Hermes-style) ─────────────────────────────────────────


def _title_banner(text: str) -> RenderableType:
    """The big block-letter title, painted top→bottom with the amber gradient."""
    rows = big_text(text)
    lines = [Text(row, style=f"bold {TITLE_ROWS[i] if i < len(TITLE_ROWS) else GOLD}") for i, row in enumerate(rows)]
    return Group(*lines)


def _leaf_logo() -> RenderableType:
    """The logo rendered as a truecolor ASCII image (see logo_image.py)."""
    lines: list[Text] = []
    for row in LOGO_ROWS:
        t = Text()
        for ch, f, _b in row:
            style = LOGO_PALETTE[f] if f >= 0 else None
            t.append(ch, style=style)
        lines.append(t)
    return Group(*lines)


def render_splash(data: BannerData) -> RenderableType:
    title = _title_banner("CURRY LEAVES")

    left = Table.grid(padding=0)
    left.add_column()
    left.add_row(_leaf_logo())
    model_line = Text()
    model_line.append(data.model, style=ACCENT)
    model_line.append(f" · {data.provider}", style="dim")
    left.add_row(Padding(model_line, (1, 0, 0, 0)))
    left.add_row(Text(data.cwd, style="dim"))
    left.add_row(Text(f"Session: {data.session}", style="dim"))

    right_body: list[RenderableType] = []
    head = Text()
    head.append("curry-leaves", style=f"bold {ACCENT}")
    head.append(f" v{data.version} · {data.provider}", style="dim")
    right_body.append(head)

    tools_block: list[RenderableType] = [Text("Available Tools", style=f"bold {ACCENT}")]
    for label, names in data.tool_groups:
        line = Text()
        line.append(f"{label}: ", style=GOLD)
        line.append(", ".join(names), style="dim")
        tools_block.append(line)
    right_body.append(Padding(Group(*tools_block), (1, 0, 0, 0)))

    skills_block: list[RenderableType] = [Text("Available Skills", style=f"bold {ACCENT}")]
    if not data.skills:
        skills_block.append(Text("(none discovered — drop skills in ~/.curry-leaves/skills)", style="dim"))
    else:
        skills_block.append(Text(", ".join(data.skills), style="dim"))
    right_body.append(Padding(Group(*skills_block), (1, 0, 0, 0)))

    right_body.append(
        Padding(
            Text(f"{data.tool_count} tools · {data.skill_count} skills · /help for commands", style="dim"),
            (1, 0, 0, 0),
        )
    )

    right_panel = Panel(Group(*right_body), border_style=ACCENT, padding=(0, 1))

    row = Table.grid(padding=(0, 3, 0, 0))
    row.add_column()
    row.add_column(ratio=1)
    row.add_row(left, right_panel)

    footer = Group(
        Text("Welcome to curry-leaves! Type your message or /help for commands.", style=GOLD),
        Text("✦ Tip: set CURRY_LEAVES_MODEL to switch models · NO_COLOR disables colour.", style="dim"),
    )

    return Group(Padding(title, (0, 0, 1, 0)), row, Padding(footer, (1, 0, 0, 0)))


# ── committed transcript rows ─────────────────────────────────────────────────


def render_entry(entry: Entry) -> RenderableType:
    if isinstance(entry, BannerEntry):
        return render_splash(entry.data)
    if isinstance(entry, UserEntry):
        t = Text()
        t.append("› ", style="dim")
        t.append(entry.text)
        return Padding(t, (1, 0, 0, 0))
    if isinstance(entry, NoticeEntry):
        lines = [Text(l, style="dim") for l in entry.lines]
        return Padding(Group(*lines), (1, 0, 0, 0))
    if isinstance(entry, AssistantEntry):
        return Padding(render_parts(entry.parts, frame="⏺"), (1, 0, 0, 0))
    return Text("")


class TranscriptLog(RichLog):
    """Committed rows print once and stay in the terminal's scrollback (Textual's answer
    to Ink's <Static>: a RichLog only ever grows, never re-renders old content).
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(wrap=True, markup=False, **kwargs)  # type: ignore[arg-type]

    def append_entry(self, entry: Entry) -> None:
        self.write(render_entry(entry))


class LiveTurn(Static):
    """The in-progress assistant turn (re-renders as tokens stream)."""

    parts: reactive[list[Part]] = reactive(list, always_update=True)
    frame: reactive[str] = reactive("⏺")

    def render(self) -> RenderableType:
        if not self.parts:
            t = Text()
            t.append(f"{self.frame} ", style=ACCENT)
            t.append("thinking…", style="dim")
            return Padding(t, (1, 0, 0, 0))
        return Padding(render_parts(self.parts, self.frame), (1, 0, 0, 0))


# ── the bottom chrome: command menu + status line ─────────────────────────────


class CommandMenu(Static):
    """The slash-command palette shown above the input while the user is typing a
    `/command`. Filters the command list by the text after the slash; the first row is
    highlighted as the Tab/Enter completion target. Renders nothing when there's no
    match (so a stray `/` in prose doesn't pop a menu).
    """

    commands: reactive[list[tuple[str, str]]] = reactive(list, always_update=True)
    filter: reactive[str] = reactive("", always_update=True)

    def matches(self) -> list[tuple[str, str]]:
        q = self.filter.lower()
        return [(n, d) for n, d in self.commands if n.startswith(q)]

    def render(self) -> RenderableType:
        matches = self.matches()
        if not matches:
            return Text("")
        table = Table.grid(padding=(0, 1))
        table.add_column(width=12)
        table.add_column()
        for i, (name, desc) in enumerate(matches):
            style = f"bold {ACCENT}" if i == 0 else GOLD
            table.add_row(Text(f"/{name}", style=style), Text(desc, style="dim"))
        return Panel(table, border_style=GOLD, padding=(0, 1))


class StatusBar(Static):
    """The persistent bottom bar: model · context meter · state · elapsed (Hermes-style)."""

    model: reactive[str] = reactive("")
    context_used: reactive[int] = reactive(0)
    context_window: reactive[int] = reactive(0)
    status: reactive[str] = reactive("idle")
    seconds: reactive[int] = reactive(0)
    tokens: reactive[int] = reactive(0)
    frame: reactive[str] = reactive("✻")

    def render(self) -> RenderableType:
        busy = self.status != "idle"
        ratio = self.context_used / self.context_window if self.context_window > 0 else 0.0
        t = Text()
        t.append("⚡ ", style=ACCENT)
        t.append(self.model, style=f"bold {GOLD}")
        t.append(" │ ", style="dim")
        t.append(f"ctx {meter(ratio)} {fmt_k(self.context_used)}", style="dim")
        t.append(" │ ", style="dim")
        if busy:
            t.append(f"{self.frame} {STATUS_WORD.get(self.status, self.status).lower()}", style=ACCENT)
        else:
            t.append("● ready", style="green")
        t.append(" │ ", style="dim")
        t.append(f"↑{fmt_k(self.tokens)}", style="dim")
        t.append(" │ ", style="dim")
        t.append(f"{self.seconds}s", style="dim")
        return t
