#!/usr/bin/env python3
"""curry-leaves — a full-screen terminal UI (Textual) over the kernel.

A thin frontend: build an Agent + Runner, then stream each turn's events into a
Textual-driven transcript with a persistent input box and status bar. Finished turns
land in the `TranscriptLog` (Textual's answer to Ink's <Static> — write once, never
re-rendered); the active turn streams live in a `LiveTurn` widget.

    export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY=sk-...
    curry-leaves                            # (after `pip install -e .`)

Env: CURRY_LEAVES_MODEL (default claude-sonnet-4-5), CURRY_LEAVES_PROVIDER, NO_COLOR.
     Each session is recorded to <home>/sessions/<id>/ (meta.json + transcript.jsonl);
     set CURRY_LEAVES_NO_RECORD to disable.
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass
from typing import Optional

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from curry_leaves import VERSION
from curry_leaves.catalog import load_catalog
from curry_leaves.core.events import AgentEvent
from curry_leaves.core.host import ApprovalChoice, ApproveTool, AskUser
from curry_leaves.settings import resolve_default_model

from .controller import Chat
from .host import PendingRequest
from .state import (
    Action,
    AssistantEntry,
    BannerData,
    BannerEntry,
    ClearAction,
    EventAction,
    FinalizeAction,
    NoticeAction,
    StatsAction,
    State,
    SubAction,
    UserAction,
    reduce,
)
from .widgets import SPIN_FRAMES, CommandMenu, LiveTurn, StatusBar, TranscriptLog

HELP: list[tuple[str, str]] = [
    ("help", "show this help"),
    ("reset", "start a fresh conversation"),
    ("fork", "branch a new session from this conversation (/fork [n] — up through the nth user turn)"),
    ("compact", "summarize history to free the context window (/compact [focus])"),
    ("auto", "toggle auto-approve of contained changes (within repo / known hosts)"),
    ("autonomous", "toggle autonomous mode (self-drive, ask only up front)"),
    ("tools", "list the agent's tools"),
    ("skills", "list discovered skills"),
    ("model", "show the active model"),
    ("stats", "show cumulative token usage"),
    ("clear", "clear the transcript"),
    ("exit", "quit (also /quit, Ctrl-C)"),
]

# Logical toolset groupings for the splash (Hermes-style). `search_tools`, `task`, and
# `transfer` are injected by the Runner, so we seed them here rather than reading agent.tools.
TOOL_GROUP_DEFS: list[tuple[str, list[str]]] = [
    ("file", ["read", "write", "edit"]),
    ("search", ["find", "search", "search_tools"]),
    ("shell", ["bash"]),
    ("tasks", ["task_create", "task_update", "task_get", "task_list"]),
    ("web", ["web_fetch", "web_search"]),
    ("delegation", ["task", "transfer"]),
    ("clarify", ["ask"]),
    ("time", ["current_time"]),
]


def _build_tool_groups(chat: Chat) -> tuple[list[tuple[str, list[str]]], int]:
    present = {t.name for t in chat.agent.tools.tools()}
    present.add("search_tools")  # always registered by the Runner
    if len(chat.agent.subagents) > 0:
        present.add("task")
        present.add("transfer")
    groups = [(label, [n for n in names if n in present]) for label, names in TOOL_GROUP_DEFS]
    groups = [(label, names) for label, names in groups if names]
    return groups, len(present)


def _init_state(chat: Chat) -> State:
    """Seed the transcript with the startup splash (scrolls with history, like Hermes)."""
    groups, count = _build_tool_groups(chat)
    skills = [name for name, _desc in chat.skills.teasers()]
    banner = BannerData(
        model=chat.model,
        provider=chat.provider,
        cwd=chat.cwd,
        session=chat.session,
        version=VERSION,
        tool_groups=groups,
        skills=skills,
        tool_count=count,
        skill_count=len(skills),
        subagents=", ".join(s.name for s in chat.agent.subagents) or "no subagents",
    )
    return State(entries=[BannerEntry(id=0, data=banner)], live=None, status="idle", next_id=1)


@dataclass
class _Usage:
    input: int = 0
    output: int = 0
    cost: float = 0.0


class AskApproveScreen(ModalScreen[None]):
    """An `ask` question / tool-approval prompt, pushed while `chat.host.current` is set.

    Mirrors the TS <AskPrompt>/<ApprovePrompt>: a free-text answer for `ask_user`
    (Enter submits; blank -> the request's default), or a single keypress for
    `approve_tool` (y/s/a/n). Resolves the pending request's `respond` on submit,
    then pops itself.
    """

    DEFAULT_CSS = """
    AskApproveScreen {
        align: center bottom;
        background: transparent;
    }
    #prompt-box {
        width: 100%;
        border: round $accent;
        padding: 0 1;
        background: $surface;
    }
    """

    def __init__(self, pending: PendingRequest) -> None:
        super().__init__()
        self._pending = pending

    def compose(self) -> ComposeResult:
        req = self._pending.req
        with Vertical(id="prompt-box"):
            if isinstance(req, ApproveTool):
                import json

                args = json.dumps(req.args or {})
                yield Static(f"[yellow]⚠ allow {req.tool}[/yellow] [dim]({req.risk}) {args[:80]}[/dim]")
                yield Static("[dim][y] once · [s] this session · [a] always · [n] no[/dim]")
            else:
                yield Static(f"[cyan]❓ {req.question}[/cyan]")
                if req.options:
                    yield Static(f"[dim]options: {' / '.join(req.options)}[/dim]")
                yield Input(placeholder="type your answer, Enter to submit", id="ask-input")

    def on_mount(self) -> None:
        if isinstance(self._pending.req, AskUser):
            self.query_one("#ask-input", Input).focus()

    def on_key(self, event: object) -> None:
        req = self._pending.req
        if not isinstance(req, ApproveTool):
            return
        key = getattr(event, "key", "")
        c = key.lower() if isinstance(key, str) else ""
        choice: Optional[ApprovalChoice] = None
        if c == "y":
            choice = "once"
        elif c == "s":
            choice = "session"
        elif c == "a":
            choice = "always"
        elif c in ("n", "escape"):
            choice = "deny"
        if choice is not None:
            self._pending.respond(choice)
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        req = self._pending.req
        if isinstance(req, AskUser):
            answer = event.value.strip() or req.default
            self._pending.respond(answer)
            self.dismiss(None)


class CurryLeavesApp(App[None]):
    """The TUI container. Owns the transcript state, the input line, slash-command
    dispatch, and driving one turn of the Runner: it iterates `runner.stream`,
    dispatches each event into the transcript reducer, and subscribes for subagent
    activity.

    Layout: the transcript log (committed rows), the live turn, a command menu popup,
    the input box, and a status bar — the Textual analogue of App.tsx's column of
    <Static>/<LiveView>/<CommandMenu>/<InputBox>/<StatusBar>.
    """

    CSS = """
    Screen {
        layout: vertical;
    }
    #transcript {
        height: 1fr;
        border: none;
        scrollbar-gutter: stable;
    }
    #live {
        height: auto;
        max-height: 40%;
    }
    #menu {
        height: auto;
    }
    #input {
        height: auto;
        margin: 1 1 0 1;
    }
    #status {
        height: 1;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    def __init__(self, chat: Chat) -> None:
        super().__init__()
        self.chat = chat
        self.state: State = _init_state(chat)
        self.usage = _Usage()
        self.compacting = False
        self._busy_started_at: Optional[float] = None
        self._elapsed_seconds = 0
        self._spin_index = 0
        self._modal_open = False
        self._unsub_host: Optional[object] = None
        self._last_committed_id: Optional[int] = None

    # ── layout ──────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield TranscriptLog(id="transcript")
        yield LiveTurn(id="live")
        yield CommandMenu(id="menu")
        yield Input(placeholder="ask anything — /help for commands", id="input")
        yield StatusBar(id="status")

    def on_mount(self) -> None:
        transcript = self.query_one("#transcript", TranscriptLog)
        for entry in self.state.entries:
            transcript.append_entry(entry)
        self.query_one("#live", LiveTurn).display = False
        self.query_one("#menu", CommandMenu).display = False
        status = self.query_one("#status", StatusBar)
        status.model = self.chat.model
        status.context_window = self.chat.runner.model.context_window
        self.query_one("#input", Input).focus()
        # Poll the host for a pending ask/approval request (mirrors the TS onChange listener).
        self.chat.host.on_change(self._on_host_change)
        self.set_interval(0.12, self._tick_spinner)
        self.set_interval(0.25, self._tick_elapsed)

    # ── busy / timers ───────────────────────────────────────────────────────

    def _is_busy(self) -> bool:
        return self.state.live is not None or self.compacting or self.chat.host.current is not None

    def _tick_spinner(self) -> None:
        self._spin_index += 1
        frame = SPIN_FRAMES[self._spin_index % len(SPIN_FRAMES)]
        live = self.query_one("#live", LiveTurn)
        live.frame = frame
        self.query_one("#status", StatusBar).frame = frame

    def _tick_elapsed(self) -> None:
        busy = self._is_busy()
        if busy and self._busy_started_at is None:
            self._busy_started_at = time.monotonic()
        elif not busy:
            self._busy_started_at = None
            self._elapsed_seconds = 0
        if self._busy_started_at is not None:
            self._elapsed_seconds = int(time.monotonic() - self._busy_started_at)
        self.query_one("#status", StatusBar).seconds = self._elapsed_seconds

    # ── host ask/approve bridge ─────────────────────────────────────────────

    def _on_host_change(self) -> None:
        pending = self.chat.host.current
        if pending is not None and not self._modal_open:
            self._modal_open = True
            self.push_screen(AskApproveScreen(pending), callback=self._on_modal_dismiss)
        self._refresh_status()

    def _on_modal_dismiss(self, _result: None) -> None:
        self._modal_open = False

    # ── dispatch + rendering ─────────────────────────────────────────────────

    def _dispatch(self, action: Action) -> None:
        self.state = reduce(self.state, action)
        # `user`/`notice` reducers append a committed entry immediately; `finalize` (in
        # `_render_live`) commits the just-closed assistant turn once `live` goes back to None.
        if isinstance(action, (NoticeAction, UserAction)):
            transcript = self.query_one("#transcript", TranscriptLog)
            transcript.append_entry(self.state.entries[-1])
        if isinstance(action, ClearAction):
            transcript = self.query_one("#transcript", TranscriptLog)
            transcript.clear()
            self._last_committed_id = None
        self._render_live()
        self._refresh_status()

    def _render_live(self) -> None:
        live_widget = self.query_one("#live", LiveTurn)
        if self.state.live is None:
            live_widget.display = False
            live_widget.parts = []
            # If a `finalize` just committed a turn, print it into the transcript now (guarded
            # by id so this is idempotent across the several calls per action).
            if self.state.entries:
                last = self.state.entries[-1]
                if isinstance(last, AssistantEntry) and getattr(self, "_last_committed_id", None) != last.id:
                    self._last_committed_id = last.id
                    self.query_one("#transcript", TranscriptLog).append_entry(last)
        else:
            live_widget.display = True
            live_widget.parts = self.state.live

    def _refresh_status(self) -> None:
        status = self.query_one("#status", StatusBar)
        status.status = "working" if (self.compacting or self.chat.host.current) else self.state.status
        menu = self.query_one("#menu", CommandMenu)
        inp = self.query_one("#input", Input)
        busy = self._is_busy()
        pending = self.chat.host.current
        show_menu = (not busy) and (not pending) and inp.value.startswith("/") and " " not in inp.value
        menu.display = show_menu
        if show_menu:
            menu.commands = HELP
            menu.filter = inp.value[1:].lower()
        inp.disabled = busy

    # ── input handling ───────────────────────────────────────────────────────

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "input":
            self._refresh_status()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "input":
            return
        text = event.value.strip()
        event.input.value = ""
        if not text or self._is_busy():
            return
        if self._run_command(text):
            self._refresh_status()
            return
        self._run_turn(text)

    def on_key(self, event: object) -> None:
        # Tab completes the highlighted command (only meaningful while the menu is up).
        key = getattr(event, "key", "")
        if key != "tab":
            return
        menu = self.query_one("#menu", CommandMenu)
        if not menu.display:
            return
        matches = menu.matches()
        if matches:
            inp = self.query_one("#input", Input)
            inp.value = f"/{matches[0][0]} "
            inp.cursor_position = len(inp.value)
            getattr(event, "stop", lambda: None)()

    # ── slash commands ───────────────────────────────────────────────────────

    def _run_command(self, text: str) -> bool:
        """Handle a slash command. Returns True if it was one (so it isn't run as a turn)."""
        if not text.startswith("/"):
            return False
        cmd = text[1:].split()[0] if text[1:].split() else ""

        def notice(lines: list[str]) -> None:
            self._dispatch(NoticeAction(lines=lines))

        chat = self.chat
        if cmd in ("exit", "quit"):
            self.exit()
            return True
        if cmd == "help":
            notice(["commands:", *[f"  /{n.ljust(8)} {d}" for n, d in HELP]])
            return True
        if cmd == "compact":
            focus = text[1:].split(None, 1)
            focus_text = focus[1].strip() if len(focus) > 1 else ""
            notice(["compacting…"])
            self.compacting = True
            self._refresh_status()
            self._run_compact(focus_text or None)
            return True
        if cmd == "auto":
            on = chat.toggle_auto()
            notice(
                [
                    "auto-approve "
                    + ("on — contained changes (within this repo / known hosts) won't prompt" if on else "off")
                ]
            )
            return True
        if cmd == "autonomous":
            on = chat.toggle_autonomous()
            notice(["autonomous mode " + ("on — I'll self-drive after any initial questions" if on else "off")])
            return True
        if cmd == "reset":
            chat.reset()
            self._dispatch(ClearAction())
            self.usage = _Usage()
            notice(["(new conversation)"])
            return True
        if cmd == "fork":
            rest = text[1:].split(None, 1)
            arg = rest[1].strip() if len(rest) > 1 else ""
            upto_turn: Optional[int] = None
            if arg:
                try:
                    upto_turn = int(arg)
                except ValueError:
                    notice(["usage: /fork [n] — n is the user turn to fork after (0-indexed)"])
                    return True
            self._run_fork(upto_turn)
            return True
        if cmd == "tools":
            lines = ["tools:"]
            for t in chat.agent.tools.tools():
                tag = "  [deferred]" if chat.agent.tools.is_deferred(t.name) else ""
                lines.append(f"  {t.name.ljust(14)} {t.description[:60]}{tag}")
            notice(lines)
            return True
        if cmd == "skills":
            listing = chat.skills.teasers()
            if not listing:
                notice(["(no skills discovered)"])
            else:
                notice(["skills:", *[f"  {n.ljust(16)} {d[:60]}" for n, d in listing]])
            return True
        if cmd == "model":
            notice([f"model: {chat.runner.model.id} ({chat.runner.model.provider})"])
            return True
        if cmd == "stats":
            u = chat.runner.usage
            notice(
                [
                    f"usage: in {u.input} · out {u.output} · cache_read {u.cache_read} · "
                    f"total {u.total_tokens} tokens · ${u.cost.total:.4f}"
                ]
            )
            return True
        if cmd == "clear":
            self._dispatch(ClearAction())
            return True
        notice([f"unknown command: /{cmd} — try /help"])
        return True

    @work(exclusive=True)
    async def _run_compact(self, focus: Optional[str]) -> None:
        try:
            oc = await self.chat.runner.compact(focus)
            if oc.compacted:
                self._dispatch(
                    NoticeAction(
                        lines=[
                            f"compacted {oc.messages_before} → {oc.messages_after} messages",
                            "",
                            *oc.summary.split("\n"),
                        ]
                    )
                )
            else:
                self._dispatch(NoticeAction(lines=["nothing to compact — too little history yet"]))
        except Exception as err:
            self._dispatch(NoticeAction(lines=[f"compaction failed: {err}"]))
        finally:
            self.compacting = False
            self._refresh_status()

    @work(exclusive=True)
    async def _run_fork(self, upto_turn: Optional[int]) -> None:
        old_id = self.chat.session
        try:
            new_id = await self.chat.fork(upto_turn)
            self._dispatch(NoticeAction(lines=[f"forked {old_id} → {new_id}, continuing here"]))
        except Exception as err:
            self._dispatch(NoticeAction(lines=[f"fork failed: {err}"]))
        finally:
            self._refresh_status()

    # ── running a turn ───────────────────────────────────────────────────────

    @work(exclusive=True)
    async def _run_turn(self, text: str) -> None:
        self._dispatch(UserAction(text=text))
        self._render_live()  # open the live view immediately
        chat = self.chat
        started_at = time.monotonic()
        before = chat.runner.usage
        base = _Usage(input=before.input, output=before.output, cost=before.cost.total)

        def on_event(e: AgentEvent) -> None:
            if e.type == "subagent_activity":
                self.call_from_thread(self._dispatch, SubAction(e=e.event, depth=e.depth, name=e.name))

        unsubscribe = chat.runner.subscribe(on_event)
        try:
            async for e in chat.runner.stream(text):
                self._dispatch(EventAction(e=e))
                if e.type == "message_end":
                    u = chat.runner.usage
                    self.usage = _Usage(input=u.input, output=u.output, cost=u.cost.total)
        except Exception as err:
            msg = f"{type(err).__name__}: {err}"
            from curry_leaves.core.events import ev

            self._dispatch(EventAction(e=ev.error(msg, True)))
        finally:
            unsubscribe()
            u = chat.runner.usage
            self._dispatch(
                StatsAction(
                    seconds=time.monotonic() - started_at,
                    in_tok=u.input - base.input,
                    out_tok=u.output - base.output,
                    cost=u.cost.total - base.cost,
                )
            )
            self._dispatch(FinalizeAction())
            self._refresh_status()


async def _amain() -> None:
    if not sys.stdin.isatty():
        print(
            "curry-leaves's TUI needs an interactive terminal (a TTY). Use `curry-leaves-repl` for piped input.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Populate real model metadata + pricing from models.dev (cached, TTL'd). Never throws.
    await load_catalog()
    model = resolve_default_model()
    chat = Chat(model)

    # Only warn when the resolved model is a cloud provider whose key is missing. The local
    # Ollama fallback needs no key, so we say nothing and it just works.
    import os

    key = chat.api_key_var()
    if key and not os.environ.get(key):
        print(f"\x1b[33m⚠ {key} is not set — turns will fail until you export it.\x1b[0m")

    app = CurryLeavesApp(chat)
    try:
        await app.run_async()
    finally:
        await chat.close()


def main() -> None:
    try:
        asyncio.run(_amain())
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
