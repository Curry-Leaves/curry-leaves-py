#!/usr/bin/env python3
"""curry-leaves — a minimal terminal chat REPL.

A thin frontend over the kernel: it builds an Agent + Runner, reads lines, streams each
turn, and renders events. Slash commands cover only what this port supports.

    export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY=sk-...
    curry-leaves-repl                       # (after `pip install -e .`)

Env: CURRY_LEAVES_MODEL (default claude-sonnet-4-5), CURRY_LEAVES_PROVIDER, NO_COLOR.
     Each session is recorded to <home>/sessions/<id>/ (meta.json + transcript.jsonl);
     set CURRY_LEAVES_NO_RECORD to disable.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime
from typing import Optional, Union

from curry_leaves.agents import explore_agent, plan_agent
from curry_leaves.core.agent import Agent
from curry_leaves.core.messages import Message
from curry_leaves.permission import PermissionEngine, PermissionOptions, contained_approval
from curry_leaves.presets import coding_tools, web_tools
from curry_leaves.prompt import CODING_IDENTITY
from curry_leaves.providers.factory import provider_name_for_model
from curry_leaves.runner import RunConfig, Runner
from curry_leaves.session import SessionMeta, SessionStore, fork_session, open_session
from curry_leaves.settings import add_global_approval, auto_hosts, global_approvals, resolve_default_model
from curry_leaves.skills import SkillRegistry
from curry_leaves.thinking import ThinkingConfig
from curry_leaves.util.paths import repo_root

from .host import CliHost
from .render import run_turn
from .theme import BOLD, CYAN, DIM, GREEN, RESET, YELLOW

# A sentinel distinguishing "quit the REPL" from "no output to show" (None) in dispatch()'s
# return value.
_EXIT = object()


def _build_agent(model: str) -> Agent:
    return Agent(
        model,
        instructions=(
            "You are curry-leaves, a concise, careful terminal coding assistant. Use tools to ground "
            "your work; verify instead of guessing; complete the task fully."
        ),
        tools=coding_tools(),
        deferred_tools=web_tools(),
        subagents=[explore_agent(model), plan_agent(model)],
        auto_thinking=True,
    )


def _session_id() -> str:
    """A session id like 20260706_124623_417 — sortable and filesystem-safe."""
    d = datetime.now()
    date = f"{d.year:04d}{d.month:02d}{d.day:02d}"
    time = f"{d.hour:02d}{d.minute:02d}{d.second:02d}"
    return f"{date}_{time}_{d.microsecond // 1000:03d}"


class Prompter:
    """The single shared "ask a question" primitive for the whole process.

    Python has no direct equivalent of Node's `readline/promises` interface reused for both
    the main prompt loop and mid-turn ask/approval prompts. This replicates that "only one
    question active at a time" property with plain `input()`, run in a thread via
    `run_in_executor` so it never blocks the event loop.
    """

    def __init__(self) -> None:
        self._loop = asyncio.get_event_loop()

    @staticmethod
    def write(s: str) -> None:
        sys.stdout.write(s)
        sys.stdout.flush()

    async def question(self, prompt: str) -> str:
        """Ask one question and await the answer. Raises EOFError on Ctrl-D (interface
        closed) — callers decide how to treat that, mirroring the TS readline 'close' event.
        """
        self.write(prompt)
        return await self._loop.run_in_executor(None, input)


class Chat:
    def __init__(self, model: str, prompter: Prompter) -> None:
        self.model = model
        self.skills = SkillRegistry(discover=True)
        self.session = _session_id()
        self.autonomous = False
        self.auto_approve = False

        provider = "?"
        try:
            provider = provider_name_for_model(model)
        except Exception:
            pass  # unknown model → leave as ?
        cwd = os.getcwd()
        self.store: SessionStore = open_session(self.session, SessionMeta(model=model, provider=provider, cwd=cwd))
        self.host = CliHost(prompter)
        # Durable approvals seed the engine; "always" grants persist; auto-approve gated on the live flag.
        contained = contained_approval(repo_root(cwd), auto_hosts())
        self.permission = PermissionEngine(
            PermissionOptions(
                global_approvals=global_approvals(),
                on_global_approve=add_global_approval,
                auto_approve=lambda tool, risk, args: self.auto_approve and contained(tool, risk, args),
            )
        )
        self.agent = _build_agent(model)
        self.runner = self._new_runner()

    def _new_runner(self, initial_messages: Optional[list[Message]] = None) -> Runner:
        return Runner(
            self.agent,
            RunConfig(
                skills=self.skills,
                thinking=ThinkingConfig(system=CODING_IDENTITY),
                store=self.store,
                host=self.host,
                permission=self.permission,
                autonomous=self.autonomous,
                initial_messages=initial_messages,
            ),
        )

    def toggle_autonomous(self) -> bool:
        """Flip autonomous (self-drive) mode live."""
        self.autonomous = not self.autonomous
        self.runner.set_autonomous(self.autonomous)
        return self.autonomous

    def toggle_auto(self) -> bool:
        """Flip auto-approve (contained changes skip the prompt). Engine reads the live flag."""
        self.auto_approve = not self.auto_approve
        return self.auto_approve

    def reset(self) -> None:
        self.agent = _build_agent(self.model)
        self.store.mark("reset")
        self.runner = self._new_runner()

    async def fork(self, upto_turn: Optional[int] = None) -> str:
        """Branch off a brand-new session that starts with this conversation's history (up
        through the `upto_turn`-th user turn, or the whole thing if None), then keep going —
        edits from here diverge into the fork without touching the original session's transcript.
        Closes the current store (its transcript is now a completed prefix of the fork's own) and
        returns the new session id.
        """
        provider = "?"
        try:
            provider = provider_name_for_model(self.model)
        except Exception:
            pass  # unknown model → leave as ?
        new_id = _session_id()
        new_store, messages = fork_session(
            self.session,
            new_id,
            SessionMeta(model=self.model, provider=provider, cwd=os.getcwd()),
            upto_turn=upto_turn,
        )
        old_store = self.store
        old_runner = self.runner
        self.session = new_id
        self.store = new_store
        self.runner = self._new_runner(initial_messages=messages)
        await old_runner.close()
        await old_store.close()
        return new_id

    async def close(self) -> None:
        await self.runner.close()
        await self.store.close()


def _banner(chat: Chat) -> None:
    provider = "?"
    try:
        provider = provider_name_for_model(chat.model)
    except Exception:
        pass  # unknown
    tool_count = len(chat.agent.tools.tools())
    subs = ", ".join(s.name for s in chat.agent.subagents) or "none"
    print(f"{BOLD}{CYAN}curry-leaves{RESET} {DIM}· {chat.model} ({provider}) · {os.getcwd()}{RESET}")
    print(f"{DIM}tools: {tool_count} · subagents: {subs} · type /help for commands, /exit to quit{RESET}\n")

    key = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY" if provider == "openai" else None
    if key and not os.environ.get(key):
        print(f"{YELLOW}⚠ {key} is not set — turns will fail until you export it.{RESET}\n\n")


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
    ("clear", "clear the screen"),
    ("exit", "quit (also /quit, Ctrl-D)"),
]


async def _dispatch(chat: Chat, text: str) -> Union[str, None, object]:
    if not text.startswith("/"):
        return text  # not a command → run as a turn

    parts = text[1:].split(None, 1)
    cmd = parts[0] if parts else ""
    rest = parts[1].strip() if len(parts) > 1 else ""  # args after the command word

    if cmd in ("exit", "quit"):
        return _EXIT
    if cmd == "help":
        print(f"{DIM}commands:{RESET}")
        for name, desc in HELP:
            print(f"{DIM}  /{name.ljust(8)} {desc}{RESET}")
        print()
        return None
    if cmd == "reset":
        chat.reset()
        print(f"{DIM}(new conversation){RESET}\n")
        return None
    if cmd == "fork":
        upto_turn: Optional[int] = None
        if rest:
            try:
                upto_turn = int(rest)
            except ValueError:
                print(f"{DIM}usage: /fork [n] — n is the user turn to fork after (0-indexed){RESET}\n")
                return None
        old_id = chat.session
        new_id = await chat.fork(upto_turn)
        print(f"{DIM}(forked {old_id} → {new_id}, continuing here){RESET}\n")
        return None
    if cmd == "compact":
        sys.stdout.write(f"{DIM}compacting…{RESET}")
        sys.stdout.flush()
        oc = await chat.runner.compact(rest or None)
        sys.stdout.write("\r\x1b[K")  # clear the "compacting…" line
        if oc.compacted:
            print(f"{DIM}(compacted {oc.messages_before} → {oc.messages_after} messages){RESET}")
            print(f"{DIM}{oc.summary}{RESET}\n")
        else:
            print(f"{DIM}(nothing to compact — too little history yet){RESET}\n")
        return None
    if cmd == "tools":
        print(f"{DIM}tools:{RESET}")
        for t in chat.agent.tools.tools():
            tag = f"  {DIM}[deferred]{RESET}" if chat.agent.tools.is_deferred(t.name) else ""
            print(f"{DIM}  {t.name.ljust(14)}{RESET} {t.description[:70]}{tag}")
        if chat.agent.subagents:
            print(
                f"{DIM}  (also: search_tools, task, transfer — from {len(chat.agent.subagents)} subagents){RESET}"
            )
        print()
        return None
    if cmd == "skills":
        listing = chat.skills.teasers()
        if len(listing) == 0:
            print(f"{DIM}(no skills found under ~/.curry-leaves/skills or ./.curry-leaves/skills){RESET}\n")
        else:
            print(f"{DIM}skills:{RESET}")
            for name, desc in listing:
                print(f"{DIM}  {name.ljust(16)} {desc[:70]}{RESET}")
            print()
        return None
    if cmd == "model":
        print(f"{DIM}model: {chat.runner.model.id} ({chat.runner.model.provider}){RESET}\n")
        return None
    if cmd == "stats":
        u = chat.runner.usage
        print(
            f"{DIM}usage: in {u.input} · out {u.output} · cache_read {u.cache_read} · "
            f"total {u.total_tokens} tokens · ${u.cost.total:.4f}{RESET}\n"
        )
        return None
    if cmd == "auto":
        on = chat.toggle_auto()
        state = "on — contained changes (within this repo / known hosts) won't prompt" if on else "off"
        print(f"{DIM}(auto-approve {state}){RESET}\n")
        return None
    if cmd == "autonomous":
        on = chat.toggle_autonomous()
        state = "on — I'll self-drive after any initial questions" if on else "off"
        print(f"{DIM}(autonomous mode {state}){RESET}\n")
        return None
    if cmd == "clear":
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.flush()
        return None
    print(f"{DIM}unknown command: /{cmd} — try /help{RESET}\n")
    return None


async def _amain() -> None:
    # Populate real model metadata + pricing from models.dev (cached in the home dir, TTL'd).
    # Never throws — falls back to a stale cache, or an empty catalog if offline with no cache.
    from curry_leaves.catalog import load_catalog

    await load_catalog()
    model = resolve_default_model()
    prompter = Prompter()
    # The chat's host reuses `prompter` for its ask/approve prompts, so it's created first.
    chat = Chat(model, prompter)
    _banner(chat)

    prompt = f"{GREEN}you ›{RESET} "

    # question()-based loop (not an async iterator) so the host can reuse the one Prompter for
    # its own prompts mid-turn — only one question is ever active at a time.
    try:
        while True:
            try:
                raw = await prompter.question(prompt)
            except EOFError:
                break  # interface closed while awaiting (Ctrl-D)
            except KeyboardInterrupt:
                print(f"\n{DIM}(use /exit or Ctrl-D to quit){RESET}")
                continue
            line = raw.strip()
            if not line:
                continue
            action = await _dispatch(chat, line)
            if action is _EXIT:
                break
            if action is not None:
                assert isinstance(action, str)
                await run_turn(chat.runner, action)
    finally:
        await chat.close()
        print(f"\n{DIM}bye.{RESET}")


def main() -> None:
    try:
        asyncio.run(_amain())
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
