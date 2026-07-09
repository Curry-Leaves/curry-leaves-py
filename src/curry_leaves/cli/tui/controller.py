"""Chat — a small mutable holder for the live Agent + Runner the TUI drives.

`reset()` swaps in a fresh agent/runner in place, so the Textual App can keep a stable
reference (`chat`) and always read `chat.runner` for the current conversation.
"""

from __future__ import annotations

import os
import random
from datetime import datetime

from curry_leaves.agents import explore_agent, plan_agent
from curry_leaves.core.agent import Agent
from curry_leaves.permission import PermissionEngine, PermissionOptions, contained_approval
from curry_leaves.presets import coding_tools, web_tools
from curry_leaves.prompt import CODING_IDENTITY
from curry_leaves.providers.factory import provider_name_for_model
from curry_leaves.runner import RunConfig, Runner
from curry_leaves.session import SessionMeta, SessionStore, open_session
from curry_leaves.settings import add_global_approval, auto_hosts, global_approvals
from curry_leaves.skills import SkillRegistry
from curry_leaves.thinking import ThinkingConfig
from curry_leaves.util.paths import repo_root

from .host import TuiHost


def _build_agent(model: str) -> Agent:
    return Agent(
        model,
        name="curry-leaves",
        instructions=(
            "You are curry-leaves, a capable, general-purpose assistant. You can read and write files, "
            "run shell commands, search the codebase and the web, fetch pages, check the time, manage "
            "tasks, and delegate to subagents. Reach for whatever tool fits the job — code or not. Use "
            "tools to ground your work; verify instead of guessing; complete the task fully."
        ),
        # Every tool always-on (nothing deferred behind search_tools) — a true general-purpose kit.
        tools=[*coding_tools(), *web_tools()],
        subagents=[explore_agent(model), plan_agent(model)],
        auto_thinking=True,
    )


def _session_id() -> str:
    """A Hermes-style session id, e.g. 20260706_124623_417786."""
    d = datetime.now()
    date = f"{d.year:04d}{d.month:02d}{d.day:02d}"
    time = f"{d.hour:02d}{d.minute:02d}{d.second:02d}"
    micro = f"{d.microsecond // 1000:03d}{random.randint(0, 999):03d}"
    return f"{date}_{time}_{micro}"


class Chat:
    def __init__(self, model: str) -> None:
        self.model = model
        self.skills = SkillRegistry(discover=True)
        self.cwd = os.getcwd()
        self.session = _session_id()
        # Autonomous mode — the model self-drives (a prompt layer).
        self.autonomous = False
        # Auto-approve mode — contained changes (within repo / known host) skip the prompt.
        self.auto_approve = False

        provider = "?"
        try:
            provider = provider_name_for_model(model)
        except Exception:
            pass  # unknown model → leave as ?
        self.provider = provider
        self.store: SessionStore = open_session(
            self.session, SessionMeta(model=model, provider=provider, cwd=self.cwd)
        )
        # Interactive host — the App reads `host.current` to render ask/approval prompts.
        self.host = TuiHost()
        # Contained-auto-approval predicate, gated on the live `auto_approve` flag.
        contained = contained_approval(repo_root(self.cwd), auto_hosts())
        self.permission = PermissionEngine(
            PermissionOptions(
                global_approvals=global_approvals(),
                on_global_approve=add_global_approval,
                auto_approve=lambda tool, risk, args: self.auto_approve and contained(tool, risk, args),
            )
        )
        self.agent = _build_agent(model)
        self.runner = self._new_runner()

    def _new_runner(self) -> Runner:
        """Build a runner wired to the session store, interactive host, and permission gate."""
        return Runner(
            self.agent,
            RunConfig(
                skills=self.skills,
                thinking=ThinkingConfig(system=CODING_IDENTITY),
                store=self.store,
                host=self.host,
                permission=self.permission,
                autonomous=self.autonomous,
            ),
        )

    def toggle_autonomous(self) -> bool:
        """Flip autonomous (self-drive) mode live — conversation preserved. Returns the new state."""
        self.autonomous = not self.autonomous
        self.runner.set_autonomous(self.autonomous)
        return self.autonomous

    def toggle_auto(self) -> bool:
        """Flip auto-approve (contained changes skip the prompt). The engine reads the live flag."""
        self.auto_approve = not self.auto_approve
        return self.auto_approve

    def reset(self) -> None:
        # Fire-and-forget close of the old runner, mirroring the TS `void this.runner.close()`.
        import asyncio

        asyncio.ensure_future(self.runner.close())
        self.agent = _build_agent(self.model)
        # Fresh runner re-attaches the store to its new event host; mark the boundary in the transcript.
        self.store.mark("reset")
        self.runner = self._new_runner()

    async def close(self) -> None:
        """Tear down the runner and flush the session transcript."""
        await self.runner.close()
        await self.store.close()

    def api_key_var(self) -> str | None:
        """The env var that must be set for this provider, or None if unknown."""
        if self.provider == "anthropic":
            return "ANTHROPIC_API_KEY"
        if self.provider == "openai":
            return "OPENAI_API_KEY"
        return None
