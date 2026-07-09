"""Compaction — keep a long conversation inside the model's context window.

As a session grows, the message history eventually approaches the context window and the
next turn would overflow. Compaction replaces the older history with a single, information-
dense SUMMARY produced by the model, so work continues with no loss of continuity. Two modes:

  - automatic — the Runner watches context occupancy and compacts when it crosses a
                configurable fraction of the window (default 85%), before the next turn.
  - manual    — the user (or SDK caller) triggers it explicitly via `runner.compact()`
                (wired to `/compact` in the CLIs), optionally with focus instructions.

This mirrors the standard agent approach (e.g. Claude Code): a structured recap of intent,
decisions, files, commands, current state, and next steps — detailed enough that the agent
can resume as if nothing was dropped. The Runner owns WHEN; this module owns HOW.

A SEAM, like thinking: `Runner(agent, RunConfig(compaction=CompactionConfig(threshold=0.7, keep_last_messages=4)))`.
"""

from __future__ import annotations

import json
import math
from typing import Optional

from pydantic import BaseModel

from curry_leaves.core.messages import Content, Message, text_of, user_text
from curry_leaves.providers.base import Context, Model, Provider


class CompactionConfig(BaseModel):
    """Knobs a caller may pass via `RunConfig.compaction`. Everything is optional."""

    # Compact automatically when the window fills. Default true.
    auto: Optional[bool] = None
    # Fraction of the context window (0-1) that triggers auto-compaction. Default 0.85.
    threshold: Optional[float] = None
    # Recent messages to keep verbatim after the summary (snapped to a safe user boundary). Default 0.
    keep_last_messages: Optional[int] = None
    # Don't auto-compact below this many messages (avoids compacting tiny sessions). Default 6.
    min_messages: Optional[int] = None
    # Model (id or preference tier) used to summarize. Default: the agent's active model.
    model: Optional[str] = None
    # Extra guidance appended to the summary prompt (e.g. "focus on the auth refactor").
    instructions: Optional[str] = None


class CompactionOutcome(BaseModel):
    """What a compaction did — returned by `runner.compact()` and carried on the compaction event."""

    compacted: bool
    reason: str  # "auto" | "manual"
    messages_before: int
    messages_after: int
    tokens_before: int
    summary: str


_DEFAULT_AUTO = True
_DEFAULT_THRESHOLD = 0.85
_DEFAULT_KEEP_LAST_MESSAGES = 0
_DEFAULT_MIN_MESSAGES = 6

# The summarization rubric — the "best standard" structured recap.
SUMMARY_SYSTEM = (
    "You are summarizing a conversation between a user and an AI coding assistant so the assistant "
    "can continue seamlessly after older messages are dropped from its context. Capture EVERY detail "
    "needed to keep working with no loss of continuity.\n\n"
    "Write the summary under these headings (skip one only if genuinely empty):\n"
    "1. Task & intent — what the user wants, in their terms; explicit requirements and constraints.\n"
    "2. Key decisions & context — established facts, conventions, and choices (libraries, patterns, layout).\n"
    "3. Files & changes — files read or edited, with paths and a concise description of each change; "
    "include critical code shape (signatures, key snippets) that must not be forgotten.\n"
    "4. Commands & results — significant commands run and their outcomes (tests, builds, errors).\n"
    "5. Current state — what is done and verified vs. still in progress.\n"
    "6. Next steps — the immediate next action(s) and any open questions.\n\n"
    "Be specific and concrete: exact names, paths, and signatures over vague description. Do not add "
    "commentary, greetings, or questions — output only the summary."
)

_SUMMARY_REQUEST = (
    "Summarize the ACTUAL user/assistant conversation above, following the required format. "
    "Summarize only what the user and assistant actually said and did — never restate, describe, "
    "or summarize these summarization instructions themselves. If little of substance has happened "
    "yet, give a brief factual summary of that. Output only the summary."
)

# Below this many estimated tokens there's nothing worth compacting (a summary would be bigger).
_MIN_WORTH_TOKENS = 400

# Prefix on the injected summary message, so it's unmistakable in the transcript.
SUMMARY_PREAMBLE = (
    "[The earlier conversation was compacted to fit the context window. "
    "Here is a summary of everything so far — treat it as authoritative context and continue.]\n\n"
)


def _content_chars(content: list[Content]) -> int:
    """Rough token estimate (~4 chars/token) for content — good enough to avoid re-triggering."""
    n = 0
    for b in content:
        if b.type == "text":
            n += len(b.text)
        elif b.type == "thinking":
            n += len(b.thinking)
        elif b.type == "tool_call":
            n += len(json.dumps(b.arguments)) + len(b.name)
        elif b.type == "image":
            n += 1500  # images cost real tokens; a coarse flat estimate
    return n


def estimate_tokens(messages: list[Message]) -> int:
    """Estimate the token footprint of a message list (used to reset occupancy post-compaction)."""
    chars = 0
    for m in messages:
        chars += _content_chars(m.content)
    return math.ceil(chars / 4)


class Compactor:
    def __init__(self, config: Optional[CompactionConfig] = None) -> None:
        config = config or CompactionConfig()
        self.auto: bool = config.auto if config.auto is not None else _DEFAULT_AUTO
        self.threshold: float = (
            config.threshold if config.threshold is not None else _DEFAULT_THRESHOLD
        )
        self.keep_last_messages: int = max(
            0,
            config.keep_last_messages
            if config.keep_last_messages is not None
            else _DEFAULT_KEEP_LAST_MESSAGES,
        )
        self.min_messages: int = max(
            0,
            config.min_messages if config.min_messages is not None else _DEFAULT_MIN_MESSAGES,
        )
        # Model ref (id/tier) to summarize with; the Runner resolves it. None -> active model.
        self.model_ref: Optional[str] = config.model
        self._instructions: Optional[str] = config.instructions

    def should_auto(self, message_count: int, context_tokens: int, context_window: int) -> bool:
        """Whether auto-compaction should fire now, given current occupancy."""
        if not self.auto or context_window <= 0:
            return False
        if message_count < self.min_messages:
            return False
        return context_tokens >= context_window * self.threshold

    def worth_compacting(self, messages: list[Message]) -> bool:
        """
        Is there enough real conversation to be worth compacting? Below a small floor a summary would
        be LARGER than the history it replaces, so compacting is pointless (and risks the model
        summarizing its own instructions). Guards both manual and automatic paths.
        """
        return len(messages) >= 2 and estimate_tokens(messages) >= _MIN_WORTH_TOKENS

    async def compact(
        self,
        provider: Provider,
        model: Model,
        messages: list[Message],
        extra_instructions: Optional[str] = None,
    ) -> tuple[list[Message], str, int]:
        """
        Compact `messages`: summarize the leading portion and return the replacement list
        (summary message + any verbatim tail). Pure — the caller swaps its own array contents.
        Throws only if the summarization model call fails (the Runner catches and continues).

        Returns (new_messages, summary, kept_tail).
        """
        cut = self._cut_index(messages)
        if cut == 0:
            return messages, "", len(messages)  # nothing to summarize
        summary = await self._summarize(provider, model, messages[:cut], extra_instructions)
        tail = messages[cut:]
        new_messages: list[Message] = [user_text(SUMMARY_PREAMBLE + summary), *tail]
        return new_messages, summary, len(tail)

    async def _summarize(
        self,
        provider: Provider,
        model: Model,
        messages: list[Message],
        extra_instructions: Optional[str] = None,
    ) -> str:
        """Ask the model for a structured summary of `messages`."""
        system_prompt = [SUMMARY_SYSTEM]
        focus = "\n".join(f for f in [self._instructions, extra_instructions] if f)
        if focus:
            system_prompt.append(f"Additional focus requested by the user:\n{focus}")

        ctx = Context(
            system_prompt=system_prompt,
            messages=[*messages, user_text(_SUMMARY_REQUEST)],
            tools=[],
        )
        out = ""
        async for ev in provider.stream(ctx, model):
            if ev.type == "done":
                out = text_of(ev.message.content)
        return out.strip()

    def _cut_index(self, messages: list[Message]) -> int:
        """
        Where to cut for the verbatim tail. Snaps FORWARD to the next user-message boundary so the
        kept tail never starts with a dangling tool_result (a user turn is always a clean boundary:
        every prior tool_call already has its result). Returns len(messages) to keep no tail.
        """
        if self.keep_last_messages <= 0:
            return len(messages)
        target = max(0, len(messages) - self.keep_last_messages)
        for i in range(target, len(messages)):
            if messages[i].role == "user":
                return i
        return len(messages)
