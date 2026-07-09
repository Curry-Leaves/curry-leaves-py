"""Auto-thinking — size reasoning effort per task with a tiny classifier.

Spending the same reasoning budget on "rename this var" and "redesign auth" is wasteful
one way and weak the other. Before a turn, a CHEAP model reads the user's prompt and
buckets its difficulty, which maps to a concrete reasoning Effort. Any failure (no model,
junk output) falls back to a middle effort so the turn still runs.

This is a SEAM: `AutoThinking` is the built-in default, but the Runner accepts any
`Classifier`. Attach your own (a different prompt, model, or logic — even one backed by
a full Agent) via `Runner(agent, thinking=my_classifier)`; by default it uses ours.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, Optional, Protocol

from pydantic import BaseModel

from curry_leaves.core.messages import text_of, user_text
from curry_leaves.providers.base import Context, Model, Provider


class Effort(str, Enum):
    MINIMAL = "minimal"  # trivial: a question, a one-line edit — no extended thinking
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"  # architecture, tricky algorithms, multi-file design


class Classifier(Protocol):
    """The pluggable contract: given the user's prompt, return a reasoning Effort. Implement it
    however you like (a heuristic, a keyword rule, a call to any model, or a wrapped Agent).
    """

    async def classify(self, prompt_text: str) -> Effort: ...


@dataclass
class ThinkingConfig:
    """The single "thinking" seam a consumer attaches via `Runner(agent, thinking=...)`.
    It carries BOTH the persona and the difficulty classifier — the framework's two opinions
    in one place. Anything omitted falls back to the built-in default:

      - `system`   — the identity/persona prompt used as the top layer of every system prompt.
                     Omitted → the neutral DEFAULT_IDENTITY (or an identity.md override).
      - `classify` — sizes reasoning effort per turn. Omitted → the built-in AutoThinking
                     (active only when the agent sets `auto_thinking`). Provided → always active.
    """

    system: Optional[str] = None
    classify: Optional[Callable[[str], Awaitable[Effort]]] = None


def thinking_budget(effort: Effort) -> int:
    """Anthropic extended-thinking token budget for an effort (0 = off)."""
    return {
        Effort.MINIMAL: 0,
        Effort.LOW: 2048,
        Effort.MEDIUM: 8192,
        Effort.HIGH: 16384,
    }[effort]


BUCKET: dict[str, Effort] = {
    "trivial": Effort.MINIMAL,
    "moderate": Effort.LOW,
    "hard": Effort.MEDIUM,
    "very_hard": Effort.HIGH,
}

# The built-in, task-agnostic difficulty rubric. Override via AutoThinkingOptions.system.
DEFAULT_CLASSIFIER_SYSTEM = (
    "You rate the DIFFICULTY of a task for an AI agent. Reply with EXACTLY ONE word:\n"
    "- trivial   — a simple question, a lookup, or a one-step action\n"
    "- moderate  — a standard, clearly-bounded task\n"
    "- hard      — several steps, some planning, non-obvious reasoning\n"
    "- very_hard — open-ended, intricate, or ambiguous; needs deep reasoning\n"
    "Output only the single word."
)

MAX_PROMPT_CHARS = 2000


class AutoThinkingOptions(BaseModel):
    """Override the classifier's system prompt (e.g. a domain-specific difficulty rubric)."""

    system: Optional[str] = None
    # Max chars of the user prompt sent to the classifier.
    max_prompt_chars: Optional[int] = None


class AutoThinking:
    """The default Classifier: a cheap one-word difficulty call to a (usually small) model."""

    def __init__(
        self,
        provider: Provider,
        model: Model,
        options: Optional[AutoThinkingOptions] = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.options = options if options is not None else AutoThinkingOptions()

    async def classify(self, prompt_text: str) -> Effort:
        """Bucket the prompt's difficulty into an Effort. Never throws — falls back to MEDIUM."""
        limit = (
            self.options.max_prompt_chars
            if self.options.max_prompt_chars is not None
            else MAX_PROMPT_CHARS
        )
        ctx = Context(
            system_prompt=[self.options.system or DEFAULT_CLASSIFIER_SYSTEM],
            messages=[user_text(prompt_text[:limit])],
            tools=[],
        )
        try:
            out = ""
            async for ev in self.provider.stream(ctx, self.model):
                if ev.type == "done":
                    out = text_of(ev.message.content)
            return parse_effort(out)
        except Exception:
            return Effort.MEDIUM


def parse_effort(text: str) -> Effort:
    low = text.strip().lower()
    # most-specific first: "hard" is a substring of "very_hard"
    for word in ("very_hard", "hard", "moderate", "trivial"):
        if word in low:
            return BUCKET[word]
    return Effort.MEDIUM
