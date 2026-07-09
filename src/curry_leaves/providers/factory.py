"""Provider inference — build a Provider instance from a model id or provider name.

Lets the minimal API stay one-line: `Agent(model="claude-sonnet-4-5")` — the provider
is inferred.

Resolution order for the provider NAME:
  1. an explicit `Model.provider`
  2. `$CURRY_LEAVES_PROVIDER`
  3. the catalog (a known model id -> its provider)
  4. an id-prefix heuristic (claude- -> anthropic, gpt-/o1- -> openai)
"""

from __future__ import annotations

import os
from typing import Union

from curry_leaves.catalog import lookup
from curry_leaves.providers.anthropic import AnthropicProvider
from curry_leaves.providers.base import Model, Provider
from curry_leaves.providers.openai import OllamaProvider, OpenAIProvider

# Local-model families served by Ollama (OpenAI-compatible). Prefix-matched.
OLLAMA_PREFIXES = ["gemma", "llama", "qwen", "mistral", "mixtral", "phi", "deepseek", "codellama", "ollama"]


def provider_for(name: str) -> Provider:
    key = name.lower().strip()
    if key == "anthropic":
        return AnthropicProvider()
    if key == "openai":
        return OpenAIProvider()
    if key == "ollama":
        return OllamaProvider()
    raise ValueError(
        f"Unknown provider '{name}'. Known: anthropic, openai, ollama. Pass an explicit provider instance for a custom one."
    )


def provider_name_for_model(model: Union[Model, str]) -> str:
    if not isinstance(model, str):
        return model.provider

    env = os.environ.get("CURRY_LEAVES_PROVIDER")
    if env:
        return env

    info = lookup(model)
    if info:
        return info.provider

    mid = model.lower()
    if mid.startswith("claude"):
        return "anthropic"
    if mid.startswith("gpt") or mid.startswith("o1") or mid.startswith("o3") or mid.startswith("o4"):
        return "openai"
    if any(mid.startswith(p) for p in OLLAMA_PREFIXES):
        return "ollama"
    raise ValueError(
        f"Cannot infer a provider from model '{model}'. Set $CURRY_LEAVES_PROVIDER, add it to the "
        "catalog, or pass an explicit provider to the Agent."
    )


def infer_provider(model: Union[Model, str]) -> Provider:
    return provider_for(provider_name_for_model(model))
