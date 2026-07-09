"""Model catalog — real context windows + pricing for known models.

The single source of truth is models.dev (https://models.dev/api.json), a community-
maintained database of model metadata + pricing. There is deliberately NO hardcoded/seed
catalog: fabricated numbers are worse than none. `load_catalog()` fetches models.dev,
caches the raw JSON in the home dir, and refreshes on a TTL — a fresh cache is used as-is;
a stale or missing one triggers a re-fetch, falling back to the stale cache if the network
is down, and to an empty catalog only if there's nothing cached at all.

`lookup`/`resolve_model`/`compute_cost` stay synchronous by reading the in-memory CATALOG,
which load_catalog() refreshes in place. Call `await load_catalog()` once at startup. A
caller may pre-register local models (e.g. Ollama tags, which models.dev doesn't list) by
mutating CATALOG before load_catalog() — the merge preserves them. Prices are USD per 1M.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import httpx
from pydantic import BaseModel

from curry_leaves.core.messages import Cost, Usage
from curry_leaves.providers.base import Model, make_model
from curry_leaves.util.paths import home, join


class ModelInfo(BaseModel):
    id: str
    provider: str
    context_window: int
    max_output_tokens: int
    supports_thinking: Optional[bool] = None
    # USD per 1,000,000 tokens
    price_input: Optional[float] = None
    price_output: Optional[float] = None
    price_cache_read: Optional[float] = None
    price_cache_write: Optional[float] = None


# Live, in-memory catalog. Empty until load_catalog() populates it from models.dev.
CATALOG: dict[str, ModelInfo] = {}


def lookup(model_id: str) -> Optional[ModelInfo]:
    return CATALOG.get(model_id)


# ── models.dev fetch + cache ─────────────────────────────────────────────────

MODELS_DEV_URL = "https://models.dev/api.json"
DEFAULT_TTL_MS = 24 * 60 * 60 * 1000  # 24h


class LoadCatalogOptions(BaseModel):
    # Where to cache models.dev's api.json. Default: <home>/models.json
    path: Optional[str] = None
    # How long a cached file stays fresh before re-fetching. Default: 24h.
    ttl_ms: Optional[int] = None
    # Override the models.dev endpoint.
    url: Optional[str] = None
    # Re-fetch even if the cache is still fresh.
    force: bool = False


# The models.dev api.json shape, narrowed to the fields we consume, is handled as
# plain dicts (RawModel/RawProvider in TS) — no pydantic model needed since we only
# ever read via .get() and immediately flatten.

# Providers preferred on id collisions (first-party over routers/aggregators).
PROVIDER_PRIORITY = ["anthropic", "openai", "google", "xai", "mistral", "deepseek", "meta"]


def _priority_of(provider: str) -> int:
    try:
        return PROVIDER_PRIORITY.index(provider)
    except ValueError:
        return len(PROVIDER_PRIORITY)


def _flatten(api: dict[str, Any]) -> dict[str, ModelInfo]:
    """Flatten models.dev's {provider:{models:{id:{...}}}} into our flat, id-keyed shape."""
    out: dict[str, ModelInfo] = {}
    chosen_prio: dict[str, int] = {}
    for provider, entry in api.items():
        models = (entry or {}).get("models")
        if not models:
            continue
        prio = _priority_of(provider)
        for model_id, m in models.items():
            prev = chosen_prio.get(model_id)
            if prev is not None and prev <= prio:
                continue  # keep the higher-priority provider
            limit = m.get("limit") or {}
            cost = m.get("cost") or {}
            out[model_id] = ModelInfo(
                id=model_id,
                provider=provider,
                context_window=limit.get("context") or 0,
                max_output_tokens=limit.get("output") or 0,
                supports_thinking=True if m.get("reasoning") else None,
                price_input=cost.get("input"),
                price_output=cost.get("output"),
                price_cache_read=cost.get("cache_read"),
                price_cache_write=cost.get("cache_write"),
            )
            chosen_prio[model_id] = prio
    return out


def _cache_path(opts: LoadCatalogOptions) -> str:
    return opts.path if opts.path is not None else join(home(), "models.json")


def _read_cache(file: str) -> Optional[dict[str, Any]]:
    try:
        with open(file, encoding="utf8") as f:
            data: Any = json.load(f)
        return data  # type: ignore[no-any-return]
    except Exception:
        return None  # missing or corrupt — treat as no cache


def _write_cache(file: str, raw: dict[str, Any]) -> None:
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w", encoding="utf8") as f:
        json.dump(raw, f)


async def _fetch_models_dev(url: str) -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        res = await client.get(url)
    if res.status_code < 200 or res.status_code >= 300:
        raise Exception(f"models.dev fetch failed: {res.status_code} {res.reason_phrase}")
    data: Any = res.json()
    return data  # type: ignore[no-any-return]


async def load_catalog(opts: Optional[LoadCatalogOptions] = None) -> dict[str, ModelInfo]:
    """Load model metadata from models.dev into the in-memory CATALOG.

    Cache policy: if the cache file exists and is younger than `ttl_ms`, use it as-is.
    Otherwise (stale or missing) re-fetch and rewrite the cache; if the fetch fails, fall
    back to the stale cache if one exists. Fetched entries are merged into CATALOG in place,
    so any pre-registered local models are preserved. Call once at startup.
    """
    opts = opts if opts is not None else LoadCatalogOptions()
    file = _cache_path(opts)
    ttl = opts.ttl_ms if opts.ttl_ms is not None else DEFAULT_TTL_MS
    fresh = (
        not opts.force
        and os.path.exists(file)
        and (time.time() * 1000 - os.stat(file).st_mtime * 1000) < ttl
    )

    raw: Optional[dict[str, Any]] = _read_cache(file) if fresh else None
    if raw is None:
        # Stale, missing, corrupt, or forced — go to the network.
        try:
            raw = await _fetch_models_dev(opts.url if opts.url is not None else MODELS_DEV_URL)
            _write_cache(file, raw)
        except Exception:
            raw = _read_cache(file)  # network down — accept a stale cache if we have one

    if raw is not None:
        CATALOG.update(_flatten(raw))  # merge fetched into CATALOG in place
    return CATALOG


def resolve_model(
    ref: str,
    preferences: Optional[dict[str, str]] = None,
    provider: Optional[str] = None,
) -> Model:
    """Resolve a model reference to a concrete Model. `ref` is a preference name (looked up
    in `preferences`, e.g. "fast"/"plan") OR a literal model id. After the preference map,
    catalog metadata is applied; unknown ids fall back to defaults.
    """
    model_id = (preferences or {}).get(ref, ref)
    info = CATALOG.get(model_id)
    if info:
        return make_model(
            model_id,
            provider if provider is not None else info.provider,
            max_output_tokens=info.max_output_tokens,
            context_window=info.context_window,
            supports_thinking=info.supports_thinking if info.supports_thinking is not None else False,
        )
    if provider is None:
        # Same inference the Agent itself uses (env → catalog → id prefix), so an unknown id
        # like "claude-…" isn't mislabeled "openai" just because the catalog wasn't loaded.
        from curry_leaves.providers.factory import provider_name_for_model  # local: avoids import cycle

        try:
            provider = provider_name_for_model(model_id)
        except ValueError:
            provider = "openai"
    return make_model(model_id, provider)


def compute_cost(usage: Usage, model_id: str) -> Cost:
    """Turn token usage into a Cost using the catalog's per-million prices."""
    info = CATALOG.get(model_id)
    empty = Cost(input=0, output=0, cache_read=0, cache_write=0, total=0)
    if not info:
        return empty
    m = 1_000_000
    input_ = (usage.input / m) * (info.price_input or 0)
    output = (usage.output / m) * (info.price_output or 0)
    cache_read = (usage.cache_read / m) * (info.price_cache_read or 0)
    cache_write = (usage.cache_write / m) * (info.price_cache_write or 0)
    return Cost(
        input=input_,
        output=output,
        cache_read=cache_read,
        cache_write=cache_write,
        total=input_ + output + cache_read + cache_write,
    )
