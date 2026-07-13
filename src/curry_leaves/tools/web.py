"""Web tools: `web_fetch` (HTTP GET + HTML-to-text) and `web_search` (DuckDuckGo HTML
scrape). No headless browser, no bs4 — a hand-rolled regex strip is enough for reading
plain pages and DDG's server-rendered HTML results.
"""

from __future__ import annotations

import asyncio
import re
import urllib.parse
from typing import TYPE_CHECKING, Any

import httpx
import pydantic

from curry_leaves.core.blobs import truncate_with_blob
from curry_leaves.core.tools import Risk, Tool, ToolResult

if TYPE_CHECKING:
    from curry_leaves.providers.base import Context

_UA = "Mozilla/5.0 (compatible; curry-leaves/1.0)"
_MAX_FETCH_CHARS = 10_000


def _html_to_text(s: str) -> str:
    s = re.sub(r"<(script|style|noscript)[\s\S]*?</\1>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", "", s)
    s = s.replace("&nbsp;", " ")
    s = s.replace("&amp;", "&")
    s = s.replace("&lt;", "<")
    s = s.replace("&gt;", ">")
    s = s.replace("&quot;", '"')
    s = s.replace("&#39;", "'")
    s = re.sub(r"\n\s*\n\s*\n+", "\n\n", s)
    return s.strip()


def _parse_ddg(page: str, limit: int) -> list[tuple[str, str, str]]:
    results: list[tuple[str, str, str]] = []
    re_result = re.compile(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)</a>')
    for m in re_result.finditer(page):
        if len(results) >= limit:
            break
        href = m.group(1)
        uddg = re.search(r"[?&]uddg=([^&]+)", href)
        if uddg:
            try:
                href = urllib.parse.unquote(uddg.group(1))
            except Exception:
                pass  # keep raw href
        title = _html_to_text(m.group(2))
        # snippet: a result__snippet within the next stretch of the page
        after = page[m.start() : m.start() + 2000]
        sn = re.search(r'class="result__snippet"[^>]*>([\s\S]*?)</a>', after)
        snippet = _html_to_text(sn.group(1)) if sn else ""
        results.append((title, href, snippet))
    return results


class FetchArgs(pydantic.BaseModel):
    url: str = pydantic.Field(description="The http(s) URL to fetch.")


class WebFetchTool:
    """Structurally satisfies the `Tool` protocol (see core/tools.py)."""

    name = "web_fetch"
    description = (
        "Fetch a URL over HTTP(S) and return its text content (HTML stripped to readable text)."
    )
    schema: type[pydantic.BaseModel] = FetchArgs
    risk: Risk | None = "network"
    timeout: float | None = None

    async def run(self, args: FetchArgs, ctx: "Context", signal: asyncio.Event) -> ToolResult:
        if not re.match(r"^https?://", args.url, flags=re.IGNORECASE):
            return ToolResult(content=f"Not an http(s) URL: {args.url}", is_error=True)

        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                resp = await client.get(args.url, headers={"User-Agent": _UA})
                if resp.status_code >= 400:
                    raise httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}", request=resp.request, response=resp
                    )
                ctype = resp.headers.get("content-type") or ""
                body = resp.text
        except Exception as e:  # noqa: BLE001 - mirrors TS catch(e) at the fetch boundary
            return ToolResult(content=f"Failed to fetch {args.url}: {e}", is_error=True)

        text = _html_to_text(body) if "html" in ctype else body
        text = truncate_with_blob(
            text,
            _MAX_FETCH_CHARS,
            ctx.blobs,
            stored=lambda bid, total: f"... [truncated — full page at artifact://{bid}]",
        )
        return ToolResult(content=text)


def web_fetch_tool() -> Tool[Any]:
    return WebFetchTool()


class SearchArgs(pydantic.BaseModel):
    query: str = pydantic.Field(description="The search query.")
    max_results: int = pydantic.Field(default=5, description="How many results to return.")


class WebSearchTool:
    name = "web_search"
    description = "Search the web (DuckDuckGo) and return the top results as title, URL, and snippet."
    schema: type[pydantic.BaseModel] = SearchArgs
    risk: Risk | None = "network"
    timeout: float | None = None

    async def run(self, args: SearchArgs, ctx: "Context", signal: asyncio.Event) -> ToolResult:
        try:
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(args.query)}"
            async with httpx.AsyncClient(follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": _UA})
                if resp.status_code >= 400:
                    raise httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}", request=resp.request, response=resp
                    )
                page = resp.text
        except Exception as e:  # noqa: BLE001 - mirrors TS catch(e) at the fetch boundary
            return ToolResult(content=f"Search failed: {e}", is_error=True)

        results = _parse_ddg(page, args.max_results)
        if len(results) == 0:
            return ToolResult(content=f"No results for '{args.query}'.")
        return ToolResult(
            content="\n\n".join(f"{t}\n{u}\n{s[:200]}" for t, u, s in results)
        )


def web_search_tool() -> Tool[Any]:
    return WebSearchTool()
