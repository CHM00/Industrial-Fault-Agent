from __future__ import annotations

import asyncio

from .base import SearchProvider, SearchProviderError


class DuckDuckGoSearchProvider(SearchProvider):
    name = "duckduckgo"

    async def search(
        self,
        query: str,
        *,
        max_results: int,
        timeout_seconds: int,
        max_source_chars: int,
    ):
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
        except ImportError as exc:
            raise SearchProviderError(
                "SEARCH_DEPENDENCY_MISSING",
                "缺少 duckduckgo-search 或 ddgs 依赖",
            ) from exc

        def call():
            with DDGS() as client:
                return list(client.text(query[:400], max_results=max_results))

        try:
            payload = await asyncio.wait_for(asyncio.to_thread(call), timeout=timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise SearchProviderError("SEARCH_TIMEOUT", "DuckDuckGo 搜索超时", retryable=True) from exc
        except Exception as exc:
            raise SearchProviderError(
                "SEARCH_FAILED",
                f"DuckDuckGo 搜索失败: {exc}",
                retryable=True,
            ) from exc

        sources = []
        for item in payload:
            snippet = item.get("body") or item.get("snippet") or ""
            source = self.source(
                title=item.get("title", ""),
                url=item.get("href") or item.get("url") or "",
                snippet=snippet,
                content=snippet,
                query=query,
                max_source_chars=max_source_chars,
            )
            if source:
                sources.append(source)
        return sources
