from __future__ import annotations

import asyncio

from .base import SearchProvider, SearchProviderError


class TavilySearchProvider(SearchProvider):
    name = "tavily"

    async def search(
        self,
        query: str,
        *,
        max_results: int,
        timeout_seconds: int,
        max_source_chars: int,
    ):
        if not self.settings.tavily_api_key:
            raise SearchProviderError(
                "SEARCH_AUTH_MISSING",
                "未配置 TAVILY_API_KEY",
                retryable=False,
        )
        try:
            from tavily import AsyncTavilyClient
        except ImportError as exc:
            raise SearchProviderError(
                "SEARCH_DEPENDENCY_MISSING",
                "缺少 tavily-python 依赖",
            ) from exc

        try:
            client = AsyncTavilyClient(api_key=self.settings.tavily_api_key)
            payload = await asyncio.wait_for(
                client.search(
                    query[:400],
                    max_results=max_results,
                    include_raw_content=self.settings.fetch_full_page,
                ),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise SearchProviderError("SEARCH_TIMEOUT", "Tavily 搜索超时", retryable=True) from exc
        except Exception as exc:
            text = str(exc)
            retryable = any(token in text.lower() for token in ("timeout", "rate", "429", "503"))
            code = "SEARCH_RATE_LIMITED" if "429" in text or "rate" in text.lower() else "SEARCH_FAILED"
            raise SearchProviderError(code, f"Tavily 搜索失败: {text}", retryable=retryable) from exc

        sources = []
        for item in payload.get("results", []):
            source = self.source(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                content=item.get("raw_content") or item.get("content", ""),
                query=query,
                max_source_chars=max_source_chars,
            )
            if source:
                sources.append(source)
        return sources
