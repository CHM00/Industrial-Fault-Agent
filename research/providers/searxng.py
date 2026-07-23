from __future__ import annotations

import httpx

from .base import SearchProvider, SearchProviderError


class SearXNGSearchProvider(SearchProvider):
    name = "searxng"

    async def search(
        self,
        query: str,
        *,
        max_results: int,
        timeout_seconds: int,
        max_source_chars: int,
    ):
        url = self.settings.searxng_url.rstrip("/") + "/search"
        try:
            async with httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=False) as client:
                response = await client.get(url, params={"q": query[:400], "format": "json"})
                response.raise_for_status()
                payload = response.json()
        except httpx.TimeoutException as exc:
            raise SearchProviderError("SEARCH_TIMEOUT", "SearXNG 搜索超时", retryable=True) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise SearchProviderError(
                "SEARCH_FAILED",
                f"SearXNG 搜索失败: {exc}",
                retryable=True,
            ) from exc

        sources = []
        for item in payload.get("results", [])[:max_results]:
            snippet = item.get("content") or item.get("snippet") or ""
            source = self.source(
                title=item.get("title", ""),
                url=item.get("url") or item.get("link") or "",
                snippet=snippet,
                content=snippet,
                query=query,
                max_source_chars=max_source_chars,
            )
            if source:
                sources.append(source)
        return sources
