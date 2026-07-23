from __future__ import annotations

from abc import ABC, abstractmethod

from ..configuration import ResearchSettings
from ..contracts import ExternalSource, SearchAPI


class SearchProviderError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class SearchProvider(ABC):
    name: SearchAPI

    def __init__(self, settings: ResearchSettings):
        self.settings = settings

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        max_results: int,
        timeout_seconds: int,
        max_source_chars: int,
    ) -> list[ExternalSource]:
        raise NotImplementedError

    def source(
        self,
        *,
        title: str,
        url: str,
        snippet: str,
        content: str,
        query: str,
        max_source_chars: int,
    ) -> ExternalSource | None:
        try:
            return ExternalSource(
                title=(title or url or "未命名来源").strip()[:500],
                url=url,
                snippet=(snippet or "").strip()[:2000],
                content=(content or snippet or "").strip()[:max_source_chars],
                provider=self.name,
                query=query[:400],
            )
        except (TypeError, ValueError):
            return None


def create_search_provider(name: SearchAPI, settings: ResearchSettings) -> SearchProvider:
    if name == "tavily":
        from .tavily import TavilySearchProvider

        return TavilySearchProvider(settings)
    if name == "duckduckgo":
        from .duckduckgo import DuckDuckGoSearchProvider

        return DuckDuckGoSearchProvider(settings)
    if name == "searxng":
        from .searxng import SearXNGSearchProvider

        return SearXNGSearchProvider(settings)
    if name == "perplexity":
        from .perplexity import PerplexitySearchProvider

        return PerplexitySearchProvider(settings)
    raise SearchProviderError("SEARCH_PROVIDER_UNSUPPORTED", f"不支持的搜索服务: {name}")
