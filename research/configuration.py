from __future__ import annotations

import os
import warnings

from pydantic import BaseModel, Field

from .contracts import ResearchOptions


def _first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            if name == "trivily_key":
                warnings.warn(
                    "trivily_key is deprecated; use TAVILY_API_KEY",
                    DeprecationWarning,
                    stacklevel=2,
                )
            return value
    return default


class ResearchSettings(BaseModel):
    llm_api_key: str = ""
    llm_base_url: str = "https://api.siliconflow.cn/v1"
    llm_model: str = "deepseek-ai/DeepSeek-V4-Flash"
    tavily_api_key: str = ""
    perplexity_api_key: str = ""
    searxng_url: str = "http://localhost:8888"
    fetch_full_page: bool = False
    max_results_per_search: int = Field(default=3, ge=1, le=10)
    total_timeout_seconds: int = Field(default=120, ge=10, le=600)

    @classmethod
    def from_env(cls) -> "ResearchSettings":
        return cls(
            llm_api_key=_first_env("LLM_API_KEY", "ARK_API_KEY", "OPENAI_API_KEY"),
            llm_base_url=_first_env(
                "LLM_BASE_URL",
                "ARK_BASE_URL",
                "OPENAI_BASE_URL",
                default="https://api.siliconflow.cn/v1",
            ),
            llm_model=_first_env(
                "LLM_MODEL",
                "LOCAL_LLM",
                default="deepseek-ai/DeepSeek-V4-Flash",
            ),
            tavily_api_key=_first_env("TAVILY_API_KEY", "trivily_key"),
            perplexity_api_key=_first_env("PERPLEXITY_API_KEY"),
            searxng_url=_first_env("SEARXNG_URL", default="http://localhost:8888"),
            fetch_full_page=os.environ.get("FETCH_FULL_PAGE", "false").lower() == "true",
            max_results_per_search=int(os.environ.get("MAX_RESULTS_PER_SEARCH", "3")),
            total_timeout_seconds=int(os.environ.get("RESEARCH_TOTAL_TIMEOUT_SECONDS", "120")),
        )


def options_from_state(state: dict) -> ResearchOptions:
    fields = ResearchOptions.model_fields
    return ResearchOptions(**{name: state[name] for name in fields if name in state})
