from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Iterable, Literal, TypeVar
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

ModelT = TypeVar("ModelT", bound=BaseModel)

ResearchMode = Literal["off", "basic", "supervisory", "auto"]
EffectiveResearchMode = Literal["off", "basic", "supervisory"]
SearchAPI = Literal["tavily", "duckduckgo", "searxng", "perplexity"]
ResearchStatus = Literal["success", "partial", "failed", "skipped"]

_TRACKING_PARAMS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def normalize_url(url: str) -> str:
    """Return a stable URL used for validation and deduplication."""
    raw = (url or "").strip()
    parts = urlsplit(raw)
    scheme = parts.scheme.lower()
    hostname = (parts.hostname or "").lower()
    if scheme not in {"http", "https"} or not hostname:
        raise ValueError(f"unsupported source URL: {url!r}")

    port = parts.port
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{hostname}:{port}"
    else:
        netloc = hostname

    path = parts.path or "/"
    if path != "/":
        path = path.rstrip("/")
    query = urlencode(
        sorted(
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if not key.lower().startswith("utm_") and key.lower() not in _TRACKING_PARAMS
        )
    )
    return urlunsplit((scheme, netloc, path, query, ""))


class ResearchWarning(BaseModel):
    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    message: str
    task_id: str | None = None
    retryable: bool = False


class ExternalSource(BaseModel):
    source_id: str = ""
    title: str
    url: HttpUrl
    snippet: str = ""
    content: str = ""
    provider: SearchAPI
    query: str
    retrieved_at: datetime = Field(default_factory=utc_now)

    @property
    def normalized_url(self) -> str:
        return normalize_url(str(self.url))


class ResearchTask(BaseModel):
    id: str = Field(pattern=r"^task_[1-9][0-9]*$")
    description: str = Field(min_length=3, max_length=500)
    priority: int = Field(default=3, ge=1, le=5)


class ResearchTaskPlan(BaseModel):
    tasks: list[ResearchTask] = Field(min_length=1, max_length=5)


class ResearchTaskResult(BaseModel):
    task_id: str
    description: str
    summary: str = ""
    sources: list[ExternalSource] = Field(default_factory=list)
    query_count: int = Field(default=0, ge=0)
    search_count: int = Field(default=0, ge=0)
    status: Literal["success", "partial", "failed"] = "success"
    warnings: list[ResearchWarning] = Field(default_factory=list)


class ExternalResearchResult(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    requested_mode: ResearchMode
    effective_mode: EffectiveResearchMode
    status: ResearchStatus
    summary: str = ""
    sources: list[ExternalSource] = Field(default_factory=list)
    task_results: list[ResearchTaskResult] = Field(default_factory=list)
    query_count: int = Field(default=0, ge=0)
    search_count: int = Field(default=0, ge=0)
    warnings: list[ResearchWarning] = Field(default_factory=list)
    elapsed_ms: int = Field(default=0, ge=0)


class ResearchOptions(BaseModel):
    model_config = ConfigDict(validate_default=True)

    research_mode: ResearchMode = "basic"
    research_depth: int = Field(
        default_factory=lambda: _env_int("MAX_WEB_RESEARCH_LOOPS", 2), ge=1, le=5
    )
    max_research_tasks: int = Field(
        default_factory=lambda: _env_int("MAX_RESEARCH_TASKS", 3), ge=1, le=5
    )
    max_total_searches: int = Field(
        default_factory=lambda: _env_int("MAX_TOTAL_SEARCHES", 6), ge=1, le=15
    )
    # None keeps the deployment-level default; 0 explicitly disables the
    # overall research deadline for long-running web research sessions.
    research_timeout_seconds: int | None = Field(default=None, ge=0, le=3600)
    search_api: SearchAPI = Field(default_factory=lambda: os.environ.get("SEARCH_API", "tavily"))
    search_timeout_seconds: int = Field(default=20, ge=3, le=60)
    search_max_retries: int = Field(default=1, ge=0, le=3)
    max_source_chars: int = Field(default=4000, ge=500, le=10000)
    max_external_context_chars: int = Field(default=12000, ge=2000, le=30000)


def models_to_state(items: Iterable[BaseModel]) -> list[dict]:
    """Convert domain models to JSON-safe values before writing graph state."""
    return [item.model_dump(mode="json") for item in items]


def models_from_state(model_type: type[ModelT], items: Iterable[ModelT | dict]) -> list[ModelT]:
    """Restore validated domain models at a graph node or subsystem boundary."""
    return [
        item if isinstance(item, model_type) else model_type.model_validate(item)
        for item in items
    ]


def deduplicate_sources(sources: list[ExternalSource]) -> list[ExternalSource]:
    """Deduplicate sources by normalized URL and assign stable source IDs."""
    unique: list[ExternalSource] = []
    seen: set[str] = set()
    for source in sources:
        try:
            key = source.normalized_url
        except ValueError:
            continue
        if key in seen:
            continue
        seen.add(key)
        unique.append(source.model_copy(update={"source_id": f"source_{len(unique) + 1}"}))
    return unique


def warning(
    code: str,
    message: str,
    *,
    task_id: str | None = None,
    retryable: bool = False,
) -> ResearchWarning:
    return ResearchWarning(code=code, message=message, task_id=task_id, retryable=retryable)
