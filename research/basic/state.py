from __future__ import annotations

from typing import TypedDict

class BasicResearchState(TypedDict):
    task_id: str
    research_topic: str
    research_depth: int
    current_query: str
    summary: str
    sources: list[dict]
    latest_sources: list[dict]
    query_count: int
    search_count: int
    warnings: list[dict]
    stop_reason: str
