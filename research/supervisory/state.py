from __future__ import annotations

from typing import TypedDict

class SupervisoryResearchState(TypedDict):
    fault_input: str
    tasks: list[dict]
    task_depths: dict[str, int]
    task_results: list[dict]
    analysis: str
    final_synthesis: str
    warnings: list[dict]
