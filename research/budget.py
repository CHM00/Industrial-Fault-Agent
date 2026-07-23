from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .contracts import ResearchTask


def allocate_task_budgets(
    tasks: list[ResearchTask],
    *,
    research_depth: int,
    max_research_tasks: int,
    max_total_searches: int,
) -> list[tuple[ResearchTask, int]]:
    """Allocate provider-call attempts fairly across prioritized tasks."""
    ordered = sorted(tasks, key=lambda task: (-task.priority, task.id))
    actual_count = min(len(ordered), max_research_tasks, max_total_searches)
    selected = ordered[:actual_count]
    if not selected:
        return []

    attempts = [1] * actual_count
    remaining = max_total_searches - actual_count
    while remaining > 0 and any(value < research_depth for value in attempts):
        for index in range(actual_count):
            if remaining == 0:
                break
            if attempts[index] < research_depth:
                attempts[index] += 1
                remaining -= 1
    return list(zip(selected, attempts))


@dataclass
class SearchBudget:
    total_limit: int
    task_limits: dict[str, int]
    total_used: int = 0
    task_used: dict[str, int] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def acquire(self, task_id: str) -> bool:
        async with self._lock:
            task_limit = self.task_limits.get(task_id, self.total_limit)
            used = self.task_used.get(task_id, 0)
            if self.total_used >= self.total_limit or used >= task_limit:
                return False
            self.total_used += 1
            self.task_used[task_id] = used + 1
            return True

    def used_by(self, task_id: str) -> int:
        return self.task_used.get(task_id, 0)
