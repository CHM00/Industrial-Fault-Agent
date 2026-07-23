import unittest

from research.budget import allocate_task_budgets
from research.contracts import ResearchTask


class ResearchBudgetTests(unittest.TestCase):
    def test_budget_is_distributed_without_exceeding_total(self):
        tasks = [
            ResearchTask(id="task_1", description="high priority task", priority=5),
            ResearchTask(id="task_2", description="medium priority task", priority=3),
            ResearchTask(id="task_3", description="low priority task", priority=1),
        ]
        allocations = allocate_task_budgets(
            tasks,
            research_depth=3,
            max_research_tasks=3,
            max_total_searches=7,
        )
        self.assertEqual([value for _, value in allocations], [3, 2, 2])
        self.assertEqual(sum(value for _, value in allocations), 7)

    def test_budget_limits_number_of_started_tasks(self):
        tasks = [
            ResearchTask(id=f"task_{index}", description=f"research task {index}", priority=3)
            for index in range(1, 5)
        ]
        allocations = allocate_task_budgets(
            tasks,
            research_depth=2,
            max_research_tasks=4,
            max_total_searches=2,
        )
        self.assertEqual(len(allocations), 2)
        self.assertEqual(sum(value for _, value in allocations), 2)


if __name__ == "__main__":
    unittest.main()
