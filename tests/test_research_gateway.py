import asyncio
import unittest
from typing import TypedDict

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from research.configuration import ResearchSettings
from research.contracts import ExternalSource, ResearchOptions
from research.gateway import run_external_research
from research.providers import SearchProvider, SearchProviderError


class FakeProvider(SearchProvider):
    name = "tavily"

    def __init__(self, settings, failures=0):
        super().__init__(settings)
        self.calls = 0
        self.failures = failures

    async def search(self, query, *, max_results, timeout_seconds, max_source_chars):
        self.calls += 1
        if self.calls <= self.failures:
            raise SearchProviderError("SEARCH_TIMEOUT", "temporary timeout", retryable=True)
        return [
            ExternalSource(
                title=f"Source {self.calls}",
                url=f"https://example.com/source/{self.calls}",
                snippet=f"Evidence for {query}",
                content=f"Evidence for {query}",
                provider="tavily",
                query=query,
            )
        ]


class SlowFakeProvider(FakeProvider):
    async def search(self, query, *, max_results, timeout_seconds, max_source_chars):
        await asyncio.sleep(0.05)
        return await super().search(
            query,
            max_results=max_results,
            timeout_seconds=timeout_seconds,
            max_source_chars=max_source_chars,
        )


class FakeLLM:
    async def ainvoke(self, messages):
        system = messages[0].content
        if "研究主管" in system:
            return AIMessage(
                content=(
                    '{"tasks":['
                    '{"id":"task_1","description":"研究电气原因","priority":5},'
                    '{"id":"task_2","description":"研究机械原因","priority":4}'
                    "]}"
                )
            )
        if "只生成简洁" in system:
            return AIMessage(content='{"query":"industrial fault diagnosis"}')
        if "知识缺口" in system:
            return AIMessage(
                content='{"knowledge_gap":"details","follow_up_query":"industrial fault safety checks"}'
            )
        if "资料整理" in system:
            return AIMessage(content="结构化研究摘要")
        if "证据分析" in system:
            return AIMessage(content="两个任务的证据相互补充")
        if "综合专家" in system:
            return AIMessage(content="综合外部研究结论")
        return AIMessage(content="")


class InventedCitationLLM(FakeLLM):
    async def ainvoke(self, messages):
        if "综合专家" in messages[0].content:
            return AIMessage(content="综合结论 https://invented.example/not-a-source")
        return await super().ainvoke(messages)


class ResearchGatewayTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.settings = ResearchSettings(total_timeout_seconds=30)

    async def test_off_mode_never_needs_provider(self):
        result = await run_external_research("fault", ResearchOptions(research_mode="off"))
        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.search_count, 0)
        self.assertEqual(result.sources, [])

    async def test_zero_request_timeout_disables_environment_deadline(self):
        settings = self.settings.model_copy(update={"total_timeout_seconds": 0.01})
        provider = SlowFakeProvider(settings)
        result = await run_external_research(
            "motor overcurrent",
            ResearchOptions(
                research_mode="basic",
                research_depth=1,
                max_total_searches=1,
                research_timeout_seconds=0,
            ),
            settings=settings,
            provider=provider,
            llm=None,
        )
        self.assertEqual(result.status, "success")
        self.assertEqual(result.search_count, 1)

    async def test_basic_depth_is_exact_without_failures(self):
        provider = FakeProvider(self.settings)
        result = await run_external_research(
            "motor overcurrent",
            ResearchOptions(
                research_mode="basic",
                research_depth=2,
                max_total_searches=2,
                search_max_retries=1,
            ),
            settings=self.settings,
            provider=provider,
            llm=None,
        )
        self.assertEqual(result.query_count, 2)
        self.assertEqual(result.search_count, 2)
        self.assertEqual(provider.calls, 2)
        self.assertEqual(result.status, "success")

    async def test_basic_sources_survive_memory_checkpointer_serialization(self):
        class OuterState(TypedDict):
            result: dict

        provider = FakeProvider(self.settings)
        options = ResearchOptions(
            research_mode="basic",
            research_depth=1,
            max_total_searches=1,
        )

        async def research_node(state: OuterState) -> dict:
            result = await run_external_research(
                "motor bearing temperature rise",
                options,
                settings=self.settings,
                provider=provider,
                llm=None,
            )
            return {"result": result.model_dump(mode="json")}

        def research_node_sync(state: OuterState) -> dict:
            return asyncio.run(research_node(state))

        builder = StateGraph(OuterState)
        builder.add_node(
            "research",
            RunnableLambda(research_node_sync, afunc=research_node),
        )
        builder.add_edge(START, "research")
        builder.add_edge("research", END)
        graph = builder.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "research-msgpack-regression"}}

        events = await asyncio.to_thread(
            lambda: list(
                graph.stream(
                    {"result": {}},
                    config,
                    stream_mode="updates",
                )
            )
        )
        result = events[-1]["research"]["result"]
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["search_count"], 1)
        self.assertEqual(len(result["sources"]), 1)
        self.assertIsInstance(result["sources"][0]["url"], str)

        snapshot = await asyncio.to_thread(graph.get_state, config)
        self.assertEqual(snapshot.values["result"]["status"], "success")

    async def test_retry_consumes_budget(self):
        provider = FakeProvider(self.settings, failures=1)
        result = await run_external_research(
            "motor overcurrent",
            ResearchOptions(
                research_mode="basic",
                research_depth=2,
                max_total_searches=2,
                search_max_retries=1,
            ),
            settings=self.settings,
            provider=provider,
            llm=None,
        )
        self.assertEqual(result.query_count, 1)
        self.assertEqual(result.search_count, 2)
        self.assertLessEqual(result.search_count, 2)

    async def test_supervisory_reuses_basic_graph_and_respects_budget(self):
        provider = FakeProvider(self.settings)
        result = await run_external_research(
            "变频器过流，同时出现电机振动和电压波动",
            ResearchOptions(
                research_mode="supervisory",
                research_depth=2,
                max_research_tasks=3,
                max_total_searches=3,
            ),
            settings=self.settings,
            provider=provider,
            llm=FakeLLM(),
        )
        self.assertEqual(result.effective_mode, "supervisory")
        self.assertEqual(len(result.task_results), 2)
        self.assertLessEqual(result.search_count, 3)
        self.assertTrue(result.sources)
        self.assertEqual(result.summary, "综合外部研究结论")

    async def test_supervisory_removes_invented_citations(self):
        provider = FakeProvider(self.settings)
        result = await run_external_research(
            "变频器过流，同时出现电机振动和电压波动",
            ResearchOptions(
                research_mode="supervisory",
                research_depth=1,
                max_research_tasks=2,
                max_total_searches=2,
            ),
            settings=self.settings,
            provider=provider,
            llm=InventedCitationLLM(),
        )
        self.assertNotIn("https://invented.example", result.summary)
        self.assertIn(
            "UNVERIFIED_CITATION_REMOVED",
            {item.code for item in result.warnings},
        )


if __name__ == "__main__":
    unittest.main()
