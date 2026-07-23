import asyncio
import unittest
from types import SimpleNamespace

import fault_agent
import web_server
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from web_server import StartReq


class IntegrationContractTests(unittest.TestCase):
    def test_fault_graph_uses_research_gateway(self):
        nodes = fault_agent.build_app().get_graph().nodes
        self.assertIn("external_research", nodes)
        self.assertNotIn("search_agent", nodes)
        self.assertNotIn("tools_node", nodes)

    def test_start_request_defaults_remain_backward_compatible(self):
        request = StartReq(fault_input="PLC communication fault")
        self.assertEqual(request.research_mode, "basic")
        self.assertEqual(request.research_depth, 2)
        self.assertIsNone(request.research_timeout_seconds)

    def test_web_request_accepts_unlimited_research_timeout(self):
        request = StartReq(
            fault_input="PLC communication fault",
            research_timeout_seconds=0,
        )
        self.assertEqual(request.research_timeout_seconds, 0)

    def test_web_rejects_unreleased_auto_research_mode(self):
        with self.assertRaises(ValueError):
            StartReq(fault_input="fault", research_mode="auto")

    def test_openapi_only_advertises_released_research_modes(self):
        schema = web_server.app.openapi()
        modes = schema["components"]["schemas"]["StartReq"]["properties"]["research_mode"]["enum"]
        self.assertEqual(modes, ["off", "basic", "supervisory"])

    def test_p1_structured_diagnosis_fields_and_routes_are_published(self):
        request = StartReq(
            fault_input="变频器 F4 过电流", asset_id="asset-1", alarm_code="F4",
            operating_context="冷启动", measurements={"电流": 83},
            maintenance_history="昨日更换电缆",
        )
        self.assertEqual(request.measurements["电流"], 83)
        schema = web_server.app.openapi()
        expected = {
            "/api/assets", "/api/knowledge/documents",
            "/api/assets/import", "/api/assets/{asset_id}/detail",
            "/api/procedures/{session_id}/versions",
            "/api/procedures/{session_id}/adopt",
            "/api/jobs/{session_id}/export", "/api/jobs/archive-batch",
            "/api/dashboard/quality",
        }
        self.assertTrue(expected.issubset(schema["paths"]))

    def test_p1_write_routes_require_expected_roles(self):
        self.assertEqual(web_server._required_role("POST", "/api/assets"), "operator")
        self.assertEqual(web_server._required_role("POST", "/api/assets/import"), "operator")
        self.assertEqual(web_server._required_role("POST", "/api/jobs/archive-batch"), "operator")
        self.assertEqual(web_server._required_role("POST", "/api/knowledge/documents"), "expert")
        self.assertEqual(web_server._required_role("POST", "/api/procedures/s1/versions/v1/approve"), "expert")
        self.assertEqual(web_server._required_role("GET", "/api/dashboard/quality"), "expert")

    def test_sop_applicability_is_a_hard_asset_filter(self):
        scope = {
            "asset_type": "pump", "vendor": "ACME",
            "model": "PX-*", "firmware": "2.*",
        }
        self.assertTrue(web_server._sop_applies_to_asset(scope, {
            "asset_type": "pump", "vendor": "acme",
            "model": "PX-100", "firmware": "2.3",
        }))
        self.assertFalse(web_server._sop_applies_to_asset(scope, {
            "asset_type": "pump", "vendor": "OTHER",
            "model": "PX-100", "firmware": "2.3",
        }))
        self.assertFalse(web_server._sop_applies_to_asset(scope, {}))

    def test_measurement_units_and_normal_range_are_rendered(self):
        asset = {"metadata": {"measurement_template": {
            "轴承温度": {"unit": "℃", "normal_min": 0, "normal_max": 80}
        }}}
        rendered = web_server._format_measurements({"轴承温度": 85}, asset)
        self.assertIn("85 ℃", rendered)
        self.assertIn("高于正常上限", rendered)


class FakeWebGraph:
    def stream(self, payload, config, stream_mode):
        yield {
            "external_research": {
                "messages": [],
                "external_result": {
                    "schema_version": "1.0",
                    "effective_mode": "off",
                    "status": "skipped",
                    "query_count": 0,
                    "search_count": 0,
                    "sources": [],
                    "warnings": [],
                    "elapsed_ms": 0,
                },
            }
        }

    def get_state(self, config):
        return SimpleNamespace(
            next=(),
            tasks=(),
            values={
                "external_result": {
                    "schema_version": "1.0",
                    "effective_mode": "off",
                    "status": "skipped",
                    "sources": [],
                    "warnings": [],
                }
            },
        )


class FakePausedWebGraph:
    def stream(self, payload, config, stream_mode):
        yield {"audit": {"messages": []}}

    def get_state(self, config):
        return SimpleNamespace(
            next=("ask_expert",),
            tasks=(
                SimpleNamespace(
                    interrupts=(SimpleNamespace(value="expert question"),),
                ),
            ),
            values={
                "current_question_idx": 0,
                "audit_questions": ["expert question"],
                "mermaid_diagram": "flowchart TD\nstart --> check1",
                "revision_count": 0,
            },
        )


class WebStreamingContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_expert_question_includes_initial_diagram_and_revision(self):
        original_graph = web_server._agent_app
        original_langfuse = web_server._langfuse_handler
        web_server._agent_app = FakePausedWebGraph()
        web_server._langfuse_handler = lambda: None
        try:
            response = await web_server.api_start(
                StartReq(fault_input="test fault", research_mode="off")
            )
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
            body = "".join(chunks)
            self.assertIn("event: expert_question", body)
            self.assertIn('"diagram": "flowchart TD\\nstart --> check1"', body)
            self.assertIn('"revision_count": 0', body)
        finally:
            web_server._agent_app = original_graph
            web_server._langfuse_handler = original_langfuse
            with web_server._sessions_lock:
                web_server._sessions.clear()

    async def test_external_research_supports_sync_and_async_graph_execution(self):
        builder = StateGraph(fault_agent.AgentState)
        builder.add_node("external_research", fault_agent.external_research_runnable)
        builder.add_edge(START, "external_research")
        builder.add_edge("external_research", END)
        graph = builder.compile()

        sync_state = fault_agent.build_initial_state(
            "test fault",
            research_options={"research_mode": "off"},
        )
        sync_events = await asyncio.to_thread(
            lambda: list(graph.stream(sync_state, stream_mode="updates"))
        )
        self.assertEqual(
            sync_events[-1]["external_research"]["external_result"]["status"],
            "skipped",
        )

        async_state = fault_agent.build_initial_state(
            "test fault",
            research_options={"research_mode": "off"},
        )
        async_events = [
            event
            async for event in graph.astream(async_state, stream_mode="updates")
        ]
        self.assertEqual(
            async_events[-1]["external_research"]["external_result"]["status"],
            "skipped",
        )

    async def test_python310_sync_bridge_supports_async_research_interrupt_and_resume(self):
        def ask_expert(state: fault_agent.AgentState):
            return {"expert_feedbacks": [interrupt("expert question")]}

        builder = StateGraph(fault_agent.AgentState)
        builder.add_node("external_research", fault_agent.external_research_runnable)
        builder.add_node("ask_expert", ask_expert)
        builder.add_edge(START, "external_research")
        builder.add_edge("external_research", "ask_expert")
        builder.add_edge("ask_expert", END)
        graph = builder.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "python310-interrupt-test"}}
        initial_state = fault_agent.build_initial_state(
            "test fault",
            research_options={"research_mode": "off"},
        )

        original_graph = web_server._agent_app
        web_server._agent_app = graph
        try:
            first_events = [
                event
                async for event in web_server._stream_updates(initial_state, config)
            ]
            self.assertEqual(
                first_events[0]["external_research"]["external_result"]["status"],
                "skipped",
            )
            self.assertIn("__interrupt__", first_events[-1])

            paused = await web_server._get_state(config)
            self.assertTrue(paused.next)
            self.assertEqual(
                paused.tasks[0].interrupts[0].value,
                "expert question",
            )

            resumed_events = [
                event
                async for event in web_server._stream_updates(
                    Command(resume="expert feedback"),
                    config,
                )
            ]
            self.assertEqual(
                resumed_events[-1]["ask_expert"]["expert_feedbacks"],
                ["expert feedback"],
            )
            completed = await web_server._get_state(config)
            self.assertFalse(completed.next)
        finally:
            web_server._agent_app = original_graph

    async def test_start_stream_emits_research_and_done_events(self):
        original_graph = web_server._agent_app
        original_save = web_server._save_outputs
        original_langfuse = web_server._langfuse_handler
        web_server._agent_app = FakeWebGraph()
        web_server._save_outputs = lambda fault, values: {
            "diagram": None,
            "report": None,
            "memory": None,
        }
        web_server._langfuse_handler = lambda: None
        try:
            response = await web_server.api_start(
                StartReq(fault_input="test fault", research_mode="off")
            )
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
            body = "".join(chunks)
            self.assertIn("event: research_started", body)
            self.assertIn("event: research_completed", body)
            self.assertIn("event: done", body)
            self.assertIn('"request_id"', body)
        finally:
            web_server._agent_app = original_graph
            web_server._save_outputs = original_save
            web_server._langfuse_handler = original_langfuse


if __name__ == "__main__":
    unittest.main()
