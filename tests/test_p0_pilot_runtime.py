import unittest
import os
import uuid
import json
from pathlib import Path
from typing import TypedDict

from governance import DiagnosisConcurrency, SlidingWindowRateLimiter
from runtime_store import JobRecord, JobStore
from safety import assess_fault_safety
from security import ApiKeyAuth

try:
    from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: F401
    HAS_SQLITE_CHECKPOINTER = True
except ImportError:
    HAS_SQLITE_CHECKPOINTER = False


class SafetyGateTests(unittest.TestCase):
    def test_high_voltage_requires_approval(self):
        result = assess_fault_safety("10kV 高压柜疑似放电，需要带电检查母线")
        self.assertEqual(result["risk_level"], "critical")
        self.assertTrue(result["requires_expert_approval"])
        self.assertIn("ENERGY_ISOLATION", {item["id"] for item in result["matched_rules"]})

    def test_protection_bypass_is_prohibited(self):
        result = assess_fault_safety("旁路安全联锁后强制启动")
        self.assertTrue(result["prohibited"])
        self.assertTrue(result["requires_expert_approval"])

    def test_normal_network_fault_is_low_risk(self):
        result = assess_fault_safety("PLC 与上位机通讯超时")
        self.assertEqual(result["risk_level"], "low")
        self.assertFalse(result["requires_expert_approval"])

    def test_high_risk_graph_pauses_before_research_and_can_be_denied(self):
        import fault_agent
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.types import Command

        graph = fault_agent.build_app(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": f"safety-{uuid.uuid4().hex}"}}
        initial = fault_agent.build_initial_state(
            "10kV 高压柜出现放电声，需要带电检查母线",
            research_options={"research_mode": "off"},
        )
        graph.invoke(initial, config=config)
        paused = graph.get_state(config)
        self.assertEqual(paused.next, ("safety_gate",))
        self.assertEqual(paused.tasks[0].interrupts[0].value["kind"], "safety_approval")
        self.assertFalse(paused.values.get("external_result"))

        result = graph.invoke(
            Command(resume={"approved": False, "feedback": "现场条件不满足"}),
            config=config,
        )
        self.assertTrue(result["safety_denied"])
        self.assertFalse(result.get("external_result"))


class JobStoreTests(unittest.TestCase):
    def test_job_survives_store_recreation_and_records_events(self):
        runtime_dir = Path("output/runtime")
        runtime_dir.mkdir(parents=True, exist_ok=True)
        path = runtime_dir / f"test-pilot-{uuid.uuid4().hex}.sqlite3"
        try:
            first = JobStore(path, ttl_seconds=3600)
            first.create(JobRecord(
                session_id="session-1",
                request_id="request-1",
                thread_id="thread-1",
                owner_id="operator-1",
                tenant_id="factory-a",
                status="waiting_feedback",
                fault_input="PLC fault",
                auto_mode=False,
            ))
            first.update("session-1", actor_id="expert-1", status="completed")

            second = JobStore(path, ttl_seconds=3600)
            loaded = second.get("session-1")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.status, "completed")
            self.assertEqual(len(second.events("session-1")), 2)
        finally:
            for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
                candidate.unlink(missing_ok=True)

    @unittest.skipUnless(HAS_SQLITE_CHECKPOINTER, "official SQLite checkpointer is not installed")
    def test_langgraph_checkpoint_survives_graph_recreation(self):
        from checkpointing import create_checkpointer
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import Command, interrupt

        class State(TypedDict):
            answer: str

        def pause(state: State):
            return {"answer": interrupt("approval")}

        runtime_dir = Path("output/runtime")
        runtime_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = runtime_dir / f"test-checkpoint-{uuid.uuid4().hex}.sqlite3"
        path = str(checkpoint_path)
        try:
            original = os.environ.get("CHECKPOINT_SQLITE_PATH")
            os.environ["CHECKPOINT_SQLITE_PATH"] = path
            try:
                builder = StateGraph(State)
                builder.add_node("pause", pause)
                builder.add_edge(START, "pause")
                builder.add_edge("pause", END)
                config = {"configurable": {"thread_id": "persistent-thread"}}

                saver1 = create_checkpointer("sqlite")
                graph1 = builder.compile(checkpointer=saver1)
                graph1.invoke({"answer": ""}, config=config)
                saver1._pilot_connection.close()

                saver2 = create_checkpointer("sqlite")
                graph2 = builder.compile(checkpointer=saver2)
                result = graph2.invoke(Command(resume="approved"), config=config)
                self.assertEqual(result["answer"], "approved")
                saver2._pilot_connection.close()
            finally:
                if original is None:
                    os.environ.pop("CHECKPOINT_SQLITE_PATH", None)
                else:
                    os.environ["CHECKPOINT_SQLITE_PATH"] = original
        finally:
            for candidate in (
                checkpoint_path,
                Path(str(checkpoint_path) + "-wal"),
                Path(str(checkpoint_path) + "-shm"),
            ):
                candidate.unlink(missing_ok=True)


class SecurityAndGovernanceTests(unittest.TestCase):
    def test_api_key_role_and_tenant(self):
        auth = ApiKeyAuth(
            enabled=True,
            raw_config='{"secret":{"subject":"alice","role":"expert","tenant_id":"factory-a"}}',
        )
        principal = auth.authenticate(None, "secret")
        self.assertEqual(principal.subject, "alice")
        self.assertTrue(principal.can("operator"))
        self.assertFalse(principal.can("admin"))
        self.assertIsNone(auth.authenticate(None, "wrong"))

    def test_configured_user_login_and_signed_session(self):
        password_hash = ApiKeyAuth.hash_password("correct-password")
        auth = ApiKeyAuth(
            enabled=True,
            raw_config="",
            raw_users=json.dumps({
                "alice": {
                    "display_name": "Alice Expert",
                    "password_hash": password_hash,
                    "subject": "expert-1",
                    "role": "expert",
                    "tenant_id": "factory-a",
                }
            }),
            session_secret="test-session-secret-with-at-least-32-characters",
        )
        principal = auth.login("alice", "correct-password")
        self.assertIsNotNone(principal)
        self.assertEqual(principal.display_name, "Alice Expert")
        self.assertIsNone(auth.login("alice", "wrong-password"))
        token = auth.issue_session(principal)
        restored = auth.authenticate(None, None, token)
        self.assertEqual(restored.subject, "expert-1")
        self.assertEqual(restored.role, "expert")
        self.assertIsNone(auth.authenticate(None, None, token + "tampered"))

    def test_rate_limit_is_enforced(self):
        limiter = SlidingWindowRateLimiter(requests_per_minute=2)
        self.assertTrue(limiter.allow("alice", now=10)[0])
        self.assertTrue(limiter.allow("alice", now=11)[0])
        allowed, retry = limiter.allow("alice", now=12)
        self.assertFalse(allowed)
        self.assertGreater(retry, 0)

    def test_concurrency_is_bounded(self):
        slots = DiagnosisConcurrency(maximum=1)
        self.assertTrue(slots.acquire(timeout=0))
        self.assertFalse(slots.acquire(timeout=0))
        slots.release()
        self.assertEqual(slots.active, 0)

    def test_http_authentication_and_admin_metrics_permission(self):
        import web_server
        from fastapi.testclient import TestClient

        original = web_server._auth
        web_server._auth = ApiKeyAuth(
            enabled=True,
            raw_config=(
                '{"operator-key":{"subject":"operator","role":"operator","tenant_id":"factory-a"},'
                '"admin-key":{"subject":"admin","role":"admin","tenant_id":"factory-a"}}'
            ),
        )
        try:
            client = TestClient(web_server.app)
            self.assertEqual(client.get("/health/live").status_code, 200)
            self.assertEqual(client.get("/api/jobs").status_code, 401)
            self.assertEqual(
                client.get("/api/jobs", headers={"X-API-Key": "operator-key"}).status_code,
                200,
            )
            self.assertEqual(
                client.get("/metrics", headers={"X-API-Key": "operator-key"}).status_code,
                403,
            )
            self.assertEqual(
                client.get("/metrics", headers={"X-API-Key": "admin-key"}).status_code,
                200,
            )
        finally:
            web_server._auth = original

    def test_http_user_login_me_and_logout(self):
        import web_server
        from fastapi.testclient import TestClient

        original = web_server._auth
        web_server._auth = ApiKeyAuth(
            enabled=True,
            raw_config="",
            raw_users=json.dumps({
                "operator": {
                    "password": "operator-password",
                    "subject": "operator-1",
                    "role": "operator",
                    "tenant_id": "factory-a",
                }
            }),
            session_secret="test-session-secret-with-at-least-32-characters",
        )
        try:
            client = TestClient(web_server.app)
            self.assertEqual(client.get("/api/auth/me").status_code, 401)
            login = client.post(
                "/api/auth/login",
                json={"username": "operator", "password": "operator-password"},
            )
            self.assertEqual(login.status_code, 200)
            self.assertTrue(login.cookies.get("fault_agent_session"))
            me = client.get("/api/auth/me")
            self.assertEqual(me.status_code, 200)
            self.assertEqual(me.json()["user"]["role"], "operator")
            self.assertEqual(client.get("/metrics").status_code, 403)
            self.assertEqual(client.post("/api/auth/logout").status_code, 200)
            self.assertEqual(client.get("/api/auth/me").status_code, 401)
        finally:
            web_server._auth = original


if __name__ == "__main__":
    unittest.main()
