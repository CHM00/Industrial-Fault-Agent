import unittest
from unittest.mock import patch

import httpx

import fault_agent
from langgraph.checkpoint.memory import MemorySaver


class _FailingLLM:
    def __init__(self):
        self.calls = []

    def invoke(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        raise httpx.ReadTimeout("timed out")


class _SuccessfulLLM:
    def __init__(self):
        self.timeout = None

    def invoke(self, _messages, **kwargs):
        self.timeout = kwargs["timeout"]
        return "ok"


class LLMRuntimeLimitTests(unittest.TestCase):
    def test_sdk_retry_is_disabled_and_http_phases_are_bounded(self):
        self.assertEqual(fault_agent.llm.max_retries, 0)
        timeout = fault_agent.llm.root_client._client.timeout
        self.assertEqual(timeout.connect, fault_agent.LLM_CONNECT_TIMEOUT_SECONDS)
        self.assertEqual(timeout.read, fault_agent.LLM_READ_TIMEOUT_SECONDS)
        self.assertEqual(timeout.write, fault_agent.LLM_WRITE_TIMEOUT_SECONDS)
        self.assertEqual(timeout.pool, fault_agent.LLM_POOL_TIMEOUT_SECONDS)

    @patch.object(fault_agent.time, "sleep", return_value=None)
    def test_retry_count_is_single_layer_and_attempts_receive_timeout(self, _sleep):
        model = _FailingLLM()

        with self.assertRaises(fault_agent.LLMInvocationTimeout):
            fault_agent.invoke_llm_with_retry(
                model,
                ["message"],
                max_attempts=2,
                total_timeout_seconds=10,
                delay=0,
            )

        self.assertEqual(len(model.calls), 2)
        for _, kwargs in model.calls:
            self.assertIsInstance(kwargs["timeout"], httpx.Timeout)

    @patch.object(fault_agent.time, "monotonic", return_value=100.0)
    def test_llm_call_uses_remaining_node_budget(self, _monotonic):
        model = _SuccessfulLLM()
        token = fault_agent._NODE_DEADLINE.set(105.0)
        try:
            result = fault_agent.invoke_llm_with_retry(model, ["message"])
        finally:
            fault_agent._NODE_DEADLINE.reset(token)

        self.assertEqual(result, "ok")
        self.assertEqual(model.timeout.read, 5.0)

    def test_compiled_graph_has_per_step_deadline(self):
        graph = fault_agent.build_app(checkpointer=MemorySaver())
        self.assertEqual(
            graph.step_timeout,
            fault_agent.NODE_EXECUTION_TIMEOUT_SECONDS,
        )


if __name__ == "__main__":
    unittest.main()
