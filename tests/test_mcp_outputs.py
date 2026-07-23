import unittest
from unittest.mock import mock_open, patch

import mcp_tools


class OutputPersistenceTests(unittest.TestCase):
    def test_compact_memory_keeps_provenance_and_truncates_content(self):
        compact = mcp_tools._compact_external_result(
            {
                "schema_version": "1.0",
                "sources": [
                    {
                        "source_id": "source_1",
                        "title": "Vendor manual",
                        "url": "https://example.com/manual",
                        "snippet": "summary",
                        "content": "x" * 2000,
                    }
                ],
                "task_results": [],
            }
        )
        self.assertEqual(compact["schema_version"], "1.0")
        self.assertEqual(compact["sources"][0]["source_id"], "source_1")
        self.assertEqual(len(compact["sources"][0]["content"]), 1000)

    def test_report_contains_structured_source_link(self):
        state = {
            "fault_input": "motor fault",
            "mermaid_diagram": "flowchart TD\nA-->B",
            "external_knowledge": "external summary",
            "external_result": {
                "effective_mode": "basic",
                "status": "success",
                "query_count": 1,
                "search_count": 1,
                "warnings": [],
                "sources": [
                    {
                        "source_id": "source_1",
                        "title": "Vendor manual",
                        "url": "https://example.com/manual",
                    }
                ],
            },
        }
        handle = mock_open()
        with (
            patch.object(mcp_tools, "_ensure_dirs"),
            patch("builtins.open", handle),
            patch("builtins.print"),
        ):
            mcp_tools.save_diagram_and_report("motor fault", state)
        written = "".join(
            str(call.args[0]) for call in handle().write.call_args_list if call.args
        )
        self.assertIn("source_1", written)
        self.assertIn("https://example.com/manual", written)


if __name__ == "__main__":
    unittest.main()
