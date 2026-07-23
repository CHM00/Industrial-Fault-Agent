import io
import os
import unittest
import uuid
from pathlib import Path

from artifact_export import export_checklist, export_docx
from business_store import BusinessStore
from document_ingestion import DocumentIngestionError, chunk_text, extract_text
from evidence_mapping import extract_nodes, map_evidence
from safety import assess_fault_safety


class TemporaryBusinessStore(unittest.TestCase):
    def setUp(self):
        runtime = Path("output/runtime")
        runtime.mkdir(parents=True, exist_ok=True)
        self.path = runtime / f"p1-test-{uuid.uuid4().hex}.sqlite3"
        self.store = BusinessStore(self.path)

    def tearDown(self):
        for suffix in ("", "-wal", "-shm"):
            try:
                Path(str(self.path) + suffix).unlink()
            except FileNotFoundError:
                pass


class AssetAndKnowledgeTests(TemporaryBusinessStore):
    def test_asset_lifecycle_is_tenant_isolated(self):
        parent = self.store.create_asset("factory-a", {
            "asset_code": "LINE-1", "name": "一号产线", "asset_type": "line",
        })
        asset = self.store.create_asset("factory-a", {
            "asset_code": "P-101", "name": "一号离心泵", "asset_type": "pump",
            "vendor": "ACME", "model": "PX", "criticality": "high",
            "parent_id": parent["id"],
            "metadata": {"measurement_template": {
                "轴承温度": {"unit": "℃", "normal_min": 0, "normal_max": 80}
            }},
        })
        self.assertEqual(asset["status"], "active")
        self.assertEqual(asset["parent_id"], parent["id"])
        self.assertEqual(self.store.list_assets("factory-b"), [])

        self.assertEqual(self.store.record_measurements(
            "factory-a", asset["id"], "session-1", {"轴承温度": 85}, asset
        ), 1)
        detail = self.store.asset_detail("factory-a", asset["id"])
        self.assertEqual(detail["measurement_trends"][0]["unit"], "℃")
        self.assertEqual(detail["measurement_trends"][0]["maximum"], 85.0)

        updated = self.store.update_asset("factory-a", asset["id"], {
            "firmware": "2.1", "status": "inactive"
        })
        self.assertEqual(updated["firmware"], "2.1")
        self.assertEqual(
            [item["asset_code"] for item in self.store.list_assets("factory-a")],
            ["LINE-1"],
        )
        self.assertEqual(len(self.store.list_assets("factory-a", include_inactive=True)), 2)

    def test_document_versions_permissions_incremental_index_and_status(self):
        result = self.store.import_document(
            "factory-a", "expert-1", "泵过热 SOP", "pump.md", "md",
            "离心泵轴承温度升高时，先检查润滑油，再检查联轴器对中。",
            min_role="operator", change_summary="初版",
            applicability={"asset_type": "pump", "model": "PX*"},
        )
        doc_id = result["document_id"]
        self.assertEqual(result["version"], 1)
        payload = self.store.document_for_indexing("factory-a", doc_id)
        self.assertEqual(payload["version_id"], result["version_id"])
        self.assertEqual(len(payload["chunks"]), result["chunks"])
        self.store.update_version_artifacts(
            result["version_id"], source_path="output/knowledge/pump.md",
            index_status="ready", index_backend="milvus",
        )
        listed = self.store.list_documents("factory-a", "operator")[0]
        self.assertEqual(listed["index_status"], "ready")
        self.assertEqual(listed["index_backend"], "milvus")
        self.assertEqual(listed["applicability"]["asset_type"], "pump")
        self.assertTrue(self.store.search_knowledge("factory-a", "operator", "轴承温度润滑油"))
        self.assertEqual(self.store.search_knowledge("factory-a", "viewer", "轴承温度润滑油"), [])

        with self.assertRaises(ValueError):
            self.store.import_document(
                "factory-a", "expert-1", "泵过热 SOP", "pump.md", "md",
                "离心泵轴承温度升高时，先检查润滑油，再检查联轴器对中。",
                document_id=doc_id, min_role="operator",
            )

        second = self.store.import_document(
            "factory-a", "expert-1", "泵过热 SOP", "pump.md", "md",
            "离心泵轴承温度升高时，先检查润滑油油位和油质，再检查联轴器对中。",
            document_id=doc_id, min_role="operator", change_summary="补充油质检查",
        )
        self.assertEqual(second["version"], 2)
        versions = self.store.document_versions("factory-a", doc_id, "operator")
        self.assertEqual([item["status"] for item in versions], ["active", "superseded"])

        self.assertTrue(self.store.set_document_status("factory-a", doc_id, "inactive"))
        self.assertEqual(self.store.search_knowledge("factory-a", "operator", "润滑油油质"), [])

    def test_critical_asset_requires_expert_safety_review(self):
        assessment = assess_fault_safety(
            "设备出现一般通信异常", {"criticality": "critical"}
        )
        self.assertEqual(assessment["risk_level"], "high")
        self.assertTrue(assessment["requires_expert_approval"])
        self.assertIn("CRITICAL_ASSET", [item["id"] for item in assessment["matched_rules"]])


class CaseProcedureAndMetricsTests(TemporaryBusinessStore):
    def test_confirmed_case_result_is_retrievable(self):
        asset = self.store.create_asset("factory-a", {"asset_code": "VFD-1", "name": "变频器"})
        self.store.save_case("s-1", "factory-a", asset["id"], {
            "fault_input": "变频器启动时报过电流", "filtered_context": "检查负载与电机",
            "audit_result": "通过", "mermaid_diagram": "flowchart TD\nA[检查负载]",
        })
        confirmed = self.store.confirm_case("factory-a", "s-1", {
            "confirmed_root_cause": "电机电缆绝缘破损", "resolution": "更换电机电缆并测试绝缘",
            "outcome": "resolved", "credibility": 0.95, "expert_rating": 5,
        })
        self.assertEqual(confirmed["outcome"], "resolved")
        hits = self.store.search_cases("factory-a", "电缆绝缘破损", asset_id=asset["id"])
        self.assertEqual(hits[0]["session_id"], "s-1")
        self.assertEqual(hits[0]["source_type"], "historical_case")
        dashboard = self.store.dashboard("factory-a")
        self.assertEqual(dashboard["summary"]["confirmed_cases"], 1)
        self.assertEqual(dashboard["summary"]["case_success_rate"], 1.0)

    def test_procedure_diff_approval_and_publish_lifecycle(self):
        first = self.store.create_procedure_version("s-1", "factory-a", "operator-1", "flowchart TD\nA[检查电源]", "初版")
        second = self.store.create_procedure_version("s-1", "factory-a", "expert-1", "flowchart TD\nA[检查电源] --> B[记录电压]", "增加测量")
        diff = self.store.procedure_diff("factory-a", first["id"], second["id"])
        self.assertIn("记录电压", diff)
        hunks = self.store.procedure_diff_hunks("factory-a", first["id"], second["id"])
        self.assertTrue(hunks)
        adopted = self.store.adopt_procedure_hunks(
            "factory-a", first["id"], second["id"], [item["id"] for item in hunks]
        )
        self.assertEqual(adopted, second["mermaid"])
        rejected = self.store.adopt_procedure_hunks("factory-a", first["id"], second["id"], [])
        self.assertEqual(rejected, first["mermaid"])
        with self.assertRaises(ValueError):
            self.store.decide_procedure("factory-a", second["id"], "expert-1", "published")
        approved = self.store.decide_procedure("factory-a", second["id"], "expert-1", "approved", "同意")
        self.assertEqual(approved["status"], "approved")
        published = self.store.decide_procedure("factory-a", second["id"], "expert-1", "published", "发布")
        self.assertEqual(published["status"], "published")
        with self.assertRaises(ValueError):
            self.store.decide_procedure("factory-a", second["id"], "expert-1", "approved")

    def test_quality_cost_dashboard_aggregates_usage(self):
        previous_in = os.environ.get("LLM_INPUT_COST_PER_1M")
        previous_out = os.environ.get("LLM_OUTPUT_COST_PER_1M")
        try:
            os.environ["LLM_INPUT_COST_PER_1M"] = "2"
            os.environ["LLM_OUTPUT_COST_PER_1M"] = "4"
            self.store.record_metrics("s-1", "factory-a", "asset-1", {
                "usage_events": [{"input_tokens": 1000, "output_tokens": 500}],
                "external_result": {"search_count": 2, "sources": [{}, {}]},
                "revision_count": 1, "expert_feedbacks": ["补充"], "has_gaps": False,
                "mermaid_validation": {"valid": True}, "safety_assessment": {"risk_level": "low"},
            }, elapsed_ms=2500)
            dashboard = self.store.dashboard("factory-a")
            self.assertEqual(dashboard["summary"]["jobs"], 1)
            self.assertEqual(dashboard["summary"]["tokens"], 1500)
            self.assertAlmostEqual(dashboard["summary"]["cost"], 0.004)
            self.assertEqual(dashboard["summary"]["audit_pass_rate"], 1.0)
            self.assertEqual(dashboard["summary"]["expert_modification_rate"], 1.0)
            self.assertEqual(dashboard["summary"]["confirmed_cases"], 0)
        finally:
            if previous_in is None:
                os.environ.pop("LLM_INPUT_COST_PER_1M", None)
            else:
                os.environ["LLM_INPUT_COST_PER_1M"] = previous_in
            if previous_out is None:
                os.environ.pop("LLM_OUTPUT_COST_PER_1M", None)
            else:
                os.environ["LLM_OUTPUT_COST_PER_1M"] = previous_out


class IngestionEvidenceAndExportTests(unittest.TestCase):
    def test_text_json_csv_extraction_and_rejection(self):
        text, kind = extract_text("sop.txt", "过流保护动作".encode("utf-8"))
        self.assertEqual(kind, "txt")
        self.assertIn("过流保护", text)
        json_text, _ = extract_text("sop.json", b'{"alarm": "F4"}')
        self.assertIn('"alarm": "F4"', json_text)
        csv_text, _ = extract_text("sop.csv", "报警,处理\nF4,停机".encode("utf-8"))
        self.assertIn("F4 | 停机", csv_text)
        with self.assertRaises(DocumentIngestionError):
            extract_text("unsafe.exe", b"payload")

    def test_chunk_locations_and_evidence_cover_nodes_on_both_sides(self):
        chunks = chunk_text("[第 1 页]\n检查润滑油\n\n[第 2 页]\n检查联轴器", max_chars=18, overlap=0)
        self.assertEqual(chunks[0]["location"], "第 1 页")
        self.assertEqual(chunks[1]["location"], "第 2 页")
        diagram = "flowchart TD\nA[检查润滑油] --> B{油位正常?}\nB --> C((检查联轴器))"
        nodes = extract_nodes(diagram)
        self.assertEqual([item["node_id"] for item in nodes], ["A", "B", "C"])
        mapped = map_evidence(diagram, [{
            "evidence_id": "sop:1", "source_type": "managed_sop", "title": "润滑 SOP",
            "content": "检查润滑油油位是否正常，并检查联轴器", "location": "第 2 页",
            "trust_level": "authoritative",
        }])
        self.assertEqual(len(mapped), 3)
        self.assertEqual(mapped[0]["evidence"][0]["location"], "第 2 页")
        conflict = map_evidence("flowchart TD\nA[带电检查母线]", [{
            "evidence_id": "sop:safety", "title": "安全规程",
            "content": "严禁带电检查母线，必须先隔离能源", "trust_level": "authoritative",
        }])[0]
        self.assertTrue(conflict["needs_review"])
        self.assertEqual(conflict["evidence"][0]["relation"], "contradicts")
        self.assertTrue(conflict["conflicts"])

    def test_word_and_field_checklist_exports(self):
        state = {
            "request_id": "req-1", "fault_input": "离心泵轴承温升",
            "structured_context": "设备：P-101", "audit_result": "审计通过",
            "safety_assessment": {"risk_level": "medium", "controls": ["停机挂牌"]},
            "mermaid_diagram": "flowchart TD\nA[检查润滑油]",
            "evidence_mappings": [{
                "node_id": "A", "label": "检查润滑油",
                "evidence": [{"evidence_id": "sop:1", "title": "泵 SOP", "confidence": 0.9, "location": "第 3 页"}],
            }],
        }
        docx, docx_name, docx_type = export_docx(state)
        self.assertTrue(docx.startswith(b"PK"))
        self.assertTrue(docx_name.endswith(".docx"))
        self.assertIn("wordprocessingml", docx_type)
        checklist, filename, media_type = export_checklist(state)
        self.assertTrue(checklist.startswith(b"\xef\xbb\xbf"))
        self.assertIn("检查润滑油", checklist.decode("utf-8-sig"))
        self.assertTrue(filename.endswith("_检查单.csv"))
        self.assertIn("text/csv", media_type)


if __name__ == "__main__":
    unittest.main()
