import unittest
from types import SimpleNamespace
from unittest.mock import patch

import fault_agent
from mermaid_pipeline import (
    MermaidProcessingError,
    MermaidValidationResult,
    extract_mermaid,
    normalize_mermaid,
    validate_mermaid,
)


class MermaidPipelineTests(unittest.TestCase):
    def test_extracts_fence_and_normalizes_line_breaks(self):
        diagram = extract_mermaid(
            "说明\n```mermaid\r\nflowchart TD\r\n    A[\"第一行<br>第二行\"]\r\n```"
        )
        self.assertEqual(
            diagram,
            'flowchart TD\n    A["第一行<br/>第二行"]',
        )

    def test_rejects_unsafe_mermaid_directives(self):
        with self.assertRaises(MermaidProcessingError):
            normalize_mermaid(
                'flowchart TD\n    A["开始"]\n    click A javascript:alert(1)'
            )

    def test_official_parser_accepts_valid_chinese_flowchart(self):
        result = validate_mermaid(
            'flowchart TD\n'
            '    A["开始"] --> B{"是否正常？"}\n'
            '    B -->|是| C["结束"]'
        )
        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.code, "VALID")

    def test_official_parser_rejects_unquoted_parenthesis_in_node(self):
        result = validate_mermaid(
            "flowchart TD\n"
            "    C --> D{润滑状态是否正常?<br>(油脂量适中、无变质)}"
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.code, "SYNTAX_ERROR")


class MermaidRepairTests(unittest.TestCase):
    @patch.object(fault_agent, "invoke_llm_with_retry")
    @patch.object(fault_agent, "validate_mermaid")
    def test_repairs_once_after_syntax_error(self, mock_validate, mock_invoke):
        mock_validate.side_effect = [
            MermaidValidationResult(False, "SYNTAX_ERROR", "unexpected token"),
            MermaidValidationResult(True, "VALID", diagram_type="flowchart"),
        ]
        mock_invoke.return_value = SimpleNamespace(
            content='```mermaid\nflowchart TD\n    A["开始"] --> B["结束"]\n```'
        )

        diagram, validation = fault_agent._validated_mermaid_from_response(
            "flowchart TD\n    A[开始 --> B",
            stage="测试阶段",
        )

        self.assertIn('A["开始"]', diagram)
        self.assertTrue(validation["valid"])
        self.assertTrue(validation["repaired"])
        self.assertEqual(mock_validate.call_count, 2)
        mock_invoke.assert_called_once()

    @patch.object(fault_agent, "invoke_llm_with_retry")
    @patch.object(fault_agent, "validate_mermaid")
    def test_does_not_repair_validator_infrastructure_error(
        self,
        mock_validate,
        mock_invoke,
    ):
        mock_validate.return_value = MermaidValidationResult(
            False,
            "VALIDATOR_UNAVAILABLE",
            "node not found",
        )

        with self.assertRaisesRegex(RuntimeError, "VALIDATOR_UNAVAILABLE"):
            fault_agent._validated_mermaid_from_response(
                'flowchart TD\n    A["开始"]',
                stage="测试阶段",
            )

        mock_invoke.assert_not_called()

    @patch.object(fault_agent, "invoke_llm_with_retry")
    @patch.object(fault_agent, "validate_mermaid")
    def test_fails_after_exactly_one_unsuccessful_repair(
        self,
        mock_validate,
        mock_invoke,
    ):
        mock_validate.side_effect = [
            MermaidValidationResult(False, "SYNTAX_ERROR", "first error"),
            MermaidValidationResult(False, "SYNTAX_ERROR", "second error"),
        ]
        mock_invoke.return_value = SimpleNamespace(
            content="flowchart TD\n    A[仍然错误 --> B"
        )

        with self.assertRaisesRegex(RuntimeError, "自动修复一次后仍未通过"):
            fault_agent._validated_mermaid_from_response(
                "flowchart TD\n    A[开始 --> B",
                stage="测试阶段",
            )

        self.assertEqual(mock_validate.call_count, 2)
        mock_invoke.assert_called_once()


if __name__ == "__main__":
    unittest.main()
