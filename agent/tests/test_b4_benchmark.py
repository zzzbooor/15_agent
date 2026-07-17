from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from b4_benchmark import expected_calls_match  # noqa: E402
from b4_acceptance import run_acceptance  # noqa: E402


class BenchmarkTests(unittest.TestCase):
    def test_expected_call_matching_is_order_independent(self) -> None:
        actual = [
            {"name": "calculator", "args": {"expression": "2+2"}},
            {"name": "file_reader", "args": {"path": "docs/a.txt", "max_chars": 100}},
        ]
        expected = [
            {"name": "file_reader", "args_contains": {"path": "docs/a.txt"}},
            {"name": "calculator"},
        ]
        valid, errors = expected_calls_match(actual, expected)
        self.assertTrue(valid)
        self.assertEqual(errors, [])

    def test_expected_call_matching_reports_missing_arguments(self) -> None:
        valid, errors = expected_calls_match(
            [{"name": "file_reader", "args": {"path": "wrong"}}],
            [{"name": "file_reader", "args_contains": {"path": "docs/a.txt"}}],
        )
        self.assertFalse(valid)
        self.assertTrue(errors)

    def test_expression_whitespace_and_query_substring_are_semantic(self) -> None:
        actual = [
            {"name": "calculator", "args": {"expression": "23*17+9"}},
            {"name": "local_file_search", "args": {"query": "about TOOL CALLING locally", "root_dir": "docs"}},
        ]
        expected = [
            {"name": "calculator", "args_contains": {"expression": "23 * 17 + 9"}},
            {"name": "local_file_search", "args_contains": {"query": "tool calling", "root_dir": "docs"}},
        ]
        valid, errors = expected_calls_match(actual, expected)
        self.assertTrue(valid)
        self.assertEqual(errors, [])

    def test_acceptance_uses_capability_gates_without_hiding_failed_rows(self) -> None:
        benchmark = {
            "coverage_complete": True,
            "profile_success_counts": {"qwen35_4b": 1, "qwen3_1_7b": 2},
            "binding_success_counts": {"prompt_json": 1, "native_tools": 2},
            "multi_tool_success_count": 3,
            "usage_evidence_count": 4,
            "run_count": 4,
            "failed_run_count": 1,
            "all_cases_successful": False,
        }
        with tempfile.TemporaryDirectory() as temporary, patch(
            "b4_acceptance.run_benchmark", return_value=benchmark
        ), patch(
            "b4_acceptance.run_plan_execute", return_value={"status": "success"}
        ):
            result = run_acceptance("cases", "task", "models", "tools", temporary)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["individual_benchmark_failures"], 1)
        self.assertFalse(result["all_individual_runs_successful"])


if __name__ == "__main__":
    unittest.main()
