from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
for path in (PROJECT_ROOT, CODE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from b2_run_skill import main, run_skill
from skills.core.catalog import list_skill_names
from skills.core.invoker import invoke_skill


class B2ContractTests(unittest.TestCase):
    def test_catalog_has_exact_basic_and_expected_advanced_skills(self) -> None:
        self.assertEqual(
            list_skill_names("basic"),
            ["calculator", "file_reader", "format_converter", "local_file_search", "table_analyzer"],
        )
        self.assertEqual(
            list_skill_names("advanced"),
            ["analyze_and_convert", "code_executor", "read_and_convert"],
        )

    def test_five_normal_and_five_error_samples_have_one_contract(self) -> None:
        normal = {
            "calculator": {"expression": "23 * 17 + 9"},
            "file_reader": {"path": "docs/agent_intro.txt", "max_chars": 2000},
            "local_file_search": {"query": "工具编排", "root_dir": "docs", "top_k": 3},
            "table_analyzer": {"path": "tables/results.csv", "describe": True},
            "format_converter": {"text": "a: 1\nb: 2", "target_format": "markdown"},
        }
        errors = {
            "calculator": {"expression": "23 / 0"},
            "file_reader": {"path": "docs/missing.txt"},
            "local_file_search": {"query": "Agent", "root_dir": "missing"},
            "table_analyzer": {"path": "tables/missing.csv"},
            "format_converter": {"text": "a: 1", "target_format": "xml"},
        }
        expected_codes = {
            "calculator": "EXECUTION_ERROR",
            "file_reader": "FILE_NOT_FOUND",
            "local_file_search": "FILE_NOT_FOUND",
            "table_analyzer": "FILE_NOT_FOUND",
            "format_converter": "UNSUPPORTED_OPERATION",
        }
        with tempfile.TemporaryDirectory() as directory:
            for skill_name, payload in normal.items():
                with self.subTest(skill=skill_name, kind="normal"):
                    result = run_skill(
                        skill_name,
                        payload,
                        str(PROJECT_ROOT / "data"),
                        directory,
                    )
                    self.assertEqual(result["status"], "success")
                    self.assertIsNone(result["error"])
                    json.dumps(result, ensure_ascii=False)
            for skill_name, payload in errors.items():
                with self.subTest(skill=skill_name, kind="error"):
                    result = run_skill(
                        skill_name,
                        payload,
                        str(PROJECT_ROOT / "data"),
                        directory,
                    )
                    self.assertEqual(result["status"], "error")
                    self.assertEqual(result["error"]["code"], expected_codes[skill_name])
                    json.dumps(result, ensure_ascii=False)

    def test_cli_writes_result_and_log_for_business_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.json"
            input_path.write_text('{"expression":"1/0"}', encoding="utf-8")
            output = root / "output"
            exit_code = main(
                [
                    "--skill",
                    "calculator",
                    "--input",
                    str(input_path),
                    "--outdir",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 0)
            result = json.loads((output / "calculator_result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["error"]["code"], "EXECUTION_ERROR")
            log_line = json.loads((output / "skill_run_log.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(log_line["error_code"], "EXECUTION_ERROR")

    def test_missing_parameter_matches_b3_error_contract(self) -> None:
        result = invoke_skill("calculator", {})
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "PARAM_MISSING")
        self.assertEqual(result["error"]["details"]["missing"], ["expression"])

    def test_initialization_and_non_json_input_still_return_json_envelopes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            invalid_limits = Path(directory) / "limits.json"
            invalid_limits.write_text('{"search":{"max_files":1.5}}', encoding="utf-8")
            configured = invoke_skill(
                "calculator",
                {"expression": "1+1"},
                limits_config=invalid_limits,
            )
        self.assertEqual(configured["status"], "error")
        self.assertEqual(configured["error"]["code"], "PARAM_OUT_OF_RANGE")
        malformed = invoke_skill("calculator", {"expression": {1, 2}})
        self.assertEqual(malformed["error"]["code"], "SERIALIZATION_ERROR")
        json.dumps(malformed, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
