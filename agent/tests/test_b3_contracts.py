from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
for path in (str(CODE), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from b3_tool_layer import execute_tool_calls, get_auto_tools_schema, get_tools_schema
from skills.core.invoker import invoke_skill


TOOLS_CONFIG = ROOT / "configs" / "tools.yaml"


class B3ContractTests(unittest.TestCase):
    def test_auto_schema_resolves_real_annotation_types(self) -> None:
        schema = get_auto_tools_schema(str(TOOLS_CONFIG), "basic_tools")
        by_name = {item["function"]["name"]: item["function"] for item in schema}
        reader = by_name["file_reader"]["parameters"]
        self.assertEqual(reader["properties"]["max_chars"]["type"], "integer")
        self.assertEqual(reader["properties"]["max_chars"]["default"], 2000)
        self.assertEqual(reader["required"], ["path"])
        search_types = by_name["local_file_search"]["parameters"]["properties"]["file_types"]
        self.assertIn({"type": "array", "items": {"type": "string"}}, search_types["anyOf"])
        self.assertNotIn("data_root", by_name["local_file_search"]["parameters"]["properties"])

    def test_configured_public_schema_uses_function_source(self) -> None:
        schema = get_tools_schema(str(TOOLS_CONFIG), "basic_tools")
        by_name = {item["function"]["name"]: item["function"] for item in schema}
        self.assertEqual(by_name["table_analyzer"]["parameters"]["properties"]["describe"]["type"], "boolean")
        self.assertEqual(by_name["local_file_search"]["parameters"]["properties"]["top_k"]["type"], "integer")

    def test_b3_and_b2_invoker_return_same_skill_result(self) -> None:
        direct = invoke_skill("calculator", {"expression": "23 * 17 + 9"})
        with tempfile.TemporaryDirectory() as temporary:
            messages = execute_tool_calls(
                [{"id": "call_calc", "name": "calculator", "args": {"expression": "23 * 17 + 9"}}],
                str(TOOLS_CONFIG),
                "basic_tools",
                temporary,
            )
        via_b3 = json.loads(messages[0]["content"])
        self.assertEqual(via_b3["status"], direct["status"])
        self.assertEqual(via_b3["input"], direct["input"])
        self.assertEqual(via_b3["output"], direct["output"])
        self.assertEqual(via_b3["error"], direct["error"])
        self.assertEqual(messages[0]["tool_call_id"], "call_calc")

    def test_schema_validation_uses_shared_error_envelope(self) -> None:
        messages = execute_tool_calls(
            [{"id": "missing", "name": "calculator", "args": {}}],
            str(TOOLS_CONFIG),
            "basic_tools",
        )
        result = json.loads(messages[0]["content"])
        self.assertEqual(messages[0]["status"], "error")
        self.assertEqual(result["error"]["code"], "PARAM_MISSING")
        self.assertFalse(result["error"]["retryable"])
        self.assertEqual(result["error"]["details"]["missing"], ["expression"])

    def test_unknown_tool_is_structured_not_fatal(self) -> None:
        messages = execute_tool_calls(
            [{"id": "unknown", "name": "not_registered", "args": {}}],
            str(TOOLS_CONFIG),
            "basic_tools",
        )
        result = json.loads(messages[0]["content"])
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "UNSUPPORTED_OPERATION")


if __name__ == "__main__":
    unittest.main()
