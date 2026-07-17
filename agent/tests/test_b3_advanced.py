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

from b3_advanced import compare_schema_descriptions, execute_with_retry_cache_stats
from b3_tool_layer import get_tools_schema
from common.schemas import make_tool_message


TOOLS_CONFIG = ROOT / "configs" / "tools.yaml"


def _message(call: dict, status: str, *, retryable: bool = False) -> dict:
    result = {
        "skill_name": call["name"],
        "status": status,
        "input": call.get("args", {}),
        "output": {"result": 400} if status == "success" else None,
        "error": None
        if status == "success"
        else {
            "code": "TIMEOUT" if retryable else "PARAM_INVALID",
            "type": "SyntheticError",
            "message": "synthetic failure",
            "retryable": retryable,
            "details": {},
        },
        "latency_ms": 1.0,
    }
    return make_tool_message(call["id"], call["name"], json.dumps(result), status)


class B3AdvancedTests(unittest.TestCase):
    def test_retry_only_when_error_is_retryable(self) -> None:
        calls = {"count": 0}

        def executor(tool_calls, tools_config, toolset, outdir):
            calls["count"] += 1
            status = "error" if calls["count"] == 1 else "success"
            return [_message(tool_calls[0], status, retryable=status == "error")]

        with tempfile.TemporaryDirectory() as temporary:
            report = execute_with_retry_cache_stats(
                [{"id": "retry", "name": "calculator", "args": {"expression": "1+1"}}],
                TOOLS_CONFIG,
                "basic_tools",
                temporary,
                max_retries=2,
                executor=executor,
            )
        self.assertEqual(calls["count"], 2)
        self.assertEqual(report["retry_count"], 1)
        self.assertEqual(report["success_count"], 1)

    def test_non_retryable_error_is_not_retried_or_cached(self) -> None:
        calls = {"count": 0}

        def executor(tool_calls, tools_config, toolset, outdir):
            calls["count"] += 1
            return [_message(tool_calls[0], "error", retryable=False)]

        with tempfile.TemporaryDirectory() as temporary:
            for suffix in ("one", "two"):
                execute_with_retry_cache_stats(
                    [{"id": suffix, "name": "calculator", "args": {"expression": "bad"}}],
                    TOOLS_CONFIG,
                    "basic_tools",
                    temporary,
                    max_retries=3,
                    executor=executor,
                )
        self.assertEqual(calls["count"], 2)

    def test_success_cache_reuses_result_with_new_call_id(self) -> None:
        calls = {"count": 0}

        def executor(tool_calls, tools_config, toolset, outdir):
            calls["count"] += 1
            return [_message(tool_calls[0], "success")]

        with tempfile.TemporaryDirectory() as temporary:
            first = execute_with_retry_cache_stats(
                [{"id": "first", "name": "calculator", "args": {"expression": "2+2"}}],
                TOOLS_CONFIG,
                "basic_tools",
                temporary,
                executor=executor,
            )
            second = execute_with_retry_cache_stats(
                [{"id": "second", "name": "calculator", "args": {"expression": "2+2"}}],
                TOOLS_CONFIG,
                "basic_tools",
                temporary,
                executor=executor,
            )
            messages = json.loads((Path(temporary) / "b3_advanced_tool_messages.json").read_text(encoding="utf-8"))
        self.assertEqual(calls["count"], 1)
        self.assertEqual(first["cache_hit_count"], 0)
        self.assertEqual(second["cache_hit_count"], 1)
        self.assertEqual(messages[0]["tool_call_id"], "second")

    def test_schema_comparison_calls_injected_selector(self) -> None:
        schema = get_tools_schema(str(TOOLS_CONFIG), "basic_tools")
        invocations = []

        def selector(task, tools_schema):
            invocations.append((task, tools_schema))
            return {"selected_tool": "calculator", "backend": "test-double"}

        cases = [{"task": "calculate 1+1", "expected_tool": "calculator"}]
        with tempfile.TemporaryDirectory() as temporary:
            report = compare_schema_descriptions(schema, cases, temporary, selector=selector)
        self.assertEqual(len(invocations), 2)
        self.assertEqual(report["metric"], "observed selector accuracy")
        self.assertEqual(report["variants"]["full_description"]["accuracy"], 1.0)

    def test_schema_comparison_without_selector_is_explicit_not_run(self) -> None:
        schema = get_tools_schema(str(TOOLS_CONFIG), "basic_tools")
        with tempfile.TemporaryDirectory() as temporary:
            report = compare_schema_descriptions(
                schema,
                [{"task": "calculate", "expected_tool": "calculator"}],
                temporary,
                selector=None,
            )
        self.assertEqual(report["status"], "not_run")
        self.assertIn("real model-backed selector", report["reason"])


if __name__ == "__main__":
    unittest.main()
