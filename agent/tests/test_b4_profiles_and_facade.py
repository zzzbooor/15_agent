from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from b4_core.profiles import choose_profile, normalized_profiles  # noqa: E402
from b4_local_agent_llm import generate_ai_message  # noqa: E402


CONFIG = {
    "runtime": {"default_profile": "large"},
    "profiles": {
        "large": {"model_name_or_path": "/large"},
        "small": {"model_name_or_path": "/small"},
    },
    "routing": {
        "planner_profile": "large",
        "fast_profile": "small",
        "fast_max_input_chars": 100,
        "fast_max_tools": 2,
    },
}


class ProfileAndFacadeTests(unittest.TestCase):
    def test_legacy_model_config_becomes_default_profile(self) -> None:
        profiles = normalized_profiles({"model": {"model_name_or_path": "/model"}})
        self.assertIn("default", profiles)

    def test_router_switches_by_strategy_and_size(self) -> None:
        short = choose_profile(CONFIG, [{"content": "short"}], [], strategy="react")
        planner = choose_profile(CONFIG, [{"content": "short"}], [], strategy="plan_execute")
        explicit = choose_profile(CONFIG, [], [], requested="small")
        self.assertEqual(short.name, "small")
        self.assertEqual(planner.name, "large")
        self.assertEqual(explicit.reason, "explicit profile override")

    def test_mock_keeps_old_positional_signature_and_adds_metadata(self) -> None:
        messages = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "read docs/agent_intro.txt"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            result = generate_ai_message(
                "unused.yaml",
                messages,
                [],
                "mock",
                temporary,
                "compat",
            )
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["ai_message"]["tool_calls"][0]["name"], "file_reader")
            self.assertEqual(result["usage"]["total_tokens"], 0)
            self.assertEqual(result["profile"], "mock")
            raw = json.loads((Path(temporary) / "compat_raw_model_output.json").read_text(encoding="utf-8"))
            self.assertIn("metadata", raw)

    def test_schema_invalid_model_call_is_not_reported_as_success(self) -> None:
        raw = SimpleNamespace(
            raw_text='{"content":"","tool_calls":[{"name":"calculator","args":{}}]}',
            prompt_text="prompt",
            metadata=lambda: {
                "profile": "fake",
                "binding": "prompt_json",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )
        parsed = SimpleNamespace(
            ai_message={
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "name": "calculator", "args": {}}],
            },
            candidate={"content": "", "tool_calls": [{"name": "calculator", "args": {}}]},
        )
        decision = SimpleNamespace(
            raw=raw,
            parsed=parsed,
            error=None,
            tool_call_validation={"valid": False, "errors": [{"message": "missing expression"}]},
        )
        engine = SimpleNamespace(generate_ai_message=lambda *args, **kwargs: decision)
        messages = [{"role": "user", "content": "calculate"}]
        schema = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                },
            }
        ]
        with patch("b4_local_agent_llm.DecisionEngine", return_value=engine):
            result = generate_ai_message("unused.yaml", messages, schema, "prompt_json")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["type"], "ToolCallSchemaValidationError")
        self.assertEqual(result["ai_message"]["tool_calls"], [])


if __name__ == "__main__":
    unittest.main()
