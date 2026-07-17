from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import local_model_hooks as hooks


def _result(*, content: str = "", calls: list[dict] | None = None) -> dict:
    return {
        "status": "success",
        "error": None,
        "ai_message": {"role": "assistant", "content": content, "tool_calls": calls or []},
        "profile": "qwen3_1_7b",
        "binding": "native_tools",
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        "tool_call_validation": {"valid": True, "errors": []},
    }


class LocalModelHookTests(unittest.TestCase):
    @patch.object(hooks, "generate_ai_message")
    def test_selector_requires_one_real_call(self, generate) -> None:
        generate.return_value = _result(calls=[{"id": "call_1", "name": "calculator", "args": {}}])
        selected = hooks.select_tool_with_local_model("calculate", [])
        self.assertEqual(selected["selected_tool"], "calculator")
        self.assertEqual(generate.call_args.kwargs["binding"], "native_tools")

    @patch.object(hooks, "generate_ai_message")
    def test_summary_enforces_character_cap(self, generate) -> None:
        generate.return_value = _result(content="123456")
        summarized = hooks.summarize_memory_with_local_model("source", 4)
        self.assertEqual(summarized["summary"], "1234")
        self.assertTrue(summarized["postprocess_truncated"])

    @patch.object(hooks, "generate_ai_message")
    def test_ab_hooks_preserve_model_metadata(self, generate) -> None:
        generate.return_value = _result(content="answer")
        answer = hooks.answer_with_memory_local_model("q", "memory")
        evaluation = hooks.evaluate_bad_memory_local_model("q", "a", "b", "bad")
        self.assertEqual(answer["answer"], "answer")
        self.assertEqual(evaluation["status"], "completed")
        self.assertEqual(generate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
