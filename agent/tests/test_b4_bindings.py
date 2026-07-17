from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from b4_core.bindings import (  # noqa: E402
    NativeToolBindingUnsupported,
    NativeToolsBinding,
    normalize_chat_messages,
)


class FakeTokenizer:
    def __init__(self, include_tools: bool = True):
        self.include_tools = include_tools
        self.calls = []

    def apply_chat_template(self, messages, **options):
        self.calls.append((messages, options))
        tools = options.get("tools") or []
        rendered = json.dumps(messages, ensure_ascii=False)
        if self.include_tools:
            rendered += json.dumps(tools, ensure_ascii=False)
        if options.get("tokenize"):
            return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
        return rendered


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


class BindingTests(unittest.TestCase):
    def test_post_system_messages_are_merged_at_front(self) -> None:
        messages = [
            {"role": "system", "content": "first"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "answer", "tool_calls": []},
            {"role": "system", "content": "new policy"},
            {"role": "user", "content": "next"},
        ]
        normalized = normalize_chat_messages(messages)
        self.assertEqual(normalized[0]["role"], "system")
        self.assertIn("first", normalized[0]["content"])
        self.assertIn("new policy", normalized[0]["content"])
        self.assertFalse(any(item["role"] == "system" for item in normalized[1:]))

    def test_flat_assistant_calls_become_native_contract(self) -> None:
        normalized = normalize_chat_messages(
            [
                {"role": "user", "content": "calculate"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "c1", "name": "calculator", "args": {"expression": "2+2"}}],
                },
            ]
        )
        call = normalized[-1]["tool_calls"][0]
        self.assertEqual(call["type"], "function")
        self.assertEqual(call["function"]["arguments"], {"expression": "2+2"})

    def test_native_binding_verifies_tools_were_rendered(self) -> None:
        binding = NativeToolsBinding()
        binding.render(FakeTokenizer(include_tools=True), [{"role": "user", "content": "x"}], TOOLS, "")
        with self.assertRaises(NativeToolBindingUnsupported):
            binding.render(FakeTokenizer(include_tools=False), [{"role": "user", "content": "x"}], TOOLS, "")


if __name__ == "__main__":
    unittest.main()
