from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from b4_core.engine import RawGeneration, TokenUsage  # noqa: E402
from b4_core.planning import generate_validated_plan, plan_layers, validate_plan  # noqa: E402


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    }
]


def plan(step_id="step_1", depends_on=None):
    return {
        "goal": "calculate",
        "steps": [
            {
                "id": step_id,
                "description": "calculate exactly",
                "tool_name": "calculator",
                "arguments": {"expression": "2+2"},
                "depends_on": depends_on or [],
            }
        ],
    }


class FakePlanningEngine:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.messages_seen = []

    def prompt(self, name):
        return "return a plan"

    def generate_raw(self, messages, tools_schema, **kwargs):
        self.messages_seen.append(messages)
        text = self.outputs.pop(0)
        return RawGeneration(
            raw_text=text,
            prompt_text="prompt",
            usage=TokenUsage(10, 5, 15),
            profile="planner",
            route_reason="test",
            binding="plain",
            model_family="test",
            native_parser=None,
            cache_hit=False,
            load_latency_ms=0.0,
            inference_latency_ms=1.0,
        )


class PlanningTests(unittest.TestCase):
    def test_valid_plan_and_parallel_layers(self) -> None:
        candidate = plan()
        candidate["steps"].append(
            {
                "id": "step_2",
                "description": "another calculation",
                "tool_name": "calculator",
                "arguments": {"expression": "3+3"},
                "depends_on": [],
            }
        )
        self.assertTrue(validate_plan(candidate, TOOLS)["valid"])
        self.assertEqual(len(plan_layers(candidate)[0]), 2)

    def test_unknown_tool_and_cycle_are_rejected(self) -> None:
        unknown = plan()
        unknown["steps"][0]["tool_name"] = "missing"
        self.assertFalse(validate_plan(unknown, TOOLS)["valid"])

        cyclic = plan("step_1", ["step_2"])
        cyclic["steps"].append(
            {
                "id": "step_2",
                "description": "second",
                "tool_name": "calculator",
                "arguments": {"expression": "3+3"},
                "depends_on": ["step_1"],
            }
        )
        result = validate_plan(cyclic, TOOLS)
        self.assertFalse(result["valid"])
        self.assertTrue(any("cycle" in item["message"] for item in result["errors"]))

    def test_planner_gets_exactly_one_correction_attempt(self) -> None:
        engine = FakePlanningEngine(["not json", __import__("json").dumps(plan())])
        validated = generate_validated_plan(engine, "calculate", TOOLS, profile="planner")
        self.assertEqual(len(validated.attempts), 2)
        self.assertEqual(len(engine.messages_seen), 2)
        self.assertIn("failed validation", engine.messages_seen[1][-1]["content"])


if __name__ == "__main__":
    unittest.main()
