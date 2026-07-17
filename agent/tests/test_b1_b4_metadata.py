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

from b1_agent_runtime import run_agent


class B1B4MetadataTests(unittest.TestCase):
    def test_mock_loop_records_b4_metadata_without_memory_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = run_agent(
                str(PROJECT_ROOT / "data" / "b4_eval" / "b1_mock_runtime.json"),
                str(PROJECT_ROOT / "configs" / "tools.yaml"),
                str(PROJECT_ROOT / "configs" / "memory.yaml"),
                str(PROJECT_ROOT / "configs" / "model.yaml"),
                temporary,
                "mock",
            )
            trace = json.loads((Path(temporary) / "trace.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "success")
        self.assertEqual(trace["llm"]["profiles_used"], ["mock"])
        self.assertEqual(trace["llm"]["bindings_used"], ["mock"])
        self.assertEqual(trace["llm"]["usage"]["total_tokens"], 0)
        self.assertEqual(trace["memory_save"]["status"], "not_requested")
        self.assertTrue(all("tool_call_validation" in turn for turn in trace["turns"]))


if __name__ == "__main__":
    unittest.main()
