from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from b5_memory import load_memory, save_memory, search_memory_keyword


class TemporaryMemory:
    def __init__(self, root: Path, *, max_chars: int = 2000, top_k: int = 1) -> None:
        self.root = root
        self.memory = root / "memory"
        (self.memory / "global").mkdir(parents=True)
        (self.memory / "conversations").mkdir(parents=True)
        self.config = root / "memory.json"
        self.config.write_text(
            json.dumps(
                {
                    "memory": {
                        "root_dir": "memory",
                        "global_memory_dir": "global",
                        "conversation_memory_dir": "conversations",
                        "index_path": "memory_index.json",
                        "max_memory_chars": max_chars,
                        "retrieval": {"keyword_top_k": top_k, "keyword_min_score": 0.0},
                    }
                }
            ),
            encoding="utf-8",
        )
        self.index = {
            "global_apple": {
                "memory_id": "global_apple",
                "memory_type": "global",
                "title": "Apple guide",
                "summary": "apple orchard notes",
                "path": "global/apple.md",
            },
            "global_space": {
                "memory_id": "global_space",
                "memory_type": "global",
                "title": "Space guide",
                "summary": "rocket orbit",
                "path": "global/space.md",
            },
            "conversation_explicit": {
                "memory_id": "conversation_explicit",
                "memory_type": "conversation",
                "title": "Explicit history",
                "summary": "selected by id",
                "path": "conversations/explicit.md",
            },
        }
        (self.memory / "global" / "apple.md").write_text("apple fruit orchard", encoding="utf-8")
        (self.memory / "global" / "space.md").write_text("rocket moon orbit", encoding="utf-8")
        (self.memory / "conversations" / "explicit.md").write_text("explicit selected memory", encoding="utf-8")
        (self.memory / "memory_index.json").write_text(json.dumps(self.index), encoding="utf-8")


class B5MemoryTests(unittest.TestCase):
    def test_query_drives_global_top_k_and_explicit_id_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = TemporaryMemory(Path(temporary), top_k=1)
            result = load_memory(
                str(fixture.config),
                ["conversation_explicit"],
                True,
                "apple orchard",
            )
        self.assertEqual(
            [item["memory_id"] for item in result["selected_memory_docs"]],
            ["global_apple", "conversation_explicit"],
        )
        self.assertTrue(result["query_ranking"]["applied"])
        self.assertGreater(result["selected_memory_docs"][0]["retrieval_score"], 0)

    def test_keyword_search_filters_zero_score_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = TemporaryMemory(Path(temporary), top_k=5)
            result = search_memory_keyword(fixture.config, "rocket", 5, include_content=False)
        self.assertEqual([item["memory_id"] for item in result["results"]], ["global_space"])
        self.assertTrue(all(item["score"] > 0 for item in result["results"]))

    def test_character_budget_and_missing_id_errors_remain_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = TemporaryMemory(Path(temporary), max_chars=5)
            result = load_memory(str(fixture.config), ["conversation_explicit", "missing"], False, "query")
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["total_chars"], 5)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["errors"][0]["type"], "MemoryNotFound")

    def test_save_memory_writes_only_to_configured_temporary_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = TemporaryMemory(root)
            messages = root / "messages.json"
            trace = root / "trace.json"
            answer = root / "answer.md"
            messages.write_text("[]", encoding="utf-8")
            trace.write_text("{}", encoding="utf-8")
            answer.write_text("temporary answer", encoding="utf-8")
            result = save_memory(
                str(fixture.config),
                "temp_case",
                "conversation",
                str(messages),
                str(trace),
                str(answer),
            )
            target = fixture.memory / result["path"]
            self.assertTrue(target.is_file())
            self.assertTrue(str(target.resolve()).startswith(str(fixture.memory.resolve())))


if __name__ == "__main__":
    unittest.main()
