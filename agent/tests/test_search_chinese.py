from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skills.core.errors import ErrorCode, SkillFault
from skills.core.context import bind_context, make_context
from skills.local_file_search import local_file_search


class ChineseSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        docs = self.root / "docs"
        docs.mkdir()
        (docs / "workflow.md").write_text(
            "检索演示文档。工具编排强调模型先选择工具，再由运行时执行。",
            encoding="utf-8",
        )
        (docs / "memory.txt").write_text(
            "记忆模块负责保存历史上下文。",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_chinese_phrase_ranks_expected_document_first(self) -> None:
        result = local_file_search("工具编排", root_dir="docs", top_k=3, data_root=str(self.root))
        self.assertTrue(result["results"])
        self.assertEqual(result["results"][0]["path"], "docs/workflow.md")
        self.assertTrue(result["results"][0]["phrase_match"])
        self.assertGreater(result["results"][0]["score"], 0)

    def test_unrelated_query_returns_no_zero_score_documents(self) -> None:
        result = local_file_search("quantum-zebra-987654", root_dir="docs", data_root=str(self.root))
        self.assertEqual(result["results"], [])

    def test_invalid_file_type_is_rejected(self) -> None:
        with self.assertRaises(SkillFault) as captured:
            local_file_search("工具", file_types=["py"], data_root=str(self.root))
        self.assertEqual(captured.exception.code, ErrorCode.UNSUPPORTED_OPERATION)

    def test_query_and_index_budgets_are_enforced(self) -> None:
        with self.assertRaises(SkillFault) as captured:
            local_file_search("x" * 501, data_root=str(self.root))
        self.assertEqual(captured.exception.code, ErrorCode.RESOURCE_EXHAUSTED)

        limits_path = self.root / "small_limits.json"
        limits_path.write_text(
            '{"search":{"max_query_chars":500,"max_entries":100,"max_files":100,'
            '"max_file_bytes":100000,"max_total_bytes":100000,"max_index_tokens":5,'
            '"timeout_seconds":5.0,"max_top_k":5,"snippet_radius":20,'
            '"max_skipped_records":5}}',
            encoding="utf-8",
        )
        context = make_context(self.root, limits_config=limits_path)
        with bind_context(context):
            result = local_file_search("工具", root_dir="docs")
        self.assertEqual(result["limit_reached"], "max_index_tokens")
        self.assertLessEqual(result["indexed_tokens"], 5)

    def test_external_symbolic_link_is_not_indexed(self) -> None:
        outside = self.root.parent / f"{self.root.name}_secret.txt"
        outside.write_text("外部机密 工具编排", encoding="utf-8")
        link = self.root / "docs" / "linked.txt"
        try:
            os.symlink(outside, link)
        except (OSError, NotImplementedError):
            outside.unlink(missing_ok=True)
            self.skipTest("symbolic links are unavailable for this account")
        try:
            result = local_file_search("外部机密", root_dir="docs", data_root=str(self.root))
            self.assertFalse(any(item["path"] == "docs/linked.txt" for item in result["results"]))
            self.assertTrue(any(item["code"] == ErrorCode.PERMISSION_DENIED.value for item in result["skipped_files"]))
        finally:
            link.unlink(missing_ok=True)
            outside.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
