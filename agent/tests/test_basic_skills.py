from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skills.calculator import calculator
from skills.core.errors import ErrorCode, SkillFault
from skills.file_reader import file_reader
from skills.format_converter import format_converter
from skills.table_analyzer import table_analyzer


class BasicSkillTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "docs").mkdir()
        (self.root / "tables").mkdir()
        (self.root / "docs" / "intro.txt").write_text("第一行\n第二行\n", encoding="utf-8")
        (self.root / "tables" / "sample.csv").write_text(
            "name,score\nalpha,10\nbeta,20\n",
            encoding="utf-8",
        )
        self.output = self.root / "output"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_calculator_success_and_error(self) -> None:
        self.assertEqual(calculator("23 * 17 + 9")["result"], 400)
        with self.assertRaises(SkillFault) as captured:
            calculator("1 / 0")
        self.assertEqual(captured.exception.code, ErrorCode.EXECUTION_ERROR)
        with self.assertRaises(SkillFault) as captured:
            calculator("__import__('os')")
        self.assertEqual(captured.exception.code, ErrorCode.UNSUPPORTED_OPERATION)

    def test_file_reader_is_bounded_and_rooted(self) -> None:
        result = file_reader("docs/intro.txt", max_chars=4, data_root=str(self.root))
        self.assertEqual(result["content"], "第一行\n")
        self.assertTrue(result["truncated"])
        self.assertEqual(result["source"], "docs/intro.txt")
        with self.assertRaises(SkillFault) as captured:
            file_reader("../outside.txt", data_root=str(self.root))
        self.assertEqual(captured.exception.code, ErrorCode.PATH_OUTSIDE_ROOT)

    def test_table_analyzer_streams_statistics(self) -> None:
        result = table_analyzer("tables/sample.csv", data_root=str(self.root))
        self.assertEqual(result["num_rows"], 2)
        self.assertEqual(result["num_columns"], 2)
        self.assertEqual(result["describe"]["score"]["mean"], 15.0)
        json.dumps(result, ensure_ascii=False)

    def test_format_converter_writes_only_to_output_directory(self) -> None:
        result = format_converter(
            "a: 1\nb: 2",
            "json",
            output_filename="result.json",
            output_dir=str(self.output),
        )
        generated = Path(result["generated_file_path"]).resolve()
        generated.relative_to(self.output.resolve())
        self.assertEqual(json.loads(result["formatted_text"]), {"a": "1", "b": "2"})
        with self.assertRaises(SkillFault) as captured:
            format_converter("a: 1", "markdown", "../escape.md", str(self.output))
        self.assertEqual(captured.exception.code, ErrorCode.PERMISSION_DENIED)


if __name__ == "__main__":
    unittest.main()
