from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skills.composite_skills import CompositeSkillError, analyze_and_convert, read_and_convert
from skills.core.errors import ErrorCode


class CompositeSkillTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "docs").mkdir()
        (self.root / "tables").mkdir()
        (self.root / "docs" / "intro.txt").write_text("第一点\n第二点\n", encoding="utf-8")
        (self.root / "tables" / "scores.csv").write_text(
            "name,score\na,10\nb,20\n",
            encoding="utf-8",
        )
        self.output = self.root / "acceptance"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_read_and_convert_propagates_output_directory(self) -> None:
        result = read_and_convert(
            "docs/intro.txt",
            data_root=str(self.root),
            output_dir=str(self.output),
        )
        generated = Path(result["generated_file_path"]).resolve()
        generated.relative_to(self.output.resolve())
        self.assertEqual([item["status"] for item in result["step_trace"]], ["success", "success"])

    def test_analyze_and_convert_supports_json(self) -> None:
        result = analyze_and_convert(
            "tables/scores.csv",
            target_format="json",
            data_root=str(self.root),
            output_dir=str(self.output),
        )
        converted = json.loads(result["final_output"])
        self.assertEqual(converted["num_rows"], 2)
        Path(result["generated_file_path"]).resolve().relative_to(self.output.resolve())

    def test_composite_error_preserves_failed_step_and_cause(self) -> None:
        with self.assertRaises(CompositeSkillError) as captured:
            read_and_convert(
                "docs/missing.txt",
                data_root=str(self.root),
                output_dir=str(self.output),
            )
        self.assertEqual(captured.exception.code, ErrorCode.COMPOSITE_STEP_FAILED)
        self.assertEqual(captured.exception.details["failed_step"], "file_reader")
        self.assertEqual(captured.exception.details["cause_code"], ErrorCode.FILE_NOT_FOUND.value)


if __name__ == "__main__":
    unittest.main()
