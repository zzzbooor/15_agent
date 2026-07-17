from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skills.code_executor import code_executor
from skills.core.errors import ErrorCode, SkillFault


class CodeExecutorSecurityTests(unittest.TestCase):
    def test_safe_math_and_bounded_loop(self) -> None:
        result = code_executor(
            "import math\ntotal = 0\nfor i in range(5):\n    total += i\nprint(math.sqrt(16), total)",
            timeout=3,
        )
        self.assertEqual(result["stdout"], "4.0 10\n")
        self.assertEqual(result["engine"], "restricted_ast_v1")
        self.assertFalse(result["isolation"]["user_source_executed_directly"])

    def test_common_escape_attempts_are_rejected(self) -> None:
        attacks = {
            "forbidden_import": "import os\nprint(os.getcwd())",
            "builtins_subscript": "__builtins__['open']('secret.txt')",
            "getattr_import": "getattr(__builtins__, '__import__')('os')",
            "dunder_chain": "print((1).__class__.__mro__)",
            "direct_file": "open('sentinel.txt', 'w')",
            "infinite_loop": "while True:\n    pass",
            "dynamic_compile": "compile('1+1', '<x>', 'eval')",
        }
        for name, source in attacks.items():
            with self.subTest(name=name), self.assertRaises(SkillFault) as captured:
                code_executor(source, timeout=1)
            self.assertEqual(captured.exception.code, ErrorCode.SANDBOX_VIOLATION)

    def test_resource_budgets_are_enforced(self) -> None:
        with self.assertRaises(SkillFault) as captured:
            code_executor("print('x' * 40000)")
        self.assertEqual(captured.exception.code, ErrorCode.RESOURCE_EXHAUSTED)
        with self.assertRaises(SkillFault) as captured:
            code_executor("for i in range(10001):\n    pass")
        self.assertEqual(captured.exception.code, ErrorCode.RESOURCE_EXHAUSTED)

    def test_file_write_attempt_cannot_create_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "sentinel.txt"
            source = f"open({str(sentinel)!r}, 'w')"
            with self.assertRaises(SkillFault):
                code_executor(source)
            self.assertFalse(sentinel.exists())


if __name__ == "__main__":
    unittest.main()
