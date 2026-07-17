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

from b5_advanced import (
    analyze_bad_memory_ab,
    search_memory_embeddings,
    summarize_memory_document,
    update_memory_document,
)
from test_b5_memory import TemporaryMemory


class B5AdvancedTests(unittest.TestCase):
    def test_injected_local_embedding_provider_ranks_real_vectors(self) -> None:
        def embedder(texts):
            return [[1.0, 0.0] if "apple" in text.casefold() else [0.0, 1.0] for text in texts]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = TemporaryMemory(root)
            result = search_memory_embeddings(fixture.config, "apple", 2, root / "out", embedder=embedder)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["results"][0]["memory_id"], "global_apple")
        self.assertFalse(result["fallback_used"])

    def test_embedding_failure_is_explicit_and_never_hash_fallback(self) -> None:
        def broken(_texts):
            raise RuntimeError("model unavailable")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = TemporaryMemory(root)
            result = search_memory_embeddings(fixture.config, "apple", 2, root / "out", embedder=broken)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "EMBEDDING_UNAVAILABLE")
        self.assertFalse(result["fallback_used"])

    def test_llm_summary_interface_and_not_configured_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = TemporaryMemory(root)
            not_run = summarize_memory_document(
                fixture.config, "global_apple", root / "none", summarizer=None
            )

            def summarizer(text, max_chars):
                return {"summary": "apple summary", "model": "test-model"}

            success = summarize_memory_document(
                fixture.config, "global_apple", root / "yes", summarizer=summarizer
            )
        self.assertEqual(not_run["status"], "not_run")
        self.assertFalse(not_run["fallback_used"])
        self.assertEqual(success["summary"], "apple summary")
        self.assertEqual(success["summarizer_metadata"]["model"], "test-model")

    def test_update_records_duplicate_addition_and_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = TemporaryMemory(root)
            target = fixture.memory / "conversations" / "explicit.md"
            target.write_text("# Facts\n\nmode: safe\n- exact fact\n", encoding="utf-8")
            result = update_memory_document(
                fixture.config,
                "conversation_explicit",
                ["exact fact", "new fact", "mode: unsafe"],
                root / "out",
                conflict_policy="record",
            )
            content = target.read_text(encoding="utf-8")
        self.assertTrue(result["changed"])
        self.assertEqual(len(result["duplicates"]), 1)
        self.assertEqual(result["additions"], ["new fact"])
        self.assertEqual(result["conflicts"][0]["resolution"], "recorded_without_change")
        self.assertIn("mode: safe", content)
        self.assertNotIn("mode: unsafe", content)
        self.assertIn("new fact", content)

    def test_bad_memory_ab_uses_two_real_responder_calls(self) -> None:
        calls = []

        def responder(query, context):
            calls.append(context)
            answer = "incorrect influenced answer" if "BAD CLAIM" in context else "baseline answer"
            return {"answer": answer, "model": "test-model"}

        def evaluator(query, baseline, injected, bad_memory):
            return {"status": "success", "degraded": baseline != injected}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = TemporaryMemory(root)
            result = analyze_bad_memory_ab(
                fixture.config,
                "apple",
                "BAD CLAIM",
                1,
                root / "out",
                responder=responder,
                evaluator=evaluator,
            )
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["observations"]["answer_changed"])
        self.assertTrue(result["evaluation"]["degraded"])
        self.assertFalse(result["fallback_used"])

    def test_bad_memory_ab_without_responder_is_not_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = TemporaryMemory(root)
            result = analyze_bad_memory_ab(
                fixture.config, "apple", "BAD CLAIM", 1, root / "out", responder=None
            )
        self.assertEqual(result["status"], "not_run")
        self.assertEqual(result["error"]["code"], "AB_RESPONDER_NOT_CONFIGURED")

    def test_bad_memory_ab_can_use_explicit_memory_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = TemporaryMemory(root)
            result = analyze_bad_memory_ab(
                fixture.config,
                "apple",
                "BAD CLAIM",
                1,
                root / "out",
                responder=lambda query, context: {"answer": "ok"},
                memory_ids=["global_apple"],
            )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["retrieval_mode"], "explicit_memory_ids")
        self.assertEqual(result["retrieved_memory_ids"], ["global_apple"])


if __name__ == "__main__":
    unittest.main()
