from __future__ import annotations

import argparse
import sys
from pathlib import Path

from b4_core.engine import release_model_cache
from b5_advanced import (
    analyze_bad_memory_ab,
    search_memory_embeddings,
    summarize_memory_document,
    update_memory_document,
)
from common.io_utils import read_json, read_text, write_json
from common.path_utils import resolve_cli_path
from local_model_hooks import (
    answer_with_memory_local_model,
    evaluate_bad_memory_local_model,
    summarize_memory_with_local_model,
)


def run_acceptance(
    config_path: str,
    update_input_path: str,
    bad_memory_path: str,
    outdir: str,
) -> dict:
    output_dir = Path(outdir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    update_input = read_json(update_input_path)
    if not isinstance(update_input, dict):
        raise ValueError("memory update sample must be an object")
    if not update_input.get("dry_run"):
        raise ValueError("B5 acceptance requires a dry_run memory update sample")

    embedding = search_memory_embeddings(
        config_path,
        "Agent 工具调用 Memory",
        3,
        output_dir / "embedding",
    )
    summary = summarize_memory_document(
        config_path,
        "mem_course_001",
        output_dir / "summary",
        summarizer=summarize_memory_with_local_model,
        max_chars=240,
    )
    update = update_memory_document(
        config_path,
        update_input["memory_id"],
        update_input["updates"],
        output_dir / "update",
        conflict_policy=update_input.get("conflict_policy", "record"),
        dry_run=True,
        summarizer=summarize_memory_with_local_model,
    )
    ab_result = analyze_bad_memory_ab(
        config_path,
        "根据给定记忆，Agent 基础概念中，模型负责决策、工具负责执行、运行循环负责协调吗？",
        read_text(bad_memory_path),
        1,
        output_dir / "bad_memory_ab",
        responder=answer_with_memory_local_model,
        evaluator=evaluate_bad_memory_local_model,
        memory_ids=["mem_course_001"],
    )
    checks = {
        "real_local_embedding": embedding.get("status") in {"success", "partial"}
        and embedding.get("fallback_used") is False
        and isinstance(embedding.get("embedding_dimensions"), int),
        "local_llm_summary": summary.get("status") == "success"
        and summary.get("fallback_used") is False,
        "memory_update_dry_run": update.get("status") == "success"
        and update.get("dry_run") is True,
        "bad_memory_real_ab": ab_result.get("status") == "success"
        and ab_result.get("fallback_used") is False,
    }
    result = {
        "status": "success" if all(checks.values()) else "completed_with_failures",
        "checks": checks,
        "production_memory_modified": False,
        "artifacts": {
            "embedding": "embedding/b5_embedding_search.json",
            "summary": "summary/b5_memory_summary.json",
            "update": "update/b5_memory_update_result.json",
            "bad_memory_ab": "bad_memory_ab/b5_bad_memory_ab_analysis.json",
        },
    }
    write_json(result, output_dir / "b5_acceptance_summary.json")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run real local-model B5 advanced acceptance checks.")
    parser.add_argument("--config", default="../configs/memory.yaml")
    parser.add_argument("--update_input", default="../data/b5_eval/memory_update_dry_run.json")
    parser.add_argument("--bad_memory", default="../data/b5_eval/bad_memory.txt")
    parser.add_argument("--outdir", default="../outputs/B5_acceptance")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_acceptance(
            str(resolve_cli_path(args.config)),
            str(resolve_cli_path(args.update_input)),
            str(resolve_cli_path(args.bad_memory)),
            str(resolve_cli_path(args.outdir)),
        )
        print(resolve_cli_path(args.outdir) / "b5_acceptance_summary.json")
        return 0 if result["status"] == "success" else 2
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        release_model_cache()


if __name__ == "__main__":
    raise SystemExit(main())
