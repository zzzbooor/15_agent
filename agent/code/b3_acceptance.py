from __future__ import annotations

import argparse
import sys
from pathlib import Path

from b3_advanced import compare_schema_descriptions, execute_with_retry_cache_stats
from b3_tool_layer import get_auto_tools_schema
from b4_core.engine import release_model_cache
from common.io_utils import read_json, write_json
from common.path_utils import resolve_cli_path
from local_model_hooks import select_tool_with_local_model


def run_acceptance(
    tools_config: str,
    cases_path: str,
    outdir: str,
    *,
    toolset: str = "basic_tools",
) -> dict:
    output_dir = Path(outdir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    schema = get_auto_tools_schema(tools_config, toolset, str(output_dir / "auto_schema"))
    repeated_calls = [
        {"id": "cache_first", "name": "calculator", "args": {"expression": "23 * 17 + 9"}},
        {"id": "cache_second", "name": "calculator", "args": {"expression": "23 * 17 + 9"}},
    ]
    stats = execute_with_retry_cache_stats(
        repeated_calls,
        tools_config,
        toolset,
        output_dir / "retry_cache",
        max_retries=1,
    )
    cases = read_json(cases_path)
    comparison = compare_schema_descriptions(
        schema,
        cases,
        output_dir / "schema_comparison",
        selector=select_tool_with_local_model,
    )
    checks = {
        "python_auto_schema": len(schema) >= 5,
        "success_only_cache": stats.get("success_count") == 2
        and stats.get("cache_hit_count") == 1
        and stats.get("cache_policy") == "success_only",
        "real_model_description_comparison": comparison.get("status") in {"success", "partial"}
        and comparison.get("selector", "").endswith("select_tool_with_local_model"),
    }
    result = {
        "status": "success" if all(checks.values()) else "completed_with_failures",
        "checks": checks,
        "artifacts": {
            "auto_schema": "auto_schema/b3_auto_schema_from_python.json",
            "retry_cache_stats": "retry_cache/b3_tool_call_stats.json",
            "description_comparison": "schema_comparison/b3_schema_description_comparison.json",
        },
        "description_variant_metrics": comparison.get("variants"),
    }
    write_json(result, output_dir / "b3_acceptance_summary.json")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run B3 advanced acceptance with a real local selector.")
    parser.add_argument("--tools_config", default="../configs/tools.yaml")
    parser.add_argument("--cases", default="../data/b3_eval/schema_selection_cases.json")
    parser.add_argument("--toolset", default="basic_tools")
    parser.add_argument("--outdir", default="../outputs/B3_acceptance")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_acceptance(
            str(resolve_cli_path(args.tools_config)),
            str(resolve_cli_path(args.cases)),
            str(resolve_cli_path(args.outdir)),
            toolset=args.toolset,
        )
        print(resolve_cli_path(args.outdir) / "b3_acceptance_summary.json")
        return 0 if result["status"] == "success" else 2
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        release_model_cache()


if __name__ == "__main__":
    raise SystemExit(main())
