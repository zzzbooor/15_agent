from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from b3_tool_layer import execute_tool_calls, get_tools_schema
from b4_core.engine import release_model_cache
from b4_local_agent_llm import generate_ai_message
from common.io_utils import append_jsonl, read_text, write_json, write_text
from common.path_utils import resolve_cli_path


def load_cases(path: str | Path) -> list[dict]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in read_text(source).splitlines() if line.strip()]
    else:
        rows = json.loads(read_text(source))
    if not isinstance(rows, list) or not rows:
        raise ValueError("benchmark cases must be a non-empty JSON array or JSONL file")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not isinstance(row.get("user_input"), str):
            raise ValueError(f"invalid benchmark case at index {index}")
        if not isinstance(row.get("expected_tool_calls", []), list):
            raise ValueError(f"case {row['id']} expected_tool_calls must be an array")
    return rows


def expected_calls_match(actual: list[dict], expected: list[dict]) -> tuple[bool, list[str]]:
    def value_matches(field: str, actual_value: Any, expected_value: Any) -> bool:
        if field == "expression" and isinstance(actual_value, str) and isinstance(expected_value, str):
            return "".join(actual_value.split()) == "".join(expected_value.split())
        if field == "query" and isinstance(actual_value, str) and isinstance(expected_value, str):
            return expected_value.casefold() in actual_value.casefold()
        return actual_value == expected_value

    errors: list[str] = []
    actual_names = Counter(call.get("name") for call in actual)
    expected_names = Counter(call.get("name") for call in expected)
    if actual_names != expected_names:
        errors.append(f"tool multiset mismatch: expected {dict(expected_names)}, got {dict(actual_names)}")
    unmatched = list(actual)
    for expected_call in expected:
        name = expected_call.get("name")
        required_args = expected_call.get("args_contains") or {}
        match_index = next(
            (
                index
                for index, call in enumerate(unmatched)
                if call.get("name") == name
                and all(value_matches(key, call.get("args", {}).get(key), value) for key, value in required_args.items())
            ),
            None,
        )
        if match_index is None:
            errors.append(f"no {name} call contains expected arguments {required_args}")
        else:
            unmatched.pop(match_index)
    return not errors, errors


def _summary(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(f"{row['profile']}::{row['binding']}", []).append(row)
    result: dict[str, Any] = {}
    for key, items in groups.items():
        token_values = [item["usage"]["total_tokens"] for item in items if isinstance(item.get("usage"), dict)]
        latency_values = [
            item["metadata"].get("inference_latency_ms")
            for item in items
            if isinstance(item.get("metadata"), dict)
            and isinstance(item["metadata"].get("inference_latency_ms"), (int, float))
        ]
        result[key] = {
            "case_count": len(items),
            "parse_success_rate": sum(item["parse_success"] for item in items) / len(items),
            "schema_valid_rate": sum(item["schema_valid"] for item in items) / len(items),
            "expected_call_match_rate": sum(item["expected_call_match"] for item in items) / len(items),
            "tool_execution_success_rate": sum(item["tool_execution_success"] for item in items) / len(items),
            "overall_success_rate": sum(item["success"] for item in items) / len(items),
            "mean_total_tokens": round(statistics.mean(token_values), 3) if token_values else None,
            "median_total_tokens": round(statistics.median(token_values), 3) if token_values else None,
            "mean_inference_latency_ms": round(statistics.mean(latency_values), 3) if latency_values else None,
        }
    return result


def run_benchmark(
    cases_path: str,
    model_config: str,
    tools_config: str,
    outdir: str,
    *,
    profiles: list[str],
    bindings: list[str],
    toolset: str = "basic_tools",
    execute_tools: bool = True,
) -> dict:
    cases = load_cases(cases_path)
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "benchmark_runs.jsonl"
    write_text("", jsonl_path)
    tools_schema = get_tools_schema(tools_config, toolset, str(output_dir))
    rows: list[dict] = []

    for profile in profiles:
        try:
            for binding in bindings:
                for case in cases:
                    run_dir = output_dir / profile / binding / case["id"]
                    messages = [
                        {"role": "system", "content": case.get("system_prompt", "Use tools when required.")},
                        {"role": "user", "content": case["user_input"]},
                    ]
                    try:
                        generated = generate_ai_message(
                            model_config,
                            messages,
                            tools_schema,
                            binding,
                            str(run_dir),
                            "decision",
                            profile=profile,
                            binding=binding,
                        )
                        ai_message = generated["ai_message"]
                        calls = ai_message.get("tool_calls", [])
                        expected_match, match_errors = expected_calls_match(
                            calls,
                            case.get("expected_tool_calls", []),
                        )
                        parse_success = generated["status"] == "success"
                        schema_valid = bool(generated.get("tool_call_validation", {}).get("valid"))
                        if execute_tools and calls:
                            tool_messages = execute_tool_calls(calls, tools_config, toolset, str(run_dir / "tools"))
                            execution_success = all(message.get("status") == "success" for message in tool_messages)
                        elif calls:
                            tool_messages = []
                            execution_success = True
                        else:
                            tool_messages = []
                            execution_success = not case.get("expected_tool_calls", [])
                        row = {
                            "case_id": case["id"],
                            "profile": generated.get("profile") or profile,
                            "binding": generated.get("binding") or binding,
                            "parse_success": parse_success,
                            "schema_valid": schema_valid,
                            "expected_call_match": expected_match,
                            "tool_execution_success": execution_success,
                            "success": parse_success and schema_valid and expected_match and execution_success,
                            "actual_tool_calls": calls,
                            "expected_tool_calls": case.get("expected_tool_calls", []),
                            "match_errors": match_errors,
                            "tool_messages": tool_messages,
                            "usage": generated.get("usage"),
                            "metadata": generated.get("metadata"),
                            "error": generated.get("error"),
                        }
                    except Exception as exc:
                        row = {
                            "case_id": case["id"],
                            "profile": profile,
                            "binding": binding,
                            "parse_success": False,
                            "schema_valid": False,
                            "expected_call_match": False,
                            "tool_execution_success": False,
                            "success": False,
                            "actual_tool_calls": [],
                            "expected_tool_calls": case.get("expected_tool_calls", []),
                            "match_errors": [],
                            "tool_messages": [],
                            "usage": None,
                            "metadata": None,
                            "error": {"type": type(exc).__name__, "message": str(exc)},
                        }
                    rows.append(row)
                    append_jsonl(row, jsonl_path)
        finally:
            release_model_cache()

    successful_rows = [row for row in rows if row["success"]]
    expected_run_count = len(profiles) * len(bindings) * len(cases)
    profile_success_counts = {
        profile: sum(row["success"] for row in rows if row["profile"] == profile)
        for profile in profiles
    }
    binding_success_counts = {
        binding: sum(row["success"] for row in rows if row["binding"] == binding)
        for binding in bindings
    }
    summary = {
        # Model failures are measured data, not a benchmark-runner failure.
        "status": "completed",
        "all_cases_successful": bool(rows) and all(row["success"] for row in rows),
        "matrix": {"profiles": profiles, "bindings": bindings, "cases": len(cases)},
        "run_count": len(rows),
        "expected_run_count": expected_run_count,
        "coverage_complete": len(rows) == expected_run_count,
        "successful_run_count": len(successful_rows),
        "failed_run_count": len(rows) - len(successful_rows),
        "profile_success_counts": profile_success_counts,
        "binding_success_counts": binding_success_counts,
        "usage_evidence_count": sum(isinstance(row.get("usage"), dict) for row in rows),
        "multi_tool_success_count": sum(
            row["success"] and len(row.get("actual_tool_calls", [])) >= 2 for row in rows
        ),
        "groups": _summary(rows),
    }
    write_json(rows, output_dir / "benchmark_runs.json")
    write_json(summary, output_dir / "benchmark_summary.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a real 2-model by 2-binding B4 benchmark.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--tools_config", required=True)
    parser.add_argument("--profiles", nargs="+", default=["qwen35_4b", "qwen3_1_7b"])
    parser.add_argument("--bindings", nargs="+", choices=["prompt_json", "native_tools"], default=["prompt_json", "native_tools"])
    parser.add_argument("--toolset", default="basic_tools")
    parser.add_argument("--no_execute", action="store_true")
    parser.add_argument("--outdir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_benchmark(
            str(resolve_cli_path(args.cases)),
            str(resolve_cli_path(args.model_config)),
            str(resolve_cli_path(args.tools_config)),
            str(resolve_cli_path(args.outdir)),
            profiles=args.profiles,
            bindings=args.bindings,
            toolset=args.toolset,
            execute_tools=not args.no_execute,
        )
        print(resolve_cli_path(args.outdir) / "benchmark_summary.json")
        return 0 if summary["status"] == "completed" else 2
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        release_model_cache()


if __name__ == "__main__":
    raise SystemExit(main())
