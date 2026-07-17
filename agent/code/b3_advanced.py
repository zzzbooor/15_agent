from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from b3_tool_layer import (
    _load_tools_config,
    _resolve_toolset,
    execute_tool_calls,
    get_auto_tools_schema,
    get_tools_schema,
)
from common.io_utils import read_json, write_json
from common.path_utils import resolve_cli_path
from common.schemas import make_tool_message, normalize_tool_call


ToolExecutor = Callable[[list[dict[str, Any]], str, str | None, str | None], list[dict[str, Any]]]
ToolSelector = Callable[[str, list[dict[str, Any]]], str | dict[str, Any]]


def _read_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "entries": {}}
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError("B3 cache must contain an object")
    if value.get("version") != 1 or not isinstance(value.get("entries"), dict):
        raise ValueError("B3 cache must use version=1 and contain entries")
    return value


def _cache_key(config_path: Path, toolset: str, call: dict[str, Any]) -> str:
    payload = {
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "toolset": toolset,
        "name": call["name"],
        "args": call["args"],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decode_skill_result(message: dict[str, Any]) -> dict[str, Any]:
    try:
        result = json.loads(message["content"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("ToolMessage content must contain a serialized SkillResult") from exc
    if not isinstance(result, dict) or result.get("status") not in {"success", "error"}:
        raise ValueError("ToolMessage content contains an invalid SkillResult")
    return result


def _is_retryable(result: dict[str, Any]) -> bool:
    error = result.get("error")
    return bool(isinstance(error, dict) and error.get("retryable") is True)


def _is_cacheable(config: dict[str, Any], name: str) -> bool:
    definition = config.get("tools", {}).get(name)
    return bool(isinstance(definition, dict) and definition.get("cacheable", False))


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 3)
    interpolated = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return round(interpolated, 3)


def execute_with_retry_cache_stats(
    tool_calls: list[dict[str, Any]],
    tools_config: str | Path,
    toolset: str | None,
    outdir: str | Path,
    *,
    max_retries: int = 1,
    cache_path: str | Path | None = None,
    executor: ToolExecutor = execute_tool_calls,
) -> dict[str, Any]:
    """Execute logical calls with bounded retries and a success-only cache."""

    if not isinstance(tool_calls, list):
        raise ValueError("tool_calls must be a list")
    if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")
    config_path, config = _load_tools_config(tools_config)
    selected, _ = _resolve_toolset(config, toolset)
    output_dir = Path(outdir).resolve()
    selected_cache_path = Path(cache_path).resolve() if cache_path else output_dir / "b3_success_cache.json"
    cache = _read_cache(selected_cache_path)
    entries = {
        key: value
        for key, value in cache["entries"].items()
        if isinstance(value, dict) and value.get("status") == "success"
    }
    cache["entries"] = entries
    final_messages: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    execution_latencies: list[float] = []
    for index, raw_call in enumerate(tool_calls):
        logical_started = perf_counter()
        try:
            call = normalize_tool_call(raw_call, index)
        except Exception:
            call = {"id": f"call_{index + 1:03d}", "name": "unknown", "args": {}}
            normalized = False
        else:
            normalized = True
        cacheable = normalized and _is_cacheable(config, call["name"])
        key = _cache_key(config_path, selected, call) if normalized else None
        cached = entries.get(key) if key else None
        if cacheable and isinstance(cached, dict) and cached.get("status") == "success":
            result = copy.deepcopy(cached)
            message = make_tool_message(
                call["id"], call["name"], json.dumps(result, ensure_ascii=False, separators=(",", ":")), "success"
            )
            final_messages.append(message)
            records.append(
                {
                    "logical_index": index + 1,
                    "tool_call_id": call["id"],
                    "name": call["name"],
                    "status": "success",
                    "attempts": 0,
                    "retries": 0,
                    "cache_hit": True,
                    "cacheable": True,
                    "execution_latency_ms": 0.0,
                    "logical_latency_ms": round((perf_counter() - logical_started) * 1000, 3),
                    "error_code": None,
                }
            )
            continue
        attempts = 0
        execution_latency = 0.0
        message: dict[str, Any] | None = None
        result: dict[str, Any] | None = None
        while attempts <= max_retries:
            attempts += 1
            attempt_started = perf_counter()
            messages = executor([raw_call], str(config_path), selected, str(output_dir))
            duration = round((perf_counter() - attempt_started) * 1000, 3)
            execution_latency += duration
            execution_latencies.append(duration)
            if not isinstance(messages, list) or len(messages) != 1:
                raise ValueError("B3 executor must return exactly one ToolMessage for one tool call")
            message = messages[0]
            result = _decode_skill_result(message)
            if result["status"] == "success" or not _is_retryable(result) or attempts > max_retries:
                break
        assert message is not None and result is not None
        final_messages.append(message)
        if cacheable and result["status"] == "success" and key:
            entries[key] = copy.deepcopy(result)
        error = result.get("error") or {}
        records.append(
            {
                "logical_index": index + 1,
                "tool_call_id": call["id"],
                "name": call["name"],
                "status": result["status"],
                "attempts": attempts,
                "retries": max(0, attempts - 1),
                "cache_hit": False,
                "cacheable": cacheable,
                "execution_latency_ms": round(execution_latency, 3),
                "logical_latency_ms": round((perf_counter() - logical_started) * 1000, 3),
                "error_code": error.get("code") if isinstance(error, dict) else None,
            }
        )
    success_count = sum(item["status"] == "success" for item in records)
    failure_count = len(records) - success_count
    error_codes = Counter(item["error_code"] for item in records if item["error_code"])
    report = {
        "status": "success" if failure_count == 0 else "partial",
        "logical_call_count": len(records),
        "success_count": success_count,
        "failure_count": failure_count,
        "success_rate": round(success_count / len(records), 6) if records else 0.0,
        "cache_hit_count": sum(item["cache_hit"] for item in records),
        "cache_miss_count": sum(not item["cache_hit"] for item in records),
        "execution_attempt_count": sum(item["attempts"] for item in records),
        "retry_count": sum(item["retries"] for item in records),
        "execution_latency_ms": {
            "total": round(sum(execution_latencies), 3),
            "mean": round(sum(execution_latencies) / len(execution_latencies), 3)
            if execution_latencies
            else 0.0,
            "p50": _percentile(execution_latencies, 0.50),
            "p95": _percentile(execution_latencies, 0.95),
        },
        "error_code_counts": dict(sorted(error_codes.items())),
        "cache_policy": "success_only",
        "records": records,
    }
    write_json(final_messages, output_dir / "b3_advanced_tool_messages.json")
    write_json(cache, selected_cache_path)
    write_json(report, output_dir / "b3_tool_call_stats.json")
    return report


def _description_variants(tools_schema: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    full = copy.deepcopy(tools_schema)
    name_only = copy.deepcopy(tools_schema)
    for item in name_only:
        function = item.get("function", {})
        name = function.get("name", "")
        function["description"] = name.replace("_", " ")
        properties = function.get("parameters", {}).get("properties", {})
        for parameter_name, schema in properties.items():
            if isinstance(schema, dict):
                schema["description"] = parameter_name.replace("_", " ")
    return {"full_description": full, "name_only": name_only}


def _selector_identity(selector: ToolSelector) -> str:
    return f"{getattr(selector, '__module__', 'unknown')}:{getattr(selector, '__qualname__', type(selector).__name__)}"


def compare_schema_descriptions(
    tools_schema: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    outdir: str | Path,
    *,
    selector: ToolSelector | None,
) -> dict[str, Any]:
    """Evaluate descriptions through an injected selector; never fabricate model choices."""

    output_dir = Path(outdir).resolve()
    if not isinstance(cases, list) or not cases:
        raise ValueError("schema comparison requires a non-empty case list")
    for index, case in enumerate(cases, 1):
        if not isinstance(case, dict) or not isinstance(case.get("task"), str):
            raise ValueError(f"comparison case {index} must contain a task string")
        if not isinstance(case.get("expected_tool"), str):
            raise ValueError(f"comparison case {index} must contain expected_tool")
    if selector is None:
        report = {
            "status": "not_run",
            "reason": "No selector was injected. A real model-backed selector is required for this comparison.",
            "case_count": len(cases),
            "variants": {},
            "cases": [],
        }
        write_json(report, output_dir / "b3_schema_description_comparison.json")
        return report
    rows: list[dict[str, Any]] = []
    variants = _description_variants(tools_schema)
    for variant_name, schema in variants.items():
        for case_index, case in enumerate(cases, 1):
            started = perf_counter()
            try:
                selection = selector(case["task"], copy.deepcopy(schema))
                if isinstance(selection, str):
                    predicted = selection
                    metadata: dict[str, Any] = {}
                elif isinstance(selection, dict) and isinstance(selection.get("selected_tool"), str):
                    predicted = selection["selected_tool"]
                    metadata = {key: value for key, value in selection.items() if key != "selected_tool"}
                else:
                    raise ValueError("selector must return a tool name or {selected_tool: ...}")
                error = None
            except Exception as exc:
                predicted = None
                metadata = {}
                error = {"type": type(exc).__name__, "message": str(exc)}
            rows.append(
                {
                    "variant": variant_name,
                    "case_index": case_index,
                    "task": case["task"],
                    "expected_tool": case["expected_tool"],
                    "predicted_tool": predicted,
                    "correct": predicted == case["expected_tool"],
                    "latency_ms": round((perf_counter() - started) * 1000, 3),
                    "selector_metadata": metadata,
                    "error": error,
                }
            )
    summaries: dict[str, Any] = {}
    for name in variants:
        selected_rows = [row for row in rows if row["variant"] == name]
        completed = [row for row in selected_rows if row["error"] is None]
        correct = sum(row["correct"] for row in selected_rows)
        summaries[name] = {
            "case_count": len(selected_rows),
            "completed_count": len(completed),
            "correct_count": correct,
            "accuracy": round(correct / len(selected_rows), 6) if selected_rows else 0.0,
            "mean_latency_ms": round(
                sum(row["latency_ms"] for row in selected_rows) / len(selected_rows), 3
            )
            if selected_rows
            else 0.0,
        }
    report = {
        "status": "success" if all(row["error"] is None for row in rows) else "partial",
        "selector": _selector_identity(selector),
        "metric": "observed selector accuracy",
        "variants": summaries,
        "cases": rows,
    }
    write_json(report, output_dir / "b3_schema_description_comparison.json")
    return report


def _load_callable(spec: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise ValueError("callable must use module:function syntax")
    module_name, attribute = spec.split(":", 1)
    module = importlib.import_module(module_name)
    value = getattr(module, attribute)
    if not callable(value):
        raise TypeError(f"configured object is not callable: {spec}")
    return value


def _read_cases(path: str | Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list):
        raise ValueError("cases file must be a list or an object containing cases")
    return cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run honest B3 advanced evaluations.")
    parser.add_argument("--tools_config", required=True)
    parser.add_argument("--toolset", default=None)
    parser.add_argument("--outdir", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("auto-schema")
    execute = subparsers.add_parser("retry-cache")
    execute.add_argument("--tool_calls", required=True)
    execute.add_argument("--cache_path")
    execute.add_argument("--max_retries", type=int, default=1)
    compare = subparsers.add_parser("compare-descriptions")
    compare.add_argument("--cases", required=True)
    compare.add_argument("--selector", help="Real selector as module:function; omission records not_run.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config_path = resolve_cli_path(args.tools_config)
        outdir = resolve_cli_path(args.outdir)
        _, config = _load_tools_config(config_path)
        toolset = args.toolset or config.get("default_toolset")
        if args.command == "auto-schema":
            get_auto_tools_schema(str(config_path), toolset, str(outdir))
            result_path = outdir / "b3_auto_schema_from_python.json"
        elif args.command == "retry-cache":
            payload = read_json(resolve_cli_path(args.tool_calls))
            calls = payload.get("tool_calls") if isinstance(payload, dict) else payload
            execute_with_retry_cache_stats(
                calls,
                config_path,
                toolset,
                outdir,
                max_retries=args.max_retries,
                cache_path=resolve_cli_path(args.cache_path) if args.cache_path else None,
            )
            result_path = outdir / "b3_tool_call_stats.json"
        else:
            schema = get_tools_schema(str(config_path), toolset)
            selector = _load_callable(args.selector) if args.selector else None
            compare_schema_descriptions(
                schema,
                _read_cases(resolve_cli_path(args.cases)),
                outdir,
                selector=selector,
            )
            result_path = outdir / "b3_schema_description_comparison.json"
        print(result_path)
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
