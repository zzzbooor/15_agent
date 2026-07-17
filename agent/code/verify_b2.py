from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

from common.io_utils import read_yaml, write_json, write_text
from common.path_utils import bootstrap_project_root, resolve_cli_path


PROJECT_ROOT = bootstrap_project_root()

from b2_run_skill import run_skill
from skills.core.catalog import SKILL_SPECS


BASIC_SKILLS = {
    "calculator",
    "file_reader",
    "local_file_search",
    "table_analyzer",
    "format_converter",
}
ADVANCED_SKILLS = {"read_and_convert", "analyze_and_convert", "code_executor"}


def _within(path: str, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(root.resolve())
        return True
    except (ValueError, TypeError):
        return False


def _case_path(outdir: Path, group: str, name: str) -> Path:
    return outdir / group / f"{name}.json"


def _run_case(
    matrix: list[dict[str, Any]],
    *,
    name: str,
    group: str,
    skill_name: str,
    payload: dict[str, Any],
    expected_status: str,
    data_root: Path,
    outdir: Path,
    limits_config: Path,
    expected_code: str | None = None,
    extra_check=None,
) -> dict[str, Any]:
    artifact_dir = outdir / "generated" / group / name
    result = run_skill(
        skill_name,
        payload,
        str(data_root),
        str(artifact_dir),
        str(limits_config),
    )
    passed = result.get("status") == expected_status
    reasons: list[str] = []
    if not passed:
        reasons.append(f"expected status={expected_status}, actual={result.get('status')}")
    if expected_code is not None:
        actual_code = (result.get("error") or {}).get("code")
        if actual_code != expected_code:
            passed = False
            reasons.append(f"expected error code={expected_code}, actual={actual_code}")
    if extra_check is not None:
        ok, message = extra_check(result, artifact_dir)
        if not ok:
            passed = False
            reasons.append(message)
    result_path = _case_path(outdir, group, name)
    write_json(result, result_path)
    matrix.append(
        {
            "name": name,
            "group": group,
            "skill_name": skill_name,
            "passed": passed,
            "reasons": reasons,
            "result_path": result_path.relative_to(outdir).as_posix(),
        }
    )
    return result


def _validate_catalog_and_config(matrix: list[dict], tools_config: Path, outdir: Path) -> None:
    config = read_yaml(tools_config)
    configured_basic = set(config.get("toolsets", {}).get("basic_tools", []))
    configured_advanced = set(config.get("toolsets", {}).get("advanced_tools", []))
    catalog_names = set(SKILL_SPECS)
    reasons: list[str] = []
    if configured_basic != BASIC_SKILLS:
        reasons.append(f"basic_tools mismatch: {sorted(configured_basic)}")
    if configured_advanced != BASIC_SKILLS | ADVANCED_SKILLS:
        reasons.append(f"advanced_tools mismatch: {sorted(configured_advanced)}")
    if catalog_names != BASIC_SKILLS | ADVANCED_SKILLS:
        reasons.append(f"catalog mismatch: {sorted(catalog_names)}")
    for name, spec in SKILL_SPECS.items():
        try:
            function = getattr(importlib.import_module(spec.module), spec.function)
        except (ImportError, AttributeError) as exc:
            reasons.append(f"cannot load {name}: {exc}")
        else:
            if not callable(function):
                reasons.append(f"catalog entry is not callable: {name}")
    result = {
        "configured_basic": sorted(configured_basic),
        "configured_advanced": sorted(configured_advanced),
        "catalog_names": sorted(catalog_names),
        "reasons": reasons,
    }
    path = outdir / "contract" / "catalog_and_config.json"
    write_json(result, path)
    matrix.append(
        {
            "name": "catalog_and_tools_config",
            "group": "contract",
            "skill_name": None,
            "passed": not reasons,
            "reasons": reasons,
            "result_path": path.relative_to(outdir).as_posix(),
        }
    )


def run_acceptance(tools_config: Path, limits_config: Path, data_root: Path, outdir: Path) -> dict:
    matrix: list[dict[str, Any]] = []
    _validate_catalog_and_config(matrix, tools_config, outdir)

    normal_cases = {
        "calculator": {"expression": "23 * 17 + 9"},
        "file_reader": {"path": "docs/agent_intro.txt", "max_chars": 2000},
        "local_file_search": {
            "query": "工具编排",
            "root_dir": "docs",
            "file_types": ["txt", "md"],
            "top_k": 3,
        },
        "table_analyzer": {"path": "tables/results.csv", "max_rows_preview": 5, "describe": True},
        "format_converter": {
            "text": "a: 1\nb: 2",
            "target_format": "markdown",
            "output_filename": "basic_conversion.md",
        },
    }
    error_cases = {
        "calculator": ({"expression": "23 / 0"}, "EXECUTION_ERROR"),
        "file_reader": ({"path": "docs/missing.txt"}, "FILE_NOT_FOUND"),
        "local_file_search": ({"query": "Agent", "root_dir": "missing"}, "FILE_NOT_FOUND"),
        "table_analyzer": ({"path": "tables/missing.csv"}, "FILE_NOT_FOUND"),
        "format_converter": ({"text": "a: 1", "target_format": "xml"}, "UNSUPPORTED_OPERATION"),
    }
    for skill_name, payload in normal_cases.items():
        check = None
        if skill_name == "local_file_search":
            check = lambda result, _: (
                bool(result.get("output", {}).get("results"))
                and result["output"]["results"][0]["path"] == "docs/search_skill_demo.md",
                "Chinese query did not rank docs/search_skill_demo.md first",
            )
        elif skill_name == "format_converter":
            check = lambda result, artifact_dir: (
                _within(result.get("output", {}).get("generated_file_path", ""), artifact_dir),
                "generated file escaped the case output directory",
            )
        _run_case(
            matrix,
            name=f"{skill_name}_normal",
            group="basic",
            skill_name=skill_name,
            payload=payload,
            expected_status="success",
            data_root=data_root,
            outdir=outdir,
            limits_config=limits_config,
            extra_check=check,
        )
    for skill_name, (payload, expected_code) in error_cases.items():
        _run_case(
            matrix,
            name=f"{skill_name}_error",
            group="basic",
            skill_name=skill_name,
            payload=payload,
            expected_status="error",
            expected_code=expected_code,
            data_root=data_root,
            outdir=outdir,
            limits_config=limits_config,
        )

    _run_case(
        matrix,
        name="missing_required_parameter",
        group="contract",
        skill_name="calculator",
        payload={},
        expected_status="error",
        expected_code="PARAM_MISSING",
        data_root=data_root,
        outdir=outdir,
        limits_config=limits_config,
    )
    _run_case(
        matrix,
        name="search_query_budget",
        group="limits",
        skill_name="local_file_search",
        payload={"query": "x" * 501, "root_dir": "docs"},
        expected_status="error",
        expected_code="RESOURCE_EXHAUSTED",
        data_root=data_root,
        outdir=outdir,
        limits_config=limits_config,
    )
    tiny_limits = outdir / "policies" / "search_tiny_limits.json"
    write_json({"search": {"max_index_tokens": 5}}, tiny_limits)
    _run_case(
        matrix,
        name="search_index_token_budget",
        group="limits",
        skill_name="local_file_search",
        payload={"query": "工具", "root_dir": "docs"},
        expected_status="success",
        data_root=data_root,
        outdir=outdir,
        limits_config=tiny_limits,
        extra_check=lambda result, _: (
            result.get("output", {}).get("limit_reached") == "max_index_tokens"
            and result.get("output", {}).get("indexed_tokens", 0) <= 5,
            "search did not stop at the configured index-token budget",
        ),
    )

    def composite_output_check(result: dict, artifact_dir: Path) -> tuple[bool, str]:
        output = result.get("output") or {}
        traces = output.get("step_trace") or []
        passed = (
            _within(output.get("generated_file_path", ""), artifact_dir)
            and traces
            and all(item.get("status") == "success" for item in traces)
        )
        return passed, "composite output/step trace is invalid or escaped its output directory"

    _run_case(
        matrix,
        name="read_and_convert_success",
        group="advanced",
        skill_name="read_and_convert",
        payload={"path": "docs/agent_intro.txt", "target_format": "markdown"},
        expected_status="success",
        data_root=data_root,
        outdir=outdir,
        limits_config=limits_config,
        extra_check=composite_output_check,
    )
    _run_case(
        matrix,
        name="analyze_and_convert_success",
        group="advanced",
        skill_name="analyze_and_convert",
        payload={"path": "tables/results.csv", "target_format": "json"},
        expected_status="success",
        data_root=data_root,
        outdir=outdir,
        limits_config=limits_config,
        extra_check=composite_output_check,
    )
    _run_case(
        matrix,
        name="composite_cause_error",
        group="advanced",
        skill_name="read_and_convert",
        payload={"path": "docs/missing.txt", "target_format": "markdown"},
        expected_status="error",
        expected_code="COMPOSITE_STEP_FAILED",
        data_root=data_root,
        outdir=outdir,
        limits_config=limits_config,
        extra_check=lambda result, _: (
            (result.get("error") or {}).get("details", {}).get("cause_code") == "FILE_NOT_FOUND",
            "composite failure did not preserve FILE_NOT_FOUND as the cause",
        ),
    )
    _run_case(
        matrix,
        name="restricted_math_success",
        group="security",
        skill_name="code_executor",
        payload={"code": "import math\nprint(math.sqrt(16))", "timeout": 3},
        expected_status="success",
        data_root=data_root,
        outdir=outdir,
        limits_config=limits_config,
        extra_check=lambda result, _: (
            result.get("output", {}).get("stdout") == "4.0\n"
            and result.get("output", {}).get("isolation", {}).get("user_source_executed_directly") is False,
            "restricted interpreter did not produce the expected safe result",
        ),
    )
    attacks = {
        "forbidden_import": "import os\nprint(os.getcwd())",
        "builtins_escape": "__builtins__['open']('forbidden.txt', 'w')",
        "getattr_escape": "getattr(__builtins__, '__import__')('os')",
        "dunder_escape": "print((1).__class__.__mro__)",
        "infinite_loop": "while True:\n    pass",
    }
    for name, source in attacks.items():
        _run_case(
            matrix,
            name=name,
            group="security",
            skill_name="code_executor",
            payload={"code": source, "timeout": 1},
            expected_status="error",
            expected_code="SANDBOX_VIOLATION",
            data_root=data_root,
            outdir=outdir,
            limits_config=limits_config,
        )
    _run_case(
        matrix,
        name="loop_iteration_budget",
        group="limits",
        skill_name="code_executor",
        payload={"code": "for i in range(10001):\n    pass", "timeout": 3},
        expected_status="error",
        expected_code="RESOURCE_EXHAUSTED",
        data_root=data_root,
        outdir=outdir,
        limits_config=limits_config,
    )

    passed_count = sum(1 for item in matrix if item["passed"])
    summary = {
        "status": "success" if passed_count == len(matrix) else "error",
        "passed": passed_count,
        "failed": len(matrix) - passed_count,
        "total": len(matrix),
        "matrix": matrix,
        "tools_config": str(tools_config),
        "limits_config": str(limits_config),
        "data_root": str(data_root),
    }
    write_json(summary, outdir / "acceptance_matrix.json")
    lines = [
        "# B2 Acceptance Report",
        "",
        f"- Status: {summary['status']}",
        f"- Passed: {summary['passed']}/{summary['total']}",
        f"- Failed: {summary['failed']}",
        "",
        "| Group | Case | Skill | Result |",
        "|---|---|---|---|",
    ]
    for item in matrix:
        outcome = "PASS" if item["passed"] else "FAIL: " + "; ".join(item["reasons"])
        lines.append(f"| {item['group']} | {item['name']} | {item['skill_name'] or '-'} | {outcome} |")
    write_text("\n".join(lines) + "\n", outdir / "acceptance_report.md")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the deterministic B2 acceptance matrix.")
    parser.add_argument("--tools-config", default=str(PROJECT_ROOT / "configs" / "tools.yaml"))
    parser.add_argument("--limits-config", default=str(PROJECT_ROOT / "configs" / "skill_limits.yaml"))
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--outdir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_acceptance(
        resolve_cli_path(args.tools_config),
        resolve_cli_path(args.limits_config),
        resolve_cli_path(args.data_root),
        resolve_cli_path(args.outdir),
    )
    print(resolve_cli_path(args.outdir) / "acceptance_report.md")
    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
