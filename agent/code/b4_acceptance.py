from __future__ import annotations

import argparse
import sys
from pathlib import Path

from b4_benchmark import run_benchmark
from b4_core.engine import release_model_cache
from b4_plan_execute import run_plan_execute
from common.io_utils import write_json
from common.path_utils import resolve_cli_path


def run_acceptance(
    cases: str,
    task: str,
    model_config: str,
    tools_config: str,
    outdir: str,
) -> dict:
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = run_benchmark(
        cases,
        model_config,
        tools_config,
        str(output_dir / "two_by_two"),
        profiles=["qwen35_4b", "qwen3_1_7b"],
        bindings=["prompt_json", "native_tools"],
        toolset="basic_tools",
        execute_tools=True,
    )
    plan = run_plan_execute(
        task,
        tools_config,
        model_config,
        str(output_dir / "plan_execute"),
        toolset="basic_tools",
        planner_profile="qwen35_4b",
        synthesis_profile="qwen35_4b",
        execution_binding="native_tools",
    )
    profile_counts = benchmark.get("profile_success_counts") or {}
    binding_counts = benchmark.get("binding_success_counts") or {}
    gates = {
        "complete_two_by_two_coverage": benchmark.get("coverage_complete") is True,
        "every_model_has_successful_tool_calling": all(
            profile_counts.get(name, 0) >= 1 for name in ("qwen35_4b", "qwen3_1_7b")
        ),
        "every_binding_has_successful_tool_calling": all(
            binding_counts.get(name, 0) >= 1 for name in ("prompt_json", "native_tools")
        ),
        "multiple_tool_calls_and_messages": benchmark.get("multi_tool_success_count", 0) >= 1,
        "real_token_usage_recorded": benchmark.get("usage_evidence_count", 0)
        == benchmark.get("run_count", -1),
        "plan_and_execute": plan.get("status") == "success",
    }
    summary = {
        "status": "success" if all(gates.values()) else "completed_with_failures",
        "capability_gates": gates,
        "benchmark_summary": benchmark,
        "individual_benchmark_failures": benchmark.get("failed_run_count"),
        "all_individual_runs_successful": benchmark.get("all_cases_successful"),
        "plan_execute_status": plan.get("status"),
        "plan_execute_report": "plan_execute/plan_execute_report.json",
        "benchmark_report": "two_by_two/benchmark_summary.json",
        "note": (
            "Acceptance uses capability-coverage gates. Individual model/binding failures remain visible "
            "as comparison evidence and are never converted into successful rows. No deterministic fallback is used."
        ),
    }
    write_json(summary, output_dir / "b4_acceptance_summary.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the lightweight real-model B4 acceptance suite.")
    parser.add_argument("--cases", default="../data/b4_eval/smoke_cases.jsonl")
    parser.add_argument("--task", default="../data/b4_eval/plan_read_calc.json")
    parser.add_argument("--model_config", default="../configs/model.yaml")
    parser.add_argument("--tools_config", default="../configs/tools.yaml")
    parser.add_argument("--outdir", default="../outputs/B4_acceptance")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_acceptance(
            str(resolve_cli_path(args.cases)),
            str(resolve_cli_path(args.task)),
            str(resolve_cli_path(args.model_config)),
            str(resolve_cli_path(args.tools_config)),
            str(resolve_cli_path(args.outdir)),
        )
        print(resolve_cli_path(args.outdir) / "b4_acceptance_summary.json")
        return 0 if result["status"] == "success" else 2
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        release_model_cache()


if __name__ == "__main__":
    raise SystemExit(main())
