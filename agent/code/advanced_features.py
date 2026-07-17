from __future__ import annotations

"""Compatibility entry for the advanced acceptance suite.

Earlier revisions generated proxy/hash reports from this filename. The entry
now delegates only to the real implementations and never manufactures model
or vector results.
"""

import argparse
import sys
from pathlib import Path

from b3_acceptance import run_acceptance as run_b3_acceptance
from b4_acceptance import run_acceptance as run_b4_acceptance
from b4_core.engine import release_model_cache
from b5_acceptance import run_acceptance as run_b5_acceptance
from common.io_utils import write_json
from common.path_utils import resolve_cli_path
from verify_b2 import run_acceptance as run_b2_acceptance
from verify_b3_b5 import run_verification


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run truthful advanced checks; no proxy model evidence.")
    parser.add_argument("--tools_config", default="../configs/tools.yaml")
    parser.add_argument("--memory_config", default="../configs/memory.yaml")
    parser.add_argument("--model_config", default="../configs/model.yaml")
    parser.add_argument("--limits_config", default="../configs/skill_limits.yaml")
    parser.add_argument("--toolset", default="advanced_tools")
    parser.add_argument("--run_real_models", action="store_true")
    parser.add_argument("--outdir", default="../outputs/advanced_acceptance")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output_dir = resolve_cli_path(args.outdir)
        tools_config = resolve_cli_path(args.tools_config)
        memory_config = resolve_cli_path(args.memory_config)
        model_config = resolve_cli_path(args.model_config)
        limits_config = resolve_cli_path(args.limits_config)
        b2 = run_b2_acceptance(
            tools_config,
            limits_config,
            PROJECT_ROOT / "data",
            output_dir / "b2",
        )
        contracts = run_verification(PROJECT_ROOT)
        write_json(contracts, output_dir / "b3_b5_contracts.json")
        real: dict[str, object]
        if args.run_real_models:
            b3 = run_b3_acceptance(
                str(tools_config),
                str(PROJECT_ROOT / "data" / "b3_eval" / "schema_selection_cases.json"),
                str(output_dir / "b3_real"),
                toolset="basic_tools",
            )
            release_model_cache()
            b4 = run_b4_acceptance(
                str(PROJECT_ROOT / "data" / "b4_eval" / "smoke_cases.jsonl"),
                str(PROJECT_ROOT / "data" / "b4_eval" / "plan_read_calc.json"),
                str(model_config),
                str(tools_config),
                str(output_dir / "b4_real"),
            )
            release_model_cache()
            b5 = run_b5_acceptance(
                str(memory_config),
                str(PROJECT_ROOT / "data" / "b5_eval" / "memory_update_dry_run.json"),
                str(PROJECT_ROOT / "data" / "b5_eval" / "bad_memory.txt"),
                str(output_dir / "b5_real"),
            )
            real = {"status": "completed", "b3": b3, "b4": b4, "b5": b5}
            all_passed = all(item.get("status") == "success" for item in (b3, b4, b5))
            status = "success" if b2["status"] == contracts["status"] == "success" and all_passed else "error"
        else:
            real = {
                "status": "not_run",
                "reason": "Pass --run_real_models to generate real local-model evidence.",
            }
            status = "structural_checks_only" if b2["status"] == contracts["status"] == "success" else "error"
        report = {
            "status": status,
            "b2": b2,
            "b3_b5_contracts": contracts,
            "real_model_acceptance": real,
            "proxy_or_hash_fallback_used": False,
        }
        write_json(report, output_dir / "advanced_acceptance_summary.json")
        print(output_dir / "advanced_acceptance_summary.json")
        return 0 if status in {"success", "structural_checks_only"} else 2
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        release_model_cache()


if __name__ == "__main__":
    raise SystemExit(main())
