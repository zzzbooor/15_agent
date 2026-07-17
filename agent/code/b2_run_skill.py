from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common.io_utils import append_jsonl, read_json, write_json
from common.logging_utils import now_iso
from common.path_utils import bootstrap_project_root, resolve_cli_path


PROJECT_ROOT = bootstrap_project_root()

from skills.core.catalog import SKILL_SPECS
from skills.core.invoker import invoke_skill


# Retained for callers that previously inspected this public mapping.
SKILL_MODULES = {name: spec.module for name, spec in SKILL_SPECS.items()}


def run_skill(
    skill_name: str,
    input_data: dict,
    data_root: str | None = None,
    output_dir: str | None = None,
    limits_config: str | None = None,
) -> dict:
    """Run one catalogued Skill and return a complete SkillResult."""

    return invoke_skill(
        skill_name,
        input_data,
        data_root=data_root,
        output_dir=output_dir,
        limits_config=limits_config,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one local Agent skill.")
    parser.add_argument("--skill", required=True, choices=sorted(SKILL_MODULES))
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument(
        "--limits_config",
        default=None,
        help="Optional central B2 resource policy file; defaults to configs/skill_limits.yaml.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        input_path = resolve_cli_path(args.input)
        outdir = resolve_cli_path(args.outdir)
        input_data = read_json(input_path)
        data_root = str(resolve_cli_path(args.data_root)) if args.data_root else None
        limits_config = str(resolve_cli_path(args.limits_config)) if args.limits_config else None
        outdir.mkdir(parents=True, exist_ok=True)
        result = run_skill(args.skill, input_data, data_root, str(outdir), limits_config)
        result_path = outdir / f"{args.skill}_result.json"
        write_json(result, result_path)
        append_jsonl(
            {
                "timestamp": now_iso(),
                "skill_name": args.skill,
                "status": result["status"],
                "error_code": (result.get("error") or {}).get("code"),
                "result_path": str(result_path),
                "latency_ms": result["latency_ms"],
            },
            outdir / "skill_run_log.jsonl",
        )
        print(result_path)
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
