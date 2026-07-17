from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from b3_tool_layer import execute_tool_calls, get_tools_schema
from b4_core.engine import DecisionEngine, release_model_cache
from b4_core.planning import PlanValidationError, generate_validated_plan, plan_layers
from b4_local_agent_llm import generate_ai_message
from common.io_utils import read_json, write_json, write_text
from common.logging_utils import now_iso
from common.path_utils import resolve_cli_path
from common.schemas import make_ai_message


def run_plan_execute(
    task_file: str,
    tools_config: str,
    model_config: str,
    outdir: str,
    *,
    toolset: str | None = None,
    planner_profile: str | None = None,
    synthesis_profile: str | None = None,
    execution_binding: str = "native_tools",
) -> dict:
    payload = read_json(task_file)
    if not isinstance(payload, dict) or not isinstance(payload.get("user_input"), str):
        raise ValueError("task file must contain a user_input string")
    selected_toolset = toolset or payload.get("toolset", "basic_tools")
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tools_schema = get_tools_schema(tools_config, selected_toolset, str(output_dir))
    engine = DecisionEngine(model_config)

    try:
        validated = generate_validated_plan(
            engine,
            payload["user_input"],
            tools_schema,
            profile=planner_profile,
            max_steps=int(payload.get("max_plan_steps", 6)),
        )
    except PlanValidationError as exc:
        write_json(exc.attempts, output_dir / "planner_attempts.json")
        report = {
            "status": "error",
            "source": "local_llm",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "planner_attempts_path": "planner_attempts.json",
        }
        write_json(report, output_dir / "plan_execute_report.json")
        return report

    write_json(validated.attempts, output_dir / "planner_attempts.json")
    write_json(validated.plan, output_dir / "validated_plan.json")
    system_prompt = payload.get("system_prompt") or engine.prompt("synthesizer")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": payload["user_input"]},
    ]
    all_tool_messages: list[dict] = []
    execution_trace: list[dict] = []
    for layer_index, layer in enumerate(plan_layers(validated.plan), 1):
        calls = [
            {
                "id": f"plan_{step['id']}",
                "name": step["tool_name"],
                "args": step["arguments"],
            }
            for step in layer
        ]
        ai_message = make_ai_message("", calls)
        messages.append(ai_message)
        round_dir = output_dir / f"tool_round_{layer_index:02d}"
        tool_messages = execute_tool_calls(calls, tools_config, selected_toolset, str(round_dir))
        messages.extend(tool_messages)
        all_tool_messages.extend(tool_messages)
        execution_trace.append(
            {
                "layer": layer_index,
                "step_ids": [step["id"] for step in layer],
                "ai_message": ai_message,
                "tool_messages": tool_messages,
            }
        )

    final = generate_ai_message(
        model_config,
        messages,
        [],
        execution_binding,
        str(output_dir / "synthesis"),
        "final",
        profile=synthesis_profile or planner_profile,
        binding=execution_binding,
        strategy="react",
    )
    final_message = final["ai_message"]
    messages.append(final_message)
    final_has_calls = bool(final_message.get("tool_calls"))
    tool_failures = [message for message in all_tool_messages if message.get("status") != "success"]
    if final["status"] != "success" or final_has_calls:
        status = "error"
    elif tool_failures:
        status = "partial"
    else:
        status = "success"

    report = {
        "status": status,
        "source": "local_llm",
        "task": payload["user_input"],
        "toolset": selected_toolset,
        "planner_profile": validated.attempts[-1]["metadata"]["profile"],
        "synthesis_profile": final.get("profile"),
        "execution_binding": execution_binding,
        "plan_step_count": len(validated.plan["steps"]),
        "tool_call_count": sum(len(item["ai_message"]["tool_calls"]) for item in execution_trace),
        "tool_message_count": len(all_tool_messages),
        "planner_corrected": len(validated.attempts) == 2,
        "planner_usage": [item["metadata"].get("usage") for item in validated.attempts],
        "synthesis_usage": final.get("usage"),
        "final_ai_message": final_message,
        "error": (
            {"type": "UnexpectedToolCall", "message": "synthesis requested another tool"}
            if final_has_calls
            else final.get("error")
        ),
        "generated_at": now_iso(),
    }
    write_json(execution_trace, output_dir / "plan_trace.json")
    write_json(all_tool_messages, output_dir / "tool_messages.json")
    write_json(messages, output_dir / "messages.json")
    write_json(final_message, output_dir / "final_ai_message.json")
    write_text(final_message.get("content", "").strip() + "\n", output_dir / "final_answer.md")
    write_json(report, output_dir / "plan_execute_report.json")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an LLM-generated and validated Plan-and-Execute task.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--tools_config", required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--toolset")
    parser.add_argument("--planner_profile")
    parser.add_argument("--synthesis_profile")
    parser.add_argument("--execution_binding", choices=["prompt_json", "native_tools"], default="native_tools")
    parser.add_argument("--outdir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_plan_execute(
            str(resolve_cli_path(args.task)),
            str(resolve_cli_path(args.tools_config)),
            str(resolve_cli_path(args.model_config)),
            str(resolve_cli_path(args.outdir)),
            toolset=args.toolset,
            planner_profile=args.planner_profile,
            synthesis_profile=args.synthesis_profile,
            execution_binding=args.execution_binding,
        )
        print(resolve_cli_path(args.outdir) / "plan_execute_report.json")
        return 0 if result["status"] in {"success", "partial"} else 2
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        release_model_cache()


if __name__ == "__main__":
    raise SystemExit(main())
