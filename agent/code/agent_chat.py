from __future__ import annotations

import argparse
import os
from contextlib import redirect_stdout
from pathlib import Path

from b1_agent_runtime import run_agent
from common.io_utils import read_json, write_json
from common.path_utils import resolve_cli_path
from trace_report import generate_trace_html


DEMO_PROMPTS = [
    "请使用计算器精确计算 23 * 17 + 9。",
    "请读取 docs/agent_intro.txt，并总结三条要点。",
    "请在 docs 目录搜索与 tool calling 有关的本地资料，并总结最相关内容。",
    "请分析 tables/results.csv，告诉我行数、列数和主要数值统计。",
    "请把‘模型负责决策，工具负责执行，记忆提供上下文’转换为 Markdown 列表并保存为 teacher_summary.md。",
]


def _split_ids(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _print_turn_summary(turn_dir: Path) -> None:
    trace = read_json(turn_dir / "trace.json")
    tool_names = []
    for turn in trace.get("turns", []):
        ai_message = turn.get("ai_message") or {}
        for call in ai_message.get("tool_calls", []):
            tool_names.append(call.get("name", "unknown"))
    if tool_names:
        print("调用 Skill：", ", ".join(tool_names), sep="")
    print(f"本轮证据：{turn_dir}")


def _print_demo_prompts() -> None:
    print("建议依次输入：")
    for index, prompt in enumerate(DEMO_PROMPTS, 1):
        print(f"  {index}. {prompt}")


def _print_skills(tools_config: Path, toolset: str) -> None:
    from b3_tool_layer import get_tools_schema

    schema = get_tools_schema(str(tools_config), toolset)
    print(f"可用 Skill/Tool（{toolset}）：")
    for item in schema:
        function = item.get("function") or {}
        print(f"  - {function.get('name', 'unknown')}: {function.get('description', '')}")


def run_chat(args: argparse.Namespace) -> None:
    outdir = resolve_cli_path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tools_config = resolve_cli_path(args.tools_config)
    memory_config = resolve_cli_path(args.memory_config)
    model_config = resolve_cli_path(args.model_config)
    system_prompt = resolve_cli_path(args.system_prompt_path)
    selected_memory_ids = _split_ids(args.selected_memory_ids)

    print("Local Agent Chat")
    print("命令：/skills 查看能力，/demo 查看演示问题，/memory 查看记忆，/exit 退出。")
    print(
        f"llm_mode={args.llm_mode}, profile={args.model_profile or 'auto'}, "
        f"binding={args.tool_binding or 'auto'}, output={outdir}"
    )

    turn_index = 1
    while True:
        try:
            user_input = input("\nUser> ").strip()
        except EOFError:
            print()
            break
        if not user_input:
            continue
        if user_input in {"/exit", "exit", "quit"}:
            break
        if user_input == "/memory":
            print("当前会话已选择/检索的 Memory ID：", ", ".join(selected_memory_ids) or "无", sep="")
            continue
        if user_input == "/skills":
            _print_skills(tools_config, args.toolset)
            continue
        if user_input == "/demo":
            _print_demo_prompts()
            continue

        conversation_id = f"{args.conversation_id}_{turn_index:03d}"
        turn_dir = outdir / f"turn_{turn_index:03d}"
        turn_dir.mkdir(parents=True, exist_ok=True)
        runtime_input = {
            "conversation_id": conversation_id,
            "user_input": user_input,
            "system_prompt_path": str(system_prompt),
            "selected_memory_ids": selected_memory_ids,
            "use_global_memory": args.use_global_memory,
            "toolset": args.toolset,
            "max_turns": args.max_turns,
            "save_memory": args.save_memory,
            "model_profile": args.model_profile,
            "tool_binding": args.tool_binding,
            "decision_strategy": args.decision_strategy,
        }
        runtime_path = turn_dir / "runtime_input.json"
        write_json(runtime_input, runtime_path)

        print("Agent> running...")
        try:
            with open(os.devnull, "w", encoding="utf-8") as quiet_output:
                with redirect_stdout(quiet_output):
                    result = run_agent(
                        str(runtime_path),
                        str(tools_config),
                        str(memory_config),
                        str(model_config),
                        str(turn_dir),
                        args.llm_mode,
                        model_profile=args.model_profile,
                        tool_binding=args.tool_binding,
                        decision_strategy=args.decision_strategy,
                    )
            generate_trace_html(turn_dir, turn_dir / "agent_trace.html", f"Chat Turn {turn_index}")
            answer = result.get("final_answer", "").strip()
            print("\nAgent>")
            print(answer if answer else "(empty answer)")
            _print_turn_summary(turn_dir)
            saved = result.get("saved_memory") or {}
            selected = result.get("selected_memory") or {}
            for document in selected.get("selected_memory_docs", []):
                memory_id = document.get("memory_id") if isinstance(document, dict) else None
                if memory_id and memory_id not in selected_memory_ids:
                    selected_memory_ids.append(memory_id)
            memory_id = saved.get("memory_id")
            if memory_id and memory_id not in selected_memory_ids:
                selected_memory_ids.append(memory_id)
        except Exception as exc:
            print(f"Agent error: {type(exc).__name__}: {exc}")
        turn_index += 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive command-line chat for the local Agent.")
    parser.add_argument("--tools_config", default="../configs/tools.yaml")
    parser.add_argument("--memory_config", default="../configs/memory.yaml")
    parser.add_argument("--model_config", default="../configs/model.yaml")
    parser.add_argument("--system_prompt_path", default="../prompts/local_tool_agent.txt")
    parser.add_argument("--llm_mode", choices=["mock", "prompt_json", "native_tools"], default="mock")
    parser.add_argument("--model_profile")
    parser.add_argument("--tool_binding", choices=["prompt_json", "native_tools"])
    parser.add_argument("--decision_strategy", choices=["react", "plan_execute"], default="react")
    parser.add_argument("--toolset", default="basic_tools")
    parser.add_argument("--conversation_id", default="chat_demo")
    parser.add_argument("--selected_memory_ids", default="")
    parser.add_argument("--use_global_memory", action="store_true", default=True)
    parser.add_argument("--no_global_memory", dest="use_global_memory", action="store_false")
    parser.add_argument("--max_turns", type=int, default=3)
    parser.add_argument("--save_memory", choices=["none", "conversation", "global"], default="conversation")
    parser.add_argument("--outdir", default="../outputs/chat_demo")
    return parser


def main(argv: list[str] | None = None) -> int:
    run_chat(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
