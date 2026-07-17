from __future__ import annotations

import argparse
import html
from pathlib import Path

from b1_agent_runtime import run_agent
from b3_tool_layer import execute_tool_calls, get_tools_schema
from b5_memory import load_memory, save_memory
from common.io_utils import read_json, read_text, write_json, write_text
from common.path_utils import resolve_cli_path
from trace_report import generate_trace_html


SHOWCASE_CASES = [
    {
        "id": "case_1_read_file",
        "title": "Read a local file and summarize",
        "user_input": "帮我阅读 docs/agent_intro.txt，总结三条中文要点。",
        "expected_tools": ["file_reader"],
        "fallback_tool_calls": [
            {"id": "call_001", "name": "file_reader", "args": {"path": "docs/agent_intro.txt", "max_chars": 2000}}
        ],
    },
    {
        "id": "case_2_calculator",
        "title": "Exact calculator tool use",
        "user_input": "请计算 23 * 17 + 9，只给出计算结果并说明使用了什么工具。",
        "expected_tools": ["calculator"],
        "fallback_tool_calls": [
            {"id": "call_001", "name": "calculator", "args": {"expression": "23 * 17 + 9"}}
        ],
    },
    {
        "id": "case_3_search_docs",
        "title": "Search local documents",
        "user_input": "请搜索 docs 目录里和 tool calling 有关的内容，并用中文总结找到的信息。",
        "expected_tools": ["local_file_search"],
        "fallback_tool_calls": [
            {
                "id": "call_001",
                "name": "local_file_search",
                "args": {"query": "tool calling", "root_dir": "docs", "file_types": ["txt", "md"], "top_k": 3},
            }
        ],
    },
    {
        "id": "case_4_table_analyze",
        "title": "Analyze a CSV table",
        "user_input": "请分析 tables/results.csv 这个表格，说明行列数和主要数值统计。",
        "expected_tools": ["table_analyzer"],
        "fallback_tool_calls": [
            {
                "id": "call_001",
                "name": "table_analyzer",
                "args": {"path": "tables/results.csv", "max_rows_preview": 5, "describe": True},
            }
        ],
    },
    {
        "id": "case_5_memory_followup",
        "title": "Use saved memory in a follow-up",
        "user_input": "结合上一次保存的记忆，说明这个 Agent 系统由哪些模块配合完成任务。",
        "expected_tools": [],
        "use_previous_memory": True,
    },
]


def _used_tools(trace: dict) -> list[str]:
    names = []
    for turn in trace.get("turns", []):
        ai_message = turn.get("ai_message") or {}
        for call in ai_message.get("tool_calls", []):
            names.append(call.get("name", "unknown"))
    return names


def _expected_tools_satisfied(expected: list[str], used: list[str]) -> bool:
    return all(name in used for name in expected)


def _fallback_answer(tool_messages: list[dict]) -> str:
    lines = ["Showcase fallback used deterministic B3/B2 tool execution."]
    for message in tool_messages:
        try:
            result = read_json_from_string(message.get("content", "{}"))
        except Exception:
            result = {}
        name = result.get("skill_name", message.get("name", "unknown"))
        status = result.get("status", message.get("status", "unknown"))
        output = result.get("output") or {}
        if status != "success":
            lines.append(f"- {name}: failed: {result.get('error')}")
        elif name == "file_reader":
            content = output.get("content", "")
            points = [item.strip(" 。") for item in content.splitlines() if item.strip()][:3]
            lines.append("- file_reader: " + "；".join(points))
        elif name == "calculator":
            lines.append(f"- calculator: result = {output.get('result')}")
        elif name == "local_file_search":
            snippets = [item.get("snippet", "") for item in output.get("results", [])[:3]]
            lines.append("- local_file_search: " + " | ".join(snippets))
        elif name == "table_analyzer":
            lines.append(
                f"- table_analyzer: {output.get('num_rows')} rows, {output.get('num_columns')} columns, "
                f"stats={output.get('describe')}"
            )
        else:
            lines.append(f"- {name}: {output}")
    return "\n".join(lines)


def read_json_from_string(text: str) -> dict:
    import json

    value = json.loads(text)
    return value if isinstance(value, dict) else {}


def _write_fallback_case(
    case_dir: Path,
    case: dict,
    runtime: dict,
    tools_config: Path,
    memory_config: Path,
    toolset: str,
) -> tuple[str, list[str], str | None]:
    tool_calls = case.get("fallback_tool_calls") or []
    selected_memory = load_memory(
        str(memory_config),
        runtime.get("selected_memory_ids", []),
        bool(runtime.get("use_global_memory", True)),
        runtime.get("user_input", ""),
        str(case_dir),
    )
    tools_schema = get_tools_schema(str(tools_config), toolset, str(case_dir))
    tool_messages = execute_tool_calls(tool_calls, str(tools_config), toolset, str(case_dir)) if tool_calls else []
    final_answer = _fallback_answer(tool_messages) if tool_messages else "No tool fallback was required for this case."
    messages = [
        {"role": "system", "content": "Showcase deterministic fallback path. It demonstrates B3 executing B2 tools."},
        {"role": "user", "content": runtime["user_input"]},
    ]
    if tool_calls:
        messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
        messages.extend(tool_messages)
    messages.append({"role": "assistant", "content": final_answer, "tool_calls": []})
    write_json(messages, case_dir / "messages.json")
    write_text(final_answer + "\n", case_dir / "final_answer.md")
    used_tools = [call.get("name", "unknown") for call in tool_calls]
    trace = {
        "conversation_id": runtime["conversation_id"],
        "execution_mode": "showcase_fallback",
        "status": "diagnostic_only",
        "toolset": toolset,
        "max_turns": runtime["max_turns"],
        "tool_rounds_used": 1 if tool_calls else 0,
        "llm_call_count": 0,
        "turns": [
            {
                "turn_index": 1,
                "user_turn_index": 1,
                "user_input": runtime["user_input"],
                "ai_message": {"role": "assistant", "content": "", "tool_calls": tool_calls} if tool_calls else {"role": "assistant", "content": final_answer, "tool_calls": []},
                "llm_status": "diagnostic_fallback_no_llm",
                "llm_error": None,
                "tool_messages": tool_messages,
                "latency_ms": None,
            }
        ],
        "final_answer_path": "final_answer.md",
        "memory_save": {"requested": "conversation", "status": "not_requested"},
        "warnings": ["real model path did not satisfy this showcase case; deterministic B3/B2 fallback was used"],
        "error": None,
    }
    write_json(trace, case_dir / "trace.json")
    saved_memory_id = None
    try:
        saved = save_memory(
            str(memory_config),
            runtime["conversation_id"],
            "conversation",
            str(case_dir / "messages.json"),
            str(case_dir / "trace.json"),
            str(case_dir / "final_answer.md"),
            str(case_dir),
        )
        saved_memory_id = saved.get("memory_id")
        trace["memory_save"] = {"requested": "conversation", "status": "success"}
        write_json(trace, case_dir / "trace.json")
    except Exception as exc:
        trace["memory_save"] = {"requested": "conversation", "status": "error", "error": {"type": type(exc).__name__, "message": str(exc)}}
        write_json(trace, case_dir / "trace.json")
    return final_answer, used_tools, saved_memory_id


def _write_overview_html(outdir: Path, rows: list[dict]) -> Path:
    cards = []
    for row in rows:
        status = html.escape(row["status"])
        tools = ", ".join(row.get("used_tools") or ["none"])
        fallback = "yes" if row.get("fallback_used") else "no"
        answer = html.escape(row.get("final_answer", "")[:500])
        cards.append(
            "<section>"
            f"<h2>{html.escape(row['title'])}</h2>"
            f"<p><strong>Status:</strong> {status}</p>"
            f"<p><strong>Fallback used:</strong> {fallback}</p>"
            f"<p><strong>Used tools:</strong> {html.escape(tools)}</p>"
            f"<p><strong>Trace:</strong> <a href=\"{html.escape(row['trace_html'])}\">{html.escape(row['trace_html'])}</a></p>"
            f"<pre>{answer}</pre>"
            "</section>"
        )
    text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agent Showcase</title>
  <style>
    body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 0; background: #f7f8fa; color: #1f2937; }}
    header {{ background: #0f172a; color: white; padding: 28px 40px; }}
    main {{ max-width: 1040px; margin: 0 auto; padding: 24px; }}
    section {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 18px; margin-bottom: 16px; }}
    pre {{ white-space: pre-wrap; background: #111827; color: #f9fafb; padding: 12px; border-radius: 6px; }}
  </style>
</head>
<body>
  <header><h1>Agent Showcase</h1><p>Five acceptance-ready demonstrations.</p></header>
  <main>{''.join(cards)}</main>
</body>
</html>
"""
    target = outdir / "showcase_report.html"
    write_text(text, target)
    return target


def run_showcase(args: argparse.Namespace) -> dict:
    outdir = resolve_cli_path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tools_config = resolve_cli_path(args.tools_config)
    memory_config = resolve_cli_path(args.memory_config)
    model_config = resolve_cli_path(args.model_config)
    system_prompt = resolve_cli_path(args.system_prompt_path)

    rows = []
    previous_memory_ids: list[str] = []
    for index, case in enumerate(SHOWCASE_CASES, 1):
        case_dir = outdir / case["id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        selected_memory_ids = previous_memory_ids[:] if case.get("use_previous_memory") else []
        runtime = {
            "conversation_id": f"showcase_{index:02d}_{case['id']}",
            "user_input": case["user_input"],
            "system_prompt_path": str(system_prompt),
            "selected_memory_ids": selected_memory_ids,
            "use_global_memory": True,
            "toolset": args.toolset,
            "max_turns": args.max_turns,
            "save_memory": "conversation",
            "model_profile": args.model_profile,
            "tool_binding": args.tool_binding,
            "decision_strategy": args.decision_strategy,
        }
        runtime_path = case_dir / "runtime_input.json"
        write_json(runtime, runtime_path)
        try:
            result = run_agent(
                str(runtime_path),
                str(tools_config),
                str(memory_config),
                str(model_config),
                str(case_dir),
                args.llm_mode,
                model_profile=args.model_profile,
                tool_binding=args.tool_binding,
                decision_strategy=args.decision_strategy,
            )
            trace = read_json(case_dir / "trace.json")
            trace_html = generate_trace_html(case_dir, case_dir / "agent_trace.html", case["title"])
            used_tools = _used_tools(trace)
            saved = result.get("saved_memory") or {}
            memory_id = saved.get("memory_id")
            if memory_id:
                previous_memory_ids.append(memory_id)
            status = result.get("status", "unknown")
            final_answer = result.get("final_answer", "")
            error = None
            fallback_used = False
            if case["expected_tools"] and (
                status not in {"success", "partial"}
                or not _expected_tools_satisfied(case["expected_tools"], used_tools)
            ) and args.allow_diagnostic_fallback:
                final_answer, used_tools, fallback_memory_id = _write_fallback_case(
                    case_dir,
                    case,
                    runtime,
                    tools_config,
                    memory_config,
                    args.toolset,
                )
                trace_html = generate_trace_html(case_dir, case_dir / "agent_trace.html", case["title"])
                status = "diagnostic_only"
                fallback_used = True
                if fallback_memory_id:
                    previous_memory_ids.append(fallback_memory_id)
            elif case["expected_tools"] and not _expected_tools_satisfied(case["expected_tools"], used_tools):
                status = "failed_acceptance"
                error = {
                    "type": "ExpectedToolNotCalled",
                    "message": f"expected {case['expected_tools']}, model called {used_tools}",
                }
        except Exception as exc:
            fallback_used = False
            if case.get("fallback_tool_calls") and args.allow_diagnostic_fallback:
                final_answer, used_tools, fallback_memory_id = _write_fallback_case(
                    case_dir,
                    case,
                    runtime,
                    tools_config,
                    memory_config,
                    args.toolset,
                )
                trace_html = generate_trace_html(case_dir, case_dir / "agent_trace.html", case["title"])
                status = "diagnostic_only"
                error = {"type": type(exc).__name__, "message": str(exc), "handled_by_fallback": True}
                fallback_used = True
                if fallback_memory_id:
                    previous_memory_ids.append(fallback_memory_id)
            else:
                trace_html = case_dir / "agent_trace.html"
                used_tools = []
                status = "error"
                final_answer = ""
                error = {"type": type(exc).__name__, "message": str(exc)}
        rows.append(
            {
                "case_id": case["id"],
                "title": case["title"],
                "status": status,
                "expected_tools": case["expected_tools"],
                "used_tools": used_tools,
                "final_answer": final_answer,
                "trace_html": str(trace_html.relative_to(outdir)),
                "outdir": str(case_dir),
                "error": error,
                "fallback_used": fallback_used,
            }
        )

    report_lines = [
        "# Agent Showcase Report",
        "",
        f"- LLM mode: `{args.llm_mode}`",
        f"- Toolset: `{args.toolset}`",
        f"- Cases: `{len(rows)}`",
        "",
        "| Case | Status | Fallback | Expected Tools | Used Tools | Trace |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        report_lines.append(
            f"| {row['title']} | `{row['status']}` | `{'yes' if row['fallback_used'] else 'no'}` | `{', '.join(row['expected_tools']) or 'none'}` | "
            f"`{', '.join(row['used_tools']) or 'none'}` | `{row['trace_html']}` |"
        )
    report_lines.append("")
    for row in rows:
        report_lines.append(f"## {row['title']}")
        if row["error"]:
            report_lines.append(f"Error: `{row['error']['type']}: {row['error']['message']}`")
        else:
            report_lines.append(read_text(Path(row["outdir"]) / "final_answer.md").strip())
        report_lines.append("")
    write_text("\n".join(report_lines).rstrip() + "\n", outdir / "showcase_report.md")
    overview_html = _write_overview_html(outdir, rows)
    result = {
        "status": "success"
        if all(row["status"] in {"success", "partial"} and not row["fallback_used"] for row in rows)
        else "partial",
        "cases": rows,
        "html": str(overview_html),
    }
    write_json(result, outdir / "showcase_summary.json")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run five acceptance-ready Agent showcase cases.")
    parser.add_argument("--tools_config", default="../configs/tools.yaml")
    parser.add_argument("--memory_config", default="../configs/memory.yaml")
    parser.add_argument("--model_config", default="../configs/model.yaml")
    parser.add_argument("--system_prompt_path", default="../prompts/local_tool_agent.txt")
    parser.add_argument("--llm_mode", choices=["mock", "prompt_json", "native_tools"], default="prompt_json")
    parser.add_argument("--model_profile")
    parser.add_argument("--tool_binding", choices=["prompt_json", "native_tools"])
    parser.add_argument("--decision_strategy", choices=["react", "plan_execute"], default="react")
    parser.add_argument(
        "--allow_diagnostic_fallback",
        action="store_true",
        help="Run deterministic B2/B3 diagnostics after a model failure; never counts as acceptance success.",
    )
    parser.add_argument("--toolset", default="basic_tools")
    parser.add_argument("--max_turns", type=int, default=3)
    parser.add_argument("--outdir", default="../outputs/showcase")
    return parser


def main(argv: list[str] | None = None) -> int:
    result = run_showcase(build_parser().parse_args(argv))
    print(result["html"])
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
