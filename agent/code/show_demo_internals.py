from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from common.io_utils import write_json, write_text
from common.path_utils import resolve_cli_path


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _shorten(text: str, maximum: int = 120) -> str:
    compact = " ".join(str(text).split())
    return compact if len(compact) <= maximum else compact[: maximum - 3] + "..."


def build_summary(session_dir: Path) -> dict[str, Any]:
    turn_dirs = sorted(path for path in session_dir.glob("turn_*") if path.is_dir())
    if not turn_dirs:
        raise ValueError(f"会话目录中没有 turn_* 结果：{session_dir}")

    interactions: list[dict[str, Any]] = []
    b3_calls: list[dict[str, Any]] = []
    b4_calls: list[dict[str, Any]] = []
    selected_memory_ids: set[str] = set()
    saved_memory_ids: set[str] = set()
    tool_schema_names: set[str] = set()
    total_tokens = 0
    session_mode = _read_json(session_dir / "session_mode.json", {})

    for turn_dir in turn_dirs:
        runtime = _read_json(turn_dir / "runtime_input.json", {})
        trace = _read_json(turn_dir / "trace.json", {})
        messages = _read_json(turn_dir / "messages.json", [])
        tool_messages = _read_json(turn_dir / "tool_messages.json", [])
        selected = _read_json(turn_dir / "selected_memory.json", {})
        saved = _read_json(turn_dir / "saved_memory.json", {})
        schema = _read_json(turn_dir / "tools_schema.json", [])

        for item in schema if isinstance(schema, list) else []:
            name = (item.get("function") or {}).get("name") if isinstance(item, dict) else None
            if isinstance(name, str):
                tool_schema_names.add(name)

        skills: list[dict[str, Any]] = []
        for message in tool_messages if isinstance(tool_messages, list) else []:
            try:
                result = json.loads(message.get("content", "{}"))
            except (json.JSONDecodeError, AttributeError):
                result = {}
            skills.append(
                {
                    "name": message.get("name"),
                    "status": result.get("status", message.get("status")),
                    "output_preview": _shorten(json.dumps(result.get("output"), ensure_ascii=False)),
                    "error": result.get("error"),
                }
            )

        roles = [message.get("role") for message in messages if isinstance(message, dict)]
        interaction = {
            "turn": turn_dir.name,
            "user_input": runtime.get("user_input", ""),
            "status": trace.get("status"),
            "roles": roles,
            "tool_rounds": trace.get("tool_rounds_used"),
            "llm_calls": trace.get("llm_call_count"),
            "skills": skills,
            "final_answer": _read_json(turn_dir / "checkpoint.json", {}).get("final_answer", ""),
            "trace_path": str(turn_dir / "trace.json"),
        }
        if not interaction["final_answer"]:
            try:
                interaction["final_answer"] = (turn_dir / "final_answer.md").read_text(encoding="utf-8").strip()
            except OSError:
                pass
        interactions.append(interaction)

        b3_calls.extend(_read_jsonl(turn_dir / "tool_call_log.jsonl"))
        llm = trace.get("llm") or {}
        usage = llm.get("usage") or {}
        if isinstance(usage.get("total_tokens"), int):
            total_tokens += usage["total_tokens"]
        for raw_path in sorted((turn_dir / "llm_calls").glob("*_raw_model_output.json")):
            raw = _read_json(raw_path, {})
            b4_calls.append(
                {
                    "path": str(raw_path),
                    "status": raw.get("status"),
                    "profile": raw.get("profile"),
                    "binding": raw.get("binding"),
                    "usage": raw.get("usage"),
                }
            )

        for document in selected.get("selected_memory_docs", []) if isinstance(selected, dict) else []:
            memory_id = document.get("memory_id") if isinstance(document, dict) else None
            if isinstance(memory_id, str):
                selected_memory_ids.add(memory_id)
        if isinstance(saved, dict) and isinstance(saved.get("memory_id"), str):
            saved_memory_ids.add(saved["memory_id"])

    successful_b3 = sum(row.get("status") == "success" for row in b3_calls)
    return {
        "session_dir": str(session_dir),
        "turn_count": len(interactions),
        "interactions": interactions,
        "b1": {
            "successful_turns": sum(item["status"] == "success" for item in interactions),
            "message_flows": [item["roles"] for item in interactions],
            "tool_rounds": sum(int(item["tool_rounds"] or 0) for item in interactions),
            "llm_calls": sum(int(item["llm_calls"] or 0) for item in interactions),
        },
        "b2": {
            "skills_used": sorted({skill["name"] for item in interactions for skill in item["skills"] if skill["name"]}),
            "executions": [skill for item in interactions for skill in item["skills"]],
        },
        "b3": {
            "schema_tools": sorted(tool_schema_names),
            "call_count": len(b3_calls),
            "successful_calls": successful_b3,
            "error_calls": len(b3_calls) - successful_b3,
        },
        "b4": {
            "generation_count": len(b4_calls),
            "profiles": sorted({row["profile"] for row in b4_calls if row["profile"]}),
            "bindings": sorted({row["binding"] for row in b4_calls if row["binding"]}),
            "total_tokens": total_tokens,
            "raw_artifacts": b4_calls,
        },
        "b5": {
            "memory_mode": session_mode.get("memory_mode", "unknown"),
            "memory_root": session_mode.get("memory_root", ""),
            "selected_memory_ids": sorted(selected_memory_ids),
            "saved_memory_ids": sorted(saved_memory_ids),
            "note": (
                "使用项目独立的持久化 Memory；清理 interactive 输出不会删除它，也不会修改正式 memory/。"
                if session_mode.get("memory_mode") == "persistent"
                else "使用会话输出目录内的临时 Memory，不修改正式 memory/。"
            ),
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    message_flows: list[str] = []
    for roles in summary["b1"]["message_flows"]:
        flow = " → ".join(str(role) for role in roles)
        if flow and flow not in message_flows:
            message_flows.append(flow)
    lines = [
        "# Agent 现场交互与内部模块汇总",
        "",
        f"- 会话目录：`{summary['session_dir']}`",
        f"- 交互轮数：`{summary['turn_count']}`",
        f"- 成功轮数：`{summary['b1']['successful_turns']}`",
        "",
        "## 一、现场人机交互与 B2 Skill",
        "",
        "| 轮次 | 用户指令 | Agent 选择的 Skill | 状态 | 回答摘要 |",
        "|---|---|---|---|---|",
    ]
    for item in summary["interactions"]:
        skill_names = ", ".join(skill["name"] or "unknown" for skill in item["skills"]) or "无需工具"
        lines.append(
            f"| {item['turn']} | {_shorten(item['user_input'], 55)} | {skill_names} | "
            f"{item['status']} | {_shorten(item['final_answer'], 70)} |"
        )

    lines.extend(
        [
            "",
            "## 二、B1 Agent Runtime",
            "",
            f"- 工具轮次总数：`{summary['b1']['tool_rounds']}`",
            f"- 模型调用总数：`{summary['b1']['llm_calls']}`",
            f"- 本次实际消息流：`{'；'.join(message_flows) or '无'}`",
            "- 逐轮执行细节位于各 `turn_*/trace.json`。",
            "",
            "## 三、B3 Tool 层",
            "",
            f"- Schema 中的工具：`{', '.join(summary['b3']['schema_tools'])}`",
            f"- 实际工具调用：`{summary['b3']['call_count']}`",
            f"- 成功/失败：`{summary['b3']['successful_calls']}/{summary['b3']['error_calls']}`",
            "- 参数校验和 SkillResult 位于各 `turn_*/tool_call_log.jsonl`、`tool_messages.json`。",
            "",
            "## 四、B4 本地模型",
            "",
            f"- 模型 profile：`{', '.join(summary['b4']['profiles'])}`",
            f"- 工具绑定：`{', '.join(summary['b4']['bindings'])}`",
            f"- 真实生成次数：`{summary['b4']['generation_count']}`",
            f"- 总 token：`{summary['b4']['total_tokens']}`",
            "- 原始 prompt、raw output、AIMessage 位于各 `turn_*/llm_calls/`。",
            "",
            "## 五、B5 Memory",
            "",
            f"- Memory 模式：`{summary['b5']['memory_mode']}`",
            f"- Memory 目录：`{summary['b5']['memory_root'] or '未记录'}`",
            f"- 注入的 Memory ID：`{', '.join(summary['b5']['selected_memory_ids']) or '无'}`",
            f"- 新保存的 Memory ID：`{', '.join(summary['b5']['saved_memory_ids']) or '无'}`",
            f"- {summary['b5']['note']}",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize one interactive Agent demo by B1-B5 module.")
    parser.add_argument("--session", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        session_dir = resolve_cli_path(args.session)
        summary = build_summary(session_dir)
        markdown = render_markdown(summary)
        write_json(summary, session_dir / "INTERNALS_SUMMARY.json")
        write_text(markdown + "\n", session_dir / "INTERNALS_SUMMARY.md")
        print(markdown)
        print(f"\n内部汇总：{session_dir / 'INTERNALS_SUMMARY.md'}")
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
