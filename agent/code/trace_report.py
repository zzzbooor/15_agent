from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from common.io_utils import read_json, read_text, write_text
from common.path_utils import resolve_cli_path


def _load_json(path: Path, default: Any) -> Any:
    return read_json(path) if path.exists() else default


def _load_text(path: Path, default: str = "") -> str:
    return read_text(path) if path.exists() else default


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _pretty_json(value: Any, max_chars: int = 1600) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n..."
    return _escape(text)


def _safe_status(value: str | None) -> str:
    return value if value in {"success", "partial", "error", "running", "max_turns_exceeded", "llm_parse_error"} else "unknown"


def _infer_tool_reason(name: str, args: dict, user_input: str) -> str:
    if name == "file_reader":
        return f"User asked about a local file, so the agent reads `{args.get('path', '')}` before answering."
    if name == "calculator":
        return "User request contains an arithmetic task, so the agent uses the calculator for exact computation."
    if name == "local_file_search":
        return "User request asks for search or retrieval, so the agent scans local documents for matching snippets."
    if name == "table_analyzer":
        return "User request points to tabular data, so the agent analyzes the CSV/TSV before summarizing."
    if name == "format_converter":
        return "User request asks for a format conversion, so the agent converts text into the requested output."
    return f"The agent selected `{name}` based on the current user request: {user_input[:120]}"


def _message_card(message: dict, index: int, current_user: str) -> str:
    role = message.get("role", "unknown")
    role_class = html.escape(role)
    title = f"{index}. {role}"
    content = message.get("content", "")
    blocks = [f'<article class="message {role_class}">', f"<h3>{_escape(title)}</h3>"]
    if role == "assistant" and message.get("tool_calls"):
        blocks.append('<div class="tool-calls">')
        for call in message.get("tool_calls", []):
            name = call.get("name", "unknown")
            args = call.get("args", {})
            reason = _infer_tool_reason(name, args if isinstance(args, dict) else {}, current_user)
            blocks.append(
                "<div class=\"tool-call\">"
                f"<strong>{_escape(name)}</strong>"
                f"<p>{_escape(reason)}</p>"
                f"<pre>{_pretty_json(args, 800)}</pre>"
                "</div>"
            )
        blocks.append("</div>")
    elif role == "tool":
        try:
            payload = json.loads(content)
        except Exception:
            payload = content
        blocks.append(f"<pre>{_pretty_json(payload)}</pre>")
    else:
        blocks.append(f"<p>{_escape(content)}</p>")
    blocks.append("</article>")
    return "\n".join(blocks)


def generate_trace_html(outdir: str | Path, html_path: str | Path | None = None, title: str | None = None) -> Path:
    outdir = Path(outdir).resolve()
    html_path = Path(html_path).resolve() if html_path else outdir / "agent_trace.html"
    trace = _load_json(outdir / "trace.json", {})
    messages = _load_json(outdir / "messages.json", [])
    selected_memory = _load_json(outdir / "selected_memory.json", {})
    tools_schema = _load_json(outdir / "tools_schema.json", [])
    final_answer = _load_text(outdir / "final_answer.md", "").strip()

    status = _safe_status(trace.get("status"))
    title = title or f"Agent Trace - {trace.get('conversation_id', outdir.name)}"
    tool_names = [
        item.get("function", {}).get("name", "unknown")
        for item in tools_schema
        if isinstance(item, dict)
    ]
    memory_docs = selected_memory.get("selected_memory_docs", []) if isinstance(selected_memory, dict) else []

    current_user = ""
    cards = []
    for index, message in enumerate(messages, 1):
        if message.get("role") == "user":
            current_user = message.get("content", "")
        cards.append(_message_card(message, index, current_user))

    memory_cards = []
    for item in memory_docs:
        memory_cards.append(
            "<div class=\"memory-card\">"
            f"<strong>{_escape(item.get('memory_id', ''))}</strong>"
            f"<span>{_escape(item.get('memory_type', ''))}</span>"
            f"<p>{_escape(item.get('title', ''))}</p>"
            f"<pre>{_escape(str(item.get('content', ''))[:700])}</pre>"
            "</div>"
        )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_escape(title)}</title>
  <style>
    body {{ margin: 0; font-family: Arial, "Microsoft YaHei", sans-serif; background: #f7f8fa; color: #1f2937; }}
    header {{ background: #111827; color: white; padding: 28px 42px; }}
    header h1 {{ margin: 0 0 10px; font-size: 28px; }}
    header p {{ margin: 0; color: #d1d5db; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 28px 24px 48px; }}
    section {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 22px; margin-bottom: 18px; }}
    h2 {{ margin: 0 0 14px; font-size: 20px; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .metric {{ border: 1px solid #e5e7eb; border-radius: 6px; padding: 12px; background: #fbfdff; }}
    .metric span {{ display: block; color: #6b7280; font-size: 12px; }}
    .metric strong {{ font-size: 18px; }}
    .status-success {{ color: #047857; }}
    .status-partial, .status-max_turns_exceeded, .status-llm_parse_error {{ color: #b45309; }}
    .status-error {{ color: #b91c1c; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .chip {{ background: #eef2ff; color: #3730a3; border-radius: 999px; padding: 6px 10px; font-size: 13px; }}
    .message {{ border-left: 5px solid #9ca3af; padding: 14px 16px; margin: 12px 0; background: #fff; border-radius: 6px; }}
    .message h3 {{ margin: 0 0 8px; font-size: 16px; }}
    .message.system {{ border-left-color: #6366f1; }}
    .message.user {{ border-left-color: #0891b2; }}
    .message.assistant {{ border-left-color: #16a34a; }}
    .message.tool {{ border-left-color: #f59e0b; }}
    pre {{ white-space: pre-wrap; overflow-x: auto; background: #111827; color: #f9fafb; border-radius: 6px; padding: 12px; font-size: 12px; }}
    .tool-call, .memory-card {{ border: 1px solid #e5e7eb; border-radius: 6px; padding: 12px; margin: 10px 0; background: #fafafa; }}
    .memory-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }}
    .answer {{ font-size: 16px; line-height: 1.7; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <header>
    <h1>{_escape(title)}</h1>
    <p>Generated from {_escape(str(outdir))}</p>
  </header>
  <main>
    <section>
      <h2>Run Summary</h2>
      <div class="summary">
        <div class="metric"><span>Status</span><strong class="status-{_escape(status)}">{_escape(status)}</strong></div>
        <div class="metric"><span>Conversation</span><strong>{_escape(trace.get('conversation_id', 'unknown'))}</strong></div>
        <div class="metric"><span>Tool Rounds</span><strong>{_escape(trace.get('tool_rounds_used', 0))}</strong></div>
        <div class="metric"><span>LLM Calls</span><strong>{_escape(trace.get('llm_call_count', 0))}</strong></div>
        <div class="metric"><span>Memory Save</span><strong>{_escape((trace.get('memory_save') or {}).get('status', 'n/a'))}</strong></div>
      </div>
    </section>
    <section>
      <h2>Final Answer</h2>
      <div class="answer">{_escape(final_answer)}</div>
    </section>
    <section>
      <h2>Memory Used</h2>
      <div class="memory-grid">{''.join(memory_cards) if memory_cards else '<p>No memory documents were loaded.</p>'}</div>
    </section>
    <section>
      <h2>Available Tools</h2>
      <div class="chips">{''.join(f'<span class="chip">{_escape(name)}</span>' for name in tool_names)}</div>
    </section>
    <section>
      <h2>Message Timeline</h2>
      {''.join(cards)}
    </section>
  </main>
</body>
</html>
"""
    write_text(html_text, html_path)
    return html_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a readable HTML report from one Agent output directory.")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--html")
    parser.add_argument("--title")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = generate_trace_html(resolve_cli_path(args.outdir), resolve_cli_path(args.html) if args.html else None, args.title)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
