from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path

from b4_core.bindings import NativeToolBindingUnsupported
from b4_core.engine import DecisionEngine
from b4_core.native_parsers import parse_prompt_json
from common.io_utils import append_jsonl, read_json, write_json
from common.logging_utils import now_iso
from common.path_utils import resolve_cli_path
from common.schemas import make_ai_message, validate_messages


PARSE_ERROR_CONTENT = "模型输出无法解析为有效的工具调用或最终回答。"


def _artifact_paths(artifact_dir: str | Path, stem: str | None) -> tuple[Path, Path, Path]:
    directory = Path(artifact_dir)
    prefix = f"{stem}_" if stem else ""
    return (
        directory / f"{prefix}raw_model_output.json",
        directory / f"{prefix}ai_message.json",
        directory / "llm_run_log.jsonl",
    )


def _parse_model_output(raw_text: str) -> tuple[dict, dict]:
    """Compatibility helper retained for earlier unit tests and notebooks."""

    parsed = parse_prompt_json(raw_text, "legacy")
    return parsed.candidate, parsed.ai_message


def _mock_generate(messages: list[dict]) -> dict:
    last_user_index = max(
        (index for index, message in enumerate(messages) if message.get("role") == "user"),
        default=0,
    )
    user_text = messages[last_user_index].get("content", "") if messages else ""
    current = messages[last_user_index + 1 :]
    tool_messages = [message for message in current if message.get("role") == "tool"]
    if not tool_messages:
        path_match = re.search(r"docs/[A-Za-z0-9_.\-/]+", user_text)
        path = path_match.group(0) if path_match else "docs/agent_intro.txt"
        return make_ai_message(
            "",
            [{"id": "call_mock_001", "name": "file_reader", "args": {"path": path, "max_chars": 2000}}],
        )

    latest = tool_messages[-1]
    try:
        result = json.loads(latest.get("content", ""))
    except json.JSONDecodeError:
        result = {"status": "error", "error": {"message": "ToolMessage is not JSON"}}
    if latest.get("status") != "success" or result.get("status") != "success":
        error = result.get("error") or {}
        detail = error.get("message", "unknown tool error") if isinstance(error, dict) else str(error)
        return make_ai_message(f"工具调用失败：{detail}", [])
    output = result.get("output")
    return make_ai_message("工具执行成功：" + json.dumps(output, ensure_ascii=False), [])


def _write_artifacts(
    artifact_dir: str | Path,
    artifact_stem: str | None,
    raw_record: dict,
    ai_message: dict,
) -> None:
    raw_path, message_path, log_path = _artifact_paths(artifact_dir, artifact_stem)
    write_json(raw_record, raw_path)
    write_json(ai_message, message_path)
    append_jsonl(
        {
            "timestamp": raw_record["generated_at"],
            "mode": raw_record["mode"],
            "binding": raw_record.get("binding"),
            "profile": raw_record.get("profile"),
            "status": raw_record["status"],
            "usage": raw_record.get("usage"),
            "raw_output_path": str(raw_path),
            "ai_message_path": str(message_path),
            "error": raw_record.get("error"),
        },
        log_path,
    )


def generate_ai_message(
    model_config: str,
    messages: list[dict],
    tools_schema: list[dict],
    mode: str = "prompt_json",
    artifact_dir: str | None = None,
    artifact_stem: str | None = None,
    *,
    profile: str | None = None,
    binding: str | None = None,
    strategy: str = "react",
) -> dict:
    """Generate one standard flat-contract AIMessage.

    The first six parameters intentionally retain their original positions so
    B1 can call this facade without changes. New routing controls are keyword
    only and all additional return fields are optional metadata for B1.
    """

    validated_messages = validate_messages(deepcopy(messages))
    if not isinstance(tools_schema, list):
        raise ValueError("tools_schema must be an array")
    if mode not in {"mock", "prompt_json", "native_tools"}:
        raise ValueError("mode must be mock, prompt_json, or native_tools")
    if binding is not None and binding not in {"prompt_json", "native_tools"}:
        raise ValueError("binding must be prompt_json or native_tools")

    generated_at = now_iso()
    effective_binding = binding or ("native_tools" if mode == "native_tools" else "prompt_json")
    call_prefix = artifact_stem or generated_at

    if mode == "mock":
        ai_message = _mock_generate(validated_messages)
        candidate = {"content": ai_message["content"], "tool_calls": ai_message["tool_calls"]}
        metadata = {
            "profile": "mock",
            "route_reason": "mock mode requested",
            "binding": "mock",
            "model_family": "mock",
            "native_parser": None,
            "cache_hit": False,
            "load_latency_ms": 0.0,
            "inference_latency_ms": 0.0,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        raw_text = json.dumps(candidate, ensure_ascii=False)
        prompt_text = ""
        validation = {"valid": True, "errors": []}
        status = "success"
        error = None
    else:
        engine = DecisionEngine(model_config)
        try:
            decision = engine.generate_ai_message(
                validated_messages,
                tools_schema,
                binding=effective_binding,
                profile=profile,
                strategy=strategy,
                call_prefix=call_prefix,
            )
        except NativeToolBindingUnsupported as exc:
            ai_message = make_ai_message(PARSE_ERROR_CONTENT, [])
            candidate = None
            raw_text = ""
            prompt_text = ""
            validation = {"valid": False, "errors": [{"message": str(exc)}]}
            status = "error"
            error = {"type": type(exc).__name__, "message": str(exc)}
            metadata = {
                "profile": profile,
                "route_reason": "native binding capability check failed",
                "binding": effective_binding,
                "usage": None,
            }
        else:
            raw_text = decision.raw.raw_text
            prompt_text = decision.raw.prompt_text
            metadata = decision.raw.metadata()
            validation = decision.tool_call_validation
            if (
                decision.error is None
                and decision.parsed is not None
                and validation.get("valid") is True
            ):
                ai_message = decision.parsed.ai_message
                candidate = decision.parsed.candidate
                status = "success"
                error = None
            elif decision.error is None and decision.parsed is not None:
                ai_message = make_ai_message(PARSE_ERROR_CONTENT, [])
                candidate = decision.parsed.candidate
                status = "error"
                error = {
                    "type": "ToolCallSchemaValidationError",
                    "message": "model tool calls do not match the supplied schema",
                    "details": validation.get("errors", []),
                }
            else:
                ai_message = make_ai_message(PARSE_ERROR_CONTENT, [])
                candidate = None
                status = "error"
                error = decision.error

    usage = metadata.get("usage")
    raw_record = {
        "mode": mode,
        "binding": metadata.get("binding", effective_binding),
        "profile": metadata.get("profile", profile),
        "backend": "mock" if mode == "mock" else "transformers",
        "raw_text": raw_text,
        "prompt_text": prompt_text,
        "parsed_candidate": candidate,
        "tool_call_validation": validation,
        "usage": usage,
        "metadata": metadata,
        "status": status,
        "error": error,
        "generated_at": generated_at,
    }
    if artifact_dir:
        _write_artifacts(artifact_dir, artifact_stem, raw_record, ai_message)
    return {
        "ai_message": ai_message,
        "status": status,
        "error": error,
        "metadata": metadata,
        "usage": usage,
        "profile": metadata.get("profile", profile),
        "binding": metadata.get("binding", effective_binding),
        "tool_call_validation": validation,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate one AIMessage with a local or mock LLM.")
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--messages", required=True)
    parser.add_argument("--tools_schema", required=True)
    parser.add_argument("--mode", choices=["mock", "prompt_json", "native_tools"], required=True)
    parser.add_argument("--binding", choices=["prompt_json", "native_tools"])
    parser.add_argument("--profile")
    parser.add_argument("--strategy", choices=["react", "plan_execute"], default="react")
    parser.add_argument("--outdir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outdir = resolve_cli_path(args.outdir)
        result = generate_ai_message(
            str(resolve_cli_path(args.model_config)),
            read_json(resolve_cli_path(args.messages)),
            read_json(resolve_cli_path(args.tools_schema)),
            args.mode,
            str(outdir),
            profile=args.profile,
            binding=args.binding,
            strategy=args.strategy,
        )
        print(outdir / "ai_message.json")
        return 0 if result["status"] == "success" else 2
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
