from __future__ import annotations

"""Small real-model adapters used by the B3/B5 advanced demonstrations.

The hooks reuse B4's local-only inference facade. They intentionally expose
plain Python callables so B3/B5 stay decoupled from a particular LLM backend.
"""

import itertools
import os
from pathlib import Path
from typing import Any

from b4_local_agent_llm import generate_ai_message


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = PROJECT_ROOT / "configs" / "model.yaml"
DEFAULT_PROFILE = "qwen3_1_7b"
_CALL_INDEX = itertools.count(1)


def _artifact_target(label: str) -> tuple[str | None, str | None]:
    configured = os.environ.get("AGENT_HOOK_ARTIFACT_DIR")
    if not configured:
        return None, None
    directory = Path(configured).expanduser().resolve()
    stem = f"{next(_CALL_INDEX):03d}_{label}"
    return str(directory), stem


def _local_completion(
    messages: list[dict[str, Any]],
    tools_schema: list[dict[str, Any]] | None = None,
    *,
    label: str,
    profile: str = DEFAULT_PROFILE,
) -> dict[str, Any]:
    artifact_dir, artifact_stem = _artifact_target(label)
    result = generate_ai_message(
        str(MODEL_CONFIG),
        messages,
        tools_schema or [],
        "native_tools",
        artifact_dir,
        artifact_stem,
        profile=profile,
        binding="native_tools",
    )
    if result.get("status") != "success":
        error = result.get("error") or {}
        raise RuntimeError(f"local model call failed: {error.get('type')}: {error.get('message')}")
    return result


def select_tool_with_local_model(task: str, tools_schema: list[dict[str, Any]]) -> dict[str, Any]:
    """Select exactly one tool for B3's schema-description comparison."""

    result = _local_completion(
        [
            {
                "role": "system",
                "content": (
                    "Select the single best tool for the task. Call that tool once with valid arguments. "
                    "Do not answer the task directly."
                ),
            },
            {"role": "user", "content": task},
        ],
        tools_schema,
        label="b3_selector",
    )
    calls = result["ai_message"].get("tool_calls") or []
    if len(calls) != 1:
        raise ValueError(f"selector expected one tool call, received {len(calls)}")
    return {
        "selected_tool": calls[0]["name"],
        "profile": result.get("profile"),
        "binding": result.get("binding"),
        "usage": result.get("usage"),
        "tool_call_validation": result.get("tool_call_validation"),
    }


def summarize_memory_with_local_model(text: str, max_chars: int) -> dict[str, Any]:
    """Produce an LLM memory summary with an enforced output character cap."""

    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")
    result = _local_completion(
        [
            {
                "role": "system",
                "content": (
                    "Summarize the memory faithfully in Chinese. Preserve concrete facts and uncertainty. "
                    f"Return plain text no longer than {max_chars} characters."
                ),
            },
            {"role": "user", "content": text},
        ],
        label="b5_summary",
    )
    content = result["ai_message"].get("content", "").strip()
    if not content:
        raise ValueError("local summarizer returned empty content")
    truncated = len(content) > max_chars
    return {
        "summary": content[:max_chars],
        "profile": result.get("profile"),
        "binding": result.get("binding"),
        "usage": result.get("usage"),
        "postprocess_truncated": truncated,
    }


def answer_with_memory_local_model(query: str, memory_context: str) -> dict[str, Any]:
    """Answer once for the B5 bad-memory A/B experiment."""

    result = _local_completion(
        [
            {
                "role": "system",
                "content": (
                    "Answer the question using the supplied memory only. If memory conflicts or is insufficient, "
                    "state that explicitly. Keep the answer concise."
                ),
            },
            {
                "role": "user",
                "content": f"Memory context:\n{memory_context}\n\nQuestion:\n{query}",
            },
        ],
        label="b5_ab_answer",
    )
    answer = result["ai_message"].get("content", "").strip()
    if not answer:
        raise ValueError("local responder returned empty content")
    return {
        "answer": answer,
        "profile": result.get("profile"),
        "binding": result.get("binding"),
        "usage": result.get("usage"),
    }


def evaluate_bad_memory_local_model(
    query: str,
    baseline_answer: str,
    injected_answer: str,
    bad_memory: str,
) -> dict[str, Any]:
    """Ask the local model to describe the observed A/B change without hiding it."""

    result = _local_completion(
        [
            {
                "role": "system",
                "content": "Compare two answers and explain how the injected memory changed the second answer.",
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{query}\n\nBaseline answer:\n{baseline_answer}\n\n"
                    f"Answer with injected memory:\n{injected_answer}\n\nInjected memory:\n{bad_memory}"
                ),
            },
        ],
        label="b5_ab_evaluation",
    )
    return {
        "status": "completed",
        "model_assessment": result["ai_message"].get("content", "").strip(),
        "profile": result.get("profile"),
        "binding": result.get("binding"),
        "usage": result.get("usage"),
    }


__all__ = [
    "answer_with_memory_local_model",
    "evaluate_bad_memory_local_model",
    "select_tool_with_local_model",
    "summarize_memory_with_local_model",
]
