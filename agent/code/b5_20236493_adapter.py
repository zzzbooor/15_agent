from __future__ import annotations

"""Compatibility wrapper for B1's B5 integration point.

Earlier revisions pointed B1 at an external teammate path under /root/siton-tmp.
The project must be self-contained, so this adapter now delegates to the local
code/b5_memory.py implementation while preserving the import name used by B1.
"""

from b5_memory import load_memory as _local_load_memory
from b5_memory import save_memory as _local_save_memory


def load_memory(
    memory_config: str | None = None,
    selected_memory_ids: list[str] | None = None,
    use_global_memory: bool = True,
    query: str | None = "",
    outdir: str | None = None,
) -> dict:
    if not memory_config:
        raise ValueError("memory_config is required")
    return _local_load_memory(
        memory_config,
        selected_memory_ids or [],
        use_global_memory,
        query,
        outdir,
    )


def save_memory(
    memory_config: str | None = None,
    conversation_id: str = "",
    save_type: str = "conversation",
    messages_path: str | None = None,
    trace_path: str | None = None,
    answer_path: str | None = None,
    outdir: str | None = None,
) -> dict:
    if not memory_config:
        raise ValueError("memory_config is required")
    if not messages_path or not trace_path or not answer_path:
        raise ValueError("messages_path, trace_path, and answer_path are required")
    return _local_save_memory(
        memory_config,
        conversation_id,
        save_type,
        messages_path,
        trace_path,
        answer_path,
        outdir,
    )
