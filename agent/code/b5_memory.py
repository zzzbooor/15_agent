from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from common.io_utils import append_jsonl, read_json, read_text, read_yaml, write_json, write_text
from common.logging_utils import now_iso
from common.path_utils import resolve_cli_path, resolve_from_file


_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff]+", re.IGNORECASE)


def _memory_paths(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    config = read_yaml(path)
    if not isinstance(config, dict) or not isinstance(config.get("memory"), dict):
        raise ValueError("memory.yaml must define one memory object")
    memory = config["memory"]
    required = ["root_dir", "global_memory_dir", "conversation_memory_dir", "index_path", "max_memory_chars"]
    missing = [name for name in required if name not in memory]
    if missing:
        raise ValueError(f"memory.yaml missing: {', '.join(missing)}")
    root = resolve_from_file(memory["root_dir"], path)
    max_chars = memory["max_memory_chars"]
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0:
        raise ValueError("max_memory_chars must be a positive integer")
    retrieval = memory.get("retrieval", {})
    if not isinstance(retrieval, dict):
        raise ValueError("memory.retrieval must be an object")
    keyword_top_k = retrieval.get("keyword_top_k", 3)
    if not isinstance(keyword_top_k, int) or isinstance(keyword_top_k, bool) or keyword_top_k <= 0:
        raise ValueError("memory.retrieval.keyword_top_k must be a positive integer")
    return {
        "config_path": path,
        "config": config,
        "memory_config": memory,
        "root": root,
        "global": root / memory["global_memory_dir"],
        "conversations": root / memory["conversation_memory_dir"],
        "index": root / memory["index_path"],
        "max_chars": max_chars,
        "keyword_top_k": keyword_top_k,
        "keyword_min_score": float(retrieval.get("keyword_min_score", 0.0)),
    }


def _read_index(index_path: Path) -> dict[str, dict[str, Any]]:
    if not index_path.exists():
        return {}
    index = read_json(index_path)
    if not isinstance(index, dict):
        raise ValueError("memory_index.json must be an object")
    invalid = [key for key, value in index.items() if not isinstance(key, str) or not isinstance(value, dict)]
    if invalid:
        raise ValueError("memory_index.json values must be metadata objects")
    return index


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for chunk in _TOKEN_PATTERN.findall(text.casefold()):
        if re.fullmatch(r"[\u3400-\u4dbf\u4e00-\u9fff]+", chunk):
            if len(chunk) <= 8:
                tokens.append(chunk)
            tokens.extend(chunk)
            tokens.extend(chunk[index : index + 2] for index in range(len(chunk) - 1))
        else:
            tokens.append(chunk)
    return tokens


def _document_path(paths: dict[str, Any], memory_id: str, metadata: dict[str, Any]) -> tuple[Path, str]:
    relative_path = metadata.get("path")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"memory path is missing for {memory_id}")
    document_path = (paths["root"] / relative_path).resolve()
    try:
        document_path.relative_to(Path(paths["root"]).resolve())
    except ValueError as exc:
        raise ValueError(f"memory path escapes root for {memory_id}") from exc
    if not document_path.is_file():
        raise FileNotFoundError(f"memory file not found: {relative_path}")
    return document_path, relative_path


def _load_document(paths: dict[str, Any], memory_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
    document_path, relative_path = _document_path(paths, memory_id, metadata)
    content = read_text(document_path)
    return {
        "memory_id": memory_id,
        "memory_type": metadata.get("memory_type"),
        "title": metadata.get("title", memory_id),
        "summary": metadata.get("summary", ""),
        "path": relative_path,
        "content": content,
        "metadata": metadata,
    }


def _bm25_scores(query_tokens: list[str], documents: list[list[str]]) -> list[float]:
    if not documents or not query_tokens:
        return [0.0] * len(documents)
    document_count = len(documents)
    average_length = sum(len(tokens) for tokens in documents) / document_count or 1.0
    query_counts = Counter(query_tokens)
    document_frequencies = {
        token: sum(token in set(document) for document in documents) for token in query_counts
    }
    k1 = 1.5
    b = 0.75
    scores: list[float] = []
    for tokens in documents:
        counts = Counter(tokens)
        length = len(tokens) or 1
        score = 0.0
        for token, query_frequency in query_counts.items():
            frequency = counts.get(token, 0)
            if frequency == 0:
                continue
            document_frequency = document_frequencies[token]
            inverse_frequency = math.log(1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
            denominator = frequency + k1 * (1.0 - b + b * length / average_length)
            score += query_frequency * inverse_frequency * frequency * (k1 + 1.0) / denominator
        scores.append(score)
    return scores


def _iter_documents(
    paths: dict[str, Any],
    index: dict[str, dict[str, Any]],
    memory_ids: Iterable[str] | None = None,
    memory_types: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ids = list(memory_ids) if memory_ids is not None else sorted(index)
    documents: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for memory_id in ids:
        metadata = index.get(memory_id)
        if not isinstance(metadata, dict):
            errors.append({"memory_id": memory_id, "type": "MemoryNotFound", "message": "memory_id does not exist"})
            continue
        if memory_types and metadata.get("memory_type") not in memory_types:
            continue
        try:
            documents.append(_load_document(paths, memory_id, metadata))
        except Exception as exc:
            errors.append({"memory_id": memory_id, "type": type(exc).__name__, "message": str(exc)})
    return documents, errors


def search_memory_keyword(
    config_path: str | Path,
    query: str,
    top_k: int | None = None,
    *,
    memory_ids: Iterable[str] | None = None,
    memory_types: set[str] | None = None,
    include_content: bool = True,
) -> dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    paths = _memory_paths(config_path)
    selected_top_k = paths["keyword_top_k"] if top_k is None else top_k
    if not isinstance(selected_top_k, int) or isinstance(selected_top_k, bool) or selected_top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    index = _read_index(paths["index"])
    documents, errors = _iter_documents(paths, index, memory_ids, memory_types)
    query_tokens = _tokenize(query)
    document_tokens = [
        _tokenize(f"{document['title']} {document['summary']} {document['content']}") for document in documents
    ]
    scores = _bm25_scores(query_tokens, document_tokens)
    rows: list[dict[str, Any]] = []
    for document, score in zip(documents, scores):
        if score <= paths["keyword_min_score"]:
            continue
        row = {
            "memory_id": document["memory_id"],
            "memory_type": document["memory_type"],
            "title": document["title"],
            "summary": document["summary"],
            "path": document["path"],
            "score": round(float(score), 6),
            "retrieval_mode": "keyword_bm25",
            "snippet": document["content"][:240].replace("\n", " ").strip(),
        }
        if include_content:
            row["content"] = document["content"]
        rows.append(row)
    rows.sort(key=lambda item: (-item["score"], item["memory_id"]))
    rows = rows[:selected_top_k]
    return {
        "status": "partial" if errors else "success",
        "mode": "keyword_bm25",
        "query": query,
        "query_tokens": query_tokens,
        "top_k": selected_top_k,
        "candidate_count": len(documents),
        "results": rows,
        "errors": errors,
    }


def load_memory(
    config_path: str,
    selected_memory_ids: list[str],
    use_global_memory: bool,
    query: str | None = None,
    outdir: str | None = None,
) -> dict[str, Any]:
    """Load explicit memories and query-ranked global memories for the stable B1 path."""

    if not isinstance(selected_memory_ids, list) or not all(isinstance(item, str) for item in selected_memory_ids):
        raise ValueError("selected_memory_ids must be a list of strings")
    if not isinstance(use_global_memory, bool):
        raise ValueError("use_global_memory must be boolean")
    paths = _memory_paths(config_path)
    index = _read_index(paths["index"])
    global_ids = sorted(
        memory_id for memory_id, item in index.items() if item.get("memory_type") == "global"
    )
    ranking: dict[str, Any] | None = None
    ranked_global_ids: list[str] = global_ids
    if use_global_memory and isinstance(query, str) and query.strip():
        ranking = search_memory_keyword(
            config_path,
            query,
            paths["keyword_top_k"],
            memory_ids=global_ids,
            memory_types={"global"},
            include_content=False,
        )
        ranked_global_ids = [item["memory_id"] for item in ranking["results"]]
    ordered_ids: list[str] = []
    if use_global_memory:
        ordered_ids.extend(ranked_global_ids)
    ordered_ids.extend(selected_memory_ids)
    ordered_ids = list(dict.fromkeys(ordered_ids))
    score_by_id = {
        item["memory_id"]: item["score"] for item in (ranking or {}).get("results", [])
    }
    documents, errors = _iter_documents(paths, index, ordered_ids)
    docs: list[dict[str, Any]] = []
    remaining = int(paths["max_chars"])
    any_truncated = False
    for document in documents:
        original = document["content"]
        included = original[:remaining] if remaining > 0 else ""
        truncated = len(included) < len(original)
        any_truncated = any_truncated or truncated
        if not included:
            continue
        docs.append(
            {
                "memory_id": document["memory_id"],
                "memory_type": document["memory_type"],
                "title": document["title"],
                "path": document["path"],
                "content": included,
                "original_chars": len(original),
                "included_chars": len(included),
                "truncated": truncated,
                "retrieval_mode": "keyword_bm25" if document["memory_id"] in score_by_id else "explicit_id",
                "retrieval_score": score_by_id.get(document["memory_id"]),
            }
        )
        remaining -= len(included)
    if ranking:
        errors = list(ranking.get("errors", [])) + errors
    deduplicated_errors = list(
        {
            (item.get("memory_id"), item.get("type"), item.get("message")): item for item in errors
        }.values()
    )
    if deduplicated_errors and docs:
        status = "partial"
    elif deduplicated_errors:
        status = "error"
    else:
        status = "success"
    result = {
        "status": status,
        "query": query,
        "query_ranking": {
            "applied": ranking is not None,
            "mode": "keyword_bm25" if ranking is not None else None,
            "top_k": paths["keyword_top_k"] if ranking is not None else None,
            "candidate_count": ranking.get("candidate_count", 0) if ranking else 0,
        },
        "selected_memory_docs": docs,
        "max_memory_chars": paths["max_chars"],
        "total_chars": sum(item["included_chars"] for item in docs),
        "truncated": any_truncated,
        "errors": deduplicated_errors,
    }
    if outdir:
        output_dir = Path(outdir)
        write_json(result, output_dir / "selected_memory.json")
        append_jsonl(
            {
                "timestamp": now_iso(),
                "operation": "load",
                "status": status,
                "query_ranking": result["query_ranking"],
                "selected_ids": [item["memory_id"] for item in docs],
                "errors": deduplicated_errors,
            },
            output_dir / "memory_log.jsonl",
        )
    return result


def _safe_conversation_id(conversation_id: str) -> str:
    if not isinstance(conversation_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", conversation_id):
        raise ValueError("conversation_id may only contain letters, numbers, dot, underscore, and hyphen")
    return conversation_id


def save_memory(
    config_path: str,
    conversation_id: str,
    save_type: str,
    messages_path: str,
    trace_path: str,
    answer_path: str,
    outdir: str | None = None,
) -> dict[str, Any]:
    conversation_id = _safe_conversation_id(conversation_id)
    if save_type not in {"conversation", "global"}:
        raise ValueError("save_type must be conversation or global")
    paths = _memory_paths(config_path)
    messages = read_json(messages_path)
    trace = read_json(trace_path)
    answer = read_text(answer_path).strip()
    if not isinstance(messages, list) or not isinstance(trace, dict):
        raise ValueError("messages must be an array and trace must be an object")
    now = now_iso()
    memory_id = f"mem_{save_type}_{conversation_id}"
    target_dir = paths["conversations"] if save_type == "conversation" else paths["global"]
    relative_dir = "conversations" if save_type == "conversation" else "global"
    target_path = Path(target_dir) / f"{conversation_id}.md"
    relative_path = f"{relative_dir}/{conversation_id}.md"
    title = f"{save_type.title()} {conversation_id}"
    summary = answer[:200]
    markdown = (
        f"# {title}\n\n"
        f"- memory_id: `{memory_id}`\n"
        f"- conversation_id: `{conversation_id}`\n"
        f"- created_or_updated_at: `{now}`\n\n"
        "## Final Answer\n\n"
        f"{answer}\n\n"
        "## Messages\n\n```json\n"
        f"{json.dumps(messages, ensure_ascii=False, indent=2)}\n```\n\n"
        "## Trace\n\n```json\n"
        f"{json.dumps(trace, ensure_ascii=False, indent=2)}\n```\n"
    )
    write_text(markdown, target_path)
    index = _read_index(paths["index"])
    existing = index.get(memory_id, {})
    created_at = existing.get("created_at", now)
    index[memory_id] = {
        "memory_id": memory_id,
        "memory_type": save_type,
        "title": title,
        "summary": summary,
        "path": relative_path,
        "conversation_id": conversation_id,
        "created_at": created_at,
        "updated_at": now,
    }
    write_json(index, paths["index"])
    result = {
        "status": "success",
        "memory_id": memory_id,
        "memory_type": save_type,
        "conversation_id": conversation_id,
        "title": title,
        "summary": summary,
        "path": relative_path,
        "index_path": Path(paths["index"]).name,
        "created_at": created_at,
        "updated_at": now,
        "source_paths": {
            "messages": str(messages_path),
            "trace": str(trace_path),
            "answer": str(answer_path),
        },
    }
    if outdir:
        output_dir = Path(outdir)
        write_json(result, output_dir / "saved_memory.json")
        append_jsonl(
            {"timestamp": now, "operation": "save", "status": "success", "memory_id": memory_id},
            output_dir / "memory_log.jsonl",
        )
    return result


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select or save local memory documents.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--select_memory_ids", nargs="*")
    parser.add_argument("--use_global_memory", type=parse_bool)
    parser.add_argument("--query")
    parser.add_argument("--save_type", choices=["conversation", "global"])
    parser.add_argument("--save_input_path")
    parser.add_argument("--outdir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config_path = resolve_cli_path(args.config)
        outdir = resolve_cli_path(args.outdir)
        if args.save_type or args.save_input_path:
            if not args.save_type or not args.save_input_path:
                raise ValueError("--save_type and --save_input_path must be provided together")
            input_path = resolve_cli_path(args.save_input_path)
            payload = read_json(input_path)
            if payload.get("save_type") != args.save_type:
                raise ValueError("CLI save_type must match memory_save_input.json")
            base = input_path.parent
            save_memory(
                str(config_path),
                payload["conversation_id"],
                args.save_type,
                str((base / payload["messages_path"]).resolve()),
                str((base / payload["trace_path"]).resolve()),
                str((base / payload["answer_path"]).resolve()),
                str(outdir),
            )
            print(outdir / "saved_memory.json")
        else:
            if args.select_memory_ids is None and args.use_global_memory is None:
                raise ValueError("select mode requires --select_memory_ids or --use_global_memory")
            load_memory(
                str(config_path),
                args.select_memory_ids or [],
                bool(args.use_global_memory),
                args.query,
                str(outdir),
            )
            print(outdir / "selected_memory.json")
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
