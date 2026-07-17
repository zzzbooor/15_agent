from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from b5_memory import (
    _document_path,
    _iter_documents,
    _load_document,
    _memory_paths,
    _read_index,
    search_memory_keyword,
)
from common.io_utils import read_json, read_text, write_json, write_text
from common.logging_utils import now_iso
from common.path_utils import resolve_cli_path, resolve_from_file


Embedder = Callable[[list[str]], list[list[float]]]
Summarizer = Callable[[str, int], str | dict[str, Any]]
Responder = Callable[[str, str], str | dict[str, Any]]
Evaluator = Callable[[str, str, str, str], dict[str, Any]]


def _error(code: str, exc: Exception | str) -> dict[str, Any]:
    return {
        "code": code,
        "type": type(exc).__name__ if isinstance(exc, Exception) else code,
        "message": str(exc),
    }


def _callable_identity(value: Callable[..., Any]) -> str:
    return f"{getattr(value, '__module__', 'unknown')}:{getattr(value, '__qualname__', type(value).__name__)}"


def _load_callable(spec: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise ValueError("callable must use module:function syntax")
    module_name, attribute = spec.split(":", 1)
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise TypeError(f"configured object is not callable: {spec}")
    return value


class LocalTransformerEmbedder:
    """Lazy, local-files-only embedding provider using mean pooling."""

    def __init__(self, config_path: str | Path) -> None:
        paths = _memory_paths(config_path)
        config = paths["memory_config"].get("embedding", {})
        if not isinstance(config, dict) or not config.get("enabled", False):
            raise RuntimeError("local embedding is disabled in memory.yaml")
        model_setting = config.get("model_name_or_path")
        tokenizer_setting = config.get("tokenizer_name_or_path", model_setting)
        if not isinstance(model_setting, str) or not model_setting.strip():
            raise ValueError("memory.embedding.model_name_or_path is required")
        if not isinstance(tokenizer_setting, str) or not tokenizer_setting.strip():
            raise ValueError("memory.embedding.tokenizer_name_or_path is required")
        self.model_path = resolve_from_file(model_setting, paths["config_path"])
        self.tokenizer_path = resolve_from_file(tokenizer_setting, paths["config_path"])
        if not self.model_path.exists() or not self.tokenizer_path.exists():
            raise FileNotFoundError(
                f"local embedding model/tokenizer does not exist: {self.model_path}, {self.tokenizer_path}"
            )
        self.device = str(config.get("device", "cpu"))
        self.max_length = int(config.get("max_length", 512))
        self.trust_remote_code = bool(config.get("trust_remote_code", False))
        self.torch_dtype = str(config.get("torch_dtype", "auto"))
        self._tokenizer: Any = None
        self._model: Any = None

    def _load(self) -> tuple[Any, Any, Any]:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("local embedding requires torch and transformers") from exc
        if self._tokenizer is None or self._model is None:
            dtype_options = {
                "auto": "auto",
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            if self.torch_dtype not in dtype_options:
                raise ValueError(f"unsupported embedding torch_dtype: {self.torch_dtype}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.tokenizer_path),
                local_files_only=True,
                trust_remote_code=self.trust_remote_code,
            )
            self._model = AutoModel.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                trust_remote_code=self.trust_remote_code,
                dtype=dtype_options[self.torch_dtype],
            ).to(self.device)
            self._model.eval()
        return torch, self._tokenizer, self._model

    def __call__(self, texts: list[str]) -> list[list[float]]:
        torch, tokenizer, model = self._load()
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        with torch.no_grad():
            output = model(**encoded)
        hidden = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.detach().cpu().tolist()


def _validate_vectors(vectors: Any, expected_count: int) -> list[list[float]]:
    if not isinstance(vectors, list) or len(vectors) != expected_count:
        raise ValueError(f"embedder must return {expected_count} vectors")
    dimensions: int | None = None
    normalized: list[list[float]] = []
    for vector in vectors:
        if not isinstance(vector, list) or not vector:
            raise ValueError("every embedding must be a non-empty list")
        values = [float(value) for value in vector]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("embeddings must contain finite numbers")
        if dimensions is None:
            dimensions = len(values)
        elif len(values) != dimensions:
            raise ValueError("all embeddings must have the same dimensions")
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0:
            raise ValueError("embedding vector must not be all zeros")
        normalized.append([value / norm for value in values])
    return normalized


def search_memory_embeddings(
    config_path: str | Path,
    query: str,
    top_k: int,
    outdir: str | Path,
    *,
    embedder: Embedder | None = None,
) -> dict[str, Any]:
    output_dir = Path(outdir).resolve()
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    paths = _memory_paths(config_path)
    index = _read_index(paths["index"])
    documents, document_errors = _iter_documents(paths, index)
    provider = embedder
    try:
        if provider is None:
            provider = LocalTransformerEmbedder(config_path)
        texts = [
            query,
            *[
                f"{document['title']}\n{document['summary']}\n{document['content']}"
                for document in documents
            ],
        ]
        started = perf_counter()
        vectors = _validate_vectors(provider(texts), len(texts))
        embedding_latency = round((perf_counter() - started) * 1000, 3)
    except Exception as exc:
        result = {
            "status": "error",
            "mode": "local_embedding",
            "query": query,
            "top_k": top_k,
            "results": [],
            "error": _error("EMBEDDING_UNAVAILABLE", exc),
            "fallback_used": False,
            "document_errors": document_errors,
        }
        write_json(result, output_dir / "b5_embedding_search.json")
        return result
    query_vector = vectors[0]
    rows = []
    for document, vector in zip(documents, vectors[1:]):
        score = sum(left * right for left, right in zip(query_vector, vector))
        rows.append(
            {
                "memory_id": document["memory_id"],
                "memory_type": document["memory_type"],
                "title": document["title"],
                "path": document["path"],
                "score": round(score, 6),
                "snippet": document["content"][:240].replace("\n", " ").strip(),
            }
        )
    rows.sort(key=lambda item: (-item["score"], item["memory_id"]))
    result = {
        "status": "partial" if document_errors else "success",
        "mode": "local_embedding",
        "query": query,
        "top_k": top_k,
        "candidate_count": len(documents),
        "embedding_dimensions": len(query_vector),
        "embedding_latency_ms": embedding_latency,
        "provider": _callable_identity(provider),
        "fallback_used": False,
        "results": rows[:top_k],
        "document_errors": document_errors,
        "error": None,
    }
    write_json(result, output_dir / "b5_embedding_search.json")
    return result


def _call_summarizer(summarizer: Summarizer, text: str, max_chars: int) -> tuple[str, dict[str, Any]]:
    value = summarizer(text, max_chars)
    if isinstance(value, str):
        summary = value
        metadata: dict[str, Any] = {}
    elif isinstance(value, dict) and isinstance(value.get("summary"), str):
        summary = value["summary"]
        metadata = {key: item for key, item in value.items() if key != "summary"}
    else:
        raise ValueError("summarizer must return text or {summary: text}")
    summary = summary.strip()
    if not summary:
        raise ValueError("summarizer returned an empty summary")
    return summary, metadata


def summarize_memory_document(
    config_path: str | Path,
    memory_id: str,
    outdir: str | Path,
    *,
    summarizer: Summarizer | None,
    max_chars: int = 240,
) -> dict[str, Any]:
    output_dir = Path(outdir).resolve()
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")
    paths = _memory_paths(config_path)
    index = _read_index(paths["index"])
    metadata = index.get(memory_id)
    if not isinstance(metadata, dict):
        raise KeyError(f"memory_id does not exist: {memory_id}")
    document = _load_document(paths, memory_id, metadata)
    if summarizer is None:
        result = {
            "status": "not_run",
            "memory_id": memory_id,
            "summary": None,
            "error": _error("LLM_SUMMARIZER_NOT_CONFIGURED", "No LLM summarizer was injected."),
            "fallback_used": False,
        }
        write_json(result, output_dir / "b5_memory_summary.json")
        return result
    started = perf_counter()
    try:
        summary, metadata_result = _call_summarizer(summarizer, document["content"], max_chars)
    except Exception as exc:
        result = {
            "status": "error",
            "memory_id": memory_id,
            "summary": None,
            "error": _error("LLM_SUMMARY_FAILED", exc),
            "fallback_used": False,
        }
    else:
        result = {
            "status": "success",
            "memory_id": memory_id,
            "summary": summary,
            "summary_chars": len(summary),
            "requested_max_chars": max_chars,
            "summarizer": _callable_identity(summarizer),
            "summarizer_metadata": metadata_result,
            "latency_ms": round((perf_counter() - started) * 1000, 3),
            "error": None,
            "fallback_used": False,
        }
    write_json(result, output_dir / "b5_memory_summary.json")
    return result


def _normalize_statement(value: str) -> str:
    value = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", value.strip())
    return re.sub(r"\s+", " ", value).casefold()


def _fact_parts(value: str) -> tuple[str, str] | None:
    stripped = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", value.strip())
    match = re.match(r"^([^:：\n]{1,80})[:：]\s*(.+)$", stripped)
    if not match:
        return None
    return _normalize_statement(match.group(1)), match.group(2).strip()


def update_memory_document(
    config_path: str | Path,
    memory_id: str,
    updates: list[str],
    outdir: str | Path,
    *,
    conflict_policy: str = "record",
    dry_run: bool = False,
    summarizer: Summarizer | None = None,
) -> dict[str, Any]:
    if conflict_policy not in {"record", "replace", "append"}:
        raise ValueError("conflict_policy must be record, replace, or append")
    if not isinstance(updates, list) or not updates or not all(isinstance(item, str) and item.strip() for item in updates):
        raise ValueError("updates must be a non-empty list of non-empty strings")
    paths = _memory_paths(config_path)
    index = _read_index(paths["index"])
    metadata = index.get(memory_id)
    if not isinstance(metadata, dict):
        raise KeyError(f"memory_id does not exist: {memory_id}")
    document_path, relative_path = _document_path(paths, memory_id, metadata)
    existing = read_text(document_path)
    lines = existing.splitlines()
    normalized_lines = {_normalize_statement(line) for line in lines if _normalize_statement(line)}
    fact_lines: dict[str, tuple[int, str, str]] = {}
    for line_index, line in enumerate(lines):
        parts = _fact_parts(line)
        if parts:
            fact_lines.setdefault(parts[0], (line_index, parts[1], line))
    duplicates: list[dict[str, Any]] = []
    additions: list[str] = []
    conflicts: list[dict[str, Any]] = []
    replacement_indexes: dict[int, str] = {}
    for update in updates:
        normalized = _normalize_statement(update)
        if normalized in normalized_lines:
            duplicates.append({"incoming": update, "reason": "exact_normalized_match"})
            continue
        parts = _fact_parts(update)
        if parts and parts[0] in fact_lines:
            line_index, old_value, old_line = fact_lines[parts[0]]
            if _normalize_statement(old_value) == _normalize_statement(parts[1]):
                duplicates.append({"incoming": update, "reason": "same_key_and_value"})
                continue
            conflict = {
                "key": parts[0],
                "existing": old_line,
                "incoming": update,
                "policy": conflict_policy,
                "resolution": "recorded_without_change",
            }
            if conflict_policy == "replace":
                replacement_indexes[line_index] = update.strip()
                conflict["resolution"] = "existing_replaced"
            elif conflict_policy == "append":
                additions.append(update.strip())
                conflict["resolution"] = "incoming_appended"
            conflicts.append(conflict)
            continue
        additions.append(update.strip())
        normalized_lines.add(normalized)
    for line_index, replacement in replacement_indexes.items():
        lines[line_index] = replacement
    if additions:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend([f"## Memory Update {now_iso()}", "", *[f"- {item}" for item in additions]])
    updated_content = "\n".join(lines).rstrip() + "\n"
    summary = metadata.get("summary", "")
    summary_info: dict[str, Any] = {"updated": False, "method": "preserved"}
    if summarizer is not None and (additions or replacement_indexes):
        summary, summarizer_metadata = _call_summarizer(summarizer, updated_content, 240)
        summary_info = {
            "updated": True,
            "method": "injected_llm",
            "summarizer": _callable_identity(summarizer),
            "metadata": summarizer_metadata,
        }
    changed = updated_content != existing
    timestamp = now_iso()
    if changed and not dry_run:
        write_text(updated_content, document_path)
        metadata = dict(metadata)
        metadata["summary"] = summary
        metadata["updated_at"] = timestamp
        metadata["update_count"] = int(metadata.get("update_count", 0)) + 1
        index[memory_id] = metadata
        write_json(index, paths["index"])
    result = {
        "status": "success",
        "memory_id": memory_id,
        "path": relative_path,
        "changed": changed,
        "dry_run": bool(dry_run),
        "conflict_policy": conflict_policy,
        "additions": additions,
        "duplicates": duplicates,
        "conflicts": conflicts,
        "replacement_count": len(replacement_indexes),
        "summary": summary_info,
        "updated_at": timestamp if changed else metadata.get("updated_at"),
    }
    output_dir = Path(outdir).resolve()
    write_json(result, output_dir / "b5_memory_update_result.json")
    return result


def _response_value(value: str | dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if isinstance(value, str):
        return value.strip(), {}
    if isinstance(value, dict) and isinstance(value.get("answer"), str):
        return value["answer"].strip(), {key: item for key, item in value.items() if key != "answer"}
    raise ValueError("responder must return text or {answer: text}")


def _bounded_memory_context(rows: list[dict[str, Any]], max_chars: int) -> tuple[str, bool]:
    parts: list[str] = []
    used = 0
    truncated = False
    for item in rows:
        opening = f'<memory id="{item["memory_id"]}">\n'
        closing = "\n</memory>"
        separator = "\n\n" if parts else ""
        overhead = len(separator) + len(opening) + len(closing)
        remaining = max_chars - used - overhead
        if remaining <= 0:
            truncated = True
            break
        content = str(item.get("content", ""))
        selected = content[:remaining]
        truncated = truncated or len(selected) < len(content)
        part = separator + opening + selected + closing
        parts.append(part)
        used += len(part)
        if len(selected) < len(content):
            break
    return "".join(parts), truncated


def analyze_bad_memory_ab(
    config_path: str | Path,
    query: str,
    bad_memory: str,
    top_k: int,
    outdir: str | Path,
    *,
    responder: Responder | None,
    evaluator: Evaluator | None = None,
    memory_ids: list[str] | None = None,
) -> dict[str, Any]:
    output_dir = Path(outdir).resolve()
    if not isinstance(bad_memory, str) or not bad_memory.strip():
        raise ValueError("bad_memory must be a non-empty string")
    paths = _memory_paths(config_path)
    if memory_ids is not None:
        if not isinstance(memory_ids, list) or not memory_ids or not all(
            isinstance(item, str) and item for item in memory_ids
        ):
            raise ValueError("memory_ids must be a non-empty string list when provided")
        index = _read_index(paths["index"])
        selected_documents = []
        for memory_id in memory_ids:
            metadata = index.get(memory_id)
            if not isinstance(metadata, dict):
                raise KeyError(f"memory_id does not exist: {memory_id}")
            selected_documents.append(_load_document(paths, memory_id, metadata))
        retrieval = {
            "status": "success",
            "mode": "explicit_memory_ids",
            "results": selected_documents,
        }
    else:
        retrieval = search_memory_keyword(config_path, query, top_k, include_content=True)
    baseline_context, context_truncated = _bounded_memory_context(
        retrieval["results"],
        paths["max_chars"],
    )
    injected_context = baseline_context + f"\n\n<memory id=\"injected_bad_memory\">\n{bad_memory}\n</memory>"
    if responder is None:
        result = {
            "status": "not_run",
            "query": query,
            "retrieval": {key: value for key, value in retrieval.items() if key != "results"},
            "baseline_answer": None,
            "injected_answer": None,
            "error": _error("AB_RESPONDER_NOT_CONFIGURED", "No model-backed responder was injected."),
            "fallback_used": False,
        }
        write_json(result, output_dir / "b5_bad_memory_ab_analysis.json")
        return result
    started = perf_counter()
    try:
        baseline_answer, baseline_metadata = _response_value(responder(query, baseline_context))
        injected_answer, injected_metadata = _response_value(responder(query, injected_context))
        if not baseline_answer or not injected_answer:
            raise ValueError("responder returned an empty answer")
        evaluation = evaluator(query, baseline_answer, injected_answer, bad_memory) if evaluator else {
            "status": "not_run",
            "reason": "No evaluator was injected; no correctness claim is made.",
        }
        if not isinstance(evaluation, dict):
            raise ValueError("evaluator must return an object")
    except Exception as exc:
        result = {
            "status": "error",
            "query": query,
            "baseline_answer": None,
            "injected_answer": None,
            "error": _error("AB_ANALYSIS_FAILED", exc),
            "fallback_used": False,
        }
    else:
        result = {
            "status": "success",
            "query": query,
            "bad_memory": bad_memory,
            "retrieved_memory_ids": [item["memory_id"] for item in retrieval["results"]],
            "retrieval_mode": retrieval.get("mode"),
            "context_budget_chars": paths["max_chars"],
            "baseline_context_chars": len(baseline_context),
            "baseline_context_truncated": context_truncated,
            "baseline": {
                "context": baseline_context,
                "answer": baseline_answer,
                "metadata": baseline_metadata,
            },
            "with_bad_memory": {
                "context": injected_context,
                "answer": injected_answer,
                "metadata": injected_metadata,
            },
            "observations": {
                "answer_changed": baseline_answer != injected_answer,
                "baseline_chars": len(baseline_answer),
                "injected_chars": len(injected_answer),
            },
            "evaluation": evaluation,
            "responder": _callable_identity(responder),
            "evaluator": _callable_identity(evaluator) if evaluator else None,
            "latency_ms": round((perf_counter() - started) * 1000, 3),
            "error": None,
            "fallback_used": False,
        }
    write_json(result, output_dir / "b5_bad_memory_ab_analysis.json")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run B5 advanced memory operations without fake fallbacks.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    keyword = subparsers.add_parser("keyword")
    keyword.add_argument("--query", required=True)
    keyword.add_argument("--top_k", type=int, default=3)
    embedding = subparsers.add_parser("embedding")
    embedding.add_argument("--query", required=True)
    embedding.add_argument("--top_k", type=int, default=3)
    summary = subparsers.add_parser("summarize")
    summary.add_argument("--memory_id", required=True)
    summary.add_argument("--summarizer", help="LLM summarizer as module:function")
    summary.add_argument("--max_chars", type=int, default=240)
    update = subparsers.add_parser("update")
    update.add_argument("--input", required=True, help="JSON with memory_id and updates")
    update.add_argument("--summarizer", help="Optional LLM summarizer as module:function")
    ab = subparsers.add_parser("ab")
    ab.add_argument("--query", required=True)
    ab.add_argument("--bad_memory", required=True, help="UTF-8 text file")
    ab.add_argument("--top_k", type=int, default=3)
    ab.add_argument("--responder", help="Model responder as module:function")
    ab.add_argument("--evaluator", help="Optional evaluator as module:function")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config_path = resolve_cli_path(args.config)
        outdir = resolve_cli_path(args.outdir)
        if args.command == "keyword":
            result = search_memory_keyword(config_path, args.query, args.top_k, include_content=False)
            write_json(result, outdir / "b5_keyword_search.json")
            result_path = outdir / "b5_keyword_search.json"
        elif args.command == "embedding":
            search_memory_embeddings(config_path, args.query, args.top_k, outdir)
            result_path = outdir / "b5_embedding_search.json"
        elif args.command == "summarize":
            summarizer = _load_callable(args.summarizer) if args.summarizer else None
            summarize_memory_document(
                config_path,
                args.memory_id,
                outdir,
                summarizer=summarizer,
                max_chars=args.max_chars,
            )
            result_path = outdir / "b5_memory_summary.json"
        elif args.command == "update":
            payload = read_json(resolve_cli_path(args.input))
            if not isinstance(payload, dict):
                raise ValueError("update input must be an object")
            summarizer = _load_callable(args.summarizer) if args.summarizer else None
            update_memory_document(
                config_path,
                payload["memory_id"],
                payload["updates"],
                outdir,
                conflict_policy=payload.get("conflict_policy", "record"),
                dry_run=bool(payload.get("dry_run", False)),
                summarizer=summarizer,
            )
            result_path = outdir / "b5_memory_update_result.json"
        else:
            responder = _load_callable(args.responder) if args.responder else None
            evaluator = _load_callable(args.evaluator) if args.evaluator else None
            analyze_bad_memory_ab(
                config_path,
                args.query,
                read_text(resolve_cli_path(args.bad_memory)),
                args.top_k,
                outdir,
                responder=responder,
                evaluator=evaluator,
            )
            result_path = outdir / "b5_bad_memory_ab_analysis.json"
        print(result_path)
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
