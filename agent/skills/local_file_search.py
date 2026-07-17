from __future__ import annotations

import os
import time
from pathlib import Path

from skills.core.context import current_context
from skills.core.errors import ErrorCode, SkillFault, normalize_exception
from skills.core.filesystem import resolve_under_root
from skills.search import normalize_text, rank_documents, tokenize


def _candidate_files(
    root: Path,
    extensions: set[str],
    deadline: float,
    max_entries: int,
    state: dict,
):
    """Yield files without materializing the complete directory tree."""

    pending = [root]
    while pending:
        if time.monotonic() >= deadline:
            state["timed_out"] = True
            return
        directory = pending.pop()
        try:
            iterator = os.scandir(directory)
        except OSError as exc:
            state["walk_errors"].append((directory, exc))
            continue
        with iterator:
            for entry in iterator:
                if time.monotonic() >= deadline:
                    state["timed_out"] = True
                    return
                state["entries_seen"] += 1
                if state["entries_seen"] > max_entries:
                    state["limit_reached"] = "max_entries"
                    return
                try:
                    if entry.is_symlink():
                        state["walk_errors"].append(
                            (
                                Path(entry.path),
                                SkillFault(
                                    ErrorCode.PERMISSION_DENIED,
                                    "symbolic links are not indexed",
                                ),
                            )
                        )
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False) and Path(entry.name).suffix.casefold() in extensions:
                        yield Path(entry.path)
                except OSError as exc:
                    state["walk_errors"].append((Path(entry.path), exc))


def _snippet(text: str, query: str, matched_terms: list[str], radius: int) -> str:
    normalized = normalize_text(text)
    candidates = [normalize_text(query).strip(), *(normalize_text(term) for term in matched_terms)]
    positions = [normalized.find(term) for term in candidates if term and normalized.find(term) >= 0]
    anchor = min(positions) if positions else 0
    start = max(0, anchor - radius)
    end = min(len(text), anchor + radius)
    prefix = "..." if start else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end].replace("\n", " ").strip() + suffix


def _record_skip(records: list[dict], path: Path, root: Path, fault: SkillFault, maximum: int) -> None:
    if len(records) >= maximum:
        return
    try:
        display_path = path.relative_to(root).as_posix()
    except ValueError:
        display_path = path.name
    records.append({"path": display_path, "code": fault.code.value, "message": fault.message})


def local_file_search(
    query: str,
    root_dir: str = "docs",
    file_types: list[str] | None = None,
    top_k: int = 5,
    *,
    data_root: str | None = None,
) -> dict:
    """Search bounded local text files with mixed Chinese/English BM25 ranking."""

    context = current_context(data_root)
    limits = context.limits.search
    if not isinstance(query, str) or not query.strip():
        raise SkillFault(ErrorCode.PARAM_INVALID, "query must be a non-empty string")
    if len(query) > limits.max_query_chars:
        raise SkillFault(
            ErrorCode.RESOURCE_EXHAUSTED,
            f"query is too long (maximum {limits.max_query_chars} characters)",
        )
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise SkillFault(ErrorCode.PARAM_INVALID, "top_k must be a positive integer")
    if file_types is not None and (
        not isinstance(file_types, list) or not all(isinstance(item, str) for item in file_types)
    ):
        raise SkillFault(ErrorCode.PARAM_INVALID, "file_types must be an array of strings")
    extensions = file_types or ["txt", "md"]
    normalized_extensions = {f".{item.casefold().lstrip('.')}" for item in extensions}
    if not normalized_extensions or not normalized_extensions.issubset({".txt", ".md"}):
        raise SkillFault(ErrorCode.UNSUPPORTED_OPERATION, "local_file_search only supports txt and md")
    query_tokens = tokenize(query)
    if not query_tokens:
        raise SkillFault(ErrorCode.PARAM_INVALID, "query contains no searchable terms")

    search_root, resolved_data_root = resolve_under_root(root_dir, context.data_root, must_exist=True)
    if not search_root.is_dir():
        raise SkillFault(ErrorCode.FILE_NOT_FOUND, f"search directory not found: {root_dir}")

    deadline = time.monotonic() + limits.timeout_seconds
    documents: list[dict] = []
    skipped: list[dict] = []
    scanned_files = 0
    total_bytes = 0
    indexed_tokens = 0
    timed_out = False
    limit_reached: str | None = None
    walk_state = {
        "entries_seen": 0,
        "timed_out": False,
        "limit_reached": None,
        "walk_errors": [],
    }

    for discovered in _candidate_files(
        search_root,
        normalized_extensions,
        deadline,
        limits.max_entries,
        walk_state,
    ):
        if time.monotonic() >= deadline:
            timed_out = True
            break
        if scanned_files >= limits.max_files:
            limit_reached = "max_files"
            break
        scanned_files += 1
        try:
            path, _ = resolve_under_root(discovered, resolved_data_root, must_exist=True)
            size = path.stat().st_size
            if size > limits.max_file_bytes:
                raise SkillFault(
                    ErrorCode.FILE_TOO_LARGE,
                    f"file exceeds per-file search limit ({size} bytes)",
                )
            if total_bytes + size > limits.max_total_bytes:
                limit_reached = "max_total_bytes"
                break
            text = path.read_text(encoding="utf-8", errors="strict")
            total_bytes += size
        except Exception as exc:
            _record_skip(
                skipped,
                discovered,
                resolved_data_root,
                normalize_exception(exc),
                limits.max_skipped_records,
            )
            continue
        if time.monotonic() >= deadline:
            timed_out = True
            break
        tokens = tokenize(text)
        filename_tokens = tokenize(path.name)
        if indexed_tokens + len(tokens) + len(filename_tokens) > limits.max_index_tokens:
            limit_reached = "max_index_tokens"
            break
        indexed_tokens += len(tokens) + len(filename_tokens)
        documents.append(
            {
                "relative_path": path.relative_to(resolved_data_root).as_posix(),
                "filename": path.name,
                "filename_tokens": filename_tokens,
                "text": text,
                "tokens": tokens,
            }
        )

    timed_out = timed_out or bool(walk_state["timed_out"])
    limit_reached = limit_reached or walk_state["limit_reached"]
    for path, error in walk_state["walk_errors"]:
        _record_skip(skipped, path, resolved_data_root, normalize_exception(error), limits.max_skipped_records)

    applied_top_k = min(top_k, limits.max_top_k)
    try:
        ranked = rank_documents(query, documents, deadline=deadline)
    except TimeoutError:
        ranked = []
        timed_out = True
    results = []
    for item in ranked[:applied_top_k]:
        document = documents[item["document_index"]]
        results.append(
            {
                "path": document["relative_path"],
                "score": round(float(item["score"]), 6),
                "snippet": _snippet(document["text"], query, item["matched_terms"], limits.snippet_radius),
                "matched_terms": item["matched_terms"],
                "phrase_match": item["phrase_match"],
                "filename_match": item["filename_match"],
            }
        )
    return {
        "results": results,
        "total_scanned": scanned_files,
        "indexed_files": len(documents),
        "total_bytes": total_bytes,
        "indexed_tokens": indexed_tokens,
        "entries_seen": walk_state["entries_seen"],
        "timed_out": timed_out,
        "limit_reached": limit_reached,
        "skipped_files": skipped,
        "query_terms": sorted({token.split(":", 1)[-1] for token in query_tokens}),
        "applied_top_k": applied_top_k,
    }
