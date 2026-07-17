from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from b1_agent_runtime import run_agent
from b3_advanced import execute_with_retry_cache_stats
from b3_tool_layer import execute_tool_calls, get_auto_tools_schema
from b5_advanced import search_memory_embeddings, update_memory_document
from b5_memory import load_memory, search_memory_keyword
from common.io_utils import write_json
from common.path_utils import PROJECT_ROOT, resolve_cli_path
from skills.core.invoker import invoke_skill


def _temporary_memory(root: Path) -> Path:
    memory_root = root / "memory"
    (memory_root / "global").mkdir(parents=True)
    (memory_root / "conversations").mkdir(parents=True)
    (memory_root / "global" / "agent.md").write_text(
        "Agent tools execute calculator and file tasks.", encoding="utf-8"
    )
    (memory_root / "global" / "travel.md").write_text(
        "Travel notes discuss trains and hotels.", encoding="utf-8"
    )
    (memory_root / "conversations" / "selected.md").write_text(
        "Explicit conversation memory.", encoding="utf-8"
    )
    index = {
        "global_agent": {
            "memory_id": "global_agent",
            "memory_type": "global",
            "title": "Agent tools",
            "summary": "calculator and file tools",
            "path": "global/agent.md",
        },
        "global_travel": {
            "memory_id": "global_travel",
            "memory_type": "global",
            "title": "Travel",
            "summary": "trains and hotels",
            "path": "global/travel.md",
        },
        "conversation_selected": {
            "memory_id": "conversation_selected",
            "memory_type": "conversation",
            "title": "Selected",
            "summary": "explicit",
            "path": "conversations/selected.md",
        },
    }
    (memory_root / "memory_index.json").write_text(json.dumps(index), encoding="utf-8")
    config_path = root / "memory.json"
    config_path.write_text(
        json.dumps(
            {
                "memory": {
                    "root_dir": "memory",
                    "global_memory_dir": "global",
                    "conversation_memory_dir": "conversations",
                    "index_path": "memory_index.json",
                    "max_memory_chars": 2000,
                    "retrieval": {"keyword_top_k": 1, "keyword_min_score": 0.0},
                }
            }
        ),
        encoding="utf-8",
    )
    return config_path


def run_verification(project_root: Path) -> dict[str, Any]:
    tools_config = project_root / "configs" / "tools.yaml"
    checks: list[dict[str, Any]] = []
    schema = get_auto_tools_schema(str(tools_config), "basic_tools")
    by_name = {item["function"]["name"]: item["function"] for item in schema}
    checks.append(
        {
            "name": "b3_auto_schema_types",
            "passed": by_name["file_reader"]["parameters"]["properties"]["max_chars"]["type"]
            == "integer"
            and by_name["table_analyzer"]["parameters"]["properties"]["describe"]["type"]
            == "boolean",
        }
    )
    direct = invoke_skill("calculator", {"expression": "23 * 17 + 9"})
    with tempfile.TemporaryDirectory(prefix="verify_b3_b5_") as temporary:
        root = Path(temporary)
        b3_output = root / "b3"
        messages = execute_tool_calls(
            [{"id": "verify_calc", "name": "calculator", "args": {"expression": "23 * 17 + 9"}}],
            str(tools_config),
            "basic_tools",
            str(b3_output),
        )
        via_b3 = json.loads(messages[0]["content"])
        checks.append(
            {
                "name": "b3_b2_invoker_integration",
                "passed": via_b3["status"] == direct["status"]
                and via_b3["output"] == direct["output"]
                and via_b3["error"] == direct["error"],
            }
        )
        stats = execute_with_retry_cache_stats(
            [
                {"id": "cache_1", "name": "calculator", "args": {"expression": "2+2"}},
                {"id": "cache_2", "name": "calculator", "args": {"expression": "2+2"}},
            ],
            tools_config,
            "basic_tools",
            b3_output / "advanced",
        )
        checks.append(
            {
                "name": "b3_success_only_cache",
                "passed": stats["success_count"] == 2 and stats["cache_hit_count"] == 1,
            }
        )
        memory_config = _temporary_memory(root / "b5_fixture")
        selected = load_memory(
            str(memory_config), ["conversation_selected"], True, "calculator tools", str(root / "b5_load")
        )
        selected_ids = [item["memory_id"] for item in selected["selected_memory_docs"]]
        checks.append(
            {
                "name": "b5_query_top_k_and_explicit_id",
                "passed": selected_ids == ["global_agent", "conversation_selected"],
            }
        )
        keyword = search_memory_keyword(memory_config, "hotels", 5, include_content=False)
        checks.append(
            {
                "name": "b5_positive_keyword_ranking",
                "passed": [item["memory_id"] for item in keyword["results"]] == ["global_travel"]
                and all(item["score"] > 0 for item in keyword["results"]),
            }
        )

        def embedder(texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] if "calculator" in text.casefold() else [0.0, 1.0] for text in texts]

        embedding = search_memory_embeddings(
            memory_config,
            "calculator",
            1,
            root / "b5_embedding",
            embedder=embedder,
        )
        checks.append(
            {
                "name": "b5_injected_embedding_no_hash_fallback",
                "passed": embedding["status"] == "success"
                and embedding["results"][0]["memory_id"] == "global_agent"
                and embedding["fallback_used"] is False,
            }
        )
        update = update_memory_document(
            memory_config,
            "conversation_selected",
            ["new verified fact"],
            root / "b5_update",
            dry_run=True,
        )
        checks.append(
            {
                "name": "b5_update_dry_run",
                "passed": update["changed"] is True
                and update["dry_run"] is True
                and "new verified fact"
                not in (root / "b5_fixture" / "memory" / "conversations" / "selected.md").read_text(
                    encoding="utf-8"
                ),
            }
        )
        runtime_input = root / "runtime_input.json"
        write_json(
            {
                "conversation_id": "verify_b3_b5_integration",
                "execution_mode": "integrated",
                "user_input": "Read docs/agent_intro.txt and summarize it.",
                "system_prompt_path": str(project_root / "prompts" / "local_tool_agent.txt"),
                "selected_memory_ids": [],
                "use_global_memory": False,
                "toolset": "basic_tools",
                "max_turns": 2,
                "save_memory": "none",
            },
            runtime_input,
        )
        integrated = run_agent(
            str(runtime_input),
            str(tools_config),
            str(memory_config),
            str(project_root / "configs" / "model.yaml"),
            str(root / "b1_integration"),
            "mock",
        )
        integrated_messages = json.loads(
            (root / "b1_integration" / "messages.json").read_text(encoding="utf-8")
        )
        checks.append(
            {
                "name": "b1_b3_b5_mock_integration",
                "passed": integrated["status"] == "success"
                and [message["role"] for message in integrated_messages]
                == ["system", "user", "assistant", "tool", "assistant"],
            }
        )
    passed_count = sum(bool(item["passed"]) for item in checks)
    return {
        "status": "success" if passed_count == len(checks) else "error",
        "check_count": len(checks),
        "passed_count": passed_count,
        "failed_count": len(checks) - passed_count,
        "checks": checks,
        "production_memory_modified": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify B3/B5 contracts using temporary memory state.")
    parser.add_argument("--outdir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_verification(PROJECT_ROOT)
        if args.outdir:
            outdir = resolve_cli_path(args.outdir)
            write_json(report, outdir / "b3_b5_verification.json")
            print(outdir / "b3_b5_verification.json")
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "success" else 1
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
