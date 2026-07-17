from __future__ import annotations

import argparse
import json
import re
import sys
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from common.io_utils import append_jsonl, read_json, read_text, read_yaml, write_json, write_text
from common.logging_utils import now_iso
from common.path_utils import resolve_cli_path, resolve_from_file


# ==========================================
# 1. B5 独立本地 LLM 引擎 (100% 复刻 B4 安全加载逻辑)
# ==========================================

_MODEL_CACHE: dict[tuple, Tuple[Any, Any]] = {}

def _dtype_value(torch_module: Any, configured: str) -> Any:
    """完全复刻 B4 的 dtype 安全映射逻辑"""
    if configured == "auto": return "auto"
    mapping = {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }
    if configured not in mapping:
        raise ValueError(f"unsupported torch_dtype: {configured}")
    return mapping[configured]

def _load_local_llm_bundle(model_path: str, tokenizer_path: str) -> Tuple[Any, Any]:
    """安全加载本地模型，带缓存机制，绝不联网"""
    cache_key = (str(model_path), str(tokenizer_path), "local_true", "trust_true", "bfloat16", "auto")
    if cache_key in _MODEL_CACHE:
        print("[B5] Model cache hit", file=sys.stderr, flush=True)
        return _MODEL_CACHE[cache_key]

    print(f"[B5] Loading independent local LLM: {model_path}...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # 严格对齐 B4：local_files_only=True 是解决 Repo id 报错的关键
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=_dtype_value(torch, "bfloat16"),
        device_map="auto"
    )
    model.eval()
    _MODEL_CACHE[cache_key] = (tokenizer, model)
    print("[B5] Local LLM loaded successfully.", flush=True)
    return tokenizer, model


class LocalLLMEngine:
    def __init__(self, model_path: str, tokenizer_path: str):
        self.tokenizer, self.model = _load_local_llm_bundle(model_path, tokenizer_path)
        
    def generate(self, prompt: str, max_new_tokens: int = 1024) -> str:
        messages = [{"role": "user", "content": prompt}]
        # 对齐 B4：使用 chat_template 并关闭 thinking 模式
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, 
                temperature=1.0, pad_token_id=self.tokenizer.eos_token_id
            )
        new_tokens = outputs[0][inputs.input_ids.shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def get_embedding(self, text: str) -> List[float]:
        """利用 Qwen 最后一层 Hidden States 提取向量"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
        last_hidden = outputs.hidden_states[-1]
        mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
        sum_emb = torch.sum(last_hidden * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_emb = sum_emb / sum_mask
        
        norm = torch.norm(mean_emb, dim=1, keepdim=True)
        return (mean_emb / norm)[0].cpu().tolist()


def _get_llm_engine(paths: dict) -> LocalLLMEngine:
    model_path = paths.get("llm_model_path")
    tokenizer_path = paths.get("llm_tokenizer_path", model_path)
    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(f"Local LLM path not found: {model_path}. Check memory.yaml")
    return LocalLLMEngine(model_path, tokenizer_path)


# ==========================================
# 2. 核心存储管理 (Index + Vector/LLM Graphs)
# ==========================================

class MemoryStore:
    def __init__(self, paths: dict):
        self.paths = paths
        self.index_path = paths["index"]
        self.graph_path = paths["graph_path"]
        self.index = read_json(self.index_path) if self.index_path.exists() else {}
        self.graphs = read_json(self.graph_path) if self.graph_path.exists() else {"vector_graph": {}, "llm_graph": {}}
        if "vector_graph" not in self.graphs: self.graphs["vector_graph"] = {}
        if "llm_graph" not in self.graphs: self.graphs["llm_graph"] = {}
        self._sync_embeddings()
        self._sync_vector_graph()

    def _save(self):
        write_json(self.index, self.index_path)
        write_json(self.graphs, self.graph_path)

    def _sync_embeddings(self):
        updated = False
        engine = None
        for mid, meta in self.index.items():
            if "embedding" not in meta or not meta["embedding"]:
                print(f"[B5] Generating embedding for missing memory: {mid} (Using Local Qwen)")
                text = meta.get("summary", "") or meta.get("title", "")
                if not text:
                    doc_path = (self.paths["root"] / meta["path"]).resolve()
                    if doc_path.exists(): text = read_text(doc_path)[:500]
                try:
                    if engine is None: engine = _get_llm_engine(self.paths)
                    meta["embedding"] = engine.get_embedding(text)
                    updated = True
                except Exception as e:
                    print(f"[B5] Warning: Failed to get embedding for {mid}: {e}")
        if updated: self._save()

    def _sync_vector_graph(self):
        valid_ids = [mid for mid in self.index.keys() if self.index[mid].get("embedding")]
        if not valid_ids: return
        matrix = np.array([self.index[mid]["embedding"] for mid in valid_ids], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix_norm = matrix / np.where(norms == 0, 1, norms)
        sims = np.dot(matrix_norm, matrix_norm.T)
        for i, mid in enumerate(valid_ids):
            sims[i, i] = -1.0
            top_k = min(4, len(valid_ids) - 1)
            if top_k > 0:
                top_indices = np.argsort(sims[i])[-top_k:]
                self.graphs["vector_graph"][mid] = [valid_ids[idx] for idx in top_indices]
        self._save()

    def _call_llm_score(self, text_a: str, text_b: str) -> int:
        prompt = f"""你是一个记忆关联分析专家。请评估以下两段记忆文本的关联程度。
请针对以下5个指标进行打分（满足得1分，不满足得0分）：
1. 因果关系 2. 时序邻近 3. 主题一致 4. 内容强相关 5. 相同主体
记忆A：{text_a[:800]}
记忆B：{text_b[:800]}
请直接输出JSON：{{"score": <总分0-5>, "reason": "<简短理由>"}}"""
        try:
            engine = _get_llm_engine(self.paths)
            raw = engine.generate(prompt, max_new_tokens=256)
            match = re.search(r'"score"\s*:\s*(\d+)', raw)
            return int(match.group(1)) if match else 0
        except Exception as e:
            print(f"[B5] Warning: LLM score failed: {e}")
            return 0

    def _update_llm_graph_for_new_node(self, new_id: str, new_text: str):
        old_ids = [mid for mid in self.index.keys() if mid != new_id and self.index[mid].get("embedding")]
        if not old_ids: return
        print(f"[B5] Calculating LLM graph edges for {new_id} against {len(old_ids)} existing memories...")
        scores = []
        for old_id in old_ids:
            old_meta = self.index[old_id]
            old_text = old_meta.get("summary", "") or old_meta.get("title", "")
            if not old_text:
                doc_path = (self.paths["root"] / old_meta["path"]).resolve()
                if doc_path.exists(): old_text = read_text(doc_path)[:500]
            scores.append((old_id, self._call_llm_score(new_text, old_text)))
        scores.sort(key=lambda x: x[1], reverse=True)
        top_4 = [item[0] for item in scores[:4]]
        self.graphs["llm_graph"][new_id] = top_4
        for n_id in top_4:
            if n_id not in self.graphs["llm_graph"]: self.graphs["llm_graph"][n_id] = []
            if new_id not in self.graphs["llm_graph"][n_id]: self.graphs["llm_graph"][n_id].append(new_id)
        self._save()

    def add_memory(self, memory_id: str, text_for_embedding: str, text_for_llm: str):
        try:
            engine = _get_llm_engine(self.paths)
            self.index[memory_id]["embedding"] = engine.get_embedding(text_for_embedding)
        except Exception as e:
            print(f"[B5] Warning: Embedding failed for {memory_id}: {e}")
        self._sync_vector_graph()
        self._update_llm_graph_for_new_node(memory_id, text_for_llm)

    def search_vector(self, query_vec: List[float], top_k: int = 5) -> List[str]:
        valid_ids = [mid for mid in self.index.keys() if self.index[mid].get("embedding")]
        if not valid_ids: return []
        matrix = np.array([self.index[mid]["embedding"] for mid in valid_ids], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix_norm = matrix / np.where(norms == 0, 1, norms)
        q = np.array(query_vec, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0: q = q / q_norm
        sims = np.dot(matrix_norm, q)
        top_indices = np.argsort(sims)[-top_k:][::-1]
        return [valid_ids[idx] for idx in top_indices]

    def search_graph(self, seed_id: str, k_hops: int, graph_type: str) -> List[str]:
        graph_dict = self.graphs.get(f"{graph_type}_graph", {})
        if seed_id not in self.index: return []
        visited = {seed_id}
        queue = [(seed_id, 0)]
        while queue:
            node, depth = queue.pop(0)
            if depth < k_hops:
                for neighbor in graph_dict.get(node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
        return list(visited)


# ==========================================
# 3. 核心业务逻辑
# ==========================================

def _memory_paths(config_path: str | Path) -> dict:
    path = Path(config_path).resolve()
    config = read_yaml(path)
    memory = config["memory"]
    root = resolve_from_file(memory["root_dir"], path)
    return {
        "root": root, "global": root / memory["global_memory_dir"],
        "conversations": root / memory["conversation_memory_dir"],
        "index": root / memory["index_path"], "max_chars": memory["max_memory_chars"],
        "graph_path": resolve_from_file(memory.get("graph_path", "../memory/memory_graph.json"), path),
        "llm_model_path": memory.get("llm_model_path"),
        "llm_tokenizer_path": memory.get("llm_tokenizer_path"),
    }

def _safe_conversation_id(conversation_id: str) -> str:
    if not isinstance(conversation_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", conversation_id):
        raise ValueError("conversation_id may only contain letters, numbers, dot, underscore, and hyphen")
    return conversation_id

def manage_length(docs_content: List[str], max_chars: int, mode: str, paths: dict) -> str:
    if mode == "llm_compress":
        full_text = "\n\n---\n\n".join([c for c in docs_content if c])
        if not full_text: return ""
        if len(full_text) <= max_chars: return full_text
        prompt = f"请将以下文本压缩到 {max_chars} 字符以内，保留核心事实与逻辑。\n文本：\n{full_text}\n压缩后的文本："
        engine = _get_llm_engine(paths)
        return engine.generate(prompt, max_new_tokens=max_chars + 200)[:max_chars]
    else:
        res, total = [], 0
        for doc in docs_content:
            if not doc: continue
            if total + len(doc) > max_chars:
                res.append(doc[:max_chars - total])
                break
            res.append(doc)
            total += len(doc)
        return "\n\n".join(res)

def load_memory(config_path: str, retrieval_mode: str, query: str, selected_ids: list,
                use_global: bool, top_k: int, k_hops: int, graph_type: str, length_mode: str, outdir: str):
    paths = _memory_paths(config_path)
    store = MemoryStore(paths)
    target_ids = []
    if retrieval_mode == "id":
        if use_global: target_ids.extend([k for k, v in store.index.items() if v.get("memory_type") == "global"])
        target_ids.extend(selected_ids)
    elif retrieval_mode == "vector":
        if not query: raise ValueError("Vector retrieval requires --query")
        q_vec = _get_llm_engine(paths).get_embedding(query)
        target_ids = store.search_vector(q_vec, top_k=top_k)
    elif retrieval_mode == "graph":
        if not query: raise ValueError("Graph retrieval requires --query")
        q_vec = _get_llm_engine(paths).get_embedding(query)
        seeds = store.search_vector(q_vec, top_k=1)
        if seeds: target_ids = store.search_graph(seeds[0], k_hops=k_hops, graph_type=graph_type)
    target_ids = list(dict.fromkeys(target_ids))
    docs_content, errors = [], []
    for mid in target_ids:
        meta = store.index.get(mid)
        if not meta: errors.append({"memory_id": mid, "error": "Not found"})
        else:
            doc_path = (paths["root"] / meta["path"]).resolve()
            docs_content.append(read_text(doc_path) if doc_path.exists() else "")
    final_content = manage_length(docs_content, paths["max_chars"], length_mode, paths)
    result = {
        "status": "success" if not errors else "partial", "retrieval_mode": retrieval_mode,
        "graph_type": graph_type if retrieval_mode == "graph" else "N/A",
        "target_ids": target_ids, "final_content": final_content, "total_chars": len(final_content), "errors": errors
    }
    if outdir: write_json(result, Path(outdir) / "selected_memory.json")
    return result

def save_memory(config_path: str, conversation_id: str, save_type: str,
                messages_path: str, trace_path: str, answer_path: str, outdir: str):
    conversation_id = _safe_conversation_id(conversation_id)
    paths = _memory_paths(config_path)
    store = MemoryStore(paths)
    messages, trace = read_json(messages_path), read_json(trace_path)
    answer = read_text(answer_path).strip()
    now = now_iso()
    memory_id = f"mem_{save_type}_{conversation_id}"
    target_dir = paths["conversations"] if save_type == "conversation" else paths["global"]
    relative_dir = "conversations" if save_type == "conversation" else "global"
    target_path = Path(target_dir) / f"{conversation_id}.md"
    relative_path = f"{relative_dir}/{conversation_id}.md"
    markdown = (f"# {save_type.title()} {conversation_id}\n\n- memory_id: `{memory_id}`\n- created_at: `{now}`\n\n## Final Answer\n\n{answer}\n\n## Messages\n\n```json\n{json.dumps(messages, ensure_ascii=False, indent=2)}\n```\n\n## Trace\n\n```json\n{json.dumps(trace, ensure_ascii=False, indent=2)}\n```\n")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    write_text(markdown, target_path)
    existing = store.index.get(memory_id, {})
    store.index[memory_id] = {
        "memory_id": memory_id, "memory_type": save_type, "title": f"{save_type.title()} {conversation_id}",
        "summary": answer[:200], "path": relative_path, "conversation_id": conversation_id, 
        "created_at": existing.get("created_at", now), "updated_at": now
    }
    store.add_memory(memory_id, answer[:200], answer)
    result = {"status": "success", "memory_id": memory_id, "path": relative_path}
    if outdir: write_json(result, Path(outdir) / "saved_memory.json")
    return result

def self_check_memory(config_path: str, memory_id: str, outdir: str):
    paths = _memory_paths(config_path)
    store = MemoryStore(paths)
    meta = store.index.get(memory_id)
    if not meta: return {"status": "error", "message": "Not found"}
    content = read_text((paths["root"] / meta["path"]).resolve())
    prompt = f"评估以下记忆的质量(准确性、完整性、相关性)，输出0-100的置信度和理由。\n格式：\n置信度: <分数>\n理由: <理由>\n\n内容：\n{content}"
    llm_output = _get_llm_engine(paths).generate(prompt, max_new_tokens=512)
    confidence = 0
    for line in llm_output.split('\n'):
        if '置信度' in line or 'confidence' in line.lower():
            nums = re.findall(r'\d+', line)
            if nums: confidence = int(nums[0]); break
    result = {"status": "success", "memory_id": memory_id, "confidence": confidence, "llm_reasoning": llm_output}
    if outdir: write_json(result, Path(outdir) / "self_check_result.json")
    return result


# ==========================================
# 4. CLI 入口
# ==========================================

def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1", "yes"}: return True
    if lowered in {"false", "0", "no"}: return False
    raise argparse.ArgumentTypeError("expected true or false")

def build_parser():
    parser = argparse.ArgumentParser(description="Independent B5: Dual Graph Memory with Local Qwen")
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--select_memory_ids", nargs="*", default=[])
    parser.add_argument("--use_global_memory", type=parse_bool, default=False)
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--save_type", choices=["conversation", "global"])
    parser.add_argument("--save_input_path")
    parser.add_argument("--retrieval_mode", choices=["id", "vector", "graph"], default="id")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--k_hops", type=int, default=1)
    parser.add_argument("--graph_type", choices=["vector", "llm"], default="vector")
    parser.add_argument("--length_mode", choices=["truncate", "llm_compress"])
    parser.add_argument("--self_check_id")
    return parser

def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        config_path = resolve_cli_path(args.config)
        outdir = resolve_cli_path(args.outdir)
        if args.length_mode is None:
            args.length_mode = "llm_compress" if args.retrieval_mode in ["vector", "graph"] else "truncate"
        if args.self_check_id:
            self_check_memory(str(config_path), args.self_check_id, str(outdir))
            print(outdir / "self_check_result.json")
        elif args.save_type:
            if not args.save_input_path: raise ValueError("--save_input_path is required")
            input_path = resolve_cli_path(args.save_input_path)
            payload = read_json(input_path)
            base = input_path.parent
            save_memory(str(config_path), payload["conversation_id"], args.save_type,
                        str((base / payload["messages_path"]).resolve()),
                        str((base / payload["trace_path"]).resolve()),
                        str((base / payload["answer_path"]).resolve()), str(outdir))
            print(outdir / "saved_memory.json")
        else:
            load_memory(str(config_path), args.retrieval_mode, args.query, args.select_memory_ids,
                        args.use_global_memory, args.top_k, args.k_hops, args.graph_type,
                        args.length_mode, str(outdir))
            print(outdir / "selected_memory.json")
        return 0
    except Exception as e:
        print(f"fatal: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())