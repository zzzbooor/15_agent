#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${AGENT_PYTHON:-/opt/conda/bin/python}"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/interactive/current"
PERSISTENT_MEMORY_ROOT="${AGENT_PERSISTENT_MEMORY_ROOT:-${PROJECT_ROOT}/persistent_memory/teacher_demo}"
MEMORY_MODE="persistent"
RESET_MEMORY=false

show_usage() {
  cat <<'EOF'
用法：
  ./start_agent_chat.sh                       教师演示（默认启用独立持久化 Memory）
  ./start_agent_chat.sh --persistent-memory   显式启用跨会话记忆
  ./start_agent_chat.sh --temporary-memory    使用临时只读 Memory，退出后不保留对话
  ./start_agent_chat.sh --reset-memory        清空独立持久化 Memory 后开始新会话
  ./start_agent_chat.sh --help                查看帮助

交互命令：/skills、/demo、/memory、/exit
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --persistent-memory) MEMORY_MODE="persistent" ;;
    --temporary-memory) MEMORY_MODE="temporary" ;;
    --reset-memory) RESET_MEMORY=true ;;
    --help|-h) show_usage; exit 0 ;;
    *) echo "未知参数：$1" >&2; show_usage >&2; exit 2 ;;
  esac
  shift
done

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "错误：Python 解释器不存在或不可执行：${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${PROJECT_ROOT}/code/agent_chat.py" ]]; then
  echo "错误：未找到 Agent 入口：${PROJECT_ROOT}/code/agent_chat.py" >&2
  exit 1
fi
if [[ "${RESET_MEMORY}" == true && "${MEMORY_MODE}" != "persistent" ]]; then
  echo "错误：--reset-memory 只能用于持久化模式。" >&2
  exit 2
fi

case "$(readlink -m -- "${OUTPUT_DIR}")/" in
  "${PROJECT_ROOT}/outputs/interactive/"*) ;;
  *) echo "拒绝使用非交互输出目录：${OUTPUT_DIR}" >&2; exit 1 ;;
esac
case "$(readlink -m -- "${PERSISTENT_MEMORY_ROOT}")/" in
  "${PROJECT_ROOT}/persistent_memory/"*) ;;
  *) echo "拒绝使用项目外的持久化 Memory：${PERSISTENT_MEMORY_ROOT}" >&2; exit 1 ;;
esac

rm -rf -- "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

if [[ "${MEMORY_MODE}" == "persistent" ]]; then
  if [[ "${RESET_MEMORY}" == true ]]; then
    rm -rf -- "${PERSISTENT_MEMORY_ROOT}"
  fi
  MEMORY_ROOT="${PERSISTENT_MEMORY_ROOT}"
  SAVE_MEMORY="global"
else
  MEMORY_ROOT="${OUTPUT_DIR}/temporary_memory"
  SAVE_MEMORY="none"
fi

mkdir -p "${MEMORY_ROOT}/global" "${MEMORY_ROOT}/conversations"

if [[ ! -f "${MEMORY_ROOT}/global/demo_memory.md" ]]; then
  cat > "${MEMORY_ROOT}/global/demo_memory.md" <<'EOF'
# 现场演示基础记忆

- 回答语言偏好：中文
- 演示计算表达式：23 * 17 + 9
- 说明：这是教师演示 Memory 的初始内容。
EOF
fi

if [[ ! -f "${MEMORY_ROOT}/memory_index.json" ]]; then
  cat > "${MEMORY_ROOT}/memory_index.json" <<'EOF'
{
  "mem_teacher_demo": {
    "memory_id": "mem_teacher_demo",
    "memory_type": "global",
    "title": "现场演示基础记忆",
    "summary": "回答使用中文；演示计算表达式为 23 * 17 + 9。",
    "path": "global/demo_memory.md"
  }
}
EOF
fi

if [[ ! -f "${MEMORY_ROOT}/memory_graph.json" ]]; then
  printf '{}\n' > "${MEMORY_ROOT}/memory_graph.json"
fi

cat > "${OUTPUT_DIR}/demo_memory.yaml" <<EOF
memory:
  root_dir: ${MEMORY_ROOT}
  global_memory_dir: global
  conversation_memory_dir: conversations
  index_path: memory_index.json
  max_memory_chars: 3500
  graph_path: memory_graph.json
  retrieval:
    keyword_top_k: 3
    keyword_min_score: 0.0
  embedding:
    enabled: false
    model_name_or_path: ""
    tokenizer_name_or_path: ""
    local_files_only: true
    trust_remote_code: false
    device: cpu
    max_length: 512
EOF

cat > "${OUTPUT_DIR}/demo_model.yaml" <<EOF
runtime:
  default_mode: native_tools
  default_profile: qwen35_4b
  default_binding: native_tools

profiles:
  qwen35_4b:
    backend: transformers
    family: qwen3.5
    native_parser: qwen35_xml
    model_name_or_path: /root/15/assignment_B/Qwen3.5-4B
    tokenizer_name_or_path: /root/15/assignment_B/Qwen3.5-4B
    local_files_only: true
    trust_remote_code: true
    torch_dtype: bfloat16
    device: cuda
    device_map: null
    max_input_tokens: 4096
    generation:
      max_new_tokens: 384
      do_sample: false
      stop_after_tool_call: true

context:
  max_input_tokens: 4096

prompts:
  native_tools: ${PROJECT_ROOT}/prompts/b4_native_tools.txt
EOF

cat > "${OUTPUT_DIR}/session_mode.json" <<EOF
{
  "memory_mode": "${MEMORY_MODE}",
  "memory_root": "${MEMORY_ROOT}",
  "save_memory": "${SAVE_MEMORY}"
}
EOF

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1

SESSION_ID="teacher_demo_$(date +%Y%m%d_%H%M%S)_$$"

echo "============================================================"
echo "  B方向本地 Agent：B2 交互 + B1/B3/B4/B5 自动汇总"
echo "============================================================"
echo "Memory 模式：${MEMORY_MODE}"
echo "Memory 目录：${MEMORY_ROOT}"
if [[ "${MEMORY_MODE}" == "persistent" ]]; then
  echo "本轮对话会保存到独立持久化 Memory，退出重启后可按问题相关性检索。"
else
  echo "本轮使用临时 Memory，退出后不会保留对话。"
fi
echo
echo "请在 User> 后依次输入下面五条 B2 演示指令："
echo "1. 请使用计算器精确计算 23 * 17 + 9。"
echo "2. 请读取 docs/agent_intro.txt，并总结三条要点。"
echo "3. 请在 docs 目录搜索与 tool calling 有关的本地资料，并总结最相关内容。"
echo "4. 请分析 tables/results.csv，告诉我行数、列数和主要数值统计。"
echo "5. 请把‘模型负责决策，工具负责执行，记忆提供上下文’转换为 Markdown 列表并保存为 teacher_summary.md。"
echo
echo "跨会话记忆测试：退出前告诉 Agent 一个口令；重新运行本脚本后询问该口令。"
echo "命令：/skills 查看能力，/demo 重看问题，/memory 查看已使用的 Memory ID，/exit 退出。"
echo

cd "${PROJECT_ROOT}/code"
"${PYTHON_BIN}" agent_chat.py \
  --tools_config "${PROJECT_ROOT}/configs/tools.yaml" \
  --memory_config "${OUTPUT_DIR}/demo_memory.yaml" \
  --model_config "${OUTPUT_DIR}/demo_model.yaml" \
  --system_prompt_path "${PROJECT_ROOT}/prompts/local_tool_agent.txt" \
  --llm_mode native_tools \
  --model_profile qwen35_4b \
  --tool_binding native_tools \
  --decision_strategy react \
  --toolset basic_tools \
  --max_turns 4 \
  --save_memory "${SAVE_MEMORY}" \
  --selected_memory_ids "mem_teacher_demo" \
  --conversation_id "${SESSION_ID}" \
  --outdir "${OUTPUT_DIR}"

if find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'turn_*' -print -quit | grep -q .; then
  echo
  echo "============================================================"
  echo "  对话结束：按 B1/B2/B3/B4/B5 汇总本次真实运行"
  echo "============================================================"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/code/show_demo_internals.py" --session "${OUTPUT_DIR}"
else
  echo "本次没有执行任务，因此未生成内部汇总。"
fi

if [[ "${EUID}" -eq 0 ]] && id zhz >/dev/null 2>&1; then
  chown -R zhz:zhz "${OUTPUT_DIR}"
  if [[ "${MEMORY_MODE}" == "persistent" ]]; then
    chown -R zhz:zhz "${PERSISTENT_MEMORY_ROOT}"
  fi
fi

echo
echo "再次运行 ./start_agent_chat.sh 会覆盖交互输出，但保留独立持久化 Memory。"
echo "清理交互输出：./acceptance.sh clean-interactive"
echo "重置持久化记忆：./start_agent_chat.sh --reset-memory"
