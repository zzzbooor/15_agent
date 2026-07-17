#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${AGENT_PYTHON:-/opt/conda/bin/python}"
OUTPUT_ROOT="${AGENT_ACCEPTANCE_OUT:-${PROJECT_ROOT}/outputs/acceptance}"
OUTPUT_ROOT="$(readlink -m -- "${OUTPUT_ROOT}")"

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "错误：Python 解释器不可用：${PYTHON_BIN}" >&2
  exit 1
fi

case "${OUTPUT_ROOT}/" in
  "${PROJECT_ROOT}/outputs/"*) ;;
  *)
    echo "错误：验收输出必须位于 ${PROJECT_ROOT}/outputs/ 内" >&2
    exit 1
    ;;
esac

safe_reset_dir() {
  local target
  target="$(readlink -m -- "$1")"
  case "${target}/" in
    "${OUTPUT_ROOT}/"*) ;;
    *)
      echo "拒绝清理非验收目录：${target}" >&2
      exit 1
      ;;
  esac
  rm -rf -- "${target}"
  mkdir -p -- "${target}"
}

normalize_owner() {
  if [[ "${EUID}" -eq 0 ]] && id zhz >/dev/null 2>&1; then
    chown -R zhz:zhz "${OUTPUT_ROOT}"
  fi
}

require_model_access() {
  if [[ ! -r /root/15/assignment_B/Qwen3.5-4B/config.json ]]; then
    echo "错误：当前用户不能读取本地模型。请使用 root 运行该验收阶段。" >&2
    exit 1
  fi
}

print_json() {
  "${PYTHON_BIN}" -m json.tool "$1"
}

run_unit() {
  local target="${OUTPUT_ROOT}/UNIT"
  safe_reset_dir "${target}"
  echo "[UNIT] 运行 Linux 全量单元、契约与安全测试"
  (
    cd "${PROJECT_ROOT}"
    "${PYTHON_BIN}" -m unittest discover -s tests -v
  ) 2>&1 | tee "${target}/unittest.log"
  echo "[UNIT] 通过，日志：${target}/unittest.log"
  normalize_owner
}

run_b2() {
  local target="${OUTPUT_ROOT}/B2"
  safe_reset_dir "${target}"
  echo "[B2] 基础 Skill、异常、复合 Skill、sandbox 与资源预算"
  (
    cd "${PROJECT_ROOT}/code"
    "${PYTHON_BIN}" verify_b2.py --outdir "${target}"
  )
  cat "${target}/acceptance_report.md"
  normalize_owner
}

run_b3() {
  require_model_access
  local target="${OUTPUT_ROOT}/B3"
  safe_reset_dir "${target}"
  echo "[B3] 契约联调、自动 Schema、缓存/重试及真实模型描述对比"
  (
    cd "${PROJECT_ROOT}/code"
    "${PYTHON_BIN}" verify_b3_b5.py --outdir "${target}/contracts"
    AGENT_HOOK_ARTIFACT_DIR="${target}/model_calls" \
      "${PYTHON_BIN}" b3_acceptance.py --outdir "${target}/real"
  )
  print_json "${target}/real/b3_acceptance_summary.json"
  normalize_owner
}

run_b4() {
  require_model_access
  local target="${OUTPUT_ROOT}/B4"
  safe_reset_dir "${target}"
  echo "[B4] 2模型×2绑定、多工具、token/延迟与 Plan-and-Execute"
  (
    cd "${PROJECT_ROOT}/code"
    "${PYTHON_BIN}" b4_acceptance.py --outdir "${target}"
  )
  print_json "${target}/b4_acceptance_summary.json"
  normalize_owner
}

run_b5() {
  require_model_access
  local target="${OUTPUT_ROOT}/B5"
  safe_reset_dir "${target}"
  echo "[B5] 真实向量、LLM 摘要、dry-run 更新与错误 Memory A/B"
  (
    cd "${PROJECT_ROOT}/code"
    AGENT_HOOK_ARTIFACT_DIR="${target}/model_calls" \
      "${PYTHON_BIN}" b5_acceptance.py --outdir "${target}"
  )
  print_json "${target}/b5_acceptance_summary.json"
  normalize_owner
}

run_full() {
  require_model_access
  local target="${OUTPUT_ROOT}/FULL"
  safe_reset_dir "${target}"
  echo "[FULL] B1→B5 真实 Agent 闭环"
  (
    cd "${PROJECT_ROOT}/code"
    "${PYTHON_BIN}" run_full_demo.py \
      --input "${PROJECT_ROOT}/data/acceptance/full_runtime.json" \
      --tools_config "${PROJECT_ROOT}/configs/tools.yaml" \
      --memory_config "${PROJECT_ROOT}/configs/memory.yaml" \
      --model_config "${PROJECT_ROOT}/configs/model.yaml" \
      --llm_mode native_tools \
      --model_profile qwen35_4b \
      --tool_binding native_tools \
      --outdir "${target}"
  )
  cat "${target}/demo_report.md"
  normalize_owner
}

show_summary() {
  echo "验收输出根目录：${OUTPUT_ROOT}"
  for item in UNIT B2 B3 B4 B5 FULL; do
    if [[ -d "${OUTPUT_ROOT}/${item}" ]]; then
      printf '%-5s %s\n' "${item}" "已生成"
    else
      printf '%-5s %s\n' "${item}" "未运行"
    fi
  done
  echo
  [[ -f "${OUTPUT_ROOT}/B2/acceptance_report.md" ]] && grep -E 'Status:|Passed:|Failed:' "${OUTPUT_ROOT}/B2/acceptance_report.md" || true
  [[ -f "${OUTPUT_ROOT}/B3/real/b3_acceptance_summary.json" ]] && grep -m1 '"status"' "${OUTPUT_ROOT}/B3/real/b3_acceptance_summary.json" || true
  [[ -f "${OUTPUT_ROOT}/B4/b4_acceptance_summary.json" ]] && grep -m1 '"status"' "${OUTPUT_ROOT}/B4/b4_acceptance_summary.json" || true
  [[ -f "${OUTPUT_ROOT}/B5/b5_acceptance_summary.json" ]] && grep -m1 '"status"' "${OUTPUT_ROOT}/B5/b5_acceptance_summary.json" || true
  [[ -f "${OUTPUT_ROOT}/FULL/trace.json" ]] && grep -m1 '"status"' "${OUTPUT_ROOT}/FULL/trace.json" || true
}

clean_outputs() {
  local stage="${1:-all}"
  local target
  case "${stage,,}" in
    all) target="${OUTPUT_ROOT}" ;;
    unit) target="${OUTPUT_ROOT}/UNIT" ;;
    b2) target="${OUTPUT_ROOT}/B2" ;;
    b3) target="${OUTPUT_ROOT}/B3" ;;
    b4) target="${OUTPUT_ROOT}/B4" ;;
    b5) target="${OUTPUT_ROOT}/B5" ;;
    full) target="${OUTPUT_ROOT}/FULL" ;;
    *)
      echo "未知清理阶段：${stage}" >&2
      exit 2
      ;;
  esac
  target="$(readlink -m -- "${target}")"
  case "${target}/" in
    "${PROJECT_ROOT}/outputs/"*) ;;
    *) echo "拒绝清理非 outputs 目录：${target}" >&2; exit 1 ;;
  esac
  rm -rf -- "${target}"
  mkdir -p -- "${OUTPUT_ROOT}"
  echo "已清理：${target}"
  normalize_owner
}

clean_interactive() {
  local target="${PROJECT_ROOT}/outputs/interactive"
  rm -rf -- "${target}"
  mkdir -p -- "${target}"
  echo "已清理交互演示输出：${target}"
  if [[ "${EUID}" -eq 0 ]] && id zhz >/dev/null 2>&1; then chown -R zhz:zhz "${target}"; fi
}

show_usage() {
  cat <<'EOF'
用法：
  ./acceptance.sh unit             运行 68 项轻量测试
  ./acceptance.sh b2               运行 B2 验收
  ./acceptance.sh b3               运行 B3 验收
  ./acceptance.sh b4               运行 B4 真实模型验收
  ./acceptance.sh b5               运行 B5 验收
  ./acceptance.sh full             运行完整 Agent 闭环
  ./acceptance.sh all              按顺序运行全部阶段
  ./acceptance.sh summary          查看当前结果摘要
  ./acceptance.sh clean [阶段]     清理 all/unit/b2/b3/b4/b5/full
  ./acceptance.sh clean-interactive 清理对话演示输出

每个阶段运行前会自动删除自己的上一份结果，不会不断生成新目录。
EOF
}

mkdir -p -- "${OUTPUT_ROOT}"

command_name="${1:-help}"
case "${command_name,,}" in
  unit) run_unit ;;
  b2) run_b2 ;;
  b3) run_b3 ;;
  b4) run_b4 ;;
  b5) run_b5 ;;
  full) run_full ;;
  all)
    require_model_access
    run_unit
    run_b2
    run_b3
    run_b4
    run_b5
    run_full
    show_summary
    ;;
  summary) show_summary ;;
  clean) clean_outputs "${2:-all}" ;;
  clean-interactive) clean_interactive ;;
  help|-h|--help) show_usage ;;
  *) echo "未知命令：${command_name}" >&2; show_usage; exit 2 ;;
esac
