# 团队项目 README（B 方向本地 Agent）

> 说明：本 README 描述 B 方向团队项目“本地 Agent”的整体设计、模块协作、依赖资源与复现方式。项目在第二台服务器上以本地大模型、受限本地工具与文件式 Memory 实现完整闭环，不依赖联网检索、外部 Web API、LoRA/PEFT、DPO 或任何训练/后训练流程。

---

## 1. 项目概述

### 1.1 项目名称

`B 方向本地 Agent（Local Tool-using Agent with Local LLM & File Memory）`

### 1.2 项目目标

本项目面向“在受限本地环境下构建一个可用的工具型对话 Agent”的任务场景，要解决的核心问题是：

- 在不联网、不调用外部闭源 API 的前提下，让本地开源大模型能够根据用户请求自主选择并调用受限本地工具（计算器、文件读取、本地检索、表格分析、格式转换、沙盒代码执行等）；
- 在多轮对话中维护可检索的文件式 Memory，支持跨会话相关性检索与注入；
- 把“模型决策—工具执行—记忆上下文”三件事组合成一个可验收、可复现、可演示的完整系统，并给出无大模型依赖的契约测试与真实模型验收两条路径。

最终实现的核心能力：一次用户请求 → 本地大模型生成标准 AIMessage（含 0 或多个 tool_calls）→ B3 校验并执行工具 → B2 Skill 返回结构化结果 → 模型基于工具结果给出最终回答，并可选地把本次对话保存进 Memory 供下次检索。

### 1.3 当前完成情况

| 类型 | 完成情况 |
|---|---|
| 基础要求 | B1 消息循环、B2 五个基础 Skill、B3 Tool Schema 生成/校验/执行、B4 本地模型加载与标准 AIMessage 输出、B5 文件式 Memory 检索/注入/保存 均已完成并通过验收 |
| 进阶要求 | B1 多轮/历史压缩/checkpoint/batch；B2 增强 BM25 检索、沙盒代码、复合 Skill、统一错误码、资源预算；B3 缓存/重试/统计/真实模型描述对比；B4 2模型×2绑定 benchmark、多工具同消息、Plan-and-Execute、能力覆盖门槛；B5 真实 Qwen3-0.6B 向量检索、LLM 摘要、Memory 去重/冲突/dry-run、错误 Memory A/B |
| 支持的主要任务类型 | 单轮/多轮工具型问答、文件阅读与总结、本地资料检索、CSV/TSV 表格分析、Markdown/JSON 格式转换、受限 Python 沙盒执行、跨会话记忆问答、Plan-and-Execute 多步任务 |
| 当前限制 | 真实推理需单张 CUDA GPU 与服务器本地模型权重；单模型缓存一次仅驻留一个 profile；fixture 模式不能保存 Memory；不包含任何训练/微调流程 |

---

## 2. 整体流程与模块结构

### 2.1 模块边界

| 模块 / 阶段 | 入口文件 / 入口函数 | 主要职责 | 输入 | 输出 |
|---|---|---|---|---|
| B1 Agent Runtime | `code/b1_agent_runtime.py::run_agent` | 多轮消息循环、checkpoint、batch、trace、prompt 模板切换、历史压缩 | runtime_input.json + tools/memory/model 配置 | messages.json、trace.json、final_answer.md、checkpoint.json |
| B2 Skills | `skills/*.py`、`code/b2_run_skill.py` | 真正执行受限本地工具，返回结构化 SkillResult | 工具参数 JSON | SkillResult JSON、可选生成文件 |
| B3 Tool Layer | `code/b3_tool_layer.py::get_tools_schema/execute_tool_calls`、`code/b3_advanced.py` | 从函数签名生成 Tool Schema、校验模型参数、调度 B2、缓存/重试/统计 | tool_calls(list) + tools.yaml | 标准 ToolMessage 列表、统计报告 |
| B4 Local Agent LLM | `code/b4_local_agent_llm.py::generate_ai_message`、`code/b4_core/` | 本地模型加载、prompt_json/native_tools 两绑定、原生方言解析、Plan-and-Execute | messages + tools_schema + model.yaml | 标准 AIMessage、raw_model_output.json、ai_message.json |
| B5 Memory | `code/b5_memory.py`、`code/b5_advanced.py`、`code/b5_20236493_adapter.py` | 文件式 Memory 索引/检索/保存、向量检索、LLM 摘要、A/B 与 dry-run 更新 | memory.yaml + 用户输入 + ids | selected_memory、saved_memory、索引/图 |
| 完整闭环 | `code/run_full_demo.py`、`./start_agent_chat.sh` | 把 B1–B5 串成一次真实模型闭环 | full_runtime.json + 三份配置 | demo_report.md、trace.json、各模块日志 |
| 验收总入口 | `./acceptance.sh` | 一键运行 unit/b2/b3/b4/b5/full 各阶段 | 阶段名 | 各阶段验收产物与 summary |

### 2.2 系统架构图或流程图

```text
┌─────────────────────────── B1 Agent Runtime ───────────────────────────┐
│                                                                         │
│   runtime_input.json ──► 校验 ──► 选 mode(fixture / integrated)         │
│                                  │                                      │
│          ┌───────────────────────┴───────────────────────┐              │
│          ▼ integrated                                    ▼ fixture       │
│   ┌─ B5 load_memory ──► selected_memory                  │ 预设 memory/  │
│   │  (keyword + Qwen3-0.6B 向量)                         │ tools/ai/tool │
│   ├─ B3 get_tools_schema ──► tools_schema                │ messages      │
│   │                                                       │              │
│   ▼ 组装 system + memory + user ──► messages             │              │
│   │                                                       │              │
│   │  ┌─────────────── 循环 (max_turns) ──────────────┐    │              │
│   │  │                                                │   │              │
│   │  │  B4 generate_ai_message  ◄── messages          │   │              │
│   │  │   (Qwen3.5-4B / Qwen3-1.7B)                    │   │              │
│   │  │   prompt_json / native_tools                   │   │              │
│   │  │            │                                    │   │              │
│   │  │            ▼ AIMessage {content, tool_calls}    │   │              │
│   │  │            │                                    │   │              │
│   │  │  无 tool_calls? ──► final_answer, 结束          │   │              │
│   │  │            │ 有 tool_calls                      │   │              │
│   │  │            ▼                                    │   │              │
│   │  │  B3 execute_tool_calls                          │   │              │
│   │  │   ├─ Schema 校验                                │   │              │
│   │  │   ├─ 缓存 / 重试 / 统计                         │   │              │
│   │  │   └─ B2 invoke(skill) ──► SkillResult           │   │              │
│   │  │            │                                    │   │              │
│   │  │            ▼ ToolMessage[]                      │   │              │
│   │  └────────────┴── 追加到 messages ─────────────────┘   │              │
│   │                                                          │              │
│   ▼ 写 messages / trace / final_answer / checkpoint         │              │
│   └─ B5 save_memory (conversation / global)                 │              │
└─────────────────────────────────────────────────────────────────────────┘

Plan-and-Execute 旁路（B4 独立入口 b4_plan_execute.py）：
  task → LLM 生成 JSON 计划 → validate_plan → plan_layers →
  每层并发 execute_tool_calls → LLM 汇总 → final_answer
```

### 2.3 一次完整任务或实验的流程

1. **原始输入**：用户在 `data/acceptance/full_runtime.json`（或交互式终端）中给出 `user_input`，例如“请同时读取 docs/agent_intro.txt，并使用计算器计算 23 * 17 + 9，最后合并回答。”，并指定 `toolset=basic_tools`、`max_turns=3`、`model_profile=qwen35_4b`、`tool_binding=native_tools`。
2. **预处理与上下文构造**：B1 校验输入；integrated 模式下调用 B5 `load_memory` 按关键词+向量检索相关 Memory，调用 B3 `get_tools_schema` 从 `configs/tools.yaml` 与函数签名生成 Tool Schema；把 system prompt + memory 片段 + user_input 组装成 messages。
3. **核心处理（模型决策）**：B1 调 B4 `generate_ai_message`，B4 按 profile 路由选择 Qwen3.5-4B 或 Qwen3-1.7B，用 native_tools 绑定 `tokenizer.apply_chat_template(..., tools=...)` 渲染 prompt，`model.generate` 生成原始文本，再用 `qwen35_xml`/`qwen3_json` 方言解析为标准 AIMessage 并做 Schema 校验。
4. **中间结果传递**：AIMessage 追加到 messages；若含 tool_calls，B1 调 B3 `execute_tool_calls`，B3 校验参数后通过共享 invoker 调用 B2 Skill（file_reader/calculator 等），得到标准 ToolMessage 列表并追加回 messages。
5. **循环与最终输出**：重复“B4 决策 → B3 执行”直到模型给出不含 tool_calls 的最终回答，或触发 `max_turns` 保护；预期消息流为 `system → user → assistant(tool_calls) → tool → tool → assistant(final)`。
6. **持久化与日志**：B1 写 `messages.json`、`tool_messages.json`、`trace.json`（含每轮 B4 profile/binding/真实 token/时延）、`final_answer.md`、`checkpoint.json`、`runtime_log.jsonl`；B4 每次调用写 `raw_model_output.json`/`ai_message.json`/`llm_run_log.jsonl`；若 `save_memory≠none` 且闭环成功，B5 把本次对话保存为 Memory 并更新索引/图。

---

## 3. 模型、数据集与外部资源

### 3.1 模型说明

| 项目 | 内容 |
|---|---|
| 使用模型 | Qwen3.5-4B（决策/规划）、Qwen3-1.7B（短任务对比）、Qwen3-0.6B（B5 向量检索）、Qwen3-1.7B（B5 LLM 摘要 hook） |
| 模型来源 | 服务器本地路径，`local_files_only=true`，不联网下载 |
| 项目内相对路径 | 由 `configs/model.yaml`（profiles）与 `configs/memory.yaml`（embedding）声明，不在仓库内存储权重 |
| 是否需要 GPU | 需要（真实推理需单张 CUDA GPU，如 NVIDIA H200） |
| 是否需要联网运行 | 不需要（离线模式 `TRANSFORMERS_OFFLINE=1`、`HF_HUB_OFFLINE=1`） |

| 模型 | 服务器本地路径 | 用途 |
|---|---|---|
| Qwen3.5-4B | `/root/15/assignment_B/Qwen3.5-4B` | B4 默认决策、规划与复杂任务，native_parser=qwen35_xml |
| Qwen3-1.7B | `/root/siton-data-1f55405a64d24fe2819a81c90df30517/model/Qwen3-1.7B` | B4 fast profile 与对比评测，native_parser=qwen3_json；B5 Memory 摘要 hook |
| Qwen3-0.6B | `/root/siton-data-1f55405a64d24fe2819a81c90df30517/backup/finetune_demo/Qwen3-0.6B` | B5 真实 hidden-state mean pooling + cosine 向量检索 |

```bash
# 模型已在服务器本地，无需下载。运行真实模型前显式开启离线模式：
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
cd /home/15/agent/code
```

### 3.2 数据集 / 示例数据说明

本项目不使用任何外部训练数据集；所有数据均为项目自带的演示/验收样例。

| 数据或文件 | 用途 | 来源 | 项目内相对路径 |
|---|---|---|---|
| agent_intro.txt | 文件阅读/总结样例 | 项目自带 | `data/docs/agent_intro.txt` |
| search_skill_demo.md、tool_calling.md | 本地检索样例 | 项目自带 | `data/docs/` |
| results.csv | 表格分析样例 | 项目自带 | `data/tables/results.csv` |
| B1 fixture 输入与预设消息 | fixture 模式隔离调试 B1 | 项目自带 | `data/b1_fixtures/` |
| B1 多轮/批处理输入 | 演示 B1 进阶 | 项目自带 | `data/b1_advanced/` |
| B4 smoke/计划/全矩阵用例 | B4 验收 | 项目自带 | `data/b4_eval/` |
| B3 工具调用样例（正/负） | B3 Schema 校验验收 | 项目自带 | `data/messages/`、`data/tool_inputs/` |
| B5 Memory 样例 | B5 检索/更新/A-B 验收 | 项目自带 | `data/b5_eval/`、`data/memory_inputs/` |
| full_runtime.json | 完整闭环输入 | 项目自带 | `data/acceptance/full_runtime.json` |
| 预置 Memory 文档 | 跨会话记忆演示 | 项目自带 | `memory/global/`、`memory/conversations/` |

```bash
# 所有样例数据均随项目自带，无需下载或解压。
# 跨会话记忆独立保存在 persistent_memory/teacher_demo/，不会被 clean-interactive 删除。
```

---

## 4. 环境安装

### 4.1 运行环境

| 项目 | 要求 |
|---|---|
| Python 版本 | 3.10.18（正式验收解释器 `/opt/conda/bin/python`） |
| 操作系统 / 服务器环境 | Linux（服务器 `ssh -p 20098 root@202.199.13.141`，NVIDIA H200） |
| GPU 要求 | 真实推理需单张 CUDA GPU；B1 fixture/mock 与无大模型契约测试可在 CPU 运行 |
| 主要依赖 | PyYAML、torch>=2.5、transformers>=5.12、safetensors、tokenizers、huggingface_hub、numpy（详见 `requirements.txt`） |

### 4.2 安装步骤

```bash
# 验收服务器已预装 /opt/conda/bin/python (3.10.18) 与匹配 CUDA 的 torch。
# 如需在新环境复现，最小依赖安装：
pip install "PyYAML>=6.0,<7" "torch>=2.5,<3" "transformers>=5.12,<6" \
            "safetensors>=0.8,<1" "tokenizers>=0.22,<1" \
            "huggingface_hub>=1.21,<2" "numpy>=2.2,<3"

# 注意：不要仅凭 requirements.txt 替换已安装的 torch wheel，
# 其 CUDA 后缀必须与宿主驱动/toolkit 匹配。
```

常见环境问题：

- **模型路径不可读**：`/root/15/assignment_B/Qwen3.5-4B` 等需 root 权限，`./acceptance.sh` 的真实模型阶段会做 `require_model_access` 检查。
- **CUDA 不可用**：B4 真实推理会显式报 `profile requests cuda, but CUDA is unavailable`；可退回 fixture/mock 或无大模型契约测试。
- **模板不支持 tools 参数**：native_tools 绑定会抛 `NativeToolBindingUnsupported`，明确失败而非静默退化为 prompt_json。
- **transformers 版本**：`apply_chat_template` 的 `enable_thinking` 参数在不支持的版本上会被自动移除并重试。

---

## 5. 输入文件与配置文件说明

### 5.1 主要配置文件

| 配置文件 | 作用 | 需要修改的字段 |
|---|---|---|
| `configs/model.yaml` | B4 模型 profile 注册、路由、生成参数、prompt 路径 | `profiles.*.model_name_or_path`、`routing.fast_profile`、`generation.max_new_tokens` |
| `configs/tools.yaml` | 工具集与工具定义（module/function/参数/returns） | `toolsets.basic_tools`、`tools.*`、`settings.data_root` |
| `configs/memory.yaml` | B5 Memory 根目录、检索 top-k、向量模型 | `memory.root_dir`、`retrieval.keyword_top_k`、`embedding.model_name_or_path` |
| `configs/skill_limits.yaml` | B2 Skill 资源预算（查询长度/文件数/字节/top-k/超时/循环/CPU/内存） | 各 Skill 的上限阈值 |

### 5.2 主要输入文件

| 输入文件 | 用途 | 适用场景 |
|---|---|---|
| `data/b1_fixtures/b1_fixture_input.json` | fixture 模式验证 B1 循环不依赖真实模型 | 模块演示 |
| `data/b1_advanced/runtime_multi_turn.json` | 验证 B1 多轮/历史压缩/prompt 模板 | 模块演示 |
| `data/b1_advanced/batch_runtime_inputs.json` | 验证 B1 批处理 | 模块演示 |
| `data/runtime_input.json` | integrated 单轮基础样例 | 推理 |
| `data/acceptance/full_runtime.json` | 真实模型完整闭环 | 完整系统 |
| `data/b4_eval/smoke_cases.jsonl` | B4 单工具与多工具同消息 | 评测 |
| `data/b4_eval/plan_read_calc.json` | B4 Plan-and-Execute 任务 | 评测 |
| `data/b4_eval/cases.jsonl` | B4 8 案例全矩阵 | 评测 |
| `data/messages/b3_tool_call_*.json` | B3 正/负 Schema 校验样例 | 异常样例 |
| `data/tool_inputs/tool_input_*_error.json` | B2/B3 异常样例（缺参/越界/危险代码/除零） | 异常样例 |
| `data/b5_eval/bad_memory.txt` | B5 错误 Memory A/B | 异常样例 |

---

## 6. 完整流程 Demo 运行

### 6.1 Demo 样例说明

| Demo | 输入文件 / 输入内容 | 演示目的 |
|---|---|---|
| 教师现场交互演示 | 终端输入自然语言（`./start_agent_chat.sh`） | 真实 Qwen3.5 + native_tools + 跨会话 Memory 的完整人机闭环 |
| 完整闭环自动验收 | `data/acceptance/full_runtime.json` | B1→B5 真实模型一次性闭环，产出可校验的 trace 与报告 |
| 五案例 Showcase | `run_showcase.py` 默认 5 个 case | 覆盖五个 B2 Skill 的真实模型端到端演示 |
| B4 一键验收 | `data/b4_eval/smoke_cases.jsonl` + `plan_read_calc.json` | 2模型×2绑定 + 多工具 + Plan-and-Execute 能力门槛 |
| B2 一键验收 | `verify_b2.py` 默认用例 | 五个基础 Skill 的正常/异常/安全/资源预算矩阵 |

### 6.2 运行命令

教师现场演示（推荐）：

```bash
ssh -p 20098 root@202.199.13.141
cd /home/15/agent
./start_agent_chat.sh
# 交互输入自然语言；/skills 查看 Skill；/memory 查看 Memory ID；/exit 退出并汇总
```

完整闭环自动验收：

```bash
cd /home/15/agent/code
export TRANSFORMERS_OFFLINE=1
/opt/conda/bin/python run_full_demo.py \
  --input ../data/acceptance/full_runtime.json \
  --tools_config ../configs/tools.yaml \
  --memory_config ../configs/memory.yaml \
  --model_config ../configs/model.yaml \
  --llm_mode native_tools \
  --model_profile qwen35_4b \
  --tool_binding native_tools \
  --outdir ../outputs/full_acceptance
```

一键分阶段验收（`./acceptance.sh`）：

```bash
cd /home/15/agent
./acceptance.sh unit            # 68 项轻量单元/契约/安全测试
./acceptance.sh b2              # B2 验收
./acceptance.sh b3              # B3 契约 + 真实模型描述对比
./acceptance.sh b4              # B4 2模型×2绑定 + Plan-and-Execute
./acceptance.sh b5              # B5 真实向量 + 摘要 + A/B
./acceptance.sh full            # B1→B5 真实闭环
./acceptance.sh all             # 顺序运行全部阶段
./acceptance.sh summary         # 查看当前结果摘要
./acceptance.sh clean [阶段]    # 清理 all/unit/b2/b3/b4/b5/full
./acceptance.sh clean-interactive  # 清理对话演示输出
```

### 6.3 关键参数说明

| 参数 | 说明 |
|---|---|
| `--input` / `--batch_input` | B1 单任务/批任务输入 JSON |
| `--tools_config` | `configs/tools.yaml`，声明工具集与工具定义 |
| `--memory_config` | `configs/memory.yaml`，声明 Memory 根目录与向量模型 |
| `--model_config` | `configs/model.yaml`，声明 B4 profile/路由/生成参数 |
| `--llm_mode` | `mock`/`prompt_json`/`native_tools`；fixture 模式由 input 决定 |
| `--model_profile` | 覆盖 B4 profile，如 `qwen35_4b`/`qwen3_1_7b` |
| `--tool_binding` | `prompt_json`/`native_tools` |
| `--decision_strategy` | `react`/`plan_execute`（B1 中仅影响 profile 路由，完整规划用 `b4_plan_execute.py`） |
| `--outdir` | 输出目录 |
| `--resume` | B1 从 checkpoint.json 断点恢复 |
| `--temporary-memory` / `--reset-memory` | 交互脚本 Memory 模式控制 |

### 6.4 运行成功的判断方式

- **交互演示**：终端能正常多轮对话，`/exit` 后打印 B1/B3/B4/B5 汇总，`outputs/interactive/current/INTERNALS_SUMMARY.md` 存在且各模块证据齐全。
- **完整闭环**：`outputs/full_acceptance/trace.json` 的 `status=success`，消息流为 `system → user → assistant(tool_calls) → tool → tool → assistant(final)`，`final_answer.md` 非空。
- **B4 验收**：`b4_acceptance_summary.json` 的 `status=success`，`capability_gates` 全部为 true（两模型/两绑定各至少 1 次成功、多工具成功≥1、真实 token 记录齐全、计划成功）。
- **B2 验收**：`acceptance_report.md` 的 `Status: PASS`，正常/异常/安全/资源预算用例全部通过。
- **B3/B5 验收**：对应 `*_acceptance_summary.json` 的 `status=success`；B5 的 `production_memory_modified=false`（默认 dry-run）。
- **单元测试**：`./acceptance.sh unit` 输出 `OK` 与通过数。

---

## 7. 输出文件与结果说明

### 7.1 主要输出文件

| 输出文件 | 生成模块 / 阶段 | 格式 | 说明 |
|---|---|---|---|
| `messages.json` | B1 | JSON | 完整消息数组（system/user/assistant/tool） |
| `tool_messages.json` | B1（integrated） | JSON | 本轮所有 ToolMessage |
| `trace.json` | B1 | JSON | 会话状态、轮次、B4 profile/binding/usage、memory_save、warnings |
| `final_answer.md` | B1 | Markdown | 最终回答文本 |
| `checkpoint.json` | B1 | JSON | 可 resume 的完整运行态 |
| `runtime_log.jsonl` | B1（integrated） | JSONL | 累计运行记录 |
| `raw_model_output.json` | B4 | JSON | 真实 prompt/raw_text/parsed_candidate/usage/metadata |
| `ai_message.json` | B4 | JSON | 标准化 AIMessage |
| `llm_run_log.jsonl` | B4 | JSONL | B4 累计运行记录 |
| `<skill>_result.json`、`skill_run_log.jsonl` | B2 | JSON/JSONL | 单 Skill 执行结果与日志 |
| `acceptance_report.md`、`acceptance_matrix.json` | B2 验收 | Markdown/JSON | B2 正常/异常/安全/资源预算矩阵 |
| `b3_acceptance_summary.json` | B3 验收 | JSON | Schema/缓存/重试/统计与真实模型描述对比 |
| `benchmark_summary.json`、`benchmark_runs.jsonl` | B4 验收 | JSON/JSONL | 2×2 矩阵 token/时延/成功率 |
| `b4_acceptance_summary.json` | B4 验收 | JSON | 能力门槛判读 |
| `validated_plan.json`、`plan_trace.json`、`plan_execute_report.json` | B4 Plan-and-Execute | JSON | 计划/分层执行/报告 |
| `b5_acceptance_summary.json` | B5 验收 | JSON | 向量/摘要/A-B/dry-run 结果 |
| `demo_report.md` | 完整闭环 | Markdown | 闭环汇总报告 |
| `selected_memory.json` | B5 | JSON | 本轮检索/注入的 Memory |
| `agent_trace.html` | `trace_report.py` | HTML | 可视化 trace |

### 7.2 运行截图或结果图例

```text
[在此处插入 start_agent_chat.sh 交互截图]
[在此处插入 full_acceptance/trace.json status 字段截图]
[在此处插入 b4_acceptance_summary.json capability_gates 截图]
[在此处插入 agent_trace.html 可视化截图]
```

示例占位：

![运行结果占位](docs/images/demo_result_placeholder.png)

---

## 8. 协作实现说明

- **模块输入输出格式约定**：全队共用 `common/schemas.py` 的扁平消息契约——AIMessage 为 `{role, content, tool_calls:[{id,name,args}]}`，ToolMessage 为 `{role, tool, tool_call_id, name, content, status}`；B2 与 B3 共用同一 `SkillResult` 错误信封（`status/skill_name/input/output/error/latency_ms`，`error` 含 `code/type/message/retryable/details`），缺参、越界、文件错误、沙盒错误在两层不产生不同格式。
- **配置驱动联调**：通过 `configs/tools.yaml`、`configs/memory.yaml`、`configs/model.yaml`、`configs/skill_limits.yaml` 把工具集、Memory、模型 profile、资源预算外部化，成员可独立修改自己的配置而不动代码；`data/` 下按 `b1_*`/`b3_*`/`b4_eval`/`b5_eval`/`messages`/`tool_inputs` 分目录放置各模块验收样例，降低联调成本。
- **数据格式不一致的处理**：B1 会在工具回合后追加 system 提示，但 Qwen chat template 要求 system 只能在开头，B4 `bindings.py::normalize_chat_messages` 将多个 system 块合并为一条前缀；native_tools 绑定加 `NativeToolBindingUnsupported` 检查，若工具名未出现在 prompt_text 中即明确失败，避免静默退化为 prompt_json。
- **接口签名兼容**：B4 `generate_ai_message` 保留原前六参数位置不变，新增 `profile/binding/strategy` 为 keyword-only，B1 旧调用无须改动即可注入新路由能力。
- **验收与回归**：`tests/` 下 68 项单元/契约/安全测试不加载真实大模型，PR 联调时先跑 `./acceptance.sh unit`；真实模型阶段（B3/B4/B5/full）由 `./acceptance.sh` 一键串行，每阶段运行前自动清理上一份结果，避免不断生成新目录。
- **多模块配合完成的功能**：完整闭环（B1 编排 + B4 决策 + B3 校验/执行 + B2 Skill + B5 Memory 注入与保存）、Plan-and-Execute（B4 生成计划 + B3 分层执行 + B2 Skill + B4 汇总）、跨会话记忆（B5 保存 + B5 检索 + B1 注入 + B4 决策）、真实模型描述对比（B4 生成 + B3 Schema 对比 + 统计）都是多模块配合后才完成的。

---

## 9. 已知问题与改进方向

| 问题 | 当前原因 | 可能改进 |
|---|---|---|
| `decision_strategy=plan_execute` 在 B1 单次决策接口中只影响 B4 profile 路由，不等于完整规划 | B1 单次接口没有计划状态与跨步骤依赖的编排能力 | 完整 Plan-and-Execute 需用 `b4_plan_execute.py`；后续可把计划状态纳入 B1 checkpoint |
| SingleModelPool 仅缓存一个模型，2×2 benchmark 切换 profile 时需重新加载 | 避免多模型同时驻留撑爆单卡显存 | 多卡环境下可扩展为按 profile 缓存多模型 |
| 真实推理需 root 权限读取服务器本地模型 | 模型权重放在 `/root/15/...` 受控路径 | 可把模型迁移到普通用户可读目录或配置权限组 |
| B1 fixture 模式不能保存 Memory | fixture 要求 `save_memory=none` 以保证隔离 | 可新增半集成模式支持 fixture+Memory 联调 |
| native_tools 绑定依赖模板真实支持 tools 参数 | 不同模型模板支持度不一 | 已做能力检查并明确失败；可扩展方言适配器覆盖更多模型 |
| 某个模型/绑定的失败仍会出现在 benchmark 中 | 设计如此：失败是对比数据，不伪装为成功 | 可补充失败原因聚类与可视化 |
| 跨会话 Memory 默认写入 `persistent_memory/teacher_demo/` | 为教师演示跨会话记忆保留 | 已提供 `--temporary-memory`/`--reset-memory` 开关，正式验收可用 `--temporary-memory` |
| 项目不包含任何训练/微调流程 | 任务范围限定为推理与工具使用 | 如需个性化，可后续接入 LoRA/PEFT，但当前不在范围内 |
