# B 方向本地 Agent 项目

本项目按 B 方向任务书实现 B1–B5，并以第二台服务器上的本地模型、受限本地工具和文件式 Memory 完成闭环。正式验收解释器为：

```text
/opt/conda/bin/python  (Python 3.10.18)
```

模型与数据均使用服务器已有本地路径。运行真实模型时建议显式开启离线模式：

```bash
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
cd /home/15/agent/code
```

项目不包含联网检索、外部 Web API、LoRA/PEFT、DPO 或任何训练/后训练流程。

教师现场验收以真实人机交互为主。直接阅读 `ACCEPTANCE_GUIDE.md`，并运行：

```bash
./start_agent_chat.sh
```

它会先依次展示五个 B2 Skill，退出对话后自动汇总本次运行中的 B1/B3/B4/B5 证据。默认启用位于 `persistent_memory/teacher_demo/` 的独立持久化 Memory，可跨进程检索相关历史；`--temporary-memory` 可切换为不保存模式。交互结果固定写入 `outputs/interactive/current/`，重复运行会覆盖旧结果。

## 1. 模块关系

```text
B1 Agent Runtime
  ├─ B5：检索并注入 Memory
  ├─ B4：本地模型生成标准 AIMessage
  ├─ B3：校验 Tool Schema、执行 tool_calls、返回 ToolMessage
  │    └─ B2：真正执行 Skill
  └─ B5：按配置保存对话 Memory
```

- `code/b1_agent_runtime.py`：多轮消息循环、检查点、批处理、trace。
- `skills/`：B2 基础与进阶 Skill。
- `code/b3_tool_layer.py`、`code/b3_advanced.py`：Schema、调度、缓存、重试、统计。
- `code/b4_core/`、`code/b4_local_agent_llm.py`：本地模型加载、两类绑定和输出解析。
- `code/b4_plan_execute.py`：真正的 LLM Plan-and-Execute。
- `code/b5_memory.py`、`code/b5_advanced.py`：Memory 检索、向量、摘要、更新和 A/B。
- `configs/`：模型、工具、Memory 和 Skill 资源策略。
- `tests/`：不加载真实大模型的契约、安全与回归测试。

## 2. Skill 与 Tool Schema 的区别

这是 B2/B3 的关键边界：

- **Skill** 是可执行的 Python 函数，例如 `skills.calculator.calculator`。它读取参数、执行逻辑并产生真实结果或结构化错误。
- **Tool Schema** 是提供给模型的声明式 JSON Schema，包括工具名称、用途、参数类型、必填项和返回值说明；它本身不执行任何任务。
- B3 根据函数类型注解和 `configs/tools.yaml` 生成/校验 Tool Schema，然后通过共享 invoker 调用 B2 Skill。
- B2 与 B3 共用同一个 `SkillResult` 错误信封，因此缺参、越界、文件错误和 sandbox 错误不会在两层产生不同格式。

标准 `SkillResult`：

```json
{
  "skill_name": "calculator",
  "status": "error",
  "input": {},
  "output": null,
  "error": {
    "code": "PARAM_MISSING",
    "type": "SkillFault",
    "message": "missing required parameters: expression",
    "retryable": false,
    "details": {"missing": ["expression"]}
  },
  "latency_ms": 0.1
}
```

## 3. B2：基础与进阶 Skill

### 3.1 基础要求

五个基础 Skill 均可独立 CLI 测试，并返回 JSON 可序列化结果：

| Skill | 主要能力 |
|---|---|
| `calculator` | AST 白名单算术，限制表达式长度、指数和数值大小 |
| `file_reader` | 只读数据根目录内 UTF-8 txt/md，限制文件和返回长度 |
| `local_file_search` | 中英文混合分词、中文 1/2/3-gram、BM25、短语加权、零分过滤 |
| `table_analyzer` | 流式 CSV/TSV 预览与数值统计，限制文件与行数 |
| `format_converter` | Markdown/JSON 转换，输出限制在指定目录 |

单个 Skill 示例：

```bash
/opt/conda/bin/python b2_run_skill.py \
  --skill calculator \
  --input ../data/tool_inputs/tool_input_calculator.json \
  --outdir ../outputs/B2_single
```

业务错误仍写入 `<skill>_result.json` 和 `skill_run_log.jsonl`，CLI 返回 0；配置/输入文件损坏等致命错误返回 1。

### 3.2 进阶要求

- 增强检索：Unicode NFKC、中文 n-gram、BM25、短语/文件名加权。
- 沙盒代码：用户源码不会交给 `exec/eval`；受限 AST 解释器运行在隔离子进程，Linux 下设置 RLIMIT，并在 root 启动时降权到 `nobody`。
- 复合 Skill：`read_and_convert`、`analyze_and_convert`，保留每一步 trace 和底层错误原因。
- 统一错误码：`PARAM_MISSING`、`FILE_NOT_FOUND`、`RESOURCE_EXHAUSTED`、`SANDBOX_VIOLATION`、`COMPOSITE_STEP_FAILED` 等。
- 高风险/高耗时限制：查询长度、目录项、文件数、单文件/总字节、索引 token、top-k、超时、代码操作数、循环次数、CPU、内存、输出和进程数均由 `configs/skill_limits.yaml` 集中控制。
- 目录安全：拒绝数据根目录外路径和符号链接逃逸。

一键 B2 验收：

```bash
/opt/conda/bin/python verify_b2.py --outdir ../outputs/B2_acceptance
```

核心产物：

- `acceptance_report.md`
- `acceptance_matrix.json`
- 每个正常、异常、安全和资源预算案例的独立 JSON

## 4. B3：Tool 层

B3 完成以下工作：

- 使用 `get_type_hints()` 从真实函数签名生成 Tool Schema。
- 对模型参数做必填、未知字段、类型和数值范围校验。
- 调用 B2 的 `make_context + invoke_callable`，保持同一 SkillResult。
- 只缓存成功结果；当前仅对确定性 calculator 开启持久缓存。
- 仅当 `error.retryable=true` 时重试。
- 输出调用次数、成功率、缓存命中、重试、p50/p95 延迟和错误码统计。
- 用真实 B4 本地模型比较“完整描述”和“弱化描述”两组 Schema 的工具选择准确率。

无大模型契约检查：

```bash
/opt/conda/bin/python verify_b3_b5.py --outdir ../outputs/B3_B5_contracts
```

真实模型 B3 进阶验收：

```bash
AGENT_HOOK_ARTIFACT_DIR=../outputs/B3_acceptance/model_calls \
/opt/conda/bin/python b3_acceptance.py --outdir ../outputs/B3_acceptance
```

## 5. B4：本地 Agent LLM

### 5.1 模型与两种绑定

`configs/model.yaml` 定义两个真实 profile：

| Profile | 本地模型 | 用途 |
|---|---|---|
| `qwen35_4b` | Qwen3.5-4B | 默认决策、规划和复杂任务 |
| `qwen3_1_7b` | Qwen3-1.7B | 短任务和对比评测 |

加载方式不依赖 `accelerate`：`from_pretrained(dtype=...)` 后移动到单张 CUDA GPU，并保持单模型缓存，切换 profile 时释放上一模型。

两种工具绑定：

- `prompt_json`：将 Tool Schema 注入 system prompt，模型必须返回扁平 JSON 信封。
- `native_tools`：调用 tokenizer 原生 `apply_chat_template(..., tools=...)`；若模板忽略 tools 参数则明确失败。

两种模式都输出：

- `raw_model_output.json`：真实 prompt、原始生成文本、解析候选、模型/绑定、token 和延迟。
- `ai_message.json`：标准 `{role, content, tool_calls}`。
- `llm_run_log.jsonl`：累计运行记录。

Schema 校验失败会返回 `status=error`，B1 不会继续执行不合法的工具调用。

### 5.2 多工具与 Plan-and-Execute

- `data/b4_eval/smoke_cases.jsonl` 要求一个 AIMessage 同时产生 `file_reader + calculator` 两个 tool_calls，B3 返回两个对应 ToolMessage。
- `b4_plan_execute.py` 执行“LLM 生成 JSON 计划 → 严格校验/最多一次纠错 → 按依赖层调用 B3 → LLM 汇总”。
- B1 的 `decision_strategy=plan_execute` 只影响 profile 路由，不等同于完整规划；完整验收必须使用 `b4_plan_execute.py` 或 `b4_acceptance.py`。

一键真实 B4 验收：

```bash
/opt/conda/bin/python b4_acceptance.py --outdir ../outputs/B4_acceptance
```

该命令运行 2 模型 × 2 绑定、真实 token/时延/工具成功率对比以及 Plan-and-Execute。验收采用能力覆盖门槛：两个模型和两种绑定都必须至少有成功证据，多工具和计划必须成功；某个模型/绑定的失败仍保留为比较数据，不会改写成成功记录。

完整 8 案例矩阵：

```bash
/opt/conda/bin/python b4_benchmark.py \
  --cases ../data/b4_eval/cases.jsonl \
  --model_config ../configs/model.yaml \
  --tools_config ../configs/tools.yaml \
  --outdir ../outputs/B4_benchmark_full
```

## 6. B5：Memory

基础功能包括 global/conversation 文档、索引、query Top-K、显式 ID 保留、字符预算、保存和日志。

进阶功能包括：

- 真实本地 Qwen3-0.6B hidden-state mean pooling + cosine 向量检索；不联网、不使用哈希伪向量。
- 通过 B4 Qwen3-1.7B hook 完成 Memory 摘要。
- 指定 Memory 的去重、补充、冲突记录/替换/追加；验收默认 dry-run。
- 使用同一真实本地模型分别回答“正常 Memory”和“注入错误 Memory”，保存 A/B 回答及模型评估。

一键真实 B5 进阶验收：

```bash
AGENT_HOOK_ARTIFACT_DIR=../outputs/B5_acceptance/model_calls \
/opt/conda/bin/python b5_acceptance.py --outdir ../outputs/B5_acceptance
```

`b5_acceptance_summary.json` 固定记录 `production_memory_modified=false`。

## 7. B1 与完整闭环

B1 支持：

- `fixture` 与 `integrated` 两种运行方式；fixture 仅用于隔离调试。
- 多轮 user turn、重复工具循环、最大轮次保护。
- checkpoint 保存/恢复、批处理、历史压缩和 prompt 切换。
- B4 profile/binding/真实 token 与耗时进入每轮 trace。

教师现场一键演示（推荐）：

```bash
cd /home/15/agent
./start_agent_chat.sh
```

脚本会自动设置离线环境、使用 Qwen3.5/native tools，并在屏幕上给出五个 B2 Skill 的自然语言演示指令。输入 `/exit` 后自动打印 B1/B3/B4/B5 汇总。每次运行覆盖 `outputs/interactive/current/`，而跨会话记忆独立保存在 `persistent_memory/teacher_demo/`。

正式完整闭环必须使用真实模型：

```bash
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

预期消息流：

```text
system -> user -> assistant(tool_calls) -> tool -> tool -> assistant(final)
```

主要产物：`messages.json`、`tool_messages.json`、`trace.json`、`final_answer.md`、`demo_report.md`、`selected_memory.json`、B3/B4/B5 日志和原始模型文件。

## 8. 回归测试与总入口

全套轻量测试：

```bash
cd /home/15/agent
PYTHONDONTWRITEBYTECODE=1 /opt/conda/bin/python -m unittest discover -s tests -v
```

结构/安全检查总入口：

```bash
cd /home/15/agent/code
/opt/conda/bin/python advanced_features.py --outdir ../outputs/advanced_structural
```

同时运行所有真实模型高级验收：

```bash
/opt/conda/bin/python advanced_features.py \
  --run_real_models \
  --outdir ../outputs/advanced_acceptance
```

没有 `--run_real_models` 时，总入口会明确标记 `structural_checks_only`，不会生成或声称已有模型证据。

## 9. 验收状态判读

- `success`：该入口定义的检查全部达到门槛。
- `partial`：完成运行但存在非致命子项失败。
- `completed_with_failures`：所有对比已执行，但未达到能力门槛。
- `not_run`：缺少真实模型 hook 或用户未要求执行；绝不伪装为完成。
- mock、fixture 和 `--allow_diagnostic_fallback` 只用于调试，不能作为正式 B4 或完整系统验收证据。
