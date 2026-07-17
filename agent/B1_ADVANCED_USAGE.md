# B1 进阶运行说明

所有命令从 `/home/15/agent/code` 运行，解释器使用 `/opt/conda/bin/python`。

## 多轮、历史压缩与 checkpoint

```bash
cd /home/15/agent/code

/opt/conda/bin/python b1_agent_runtime.py \
  --input ../data/b1_advanced/runtime_multi_turn.json \
  --tools_config ../configs/tools.yaml \
  --memory_config ../configs/memory.yaml \
  --model_config ../configs/model.yaml \
  --llm_mode mock \
  --outdir ../outputs/B1_advanced_multi_turn
```

该输入演示：

- `conversation_turns`：一次运行包含多个用户轮次。
- `prompt_template_paths`：按轮次追加 system prompt。
- `history_summary_max_chars`：超过预算后压缩较早历史。
- `checkpoint.json`：每次模型决策和工具执行后保存状态。

恢复命令：

```bash
/opt/conda/bin/python b1_agent_runtime.py \
  --input ../data/b1_advanced/runtime_multi_turn.json \
  --tools_config ../configs/tools.yaml \
  --memory_config ../configs/memory.yaml \
  --model_config ../configs/model.yaml \
  --outdir ../outputs/B1_advanced_multi_turn \
  --resume ../outputs/B1_advanced_multi_turn/checkpoint.json
```

## Batch

```bash
/opt/conda/bin/python b1_agent_runtime.py \
  --batch_input ../data/b1_advanced/batch_runtime_inputs.json \
  --tools_config ../configs/tools.yaml \
  --memory_config ../configs/memory.yaml \
  --model_config ../configs/model.yaml \
  --llm_mode mock \
  --outdir ../outputs/B1_advanced_batch
```

每个任务写入独立 `task_NNN_<conversation_id>/`，汇总为 `batch_report.json`。

## 真实模型闭环

```bash
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

/opt/conda/bin/python b1_agent_runtime.py \
  --input ../data/acceptance/full_runtime.json \
  --tools_config ../configs/tools.yaml \
  --memory_config ../configs/memory.yaml \
  --model_config ../configs/model.yaml \
  --llm_mode native_tools \
  --model_profile qwen35_4b \
  --tool_binding native_tools \
  --decision_strategy react \
  --outdir ../outputs/B1_real
```

`decision_strategy=plan_execute` 在 B1 单次决策接口中只用于 B4 profile 路由。完整 Plan-and-Execute 需要 `b4_plan_execute.py`，因为它还需要 tools_config、计划状态和跨步骤依赖。

## 稳定接口

```python
from b3_tool_layer import get_tools_schema, execute_tool_calls
from b4_local_agent_llm import generate_ai_message
from b5_memory import load_memory, save_memory
```

- `get_tools_schema(...) -> list[ToolSchema]`
- `execute_tool_calls(...) -> list[ToolMessage]`
- `generate_ai_message(..., *, profile=None, binding=None, strategy="react") -> dict`
- `load_memory(...) -> selected_memory`
- `save_memory(...) -> saved_memory`

B1 只负责消息和状态编排；B3 执行工具、B4 负责本地模型决策、B5 负责 Memory。
