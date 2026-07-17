# 交互与 Showcase

## 真实模型交互

推荐使用项目根目录的一键脚本：

```bash
cd /home/15/agent
./start_agent_chat.sh
```

默认输出到 `outputs/interactive/current/`，并把跨会话记忆独立保存到 `persistent_memory/teacher_demo/`。每次启动覆盖交互证据，但保留持久化记忆。

```bash
./start_agent_chat.sh --temporary-memory
./start_agent_chat.sh --reset-memory
```

也可以直接运行 Python 入口：

```bash
cd /home/15/agent/code
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

/opt/conda/bin/python agent_chat.py \
  --llm_mode native_tools \
  --model_profile qwen35_4b \
  --tool_binding native_tools \
  --toolset basic_tools \
  --outdir ../outputs/chat_real
```

输入 `/exit` 结束，输入 `/memory` 查看本次会话已选择的 Memory ID。

## 五案例 Showcase

```bash
/opt/conda/bin/python run_showcase.py \
  --llm_mode native_tools \
  --model_profile qwen35_4b \
  --tool_binding native_tools \
  --outdir ../outputs/showcase_real
```

正式验收不要传 `--allow_diagnostic_fallback`。模型未调用预期工具时，case 会标记 `failed_acceptance` 或 `error`，不会自动改写成成功。

只有排查 B2/B3 时才可显式启用诊断兜底：

```bash
/opt/conda/bin/python run_showcase.py \
  --llm_mode native_tools \
  --allow_diagnostic_fallback \
  --outdir ../outputs/showcase_diagnostic
```

该模式产生的 case 状态是 `diagnostic_only`，总报告不会判为验收成功。

## Trace HTML

```bash
/opt/conda/bin/python trace_report.py \
  --run_dir ../outputs/full_acceptance \
  --output ../outputs/full_acceptance/agent_trace.html \
  --title "Full Agent Acceptance"
```
