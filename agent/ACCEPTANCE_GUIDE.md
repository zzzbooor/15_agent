# B 方向 Agent 教师现场演示指南

## 1. 统一启动入口

```bash
ssh -p 20098 root@202.199.13.141
cd /home/15/agent
./start_agent_chat.sh
```

这一个脚本同时完成：

1. 启动真实 Qwen3.5 Agent；
2. 交互展示五个 B2 Skill；
3. 输入 `/exit` 后汇总本次 B1、B3、B4、B5 证据；
4. 默认把对话写入独立持久化 Memory，下一次重新启动仍可按问题相关性检索。

## 2. B2 五个 Skill

启动后可先输入 `/skills`，再依次输入：

```text
请使用计算器精确计算 23 * 17 + 9。
请读取 docs/agent_intro.txt，并总结三条要点。
请在 docs 目录搜索与 tool calling 有关的本地资料，并总结最相关内容。
请分析 tables/results.csv，告诉我行数、列数和主要数值统计。
请把“模型负责决策，工具负责执行，记忆提供上下文”转换为 Markdown 列表并保存为 teacher_summary.md。
```

每轮终端只重点显示 Agent 回答、实际调用的 Skill 和证据目录。

## 3. 跨会话 Memory 演示

默认 `./start_agent_chat.sh` 已开启独立持久化模式。第一次启动时输入：

```text
请记住：本次验收口令是蓝鲸42，项目负责人姓张。
```

看到 Agent 回答后输入：

```text
/exit
```

重新启动：

```bash
./start_agent_chat.sh
```

询问：

```text
我上一次告诉你的验收口令和项目负责人信息是什么？
```

Agent 会根据新问题检索上一段会话保存的相关 Global Memory。输入 `/memory` 可查看本次已经选择或检索到的 Memory ID。

独立持久化目录为：

```text
/home/15/agent/persistent_memory/teacher_demo/
```

它不属于正式 `memory/`，也不会被 `clean-interactive` 删除。

如需回到无历史状态：

```bash
./start_agent_chat.sh --reset-memory
```

如需一次完全不保存记忆的演示：

```bash
./start_agent_chat.sh --temporary-memory
```

## 4. 退出后的内部汇总

输入 `/exit` 后，脚本自动展示：

- B1：成功轮数、工具轮次、模型调用次数和消息闭环；
- B3：Tool Schema、实际调用次数、成功/失败；
- B4：本地模型 profile、native tools、token 和原始模型产物；
- B5：Memory 模式、独立目录、本次检索的 ID 和新保存的 ID。

汇总文件：

```bash
cat outputs/interactive/current/INTERNALS_SUMMARY.md
```

## 5. 输出与清理

交互证据固定写入：

```text
outputs/interactive/current/
```

每次启动会覆盖上一份交互证据，但不会删除持久化 Memory。

只清理交互输出：

```bash
./acceptance.sh clean-interactive
```

清理自动验收输出：

```bash
./acceptance.sh clean all
```

重置跨会话记忆：

```bash
./start_agent_chat.sh --reset-memory
```

## 6. 备用自动验收

老师临时要求时再按需运行：

```bash
./acceptance.sh unit
./acceptance.sh b2
./acceptance.sh b4
./acceptance.sh b5
./acceptance.sh full
```
