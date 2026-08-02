# 现代 Agent 框架与 Harness 手写教程

> 从零手写一套现代 Agent 框架和一个 Harness 工程。学完主线,你能亲手造出一个 "mini Claude Code";学完深入篇,你能亲手造出一个 "mini LangGraph"。两条路线合起来,就是 2026 年 Agent 工程的完整版图。

## 这份教程在教什么

市面上大多数 "Agent 教程" 教的是怎么**调用**一个框架。这份教程不一样:它带你**重新发明**框架本身——而且是按 2026 年真实的主流架构来发明。

2026 年的一个关键事实是:Agent 工程的重心已经从 "框架" 转移到了 **Harness(执行壳)**。同一个模型放进不同 harness,实际效果会受到上下文管理、权限、会话、工具执行和沙箱等基础设施的显著影响。**模型决定上限,harness 决定你能拿到上限的百分之多少。**

所以这份教程教两样东西,缺一不可:

- **Agent 框架(内核)** —— `tinycore`:消息、模型接口、工具系统、主循环、上下文工程、事件流。对应 Claude Code / Codex 这类主流 harness 的内核层。
- **Harness(控制面)** —— `tinyharness`:会话持久化、权限与审批、钩子、子代理、技能、MCP 客户端。对应 Claude Agent SDK / Codex CLI 的产品能力层。

两者组装起来的终局示例 [`code/examples/mini_code.py`](code/examples/mini_code.py),就是一个能在你的工作区里读写文件、跑命令、要审批、可恢复会话的迷你编码 Agent——一个五脏俱全的 mini Claude Code。

## 两条路线:主线与深入篇

2026 年的主流 Agent 架构收敛成了**两大家族**,它们对同一个问题("一个 while 循环怎么撑起真实产品")给出了两种答案:

| | 家族 A:循环 + Harness | 家族 B:图运行时 |
|---|---|---|
| 哲学 | 控制流交给**模型**,工程做厚**壳** | 控制流写进**代码**,建模为显式的图 |
| 代表 | Claude Code / Claude Agent SDK、Codex CLI / OpenAI Agents SDK、Gemini CLI | LangGraph + LangChain 1.x、Microsoft Agent Framework、Google ADK |
| 本教程 | **主线(第 0-11 章)**:`tinycore` + `tinyharness` | **深入篇(runtime/R1-R7)**:`tinygraph` + `tinyagent` |

两条路线不是对立而是互补:主线是当下产品形态的主流,深入篇是企业工作流编排的主流,而且图运行时把 "状态、并发、恢复" 这些问题解得最形式化——学完主线再读深入篇,你会看到两家在最底层殊途同归。第 10 章专门讲清 "什么时候该选哪条路"。

## 学习路径

> 建议按顺序读主线。每一章都建立在前一章暴露的问题之上,跳读会丢失 "为什么需要这个东西" 的动机。

**主线:内核篇(造框架)**

| 章节 | 标题 | 你会手写出 |
|---|---|---|
| [[00-全景-Agent框架与Harness]] | 全景:Agent、框架与 Harness | 2026 格局地图、四层架构、方法论 |
| [[01-最小Agent循环]] | 最小 Agent 循环 | 消息、模型接口、`@tool`、~100 行主循环 |
| [[02-工具系统与执行环境]] | 工具系统与执行环境 | 编码工具八件套、路径 jail、并行执行、错误即消息 |
| [[03-上下文工程]] | 上下文工程 | system prompt 装配、记忆文件、压缩、结构化笔记 |
| [[04-事件流与流式]] | 事件流与流式 | 事件协议、逐 token 流式、打断与中途插话 |

**主线:Harness 篇(造壳)**

| 章节 | 标题 | 你会手写出 |
|---|---|---|
| [[05-会话与持久化]] | 会话与持久化 | JSONL 事件日志、重放、恢复、分叉 |
| [[06-权限与沙箱]] | 权限与沙箱 | 权限模式、规则、审批回调、两种信任姿态 |
| [[07-钩子与技能]] | 钩子与技能 | 生命周期钩子、SKILL.md 渐进披露 |
| [[08-子代理与多智能体]] | 子代理与多智能体 | task 工具、上下文隔离、并行扇出 |
| [[09-MCP-工具的生态协议]] | MCP:工具的生态协议 | 最小 MCP 客户端 + server、Harness 门面组装 |
| [[10-第二条路线-图运行时]] | 第二条路线:图运行时 | 选型判据;通往深入篇的桥 |
| [[11-收尾-对照真实框架]] | 收尾:对照真实框架 | 与 Claude Agent SDK / Agents SDK / LangGraph 逐层对照、源码阅读路线 |

**深入篇:图运行时(runtime/,可在主线后读,也可从第 10 章跳入)**

[[R0-深入篇导读|导读 R0]] · R1 循环的局限 · R2 构建图运行时(Pregel/BSP) · R3 持久化与检查点 · R4 流式与人在回路 · R5 重建 Agent 抽象层 · R6 中间件系统 · R7 进阶与生产化

## 配套代码

```
code/
├── tinycore/     # 主线内核:messages/models/tools/toolkit/context/events/loop
├── tinyharness/    # 主线控制面:session/permissions/hooks/subagents/skills/mcp/harness
├── tinygraph/      # 深入篇运行时(对应 LangGraph)
├── tinyagent/      # 深入篇 Agent 层(对应 LangChain)
└── examples/       # 每章示例 + mini_code.py(终局 demo)+ mcp_demo_server.py
```

**运行环境**:Python 3.10+,运行时零第三方依赖。主线 `k01-k08` 示例使用内置的 `FakeModel`(照剧本出牌的桩模型),可离线运行；深入篇按示例区分是否需要 SDK 和 API key:

| 示例 | 运行条件 |
|---|---|
| `04_checkpoint.py`、`05_stream_hitl.py`、`08_advanced.py` | 无需 API key |
| `03_graph_runtime.py` | 示例 A 无需 key,示例 B 需要 key |
| `07_middleware.py` | 示例 C/D 无需 key,示例 A/B 需要 key |
| `01_minimal_agent.py`、`06_create_agent.py` | 需要 SDK 和 API key |

对接真实模型时安装相应 SDK,支持 Anthropic / OpenAI 及 OpenAI 兼容服务(DeepSeek / Qwen / Kimi / 本地 vLLM):

```bash
pip install -r code/requirements.txt     # 只为对接真实模型;离线示例可跳过
cp code/.env.example code/.env           # 填 TINYAGENT_MODEL 与你的 key
python code/examples/k01_loop.py        # 先运行一个无需 key 的主线示例
```

`TINYAGENT_MODEL` 形如 `"provider:model"`(如 `anthropic:claude-sonnet-5` 或 `deepseek:deepseek-chat`),换 provider 只改这一行。

## 如何使用这份教程

1. **跟着敲**。每章代码建议亲手敲一遍。主线示例不需要 API key；深入篇按上表准备环境。
2. **先读 "为什么",再读 "怎么做"**。每章开头都先讲这一章要解决的问题;理解了动机,代码只是动机的自然结果。
3. **对照真实系统**。每个抽象我们都指出它对应 Claude Code / Agent SDK / LangGraph 里的什么。教程代码是简化版,概念是 1:1 的。

---

下一步 → [[00-全景-Agent框架与Harness]]
