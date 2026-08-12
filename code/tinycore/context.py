"""上下文工程 —— 决定"这一轮模型到底看到什么"（第 3 章）。

三个职责：
1. **装配（assemble）**：system prompt 不是历史的一部分，而是每轮现场拼装：
   身份指令 + 环境信息 + 项目记忆文件（AGENTS.md）+ 额外注入。这保证了
   改记忆文件立即生效、恢复会话不会带出陈旧指令。
2. **测量（estimate_tokens）**：优先用 provider 报告的真实用量（AIMessage.usage），
   兜底用"字符数/4"粗估。测不到就管不了。
3. **压缩（maybe_compact / clear_old_tool_results）**：上下文是稀缺资源。
   超过预算时，把早期历史用 LLM 摘要成一段"前情提要"重开，或先做更轻的
   "旧工具结果清理"。压缩点必须落在**安全边界**上——不能把一条带 tool_calls
   的 AI 消息和它的工具结果拆开（provider 会直接报错）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

from .messages import AIMessage, HumanMessage, Messages, SystemMessage, ToolMessage, Usage

# ---------------------------------------------------------------------------
# 测量
# ---------------------------------------------------------------------------


def estimate_tokens(messages: Messages) -> int:
    """估算这串消息的 token 规模。

    真实 token 数只有 provider 的分词器知道；工程上用两级近似就够做决策了：
    1. 若最近一条 AI 消息带 usage，它的 input_tokens 就是"上一轮全部历史"的
       **精确值**，在此基础上加上之后新增消息的粗估；
    2. 否则全部按 "字符数 // 4 + 每条消息 8 token 开销" 粗估（中英混合的保守值）。
    """
    base, start = 0, 0
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, AIMessage) and m.usage.input_tokens:
            base = m.usage.input_tokens + m.usage.output_tokens
            start = i + 1
            break
    rough = sum(len(m.content or "") // 4 + 8 for m in messages[start:])
    return base + rough


# ---------------------------------------------------------------------------
# 项目记忆文件（AGENTS.md / CLAUDE.md 惯例）
# ---------------------------------------------------------------------------


def load_memory_files(
    cwd: str, names: Tuple[str, ...] = ("AGENTS.md", "CLAUDE.md")
) -> str:
    """从 cwd 逐级向上收集项目记忆文件，外层在前、内层在后（内层更具体、优先级更高）。

    这是 2026 年所有主流 harness 的共同惯例：把"项目怎么构建、规范是什么、
    别碰哪些目录"写进仓库里的 markdown，Agent 每次启动自动读入——
    **记忆放在文件系统里，而不是模型里**。
    """
    chunks: List[str] = []
    d = os.path.abspath(cwd)
    seen = []
    while True:
        for name in names:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                seen.append(p)
                break  # 同目录多个候选只取第一个
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    for p in reversed(seen):  # 根目录的在前，越靠近 cwd 的越后（后者覆盖前者）
        with open(p, "r", encoding="utf-8") as f:
            chunks.append(f"<memory source={p!r}>\n{f.read().strip()}\n</memory>")
    return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# 压缩
# ---------------------------------------------------------------------------

_SUMMARIZE_PROMPT = """\
你是对话压缩器。把下面这段 Agent 工作记录压缩成一份"前情提要"，供同一个 Agent \
继续工作使用。必须保留：用户的原始目标与约束、已完成的事与关键结论、正在进行的事、\
尚未解决的问题、重要的文件路径/命令/数据。省略：寒暄、失败后已被纠正的弯路细节、\
工具输出的原文（只留结论）。用条目式中文输出。"""

# 把结构化消息列表转成纯文本对话记录，供摘要模型阅读理解。(json->str)
def _render_transcript(messages: Messages) -> str:
    lines = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            calls = "; ".join(f"{c.name}({c.args})" for c in m.tool_calls)
            lines.append(f"[assistant 调用工具] {calls}")
            if m.content:
                lines.append(f"[assistant] {m.content}")
        else:
            lines.append(f"[{m.role}] {m.content}")
    return "\n".join(lines)


def _safe_cut(messages: Messages, keep_recent: int) -> int:
    """返回切分点 cut：messages[cut:] 保留、messages[:cut] 送去摘要。

    向前回退直到保留段不以 ToolMessage 开头——否则这些结果对应的
    AIMessage(tool_calls) 被切走了，provider 会拒绝这种"孤儿工具结果"。
    """
    cut = max(0, len(messages) - keep_recent)
    while cut > 0 and isinstance(messages[cut], ToolMessage):
        cut -= 1
    return cut


@dataclass
class ContextManager:
    """上下文预算与装配策略。挂到 Agent 上即生效（不挂 = 全量历史直发）。"""

    max_context_tokens: int = 60000     # 预算上限（按模型窗口和成本定）
    compact_threshold: float = 0.75      # 用到预算的多少比例时触发压缩
    keep_recent: int = 12                # 压缩时保留最近多少条消息
    memory_cwd: Optional[str] = None     # 从哪里加载 AGENTS.md（None=不加载）
    extra_context: str = ""              # harness 注入的附加段（技能清单等，见第 7 章）

    # ---- 装配 ----

    def assemble(self, system_prompt: str, messages: Messages) -> Messages:
        """拼出真正发给模型的消息列表：[装配好的 SystemMessage] + 历史。"""
        parts = [system_prompt.strip()] if system_prompt else []
        if self.memory_cwd:
            mem = load_memory_files(self.memory_cwd)
            if mem:
                parts.append("# 项目记忆\n" + mem)
        if self.extra_context:
            parts.append(self.extra_context.strip())
        sys = "\n\n".join(parts)
        return ([SystemMessage(content=sys)] if sys else []) + list(messages)

    # ---- 压缩 ----

    def should_compact(self, messages: Messages) -> bool:
        return estimate_tokens(messages) > self.max_context_tokens * self.compact_threshold

    def compact(self, messages: Messages, model) -> Messages:
        """把早期历史摘要成一条"前情提要"，返回新的（短得多的）历史。"""
        cut = _safe_cut(messages, self.keep_recent)
        if cut <= 1:
            return list(messages)  # 没什么可压的
        head, tail = messages[:cut], messages[cut:]
        summary_ai = model.invoke(
            [
                SystemMessage(content=_SUMMARIZE_PROMPT),
                HumanMessage(content=_render_transcript(head)),
            ]
        )
        recap = HumanMessage(
            content="[前情提要——早前对话已压缩，以下是摘要]\n" + summary_ai.content
        )
        # 陈旧锚点陷阱：尾段 AI 消息的 usage 记的是【压缩前】完整历史的账，
        # 留着会让 estimate_tokens 永远高估。压缩即换纪元，旧账单一律清零
        # （预算统计不受影响——它在每轮 turn_end 时已累计过）。
        fresh_tail = [
            replace(m, usage=Usage()) if isinstance(m, AIMessage) else m
            for m in tail
        ]
        return [recap] + fresh_tail

    @staticmethod
    def clear_old_tool_results(
        messages: Messages, keep_recent: int = 8, placeholder: str = "[旧工具结果已清理，共 {n} 字符]"
    ) -> Messages:
        """更轻量的压缩：把久远的工具结果原文替换成占位符（保留消息结构）。

        直觉：一个工具结果被消费过一次后，模型极少需要再读它的**原文**——
        结论已经沉淀进后续的 AI 消息里了。这是 2026 年各家 harness / API
        都内置的第一档压缩手段（tool result clearing）。
        """
        out: Messages = []
        boundary = max(0, len(messages) - keep_recent)
        for i, m in enumerate(messages):
            if i < boundary and isinstance(m, ToolMessage) and len(m.content) > 200:
                out.append(
                    ToolMessage(
                        content=placeholder.format(n=len(m.content)),
                        tool_call_id=m.tool_call_id,
                        name=m.name,
                    )
                )
            else:
                out.append(m)
        return out
