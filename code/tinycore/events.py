"""事件流 —— 内核对外的**唯一**输出通道（第 4 章）。

现代 harness 的一个核心设计：内核不直接"返回结果"，而是把执行过程吐成一条
事件流；UI 渲染、会话持久化（第 5 章）、可观测、审计……全都只是这条流的
不同**消费者**。层与层之间只隔一条事件流，这就是解耦的全部秘密。

事件用"一个类型 + 一个 payload dict"表达，刻意不给每种事件建类——
因为事件是**协议**不是对象模型：它要被序列化成 JSONL、推给前端、跨语言消费，
dict 是它的天然形态（对照：Claude Agent SDK 的 message 流、
OpenAI Agents SDK 的 stream events、LangGraph 的 stream_mode 输出）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict

# 事件类型清单（内核层）。harness 会在同一条流上追加自己的类型（如 approval.*）。
RUN_START = "run_start"            # 一次 run 开始 {input}
USER_MESSAGE = "user_message"      # 用户消息进入历史 {message}
TEXT_DELTA = "text_delta"          # 模型文本增量 {text}
ASSISTANT_MESSAGE = "assistant_message"  # 一条完整 AI 消息落定 {message}
TOOL_START = "tool_start"          # 工具开始执行 {call}
TOOL_RESULT = "tool_result"        # 工具结果落定 {message}
TURN_END = "turn_end"              # 一轮(模型+工具)结束 {turn, usage}
COMPACTION = "compaction"          # 发生了上下文压缩 {before_tokens, after_tokens}
RUN_END = "run_end"                # run 结束 {stop_reason, usage}
ERROR = "error"                    # 不可恢复错误 {error}


@dataclass
class Event:
    """一条事件。``data`` 里放什么由 ``type`` 决定（见上方清单注释）。"""

    type: str
    data: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        keys = ", ".join(self.data.keys())
        return f"Event({self.type}, {{{keys}}})"
