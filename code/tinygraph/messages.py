"""消息与 content blocks —— Agent 的统一消息表示。

对应 LangChain 的 langchain_core.messages。我们用 dataclass 实现一组最小但
完整的消息类型,并支持现代的 "原生工具调用(tool_calls)" 协议。
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _new_id() -> str:
    return str(uuid.uuid4())


@dataclass
class ToolCall:
    """模型发出的一次工具调用请求。

    这是 "原生工具调用" 协议的核心数据结构:模型不再输出 "Action: foo(1)"
    这种文本让我们正则解析,而是直接给出结构化的 name + args。
    """

    name: str
    args: Dict[str, Any]
    id: str = field(default_factory=_new_id)


@dataclass
class BaseMessage:
    content: str = ""
    id: str = field(default_factory=_new_id)

    @property
    def role(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        text = self.content if len(self.content) <= 60 else self.content[:57] + "..."
        return f"{type(self).__name__}({text!r})"


@dataclass
class SystemMessage(BaseMessage):
    """系统指令,定义 Agent 的角色与行为。"""

    @property
    def role(self) -> str:
        return "system"


@dataclass
class HumanMessage(BaseMessage):
    """用户输入。"""

    @property
    def role(self) -> str:
        return "user"


@dataclass
class AIMessage(BaseMessage):
    """模型的输出。可能是纯文本回答,也可能携带 tool_calls(要求调用工具)。"""

    tool_calls: List[ToolCall] = field(default_factory=list)

    @property
    def role(self) -> str:
        return "assistant"

    def __repr__(self) -> str:
        if self.tool_calls:
            calls = ", ".join(f"{c.name}({c.args})" for c in self.tool_calls)
            return f"AIMessage(tool_calls=[{calls}])"
        return super().__repr__()


@dataclass
class ToolMessage(BaseMessage):
    """工具执行结果,通过 tool_call_id 关联到发起它的那次 ToolCall。"""

    tool_call_id: str = ""
    name: str = ""

    @property
    def role(self) -> str:
        return "tool"


AnyMessage = BaseMessage
Messages = List[BaseMessage]

# 用于从 JSON(检查点)还原消息对象的类型注册表
_MESSAGE_TYPES = {
    "SystemMessage": SystemMessage,
    "HumanMessage": HumanMessage,
    "AIMessage": AIMessage,
    "ToolMessage": ToolMessage,
}


def revive(obj: Any) -> Any:
    """递归地把 "带 __type__ 标记的 dict" 还原成消息 / ToolCall 对象。

    检查点经 JSON 序列化后,消息对象变成了 dict;恢复执行时需要还原回对象。
    这正是 "可移植检查点" 的关键一环:状态要能 序列化→存盘→读回→还原。
    """
    if isinstance(obj, list):
        return [revive(x) for x in obj]
    if isinstance(obj, dict):
        t = obj.get("__type__")
        if t in _MESSAGE_TYPES:
            data = {k: revive(v) for k, v in obj.items() if k != "__type__"}
            if "tool_calls" in data:
                data["tool_calls"] = [
                    ToolCall(**tc) if isinstance(tc, dict) else tc
                    for tc in data["tool_calls"]
                ]
            return _MESSAGE_TYPES[t](**data)
        return {k: revive(v) for k, v in obj.items()}
    return obj


def to_chat_format(messages: Messages) -> List[Dict[str, Any]]:
    """把我们的消息对象转成 provider API 常见的 dict 格式(便于对接真实模型)。

    这里采用类 OpenAI/Anthropic 的通用结构;对接具体 provider 时按需调整。
    """
    out: List[Dict[str, Any]] = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": m.content,
                    "tool_calls": [
                        {"id": c.id, "name": c.name, "args": c.args}
                        for c in m.tool_calls
                    ],
                }
            )
        elif isinstance(m, ToolMessage):
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": m.tool_call_id,
                    "name": m.name,
                    "content": m.content,
                }
            )
        else:
            out.append({"role": m.role, "content": m.content})
    return out
