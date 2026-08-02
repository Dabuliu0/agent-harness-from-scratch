"""LLM 抽象层:provider 无关的统一接口 + 两种真实实现。

对应 LangChain 的 BaseChatModel。核心契约只有一句话:**输入一组消息,输出一条
AIMessage(可能带 tool_calls)**。无论底层是 Anthropic、OpenAI,还是任何
OpenAI 兼容的服务(DeepSeek / Qwen / Kimi / 本地 vLLM 等),上层引擎只认这一个
接口——这正是 "机制与策略分离":引擎定义 "怎么用模型",模型实现决定 "对接谁"。

两种实现:
  - ``AnthropicChatModel``:走 Anthropic 原生 API,工具调用是 tool_use /
    tool_result content blocks,system 是独立参数。
  - ``OpenAIChatModel``:走 OpenAI 兼容的 chat.completions API,通过 ``base_url``
    即可指向 DeepSeek / Qwen / Kimi / 本地模型,工具调用走 ``tools`` 字段。

用工厂一行切换(对应 LangChain 的同名 ``init_chat_model``)::

    model = init_chat_model("anthropic:claude-sonnet-5")
    model = init_chat_model("deepseek:deepseek-chat")
    model = init_chat_model("openai:gpt-4.1")

API key 通过环境变量读取(``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` /
``DEEPSEEK_API_KEY`` 等),绝不写进代码。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Sequence, Tuple

from .messages import (
    AIMessage,
    HumanMessage,
    Messages,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from .tools import Tool


class BaseChatModel:
    """所有聊天模型的统一接口。

    上层(引擎、create_agent、中间件)只依赖这两个方法,完全不关心 provider。
    """

    def bind_tools(self, tools: Sequence[Tool]) -> "BaseChatModel":
        """绑定一组工具,让模型在 invoke 时把它们的 schema 传给底层 API。

        返回 self(链式调用),并把工具记在 ``self._tools`` 上。
        """
        self._tools = list(tools)
        return self

    def invoke(self, messages: Messages) -> AIMessage:
        """输入消息历史,返回一条 AIMessage(纯文本或带 tool_calls)。"""
        raise NotImplementedError

    @property
    def _bound_tools(self) -> List[Tool]:
        return getattr(self, "_tools", [])


# ===========================================================================
# Anthropic 实现
# ===========================================================================
#
# Anthropic 的消息格式有三处和 OpenAI 不同,这正是 "to_chat_format 不能一招通吃"
# 的原因,所以我们在这里专门做转换:
#   1. system 不是一条消息,而是 create() 的独立参数;
#   2. 模型发起工具调用 → assistant 消息里夹一个 type=tool_use 的 content block;
#   3. 工具结果 → user 消息里夹一个 type=tool_result 的 content block(不是单独
#      的 "tool" 角色)。


def _to_anthropic(messages: Messages) -> Tuple[str, List[Dict[str, Any]]]:
    """把内部消息转成 (system 文本, Anthropic messages 列表)。"""
    system_text = ""
    out: List[Dict[str, Any]] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            system_text += (("\n" if system_text else "") + m.content)
        elif isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            blocks: List[Dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for c in m.tool_calls:
                blocks.append(
                    {"type": "tool_use", "id": c.id, "name": c.name, "input": c.args}
                )
            out.append({"role": "assistant", "content": blocks})
        elif isinstance(m, ToolMessage):
            # 工具结果以 tool_result block 放进一条 user 消息
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.tool_call_id,
                            "content": m.content,
                        }
                    ],
                }
            )
    return system_text, out


class AnthropicChatModel(BaseChatModel):
    """对接 Anthropic 原生 API(需要 ``pip install anthropic`` 与 ``ANTHROPIC_API_KEY``)。"""

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        max_tokens: int = 2048,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        import anthropic  # 延迟导入:没装也不影响用 OpenAI 实现

        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = model
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        self._tools: List[Tool] = []

    def invoke(self, messages: Messages) -> AIMessage:
        system_text, msgs = _to_anthropic(messages)
        tools_param = [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in self._bound_tools
        ]
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_text or None,
            tools=tools_param or None,
            messages=msgs,
            **self.kwargs,
        )
        text, tool_calls = "", []
        for block in resp.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(block.name, dict(block.input), block.id))
        return AIMessage(content=text, tool_calls=tool_calls)


# ===========================================================================
# OpenAI 兼容实现(OpenAI / DeepSeek / Qwen / Kimi / 本地 vLLM ……)
# ===========================================================================
#
# 凡是 "OpenAI 兼容" 的服务,只要换 base_url 和 api_key 就能用同一份代码。
# OpenAI 的格式:system 是一条普通消息;工具调用走 assistant.tool_calls
# (function.arguments 是 JSON 字符串);工具结果是独立的 "tool" 角色消息。


def _to_openai(messages: Messages) -> List[Dict[str, Any]]:
    """把内部消息转成 OpenAI chat.completions 的 messages 列表。"""
    out: List[Dict[str, Any]] = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.name,
                                "arguments": json.dumps(c.args, ensure_ascii=False),
                            },
                        }
                        for c in m.tool_calls
                    ],
                }
            )
        elif isinstance(m, ToolMessage):
            out.append(
                {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
            )
        else:
            out.append({"role": m.role, "content": m.content})
    return out


class OpenAIChatModel(BaseChatModel):
    """对接任何 OpenAI 兼容服务(需要 ``pip install openai``)。

    通过 ``base_url`` 指向不同 provider::

        OpenAIChatModel("gpt-4.1")                              # OpenAI
        OpenAIChatModel("deepseek-chat",
                        base_url="https://api.deepseek.com",
                        api_key_env="DEEPSEEK_API_KEY")
    """

    def __init__(
        self,
        model: str = "gpt-4.1",
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        **kwargs: Any,
    ) -> None:
        from openai import OpenAI  # 延迟导入

        self.client = OpenAI(
            api_key=api_key or os.environ.get(api_key_env),
            base_url=base_url,
        )
        self.model = model
        self.kwargs = kwargs
        self._tools: List[Tool] = []

    def invoke(self, messages: Messages) -> AIMessage:
        tools_param = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._bound_tools
        ]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=_to_openai(messages),
            tools=tools_param or None,
            **self.kwargs,
        )
        msg = resp.choices[0].message
        tool_calls = []
        for tc in msg.tool_calls or []:
            args = json.loads(tc.function.arguments or "{}")
            tool_calls.append(ToolCall(tc.function.name, args, tc.id))
        return AIMessage(content=msg.content or "", tool_calls=tool_calls)


# ===========================================================================
# 工厂:一行字符串切换 provider(对应 LangChain 的 init_chat_model)
# ===========================================================================

# OpenAI 兼容服务的 base_url 与 key 环境变量。要加新 provider,在这里补一行即可。
_OPENAI_COMPATIBLE = {
    "openai": (None, "OPENAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "kimi": ("https://api.moonshot.cn/v1", "MOONSHOT_API_KEY"),
}


def init_chat_model(spec: str, **kwargs: Any) -> BaseChatModel:
    """按 ``"provider:model"`` 字符串构造模型。

    示例::

        init_chat_model("anthropic:claude-sonnet-5")
        init_chat_model("deepseek:deepseek-chat")
        init_chat_model("openai:gpt-4.1")
        init_chat_model("local:qwen2.5", base_url="http://localhost:8000/v1")
    """
    provider, _, model = spec.partition(":")
    if not model:
        raise ValueError(f"spec 应形如 'provider:model',收到: {spec!r}")

    if provider == "anthropic":
        return AnthropicChatModel(model, **kwargs)

    if provider in _OPENAI_COMPATIBLE:
        base_url, key_env = _OPENAI_COMPATIBLE[provider]
        kwargs.setdefault("base_url", base_url)
        kwargs.setdefault("api_key_env", key_env)
        return OpenAIChatModel(model, **kwargs)

    # 未知 provider 一律按 OpenAI 兼容处理(本地模型 / 自建网关),要求显式给 base_url
    return OpenAIChatModel(model, **kwargs)
