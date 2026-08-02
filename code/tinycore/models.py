"""模型接口层（L0）—— provider 无关的统一契约 + 三种实现（第 1、4 章）。

契约只有两个方法：
- ``invoke(messages) -> AIMessage``           一次拿全量
- ``stream(messages) -> 迭代器``               先逐段吐 ``str`` 文本增量，
                                              **最后一项**是完整的 ``AIMessage``

三种实现：
- ``AnthropicChatModel``   Anthropic 原生 API（tool_use / tool_result blocks）
- ``OpenAIChatModel``      任何 OpenAI 兼容服务（OpenAI / DeepSeek / Qwen / Kimi / vLLM）
- ``FakeModel``            照剧本出牌的桩模型：离线跑通全部示例与测试，
                           也是理解"模型在循环里扮演什么角色"的最好教具

``init_chat_model("provider:model")`` 一行切换 provider；
``get_model()`` 从环境变量 ``TINYAGENT_MODEL`` 读取（沿用全书契约）。
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Iterator, List, Sequence, Tuple, Union

from .messages import (
    AIMessage,
    HumanMessage,
    Messages,
    SystemMessage,
    ToolCall,
    ToolMessage,
    Usage,
)
from .tools import Tool

StreamItem = Union[str, AIMessage]  # 文本增量……最后一项是完整消息


class BaseChatModel:
    """所有聊天模型的统一接口。上层（循环、harness）只依赖这几个方法。"""

    def bind_tools(self, tools: Sequence[Tool]) -> "BaseChatModel":
        self._tools = list(tools)
        return self

    def invoke(self, messages: Messages) -> AIMessage:
        raise NotImplementedError

    def stream(self, messages: Messages) -> Iterator[StreamItem]:
        """默认实现：不支持流式的模型退化为"一次性吐全文"。

        这保证了**任何**模型（包括 FakeModel）都能被流式消费者驱动——
        消费者的写法永远是"循环收增量，最后一项当结果"。
        """
        ai = self.invoke(messages)
        if ai.content:
            yield ai.content
        yield ai

    @property
    def _bound_tools(self) -> List[Tool]:
        return getattr(self, "_tools", [])


# ===========================================================================
# FakeModel：照剧本出牌的桩模型
# ===========================================================================


class FakeModel(BaseChatModel):
    """按剧本依次返回响应，用于离线示例、测试与教学。

    剧本元素可以是：
    - ``str``                       → 纯文本回答
    - ``AIMessage``                 → 原样返回（可带 tool_calls）
    - ``list[ToolCall]``            → 一条只带工具调用的 AIMessage
    - ``callable(messages) -> 上述`` → 按当轮输入动态决定（写断言/复读机都靠它）

    剧本耗尽后固定回答 "(剧本已结束)"——循环因此必然终止。
    """

    def __init__(self, script: Sequence[Any] = ()) -> None:
        self.script = list(script)
        self.calls: List[Messages] = []  # 记录每轮实际收到的消息，测试用
        self._tools: List[Tool] = []

    def invoke(self, messages: Messages) -> AIMessage:
        self.calls.append(list(messages))
        item = self.script.pop(0) if self.script else "(剧本已结束)"
        if callable(item):
            item = item(messages)
        if isinstance(item, AIMessage):
            ai = item
        elif isinstance(item, list):
            ai = AIMessage(content="", tool_calls=item)
        else:
            ai = AIMessage(content=str(item))
        if not ai.stop_reason:
            ai.stop_reason = "tool_use" if ai.tool_calls else "end_turn"
        # 用量按字符数粗估，让预算/压缩逻辑离线也能被触发
        est_in = sum(len(m.content or "") for m in messages) // 4
        ai.usage = Usage(input_tokens=est_in, output_tokens=len(ai.content) // 4)
        return ai


# ===========================================================================
# Anthropic 实现
# ===========================================================================
#
# 三处与 OpenAI 的格式差异（第 1 章详解）：
#   1. system 是 create() 的独立参数，不是一条消息；
#   2. 工具调用是 assistant 消息里的 tool_use content block；
#   3. 工具结果是 user 消息里的 tool_result block（没有独立 "tool" 角色）。


def _to_anthropic(messages: Messages) -> Tuple[str, List[Dict[str, Any]]]:
    system_text = ""
    out: List[Dict[str, Any]] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            system_text += ("\n" if system_text else "") + m.content
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
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.tool_call_id,
                            "content": m.content,
                            "is_error": m.is_error,
                        }
                    ],
                }
            )
    return system_text, out


class AnthropicChatModel(BaseChatModel):
    """Anthropic 原生 API（``pip install anthropic`` + ``ANTHROPIC_API_KEY``）。"""

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        max_tokens: int = 4096,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        import anthropic  # 延迟导入：没装也不影响其他实现

        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = model
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        self._tools: List[Tool] = []

    def _params(self, messages: Messages) -> Dict[str, Any]:
        system_text, msgs = _to_anthropic(messages)
        tools_param = [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in self._bound_tools
        ]
        p: Dict[str, Any] = dict(
            model=self.model, max_tokens=self.max_tokens, messages=msgs, **self.kwargs
        )
        if system_text:
            p["system"] = system_text
        if tools_param:
            p["tools"] = tools_param
        return p

    @staticmethod
    def _from_response(resp: Any) -> AIMessage:
        text, tool_calls = "", []
        for block in resp.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(block.name, dict(block.input), block.id))
        return AIMessage(
            content=text,
            tool_calls=tool_calls,
            stop_reason=resp.stop_reason or "",
            usage=Usage(resp.usage.input_tokens, resp.usage.output_tokens),
        )

    def invoke(self, messages: Messages) -> AIMessage:
        return self._from_response(self.client.messages.create(**self._params(messages)))

    def stream(self, messages: Messages) -> Iterator[StreamItem]:
        with self.client.messages.stream(**self._params(messages)) as s:
            for delta in s.text_stream:  # 逐段文本增量
                if delta:
                    yield delta
            yield self._from_response(s.get_final_message())


# ===========================================================================
# OpenAI 兼容实现（OpenAI / DeepSeek / Qwen / Kimi / 本地 vLLM ……）
# ===========================================================================


def _to_openai(messages: Messages) -> List[Dict[str, Any]]:
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
    """任何 OpenAI 兼容服务（``pip install openai``），换 ``base_url`` 即换 provider。"""

    def __init__(
        self,
        model: str = "gpt-5.1",
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        **kwargs: Any,
    ) -> None:
        from openai import OpenAI  # 延迟导入

        self.client = OpenAI(
            api_key=api_key or os.environ.get(api_key_env), base_url=base_url
        )
        self.model = model
        self.kwargs = kwargs
        self._tools: List[Tool] = []

    def _params(self, messages: Messages) -> Dict[str, Any]:
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
        p: Dict[str, Any] = dict(
            model=self.model, messages=_to_openai(messages), **self.kwargs
        )
        if tools_param:
            p["tools"] = tools_param
        return p

    def invoke(self, messages: Messages) -> AIMessage:
        resp = self.client.chat.completions.create(**self._params(messages))
        msg = resp.choices[0].message
        tool_calls = [
            ToolCall(tc.function.name, json.loads(tc.function.arguments or "{}"), tc.id)
            for tc in (msg.tool_calls or [])
        ]
        usage = Usage(resp.usage.prompt_tokens, resp.usage.completion_tokens) if resp.usage else Usage()
        return AIMessage(
            content=msg.content or "",
            tool_calls=tool_calls,
            stop_reason=resp.choices[0].finish_reason or "",
            usage=usage,
        )

    def stream(self, messages: Messages) -> Iterator[StreamItem]:
        # OpenAI 的流式要自己累积：文本按增量拼接，工具调用的参数是
        # **分片送达的 JSON 字符串**，必须按 index 归位、最后再整体解析。
        params = self._params(messages)
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}
        text, finish, usage = "", "", Usage()
        calls: Dict[int, Dict[str, Any]] = {}
        for chunk in self.client.chat.completions.create(**params):
            if chunk.usage:
                usage = Usage(chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish = choice.finish_reason
            delta = choice.delta
            if delta and delta.content:
                text += delta.content
                yield delta.content
            for tc in (delta.tool_calls or []) if delta else []:
                slot = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    slot["args"] += tc.function.arguments
        tool_calls = [
            ToolCall(s["name"], json.loads(s["args"] or "{}"), s["id"] or f"call_{i}")
            for i, s in sorted(calls.items())
        ]
        yield AIMessage(content=text, tool_calls=tool_calls, stop_reason=finish, usage=usage)


# ===========================================================================
# 工厂：一行字符串切换 provider
# ===========================================================================

_OPENAI_COMPATIBLE = {
    "openai": (None, "OPENAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "kimi": ("https://api.moonshot.cn/v1", "MOONSHOT_API_KEY"),
}


def init_chat_model(spec: str, **kwargs: Any) -> BaseChatModel:
    """按 ``"provider:model"`` 构造模型，如 ``anthropic:claude-sonnet-5``、
    ``deepseek:deepseek-chat``；未知 provider 按 OpenAI 兼容处理（需给 base_url）。"""
    provider, _, model = spec.partition(":")
    if provider == "fake":
        return FakeModel()
    if not model:
        raise ValueError(f"spec 应形如 'provider:model'，收到: {spec!r}")
    if provider == "anthropic":
        return AnthropicChatModel(model, **kwargs)
    if provider in _OPENAI_COMPATIBLE:
        base_url, key_env = _OPENAI_COMPATIBLE[provider]
        kwargs.setdefault("base_url", base_url)
        kwargs.setdefault("api_key_env", key_env)
        return OpenAIChatModel(model, **kwargs)
    return OpenAIChatModel(model, **kwargs)


def get_model(**kwargs: Any) -> BaseChatModel:
    """从环境变量 ``TINYAGENT_MODEL`` 读取 spec 构造模型（全书示例的统一入口）。"""
    spec = os.environ.get("TINYAGENT_MODEL", "").strip()
    if not spec:
        raise RuntimeError(
            "未设置 TINYAGENT_MODEL。示例：export TINYAGENT_MODEL='anthropic:claude-sonnet-5'"
        )
    return init_chat_model(spec, **kwargs)
