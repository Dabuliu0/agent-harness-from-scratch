"""中间件系统 —— 现代 Agent 的核心可扩展点(对应 LangChain 1.0 的 AgentMiddleware)。

中间件让你在 Agent 生命周期的 6 个时机插入自定义逻辑,而不必改 Agent 内核:

    before_agent     —— Agent 开始前(加载记忆、校验输入)
    before_model     —— 每次调模型前(裁剪/摘要历史、注入提示)
    wrap_model_call  —— 包裹模型调用(改写请求/响应、换模型)
    wrap_tool_call   —— 包裹工具调用(审批、改写参数、脱敏)
    after_model      —— 每次模型响应后(校验输出、加护栏)
    after_agent      —— Agent 结束后(保存结果、清理)

wrap_* 用 "handler-passing" 模式:你拿到 (请求, handler),可以改请求、
决定调不调 handler、改 handler 的返回值 —— 完全掌控这一层。
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from tinygraph.messages import AIMessage, BaseMessage, ToolCall


class AgentMiddleware:
    """所有中间件的基类。默认所有钩子都是 "什么都不做"(返回 None = 不改)。

    子类按需重写其中几个钩子即可。"""

    # —— 生命周期两端 ——
    def before_agent(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return None

    def after_agent(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return None

    # —— 模型调用前后 ——
    def before_model(self, messages: List[BaseMessage]) -> Optional[List[BaseMessage]]:
        return None

    def after_model(self, message: AIMessage) -> Optional[AIMessage]:
        return None

    # —— 包裹型钩子(handler-passing)——
    def wrap_model_call(
        self,
        messages: List[BaseMessage],
        handler: Callable[[List[BaseMessage]], AIMessage],
    ) -> AIMessage:
        return handler(messages)

    def wrap_tool_call(
        self,
        call: ToolCall,
        handler: Callable[[], Any],
    ) -> Any:
        return handler()


# ===========================================================================
# 三个实战中间件
# ===========================================================================
import re


class SummarizationMiddleware(AgentMiddleware):
    """当消息历史过长时,把较早的消息压缩成一条摘要,控制上下文长度。

    对应 LangChain 的 SummarizationMiddleware。这里用 "条数阈值 + 简单拼接"
    模拟,真实实现会用 LLM 生成摘要、按 token 阈值触发。
    """

    def __init__(self, max_messages: int = 6, keep_recent: int = 2) -> None:
        self.max_messages = max_messages
        self.keep_recent = keep_recent

    def before_model(self, messages: List[BaseMessage]) -> Optional[List[BaseMessage]]:
        if len(messages) <= self.max_messages:
            return None
        from tinygraph.messages import SystemMessage

        head = messages[: -self.keep_recent]
        recent = messages[-self.keep_recent :]
        summary = "；".join(
            f"{m.role}:{m.content[:20]}" for m in head if m.content
        )
        sys = SystemMessage(content=f"[早前对话摘要] {summary}")
        return [sys] + recent


class PIIRedactionMiddleware(AgentMiddleware):
    """在把消息发给模型【前】,对敏感信息(手机号/邮箱)做脱敏。

    对应 LangChain 的 PIIMiddleware。"""

    PHONE = re.compile(r"\b1[3-9]\d{9}\b")
    EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

    def before_model(self, messages: List[BaseMessage]) -> Optional[List[BaseMessage]]:
        import dataclasses

        redacted = []
        for m in messages:
            text = m.content or ""
            text = self.PHONE.sub("[手机号]", text)
            text = self.EMAIL.sub("[邮箱]", text)
            redacted.append(dataclasses.replace(m, content=text) if text != m.content else m)
        return redacted


class HumanApprovalMiddleware(AgentMiddleware):
    """对指定的敏感工具,在执行【前】要求人工审批。

    对应 LangChain 的 HumanInTheLoopMiddleware。这里用一个 approver 回调模拟
    审批决策(真实场景会配合第 5 章的 interrupt() 暂停等人工输入)。
    """

    def __init__(self, sensitive_tools: List[str], approver: Callable[[ToolCall], bool]) -> None:
        self.sensitive_tools = set(sensitive_tools)
        self.approver = approver

    def wrap_tool_call(self, call: ToolCall, handler: Callable[[], Any]) -> Any:
        if call.name in self.sensitive_tools:
            if not self.approver(call):
                return f"[已拒绝] 工具 {call.name} 未获批准,跳过执行。"
        return handler()
