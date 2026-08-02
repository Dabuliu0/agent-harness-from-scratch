"""钩子系统 —— 把 harness 的接缝开放给使用者（第 7 章）。

中间件（图运行时深入篇 R6）是**框架开发者**写代码扩展 Agent 的方式；
钩子是**产品使用者**不改一行框架代码、在关键时机挂入自己逻辑的方式
（对照 Claude Code 的 hooks：PreToolUse / PostToolUse / UserPromptSubmit / Stop…）。

我们实现五个时机：
- ``session_start``       会话开始：注入环境说明、加载状态
- ``user_prompt_submit``  用户消息提交前：校验、改写、拒绝
- ``pre_tool_use``        工具执行前：**可拦截、可改参**（编译进内核 gate）
- ``post_tool_use``       工具执行后：审计、通知、追加反馈
- ``stop``                run 结束时：收尾、断言"活干完没有"

钩子返回 ``HookResult``：block 拦截（附理由，理由会喂回模型）、
replace_args 改写参数、add_context 追加上下文。多个钩子按注册顺序依次生效。
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from tinycore.messages import ToolCall
from tinycore.tools import Gate

SESSION_START = "session_start"
USER_PROMPT_SUBMIT = "user_prompt_submit"
PRE_TOOL_USE = "pre_tool_use"
POST_TOOL_USE = "post_tool_use"
STOP = "stop"

HOOK_EVENTS = (SESSION_START, USER_PROMPT_SUBMIT, PRE_TOOL_USE, POST_TOOL_USE, STOP)


@dataclass
class HookResult:
    """一个钩子的裁决。全 None/False = "我没意见"。"""

    block: bool = False
    reason: str = ""
    replace_args: Optional[Dict[str, Any]] = None  # 仅 pre_tool_use 有效
    add_context: str = ""                          # 追加给模型的上下文


HookFn = Callable[[Dict[str, Any]], Optional[HookResult]]


class HookManager:
    def __init__(self) -> None:
        self._hooks: Dict[str, List[Tuple[str, HookFn]]] = {e: [] for e in HOOK_EVENTS}

    def register(self, event: str, fn: HookFn, matcher: str = "*") -> None:
        """挂一个钩子。matcher 对工具类事件按工具名 fnmatch（如 "bash"、"*_file"）。"""
        if event not in self._hooks:
            raise ValueError(f"未知钩子时机: {event}（可用: {HOOK_EVENTS}）")
        self._hooks[event].append((matcher, fn))

    def fire(self, event: str, payload: Dict[str, Any]) -> HookResult:
        """触发一个时机的所有钩子，合并裁决：
        任何一个 block 即 block；replace_args 后写覆盖先写；add_context 累积。"""
        merged = HookResult()
        tool_name = payload.get("call").name if isinstance(payload.get("call"), ToolCall) else None
        for matcher, fn in self._hooks.get(event, ()):
            if tool_name is not None and not fnmatch.fnmatch(tool_name, matcher):
                continue
            r = fn(payload)
            if r is None:
                continue
            if r.block:
                return HookResult(block=True, reason=r.reason or "被钩子拦截")
            if r.replace_args is not None:
                merged.replace_args = r.replace_args
                payload = {**payload, "call": ToolCall(
                    name=payload["call"].name, args=r.replace_args, id=payload["call"].id
                )}
            if r.add_context:
                merged.add_context += ("\n" if merged.add_context else "") + r.add_context
        return merged

    def as_gate(self, inner: Optional[Gate] = None) -> Gate:
        """把 pre_tool_use 编译成内核 gate，并与内层 gate（通常是权限）串联。

        顺序是刻意的：**钩子先跑、权限后跑**——钩子可能改写参数，
        权限必须裁决"改写后"的那次调用，否则会出现"审的是 A、执行的是 B"。
        """

        def gate(call: ToolCall):
            r = self.fire(PRE_TOOL_USE, {"call": call})
            if r.block:
                return r.reason
            if r.replace_args is not None:
                call = ToolCall(name=call.name, args=r.replace_args, id=call.id)
            if inner is not None:
                verdict = inner(call)
                if verdict is not None:
                    return verdict
            # 内层放行但参数被钩子改过 → 把改写后的 call 传给执行器
            return call if r.replace_args is not None else None

        return gate
