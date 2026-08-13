"""主循环 —— 整个内核的心脏（第 1、4 章）。
"""
from __future__ import annotations

import queue
from dataclasses import asdict, dataclass, field
from typing import Generator, List, Optional, Sequence, Union

from . import events as ev
from .context import ContextManager, estimate_tokens
from .events import Event
from .messages import AIMessage, HumanMessage, Messages, SystemMessage, ToolCall, Usage
from .tools import Gate, Tool, run_tool_calls


@dataclass
class Agent:
    """一个可运行的 Agent = 模型 + 工具 + 指令 + （可选的）上下文策略与闸门。"""

    model: object                      # BaseChatModel
    tools: Sequence[Tool] = ()
    system_prompt: str = ""
    max_turns: int = 40                # 安全阀：防失控循环（对应 recursion_limit）
    context: Optional[ContextManager] = None
    gate: Optional[Gate] = None        # 权限/钩子接缝（第 6、7 章从这里挂入）
    parallel_tools: bool = True        # 是否并行执行工具
    stream_text: bool = True           # False 则整段返回（省事件量）
    name: str = "agent"

    def __post_init__(self) -> None:
        self._inbox: "queue.Queue[str]" = queue.Queue()  # 中途插话（steering）
        self._interrupted = False
        self.last_messages: Messages = []

    # ---- 外部控制（可从其他线程调用）----

    def interrupt(self) -> None:
        """请求停止。在下一个消息边界生效（协作式，不会撕裂历史）。"""
        self._interrupted = True

    def inject(self, text: str) -> None:
        """中途插入一条用户消息，在下一次调模型前生效（steering）。"""
        self._inbox.put(text)

    # ---- 便捷入口：只要最终结果，不关心过程 ----

    def invoke(
        self, user_input: Union[str, Messages], prior_messages: Optional[Messages] = None
    ) -> Messages:
        gen = self.run(user_input, prior_messages)
        for _ in gen:
            pass
        return self.last_messages

    # ---- 主循环 ----

    def run(
        self, user_input: Union[str, Messages], prior_messages: Optional[Messages] = None  # 支持普通用户传一句话，批量传多条消息;支持“接着上次聊”或“重新开始”两种模式
    ) -> Generator[Event, None, Messages]: # 实时吐 Event 事件，不支持外部传值，最终返回完整历史 Messages
        """执行一次 run，yield 事件流；生成器返回值 = 结束时的完整历史。"""
        self._interrupted = False
        
        # list(可迭代对象) 的作用：它会把传入的“可迭代对象”（比如列表、元组、字符串）拆开（迭代），然后把里面的每个元素依次收集到一个新列表里。
        messages: Messages = list(prior_messages or [])
        new_inputs = (
            [HumanMessage(content=user_input)]
            if isinstance(user_input, str)
            else list(user_input)
        ) 
        yield Event(ev.RUN_START, {"agent": self.name, "input": new_inputs[0].content if new_inputs else ""})
        for m in new_inputs:
            messages.append(m)
            yield Event(ev.USER_MESSAGE, {"message": m})

        model = self.model.bind_tools(list(self.tools))
        total = Usage()
        stop_reason = "max_turns"

        for turn in range(1, self.max_turns + 1):
            # ① 消息边界：先处理外部控制（插话 / 打断）
            while not self._inbox.empty():
                m = HumanMessage(content=self._inbox.get())
                messages.append(m)
                yield Event(ev.USER_MESSAGE, {"message": m})
            if self._interrupted:
                stop_reason = "interrupted"
                break

            # ② 上下文预算检查：超了先压缩，再装配本轮真正发给模型的列表
            if self.context and self.context.should_compact(messages):
                before = estimate_tokens(messages)
                messages = self.context.compact(messages, self.model)
                yield Event(
                    ev.COMPACTION,
                    {"before_tokens": before, "after_tokens": estimate_tokens(messages)},
                )
            if self.context:
                working = self.context.assemble(self.system_prompt, messages)     # 系统提示词 + 项目记忆 + 额外上下文 一起打包贴最前面
            elif self.system_prompt:
                working = [SystemMessage(content=self.system_prompt)] + messages  # 只把系统提示词贴最前面
            else:
                working = list(messages) # 原样复制一份

            # ③ 模型决策（流式：先增量后全量）
            try:
                # ----- 模式 1：流式输出（打字机效果） -----
                if self.stream_text:
                    ai: Optional[AIMessage] = None    # 用于收集最后吐出的完整 AIMessage

                    # model.stream() 会分两次吐东西，顺序固定：
                    # ① 先吐出无数个文本碎片（str）→ 实时推给前端
                    # ② 最后吐出一个完整的 AIMessage（包含内容 + tool_calls + usage）
                    for item in model.stream(working):
                        if isinstance(item, AIMessage):
                            ai = item     # 缓存最终完整对象（不对外输出）
                        elif item:        # 非空文本碎片 → 实时显示
                            yield Event(ev.TEXT_DELTA, {"text": item})

                    # 安全闸门：如果 model.stream() 没按约定返回 AIMessage，直接报错
                    # 防止后续代码因 ai 为 None 而崩溃
                    assert ai is not None, "stream() 必须以完整 AIMessage 收尾"
                # ----- 模式 2：整体输出（普通模式） -----     
                else:
                    # 阻塞等待，直到模型一次性返回完整 AIMessage
                    ai = model.invoke(working)
                    
            # ----- 异常处理：模型挂了必须停机 -----        
            except Exception as e:
                # 先发错误事件通知外界（前端/日志）
                yield Event(ev.ERROR, {"error": f"{type(e).__name__}: {e}"})
                # ★ 和工具执行不同：模型报错无法自救，必须向上抛出，终止整个循环
                raise

            messages.append(ai)
            total += ai.usage
            yield Event(ev.ASSISTANT_MESSAGE, {"message": ai})

            # ④ 终止判据：没有工具调用 = 模型给出最终回答
            if not ai.tool_calls:
                stop_reason = ai.stop_reason or "end_turn"
                break

            # ⑤ 执行工具（gate 把关；并行但结果保序），结果喂回历史
            for call in ai.tool_calls:
                yield Event(ev.TOOL_START, {"call": call})
            results = run_tool_calls(
                ai.tool_calls,
                {t.name: t for t in self.tools},
                gate=self.gate,
                parallel=self.parallel_tools,
            )
            for r in results:
                messages.append(r)
                yield Event(ev.TOOL_RESULT, {"message": r})

            yield Event(ev.TURN_END, {"turn": turn, "usage": asdict(ai.usage)})

        self.last_messages = messages
        yield Event(
            ev.RUN_END,
            {
                "stop_reason": stop_reason,
                "usage": asdict(total),
                "final_text": final_text(messages),
            },
        )
        return messages


def final_text(messages: Messages) -> str:
    """取最后一条 AI 消息的文本（一次 run 的"最终回答"）。"""
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return m.content
    return ""
