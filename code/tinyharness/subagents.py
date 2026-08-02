"""子代理 —— 上下文隔离的多智能体（第 8 章）。

关键认知：子代理的第一价值不是"并行"，而是**上下文隔离**——
子代理可以烧掉几万 token 去翻文件、试命令，主代理的历史里只落**一段结论**。
这是上下文工程三大技术之一（另两个：压缩、结构化笔记，见第 3 章）。

实现只需要一个工具：``task(agent, prompt)``。
- 主模型调用它 = 委派一个子任务（Claude Code 的 Task 工具、
  OpenAI Agents SDK 的 agent-as-tool，同一个思想）；
- 工具内部新建一个 **全新的** Agent 循环（独立历史、独立上下文预算），
  跑完只返回最终文本；
- 主模型一轮发多个 task 调用 → 内核的并行执行器（第 2 章）让子代理
  **自动并行**——多智能体没有引入任何新机制，全是旧零件的组合。

安全：子代理的权限只能比父代理**收窄**（gate 原样下传 + 工具集由定义方限定），
深度上限静态封顶（构造时就不给更深层发 task 工具）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from tinycore.context import ContextManager
from tinycore.events import Event
from tinycore.loop import Agent, final_text
from tinycore.tools import Gate, Tool


@dataclass
class SubagentDef:
    """一个可被委派的子代理档案。description 是写给**主模型**看的：
    它靠这段话决定"什么任务派给谁"——和工具的 description 同一个道理。"""

    name: str
    description: str
    system_prompt: str
    tools: Sequence[Tool] = ()
    model: Optional[object] = None      # None = 沿用主代理的模型
    max_turns: int = 20
    max_context_tokens: int = 40_000


def make_task_tool(
    agents: Sequence[SubagentDef],
    default_model: object,
    *,
    gate: Optional[Gate] = None,            # 父代理的闸门原样下传（权限只收不放）
    on_event: Optional[Callable[[str, Event], None]] = None,  # 子代理事件外送（观测）
    max_depth: int = 2,
    _depth: int = 0,
) -> Tool:
    """构造 ``task`` 工具。深度控制是**静态**的：只有 depth+1 < max_depth 时，
    子代理的工具箱里才会再放一把 task 工具——不存在"运行时数漏了"的可能。"""
    registry: Dict[str, SubagentDef] = {a.name: a for a in agents}
    roster = "\n".join(f"- {a.name}: {a.description}" for a in agents)

    def task(agent: str, prompt: str) -> str:
        spec = registry.get(agent)
        if spec is None:
            return f"[错误] 没有名为 {agent!r} 的子代理，可用: {', '.join(registry)}"
        tools = list(spec.tools)
        if _depth + 1 < max_depth and len(registry) > 0:
            tools.append(
                make_task_tool(
                    agents, default_model, gate=gate, on_event=on_event,
                    max_depth=max_depth, _depth=_depth + 1,
                )
            )
        sub = Agent(
            model=spec.model or default_model,
            tools=tools,
            system_prompt=spec.system_prompt,
            max_turns=spec.max_turns,
            context=ContextManager(max_context_tokens=spec.max_context_tokens),
            gate=gate,
            name=f"{spec.name}@{_depth + 1}",
        )
        gen = sub.run(prompt)
        for e in gen:
            if on_event is not None:
                on_event(sub.name, e)   # 主流程可见子代理进度，但不进主历史
        return final_text(sub.last_messages) or "(子代理没有产出文本)"

    return Tool(
        name="task",
        description=(
            "把一个独立子任务委派给专职子代理执行，返回其最终结论。"
            "适用于：需要大量探索/阅读但主对话只需要结论的任务；"
            "可以在一轮里发多个 task 调用并行推进互不依赖的子任务。\n"
            "agent: 子代理名字，必须是下列之一\n" + roster + "\n"
            "prompt: 交给子代理的完整任务描述（它看不到当前对话，必须自包含）"
        ),
        parameters={
            "type": "object",
            "properties": {"agent": {"type": "string"}, "prompt": {"type": "string"}},
            "required": ["agent", "prompt"],
        },
        func=task,
    )
