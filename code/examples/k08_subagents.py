"""第 8 章示例:子代理——上下文隔离肉眼可见,并行扇出秒表可测。

python k08_subagents.py
"""
import time

import _config  # noqa: F401

from tinycore import Agent, AIMessage, FakeModel, ToolCall, final_text, tool
from tinycore.models import BaseChatModel
from tinyharness import SubagentDef, make_task_tool


@tool
def grep_repo(pattern: str) -> str:
    """在仓库里搜索(演示用)。"""
    return f"src/a.py:12:{pattern} 命中\nsrc/b.py:34:{pattern} 命中\n" + "…large output…" * 50


@tool
def slow_probe(target: str) -> str:
    """探测一个目标(耗时 0.3 秒,演示并行)。"""
    time.sleep(0.3)
    return f"{target}: 正常"


def researcher_model():
    return FakeModel([
        [ToolCall("grep_repo", {"pattern": "TODO"})],
        [ToolCall("grep_repo", {"pattern": "FIXME"})],
        "结论:仓库共 4 处待办标记,集中在 src/a.py 与 src/b.py,建议列入下个迭代。",
    ])


def main():
    # ---- 第一幕:隔离——子代理烧掉的上下文不进主历史 ----
    print("== 隔离:子代理翻箱倒柜,主历史只落一段结论 ==")
    researcher = SubagentDef(
        name="researcher",
        description="仓库调研:搜索、阅读、汇总结论",
        system_prompt="你是调研员,只输出结论",
        tools=[grep_repo],
        model=researcher_model(),
    )
    parent_model = FakeModel([
        [ToolCall("task", {"agent": "researcher", "prompt": "调研仓库里的待办标记并给结论"})],
        lambda msgs: "调研完成:" + [m for m in msgs if m.role == "tool"][-1].content[:40],
    ])
    sub_events = []
    task_tool = make_task_tool([researcher], parent_model,
                               on_event=lambda name, e: sub_events.append((name, e.type)))
    parent = Agent(model=parent_model, tools=[task_tool])
    parent.invoke("这个仓库还有哪些待办?")

    print(f"  子代理内部事件 {len(sub_events)} 个(工具调用、结果……全在子代理窗口里)")
    print(f"  主代理历史仅 {len(parent.last_messages)} 条,其中工具结果 1 条(=子代理结论)")
    print(f"  最终回答: {final_text(parent.last_messages)}")

    # ---- 第二幕:并行扇出——一轮发 3 个 task,自动并行 ----
    print("\n== 并行:一轮 3 个 task,0.3 秒的活并行只花 ~0.3 秒 ==")

    class ProbeModel(BaseChatModel):
        """无状态剧本:还没探测就发调用,探测完就复述结果。
        三个并行子代理共享同一实例也不会竞态(不 pop 任何共享剧本)。"""

        def invoke(self, messages):
            tool_msgs = [m for m in messages if m.role == "tool"]
            if not tool_msgs:
                target = messages[-1].content[-4:]
                return AIMessage(content="", tool_calls=[ToolCall("slow_probe", {"target": target})])
            return AIMessage(content=tool_msgs[-1].content)

    prober = SubagentDef(
        name="prober", description="探测一个目标", system_prompt="探测员",
        tools=[slow_probe],
        model=ProbeModel(),
    )
    parent_model2 = FakeModel([
        [ToolCall("task", {"agent": "prober", "prompt": "探测 db-1"}),
         ToolCall("task", {"agent": "prober", "prompt": "探测 db-2"}),
         ToolCall("task", {"agent": "prober", "prompt": "探测 db-3"})],
        lambda msgs: " | ".join(m.content for m in msgs if m.role == "tool"),
    ])
    task2 = make_task_tool([prober], parent_model2)
    parent2 = Agent(model=parent_model2, tools=[task2])
    t0 = time.time()
    parent2.invoke("检查三个数据库")
    dt = time.time() - t0
    print(f"  耗时 {dt:.2f}s(串行要 0.9s+);结果保序: {final_text(parent2.last_messages)[:60]}")


if __name__ == "__main__":
    main()
