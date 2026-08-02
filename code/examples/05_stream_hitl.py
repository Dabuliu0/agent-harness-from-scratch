"""第 5 章示例:流式 与 人在回路(HITL)。

  示例 A:多种 streaming 模式(values / updates)。
  示例 B:人在回路 —— Agent 想执行危险操作前 interrupt() 暂停,
          外部审批后用 Command(resume=...) 恢复继续。

运行:  python 05_stream_hitl.py   (无需 API key)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinygraph.channels import add_reducer, last_value
from tinygraph.checkpoint import InMemorySaver
from tinygraph.graph import END, START, StateGraph
from tinygraph.interrupt import Command, interrupt


# ===========================================================================
# 示例 A:流式的两种模式
# ===========================================================================
def demo_streaming():
    print("=== 示例 A:流式模式 ===")
    schema = {"count": last_value, "log": add_reducer}
    g = StateGraph(schema)

    def step(state):
        n = state.get("count", 0) + 1
        return {"count": n, "log": [f"step{n}"]}

    g.add_node("step", step)
    g.add_edge(START, "step")
    g.add_conditional_edges("step", lambda s: "step" if s["count"] < 3 else END,
                            {"step": "step", END: END})
    app = g.compile()

    print("stream_mode='updates'(每个超步:跑了谁 + 当时状态):")
    for ev in app.stream({"count": 0, "log": []}, stream_mode="updates"):
        print(f"  超步{ev['__step__']}: nodes={ev['nodes']} count={ev['state']['count']}")

    print("stream_mode='values'(每个超步:完整状态快照):")
    for snap in app.stream({"count": 0, "log": []}, stream_mode="values"):
        print(f"  count={snap['count']} log={snap['log']}")
    print()


# ===========================================================================
# 示例 B:人在回路 —— 危险操作前暂停等审批
# ===========================================================================
def demo_human_in_the_loop():
    print("=== 示例 B:人在回路(HITL)===")
    schema = {"action": last_value, "result": last_value}
    g = StateGraph(schema)

    def plan(state):
        # Agent 决定要执行一个危险操作
        return {"action": "DELETE prod database"}

    def execute(state):
        # 执行前先请求人工审批:interrupt() 会暂停图,把待审批内容抛给外部
        decision = interrupt({"approve_action": state["action"]})
        if decision == "approve":
            return {"result": f"已执行: {state['action']}"}
        return {"result": f"已拒绝: {state['action']}"}

    g.add_node("plan", plan)
    g.add_node("execute", execute)
    g.add_edge(START, "plan")
    g.add_edge("plan", "execute")
    g.add_edge("execute", END)

    saver = InMemorySaver()          # 中断/恢复必须配 checkpointer
    app = g.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "approval-1"}}

    # —— 第一次运行:跑到 execute 时 interrupt,暂停 ——
    print("启动 Agent...")
    for ev in app.stream({"action": "", "result": ""}, config):
        if "__interrupt__" in ev:
            print(f"  ⏸  图已暂停,等待人工审批:{ev['__interrupt__']}")

    # 此刻进程其实可以直接退出,状态已存进检查点。这里直接演示恢复。
    snap = app.get_state(config)
    print(f"  检查点显示 next_nodes={snap.next_nodes}(execute 还没执行)")

    # —— 人点了 "批准",用 Command(resume=...) 恢复 ——
    print("人工决定:approve")
    final = app.invoke(Command(resume="approve"), config)
    print(f"  ▶  恢复执行,结果:{final['result']}")


# ===========================================================================
# 示例 C:副作用陷阱 —— interrupt() 之前的代码会在恢复时重跑
# ===========================================================================
def demo_side_effect_pitfall():
    print("=== 示例 C:副作用陷阱(interrupt 之前的代码重跑)===")
    charge_calls = {"n": 0}          # 模拟扣款的副作用计数器

    schema = {"action": last_value, "result": last_value}
    g = StateGraph(schema)

    def plan(state):
        return {"action": "买一杯咖啡"}

    def execute(state):
        charge_calls["n"] += 1                       # ← 副作用放在 interrupt 之前(错误示范)
        print(f"  [副作用] charge_money() 第 {charge_calls['n']} 次执行")
        decision = interrupt({"approve": state["action"]})
        if decision == "approve":
            return {"result": f"已下单: {state['action']}"}
        return {"result": "已取消"}

    g.add_node("plan", plan)
    g.add_node("execute", execute)
    g.add_edge(START, "plan")
    g.add_edge("plan", "execute")
    g.add_edge("execute", END)

    app = g.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "bug-demo"}}

    for ev in app.stream({"action": "", "result": ""}, config):
        if "__interrupt__" in ev:
            print("  暂停,等待审批")
    app.invoke(Command(resume="approve"), config)
    print(f"  charge_money 总共被调用了 {charge_calls['n']} 次  ← 只下一单却扣两次款")
    print("  修复:把 interrupt() 放节点最前面,副作用放在拿到 resume 之后\n")


if __name__ == "__main__":
    demo_streaming()
    demo_human_in_the_loop()
    print()
    demo_side_effect_pitfall()
