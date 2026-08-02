"""第 4 章示例:检查点 —— 持久化、断点恢复、时间旅行。

  示例 A:用 thread_id 持续一个会话,中途 "换台机器" 仍能续上历史。
  示例 B:用 SqliteSaver 跨进程持久化(模拟重启后从数据库恢复)。
  示例 C:时间旅行 —— 列出历史检查点,从过去某一步重新往下走。

运行:  python 04_checkpoint.py   (无需 API key)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinygraph.channels import add_reducer, last_value
from tinygraph.checkpoint import InMemorySaver, SqliteSaver
from tinygraph.graph import END, START, StateGraph


def build_counter_graph(checkpointer):
    """一个简单的计数图:每步给 count +1,连续走 3 步。"""
    schema = {"count": last_value, "log": add_reducer}
    g = StateGraph(schema)

    def step(state):
        n = state.get("count", 0) + 1
        return {"count": n, "log": [f"step->{n}"]}

    def route(state):
        return "step" if state["count"] < 3 else END

    g.add_node("step", step)
    g.add_edge(START, "step")
    g.add_conditional_edges("step", route, {"step": "step", END: END})
    return g.compile(checkpointer=checkpointer)


def demo_thread_and_resume():
    print("=== 示例 A:thread + 断点状态查询 ===")
    saver = InMemorySaver()
    app = build_counter_graph(saver)
    config = {"configurable": {"thread_id": "session-1"}}

    final = app.invoke({"count": 0, "log": []}, config)
    print("最终 count:", final["count"], "| log:", final["log"])

    # 用同一个 thread_id 查询最新检查点(模拟另一处代码 / 另一台机器读状态)
    snap = app.get_state(config)
    print("从检查点读回的 count:", snap.channel_values["count"])
    print()


def demo_sqlite_restart():
    print("=== 示例 B:SQLite 跨进程持久化 ===")
    db = "/tmp/tinygraph_demo.sqlite"
    Path(db).unlink(missing_ok=True)

    # —— 第一个 "进程":跑到一半假设崩溃,我们只跑出前两步的检查点 ——
    saver1 = SqliteSaver(db)
    app1 = build_counter_graph(saver1)
    config = {"configurable": {"thread_id": "job-42"}}
    # 手动只推进部分超步,模拟 "跑到一半"
    gen = app1.stream({"count": 0, "log": []}, config, stream_mode="values")
    next(gen); next(gen)        # 只走两步就 "崩溃"(丢弃生成器)
    del gen, app1, saver1
    print("进程1:跑了两步后崩溃。")

    # —— 第二个 "进程":重新打开同一个数据库,从检查点恢复继续 ——
    saver2 = SqliteSaver(db)
    app2 = build_counter_graph(saver2)
    snap = app2.get_state(config)
    print(f"进程2:从数据库恢复,当前 count={snap.channel_values['count']}, "
          f"接下来要跑 {snap.next_nodes}")
    final = app2.invoke(None, config)      # input=None → 从检查点续跑
    print("进程2:续跑完成,最终 count:", final["count"], "| log:", final["log"])
    print()


def demo_time_travel():
    print("=== 示例 C:时间旅行 ===")
    saver = InMemorySaver()
    app = build_counter_graph(saver)
    config = {"configurable": {"thread_id": "tt-1"}}
    app.invoke({"count": 0, "log": []}, config)

    history = app.get_state_history(config)
    print("历史检查点(最新在前):")
    for cp in history:
        print(f"  step={cp.step} count={cp.channel_values.get('count')} "
              f"next={cp.next_nodes} id={cp.checkpoint_id[-8:]}")

    # 挑一个 "过去" 的检查点(count==1 那一步),从它恢复继续走
    target = [cp for cp in history if cp.channel_values.get("count") == 1][0]
    print(f"\n从 count=1 的检查点(id ...{target.checkpoint_id[-8:]})重新往下走:")
    branch_config = {"configurable": {"thread_id": "tt-1",
                                      "checkpoint_id": target.checkpoint_id}}
    snap = app.get_state(branch_config)
    print("  读回该检查点 count:", snap.channel_values["count"],
          "next:", snap.next_nodes)


if __name__ == "__main__":
    demo_thread_and_resume()
    demo_sqlite_restart()
    demo_time_travel()
