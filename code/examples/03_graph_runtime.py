"""第 3 章示例:用 tinygraph 图运行时跑两个图。

  示例 A:并行扇出 —— 三个节点并行写同一个 channel,reducer 确定性合并。
  示例 B:带环的 Agent 图 —— model 节点 ↔ tools 节点 循环,直到不再调工具。

运行:  python 03_graph_runtime.py
  示例 A 无需 API key;示例 B 需要配好模型与 key(见 _config.py)。
"""
from _config import get_model

from tinygraph.channels import add_reducer, last_value
from tinygraph.graph import END, START, StateGraph
from tinygraph.messages import HumanMessage, ToolMessage
from tinygraph.tools import tool


# ===========================================================================
# 示例 A:并行扇出 + reducer 合并
# ===========================================================================
def demo_parallel():
    print("=== 示例 A:并行扇出 ===")
    # 状态:results 是一个列表 channel(用 add_reducer 累加)
    schema = {"results": add_reducer}
    g = StateGraph(schema)

    def fetch_beijing(state):
        return {"results": [f"北京: 5°C"]}

    def fetch_shanghai(state):
        return {"results": [f"上海: 12°C"]}

    def fetch_guangzhou(state):
        return {"results": [f"广州: 22°C"]}

    for name, fn in [
        ("bj", fetch_beijing),
        ("sh", fetch_shanghai),
        ("gz", fetch_guangzhou),
    ]:
        g.add_node(name, fn)
        g.add_edge(START, name)   # 三个节点都从 START 激活 → 同一超步并行执行
        g.add_edge(name, END)

    app = g.compile()
    final = app.invoke({"results": []})
    # 不管三个节点谁先跑完,合并顺序由 tasks 固定顺序决定 → 结果可复现
    print("合并后的 results:", final["results"])
    print()


# ===========================================================================
# 示例 B:带环的 Agent 图(model ↔ tools 循环)
# ===========================================================================
@tool
def get_weather(city: str) -> str:
    """查询城市天气。"""
    return {"北京": "5°C, 有风"}.get(city, "未知")


def demo_agent_graph():
    print("=== 示例 B:带环的 Agent 图 ===")
    tools_by_name = {"get_weather": get_weather}

    model = get_model().bind_tools([get_weather])

    # 状态:messages 用 add_reducer 累加(对应 LangGraph 的 add_messages)
    schema = {"messages": add_reducer}
    g = StateGraph(schema)

    def call_model(state):
        ai = model.invoke(state["messages"])
        return {"messages": [ai]}

    def call_tools(state):
        last = state["messages"][-1]
        out = []
        for c in last.tool_calls:
            result = tools_by_name[c.name].invoke(c.args)
            out.append(ToolMessage(content=str(result), tool_call_id=c.id, name=c.name))
        return {"messages": out}

    def should_continue(state):
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    g.add_node("model", call_model)
    g.add_node("tools", call_tools)
    g.add_edge(START, "model")
    g.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "model")        # ← 这条边构成了 "环"

    app = g.compile()
    final = app.invoke({"messages": [HumanMessage("北京今天适合穿什么?")]})

    for m in final["messages"]:
        print(f"  {m!r}")
    print("\n最终回答:", final["messages"][-1].content)


if __name__ == "__main__":
    demo_parallel()
    demo_agent_graph()
