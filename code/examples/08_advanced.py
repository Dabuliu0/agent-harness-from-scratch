"""第 8 章示例:进阶 —— 多智能体(supervisor)与长期记忆(store)。

  示例 A:子图即节点 —— 一个 supervisor 图把任务分派给两个子 Agent。
  示例 B:长期记忆 Store —— 跨 thread 共享的记忆(区别于 thread 内的检查点)。

运行:  python 08_advanced.py   (无需 API key)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinygraph.channels import add_reducer, last_value
from tinygraph.graph import END, START, StateGraph


# ===========================================================================
# 示例 A:多智能体 supervisor(子图作为节点)
# ===========================================================================
def demo_supervisor():
    print("=== 示例 A:supervisor 多智能体 ===")
    # 关键洞察:一个 Agent 本身就是一张图;把它当作【另一张图的节点】,
    # 就得到多智能体。这里 supervisor 根据关键词路由到 "天气专家" 或 "数学专家"。
    schema = {"query": last_value, "answer": last_value, "route": last_value}

    def supervisor(state):
        q = state["query"]
        route = "weather" if "天气" in q else "math"
        return {"route": route}

    def weather_agent(state):
        # 真实场景这里会是 create_agent(...).invoke(...);简化为直接处理
        return {"answer": f"[天气专家] 关于「{state['query']}」:今天晴,25°C。"}

    def math_agent(state):
        return {"answer": f"[数学专家] 关于「{state['query']}」:答案是 42。"}

    g = StateGraph(schema)
    g.add_node("supervisor", supervisor)
    g.add_node("weather", weather_agent)
    g.add_node("math", math_agent)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", lambda s: s["route"],
                            {"weather": "weather", "math": "math"})
    g.add_edge("weather", END)
    g.add_edge("math", END)
    app = g.compile()

    for q in ["北京今天天气怎么样?", "帮我算个数"]:
        result = app.invoke({"query": q, "answer": "", "route": ""})
        print(f"  问:{q}\n  答:{result['answer']}")
    print()


# ===========================================================================
# 示例 B:长期记忆 Store(跨 thread)
# ===========================================================================
class InMemoryStore:
    """跨 thread 的长期记忆。对比检查点(thread 内短期状态),Store 用
    (namespace, key) 寻址,可跨会话共享 —— 比如 "用户画像"。"""

    def __init__(self):
        self._data = {}

    def put(self, namespace, key, value):
        self._data[(namespace, key)] = value

    def get(self, namespace, key):
        return self._data.get((namespace, key))

    def search(self, namespace):
        return {k[1]: v for k, v in self._data.items() if k[0] == namespace}


def demo_store():
    print("=== 示例 B:长期记忆 Store ===")
    store = InMemoryStore()
    ns = ("user", "alice")     # 命名空间:用户 alice 的记忆

    # 会话1(thread-1):记住用户偏好
    store.put(ns, "language", "中文")
    store.put(ns, "style", "简洁")
    print("  会话1 写入偏好:", store.search(ns))

    # 会话2(thread-2,完全不同的对话):仍能读到 alice 的长期偏好
    pref = store.get(ns, "language")
    print(f"  会话2 读到 alice 的语言偏好:{pref}(跨 thread 共享)")


if __name__ == "__main__":
    demo_supervisor()
    demo_store()
