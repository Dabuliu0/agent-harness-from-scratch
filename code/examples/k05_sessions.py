"""第 5 章示例:会话与持久化——续聊、崩溃恢复、分叉,三场戏全离线。

python k05_sessions.py
"""
import shutil
import tempfile

import _config  # noqa: F401

from tinycore import Agent, AIMessage, Event, FakeModel, ToolCall, final_text, tool
from tinycore import events as ev
from tinyharness import SessionStore


@tool
def get_weather(city: str) -> str:
    """查天气。"""
    return f"{city} 晴 25°C"


def main():
    root = tempfile.mkdtemp(prefix="k05-")
    store = SessionStore(root)

    # ---- 第一幕:跑一轮并落盘 ----
    s = store.create("demo")
    agent = Agent(
        model=FakeModel([[ToolCall("get_weather", {"city": "上海"})], "上海晴,25 度。"]),
        tools=[get_weather],
    )
    for _ in s.record(agent.run("上海天气?")):
        pass
    print(f"第一幕:run 完成并落盘 → {s.path}")
    print(f"  日志 {len(s.events())} 个事件,重放出 {len(s.messages())} 条历史")

    # ---- 第二幕:'进程重启'后续聊 ----
    s2 = store.open("demo")                      # 全新对象,只认文件
    prior = s2.messages()
    agent2 = Agent(
        model=FakeModel([lambda msgs: f"我记得:此前有 {len([m for m in msgs if m.role != 'system'])-1} 条历史,"
                                      f"最后聊的是「{prior[-1].content[:12]}…」"]),
        tools=[get_weather],
    )
    for _ in s2.record(agent2.run("我们刚才聊了什么?", prior_messages=prior)):
        pass
    print(f"\n第二幕:恢复会话续聊 → {final_text(agent2.last_messages)}")

    # ---- 第三幕:崩溃恢复(半轮被修剪) ----
    s2.append(Event(ev.ASSISTANT_MESSAGE, {
        "message": AIMessage(content="", tool_calls=[ToolCall("get_weather", {"city": "广州"})])
    }))  # 模拟:AI 要了工具,结果没落盘进程就死了
    n_before = len(s2.messages())
    print(f"\n第三幕:注入半轮后重放 → {n_before} 条(残缺半轮已被修剪,历史仍合法)")

    # ---- 第四幕:分叉 ----
    fork = store.fork("demo")
    print(f"\n第四幕:分叉 → 新会话 {fork.id},与原会话从此各自发展")
    print(f"  仓库现有会话: {store.list()}")

    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
