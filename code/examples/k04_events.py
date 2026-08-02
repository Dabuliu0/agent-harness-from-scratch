"""第 4 章示例:事件流——先裸看,再写一个 40 行渲染器,再演示插话。

python k04_events.py
"""
import _config  # noqa: F401

from tinycore import Agent, FakeModel, ToolCall, tool
from tinycore import events as ev


@tool
def get_weather(city: str) -> str:
    """查天气。"""
    return f"{city} 晴 25°C"


def make_model():
    return FakeModel([
        [ToolCall("get_weather", {"city": "北京"}), ToolCall("get_weather", {"city": "上海"})],
        "北京和上海都是晴天,25 度。",
    ])


def main():
    # ---- 第一遍:裸事件流 ----
    print("== 裸事件流 ==")
    agent = Agent(model=make_model(), tools=[get_weather])
    for e in agent.run("北京上海天气?"):
        print(f"  {e!r}")

    # ---- 第二遍:同一条流,渲染成终端 UI ----
    print("\n== 渲染器(事件流的另一个消费者) ==")

    def render(e):
        if e.type == ev.TEXT_DELTA:
            print(e.data["text"], end="", flush=True)
        elif e.type == ev.ASSISTANT_MESSAGE and e.data["message"].tool_calls:
            for c in e.data["message"].tool_calls:
                print(f"● {c.name}({next(iter(c.args.values()), '')})")
        elif e.type == ev.TOOL_RESULT:
            print(f"  ⎿ {e.data['message'].content[:50]}")
        elif e.type == ev.RUN_END:
            print(f"\n—— {e.data['stop_reason']}")

    agent = Agent(model=make_model(), tools=[get_weather])
    for e in agent.run("北京上海天气?"):
        render(e)

    # ---- 第三遍:中途插话(steering) ----
    print("\n== 中途插话:第一轮工具结果回来前注入一句 ==")
    model = FakeModel([
        [ToolCall("get_weather", {"city": "北京"})],
        lambda msgs: f"(第二轮)我看到的最后一条 user 是:「{[m for m in msgs if m.role=='user'][-1].content}」",
    ])
    agent = Agent(model=model, tools=[get_weather])
    gen = agent.run("北京天气?")
    for e in gen:
        if e.type == ev.TOOL_RESULT:
            agent.inject("顺便用华氏度再说一遍")   # 消息边界后、下轮调模型前生效
        render(e)


if __name__ == "__main__":
    main()
