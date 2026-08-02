"""第 6 章示例:用 create_agent 一行搭出 Agent,并验证它继承了底层能力。

  示例 A:create_agent 基本用法(对照第 1 章的手写循环 / 第 3 章的手搭图)。
  示例 B:同一个 Agent 自动获得检查点能力(多轮对话记忆)。

运行:  python 06_create_agent.py   (需要配好模型与 key,见 _config.py)
"""
from _config import get_model

from tinyagent.agent import create_agent
from tinygraph.checkpoint import InMemorySaver
from tinygraph.messages import AIMessage, HumanMessage
from tinygraph.models import BaseChatModel
from tinygraph.tools import tool


@tool
def get_weather(city: str) -> str:
    """查询城市天气。"""
    return {"北京": "5°C, 有风", "上海": "12°C, 多云"}.get(city, "未知")


def demo_basic():
    print("=== 示例 A:create_agent 基本用法 ===")
    # 一行搭出 Agent —— 内部自动搭好 model↔tools 的图
    agent = create_agent(
        model=get_model(),
        tools=[get_weather],
        system_prompt="你是贴心的生活助手。",
    )

    result = agent.invoke({"messages": [HumanMessage("北京今天适合穿什么?")]})
    print("最终回答:", result["messages"][-1].content)
    print(f"(消息历史 {len(result['messages'])} 条)")
    print()


def demo_memory():
    print("=== 示例 B:自动继承检查点(多轮记忆)===")
    # 配上 checkpointer,Agent 就自动有了跨轮记忆:同一个 thread_id 的第二轮
    # 调用会从检查点把第一轮的消息历史续上,模型因此 "记得" 上文。
    agent = create_agent(
        model=get_model(),
        tools=[],
        system_prompt="你是友好的助手,回答简洁。",
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "chat-1"}}

    r1 = agent.invoke({"messages": [HumanMessage("你好,我叫小明")]}, config)
    print("第1轮:", r1["messages"][-1].content)

    # 第二轮:同一个 thread_id,历史从检查点自动续上 → 模型记得名字
    r2 = agent.invoke({"messages": [HumanMessage("我叫什么名字?")]}, config)
    print("第2轮:", r2["messages"][-1].content)


def demo_memory_mechanism():
    print("=== 示例 C:记忆机制(桩模型,无需 key)===")
    # 不联网的桩模型:只回显 "自己这一轮看到了多少条历史",方便观察累积
    class EchoModel(BaseChatModel):
        def bind_tools(self, tools):
            self._tools = list(tools)
            return self

        def invoke(self, messages):
            human = [m for m in messages if m.role == "user"]
            return AIMessage(
                content=f"我看到了 {len(messages)} 条历史,最后一句是「{human[-1].content}」")

    agent = create_agent(model=EchoModel(), tools=[], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "chat-1"}}

    r1 = agent.invoke({"messages": [HumanMessage("第一句话")]}, config)
    print("第1轮模型看到:", r1["messages"][-1].content)
    r2 = agent.invoke({"messages": [HumanMessage("第二句话")]}, config)
    print("第2轮模型看到:", r2["messages"][-1].content)
    print("第2轮完整历史:", [f"{m.role}:{m.content[:12]}" for m in r2["messages"]])

    r3 = agent.invoke({"messages": [HumanMessage("我是新对话")]},
                      {"configurable": {"thread_id": "chat-2"}})
    print("新 thread 历史长度:", len(r3["messages"]), "← 不串味")
    print()


if __name__ == "__main__":
    demo_memory_mechanism()      # 无需 key,先跑这个看记忆机制
    demo_basic()
    demo_memory()
