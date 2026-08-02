"""第 1 章示例:最小可用 Agent —— 一个朴素的 while 循环。

运行前先配好模型与 key(见 examples/_config.py 或 code/.env.example):
    export TINYAGENT_MODEL="anthropic:claude-sonnet-5"
    export ANTHROPIC_API_KEY="sk-ant-..."

运行:  python 01_minimal_agent.py
"""
from _config import get_model

from tinygraph.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from tinygraph.tools import tool


# ---- 1) 定义两个工具 -------------------------------------------------------
@tool
def get_weather(city: str) -> str:
    """查询指定城市今天的天气。"""
    fake_db = {"北京": "5°C, 有风", "上海": "12°C, 多云", "广州": "22°C, 晴"}
    return fake_db.get(city, f"{city}: 暂无数据")


@tool
def add(a: int, b: int) -> int:
    """计算两个整数之和。"""
    return a + b


# ---- 2) 朴素 Agent 循环 ----------------------------------------------------
def run_agent(model, tools, user_input, system_prompt=None, max_iters=10):
    """Agent 的本质:让模型在循环里反复决策、调用工具,直到不再调工具。"""
    tools_by_name = {t.name: t for t in tools}
    model = model.bind_tools(tools)

    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=user_input))

    calls = 0
    for _ in range(max_iters):
        ai_msg: AIMessage = model.invoke(messages)   # 模型决策(真实 API 调用)
        calls += 1
        messages.append(ai_msg)

        if not ai_msg.tool_calls:                    # 不再调工具 → 结束
            return ai_msg.content, messages, calls

        for call in ai_msg.tool_calls:               # 执行每个被请求的工具
            print(f"  [工具调用] {call.name}({call.args})")
            result = tools_by_name[call.name].invoke(call.args)
            print(f"  [工具结果] {result}")
            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call.id,
                    name=call.name,
                )
            )

    return "(达到最大迭代次数)", messages, calls


# ---- 3) 用真实模型驱动一次完整对话 -----------------------------------------
if __name__ == "__main__":
    # 模型自己决定:先调 get_weather("北京"),拿到结果后再给出最终回答。
    # 整个决策序列由真实 LLM 产生,不是我们写死的脚本。
    model = get_model()

    print("用户:北京今天适合穿什么?")
    answer, history, calls = run_agent(
        model=model,
        tools=[get_weather, add],
        user_input="北京今天适合穿什么?",
        system_prompt="你是一个贴心的生活助手。",
    )
    print(f"\nAgent 回答:{answer}")
    print(f"\n(本轮共 {calls} 次模型调用,消息历史 {len(history)} 条)")
