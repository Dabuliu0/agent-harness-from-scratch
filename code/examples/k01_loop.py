"""第 1 章示例:最小 Agent 循环。

离线跑(默认,FakeModel 照剧本):  python k01_loop.py
接真实模型:                      python k01_loop.py --real   (需 TINYAGENT_MODEL 与 key)
"""
import sys

import _config  # noqa: F401  # 注入包路径 + 加载 .env

from tinycore import FakeModel, HumanMessage, SystemMessage, ToolCall, ToolMessage, tool


@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气。city 传中文城市名,如 "北京"。"""
    return {"北京": "5°C, 有风", "上海": "12°C, 小雨"}.get(city, f"{city} 晴, 25°C")


# ---- 教学版主循环(与第 1 章正文逐行一致) ----

def run_agent(model, tools, user_input, system_prompt=None, max_turns=10):
    tools_by_name = {t.name: t for t in tools}
    model = model.bind_tools(tools)

    messages = [HumanMessage(content=user_input)]
    for _ in range(max_turns):
        working = ([SystemMessage(content=system_prompt)] if system_prompt else []) + messages
        ai = model.invoke(working)                    # ① 模型决策
        messages.append(ai)

        if not ai.tool_calls:                         # ② 无工具调用 → 结束
            return ai.content, messages

        for call in ai.tool_calls:                    # ③ 执行工具,结果喂回
            print(f"  [工具调用] {call.name}({call.args})")
            result = tools_by_name[call.name].invoke(call.args)
            print(f"  [工具结果] {result}")
            messages.append(ToolMessage(content=result, tool_call_id=call.id, name=call.name))

    return "(达到最大轮数)", messages


def main():
    if "--real" in sys.argv:
        model = _config.get_kernel_model()
        print(f"使用真实模型: {type(model).__name__}({model.model})")
    else:
        model = FakeModel([
            [ToolCall("get_weather", {"city": "北京"})],
            "北京今天 5°C 且有风,建议穿厚外套加围巾。",
        ])
        print("使用 FakeModel(离线剧本;加 --real 切真实模型)")

    print("\n用户: 北京今天适合穿什么?")
    answer, history = run_agent(
        model, [get_weather], "北京今天适合穿什么?", "你是贴心的生活助手。"
    )
    print(f"\nAgent 回答: {answer}")
    print(f"\n最终历史({len(history)} 条):")
    for m in history:
        tag = f"{m.role}"
        if getattr(m, "tool_calls", None):
            tag += "(调工具)"
        print(f"  {tag}: {(m.content or '')[:50]}")


if __name__ == "__main__":
    main()
