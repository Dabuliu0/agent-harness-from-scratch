"""第 7 章示例:中间件系统。

  示例 A:自定义 logging 中间件 —— 看清 6 个钩子的触发时机与顺序。
  示例 B:三个实战中间件 —— 摘要、PII 脱敏、工具审批。
  示例 C:中间件顺序的语义 —— 脱敏/摘要排错会漏手机号(桩模型,无需 key)。
  示例 D:钩子触发顺序(桩模型,无需 key,等价示例 A 但不联网)。

运行:  python 07_middleware.py   (A/B 需配好模型与 key;C/D 无需 key,见 _config.py)
"""
import re

from _config import get_model

from tinyagent.agent import create_agent
from tinyagent.middleware import (
    AgentMiddleware,
    HumanApprovalMiddleware,
    PIIRedactionMiddleware,
    SummarizationMiddleware,
)
from tinygraph.messages import AIMessage, HumanMessage, ToolCall
from tinygraph.models import BaseChatModel
from tinygraph.tools import tool


@tool
def delete_file(path: str) -> str:
    """删除指定文件(敏感操作)。"""
    return f"已删除 {path}"


@tool
def read_file(path: str) -> str:
    """读取文件内容。"""
    return f"{path} 的内容..."


# ===========================================================================
# 示例 A:看清 6 个钩子的触发顺序
# ===========================================================================
class LoggingMiddleware(AgentMiddleware):
    def __init__(self, tag):
        self.tag = tag

    def before_agent(self, state):
        print(f"  [{self.tag}] before_agent")
    def before_model(self, messages):
        print(f"  [{self.tag}] before_model ({len(messages)} 条消息)")
    def wrap_model_call(self, messages, handler):
        print(f"  [{self.tag}] wrap_model_call → 进入")
        ai = handler(messages)
        print(f"  [{self.tag}] wrap_model_call ← 返回")
        return ai
    def after_model(self, message):
        print(f"  [{self.tag}] after_model")
    def wrap_tool_call(self, call, handler):
        print(f"  [{self.tag}] wrap_tool_call: {call.name}")
        return handler()
    def after_agent(self, state):
        print(f"  [{self.tag}] after_agent")


def demo_hook_order():
    print("=== 示例 A:钩子触发顺序 ===")
    agent = create_agent(
        model=get_model(),
        tools=[read_file],
        system_prompt="用户让你读文件时,调用 read_file 工具。",
        middleware=[LoggingMiddleware("MW1"), LoggingMiddleware("MW2")],
    )
    agent.invoke({"messages": [HumanMessage("读一下 a.txt")]})
    print()


# ===========================================================================
# 示例 B:三个实战中间件
# ===========================================================================
def demo_real_middlewares():
    print("=== 示例 B:摘要 + PII脱敏 + 工具审批 ===")

    # 审批器:拒绝删除操作
    def approver(call):
        decision = call.args.get("path") != "important.db"
        print(f"  [审批] {call.name}({call.args}) → {'批准' if decision else '拒绝'}")
        return decision

    agent = create_agent(
        model=get_model(),
        tools=[delete_file],
        system_prompt="用户要求删除文件时,调用 delete_file 工具。",
        middleware=[
            PIIRedactionMiddleware(),
            SummarizationMiddleware(max_messages=6, keep_recent=2),
            HumanApprovalMiddleware(sensitive_tools=["delete_file"], approver=approver),
        ],
    )

    # 输入里带敏感信息 —— 会被 PII 中间件脱敏后才发给模型
    result = agent.invoke({"messages": [
        HumanMessage("我的手机号是 13812345678,请删掉 important.db")
    ]})
    # delete_file 是敏感工具且审批被拒 → 返回拒绝提示
    tool_msgs = [m for m in result["messages"] if m.role == "tool"]
    print("  工具结果:", tool_msgs[0].content if tool_msgs else "(无)")
    print("  最终回答:", result["messages"][-1].content)


# ===========================================================================
# 示例 C:中间件顺序的语义(脱敏 vs 摘要)—— 桩模型,无需 key
# ===========================================================================
class _SpyModel(BaseChatModel):
    """把"模型实际收到的消息"原样回显,用来观察中间件流水线的产出。"""
    def bind_tools(self, tools):
        self._tools = list(tools)
        return self

    def invoke(self, messages):
        return AIMessage(content="模型看到:" + " | ".join(m.content or "" for m in messages))


def demo_middleware_order():
    print("=== 示例 C:中间件顺序的语义(无需 key)===")
    # 手机号在第 14 个字符,摘要的 content[:20] 会把它截断 → 脱敏正则失配
    history = [
        HumanMessage("请记录联系方式如下手机号 13812345678 谢谢"),
        HumanMessage("帮我记一下"), HumanMessage("第三句"), HumanMessage("第四句"),
        HumanMessage("第五句话现在问你"),
    ]

    def leak(content):
        m = re.search(r"\d{6,}", content)
        return m.group() if m else "无"

    a1 = create_agent(model=_SpyModel(), middleware=[
        PIIRedactionMiddleware(),
        SummarizationMiddleware(max_messages=4, keep_recent=2)])
    c1 = a1.invoke({"messages": history})["messages"][-1].content
    print("  顺序①脱敏在前 → 泄漏:", leak(c1))

    a2 = create_agent(model=_SpyModel(), middleware=[
        SummarizationMiddleware(max_messages=4, keep_recent=2),
        PIIRedactionMiddleware()])
    c2 = a2.invoke({"messages": history})["messages"][-1].content
    print("  顺序②摘要在前 → 泄漏:", leak(c2), " ← 残缺号码漏进了模型")
    print()


# ===========================================================================
# 示例 D:钩子触发顺序(桩模型,无需 key;等价示例 A 但不联网)
# ===========================================================================
class _ScriptedModel(BaseChatModel):
    """第一次请求工具、第二次纯文本回答 —— 制造"调一次工具再回答"的轨迹。"""
    def __init__(self):
        self.n = 0

    def bind_tools(self, tools):
        self._tools = list(tools)
        return self

    def invoke(self, messages):
        self.n += 1
        if self.n == 1:
            return AIMessage(content="", tool_calls=[ToolCall("read_file", {"path": "a.txt"})])
        return AIMessage(content="文件读好了")


def demo_hook_order_stub():
    print("=== 示例 D:钩子触发顺序(无需 key)===")
    agent = create_agent(
        model=_ScriptedModel(),
        tools=[read_file],
        middleware=[LoggingMiddleware("MW1"), LoggingMiddleware("MW2")],
    )
    agent.invoke({"messages": [HumanMessage("读一下 a.txt")]})
    print()


if __name__ == "__main__":
    demo_hook_order_stub()         # 无需 key
    demo_middleware_order()        # 无需 key
    demo_hook_order()              # 需要 key
    demo_real_middlewares()        # 需要 key
