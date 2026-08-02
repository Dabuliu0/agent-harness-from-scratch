"""create_agent —— 在 tinygraph 之上重建的 Agent 抽象(对应 LangChain 1.0)。

第 3 章我们手动搭过 "model 节点 ↔ tools 节点" 的图。每次写 Agent 都重搭一遍
太啰嗦。create_agent 把这套标准结构【自动】搭好:你只给 model + tools,它
返回一个编译好的图,自带第 4/5 章的全部能力(检查点、中断、流式)。

state/messages 契约:输入 {"messages": [...]},输出也以 messages 为中心。

中间件(第 7 章)通过 middleware 参数注入;本文件已预留所有钩子的调用点。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from tinygraph.channels import add_reducer, last_value
from tinygraph.graph import END, START, StateGraph
from tinygraph.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from tinygraph.tools import Tool


class Agent:
    """create_agent 的返回物:包装了底层 CompiledGraph,提供 invoke/stream。"""

    def __init__(self, compiled, middleware: List[Any]) -> None:
        self._compiled = compiled
        self._middleware = middleware

    def invoke(self, input: Dict[str, Any], config: Optional[Dict] = None) -> Dict[str, Any]:
        # before_agent / after_agent 钩子(第 7 章)
        for mw in self._middleware:
            input = mw.before_agent(input) or input
        result = self._compiled.invoke(input, config)
        for mw in reversed(self._middleware):
            result = mw.after_agent(result) or result
        return result

    def stream(self, input: Dict[str, Any], config: Optional[Dict] = None,
               stream_mode: str = "updates"):
        return self._compiled.stream(input, config, stream_mode=stream_mode)

    def get_state(self, config):
        return self._compiled.get_state(config)


def create_agent(
    model: Any,
    tools: Sequence[Tool] = (),
    system_prompt: Optional[str] = None,
    middleware: Sequence[Any] = (),
    checkpointer: Any = None,
) -> Agent:
    """构建一个标准 Agent 图并编译。

    图结构(就是第 3 章那张,只是自动搭好):
        START → model →(有 tool_calls?)→ tools → model → ... → END
    """
    tools_by_name = {t.name: t for t in tools}
    model = model.bind_tools(list(tools))
    mws = list(middleware)

    schema = {"messages": add_reducer}

    # ---- model 节点:调模型(前后挂 before_model / wrap_model_call / after_model)
    def call_model(state: Dict[str, Any]) -> Dict[str, Any]:
        messages = list(state["messages"])
        if system_prompt and not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=system_prompt)] + messages

        # before_model 钩子:可改写发给模型的消息(裁剪/摘要/注入记忆)
        for mw in mws:
            messages = mw.before_model(messages) or messages

        # wrap_model_call 钩子:层层包裹真正的模型调用
        def base_call(msgs: List[BaseMessage]) -> AIMessage:
            return model.invoke(msgs)

        handler = base_call
        for mw in reversed(mws):
            handler = _wrap_model(mw, handler)
        ai = handler(messages)

        # after_model 钩子:可校验/改写模型输出
        for mw in mws:
            ai = mw.after_model(ai) or ai
        return {"messages": [ai]}

    # ---- tools 节点:执行所有被请求的工具(前后挂 wrap_tool_call)
    def call_tools(state: Dict[str, Any]) -> Dict[str, Any]:
        last = state["messages"][-1]
        out: List[BaseMessage] = []
        for call in last.tool_calls:
            def base_tool(c=call):
                return tools_by_name[c.name].invoke(c.args)

            handler = base_tool
            for mw in reversed(mws):
                handler = _wrap_tool(mw, handler, call)
            result = handler()
            out.append(ToolMessage(content=str(result),
                                   tool_call_id=call.id, name=call.name))
        return {"messages": out}

    # ---- 条件边:最后一条 AI 消息有 tool_calls 就继续,否则结束
    def should_continue(state: Dict[str, Any]):
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    g = StateGraph(schema)
    g.add_node("model", call_model)
    g.add_node("tools", call_tools)
    g.add_edge(START, "model")
    g.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "model")

    compiled = g.compile(checkpointer=checkpointer)
    return Agent(compiled, mws)


# ---- 把单个中间件的 wrap_* 钩子接进 handler 链(handler-passing 模式)-------
def _wrap_model(mw, handler):
    def wrapped(messages):
        return mw.wrap_model_call(messages, handler)
    return wrapped


def _wrap_tool(mw, handler, call):
    def wrapped():
        return mw.wrap_tool_call(call, handler)
    return wrapped
