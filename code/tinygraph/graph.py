"""StateGraph —— 把计算建模成 "节点 + 边 + 状态" 的图构建器。

对应 LangGraph 的 langgraph.graph.StateGraph。本文件只负责 "声明" 这张图
(有哪些节点、哪些边、状态长什么样);真正 "执行" 它的引擎在 pregel.py。

状态 schema 用一个 dict 声明:{channel名: reducer}。例如:
    schema = {"messages": add_reducer, "count": last_value}

"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .channels import Reducer, last_value

# 两个特殊的虚拟节点名
START = "__start__"
END = "__end__"

# 节点函数签名: (state: dict) -> dict(部分状态更新)
NodeFn = Callable[[Dict[str, Any]], Dict[str, Any]]
# 条件路由函数签名: (state: dict) -> 下一个节点名 / END / 它们的列表
RouterFn = Callable[[Dict[str, Any]], Union[str, List[str]]]

class StateGraph:
    def __init__(self, schema: Dict[str, Reducer]) -> None:
        # schema: 每个状态 key 对应一个 reducer(决定该 channel 如何合并写入)
        self.schema: Dict[str, Reducer] = dict(schema)
        self.nodes: Dict[str, NodeFn] = {}
        self.edges: Dict[str, List[str]] = {}        # 固定边: from -> [to, ...]
        self.branches: Dict[str, Tuple[RouterFn, Optional[Dict[str, str]]]] = {}

    def add_node(self, name: str, fn: NodeFn) -> "StateGraph":
        if name in (START, END):
            raise ValueError(f"{name} 是保留名")
        self.nodes[name] = fn
        return self

    def add_edge(self, start: str, end: str) -> "StateGraph":
        """固定边:start 执行完后,无条件激活 end。"""
        self.edges.setdefault(start, []).append(end)
        return self

    def add_conditional_edges(
        self,
        start: str,
        router: RouterFn,
        mapping: Optional[Dict[str, str]] = None,
    ) -> "StateGraph":
        """条件边:start 执行完后,调用 router(state) 决定去哪个节点。

        router 返回值若在 mapping 里,则映射成对应节点名;否则直接当作节点名。
        返回 END 表示该分支结束。
        """
        self.branches[start] = (router, mapping)
        return self

    def set_entry_point(self, name: str) -> "StateGraph":
        return self.add_edge(START, name)

    def compile(self, checkpointer: Any = None) -> "CompiledGraph":
        # 延迟 import,避免循环依赖
        from .pregel import CompiledGraph

        return CompiledGraph(self, checkpointer=checkpointer)
