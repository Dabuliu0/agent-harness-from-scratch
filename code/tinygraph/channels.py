"""Channel(通道)与 Reducer(归并器)—— 图状态的存储与合并机制。

对应 LangGraph 的 channels。核心思想:
  - 状态的每个 key 是一个 Channel,Channel 持有当前值 + 一个单调递增的版本号。
  - 一个超步里,多个节点可能同时写同一个 Channel。如何把这些写入合并进旧值?
    由该 Channel 的 reducer 决定。
  - 默认 reducer 是 "覆盖"(后写覆盖旧值);列表类常用 "追加" reducer。
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional

# Reducer 签名: (旧值, 新写入的值) -> 合并后的新值
Reducer = Callable[[Any, Any], Any]


def last_value(current: Any, update: Any) -> Any:
    """默认 reducer:直接用新值覆盖旧值。"""
    return update


def add_reducer(current: Any, update: Any) -> Any:
    """累加 reducer:把新值追加到列表(用于消息历史等)。"""
    if current is None:
        current = []
    if isinstance(update, list):
        return current + update
    return current + [update]


class Channel:
    """一个带版本号的数据载体。

    版本号是单调递增的整数。每当 Channel 被成功写入(并经 reducer 合并)后,
    版本号 +1。

    版本号有什么用?在真实 LangGraph 里,它是【节点触发】的依据:节点订阅若干
    channel,运行时比较 "节点上次见过的版本 vs channel 当前版本",版本变大就触发
    节点。我们的简化引擎不走这条路 —— 我们直接用【边】决定下一批跑谁(见 pregel.py
    的 _next_active),版本号只用于【检查点】:存盘时一并记下,恢复时一并还原,
    让恢复后的状态和存盘那一刻逐位相同。第 3 章 3.5 节会把这两种触发模型讲透。
    """

    def __init__(self, reducer: Reducer = last_value, value: Any = None) -> None:
        self.reducer = reducer
        self.value = value
        self.version = 0

    def update(self, writes: List[Any]) -> None:
        """把本超步收集到的一组写入,依次经 reducer 合并进当前值,然后版本 +1。

        注意:多个写入按 "确定性顺序"(运行时保证的固定顺序)依次合并,
        这样并行节点写同一个 channel 的结果才是可复现的。
        """
        if not writes:
            return
        v = self.value
        for w in writes:
            v = self.reducer(v, w)
        self.value = v
        self.version += 1

    def snapshot(self) -> Any:
        return self.value

    def copy(self) -> "Channel":
        c = Channel(self.reducer, _deepish_copy(self.value))
        c.version = self.version
        return c


def _deepish_copy(value: Any) -> Any:
    """对常见的可变容器做浅层拷贝,避免节点拿到的副本被意外共享修改。"""
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value
