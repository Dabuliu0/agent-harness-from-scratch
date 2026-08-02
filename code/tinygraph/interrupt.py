"""中断原语 —— 让图能在某个节点处 "真正暂停",把状态存成检查点后退出进程,
之后再从检查点恢复。对应 LangGraph 的 interrupt() 与 Command。

关键点(也是第 5 章的主题):中断不是 "挂起一个还在跑的进程",而是
"抛异常→存档→退出"。恢复时从检查点重建状态、重跑被中断的节点 —— 但这次
interrupt() 会拿到外部注入的 resume 值,从而 "跳过" 暂停继续往下走。
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class GraphInterrupt(Exception):
    """节点内调用 interrupt() 时抛出的异常,被引擎捕获 → 存档并停下。"""

    def __init__(self, value: Any) -> None:
        super().__init__("graph interrupted")
        self.value = value


# 恢复值的传递:调用方通过 Command(resume=...) 注入,引擎放到这个上下文里,
# 节点重跑时 interrupt() 从这里取值。教学用简单方案:线程局部 / 模块级变量。
_RESUME_BOX: Dict[str, Any] = {}


def set_resume(value: Any) -> None:
    _RESUME_BOX["value"] = value
    _RESUME_BOX["consumed"] = False


def interrupt(value: Any) -> Any:
    """在节点中调用:暂停图,把 value 抛给外部(如等待人审批)。

    - 首次执行该节点:没有 resume 值 → 抛 GraphInterrupt,引擎存档并停下。
    - 携带 resume 恢复后重跑该节点:返回外部注入的 resume 值,继续往下执行。
    """
    if _RESUME_BOX.get("value") is not None and not _RESUME_BOX.get("consumed", True):
        _RESUME_BOX["consumed"] = True
        return _RESUME_BOX["value"]
    raise GraphInterrupt(value)


class Command:
    """恢复指令。Command(resume=x) 表示 "带着值 x 从中断处继续"。"""

    def __init__(self, resume: Optional[Any] = None) -> None:
        self.resume = resume
