"""工具系统 —— 把函数变成模型可请求的能力，并安全地执行它（第 1、2 章）。

三个组成：
- ``Tool``：name / description / parameters(JSON Schema) / func 四件套。
  前三样是发给模型的"说明书"，func 是藏在框架侧的真实实现。
- ``@tool``：从函数签名 + 类型注解 + docstring 自动生成说明书。
- ``run_tool_calls``：一批工具调用的执行器——**这里是"模型只请求、框架才执行"
  这条安全边界的物理位置**。gate（权限闸门）、错误即消息、并行执行、
  结果截断，全都发生在这几十行里。
"""
from __future__ import annotations

import inspect
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, get_type_hints

from .messages import ToolCall, ToolMessage

_PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _build_parameters(func: Callable) -> Dict[str, Any]:
    """从函数签名 + 类型注解构造 JSON Schema。"""
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        json_type = _PY_TO_JSON.get(hints.get(pname, str), "string")
        properties[pname] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(pname)
    return {"type": "object", "properties": properties, "required": required}


@dataclass
class Tool:
    """一个可被模型请求调用的工具。"""

    name: str
    description: str
    parameters: Dict[str, Any]
    func: Callable
    # 工具结果超过此长度会被截断——工具结果是上下文的最大污染源（第 3 章）
    max_result_chars: int = 20_000

    def invoke(self, args: Dict[str, Any]) -> str:
        result = str(self.func(**args))
        if len(result) > self.max_result_chars:
            omitted = len(result) - self.max_result_chars
            result = result[: self.max_result_chars] + f"\n…[已截断 {omitted} 字符]"
        return result


def tool(func: Callable = None, *, name: str = None, max_result_chars: int = 20_000):
    """装饰器：``@tool`` 或 ``@tool(name=..., max_result_chars=...)``。

    name 取函数名，description 取 docstring——**docstring 是写给模型的接口文档**，
    质量直接决定模型"何时调用、怎么传参"的正确率（第 2 章展开）。
    """

    def wrap(f: Callable) -> Tool:
        return Tool(
            name=name or f.__name__,
            description=(f.__doc__ or "").strip(),
            parameters=_build_parameters(f),
            func=f,
            max_result_chars=max_result_chars,
        )

    return wrap(func) if func is not None else wrap


# ---------------------------------------------------------------------------
# 执行器：安全边界的物理位置
# ---------------------------------------------------------------------------

# gate 的契约（内核唯一的"策略注入接缝"，第 6、7 章的 harness 都从这里挂进来）：
#   gate(call) -> None            放行，原样执行
#   gate(call) -> str             拒绝，字符串是给模型看的拒绝理由
#   gate(call) -> ToolCall        放行，但用改写后的调用（参数已被修正/净化）
Gate = Callable[[ToolCall], Union[None, str, ToolCall]]


def _execute_one(
    call: ToolCall,
    tools_by_name: Dict[str, Tool],
    gate: Optional[Gate],
) -> ToolMessage:
    """执行单个工具调用。任何失败都变成 is_error 的 ToolMessage 喂回模型，
    而不是抛异常炸掉循环——模型看到错误后可以自己纠正（换参数/换工具/求助）。"""
    if gate is not None:
        try:
            verdict = gate(call)
        except Exception as e:  # 闸门自身出错按拒绝处理：安全侧优先
            verdict = f"权限检查失败: {e}"
        if isinstance(verdict, str):
            return ToolMessage(
                content=f"[已拒绝] {verdict}",
                tool_call_id=call.id,
                name=call.name,
                is_error=True,
            )
        if isinstance(verdict, ToolCall):
            call = verdict

    t = tools_by_name.get(call.name)
    if t is None:
        return ToolMessage(
            content=f"[错误] 不存在名为 {call.name!r} 的工具",
            tool_call_id=call.id,
            name=call.name,
            is_error=True,
        )
    try:
        return ToolMessage(content=t.invoke(call.args), tool_call_id=call.id, name=call.name)
    except Exception:
        tb = traceback.format_exc(limit=2)
        return ToolMessage(
            content=f"[错误] 工具执行失败:\n{tb}",
            tool_call_id=call.id,
            name=call.name,
            is_error=True,
        )


def run_tool_calls(
    calls: Sequence[ToolCall],
    tools_by_name: Dict[str, Tool],
    *,
    gate: Optional[Gate] = None,
    parallel: bool = True,
    max_workers: int = 8,
) -> List[ToolMessage]:
    """执行一批工具调用，**结果顺序恒等于请求顺序**。

    模型一轮可能发多个互不依赖的调用（查三个城市天气、读五个文件），并行执行
    省的是真金白银的等待时间。但并行完成的先后是随机的——如果按完成顺序追加
    结果，同样的输入会得到不同的历史。所以这里按 ``calls`` 的原顺序收集 futures，
    保证**确定性**（图运行时深入篇会看到，这正是 Reducer 解决的同一个问题）。
    """
    if len(calls) <= 1 or not parallel:
        return [_execute_one(c, tools_by_name, gate) for c in calls]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(calls))) as pool:
        futures = [pool.submit(_execute_one, c, tools_by_name, gate) for c in calls]
        return [f.result() for f in futures]

