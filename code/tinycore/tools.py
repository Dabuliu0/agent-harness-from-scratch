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
    """程序启动时执行：将 Python 函数签名 -> AI 调用的 JSON Schema（全自动翻译机）。"""

    sig = inspect.signature(func)          # 1. 提取参数名、默认值、注解
    try:
        hints = get_type_hints(func)       # 2. 解析类型注解（如 city: str -> str）
    except Exception:
        hints = {}                         # 处理循环引用等极端情况，降级不崩溃

    properties: Dict[str, Any] = {}
    required: List[str] = []

    for pname, param in sig.parameters.items():
        # 跳过类方法绑定参数（这些不是工具参数，AI 不应传）
        if pname in ("self", "cls"):
            continue

        # 3. 类型翻译：Python 类型 -> AI 理解的 JSON 类型（str -> "string"）
        # 没写注解就默认 str（对 AI 最友好）
        json_type = _PY_TO_JSON.get(hints.get(pname, str), "string")
        properties[pname] = {"type": json_type}

        # 4. 必填判定：无默认值 -> 必填；有默认值 -> 选填（AI 不传也能跑）
        if param.default is inspect.Parameter.empty:
            required.append(pname)

    # 返回 AI 官方 Function Calling 要求的标准 Schema 结构
    return {"type": "object", "properties": properties, "required": required}


@dataclass
class Tool:
    """一个可被模型请求调用的工具。"""

    name: str                    # 模型用它指定调用谁
    description: str             # 模型靠它判断"何时该用"
    parameters: Dict[str, Any]   # JSON Schema:参数名/类型/必填
    func: Callable               # 真实现,藏在框架侧
  
    # 工具结果超过此长度会被截断——工具结果是上下文的最大污染源（第 3 章）
    max_result_chars: int = 20_000

    # “工具返回的数据，如何安全、标准地喂给大模型？”
    def invoke(self, args: Dict[str, Any]) -> str:

        # 1. 执行函数并统一转字符串，确保模型能消费，大模型（LLM）的API接口只认字符串（str）
        result = str(self.func(**args))

        # 2. 截断超长结果，防止撑爆上下文窗口（核心保护机制）
        if len(result) > self.max_result_chars:
            omitted = len(result) - self.max_result_chars
            result = result[:self.max_result_chars] + f"\n…[已截断 {omitted} 字符]"
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
# # ❌ 没有装饰器的地狱模式
# weather_tool = Tool(
#     name="get_weather",
#     description="查询指定城市的实时天气",
#     parameters={
#         "type": "object",
#         "properties": {
#             "city": {"type": "string", "description": "城市名称，如'北京'"}
#         },
#         "required": ["city"]
#     },
#     func=get_weather
# )

# # ✅ 有装饰器的天堂模式
# @tool
# def get_weather(city: str) -> str:
#     """查询指定城市的实时天气"""
#     return f"{city} 晴，22°C"
# ---------------------------------------------------------------------------



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
            verdict = gate(call)  # 3种情况:不放行（str），放行（None），放行但要没收危险工具（ToolCall）
        except Exception as e:    # 闸门自身出错按拒绝处理：安全侧优先
            verdict = f"权限检查失败: {e}"
          
        if isinstance(verdict, str): #不放行，返回工具调用结果
            return ToolMessage(
                content=f"[已拒绝] {verdict}",
                tool_call_id=call.id,
                name=call.name,
                is_error=True,
            )
        if isinstance(verdict, ToolCall): # 放行但要没收危险工具（ToolCall）
            call = verdict
          # 剩下的情况就是放行（None），什么都不改动

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

    # 情况 1：串行执行
    # 触发条件：只有 1 个调用 或 用户显式关闭并行
    # 此时直接列表推导逐个执行，顺序天然保证，无需引入线程池开销
    if len(calls) <= 1 or not parallel:
        return [_execute_one(c, tools_by_name, gate) for c in calls]

    # 情况 2：并行执行（2 个以上调用，且 parallel=True）
    # with 的作用：进入时创建线程池，退出时自动关闭并等待所有任务完成（资源安全）
    with ThreadPoolExecutor(max_workers=min(max_workers, len(calls))) as pool:

        # ① 提交所有任务到线程池，立即返回 Future 对象（"取餐小票"）
        # pool.submit(fn, *args) 等价于：把 _execute_one(c, ...) 丢给空闲线程去跑
        # 列表推导式保证了 futures 的顺序与 calls 完全一致
        futures = [pool.submit(_execute_one, c, tools_by_name, gate) for c in calls]

        # ② 按 futures 列表的原顺序逐个取结果（阻塞等待）
        # f.result() 会阻塞当前线程，直到该任务完成并返回 ToolMessage
        # 即使第 2 个任务先完成，也必须等第 1 个取完才能取第 2 个
        # 最终返回的列表顺序 == calls 的顺序（确定性保证）
        return [f.result() for f in futures]

