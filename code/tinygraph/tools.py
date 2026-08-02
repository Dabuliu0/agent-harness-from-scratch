"""工具抽象与 @tool 装饰器。

对应 LangChain 的 langchain_core.tools。核心思想:把一个普通 Python 函数
包装成 "带 schema 的、可被模型调用的工具",schema 从函数签名 + 类型注解
+ docstring 自动推断出来。
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, get_type_hints


# Python 类型 → JSON Schema 类型名 的简单映射
_PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


@dataclass
class Tool:
    """一个可被模型调用的工具。"""

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema 描述参数
    func: Callable[..., Any]

    def invoke(self, args: Dict[str, Any]) -> Any:
        """真正执行工具。模型只负责 "请求" 调用,执行永远由框架完成。"""
        return self.func(**args)

    def to_schema(self) -> Dict[str, Any]:
        """导出给模型看的工具描述(类 OpenAI/Anthropic function schema)。"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


def _build_parameters(func: Callable[..., Any]) -> Dict[str, Any]:
    """从函数签名 + 类型注解构造 JSON Schema。"""
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    properties: Dict[str, Any] = {}
    required: list = []
    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        py_type = hints.get(pname, str)
        json_type = _PY_TO_JSON.get(py_type, "string")
        properties[pname] = {"type": json_type}
        # 没有默认值的参数视为必填
        if param.default is inspect.Parameter.empty:
            required.append(pname)

    return {"type": "object", "properties": properties, "required": required}


def tool(func: Callable[..., Any]) -> Tool:
    """把普通函数装饰成 Tool。

    用法::

        @tool
        def get_weather(city: str) -> str:
            '''查询指定城市的天气。'''
            return f"{city} 晴, 25°C"

    工具的 name 取函数名,description 取 docstring,参数 schema 从签名推断。
    """
    name = func.__name__
    description = (func.__doc__ or "").strip()
    parameters = _build_parameters(func)
    return Tool(name=name, description=description, parameters=parameters, func=func)
