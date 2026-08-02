"""MCP 客户端（最小实现）—— 把外部工具生态接进内核（第 9 章）。

MCP（Model Context Protocol）解决 N×M 问题：N 个 Agent 应用 × M 个工具源，
本来要写 N×M 个集成；有了统一协议，各写一次就互通。2025-2026 年它已是
全行业的工具接入标准。

协议本体朴素得惊人：**JSON-RPC 2.0，每条消息一行 JSON**（stdio 传输时）。
生命周期三步：
1. ``initialize``               握手：交换协议版本与能力
2. ``notifications/initialized`` 客户端确认（通知，无需应答）
3. 正常使用：``tools/list`` 发现工具、``tools/call`` 调用工具

我们只实现 stdio 传输 + tools 能力——这已覆盖日常 90% 的用法；
resources / prompts / HTTP 传输的结构完全同构（第 9 章末尾给指引）。

安全提醒（第 9 章展开）：MCP 工具的 name/description 来自**外部进程**，
对模型而言是不可信输入（提示注入面）。接入前应过白名单，执行时照常过 gate。
"""
from __future__ import annotations

import json
import subprocess
from typing import Any, Dict, List, Optional

from tinycore.tools import Tool

PROTOCOL_VERSION = "2025-06-18"


class MCPError(RuntimeError):
    pass


class MCPServerStdio:
    """通过 stdio 与一个 MCP server 子进程通信的最小客户端。"""

    def __init__(self, command: List[str], name: str = "server") -> None:
        self.command = command
        self.name = name
        self.proc: Optional[subprocess.Popen] = None
        self._id = 0

    # ---- 传输层：一行进、一行出 ----

    def _send(self, msg: Dict[str, Any]) -> None:
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def _request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        self._id += 1
        self._send({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}})
        assert self.proc and self.proc.stdout
        while True:  # 跳过 server 主动发来的通知，直到等到本次 id 的应答
            line = self.proc.stdout.readline()
            if not line:
                raise MCPError(f"MCP server {self.name!r} 意外退出")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue  # server 往 stdout 打了日志之类的脏东西，忽略
            if msg.get("id") != self._id:
                continue
            if "error" in msg:
                raise MCPError(f"{method} 失败: {msg['error'].get('message')}")
            return msg.get("result")

    def _notify(self, method: str) -> None:
        self._send({"jsonrpc": "2.0", "method": method})

    # ---- 生命周期 ----

    def start(self) -> "MCPServerStdio":
        self.proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "tinyharness", "version": "0.2"},
            },
        )
        self._notify("notifications/initialized")
        return self

    def close(self) -> None:
        if self.proc:
            self.proc.terminate()
            self.proc = None

    # ---- tools 能力 ----

    def list_tools(self) -> List[Dict[str, Any]]:
        return self._request("tools/list").get("tools", [])

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        result = self._request("tools/call", {"name": name, "arguments": arguments})
        texts = [
            c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"
        ]
        out = "\n".join(t for t in texts if t)
        if result.get("isError"):
            raise MCPError(out or "工具执行失败")
        return out

    # ---- 适配：远端工具 → 内核 Tool ----

    def as_tools(self) -> List[Tool]:
        """把 server 的每个工具包装成内核 Tool，命名 ``mcp__<server>__<tool>``
        （与主流 harness 的命名惯例一致，权限规则可按前缀匹配整个 server）。"""
        out: List[Tool] = []
        for spec in self.list_tools():
            tool_name = spec["name"]

            def call(_name=tool_name, **kwargs: Any) -> str:
                return self.call_tool(_name, kwargs)

            out.append(
                Tool(
                    name=f"mcp__{self.name}__{tool_name}",
                    description=spec.get("description", ""),
                    parameters=spec.get("inputSchema", {"type": "object", "properties": {}}),
                    func=call,
                )
            )
        return out
