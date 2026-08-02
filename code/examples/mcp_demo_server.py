"""教学用最小 MCP server（第 9 章）—— 站到协议的另一侧。

不依赖任何 SDK，几十行就能实现一个合法的 MCP server：
从 stdin 逐行读 JSON-RPC 请求，往 stdout 逐行写应答。
提供两个演示工具：echo（回声）与 add（加法）。

单独测试它（体验裸协议）::

    python mcp_demo_server.py
    {"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
    {"jsonrpc":"2.0","method":"notifications/initialized"}
    {"jsonrpc":"2.0","id":2,"method":"tools/list"}
    {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"add","arguments":{"a":1,"b":2}}}
"""
import json
import sys

TOOLS = [
    {
        "name": "echo",
        "description": "原样返回输入的文本（演示用）",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "add",
        "description": "计算两个整数之和（演示用）",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
]


def _call(name, args):
    if name == "echo":
        return str(args.get("text", ""))
    if name == "add":
        return str(int(args["a"]) + int(args["b"]))
    raise ValueError(f"unknown tool: {name}")


def _reply(msg_id, result=None, error=None):
    resp = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        resp["error"] = {"code": -32000, "message": error}
    else:
        resp["result"] = result
    sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        method, msg_id = msg.get("method"), msg.get("id")
        if method == "initialize":
            _reply(msg_id, {
                "protocolVersion": msg.get("params", {}).get("protocolVersion", "2025-06-18"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "demo-server", "version": "0.1"},
            })
        elif method == "notifications/initialized":
            pass  # 通知无需应答
        elif method == "tools/list":
            _reply(msg_id, {"tools": TOOLS})
        elif method == "tools/call":
            p = msg.get("params", {})
            try:
                text = _call(p.get("name"), p.get("arguments", {}))
                _reply(msg_id, {"content": [{"type": "text", "text": text}], "isError": False})
            except Exception as e:
                _reply(msg_id, {"content": [{"type": "text", "text": str(e)}], "isError": True})
        elif msg_id is not None:
            _reply(msg_id, error=f"method not found: {method}")


if __name__ == "__main__":
    main()
