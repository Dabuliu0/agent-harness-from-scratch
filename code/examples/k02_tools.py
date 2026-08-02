"""第 2 章示例:编码工具八件套 + 执行边界。

离线:  python k02_tools.py            (FakeModel 剧本:写→跑→改→再跑)
真模型: python k02_tools.py --real     让模型自己完成同一个任务
"""
import shutil
import sys
import tempfile

import _config  # noqa: F401

from tinycore import Agent, FakeModel, ToolCall, final_text, make_coding_tools
from tinycore.tools import run_tool_calls


def main():
    ws = tempfile.mkdtemp(prefix="k02-ws-")
    tools = make_coding_tools(ws)
    print(f"工作区: {ws}")

    if "--real" in sys.argv:
        model = _config.get_kernel_model()
    else:
        model = FakeModel([
            [ToolCall("write_file", {"path": "hello.py", "content": "print('helo')\n"})],
            [ToolCall("bash", {"command": "python3 hello.py"})],
            [ToolCall("edit_file", {"path": "hello.py", "old": "helo", "new": "hello"})],
            [ToolCall("bash", {"command": "python3 hello.py"})],
            "已创建 hello.py 并修正拼写,运行输出 hello。",
        ])

    agent = Agent(model=model, tools=tools)
    for e in agent.run("创建一个打印 hello 的脚本并确认能跑"):
        if e.type == "tool_start":
            c = e.data["call"]
            print(f"● {c.name}({c.args})")
        elif e.type == "tool_result":
            print(f"  ⎿ {(e.data['message'].content or '').splitlines()[0][:70]}")
    print(f"\n最终回答: {final_text(agent.last_messages)}")

    # ---- 边界演示(不经过模型,直接打执行器) ----
    tools_by_name = {t.name: t for t in tools}
    def gist(msg):
        """从错误 ToolMessage 里抽出最有信息量的一行(通常是异常那行)。"""
        lines = [l.strip() for l in msg.content.splitlines() if l.strip()]
        return lines[-1][:80]

    print("\n[路径 jail] 尝试读 ../../etc/passwd:")
    r = run_tool_calls([ToolCall("read_file", {"path": "../../etc/passwd"})], tools_by_name)
    print(f"  {gist(r[0])} (is_error={r[0].is_error})")

    print("[错误即消息] 尝试 edit 不存在的内容:")
    r = run_tool_calls(
        [ToolCall("edit_file", {"path": "hello.py", "old": "不存在的串", "new": "x"})],
        tools_by_name,
    )
    print(f"  {gist(r[0])} (is_error={r[0].is_error})")

    shutil.rmtree(ws, ignore_errors=True)


if __name__ == "__main__":
    main()
