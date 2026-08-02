"""第 3 章示例:上下文工程——装配、测量、压缩,全程离线可见。

python k03_context.py
"""
import os
import shutil
import tempfile

import _config  # noqa: F401

from tinycore import Agent, ContextManager, FakeModel, ToolCall, tool
from tinycore.context import estimate_tokens


@tool
def read_doc(name: str) -> str:
    """读取一篇长文档。"""
    return f"《{name}》正文:" + ("这是很长的文档内容。" * 200)   # ~2000 字


def spy(messages):
    """剧本函数:回显模型本轮实际看到了什么。"""
    first = messages[0]
    return (f"我本轮看到 {len(messages)} 条消息;"
            f"第一条是 {first.role},开头「{(first.content or '')[:16]}…」")


def main():
    ws = tempfile.mkdtemp(prefix="k03-ws-")
    with open(os.path.join(ws, "AGENTS.md"), "w", encoding="utf-8") as f:
        f.write("回答一律以「收到:」开头。本项目禁止修改 legacy/ 目录。")

    # 剧本按【模型被调用的顺序】排:三次读文档 → (压缩的摘要调用) → 最终回显。
    # 注意第 4 条台词是被 ContextManager.compact() 内部那次 invoke 消耗的!
    model = FakeModel([
        [ToolCall("read_doc", {"name": "文档A"})],
        [ToolCall("read_doc", {"name": "文档B"})],
        [ToolCall("read_doc", {"name": "文档C"})],
        "- 用户要读 A/B/C 三篇文档并总结;A、B 已读完,内容高度重复",   # ← 摘要
        spy,
    ])
    cm = ContextManager(max_context_tokens=1500, keep_recent=4, memory_cwd=ws)
    agent = Agent(model=model, tools=[read_doc], context=cm)

    print("== 运行(观察 token 增长与压缩事件) ==")
    for e in agent.run("把文档 A、B、C 都读一遍然后总结"):
        if e.type == "compaction":
            print(f"  ★ COMPACTION: {e.data['before_tokens']} → {e.data['after_tokens']} tokens")
        elif e.type == "turn_end":
            print(f"  轮 {e.data['turn']} 结束")
        elif e.type == "run_end":
            print(f"\n最终回答: {e.data['final_text']}")

    print(f"\n结束时历史 ≈ {estimate_tokens(agent.last_messages)} tokens,结构:")
    for m in agent.last_messages:
        body = m.content or "(工具调用)"
        print(f"  {m.role}: {body[:44]}")

    print("\n观察点:")
    print(" 1. 历史开头出现了[前情提要](原始的 A/B 长文已被摘要替换);")
    print(" 2. spy 显示模型看到的第一条是装配出来的 system(含 AGENTS.md)——改文件,下一轮即生效;")
    print(" 3. 压缩发生在『调模型之前』,并作为事件对外可见。")
    shutil.rmtree(ws, ignore_errors=True)


if __name__ == "__main__":
    main()
