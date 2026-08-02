"""会话与持久化 —— 事件日志是唯一事实源（第 5 章）。

Harness 路线的持久化答案与图运行时的"状态快照（检查点）"不同：
**追加式事件日志（append-only JSONL）**。每发生一件事就往文件尾追加一行 JSON；
"当前对话历史"不是存出来的，而是**从日志重放（replay）推导**出来的。

这带来三个直接好处：
- 崩溃恢复：追加写天然抗崩溃——最坏丢最后一行，历史永远是合法前缀；
- 完整审计：日志里不止有消息，还有工具调用、审批、压缩……全过程可回放；
- 分叉（fork）：复制文件前缀就是"从这里开出一条平行世界"。

对照真实系统：Claude Code 的 ~/.claude/projects/**/*.jsonl 会话文件、
其 --resume/--fork-session 语义，就是这套机制的生产版。
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Iterator, List, Optional

from tinycore import events as ev
from tinycore.events import Event
from tinycore.messages import Messages, from_jsonable, to_jsonable

# 进入历史的三种事件：重放时只看它们（其余事件是过程记录，不构成历史）
_MESSAGE_EVENTS = {ev.USER_MESSAGE, ev.ASSISTANT_MESSAGE, ev.TOOL_RESULT}


class Session:
    """一个会话 = 一个 JSONL 文件。append 记录事件，messages() 重放出历史。"""

    def __init__(self, path: str, session_id: Optional[str] = None) -> None:
        self.path = path
        self.id = session_id or os.path.splitext(os.path.basename(path))[0]

    # ---- 写：只有追加，没有修改 ----

    def append(self, event: Event) -> None:
        line = json.dumps(
            {"ts": event.ts, "type": event.type, "data": to_jsonable(event.data)},
            ensure_ascii=False,
        )
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def record(self, event_stream: Iterator[Event]) -> Iterator[Event]:
        """把会话变成事件流的一个"透明消费者"：边转发边落盘。

        用法：``for e in session.record(agent.run(prompt)): ...``
        —— UI 逻辑一行不改，持久化就发生了。这就是"层间只隔一条事件流"的红利。
        """
        for e in event_stream:
            self.append(e)
            yield e

    # ---- 读：重放 ----

    def events(self) -> List[Event]:
        out: List[Event] = []
        if not os.path.exists(self.path):
            return out
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue  # 最后一行可能因崩溃而残缺：跳过即可，前缀仍然合法
                out.append(Event(d["type"], from_jsonable(d.get("data", {})), d.get("ts", 0)))
        return out

    def messages(self) -> Messages:
        """从事件日志重放出对话历史（resume 的地基）。

        细节：末尾若停着"带 tool_calls 的 AI 消息 + 不完整的工具结果"
        （崩在工具执行中途），把这半轮裁掉——恢复点必须是合法的消息边界。
        """
        msgs: Messages = [
            e.data["message"] for e in self.events() if e.type in _MESSAGE_EVENTS
        ]
        while msgs:
            last = msgs[-1]
            calls = getattr(last, "tool_calls", None)
            if calls:  # AI 消息要求了工具但结果没齐 → 裁掉这条 AI 消息
                msgs.pop()
                continue
            if getattr(last, "role", "") == "tool":
                # 工具结果凑不齐一整批也要裁（找到它所属的 AI 消息一并检查）
                i = len(msgs) - 1
                while i >= 0 and msgs[i].role == "tool":
                    i -= 1
                need = {c.id for c in getattr(msgs[i], "tool_calls", [])} if i >= 0 else set()
                got = {m.tool_call_id for m in msgs[i + 1 :]}
                if need and need != got:
                    del msgs[i:]
                    continue
            break
        return msgs


class SessionStore:
    """会话仓库：一个目录，每个会话一个 .jsonl 文件。"""

    def __init__(self, root: str = ".tinyharness/sessions") -> None:
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, session_id: str) -> str:
        return os.path.join(self.root, f"{session_id}.jsonl")

    def create(self, session_id: Optional[str] = None) -> Session:
        sid = session_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        return Session(self._path(sid), sid)

    def open(self, session_id: str) -> Session:
        path = self._path(session_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"会话不存在: {session_id}")
        return Session(path, session_id)

    def fork(self, session_id: str, upto_event: Optional[int] = None) -> Session:
        """从既有会话分叉：复制其日志（可截到第 upto_event 行），得到平行新会话。"""
        src = self.open(session_id)
        dst = self.create(f"{session_id}-fork-{uuid.uuid4().hex[:4]}")
        with open(src.path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if upto_event is not None:
            lines = lines[:upto_event]
        with open(dst.path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        return dst

    def list(self) -> List[str]:
        return sorted(
            os.path.splitext(fn)[0]
            for fn in os.listdir(self.root)
            if fn.endswith(".jsonl")
        )
