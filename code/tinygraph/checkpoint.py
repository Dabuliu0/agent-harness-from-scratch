"""Checkpointer —— 检查点持久化:存档、断点恢复、时间旅行。

对应 LangGraph 的 BaseCheckpointSaver / InMemorySaver / SqliteSaver。

核心概念:
  - Checkpoint:某一时刻图状态的完整快照(各 channel 的值 + 版本 + 下一步要跑
    的节点 + 父检查点 id)。它是【可序列化】的 —— 这是断点恢复的前提。
  - thread_id:一条独立的执行线(比如一个用户会话)。同一 thread 下的检查点
    串成一条链,最新的在尾部。
  - 时间旅行:每个超步存一个检查点,于是能 list 出整条历史、并从任意一个恢复。
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Checkpoint:
    """图状态在某一超步结束时的完整快照。"""

    thread_id: str
    checkpoint_id: str
    # 各 channel 的 {值, 版本}
    channel_values: Dict[str, Any]
    channel_versions: Dict[str, int]
    # 下一超步要激活的节点(空 = 执行已结束)
    next_nodes: List[str]
    # 父检查点 id(用于串成链 / 时间旅行),根检查点为 None
    parent_id: Optional[str] = None
    step: int = -1
    ts: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(
            {
                "thread_id": self.thread_id,
                "checkpoint_id": self.checkpoint_id,
                "channel_values": self.channel_values,
                "channel_versions": self.channel_versions,
                "next_nodes": self.next_nodes,
                "parent_id": self.parent_id,
                "step": self.step,
                "ts": self.ts,
            },
            default=_json_default,
            ensure_ascii=False,
        )

    @staticmethod
    def from_json(s: str) -> "Checkpoint":
        d = json.loads(s)
        return Checkpoint(**d)


def new_checkpoint_id() -> str:
    # 用时间前缀 + uuid,保证同一 thread 内大致按时间有序
    return f"{int(time.time()*1000):013d}-{uuid.uuid4().hex[:8]}"


def _json_default(o: Any) -> Any:
    """让 dataclass 消息等对象可被 JSON 序列化(教学用的简化序列化)。"""
    from dataclasses import asdict, is_dataclass

    if is_dataclass(o):
        d = asdict(o)
        d["__type__"] = type(o).__name__   # 记下类型,反序列化时还原
        return d
    raise TypeError(f"无法序列化: {type(o)}")


class BaseCheckpointSaver:
    """检查点存储接口。对应 LangGraph 的 BaseCheckpointSaver。"""

    def put(self, checkpoint: Checkpoint) -> None:
        raise NotImplementedError

    def get(self, thread_id: str, checkpoint_id: Optional[str] = None) -> Optional[Checkpoint]:
        """取某 thread 的最新检查点;给定 checkpoint_id 则取指定的那个。"""
        raise NotImplementedError

    def list(self, thread_id: str) -> List[Checkpoint]:
        """列出某 thread 的全部检查点,最新的在前(用于时间旅行)。"""
        raise NotImplementedError


class InMemorySaver(BaseCheckpointSaver):
    """内存版:进程内有效,重启即失。对应 LangGraph 的 InMemorySaver。"""

    def __init__(self) -> None:
        # thread_id -> [Checkpoint, ...] 按写入顺序(即时间顺序)
        self._store: Dict[str, List[Checkpoint]] = {}

    def put(self, checkpoint: Checkpoint) -> None:
        self._store.setdefault(checkpoint.thread_id, []).append(checkpoint)

    def get(self, thread_id: str, checkpoint_id: Optional[str] = None) -> Optional[Checkpoint]:
        chain = self._store.get(thread_id, [])
        if not chain:
            return None
        if checkpoint_id is None:
            return chain[-1]
        for cp in reversed(chain):
            if cp.checkpoint_id == checkpoint_id:
                return cp
        return None

    def list(self, thread_id: str) -> List[Checkpoint]:
        return list(reversed(self._store.get(thread_id, [])))


class SqliteSaver(BaseCheckpointSaver):
    """SQLite 版:跨进程持久化。对应 LangGraph 的 SqliteSaver。

    检查点序列化成 JSON 存进一张表。重启进程后仍能 get/list/恢复。
    """

    def __init__(self, path: str = ":memory:") -> None:
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS checkpoints (
                   thread_id TEXT,
                   checkpoint_id TEXT,
                   step INTEGER,
                   data TEXT,
                   PRIMARY KEY (thread_id, checkpoint_id)
               )"""
        )
        self.conn.commit()

    def put(self, checkpoint: Checkpoint) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO checkpoints VALUES (?,?,?,?)",
            (checkpoint.thread_id, checkpoint.checkpoint_id,
             checkpoint.step, checkpoint.to_json()),
        )
        self.conn.commit()

    def get(self, thread_id: str, checkpoint_id: Optional[str] = None) -> Optional[Checkpoint]:
        if checkpoint_id is None:
            row = self.conn.execute(
                "SELECT data FROM checkpoints WHERE thread_id=? ORDER BY rowid DESC LIMIT 1",
                (thread_id,),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT data FROM checkpoints WHERE thread_id=? AND checkpoint_id=?",
                (thread_id, checkpoint_id),
            ).fetchone()
        return Checkpoint.from_json(row[0]) if row else None

    def list(self, thread_id: str) -> List[Checkpoint]:
        rows = self.conn.execute(
            "SELECT data FROM checkpoints WHERE thread_id=? ORDER BY rowid DESC",
            (thread_id,),
        ).fetchall()
        return [Checkpoint.from_json(r[0]) for r in rows]
