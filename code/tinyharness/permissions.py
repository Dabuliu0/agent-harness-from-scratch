"""权限系统 —— 把"模型只请求、框架才执行"变成可配置的策略（第 6 章）。

三层决策，从粗到细：
1. **模式（mode）**：整个会话的信任姿态。readonly / ask / accept_edits / yolo
   （对应真实 harness 的 plan mode / default / acceptEdits / bypassPermissions）。
2. **规则（rules）**：对特定"工具 + 参数模式"的显式放行或拒绝，
   如简单命令 ``bash(git *)`` 放行、``bash(rm *)`` 拒绝。规则优先于模式。
3. **审批（approver）**：模式和规则都吃不准的（ask），交给人决定。
   审批回调就是 Claude Agent SDK 的 ``canUseTool``、CLI 里的那个确认框。

产出物是一个 **gate 函数**，插进内核的接缝（``Agent(gate=...)``）。
内核不认识"权限"这个词——它只知道 gate 说不行就不执行。
这就是"机制在内核、策略在 harness"的完整闭环。

要点：权限是**硬约束**。系统提示里写"请不要删文件"是许愿；
gate 里拒绝 ``rm`` 才是控制。
"""
from __future__ import annotations

import fnmatch
import json
import shlex
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from tinycore.messages import ToolCall
from tinycore.tools import Gate

ALLOW, DENY, ASK = "allow", "deny", "ask"

# 工具的默认风险等级：决定各模式下的默认动作。未知工具一律按最危险处理。
DEFAULT_RISK: Dict[str, str] = {
    "read_file": "read",
    "list_dir": "read",
    "glob_files": "read",
    "grep": "read",
    "read_skill": "read",
    "todo_write": "write",
    "write_file": "write",
    "edit_file": "write",
    "bash": "exec",
    "task": "spawn",  # 子代理：自身无害，危险在它内部的工具（继承收窄，见第 8 章）
}

# 各模式下，风险等级 → 默认动作
_MODE_TABLE: Dict[str, Dict[str, str]] = {
    "readonly":     {"read": ALLOW, "write": DENY,  "exec": DENY,  "spawn": ALLOW},
    "ask":          {"read": ALLOW, "write": ASK,   "exec": ASK,   "spawn": ALLOW},
    "accept_edits": {"read": ALLOW, "write": ALLOW, "exec": ASK,   "spawn": ALLOW},
    "yolo":         {"read": ALLOW, "write": ALLOW, "exec": ALLOW, "spawn": ALLOW},
}


def invocation_summary(call: ToolCall) -> str:
    """把一次调用压成一行可匹配、可展示的字符串，如 ``bash(git status)``。"""
    if call.name == "bash":
        head = str(call.args.get("command", ""))
    elif len(call.args) == 1:
        head = str(next(iter(call.args.values())))
    else:
        head = json.dumps(call.args, ensure_ascii=False)
    return f"{call.name}({head})"


_SHELL_CHAIN_OPERATORS = {"&&", "||", ";", "|"}
_SHELL_CONTROL_CHARS = set(";&|<>")
_NESTED_SHELLS = {"sh", "bash", "zsh", "fish", "dash", "ksh", "cmd", "powershell", "pwsh"}


def _shell_segments(command: str) -> Optional[List[str]]:
    """保守地拆出 shell 命令段；无法证明安全时返回 None。"""
    command = command.strip()
    if not command or "$(" in command or "`" in command:
        return None

    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return None

    for index, token in enumerate(tokens[:-1]):
        executable = token.rsplit("/", 1)[-1].lower()
        if executable in _NESTED_SHELLS and tokens[index + 1].lower() in {
            "-c",
            "--command",
            "-command",
            "/c",
        }:
            return None

    segments: List[str] = []
    current: List[str] = []
    for token in tokens:
        if token in _SHELL_CHAIN_OPERATORS:
            if not current:
                return None
            segments.append(shlex.join(current))
            current = []
        elif any(char in token for char in _SHELL_CONTROL_CHARS):
            return None
        else:
            current.append(token)
    if not current:
        return None
    segments.append(shlex.join(current))
    return segments


def _invocation_inner(call: ToolCall) -> str:
    summary = invocation_summary(call)
    return summary[len(call.name) + 1 : -1]


@dataclass
class PermissionRule:
    """一条显式规则：工具名 + 参数通配模式 → 动作。"""

    tool: str
    pattern: str = "*"          # 与调用参数做 fnmatch；bash 仅匹配可解析的命令段
    action: str = ALLOW

    def _matches_inner(self, inner: str) -> bool:
        return fnmatch.fnmatch(inner, self.pattern)

    def matches(self, call: ToolCall) -> bool:
        if call.name != self.tool:
            return False
        inner = _invocation_inner(call)
        if call.name == "bash":
            segments = _shell_segments(inner)
            return segments is not None and len(segments) == 1 and self._matches_inner(segments[0])
        return self._matches_inner(inner)


# 审批回调：返回 True 放行 / False 拒绝。由 CLI（input 确认框）、测试（脚本）、
# 服务端（挂起 run 等 HTTP 回调）各自实现——策略与交互形态解耦。
Approver = Callable[[ToolCall, str], bool]


@dataclass
class PermissionPolicy:
    mode: str = "ask"
    rules: List[PermissionRule] = field(default_factory=list)
    approver: Optional[Approver] = None
    risk: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_RISK))
    audit: List[Tuple[str, str]] = field(default_factory=list)  # (调用摘要, 最终决定)

    def _check_inner(self, call: ToolCall, inner: str) -> Tuple[str, str]:
        for rule in self.rules:  # 规则优先，先命中先生效
            if rule.tool == call.name and rule._matches_inner(inner):
                return rule.action, f"命中规则 {rule.tool}({rule.pattern})→{rule.action}"
        level = self.risk.get(call.name, "exec")
        action = _MODE_TABLE.get(self.mode, _MODE_TABLE["ask"]).get(level, ASK)
        return action, f"{self.mode} 模式下 {level} 级工具默认 {action}"

    def check(self, call: ToolCall) -> Tuple[str, str]:
        """纯决策（不含交互）：返回 (allow|deny|ask, 理由)。"""
        if call.name != "bash":
            return self._check_inner(call, _invocation_inner(call))

        command = str(call.args.get("command", ""))
        segments = _shell_segments(command)
        if segments is None:
            level = self.risk.get(call.name, "exec")
            action = _MODE_TABLE.get(self.mode, _MODE_TABLE["ask"]).get(level, ASK)
            return action, f"{self.mode} 模式下 bash 命令无法安全解析，默认 {action}"

        decisions = [(segment, *self._check_inner(call, segment)) for segment in segments]
        if len(decisions) == 1:
            _, action, reason = decisions[0]
            return action, reason

        if any(action == DENY for _, action, _ in decisions):
            segment, _, reason = next(item for item in decisions if item[1] == DENY)
            return DENY, f"复合命令段 {segment!r} 被拒绝：{reason}"
        if any(action == ASK for _, action, _ in decisions):
            segment, _, reason = next(item for item in decisions if item[1] == ASK)
            return ASK, f"复合命令段 {segment!r} 需要审批：{reason}"
        return ALLOW, "复合命令的每一段均已明确放行"

    def as_gate(self) -> Gate:
        """编译成内核 gate。ask 在这里落地成"问 approver"——没有审批通道就拒绝。"""

        def gate(call: ToolCall):
            action, reason = self.check(call)
            if action == ASK:
                if self.approver is None:
                    action, reason = DENY, "需要人工批准，但当前没有审批通道"
                elif self.approver(call, reason):
                    action, reason = ALLOW, "人工批准"
                else:
                    action, reason = DENY, "人工拒绝"
            self.audit.append((invocation_summary(call), action))
            if action == DENY:
                return f"权限不足：{invocation_summary(call)}（{reason}）"
            return None  # 放行

        return gate
