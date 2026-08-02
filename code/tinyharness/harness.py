"""Harness 门面 —— 把内核与控制面组装成一个可运行的 Agent 应用（第 9-11 章）。

到这里为止，每个部件都是独立可用的；``Harness`` 只做**组装**，不再引入新机制：

    模型 + 工具族(工作区) + 技能 + MCP + 子代理     ← 能力面
    权限策略 → gate ← 钩子                          ← 控制面（编译成一个 gate）
    上下文管理（记忆文件 + 预算 + 压缩）              ← 上下文面
    会话存储（事件日志，record 透明落盘）             ← 持久化面

对照真实世界：这个类相当于 Claude Agent SDK 的 ``query()`` 入口 /
OpenAI Agents SDK 的 ``Runner``——产品（CLI/IDE/服务）都从这一个门面进出，
绝不绕过它直连模型（否则权限与审计就成了摆设）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence, Union

from tinycore import events as ev
from tinycore.context import ContextManager
from tinycore.events import Event
from tinycore.loop import Agent
from tinycore.models import BaseChatModel, get_model, init_chat_model
from tinycore.toolkit import make_coding_tools
from tinycore.tools import Tool

from .hooks import HookManager, STOP, USER_PROMPT_SUBMIT
from .mcp import MCPServerStdio
from .permissions import Approver, PermissionPolicy, PermissionRule
from .session import Session, SessionStore
from .skills import load_skills, make_skill_tool, skills_section
from .subagents import SubagentDef, make_task_tool

DEFAULT_SYSTEM_PROMPT = """\
你是一个严谨的编码助手，在一个受限的工作区内帮用户完成任务。
- 动手前先看清现状（读文件/列目录），不臆测文件内容。
- 多步任务先用 todo_write 列计划，完成一步更新一步。
- 修改文件优先用 edit_file 做精确替换；只有新建/全量重写才用 write_file。
- 工具被拒绝时不要原样重试，向用户说明并寻求替代方案。
- 回答简洁，用中文。"""


@dataclass
class HarnessConfig:
    model: Union[BaseChatModel, str, None] = None   # None → 读 TINYAGENT_MODEL
    workspace: str = "./workspace"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    permission_mode: str = "ask"
    rules: Sequence[PermissionRule] = ()
    approver: Optional[Approver] = None
    skills_dir: Optional[str] = None
    mcp_servers: Sequence[MCPServerStdio] = ()      # 未 start 的实例
    subagents: Sequence[SubagentDef] = ()
    extra_tools: Sequence[Tool] = ()
    session_root: str = ".tinyharness/sessions"
    max_turns: int = 40
    max_context_tokens: int = 60_000
    enable_bash: bool = True


class Harness:
    def __init__(self, cfg: HarnessConfig) -> None:
        self.cfg = cfg
        # ① 模型
        if isinstance(cfg.model, str):
            self.model: BaseChatModel = init_chat_model(cfg.model)
        else:
            self.model = cfg.model or get_model()
        # ② 控制面：权限 + 钩子 → 一个 gate
        self.policy = PermissionPolicy(
            mode=cfg.permission_mode, rules=list(cfg.rules), approver=cfg.approver
        )
        self.hooks = HookManager()
        self.gate = self.hooks.as_gate(self.policy.as_gate())
        # ③ 能力面：工作区工具族 + 技能 + MCP + 子代理
        self.tools: List[Tool] = make_coding_tools(
            cfg.workspace, enable_bash=cfg.enable_bash
        ) + list(cfg.extra_tools)
        self.skills = load_skills(cfg.skills_dir) if cfg.skills_dir else []
        if self.skills:
            self.tools.append(make_skill_tool(self.skills))
        self._mcp: List[MCPServerStdio] = []
        for server in cfg.mcp_servers:
            server.start()
            self._mcp.append(server)
            self.tools += server.as_tools()
        if cfg.subagents:
            self.tools.append(
                make_task_tool(list(cfg.subagents), self.model, gate=self.gate)
            )
        # ④ 上下文面：预算 + 项目记忆 + 技能清单（第一级披露）
        self.context = ContextManager(
            max_context_tokens=cfg.max_context_tokens,
            memory_cwd=cfg.workspace,
            extra_context=skills_section(self.skills),
        )
        # ⑤ 持久化面
        self.sessions = SessionStore(cfg.session_root)

    # ---- 主入口 ----

    def run(self, prompt: str, session_id: Optional[str] = None) -> Iterator[Event]:
        """执行一次 run：恢复(或新建)会话 → 钩子 → 内核循环 → 透明落盘。"""
        session = (
            self.sessions.open(session_id) if session_id else self.sessions.create()
        )
        self.last_session: Session = session
        prior = session.messages()  # 事件日志重放出历史 → 天然的多轮记忆

        r = self.hooks.fire(USER_PROMPT_SUBMIT, {"prompt": prompt})
        if r.block:
            yield Event(ev.ERROR, {"error": f"输入被钩子拒绝: {r.reason}"})
            return
        if r.add_context:
            prompt = prompt + "\n\n[hook 注入]\n" + r.add_context

        self.agent = Agent(
            model=self.model,
            tools=self.tools,
            system_prompt=self.cfg.system_prompt,
            max_turns=self.cfg.max_turns,
            context=self.context,
            gate=self.gate,
        )
        for e in session.record(self.agent.run(prompt, prior_messages=prior)):
            yield e
            if e.type == ev.RUN_END:
                self.hooks.fire(STOP, {"result": e.data})

    def close(self) -> None:
        for s in self._mcp:
            s.close()
