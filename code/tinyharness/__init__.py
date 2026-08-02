"""tinyharness —— 手写的 Harness 控制面（L2）。

内核（tinycore）解决"循环怎么转"；本包解决"怎么把循环托管成一个可信的产品"：

- session      会话与持久化：事件日志是唯一事实源，可恢复、可分叉
- permissions  权限：默认询问、规则匹配、审批回调——硬约束，不是提示词
- hooks        钩子：把 harness 的接缝开放给用户配置
- subagents    子代理：上下文隔离的多智能体
- skills       技能：SKILL.md 渐进披露
- mcp          MCP 客户端：把外部工具生态接进来
- harness      门面：把以上全部组装成一个可运行的 Agent 应用

与内核的关系只有两条通道：事件流（出）与 gate（入）。
"""

__version__ = "0.2.0"

from .session import Session, SessionStore  # noqa: F401
from .permissions import ALLOW, ASK, DENY, PermissionPolicy, PermissionRule  # noqa: F401
from .hooks import HookManager, HookResult  # noqa: F401
from .subagents import SubagentDef, make_task_tool  # noqa: F401
from .skills import Skill, load_skills, make_skill_tool, skills_section  # noqa: F401
from .mcp import MCPServerStdio  # noqa: F401
from .harness import Harness, HarnessConfig  # noqa: F401
