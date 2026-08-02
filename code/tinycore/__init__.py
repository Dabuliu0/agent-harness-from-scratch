"""tinycore —— 手写的 Agent 内核（L0 模型接口 + L1 循环运行时）。

对应主流 harness（Claude Code / Codex CLI / Agent SDK）里那个"模型驱动循环"内核：
消息、模型接口、工具、上下文装配、事件流、主循环。

分层约定（贯穿全书）：
- 本包只提供**机制**：循环怎么转、工具怎么执行、事件怎么吐；
- 所有**策略**（权限、审批、钩子、会话、子代理）都在 tinyharness 里，
  通过 gate / 事件流这两个接缝注入进来。
"""

__version__ = "0.2.0"

from .messages import (  # noqa: F401
    AIMessage,
    BaseMessage,
    HumanMessage,
    Messages,
    SystemMessage,
    ToolCall,
    ToolMessage,
    Usage,
    from_jsonable,
    to_jsonable,
)
from .models import (  # noqa: F401
    AnthropicChatModel,
    BaseChatModel,
    FakeModel,
    OpenAIChatModel,
    get_model,
    init_chat_model,
)
from .tools import Tool, tool  # noqa: F401
from .events import Event  # noqa: F401
from .context import ContextManager, estimate_tokens, load_memory_files  # noqa: F401
from .loop import Agent, final_text  # noqa: F401
from .toolkit import make_coding_tools  # noqa: F401
