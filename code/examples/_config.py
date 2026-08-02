"""示例公用:从环境变量挑选并构造一个真实模型。

所有示例都通过 ``get_model()`` 拿模型,这样你只需配一次环境变量,就能用同一份
示例代码切换 provider。

配置方式(二选一):
  1. 直接设环境变量::

       export TINYAGENT_MODEL="anthropic:claude-sonnet-5"
       export ANTHROPIC_API_KEY="sk-..."

  2. 在 code/ 目录放一个 .env 文件(需 ``pip install python-dotenv``)::

       TINYAGENT_MODEL=deepseek:deepseek-chat
       DEEPSEEK_API_KEY=sk-...

``TINYAGENT_MODEL`` 形如 ``"provider:model"``,支持:
  anthropic / openai / deepseek / qwen / kimi,或任意 OpenAI 兼容服务。
缺省值是 ``anthropic:claude-sonnet-5``。
"""
import os
import sys
from pathlib import Path

# 让示例能直接 import 到上层的 tinygraph / tinyagent 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Windows 默认控制台可能使用 GBK；统一为 UTF-8，避免示例的中文和符号输出失败。
if os.name == "nt":
    for _stream in (sys.stdout, sys.stderr):
        if _stream is not None and hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")

# 可选:若装了 python-dotenv,自动加载 code/.env
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

DEFAULT_MODEL = "anthropic:claude-sonnet-5"


def load_env() -> None:
    """确保 .env 已加载、包路径已注入（import 本模块时已自动完成，此函数供显式调用）。"""


def get_model(**kwargs):
    """（图运行时深入篇示例用）按 ``TINYAGENT_MODEL`` 构造 tinygraph 的模型。"""
    from tinygraph.models import init_chat_model

    spec = os.environ.get("TINYAGENT_MODEL", DEFAULT_MODEL)
    return init_chat_model(spec, **kwargs)


def get_kernel_model(**kwargs):
    """（主线示例用）按 ``TINYAGENT_MODEL`` 构造 tinycore 的模型（支持流式）。"""
    from tinycore.models import init_chat_model

    spec = os.environ.get("TINYAGENT_MODEL", DEFAULT_MODEL)
    return init_chat_model(spec, **kwargs)
