"""mini_code —— 全书终局示例：用 tinycore + tinyharness 组装的迷你编码 Agent CLI。

这是 L3（宿主层）的完整演示：约 130 行，只做三件事——
把配置翻译成 HarnessConfig、把事件流渲染成终端界面、把审批做成确认框。
所有智能与控制都在下面两层里，宿主薄得几乎透明——这正是分层的意义。

用法（需先配好 code/.env 或环境变量 TINYAGENT_MODEL 与对应 key）::

    python mini_code.py --workspace ./ws          # 新会话
    python mini_code.py --workspace ./ws --resume 20260726-...   # 恢复会话
    python mini_code.py --mode accept_edits       # 自动接受文件编辑

REPL 内命令： /sessions 列出会话  /quit 退出；其余输入都交给 Agent。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _config import load_env  # noqa: E402

load_env()

from tinycore import events as ev  # noqa: E402
from tinyharness import Harness, HarnessConfig, PermissionRule  # noqa: E402
from tinyharness.permissions import ALLOW  # noqa: E402

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
CYAN, YELLOW, RED = "\033[36m", "\033[33m", "\033[31m"


def approver(call, reason) -> bool:
    """审批回调：权限系统说"ask"时，人来定。这就是 canUseTool 的 CLI 形态。"""
    from tinyharness.permissions import invocation_summary

    print(f"\n{YELLOW}⚠ 请求执行: {invocation_summary(call)}{RESET}")
    print(f"{DIM}  ({reason}){RESET}")
    return input("  允许吗? [y/N] ").strip().lower() in ("y", "yes")


def render(e) -> None:
    """事件流 → 终端。UI 只是事件流的一个消费者。"""
    if e.type == ev.TEXT_DELTA:
        print(e.data["text"], end="", flush=True)
    elif e.type == ev.ASSISTANT_MESSAGE and e.data["message"].tool_calls:
        for c in e.data["message"].tool_calls:
            arg = next(iter(c.args.values()), "")
            print(f"{CYAN}● {c.name}({str(arg)[:80]}){RESET}")
    elif e.type == ev.TOOL_RESULT:
        m = e.data["message"]
        head = (m.content or "").strip().splitlines()[0][:100] if m.content else ""
        color = RED if m.is_error else DIM
        print(f"{color}  ⎿ {head}{RESET}")
    elif e.type == ev.COMPACTION:
        print(f"{DIM}[上下文已压缩 {e.data['before_tokens']}→{e.data['after_tokens']} tokens]{RESET}")
    elif e.type == ev.RUN_END:
        u = e.data["usage"]
        print(f"\n{DIM}—— {e.data['stop_reason']} · ↑{u['input_tokens']} ↓{u['output_tokens']} tokens{RESET}")
    elif e.type == ev.ERROR:
        print(f"{RED}[错误] {e.data['error']}{RESET}")


def main() -> None:
    ap = argparse.ArgumentParser(description="mini Claude Code（教学版）")
    ap.add_argument("--workspace", default="./workspace")
    ap.add_argument("--mode", default="ask",
                    choices=["ask", "accept_edits", "readonly", "yolo"])
    ap.add_argument("--resume", default=None, help="要恢复的会话 id")
    ap.add_argument("--skills", default=None, help="技能目录（含 */SKILL.md）")
    args = ap.parse_args()

    h = Harness(HarnessConfig(
        workspace=args.workspace,
        permission_mode=args.mode,
        approver=approver,
        skills_dir=args.skills,
        rules=[  # 常用安全命令直接放行，减少审批噪音（第 6 章：规则优先于模式）
            PermissionRule("bash", "git status*", ALLOW),
            PermissionRule("bash", "git diff*", ALLOW),
            PermissionRule("bash", "ls*", ALLOW),
        ],
    ))
    session_id = args.resume
    print(f"{BOLD}mini_code{RESET} · 工作区 {os.path.abspath(args.workspace)} · 模式 {args.mode}")
    if session_id:
        print(f"{DIM}恢复会话 {session_id}{RESET}")

    while True:
        try:
            prompt = input(f"\n{BOLD}> {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt == "/quit":
            break
        if prompt == "/sessions":
            print("\n".join(h.sessions.list()) or "(暂无会话)")
            continue
        try:
            for e in h.run(prompt, session_id=session_id):
                render(e)
            session_id = h.last_session.id  # 本轮之后固定在同一会话上继续
        except KeyboardInterrupt:
            print(f"\n{YELLOW}[已请求中断]{RESET}")
            h.agent.interrupt()
    h.close()


if __name__ == "__main__":
    main()
