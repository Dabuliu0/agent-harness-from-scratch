"""内置工具族 —— 编码 Agent 的标准装备（第 2 章）。

2026 年的主流 harness 殊途同归地收敛到同一套核心工具：
读 / 写 / 精确编辑 / 列目录 / 通配查找 / 内容检索 / 执行命令 / 任务清单。
给模型这八件套加一个工作区，它就"有了一台电脑"。

安全设计（第 2、6 章反复回到这里）：
- 所有**文件工具**在构造时就被锁进 ``root`` 工作区（路径 jail）：
  相对路径基于 root 解析，realpath 后必须仍在 root 之内，否则 拒绝。
  这是"机制层"的安全——不靠模型自觉，单次解析物理上做不到越界；生产环境仍需处理 TOCTOU。
- ``bash`` 是唯一无法靠构造锁住的工具（命令天生能触达整个系统），
  所以它必须靠**策略层**把守：权限规则 + 审批（第 6 章），
  以及生产中的 OS 级沙箱（Seatbelt / bubblewrap / 容器）。
- 路径 jail 的 realpath 检查与后续 open 之间仍存在 TOCTOU 窗口；生产环境还要配合
  打开后校验、平台级 O_NOFOLLOW 或 OS 沙箱，不能把这段教学代码当作完整边界。
"""
from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from typing import List

from .tools import Tool, tool


def make_coding_tools(
    root: str,
    *,
    bash_timeout: int = 60,
    max_output_chars: int = 8_000,
    enable_bash: bool = True,
) -> List[Tool]:
    """构造一套锁定在 ``root`` 工作区的编码工具。"""
    root = os.path.realpath(root)
    os.makedirs(root, exist_ok=True)

    def _resolve(path: str) -> str:
        """路径 jail：任何解析结果越出 root 一律拒绝（软链接目标也参与检查）。"""
        p = os.path.realpath(os.path.join(root, path))
        if p != root and not p.startswith(root + os.sep):
            raise PermissionError(f"路径越出工作区: {path!r}")
        return p

    @tool
    def read_file(path: str, offset: int = 0, limit: int = 200) -> str:
        """读取工作区内的文本文件，返回带行号的内容。
        path: 相对工作区的路径。offset/limit: 从第 offset 行起最多读 limit 行，
        大文件请分段读取而不是一次读完。"""
        with open(_resolve(path), "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        window = lines[offset : offset + limit]
        body = "".join(f"{i + offset + 1:>5}\t{l}" for i, l in enumerate(window))
        tail = f"\n…(共 {len(lines)} 行，仅显示 {offset + 1}-{offset + len(window)})" if len(lines) > offset + limit else ""
        return (body or "(空文件)") + tail

    @tool
    def write_file(path: str, content: str) -> str:
        """在工作区内创建或整体覆盖一个文件。仅用于新建文件或全量重写；
        修改既有文件请优先用 edit_file。"""
        p = _resolve(path)
        os.makedirs(os.path.dirname(p) or root, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return f"已写入 {path}（{len(content)} 字符）"

    @tool
    def edit_file(path: str, old: str, new: str) -> str:
        """对既有文件做精确替换：old 必须与文件中某段内容**完全一致且唯一**。
        找不到或出现多处都会报错——请提供更长的上下文片段来唯一定位。"""
        p = _resolve(path)
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
        n = text.count(old)
        if n == 0:
            raise ValueError(f"未找到要替换的内容（old 与文件不一致）: {old[:80]!r}")
        if n > 1:
            raise ValueError(f"old 在文件中出现 {n} 次，无法唯一定位，请扩大片段")
        with open(p, "w", encoding="utf-8") as f:
            f.write(text.replace(old, new, 1))
        return f"已编辑 {path}"

    @tool
    def list_dir(path: str = ".") -> str:
        """列出工作区内某目录的内容（目录以 / 结尾）。"""
        p = _resolve(path)
        entries = sorted(os.listdir(p))
        out = [e + "/" if os.path.isdir(os.path.join(p, e)) else e for e in entries]
        return "\n".join(out) or "(空目录)"

    @tool
    def glob_files(pattern: str) -> str:
        """按通配符递归查找文件名，如 "**/*.py"、"src/*.md"。返回相对路径列表。"""
        hits: List[str] = []  #命中结果

        # dirpath：当前正在遍历的文件夹的绝对路径
        # dirnames：这个文件夹下的子文件夹名称列表
        # filenames：这个文件夹下的文件名称列表
        for dirpath, dirnames, filenames in os.walk(root):  #os.walk(root) 是 Python 里遍历目录树的标准方法
            dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", "node_modules")]
            for fn in filenames:
                rel = os.path.relpath(os.path.join(dirpath, fn), root)  #以root为基准构建相对路径，给人看的
                if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(fn, pattern):  #通配符搜索
                    hits.append(rel)
        return "\n".join(sorted(hits)[:200]) or "(无匹配)"

    @tool
    def grep(pattern: str, path: str = ".", max_matches: int = 50) -> str:
        """在工作区内按正则搜索文件内容，返回 "文件:行号:内容"。
        pattern: Python 正则。path: 限定搜索的子目录，默认全工作区。"""
        rx = re.compile(pattern)  #把用户输入的正则表达式字符串（比如 "def .*:"）编译成一个“正则对象” rx
        base = _resolve(path)
        hits: List[str] = []
        limit_reached = False  #停止标志，
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", "node_modules")]
            for fn in filenames:
                fp = os.path.join(dirpath, fn)
                try: 
                    with open(fp, "r", encoding="utf-8") as f:
                        for i, line in enumerate(f, 1):
                            if rx.search(line):  #用编译好的正则对象在每一行里搜索匹配
                                rel = os.path.relpath(fp, root)
                                hits.append(f"{rel}:{i}:{line.rstrip()}")  #把“文件路径:行号:去掉末尾换行符的内容” 拼成一条字符串，存入 hits 列表。
                                if len(hits) >= max_matches:
                                    limit_reached = True
                                    break
                except UnicodeDecodeError:
                    continue
                if limit_reached:
                    break
            if limit_reached:
                break
        return "\n".join(hits) or "(无匹配)"

    @tool
    def todo_write(content: str) -> str:
        """维护你的任务清单（覆盖式写入）。做多步任务时，先把计划写进来，
        每完成一步就更新状态——这份清单会帮你在长任务中保持方向。"""
        with open(os.path.join(root, ".agent_todo.md"), "w", encoding="utf-8") as f:
            f.write(content)
        return "任务清单已更新"

    @tool
    def bash(command: str) -> str:
        """在工作区目录下执行 shell 命令，返回 exit code 与输出（stdout+stderr 合并）。
        用于运行测试、git、构建等。命令有超时限制；产生大量输出时会被截断。"""
        # 注意：bash 无法被路径 jail 约束（这正是它必须过权限闸门的原因，见第 6 章）
        r = subprocess.run(       #“执行外部命令”函数
            command,              # 要执行的命令字符串，比如 "ls -la"
            shell=True,           # 通过系统的 shell（比如 bash）来执行，这样能支持管道、通配符等
            cwd=root,             # 命令的工作目录设置为工作区根目录（所以你在里面执行 `ls` 看到的就是工作区内容）
            capture_output=True,  # 捕获命令的输出（标准输出和标准错误）
            text=True,            # 以文本字符串形式返回（而不是字节），方便后续处理
            timeout=bash_timeout, # 给命令设定一个最大运行时间（默认 60 秒），防止死循环或卡死
            # r是一个 CompletedProcess 对象，里面存着：
          
            # r.stdout：命令的标准输出（正常打印的信息）
            # r.stderr：命令的标准错误（报错信息）
            # r.returncode：命令的退出码（0 通常表示成功）
        )
        out = (r.stdout or "") + (r.stderr or "")  
        if len(out) > max_output_chars:
            out = out[:max_output_chars] + f"\n…[输出已截断，共 {len(out)} 字符]"
        return f"(exit {r.returncode})\n{out.strip()}"

    tools = [read_file, write_file, edit_file, list_dir, glob_files, grep, todo_write]

    #允许执行 Bash 命令？
    if enable_bash:
        tools.append(bash)
    return tools
