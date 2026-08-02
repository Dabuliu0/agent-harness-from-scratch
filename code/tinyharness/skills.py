"""技能 —— 渐进披露的领域知识包（第 7 章）。

技能（Agent Skills / SKILL.md，2025 年由 Anthropic 提出后成为跨厂商惯例）
解决的问题是：领域知识（"怎么发版""怎么写周报""公司代码规范"）既不该
硬编码进 system prompt（token 全预付、改起来要发版），也不该做成工具
（它是知识不是能力）。

答案是**渐进披露（progressive disclosure）**，一共三级：
1. 清单级：system prompt 里只放每个技能的 name + description（几十 token）；
2. 正文级：模型判断需要时，调 ``read_skill(name)`` 把 SKILL.md 全文拉进上下文；
3. 资源级：SKILL.md 里可以引用同目录的脚本/模板，模型再用文件工具去读。

目录约定（与真实系统一致）::

    skills/
    └── release-notes/
        ├── SKILL.md          # 带 name/description 头部 + 正文指引
        └── template.md       # 可选的配套资源
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List

from tinycore.tools import Tool


@dataclass
class Skill:
    name: str
    description: str
    path: str  # SKILL.md 的路径

    def body(self) -> str:
        with open(self.path, "r", encoding="utf-8") as f:
            return f.read()


_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)


def _parse_skill(path: str) -> Skill:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    name = os.path.basename(os.path.dirname(path))
    desc = ""
    m = _FRONTMATTER.match(text)
    if m:
        for line in m.group(1).splitlines():
            k, _, v = line.partition(":")
            if k.strip() == "name" and v.strip():
                name = v.strip()
            elif k.strip() == "description":
                desc = v.strip()
    if not desc:  # 没写 frontmatter 就拿正文第一段凑合
        body = _FRONTMATTER.sub("", text).strip()
        desc = body.splitlines()[0][:120] if body else "(无描述)"
    return Skill(name=name, description=desc, path=path)


def load_skills(skills_dir: str) -> List[Skill]:
    """扫描 ``skills_dir/*/SKILL.md``，返回技能清单。目录不存在返回空。"""
    out: List[Skill] = []
    if not os.path.isdir(skills_dir):
        return out
    for entry in sorted(os.listdir(skills_dir)):
        p = os.path.join(skills_dir, entry, "SKILL.md")
        if os.path.isfile(p):
            out.append(_parse_skill(p))
    return out


def skills_section(skills: List[Skill]) -> str:
    """生成注入 system prompt 的"第一级披露"：只有名字和一句话描述。"""
    if not skills:
        return ""
    lines = "\n".join(f"- {s.name}: {s.description}" for s in skills)
    return (
        "# 可用技能\n"
        "以下技能是可按需加载的操作手册。当任务与某技能的描述匹配时，"
        "先调用 read_skill 读取其全文，再按其中的指引行事：\n" + lines
    )


def make_skill_tool(skills: List[Skill]) -> Tool:
    """第二级披露的入口：``read_skill(name)`` 把技能正文拉进上下文。"""
    by_name = {s.name: s for s in skills}

    def read_skill(name: str) -> str:
        s = by_name.get(name)
        if s is None:
            return f"[错误] 没有名为 {name!r} 的技能，可用: {', '.join(by_name) or '(无)'}"
        return f"<skill name={s.name!r} dir={os.path.dirname(s.path)!r}>\n{s.body()}\n</skill>"

    return Tool(
        name="read_skill",
        description="加载一个技能的完整操作手册。技能清单见 system prompt 的\"可用技能\"一节。",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        func=read_skill,
    )
