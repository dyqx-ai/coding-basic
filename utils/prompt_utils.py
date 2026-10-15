# -*- coding: utf-8 -*-
"""
提示词模板安全渲染工具。

背景事故：模板字符串里写了 JSON 示例 {"is_enough": true/false, ...}，
str.format() 把 {"is_enough"} 当占位符解析 → KeyError: '"is_enough"'，
导致 demo 澄清节点 / Complete 模式架构合并节点直接 500。
"""
import re

# 占位符形如 {KEY}；(?<!\{) / (?!\}) 排除已转义的 {{KEY}} 内层，避免误替换
_PLACEHOLDER_RE = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})")


def safe_format(template: str, **kwargs) -> str:
    """
    安全模板渲染（替代 str.format），语义对齐 str.format 且永不抛 KeyError：

    1. 只替换模板中 {KEY} 占位符（KEY 必须在 kwargs 中）；
    2. 未知 KEY（如 JSON 示例里的 {"is_enough"} / {true}）原样保留，不抛异常；
    3. 已按 str.format 风格写的 {{...}} 转义块先跳过占位符替换，再统一还原为单花括号，
       与 str.format 渲染结果完全一致（包括模板内嵌代码里的 f-string {{e}} → {e}）。
    """
    def _sub(m):
        key = m.group(1)
        return str(kwargs[key]) if key in kwargs else m.group(0)

    out = _PLACEHOLDER_RE.sub(_sub, template)
    return out.replace("{{", "{").replace("}}", "}")
