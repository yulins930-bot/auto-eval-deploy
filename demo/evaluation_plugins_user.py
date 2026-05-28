# -*- coding: utf-8 -*-
"""
在此文件注册你自己的行级评测插件（平台启动时自动加载）。

修改后重启 demo 即可生效。
"""
from __future__ import annotations

from typing import Any, Callable, Optional


def register(registrar: Callable[..., Any]) -> None:
    """平台入口：将自定义插件注册进全局表。"""

    def _at_least_three_points(ctx) -> Optional[bool]:
        cols = ctx.profile.get("point_columns") or []
        min_ok = int(ctx.params.get("min_ok", 3))
        matched = 0
        comparable = 0
        for c in cols:
            d = ctx.by_col.get(c)
            if not d or d.get("correct") is None:
                continue
            comparable += 1
            if d.get("correct"):
                matched += 1
        if comparable < min_ok:
            return None
        return matched >= min_ok

    registrar(
        "at_least_three_points",
        _at_least_three_points,
        label="至少 3 个采分点对",
        description="采分点列中至少 min_ok 列判对即算行正确（params.min_ok，默认 3）",
        params_schema={"min_ok": {"type": "integer", "default": 3}},
    )
