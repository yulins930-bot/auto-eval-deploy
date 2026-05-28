# -*- coding: utf-8 -*-
"""
内置 + 可扩展的行级评测插件。

在 evaluation_plugins_user.py 中可 register 自定义函数，无需改平台主代码。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

RowEvaluator = Callable[["RowEvalContext"], Optional[bool]]

_PLUGIN_META: Dict[str, Dict[str, Any]] = {}
_PLUGIN_FN: Dict[str, RowEvaluator] = {}


@dataclass
class RowEvalContext:
    """单行评测上下文（插件入参）。"""

    by_col: Dict[str, Dict[str, Any]]
    profile: Dict[str, Any]
    preds_map: Optional[Dict[str, Any]] = None
    params: Dict[str, Any] = field(default_factory=dict)


def register_row_evaluator(
    plugin_id: str,
    fn: RowEvaluator,
    *,
    label: str,
    description: str = "",
    params_schema: Optional[Dict[str, Any]] = None,
) -> None:
    pid = str(plugin_id).strip()
    if not pid:
        raise ValueError("plugin_id 不能为空")
    _PLUGIN_FN[pid] = fn
    _PLUGIN_META[pid] = {
        "id": pid,
        "label": label,
        "description": description,
        "params_schema": params_schema or {},
    }


def list_plugins() -> List[Dict[str, Any]]:
    return [dict(_PLUGIN_META[k]) for k in sorted(_PLUGIN_FN.keys())]


def get_plugin(plugin_id: str) -> Optional[RowEvaluator]:
    return _PLUGIN_FN.get(str(plugin_id).strip())


def _cols_detail(ctx: RowEvalContext, columns: List[str]) -> List[Dict[str, Any]]:
    out = []
    for c in columns:
        d = ctx.by_col.get(c)
        if d is not None:
            out.append(d)
    return out


def _numeric(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(str(v).strip().replace(",", ""))
    except ValueError:
        return None


def _eval_all_columns(ctx: RowEvalContext) -> Optional[bool]:
    cols = ctx.params.get("columns") or ctx.profile.get("ground_truth_columns") or []
    cols = [c for c in cols if c and not str(c).startswith("_")]
    comp = _cols_detail(ctx, cols)
    if not comp:
        return None
    return all(d.get("correct") for d in comp)


def _eval_points_all(ctx: RowEvalContext) -> Optional[bool]:
    cols = ctx.params.get("columns") or ctx.profile.get("point_columns") or []
    return _eval_all_columns(RowEvalContext(ctx.by_col, ctx.profile, ctx.preds_map, {"columns": cols}))


def _eval_points_ratio(ctx: RowEvalContext) -> Optional[bool]:
    """至少 min_ratio 比例的采分点列判对（默认 1.0 = 全部）。"""
    cols = ctx.params.get("columns") or ctx.profile.get("point_columns") or []
    min_ratio = float(ctx.params.get("min_ratio", 1.0))
    min_ratio = max(0.0, min(1.0, min_ratio))
    comp = _cols_detail(ctx, cols)
    if not comp:
        return None
    ok = sum(1 for d in comp if d.get("correct")) / len(comp)
    return ok >= min_ratio


def _eval_points_and_total(ctx: RowEvalContext) -> Optional[bool]:
    pts_ok = _eval_points_all(RowEvalContext(ctx.by_col, ctx.profile, ctx.preds_map, ctx.params))
    total_cols = ctx.params.get("total_columns") or ctx.profile.get("total_columns") or []
    if not total_cols:
        return pts_ok
    tc = total_cols[0]
    d = ctx.by_col.get(tc)
    if not d or d.get("correct") is None:
        return pts_ok if pts_ok is not None else None
    if pts_ok is None:
        return None
    return pts_ok and bool(d.get("correct"))


def _eval_primary_only(ctx: RowEvalContext) -> Optional[bool]:
    primary = ctx.params.get("column") or ctx.profile.get("primary_ground_truth")
    if not primary or primary not in ctx.by_col:
        return None
    return (ctx.by_col.get(primary) or {}).get("correct")


def _eval_min_columns_match(ctx: RowEvalContext) -> Optional[bool]:
    """在 ground_truths 中至少 min_count 列可比对且全部判对。"""
    cols = ctx.params.get("columns") or ctx.profile.get("ground_truth_columns") or []
    ignore = set(ctx.params.get("ignore_columns") or [])
    cols = [c for c in cols if c not in ignore and not str(c).startswith("_")]
    min_count = int(ctx.params.get("min_count", len(cols) or 1))
    comp = _cols_detail(ctx, cols)
    if len(comp) < min_count:
        return None
    use = comp[:min_count] if min_count < len(comp) else comp
    return all(d.get("correct") for d in use)


def _eval_total_tolerance(ctx: RowEvalContext) -> Optional[bool]:
    """采分点全对，且预测总分与标注总分相差不超过 tolerance。"""
    pts_ok = _eval_points_all(ctx)
    total_cols = ctx.profile.get("total_columns") or []
    if not total_cols:
        return pts_ok
    tc = total_cols[0]
    d = ctx.by_col.get(tc) or {}
    gt = _numeric(d.get("ground_truth"))
    pred = _numeric(d.get("predicted"))
    if pts_ok is None:
        return None
    if gt is None or pred is None:
        return pts_ok if ctx.params.get("points_only_if_no_total") else None
    tol = float(ctx.params.get("tolerance", 0))
    total_ok = abs(pred - gt) <= tol
    return pts_ok and total_ok


def _eval_weighted_score(ctx: RowEvalContext) -> Optional[bool]:
    """
    按列权重加权：预测加权分与标注加权分相差 <= tolerance 即判对。
    params.weights: {"采分点1": 2, "采分点2": 1, ...}
    """
    weights = ctx.params.get("weights") or {}
    if not weights:
        cols = ctx.profile.get("point_columns") or ctx.profile.get("ground_truth_columns") or []
        weights = {c: 1 for c in cols}
    pred_sum = 0.0
    gt_sum = 0.0
    w_sum = 0.0
    n = 0
    for col, w in weights.items():
        try:
            wf = float(w)
        except (TypeError, ValueError):
            continue
        d = ctx.by_col.get(col)
        if not d:
            continue
        p = _numeric(d.get("predicted"))
        g = _numeric(d.get("ground_truth"))
        if p is None or g is None:
            continue
        pred_sum += p * wf
        gt_sum += g * wf
        w_sum += wf
        n += 1
    if n == 0 or w_sum == 0:
        return None
    tol = float(ctx.params.get("tolerance", 0))
    return abs(pred_sum - gt_sum) <= tol


def _register_builtins() -> None:
    register_row_evaluator(
        "all_ground_truths",
        _eval_all_columns,
        label="全部标注列一致",
        description="已选 ground_truths 每一列都可比对且判对",
    )
    register_row_evaluator(
        "points_all",
        _eval_points_all,
        label="全部采分点一致",
        description="所有采分点列（正则识别）均判对",
    )
    register_row_evaluator(
        "points_ratio",
        _eval_points_ratio,
        label="采分点比例达标",
        description="至少 min_ratio（0~1）的采分点列判对",
        params_schema={"min_ratio": {"type": "number", "default": 1.0, "min": 0, "max": 1}},
    )
    register_row_evaluator(
        "points_and_total",
        _eval_points_and_total,
        label="采分点全对 + 总分一致",
        description="全部采分点判对且总分列一致",
    )
    register_row_evaluator(
        "primary_column",
        _eval_primary_only,
        label="仅主标注列",
        description="行正确只看 primary_ground_truth",
    )
    register_row_evaluator(
        "min_columns_all",
        _eval_min_columns_match,
        label="至少 N 列全对",
        description="可比对列数 >= min_count 且这些列全部判对",
        params_schema={
            "min_count": {"type": "integer", "default": 1},
            "ignore_columns": {"type": "array", "default": []},
        },
    )
    register_row_evaluator(
        "total_tolerance",
        _eval_total_tolerance,
        label="采分点全对 + 总分容差",
        description="采分点全对，且 |预测总分-标注总分| <= tolerance",
        params_schema={"tolerance": {"type": "number", "default": 0}},
    )
    register_row_evaluator(
        "weighted_score",
        _eval_weighted_score,
        label="加权总分容差",
        description="按 weights 加权求和，预测与标注差 <= tolerance",
        params_schema={
            "weights": {"type": "object", "default": {}},
            "tolerance": {"type": "number", "default": 0},
        },
    )


def load_user_plugins() -> None:
    """加载 demo/evaluation_plugins_user.py（若存在）。"""
    try:
        import evaluation_plugins_user as user_mod

        reg = user_mod.register
        if callable(reg):
            reg(register_row_evaluator)
    except ImportError:
        pass


_register_builtins()
load_user_plugins()
