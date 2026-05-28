# -*- coding: utf-8 -*-
"""
可配置的跑批结果评测链路（与具体数据集解耦）。

内置模式：evaluation_mode = auto | single | multi_point_* | all_columns | ...
完全自定义：evaluation_mode = custom + custom_evaluation { plugin, params }
扩展：在 evaluation_plugins_user.py 中 register_row_evaluator(...) 注册 Python 插件。
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

# 行级「是否正确」的汇总规则
ROW_RULE_PRIMARY = "primary"
ROW_RULE_ALL_POINTS = "all_points"
ROW_RULE_POINTS_AND_TOTAL = "points_and_total"
ROW_RULE_ALL_GT_COLUMNS = "all_gt_columns"
ROW_RULE_CUSTOM = "custom"
ROW_RULE_PLUGIN_PREFIX = "plugin:"

# evaluation_mode 可选值（auto 会根据列名推断 row_correct_rule）
HEADLINE_ROW_STRICT = "row_strict"
HEADLINE_MICRO = "micro"

EVALUATION_MODES = {
    "auto": "自动：≥2 个采分点列 → 整行全对+总分；否则 → 主标注列",
    "single": "单列：仅主标注列决定行是否正确",
    "multi_point_strict": "多采分点·整行：全部采分点（+总分）都对才算对",
    "multi_point_micro": "多采分点·最小单位：按每个采分点计对（如 2/3）",
    "multi_point_all": "（同整行）全部采分点列一致才算行正确",
    "multi_point_and_total": "（同整行）采分点全对且总分一致",
    "all_columns": "多列一致：已勾选的全部标注列均一致（不限「采分点」命名）",
    "per_column_only": "仅分列统计：行正确仍看主标注列，各列单独算准确率",
    "custom": "完全自定义：选用内置插件或 evaluation_plugins_user.py 注册的插件",
}

# 第二步下拉框展示顺序（突出整行 / 最小单位，旧 id 保留兼容）
UI_EVALUATION_MODE_ORDER = [
    "auto",
    "single",
    "multi_point_strict",
    "multi_point_micro",
    "all_columns",
    "per_column_only",
    "multi_point_and_total",
    "multi_point_all",
    "custom",
]


def evaluation_modes_for_ui() -> List[Dict[str, str]]:
    """供前端下拉使用的有序模式列表。"""
    seen = set()
    out: List[Dict[str, str]] = []
    for mode_id in UI_EVALUATION_MODE_ORDER:
        if mode_id in EVALUATION_MODES and mode_id not in seen:
            out.append({"id": mode_id, "label": EVALUATION_MODES[mode_id]})
            seen.add(mode_id)
    for mode_id, label in EVALUATION_MODES.items():
        if mode_id not in seen:
            out.append({"id": mode_id, "label": label})
    return out

DEFAULT_POINT_PATTERN = r"采分点\s*\d+"
DEFAULT_TOTAL_NAMES = frozenset({"总分", "total_score", "total", "sum_score", "totalScore"})


def _gt_list(field_mapping: Dict[str, Any]) -> List[str]:
    fm = dict(field_mapping or {})
    gts: List[str] = []
    raw = fm.get("ground_truths") or []
    if isinstance(raw, str):
        raw = [raw]
    for g in raw:
        gs = str(g).strip()
        if gs and gs.lower() not in ("null", "none") and gs not in gts:
            gts.append(gs)
    legacy = fm.get("ground_truth") or fm.get("primary_ground_truth")
    if legacy and str(legacy).strip().lower() not in ("null", "none"):
        lg = str(legacy).strip()
        if lg not in gts:
            gts.insert(0, lg)
    return gts


def detect_point_columns(
    gt_columns: List[str],
    *,
    pattern: Optional[str] = None,
    explicit: Optional[List[str]] = None,
) -> List[str]:
    if explicit:
        return [c for c in explicit if c in gt_columns]
    rx = re.compile(pattern or DEFAULT_POINT_PATTERN, re.I)
    pts = [c for c in gt_columns if rx.search(str(c))]

    def _idx(col: str) -> int:
        m = re.search(r"(\d+)", str(col))
        return int(m.group(1)) if m else 999

    return sorted(pts, key=_idx)


def detect_total_columns(
    gt_columns: List[str],
    *,
    explicit: Optional[List[str]] = None,
) -> List[str]:
    if explicit:
        return [c for c in explicit if c in gt_columns]
    return [c for c in gt_columns if str(c) in DEFAULT_TOTAL_NAMES or "总分" in str(c)]


def parse_custom_evaluation(field_mapping: Dict[str, Any]) -> Dict[str, Any]:
    """解析 field_mapping.custom_evaluation（dict 或 JSON 字符串）。"""
    fm = dict(field_mapping or {})
    raw = fm.get("custom_evaluation")
    if raw is None:
        ep = fm.get("evaluation_profile") or {}
        if isinstance(ep, dict):
            raw = ep.get("custom_evaluation") or ep
    if isinstance(raw, str) and raw.strip():
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {"parse_error": raw[:200]}
    if not isinstance(raw, dict):
        return {}
    return dict(raw)


def _resolve_row_rule(
    mode: str,
    point_cols: List[str],
    total_cols: List[str],
    gt_cols: List[str],
    custom_eval: Optional[Dict[str, Any]] = None,
) -> str:
    mode = (mode or "auto").strip().lower()
    if mode == "custom":
        ce = custom_eval or {}
        plugin = (ce.get("plugin") or ce.get("preset") or "").strip()
        if plugin:
            return f"{ROW_RULE_PLUGIN_PREFIX}{plugin}"
        return ROW_RULE_PRIMARY
    if mode == "single" or mode == "per_column_only":
        return ROW_RULE_PRIMARY
    if mode in ("multi_point_strict", "multi_point_and_total"):
        return ROW_RULE_POINTS_AND_TOTAL
    if mode == "multi_point_micro":
        return ROW_RULE_POINTS_AND_TOTAL if total_cols else ROW_RULE_ALL_POINTS
    if mode == "multi_point_all":
        return ROW_RULE_ALL_POINTS
    if mode == "all_columns":
        return ROW_RULE_ALL_GT_COLUMNS
    # auto
    if len(point_cols) >= 2:
        return ROW_RULE_POINTS_AND_TOTAL if total_cols else ROW_RULE_ALL_POINTS
    if len(gt_cols) >= 2 and not point_cols:
        return ROW_RULE_ALL_GT_COLUMNS
    return ROW_RULE_PRIMARY


def build_evaluation_profile(
    field_mapping: Dict[str, Any],
    columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    生成评测配置（可来自 LLM field_mapping 或第二步 UI）。
    返回 dict，供比对、指标汇总、Prompt 工程、结果页文案使用。
    """
    fm = dict(field_mapping or {})
    gt_cols = _gt_list(fm)
    ep = fm.get("evaluation_profile") or {}
    if not isinstance(ep, dict):
        ep = {}

    mode = str(fm.get("evaluation_mode") or ep.get("mode") or "auto").strip().lower()
    point_pattern = fm.get("point_column_regex") or ep.get("point_column_regex") or DEFAULT_POINT_PATTERN
    point_explicit = fm.get("point_columns") or ep.get("point_columns")
    total_explicit = fm.get("total_columns") or ep.get("total_columns")

    point_cols = detect_point_columns(
        gt_cols, pattern=point_pattern, explicit=point_explicit if point_explicit else None
    )
    total_cols = detect_total_columns(gt_cols, explicit=total_explicit if total_explicit else None)

    primary = fm.get("primary_ground_truth") or fm.get("ground_truth")
    if primary and str(primary).strip().lower() in ("null", "none", ""):
        primary = None
    if not primary and gt_cols:
        primary = gt_cols[0]

    custom_eval = parse_custom_evaluation(fm)
    row_rule = _resolve_row_rule(mode, point_cols, total_cols, gt_cols, custom_eval)
    is_multi_point = len(point_cols) >= 2
    is_multi_column = len(gt_cols) >= 2

    from comparison_spec import (
        COMPARISON_UNIT_MULTI_POINT,
        COMPARISON_UNIT_SEQUENCE_CHAR,
        ROW_HEADLINE_BOTH,
        ROW_HEADLINE_MICRO,
        ROW_HEADLINE_STRICT,
        build_comparison_spec,
    )

    cmp_spec = build_comparison_spec(fm, columns)
    comparison_unit = cmp_spec.get("comparison_unit") or COMPARISON_UNIT_SCALAR
    row_headline = cmp_spec.get("row_headline") or ROW_HEADLINE_STRICT
    column_comparison_units = cmp_spec.get("column_comparison_units") or {}

    if row_headline == ROW_HEADLINE_MICRO or (
        row_headline == ROW_HEADLINE_BOTH and comparison_unit == COMPARISON_UNIT_SEQUENCE_CHAR
    ):
        headline_metric = HEADLINE_MICRO
    elif mode == "multi_point_micro":
        headline_metric = HEADLINE_MICRO
    elif is_multi_point or mode.startswith("multi_point"):
        headline_metric = HEADLINE_ROW_STRICT
    else:
        headline_metric = HEADLINE_ROW_STRICT

    output_format = str(fm.get("output_format") or ep.get("output_format") or "").strip()
    if not output_format:
        if is_multi_point or row_rule in (ROW_RULE_ALL_POINTS, ROW_RULE_POINTS_AND_TOTAL):
            output_format = "per_point"
        elif row_rule.startswith(ROW_RULE_PLUGIN_PREFIX):
            output_format = "per_point" if point_cols else "judge_or_single"
        else:
            output_format = "judge_or_single"

    metric_note = _metric_note(
        row_rule,
        point_cols,
        total_cols,
        primary,
        mode,
        custom_eval,
        headline_metric,
        comparison_unit=comparison_unit,
        row_headline=row_headline,
    )
    mode_label = EVALUATION_MODES.get(mode, EVALUATION_MODES["auto"])
    plugin_id = ""
    if row_rule.startswith(ROW_RULE_PLUGIN_PREFIX):
        plugin_id = row_rule[len(ROW_RULE_PLUGIN_PREFIX) :]
        from evaluation_plugins import list_plugins

        for p in list_plugins():
            if p.get("id") == plugin_id:
                mode_label = f"自定义：{p.get('label', plugin_id)}"
                break

    return {
        "evaluation_mode": mode,
        "evaluation_mode_label": mode_label,
        "row_correct_rule": row_rule,
        "custom_evaluation": custom_eval,
        "custom_plugin": plugin_id,
        "custom_plugin_params": custom_eval.get("params") if isinstance(custom_eval, dict) else {},
        "point_columns": point_cols,
        "scoring_point_columns": point_cols,
        "total_columns": total_cols,
        "scoring_total_columns": total_cols,
        "ground_truth_columns": gt_cols,
        "primary_ground_truth": primary,
        "is_multi_point": is_multi_point,
        "is_multi_column": is_multi_column,
        "output_format": output_format,
        "point_column_regex": point_pattern,
        "headline_metric": headline_metric,
        "judgment_metric_note": metric_note,
        "comparison_unit": comparison_unit,
        "comparison_unit_label": cmp_spec.get("comparison_unit_label"),
        "row_headline": row_headline,
        "row_headline_label": cmp_spec.get("row_headline_label"),
        "column_comparison_units": column_comparison_units,
        "legacy_evaluation_mode": "composite" if row_rule == ROW_RULE_POINTS_AND_TOTAL else (
            "single" if row_rule == ROW_RULE_PRIMARY else "multi"
        ),
    }


def summarize_point_units(
    by_col: Dict[str, Dict[str, Any]],
    point_columns: List[str],
) -> Dict[str, Any]:
    """单行采分点最小单位统计（不含总分列）。"""
    correct = 0
    total = 0
    for c in point_columns:
        d = by_col.get(c) or {}
        if normalize_value_simple(d.get("ground_truth")) is None:
            continue
        if d.get("correct") is None:
            continue
        total += 1
        if d.get("correct"):
            correct += 1
    ratio = round(correct / total, 4) if total else None
    return {
        "correct": correct,
        "total": total,
        "label": f"{correct}/{total}" if total else None,
        "ratio": ratio,
    }


def attach_point_units_to_by_col(
    by_col: Dict[str, Dict[str, Any]],
    profile: Dict[str, Any],
) -> None:
    point_cols = profile.get("point_columns") or []
    if len(point_cols) >= 2:
        by_col["_point_units"] = summarize_point_units(by_col, point_cols)


def aggregate_micro_from_responses(
    responses: List[Dict[str, Any]],
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    """跨行汇总采分点最小单位一致率。"""
    pc = pt = 0
    for r in responses:
        if r.get("status") != "ok":
            continue
        by = r.get("ground_truth_by_column") or {}
        pu = by.get("_point_units") or {}
        if not pu.get("total"):
            point_cols = profile.get("point_columns") or []
            pu = summarize_point_units(by, point_cols)
        pc += int(pu.get("correct") or 0)
        pt += int(pu.get("total") or 0)
    acc = round(pc / pt, 4) if pt else None
    return {
        "point_units_correct": pc,
        "point_units_total": pt,
        "accuracy": acc,
        "accuracy_pct": round(acc * 100, 2) if acc is not None else None,
        "label": f"{pc}/{pt}" if pt else None,
    }


def apply_headline_to_metrics(
    result: Dict[str, Any],
    responses: List[Dict[str, Any]],
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    """按 headline_metric / comparison_unit 调整主展示准确率（保留严格指标为副指标）。"""
    from comparison_spec import COMPARISON_UNIT_SEQUENCE_CHAR, aggregate_sequence_micro_from_responses

    if profile.get("headline_metric") != HEADLINE_MICRO:
        return result

    result["row_strict_accuracy"] = result.get("accuracy")
    result["row_strict_accuracy_pct"] = result.get("accuracy_pct")
    result["row_strict_rows_correct"] = result.get("rows_correct")
    result["row_strict_rows_evaluated"] = result.get("rows_evaluated")

    if profile.get("comparison_unit") == COMPARISON_UNIT_SEQUENCE_CHAR:
        micro = aggregate_sequence_micro_from_responses(responses, profile)
        if micro.get("sequence_units_total", 0) <= 0:
            return result
        result["accuracy"] = micro.get("accuracy")
        result["accuracy_pct"] = micro.get("accuracy_pct")
        result["sequence_units_correct"] = micro.get("sequence_units_correct")
        result["sequence_units_total"] = micro.get("sequence_units_total")
        result["sequence_units_label"] = micro.get("label")
        strict_pct = result.get("row_strict_accuracy_pct")
        result["judgment_metric_note"] = (
            f"主指标：逐字一致率（{micro.get('label')}）；"
            f"整串全对率 {strict_pct}%（适合自动化门槛）"
        )
        result["headline_metric"] = HEADLINE_MICRO
        result["comparison_unit"] = COMPARISON_UNIT_SEQUENCE_CHAR
        return result

    micro = aggregate_micro_from_responses(responses, profile)
    if micro.get("point_units_total", 0) <= 0:
        return result
    result["accuracy"] = micro.get("accuracy")
    result["accuracy_pct"] = micro.get("accuracy_pct")
    result["point_units_correct"] = micro.get("point_units_correct")
    result["point_units_total"] = micro.get("point_units_total")
    result["point_units_label"] = micro.get("label")
    result["judgment_metric_note"] = (
        f"主指标：采分点最小单位一致率（{micro.get('label')} 个采分点判对）；"
        f"整行全对率 {result.get('row_strict_accuracy_pct')}% 见副指标"
    )
    result["headline_metric"] = HEADLINE_MICRO
    return result


def _metric_note(
    row_rule: str,
    point_cols: List[str],
    total_cols: List[str],
    primary: Optional[str],
    mode: str,
    custom_eval: Optional[Dict[str, Any]] = None,
    headline_metric: str = HEADLINE_ROW_STRICT,
    *,
    comparison_unit: str = "",
    row_headline: str = "",
) -> str:
    from comparison_spec import COMPARISON_UNIT_SEQUENCE_CHAR

    if comparison_unit == COMPARISON_UNIT_SEQUENCE_CHAR:
        return (
            f"主指标：主标注列「{primary or 'judge_gt'}」逐字一致；"
            f"辅指标：整串全对（自动化门槛）"
        )
    if headline_metric == HEADLINE_MICRO:
        pts = "、".join(point_cols[:8]) if point_cols else "采分点"
        return f"主指标：{pts} 按最小单位计对（如 2/3）；非整行全对"
    if row_rule.startswith(ROW_RULE_PLUGIN_PREFIX):
        pid = row_rule[len(ROW_RULE_PLUGIN_PREFIX) :]
        from evaluation_plugins import list_plugins

        desc = pid
        for p in list_plugins():
            if p.get("id") == pid:
                desc = p.get("description") or p.get("label") or pid
                break
        return f"主指标（自定义插件 {pid}）：{desc}"
    if row_rule == ROW_RULE_POINTS_AND_TOTAL:
        pts = "、".join(point_cols[:8])
        tc = total_cols[0] if total_cols else "总分"
        return f"主指标：每行全部采分点（{pts}）与「{tc}」均与人工一致"
    if row_rule == ROW_RULE_ALL_POINTS:
        return f"主指标：每行全部采分点（{'、'.join(point_cols[:8])}）均与人工一致"
    if row_rule == ROW_RULE_ALL_GT_COLUMNS:
        return "主指标：每行已选全部标注列均与人工一致"
    if mode == "per_column_only":
        return f"主指标：主标注列「{primary}」一致；各标注列另有分列准确率"
    return f"主指标：主标注列「{primary or '（未指定）'}」与人工一致"


def apply_composite_judgment(
    by_col: Dict[str, Dict[str, Any]],
    profile: Dict[str, Any],
    *,
    preds_map: Optional[Dict[str, Any]] = None,
    sum_numeric_fn: Any = None,
) -> Dict[str, Dict[str, Any]]:
    """
    按 profile.row_correct_rule 写入 _composite* 字段（不改变各列逐列比对结果）。
    sum_numeric_fn: 可选，签名 (values: List) -> Optional[float]
    """
    rule = profile.get("row_correct_rule") or ROW_RULE_PRIMARY
    point_cols = profile.get("point_columns") or []
    total_cols = profile.get("total_columns") or []
    gt_cols = profile.get("ground_truth_columns") or []

    if rule == ROW_RULE_PRIMARY:
        return by_col

    def _comparable_cols(cols: List[str]) -> List[Dict[str, Any]]:
        return [
            by_col[c]
            for c in cols
            if c in by_col
            and normalize_value_simple((by_col.get(c) or {}).get("ground_truth")) is not None
            and (by_col.get(c) or {}).get("correct") is not None
        ]

    if rule == ROW_RULE_ALL_GT_COLUMNS:
        cols = [c for c in gt_cols if not str(c).startswith("_")]
        comp = _comparable_cols(cols)
        ok = bool(comp) and all(d.get("correct") for d in comp)
        by_col["_composite"] = {
            "correct": ok if comp else None,
            "label": "全部标注列一致",
            "columns": cols,
        }
        return by_col

    from comparison_spec import attach_sequence_units_to_by_col

    attach_sequence_units_to_by_col(by_col, profile)

    if rule in (ROW_RULE_ALL_POINTS, ROW_RULE_POINTS_AND_TOTAL) and point_cols:
        attach_point_units_to_by_col(by_col, profile)
        comp = _comparable_cols(point_cols)
        row_points_ok = bool(comp) and all(d.get("correct") for d in comp)
        pred_sum = None
        if sum_numeric_fn and preds_map is not None:
            pred_sum = sum_numeric_fn(
                [preds_map.get(c) or (by_col.get(c) or {}).get("predicted") for c in point_cols]
            )
        by_col["_composite_points"] = {
            "ground_truth": {c: (by_col.get(c) or {}).get("ground_truth") for c in point_cols},
            "predicted": {c: (by_col.get(c) or {}).get("predicted") for c in point_cols},
            "predicted_sum": pred_sum,
            "correct": row_points_ok if comp else None,
            "label": "全部采分点一致",
        }
        if rule == ROW_RULE_POINTS_AND_TOTAL and total_cols:
            tc = total_cols[0]
            gt_total = (by_col.get(tc) or {}).get("ground_truth")
            pred_total = preds_map.get(tc) if preds_map else None
            if pred_total is None:
                pred_total = (by_col.get(tc) or {}).get("predicted")
            if pred_total is None and pred_sum is not None:
                pred_total = pred_sum
            total_ok = (by_col.get(tc) or {}).get("correct")
            by_col["_composite_total"] = {
                "ground_truth": gt_total,
                "predicted": pred_total,
                "correct": total_ok,
                "label": "总分一致",
            }
            if comp and total_ok is not None:
                by_col["_composite"] = {
                    "correct": row_points_ok and total_ok,
                    "label": "采分点全对且总分一致",
                }
            else:
                by_col["_composite"] = dict(by_col.get("_composite_points") or {})
        else:
            by_col["_composite"] = dict(by_col.get("_composite_points") or {})
        return by_col

    if rule.startswith(ROW_RULE_PLUGIN_PREFIX):
        return _apply_plugin_judgment(by_col, profile, preds_map=preds_map)

    return by_col


def _apply_plugin_judgment(
    by_col: Dict[str, Dict[str, Any]],
    profile: Dict[str, Any],
    *,
    preds_map: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    from evaluation_plugins import RowEvalContext, get_plugin

    rule = profile.get("row_correct_rule") or ""
    plugin_id = rule[len(ROW_RULE_PLUGIN_PREFIX) :] if rule.startswith(ROW_RULE_PLUGIN_PREFIX) else ""
    plugin_id = plugin_id or profile.get("custom_plugin") or ""
    fn = get_plugin(plugin_id)
    if not fn:
        by_col["_composite"] = {
            "correct": None,
            "label": f"未知插件: {plugin_id}",
            "error": "plugin_not_found",
        }
        return by_col
    params = dict(profile.get("custom_plugin_params") or {})
    ce = profile.get("custom_evaluation") or {}
    if isinstance(ce, dict) and ce.get("params"):
        params.update(ce.get("params") or {})
    ctx = RowEvalContext(by_col=by_col, profile=profile, preds_map=preds_map, params=params)
    try:
        ok = fn(ctx)
    except Exception as e:
        by_col["_composite"] = {
            "correct": None,
            "label": f"插件 {plugin_id} 执行失败",
            "error": str(e)[:200],
        }
        return by_col
    by_col["_composite"] = {
        "correct": ok,
        "label": profile.get("evaluation_mode_label") or f"插件:{plugin_id}",
        "plugin": plugin_id,
        "params": params,
    }
    return by_col


def row_level_correct(
    by_col: Dict[str, Dict[str, Any]],
    profile: Dict[str, Any],
) -> Optional[bool]:
    """根据评测配置汇总行级 correct。"""
    rule = profile.get("row_correct_rule") or ROW_RULE_PRIMARY
    if (rule != ROW_RULE_PRIMARY or profile.get("evaluation_mode") == "custom") and "_composite" in by_col:
        return (by_col.get("_composite") or {}).get("correct")
    primary = profile.get("primary_ground_truth")
    if primary and primary in by_col:
        return (by_col.get(primary) or {}).get("correct")
    for c, d in by_col.items():
        if str(c).startswith("_"):
            continue
        if isinstance(d, dict) and d.get("correct") is not None:
            return d.get("correct")
    return None


def normalize_value_simple(val: Any) -> Optional[str]:
    """轻量 normalize，避免 evaluation 与 platform_core 循环导入。"""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "null", "n/a"):
        return None
    return s


def prompt_output_hint(profile: Dict[str, Any], gt_columns: List[str]) -> str:
    """按评测配置生成 Prompt 追加说明（供 apply_prompt_engineering 调用）。"""
    rule = profile.get("row_correct_rule") or ROW_RULE_PRIMARY
    custom_hint = (profile.get("custom_evaluation") or {}).get("prompt_suffix")
    if custom_hint:
        return "\n\n" + str(custom_hint).strip()
    if rule.startswith(ROW_RULE_PLUGIN_PREFIX):
        pts = profile.get("point_columns") or []
        cols = pts or profile.get("ground_truth_columns") or gt_columns
        if cols:
            keys = ", ".join(json.dumps(c, ensure_ascii=False) for c in cols)
            return (
                f"\n\n【输出要求】只输出一行 JSON，键名与列名一致：{keys}。"
                "不要 markdown，不要解释。"
            )
    if rule in (ROW_RULE_ALL_POINTS, ROW_RULE_POINTS_AND_TOTAL):
        import json

        pts = profile.get("point_columns") or []
        keys = ", ".join(json.dumps(c, ensure_ascii=False) for c in pts)
        example: Dict[str, Any] = {c: 0 for c in pts}
        total_cols = profile.get("total_columns") or []
        if total_cols:
            example[total_cols[0]] = 0
        tail = (
            f"\n\n【输出要求】只输出一行 JSON，键名必须与列名完全一致（勿用①②等别名）：{keys}"
        )
        if total_cols:
            tail += f"，以及「{total_cols[0]}」（各采分点得分之和）"
        tail += f"。示例：{json.dumps(example, ensure_ascii=False)}。不要 markdown，不要解释。"
        return tail
    if rule == ROW_RULE_ALL_GT_COLUMNS:
        import json

        cols = profile.get("ground_truth_columns") or gt_columns
        keys = ", ".join(json.dumps(c, ensure_ascii=False) for c in cols)
        return (
            f"\n\n【输出要求】只输出一行 JSON，须包含各标注字段：{keys}，"
            "值与人工标注口径一致。不要 markdown，不要解释。"
        )
    return (
        "\n\n【输出要求】只输出一行 JSON，必须包含 is_correct 字段（0 或 1，与人工标注含义一致），"
        "可选 confidence（0~1）。不要 markdown，不要解释。"
    )
