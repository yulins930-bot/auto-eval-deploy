# -*- coding: utf-8 -*-
"""
比对规格（Comparison Spec）：在有限边界内描述「怎么比」，避免按任务写死指标。

comparison_unit（比对单位）
  - scalar：行内单值（0/1、可用/不可用、单标签）
  - sequence_char：等长或按金标对齐的 a/b/c 等字符序列（错别字 judge_gt）
  - multi_column：多列各自 scalar，行级由 row_correct_rule 汇总
  - multi_point：采分点列（复用现有 multi_point_* 逻辑）

row_headline（主指标）
  - strict：严格一致（整串/整行全对）
  - micro：最小单位计对（逐字、逐采分点）
  - both：主展示 micro，附 strict
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

COMPARISON_UNIT_SCALAR = "scalar"
COMPARISON_UNIT_SEQUENCE_CHAR = "sequence_char"
COMPARISON_UNIT_MULTI_COLUMN = "multi_column"
COMPARISON_UNIT_MULTI_POINT = "multi_point"

COMPARISON_UNITS = {
    COMPARISON_UNIT_SCALAR: "单值：每行一个标注（0/1、标签、对错）",
    COMPARISON_UNIT_SEQUENCE_CHAR: "字符序列：如 judge 串与标答等长的 a/b/c",
    COMPARISON_UNIT_MULTI_COLUMN: "多列：每列独立单值，行级由规则汇总",
    COMPARISON_UNIT_MULTI_POINT: "多采分点：按采分点列逐列比对",
}

ROW_HEADLINE_STRICT = "strict"
ROW_HEADLINE_MICRO = "micro"
ROW_HEADLINE_BOTH = "both"

ROW_HEADLINE_MODES = {
    ROW_HEADLINE_STRICT: "严格一致：整串/整行全对才算对",
    ROW_HEADLINE_MICRO: "微观一致：按最小单位计对率（逐字/逐采分点）",
    ROW_HEADLINE_BOTH: "双指标：主看微观，附严格一致率",
}

_ABC_SEQUENCE_RE = re.compile(r"^[abc?]+$", re.I)
_BIEZI_COL_RE = re.compile(r"^biezi_\d+$", re.I)
_CUOZI_COL_RE = re.compile(r"^cuozi_\d+$", re.I)
_POINT_COL_FALLBACK = re.compile(r"采分点\s*\d+", re.I)

# 列名提示 → 序列型金标
_SEQUENCE_GT_NAME_HINTS = ("judge_gt", "judge", "verdict_seq", "label_seq")
# 列名提示 → 二元/标量
_SCALAR_GT_NAME_HINTS = (
    "识别结果",
    "是否",
    "可用",
    "label",
    "score",
    "is_correct",
    "wrong",
)


def _strip_val(val: Any) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "null", "n/a"):
        return None
    return s


def normalize_scalar(val: Any) -> Optional[str]:
    """与 platform_core.normalize_value 对齐的轻量版（避免循环导入）。"""
    s = _strip_val(val)
    if s is None:
        return None
    sl = s.lower()
    if sl in ("1", "true", "yes", "y", "正确", "对", "right", "correct", "是"):
        return "1"
    if sl in ("0", "false", "no", "n", "错误", "错", "wrong", "incorrect", "否"):
        return "0"
    if sl in ("a", "b", "c"):
        return sl
    return s


def normalize_abc_sequence(val: Any) -> Optional[str]:
    s = _strip_val(val)
    if s is None:
        return None
    s = re.sub(r"\s+", "", s).lower()
    if not s:
        return None
    if s in ("ambiguous", "unknown", "?"):
        return "?"
    if _ABC_SEQUENCE_RE.match(s):
        return s
    return s


def looks_like_abc_sequence(val: Any) -> bool:
    s = normalize_abc_sequence(val)
    return bool(s) and s != "?" and len(s) >= 1


def sample_values_for_column(
    rows: List[Dict[str, str]],
    column: str,
    *,
    max_samples: int = 40,
) -> List[str]:
    out: List[str] = []
    for row in rows:
        v = normalize_scalar(row.get(column))
        if v is None:
            v = _strip_val(row.get(column))
        if v is not None:
            out.append(v)
        if len(out) >= max_samples:
            break
    return out


def infer_comparison_unit(
    field_mapping: Dict[str, Any],
    columns: Optional[List[str]] = None,
    *,
    sample_rows: Optional[List[Dict[str, str]]] = None,
) -> str:
    """从列映射与样本推断 comparison_unit（可被 evaluation_profile 显式覆盖）。"""
    fm = dict(field_mapping or {})
    ep = fm.get("evaluation_profile") or {}
    if not isinstance(ep, dict):
        ep = {}
    explicit = (ep.get("comparison_unit") or fm.get("comparison_unit") or "").strip().lower()
    if explicit in COMPARISON_UNITS:
        return explicit

    from evaluation import _gt_list, detect_point_columns, DEFAULT_POINT_PATTERN

    gt_cols = _gt_list(fm)
    point_pattern = fm.get("point_column_regex") or ep.get("point_column_regex") or DEFAULT_POINT_PATTERN
    point_cols = detect_point_columns(gt_cols, pattern=point_pattern)

    if len(point_cols) >= 2:
        return COMPARISON_UNIT_MULTI_POINT
    if len(gt_cols) >= 2:
        if any(_BIEZI_COL_RE.match(str(c)) for c in gt_cols) or any(
            _CUOZI_COL_RE.match(str(c)) for c in gt_cols
        ):
            return COMPARISON_UNIT_MULTI_COLUMN
        return COMPARISON_UNIT_MULTI_COLUMN

    primary = fm.get("primary_ground_truth") or fm.get("ground_truth")
    if not primary and gt_cols:
        primary = gt_cols[0]

    samples: List[str] = []
    if sample_rows and primary:
        samples = sample_values_for_column(sample_rows, primary)

    if primary:
        pn = str(primary).lower()
        if any(h in pn for h in _SEQUENCE_GT_NAME_HINTS):
            if not samples or sum(1 for s in samples if looks_like_abc_sequence(s)) >= max(
                1, len(samples) // 2
            ):
                return COMPARISON_UNIT_SEQUENCE_CHAR
        if any(h in str(primary) for h in _SCALAR_GT_NAME_HINTS):
            return COMPARISON_UNIT_SCALAR

    if samples:
        abc_n = sum(1 for s in samples if looks_like_abc_sequence(s))
        if abc_n >= max(1, len(samples) * 2 // 3):
            return COMPARISON_UNIT_SEQUENCE_CHAR
        if all(s in ("0", "1") for s in samples):
            return COMPARISON_UNIT_SCALAR

    return COMPARISON_UNIT_SCALAR


def infer_row_headline(
    comparison_unit: str,
    field_mapping: Dict[str, Any],
) -> str:
    fm = dict(field_mapping or {})
    ep = fm.get("evaluation_profile") or {}
    if not isinstance(ep, dict):
        ep = {}
    explicit = (ep.get("row_headline") or fm.get("row_headline") or "").strip().lower()
    if explicit in ROW_HEADLINE_MODES:
        return explicit

    mode = str(fm.get("evaluation_mode") or ep.get("mode") or "auto").strip().lower()
    if mode == "multi_point_micro":
        return ROW_HEADLINE_MICRO
    if comparison_unit == COMPARISON_UNIT_SEQUENCE_CHAR:
        return ROW_HEADLINE_BOTH
    if comparison_unit == COMPARISON_UNIT_MULTI_POINT:
        return ROW_HEADLINE_MICRO
    return ROW_HEADLINE_STRICT


def infer_column_comparison_units(
    field_mapping: Dict[str, Any],
    *,
    sample_rows: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, str]:
    """每标注列可覆盖 unit；默认继承任务级 comparison_unit。"""
    from evaluation import _gt_list

    default_unit = infer_comparison_unit(field_mapping, sample_rows=sample_rows)
    ep = field_mapping.get("evaluation_profile") or {}
    overrides = ep.get("column_comparison_units") if isinstance(ep, dict) else {}
    if not isinstance(overrides, dict):
        overrides = {}

    out: Dict[str, str] = {}
    for col in _gt_list(field_mapping):
        unit = str(overrides.get(col) or "").strip().lower()
        if unit in COMPARISON_UNITS:
            out[col] = unit
            continue
        if sample_rows:
            samples = sample_values_for_column(sample_rows, col)
            if samples and sum(1 for s in samples if looks_like_abc_sequence(s)) >= max(
                1, len(samples) * 2 // 3
            ):
                out[col] = COMPARISON_UNIT_SEQUENCE_CHAR
                continue
        out[col] = default_unit
    return out


def compare_sequence_char(predicted: Any, ground_truth: Any) -> Dict[str, Any]:
    p = normalize_abc_sequence(predicted)
    g = normalize_abc_sequence(ground_truth)
    if g is None:
        return {
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": None,
            "strict_correct": None,
            "micro_correct": 0,
            "micro_total": 0,
            "micro_ratio": None,
            "match_detail": "",
            "comparison_unit": COMPARISON_UNIT_SEQUENCE_CHAR,
        }
    if p is None:
        return {
            "ground_truth": g,
            "predicted": predicted,
            "correct": False,
            "strict_correct": False,
            "micro_correct": 0,
            "micro_total": len(g),
            "micro_ratio": 0.0,
            "match_detail": f"0/{len(g)}字",
            "comparison_unit": COMPARISON_UNIT_SEQUENCE_CHAR,
        }

    strict = p == g
    total = len(g)
    correct_chars = sum(1 for i in range(total) if i < len(p) and p[i] == g[i])
    ratio = round(correct_chars / total, 4) if total else None
    detail = f"{correct_chars}/{total}字"
    if not strict and len(p) != len(g):
        detail += f"（预测长{len(p)}·金标长{total}）"

    return {
        "ground_truth": g,
        "predicted": p,
        "correct": strict,
        "strict_correct": strict,
        "micro_correct": correct_chars,
        "micro_total": total,
        "micro_ratio": ratio,
        "match_detail": detail,
        "comparison_unit": COMPARISON_UNIT_SEQUENCE_CHAR,
    }


def compare_scalar(predicted: Any, ground_truth: Any) -> Dict[str, Any]:
    gt_n = normalize_scalar(ground_truth)
    pred_n = normalize_scalar(predicted)
    if gt_n is None:
        ok = None
    elif pred_n is None:
        ok = None
    else:
        ok = pred_n == gt_n
    return {
        "ground_truth": ground_truth,
        "predicted": predicted,
        "correct": ok,
        "strict_correct": ok,
        "micro_correct": 1 if ok else 0 if ok is False else 0,
        "micro_total": 1 if ok is not None else 0,
        "micro_ratio": 1.0 if ok else (0.0 if ok is False else None),
        "match_detail": "✓" if ok else ("✗" if ok is False else "?"),
        "comparison_unit": COMPARISON_UNIT_SCALAR,
    }


def compare_column_values(
    predicted: Any,
    ground_truth: Any,
    unit: str,
) -> Dict[str, Any]:
    unit = (unit or COMPARISON_UNIT_SCALAR).strip().lower()
    if unit == COMPARISON_UNIT_SEQUENCE_CHAR:
        return compare_sequence_char(predicted, ground_truth)
    return compare_scalar(predicted, ground_truth)


def summarize_sequence_units(by_col: Dict[str, Dict[str, Any]], primary: Optional[str]) -> Dict[str, Any]:
    """单行序列微观统计（写入 by_col['_sequence_units']）。"""
    cols = [primary] if primary else []
    for c, d in by_col.items():
        if str(c).startswith("_"):
            continue
        if (d or {}).get("comparison_unit") == COMPARISON_UNIT_SEQUENCE_CHAR:
            if c not in cols:
                cols.append(c)
    mc = mt = 0
    for c in cols:
        d = by_col.get(c) or {}
        if d.get("micro_total"):
            mc += int(d.get("micro_correct") or 0)
            mt += int(d.get("micro_total") or 0)
    ratio = round(mc / mt, 4) if mt else None
    return {
        "correct": mc,
        "total": mt,
        "label": f"{mc}/{mt}字" if mt else None,
        "ratio": ratio,
        "primary_column": primary,
    }


def attach_sequence_units_to_by_col(
    by_col: Dict[str, Dict[str, Any]],
    profile: Dict[str, Any],
) -> None:
    if profile.get("comparison_unit") != COMPARISON_UNIT_SEQUENCE_CHAR:
        return
    primary = profile.get("primary_ground_truth")
    su = summarize_sequence_units(by_col, primary)
    if su.get("total"):
        by_col["_sequence_units"] = su


def aggregate_sequence_micro_from_responses(
    responses: List[Dict[str, Any]],
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    mc = mt = 0
    for r in responses:
        if r.get("status") != "ok":
            continue
        by = r.get("ground_truth_by_column") or {}
        su = by.get("_sequence_units") or {}
        if not su.get("total"):
            primary = profile.get("primary_ground_truth")
            su = summarize_sequence_units(by, primary)
        mc += int(su.get("correct") or 0)
        mt += int(su.get("total") or 0)
    acc = round(mc / mt, 4) if mt else None
    return {
        "sequence_units_correct": mc,
        "sequence_units_total": mt,
        "accuracy": acc,
        "accuracy_pct": round(acc * 100, 2) if acc is not None else None,
        "label": f"{mc}/{mt}字" if mt else None,
    }


def format_sequence_match_line(by_col: Dict[str, Any], primary: Optional[str]) -> str:
    parts: List[str] = []
    if primary and primary in by_col:
        d = by_col[primary] or {}
        sym = "?"
        if d.get("strict_correct") is True:
            sym = "✓"
        elif d.get("strict_correct") is False:
            sym = "✗"
        md = d.get("match_detail") or ""
        parts.append(f"{primary}{sym}" + (f"({md})" if md else ""))
    su = by_col.get("_sequence_units") or {}
    if su.get("label") and not parts:
        parts.append(f"逐字{su['label']}")
    return " ".join(parts)


def default_prediction_key_mapping(field_mapping: Dict[str, Any]) -> Dict[str, str]:
    """未配置时：judge → judge_gt 等常见别名。"""
    from evaluation import _gt_list

    pkm = dict(field_mapping.get("prediction_key_mapping") or {})
    gt_cols = _gt_list(field_mapping)
    for col in gt_cols:
        cl = col.lower()
        if cl == "judge_gt" and "judge" not in pkm:
            pkm["judge"] = col
        if col == "识别结果（0=错误）" and "is_error" not in pkm:
            pkm["is_error"] = col
            pkm["识别结果"] = col
    return pkm


def build_comparison_spec(
    field_mapping: Dict[str, Any],
    columns: Optional[List[str]] = None,
    *,
    sample_rows: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    unit = infer_comparison_unit(field_mapping, columns, sample_rows=sample_rows)
    headline = infer_row_headline(unit, field_mapping)
    col_units = infer_column_comparison_units(field_mapping, sample_rows=sample_rows)
    return {
        "comparison_unit": unit,
        "comparison_unit_label": COMPARISON_UNITS.get(unit, unit),
        "row_headline": headline,
        "row_headline_label": ROW_HEADLINE_MODES.get(headline, headline),
        "column_comparison_units": col_units,
        "suggested_prediction_key_mapping": default_prediction_key_mapping(field_mapping),
    }
