# -*- coding: utf-8 -*-
"""自动化评测核心逻辑（本地 Flask 用）：澄清、数据准备、Prompt 渲染。"""
from __future__ import annotations

import csv
import json
import math
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import jinja2


def parse_model_json(raw_text: str) -> Dict[str, Any]:
    if not raw_text:
        return {"raw": "", "parse_error": "empty output"}
    text = raw_text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fenced:
        text = fenced.group(1)
    else:
        first, last = text.find("{"), text.rfind("}")
        if first >= 0 and last > first:
            text = text[first : last + 1]
        else:
            first, last = text.find("["), text.rfind("]")
            if first >= 0 and last > first:
                text = text[first : last + 1]
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"prediction": parsed[0] if parsed else None, "results": parsed}
        return {"prediction": parsed}
    except Exception as e:
        m = re.search(r"list\s*\[\s*[\"']([^\"']+)[\"']\s*\]", text, re.I)
        if m:
            return {"prediction": m.group(1), "results": [m.group(1)]}
        return {"raw": raw_text, "parse_error": str(e)}


def row_to_dict(columns: List[str], values: List[str]) -> Dict[str, str]:
    d: Dict[str, str] = {}
    for i, col in enumerate(columns):
        d[col] = values[i] if i < len(values) else ""
    return d


def preview_from_table(
    columns: List[str], all_rows: List[List[str]], total: int, n: int = 5
) -> Tuple[List[str], List[Dict[str, str]], int]:
    preview = [row_to_dict(columns, rv) for rv in all_rows[:n]]
    return columns, preview, total


def load_clarify_template(templates_dir: Path) -> str:
    p = templates_dir / "clarify_task.j2"
    if p.is_file():
        return p.read_text(encoding="utf-8")
    return _FALLBACK_CLARIFY_TEMPLATE


_PRIMARY_CONTENT_HINT_RE = re.compile(
    r"学生作答|作答原图|作答图|student.?answer|submission|手写|答题|worksheet|原图|scan",
    re.I,
)
_CORE_INPUT_HINT_RE = re.compile(
    r"原题|推题|题干|题目|材料|正文|内容|相似题|对比题|推荐题|候选题",
    re.I,
)


def suggest_primary_content_columns(
    columns: List[str],
    rows: Optional[List[List[str]]] = None,
) -> List[str]:
    """兼容旧名：返回 suggest_core_input_columns 的结果。"""
    return suggest_core_input_columns(columns, rows)


def suggest_core_input_columns(
    columns: List[str],
    rows: Optional[List[List[str]]] = None,
) -> List[str]:
    """启发式列出最可能的核心输入列（作答 / 原题+推题 / 题干材料等）。"""
    colset = [str(c).strip() for c in columns if c]
    scored: List[Tuple[int, str]] = []
    for c in colset:
        if c.startswith("platform_"):
            continue
        score = 0
        if _CORE_INPUT_HINT_RE.search(c):
            score += 7
        if _PRIMARY_CONTENT_HINT_RE.search(c):
            score += 6
        low = c.lower()
        if any(k in low for k in ("img", "图", "url", "image", "photo", "pic")):
            score += 2
        if any(k in c for k in ("作答", "answer", "response", "提交")):
            score += 2
        if "信息" in c and any(k in c for k in ("原题", "推题", "题干")):
            score += 4
        if score > 0:
            scored.append((score, c))
    scored.sort(key=lambda x: (-x[0], x[1]))
    out = [c for _, c in scored[:10]]
    # 教研相似题：原题+推题成对出现时优先一起列出
    pair_order = ("原题信息", "推题信息", "原题", "推题")
    for key in pair_order:
        for c in colset:
            if key in c and c not in out:
                out.insert(0, c)
    deduped: List[str] = []
    for c in out:
        if c not in deduped:
            deduped.append(c)
    return deduped[:10]


def render_clarify_prompt(
    template_text: str,
    *,
    task_description: str,
    user_output_expectation: str,
    file_path: str,
    columns_list: List[str],
    preview_rows: List[Dict[str, str]],
    annotation_column_hints: Optional[List[str]] = None,
    primary_content_hints: Optional[List[str]] = None,
    rubric_text: str = "",
) -> str:
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    tpl = env.from_string(template_text)
    hints = annotation_column_hints or []
    primary_hints = primary_content_hints or []
    rubric = (rubric_text or "").strip()
    if primary_hints:
        primary_note = "、".join(primary_hints)
    else:
        primary_note = "（未从列名自动识别；你必须结合预览判断核心输入列，并在 clarifying_questions 中追问）"
    return tpl.render(
        task_description=task_description or "(用户未填写)",
        user_output_expectation=user_output_expectation or "(用户未填写，请根据任务描述推断并询问)",
        file_path=file_path,
        columns_list=json.dumps(columns_list, ensure_ascii=False),
        columns_only_note="、".join(columns_list) if columns_list else "(无列名)",
        annotation_column_hints="、".join(hints) if hints else "（未自动识别，请根据列名与预览判断）",
        primary_content_hints=primary_note,
        rubric_text=rubric[:12000] if rubric else "（未上传；若任务依赖独立评分标准文件，请用户上传 .txt/.md）",
        sample_count=len(preview_rows),
        sample_rows_json=json.dumps(preview_rows, ensure_ascii=False, indent=2),
    )


def load_clarify_followup_template(templates_dir: Path) -> str:
    p = templates_dir / "clarify_followup.j2"
    if p.is_file():
        return p.read_text(encoding="utf-8")
    return "请根据用户补充更新 field_mapping 与 suggested_prompt，只输出 JSON。"


def render_clarify_followup_prompt(
    template_text: str,
    *,
    task_description: str,
    user_output_expectation: str,
    file_path: str,
    columns_list: List[str],
    preview_rows: List[Dict[str, str]],
    previous_clarified: Dict[str, Any],
    user_answers: Dict[str, Any],
    clarifying_questions: List[str],
    current_field_mapping: Dict[str, Any],
    current_prompt: str,
    prompt_revision_instruction: str = "",
    rubric_text: str = "",
) -> str:
    qa_lines: List[str] = []
    for i, q in enumerate(clarifying_questions or []):
        ans = (user_answers.get(str(i)) or user_answers.get(i) or "").strip()
        qa_lines.append(f"- 问：{q}\n  答：{ans or '（未填写）'}")
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    tpl = env.from_string(template_text)
    return tpl.render(
        task_description=task_description or "(用户未填写)",
        user_output_expectation=user_output_expectation or "(用户未填写)",
        file_path=file_path,
        columns_list=json.dumps(columns_list, ensure_ascii=False),
        columns_only_note="、".join(columns_list) if columns_list else "(无列名)",
        sample_count=len(preview_rows),
        sample_rows_json=json.dumps(preview_rows, ensure_ascii=False, indent=2),
        previous_clarified_json=json.dumps(previous_clarified, ensure_ascii=False, indent=2),
        user_answers_block="\n".join(qa_lines) if qa_lines else "（无澄清问题或未填写）",
        current_field_mapping_json=json.dumps(current_field_mapping or {}, ensure_ascii=False, indent=2),
        current_prompt_excerpt=(current_prompt or "")[:2500],
        prompt_revision_instruction=(prompt_revision_instruction or "").strip() or "（未填写）",
        rubric_text=((rubric_text or "").strip()[:12000] or "（未上传）"),
    )


def load_clarify_interpret_template(templates_dir: Path) -> str:
    p = templates_dir / "clarify_interpret.j2"
    if p.is_file():
        return p.read_text(encoding="utf-8")
    return (
        "请根据以下评测配置 JSON，用中文写一段给用户看的「最终解读」（3～6 句），"
        "说明任务理解、列怎么用、模型每行输出什么、如何与人工标注比对。只输出 JSON："
        '{"interpretation_summary":"..."}'
    )


def render_clarify_interpret_prompt(
    template_text: str,
    *,
    clarified: Dict[str, Any],
    user_answers: Dict[str, Any],
    clarifying_questions: List[str],
) -> str:
    qa_lines: List[str] = []
    for i, q in enumerate(clarifying_questions or []):
        ans = (user_answers.get(str(i)) or user_answers.get(i) or "").strip()
        if ans:
            qa_lines.append(f"- {q} → {ans}")
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    tpl = env.from_string(template_text)
    return tpl.render(
        clarified_json=json.dumps(clarified, ensure_ascii=False, indent=2),
        user_answers_block="\n".join(qa_lines) if qa_lines else "（无）",
    )


_ANNOTATION_HINT_RE = re.compile(
    r"识别|标注|judge|打分|采分|人工|结果|评分|label|score|对错|正确|错误|判定|grade|verdict",
    re.I,
)


def suggest_annotation_columns(
    columns: List[str],
    rows: Optional[List[List[str]]] = None,
    *,
    columns_stats: Optional[Dict[str, int]] = None,
) -> List[str]:
    """启发式列出可能的人工标注列，供澄清 LLM 与前端勾选参考。"""
    colset = [str(c).strip() for c in columns if c]
    scored: List[Tuple[int, str]] = []
    for c in colset:
        score = 0
        if _ANNOTATION_HINT_RE.search(c):
            score += 3
        if c.startswith("platform_"):
            score -= 5
        if columns_stats and columns_stats.get(c, 0) > 0:
            score += 1
        if score > 0:
            scored.append((score, c))
    scored.sort(key=lambda x: (-x[0], x[1]))
    out: List[str] = []
    seen: set[str] = set()
    for _, c in scored:
        if c not in seen:
            seen.add(c)
            out.append(c)
    if rows and len(out) < 8:
        for c in colset:
            if c in seen or c.startswith("platform_"):
                continue
            vals = []
            for rv in rows[:min(80, len(rows))]:
                rd = row_to_dict(columns, rv)
                v = normalize_value(rd.get(c))
                if v is not None:
                    vals.append(v)
            if len(vals) >= 3:
                uniq = set(vals)
                if len(uniq) <= 6 and uniq.issubset({"0", "1", "a", "b", "c"} | set(vals)):
                    out.append(c)
                    seen.add(c)
            if len(out) >= 12:
                break
    return out[:20]


def _fm_col_ok(name: Any, colset: set[str]) -> bool:
    return bool(name) and str(name).strip() in colset and str(name).lower() not in ("null", "none")


def finalize_field_mapping(fm: Dict[str, Any]) -> Dict[str, Any]:
    """合并 reference/stem/rubric、ground_truths 与 legacy ground_truth。"""
    out: Dict[str, Any] = dict(fm or {})
    ctx: List[str] = []
    for key in ("reference_column", "stem_column", "rubric_column"):
        v = out.get(key)
        if v and str(v).strip().lower() not in ("null", "none", ""):
            ctx.append(str(v).strip())
    legacy_ctx = out.get("context") or []
    if isinstance(legacy_ctx, str):
        legacy_ctx = [legacy_ctx] if legacy_ctx else []
    for c in legacy_ctx:
        cs = str(c).strip()
        if cs and cs not in ctx:
            ctx.append(cs)
    out["context"] = ctx
    out["reference_column"] = out.get("reference_column") or (ctx[0] if len(ctx) > 0 else None)
    out["stem_column"] = out.get("stem_column") or (ctx[1] if len(ctx) > 1 else None)
    out["rubric_column"] = out.get("rubric_column") or (ctx[2] if len(ctx) > 2 else None)

    gts: List[str] = []
    raw_gts = out.get("ground_truths") or []
    if isinstance(raw_gts, str):
        raw_gts = [raw_gts]
    for g in raw_gts:
        gs = str(g).strip()
        if gs and gs.lower() not in ("null", "none") and gs not in gts:
            gts.append(gs)
    legacy_gt = out.get("ground_truth")
    if legacy_gt and str(legacy_gt).strip().lower() not in ("null", "none"):
        lg = str(legacy_gt).strip()
        if lg not in gts:
            gts.insert(0, lg)
    out["ground_truths"] = gts
    primary = out.get("primary_ground_truth") or out.get("ground_truth")
    if primary and str(primary).strip().lower() in ("null", "none", ""):
        primary = None
    if primary and str(primary).strip() not in gts:
        gts.insert(0, str(primary).strip())
        out["ground_truths"] = gts
    if not primary and gts:
        primary = gts[0]
    out["primary_ground_truth"] = primary
    out["ground_truth"] = primary

    raw_core = out.get("core_inputs") or []
    if isinstance(raw_core, str):
        raw_core = [raw_core] if raw_core.strip() else []
    core_inputs: List[str] = []
    for c in raw_core:
        cs = str(c).strip()
        if cs and cs.lower() not in ("null", "none") and cs not in core_inputs:
            core_inputs.append(cs)
    pc = (out.get("primary_content") or "").strip()
    if pc and pc not in core_inputs:
        core_inputs.insert(0, pc)
    for c in ctx:
        if c and c not in core_inputs and any(k in c for k in ("原题", "推题", "题干", "相似", "对比")):
            core_inputs.append(c)
    if not core_inputs and pc:
        core_inputs = [pc]
    out["core_inputs"] = core_inputs
    if core_inputs and not pc:
        out["primary_content"] = core_inputs[0]

    ep = infer_scoring_profile([], out)
    out["evaluation_mode"] = ep.get("evaluation_mode", "auto")
    out["evaluation_profile"] = {
        "mode": ep.get("evaluation_mode"),
        "row_correct_rule": ep.get("row_correct_rule"),
        "point_columns": ep.get("point_columns"),
        "total_columns": ep.get("total_columns"),
        "point_column_regex": ep.get("point_column_regex"),
        "custom_evaluation": ep.get("custom_evaluation"),
        "custom_plugin": ep.get("custom_plugin"),
        "comparison_unit": ep.get("comparison_unit"),
        "row_headline": ep.get("row_headline"),
        "column_comparison_units": ep.get("column_comparison_units"),
    }
    out.update(ep)
    if not out.get("prediction_key_mapping"):
        from comparison_spec import default_prediction_key_mapping

        suggested = default_prediction_key_mapping(out)
        if suggested:
            out["prediction_key_mapping"] = suggested
    return out


_POINT_COL_RE = re.compile(r"采分点\s*\d+", re.I)
_TOTAL_SCORE_NAMES = ("总分", "total_score", "total", "sum_score")
_CIRCLE_DIGITS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
_TOTAL_KEY_NAMES = frozenset(
    {"总分", "total", "total_score", "sum", "sum_score", "totalScore", "满分"}
)


def scoring_point_index_from_column(column_name: str) -> Optional[int]:
    m = re.search(r"采分点\s*(\d+)", str(column_name), re.I)
    return int(m.group(1)) if m else None


def scoring_point_index_from_model_key(key: str) -> Optional[int]:
    """从模型 JSON 键名推断采分点序号（1-based），如 ①、采分点②、2。"""
    k = str(key).strip()
    if not k or k in _TOTAL_KEY_NAMES:
        return None
    if len(k) == 1 and k in _CIRCLE_DIGITS:
        return _CIRCLE_DIGITS.index(k) + 1
    m = re.search(r"采分点\s*([①②③④⑤⑥⑦⑧⑨⑩]|\d+)", k, re.I)
    if m:
        token = m.group(1)
        if token.isdigit():
            return int(token)
        if token in _CIRCLE_DIGITS:
            return _CIRCLE_DIGITS.index(token) + 1
    if re.fullmatch(r"\d+", k):
        return int(k)
    m = re.search(r"(?:point|p|项|题)\s*_?\s*(\d+)", k, re.I)
    if m:
        return int(m.group(1))
    return None


def sorted_scoring_point_columns(gt_columns: List[str]) -> List[str]:
    pts = [c for c in gt_columns if _POINT_COL_RE.search(str(c))]
    return sorted(pts, key=lambda c: scoring_point_index_from_column(c) or 999)


def column_for_scoring_index(gt_columns: List[str], index: int) -> Optional[str]:
    for col in sorted_scoring_point_columns(gt_columns):
        if scoring_point_index_from_column(col) == index:
            return col
    pts = sorted_scoring_point_columns(gt_columns)
    if 1 <= index <= len(pts):
        return pts[index - 1]
    return None


def _iter_score_dicts(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """展开 parsed 及常见嵌套容器为若干 dict，用于扫键名。"""
    if not isinstance(parsed, dict) or parsed.get("parse_error"):
        return []
    out: List[Dict[str, Any]] = [parsed]
    for name in ("scores", "fields", "labels", "predictions", "judgments", "annotations", "data", "result"):
        sub = parsed.get(name)
        if isinstance(sub, dict):
            out.append(sub)
    return out


def map_alias_keys_to_columns(
    parsed: Dict[str, Any],
    gt_columns: List[str],
    existing: Dict[str, Any],
) -> Dict[str, Any]:
    """将 ①/采分点① 等别名键映射到表头列名（采分点1…）。"""
    out = dict(existing)
    point_cols = sorted_scoring_point_columns(gt_columns)
    if not point_cols:
        return out

    keyed: List[Tuple[int, Any, str]] = []
    for container in _iter_score_dicts(parsed):
        for k, v in container.items():
            if v is None:
                continue
            idx = scoring_point_index_from_model_key(str(k))
            if idx is not None:
                keyed.append((idx, v, str(k)))

    keyed.sort(key=lambda x: x[0])
    seen_idx: set[int] = set()
    for idx, v, _ in keyed:
        if idx in seen_idx:
            continue
        seen_idx.add(idx)
        col = column_for_scoring_index(gt_columns, idx)
        if col and col not in out:
            out[col] = v

    return out


def infer_scoring_profile(columns: List[str], field_mapping: Dict[str, Any]) -> Dict[str, Any]:
    """评测配置（委托 evaluation.build_evaluation_profile，兼容旧字段名）。"""
    from evaluation import build_evaluation_profile

    return build_evaluation_profile(field_mapping, columns)


def validate_field_mapping(
    columns: List[str],
    field_mapping: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """校验 LLM 返回的列名是否存在于真实表头；修正幻觉列名并给出警告。"""
    colset = {str(c).strip() for c in columns if c}
    fm: Dict[str, Any] = dict(field_mapping or {})
    warnings: List[str] = []

    def _ok(name: Any) -> bool:
        return _fm_col_ok(name, colset)

    primary = fm.get("primary_content")
    if primary and not _ok(primary):
        warnings.append(f"模型建议的核心输入列「{primary}」不在你的文件中，请在下拉框中重选。")
        fm["primary_content"] = ""

    for role, label in (
        ("reference_column", "标答"),
        ("stem_column", "题干"),
        ("rubric_column", "参考/规则"),
    ):
        v = fm.get(role)
        if v and not _ok(v):
            warnings.append(f"模型建议的{label}列「{v}」不存在，已清空。")
            fm[role] = None

    ctx = fm.get("context") or []
    if isinstance(ctx, str):
        ctx = [ctx]
    kept_ctx = [c for c in ctx if _ok(c)]
    dropped = [c for c in ctx if c and c not in kept_ctx]
    if dropped:
        warnings.append(f"参考列 {dropped} 不存在，已忽略。")
    fm["context"] = kept_ctx

    raw_gts = fm.get("ground_truths") or []
    if isinstance(raw_gts, str):
        raw_gts = [raw_gts]
    kept_gts: List[str] = []
    for g in raw_gts:
        if _ok(g):
            kept_gts.append(str(g).strip())
        elif g:
            warnings.append(f"人工标注列「{g}」不在文件中，已忽略。")
    gt = fm.get("ground_truth")
    if gt and str(gt).lower() not in ("null", "none", "") and _ok(gt) and str(gt).strip() not in kept_gts:
        kept_gts.insert(0, str(gt).strip())
    elif gt and not _ok(gt) and str(gt).lower() not in ("null", "none", ""):
        warnings.append(
            f"模型建议的人工标注列「{gt}」不在文件中（当前仅有：{', '.join(sorted(colset)[:12])}…）。"
        )
    fm["ground_truths"] = kept_gts
    pg = fm.get("primary_ground_truth")
    if pg and not _ok(pg):
        warnings.append(f"主标注列「{pg}」无效，已改用第一个标注列。")
        fm["primary_ground_truth"] = kept_gts[0] if kept_gts else None
    em = str(fm.get("evaluation_mode") or "auto").strip().lower()
    from evaluation import EVALUATION_MODES, parse_custom_evaluation

    if em and em not in EVALUATION_MODES:
        warnings.append(f"未知评测方式「{em}」，已改回 auto。")
        fm["evaluation_mode"] = "auto"
        em = "auto"

    if em == "custom":
        ce = parse_custom_evaluation(fm)
        if ce.get("parse_error"):
            warnings.append("custom_evaluation 不是合法 JSON，已忽略自定义配置。")
            fm["evaluation_mode"] = "auto"
        else:
            plugin = (ce.get("plugin") or ce.get("preset") or "").strip()
            if not plugin:
                warnings.append("自定义模式须指定 custom_evaluation.plugin（插件 ID）。")
            else:
                from evaluation_plugins import get_plugin

                if not get_plugin(plugin):
                    warnings.append(
                        f"未找到评测插件「{plugin}」，请选内置插件或在 evaluation_plugins_user.py 注册。"
                    )

    raw_pkm = fm.get("prediction_key_mapping")
    if raw_pkm is None or raw_pkm == {}:
        fm.pop("prediction_key_mapping", None)
    elif not isinstance(raw_pkm, dict):
        warnings.append(
            "prediction_key_mapping 须为 JSON 对象（「模型输出键」→「表头标注列名」），已忽略。"
        )
        fm.pop("prediction_key_mapping", None)
    else:
        cleaned_pkm: Dict[str, str] = {}
        for mk, tk in raw_pkm.items():
            ms, ts = str(mk).strip(), str(tk).strip()
            if not ms or not ts:
                continue
            if not _ok(ts):
                warnings.append(
                    f"prediction_key_mapping 目标列「{ts}」不在文件中，已忽略键「{ms}」。"
                )
                continue
            cleaned_pkm[ms] = ts
        if cleaned_pkm:
            fm["prediction_key_mapping"] = cleaned_pkm
        else:
            fm.pop("prediction_key_mapping", None)

    fm = finalize_field_mapping(fm)
    return fm, warnings


def _flatten_prediction_kv_from_parsed(use_parsed: Dict[str, Any]) -> Dict[str, Any]:
    """收集模型 JSON 顶层及常见嵌套对象中的键值（后者覆盖前者）。"""
    flat: Dict[str, Any] = {}
    for container in _iter_score_dicts(use_parsed):
        if not isinstance(container, dict):
            continue
        for k, v in container.items():
            if k is None or str(k).startswith("_"):
                continue
            flat[str(k).strip()] = v
    return flat


def apply_user_prediction_key_mapping(
    use_parsed: Dict[str, Any],
    gt_columns: List[str],
    out: Dict[str, Any],
    mapping: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    用户/澄清阶段配置的「模型 JSON 键 → 表头标注列名」映射。
    仅当目标列尚无预测值时写入，避免覆盖已对齐的键。
    """
    if not mapping or not isinstance(mapping, dict):
        return out
    merged = dict(out)
    gt_set = {str(c).strip() for c in gt_columns if c and not str(c).startswith("_")}
    flat = _flatten_prediction_kv_from_parsed(use_parsed)
    for mk_raw, tk_raw in mapping.items():
        mk, tk = str(mk_raw).strip(), str(tk_raw).strip()
        if not mk or not tk or tk not in gt_set:
            continue
        if tk in merged and merged[tk] is not None:
            continue
        if mk not in flat or flat[mk] is None:
            continue
        merged[tk] = flat[mk]
    return merged


def ensure_suggested_prompt_primary_input_block(
    suggested_prompt: str,
    primary: Optional[str],
) -> str:
    """澄清产出的 Prompt 若未嵌入主要输入，在文首插入标准块（兼容 student_answer 别名）。"""
    if not primary or not str(primary).strip():
        return suggested_prompt or ""
    p = suggested_prompt or ""
    if prompt_references_primary_input(p, primary):
        return p
    block = (
        f"## 核心输入（须评判的核心内容，数据列「{primary}」）\n"
        "{{ primary_input }}\n"
        "{{ student_answer }}\n\n"
    )
    return block + p.lstrip()


# 兼容旧名
ensure_suggested_prompt_student_block = ensure_suggested_prompt_primary_input_block


def audit_clarified_output(
    parsed: Dict[str, Any],
    field_mapping: Dict[str, Any],
    *,
    columns: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    澄清结果质检：核心输入列与 suggested_prompt 是否把核心内容写入模板。
    必要时补全 primary_content / core_inputs、在 Prompt 文首插入核心输入块。
    """
    fm = dict(field_mapping or {})
    warnings: List[str] = []
    colset = {str(c).strip() for c in (columns or []) if c}

    cis = parsed.get("core_input_source")
    if isinstance(cis, dict):
        raw_cols = cis.get("columns") or cis.get("column") or []
        if isinstance(raw_cols, str):
            raw_cols = [raw_cols]
        core_cols = [str(c).strip() for c in raw_cols if _fm_col_ok(c, colset)]
        if core_cols:
            fm["core_inputs"] = core_cols
            if not fm.get("primary_content"):
                fm["primary_content"] = core_cols[0]

    sas = parsed.get("student_answer_source")
    if isinstance(sas, dict):
        col = (sas.get("column") or "").strip()
        if col and _fm_col_ok(col, colset) and not fm.get("primary_content"):
            fm["primary_content"] = col
        form = (sas.get("form") or "").strip()
        if col and not form:
            warnings.append("student_answer_source 建议填写 form（text / image_url / excel_embedded）。")

    primary = (fm.get("primary_content") or "").strip()
    if not primary:
        hints = suggest_core_input_columns(list(colset))
        if hints:
            fm["primary_content"] = hints[0]
            fm["core_inputs"] = hints[:4] if len(hints) > 1 else [hints[0]]
            primary = hints[0]
            warnings.append(
                f"模型未指定核心输入列，已暂选「{primary}」（请在「核心输入」下拉框确认）。"
            )
        else:
            warnings.append(
                "未识别核心输入列（primary_content）。请在映射区指定本任务须交给模型判断的列。"
            )

    suggested = parsed.get("suggested_prompt") or ""
    if primary and not prompt_references_student_content(suggested, primary):
        warnings.append(
            f"建议的 Prompt 未把核心输入写入模板（列「{primary}」）。"
            "已在文首自动插入 ## 核心输入 + {{ primary_input }}，请确认。"
        )
        suggested = ensure_suggested_prompt_primary_input_block(suggested, primary)
        parsed["suggested_prompt"] = suggested

    if primary and not (parsed.get("understood_task") or "").strip():
        warnings.append("understood_task 为空，应说明核心输入在哪一列。")
    elif primary and primary not in (parsed.get("understood_task") or ""):
        if not (isinstance(sas, dict) and (sas.get("column") or "") == primary):
            warnings.append(
                f"理解摘要未提及核心输入列「{primary}」，请核对 core_input_source / 列映射。"
            )

    fm = finalize_field_mapping(fm)
    return fm, warnings


def list_ground_truth_columns(field_mapping: Dict[str, Any]) -> List[str]:
    fm = finalize_field_mapping(dict(field_mapping or {}))
    return list(fm.get("ground_truths") or [])


def list_core_input_columns(field_mapping: Dict[str, Any]) -> List[str]:
    """任务须交给模型判断的核心列（可为多列，如原题信息+推题信息）。"""
    fm = finalize_field_mapping(dict(field_mapping or {}))
    core = list(fm.get("core_inputs") or [])
    primary = (fm.get("primary_content") or "").strip()
    if primary and primary not in core:
        core.insert(0, primary)
    return [c for c in core if c]


_PLACEHOLDER_RE = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")


def build_template_vars(
    row: Dict[str, str],
    field_mapping: Dict[str, Any],
    extra_vars: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """构建 {{ 占位 }} 变量表；含常用别名（student_answer、rubric_content 等）。"""
    fm = finalize_field_mapping(dict(field_mapping or {}))
    primary = fm.get("primary_content") or fm.get("primary")
    context_cols = fm.get("context") or []
    if isinstance(context_cols, str):
        context_cols = [context_cols]

    vars_dict: Dict[str, Any] = {}
    if primary:
        val = row.get(primary, "") if primary in row else ""
        vars_dict["primary_input"] = val
        vars_dict["主要输入"] = val
        vars_dict["student_answer"] = val
        vars_dict["学生作答"] = val
        vars_dict[primary] = val

    ref_col = fm.get("reference_column")
    if ref_col and ref_col in row:
        vars_dict["reference_answer"] = row.get(ref_col, "")
        vars_dict["correct_answer"] = row.get(ref_col, "")
    elif context_cols:
        vars_dict["correct_answer"] = row.get(context_cols[0], "")

    stem_col = fm.get("stem_column")
    if stem_col and stem_col in row:
        vars_dict["stem_content"] = row.get(stem_col, "")
        vars_dict["题干"] = vars_dict["stem_content"]

    rub_col = fm.get("rubric_column")
    if rub_col and rub_col in row:
        vars_dict["rubric_content"] = row.get(rub_col, "")

    for c in context_cols:
        if c in row:
            vars_dict[c] = row[c]
    for k, v in row.items():
        vars_dict.setdefault(k, v)

    if extra_vars:
        for k, v in extra_vars.items():
            if v is None:
                continue
            if k not in vars_dict or not str(vars_dict.get(k) or "").strip():
                vars_dict[k] = v
    return vars_dict


def list_unreplaced_placeholders(prompt: str) -> List[str]:
    """渲染后仍残留的 {{ 占位符 }} 名称列表。"""
    if not prompt:
        return []
    return [m.group(1).strip() for m in _PLACEHOLDER_RE.finditer(prompt)]


def prompt_references_primary_input(prompt: str, primary: Optional[str]) -> bool:
    """是否已把 primary_content 作为「主要输入」写入 Prompt（仅在指令里提到列名不算）。"""
    if not primary:
        return True
    p = prompt or ""
    for key in ("primary_input", "student_answer", "主要输入", "核心输入", "学生作答", primary):
        if re.search(r"\{\{\s*" + re.escape(key) + r"\s*\}\}", p):
            return True
    if re.search(r"##\s*(主要输入|核心输入|学生作答)\b", p):
        return True
    if re.search(r"(主要输入|核心输入|学生作答)[：:]\s*\S", p) and primary in p:
        return True
    if primary in p:
        m = re.search(r"##\s*(主要输入|核心输入|学生作答)[^\n]*\n([\s\S]{20,})", p)
        if m and m.group(2).strip():
            return True
    return False


prompt_references_student_content = prompt_references_primary_input


def _prepend_primary_input_body(
    prompt: str,
    primary: str,
    val: str,
    *,
    vision_has_images: bool,
) -> str:
    """跑批渲染后：在文首插入主要输入（文本直接写入；图片题写说明+URL 参考）。"""
    if vision_has_images:
        url_line = f"\n（同列文本/URL 参考：{val[:800]}）" if val else ""
        block = (
            f"## 主要输入（列「{primary}」）\n"
            f"本题主要输入为**图片**，已随本条消息附图，请结合图中内容判断。{url_line}\n\n"
        )
    else:
        body = val[:6000] if val else "（本行该列为空，请结合其余字段判断）"
        block = f"## 主要输入（列「{primary}」）\n{body}\n\n"
    return block + (prompt or "").lstrip()


def _auto_prompt_context_columns(
    row: Dict[str, str],
    field_mapping: Dict[str, Any],
) -> List[str]:
    """补足成对/参考类任务的必要上下文列，避免只把主列塞进 Prompt。"""
    fm = finalize_field_mapping(dict(field_mapping or {}))
    primary = (fm.get("primary_content") or "").strip()
    gt_cols = set(list_ground_truth_columns(fm))
    blocked_names = gt_cols | {primary, "platform_ground_truth", "platform_primary", "platform_reference"}
    out: List[str] = []

    def add(col: Any) -> None:
        cs = str(col or "").strip()
        if not cs or cs in blocked_names or cs not in row:
            return
        if normalize_value(row.get(cs)) is None:
            return
        low = cs.lower()
        if low in ("id", "top") or low.endswith("id") or "标注" in cs or "结果" in cs:
            return
        if cs not in out:
            out.append(cs)

    for c in fm.get("context") or []:
        add(c)

    # 相似题/对比题常见为「原题信息」+「推题信息」；主输入只选一列时要自动补另一列。
    pair_keywords = ("推题", "推荐题", "候选题", "相似题", "对比题", "原题")
    if any(k in primary for k in ("原题", "主题")):
        for c in row.keys():
            if any(k in str(c) for k in pair_keywords) and str(c) != primary:
                add(c)
    elif any(k in primary for k in ("推题", "推荐题", "候选题", "相似题", "对比题")):
        for c in row.keys():
            if "原题" in str(c) and str(c) != primary:
                add(c)

    return out[:4]


def ensure_prompt_essentials(
    prompt: str,
    row: Dict[str, str],
    field_mapping: Dict[str, Any],
    *,
    vision_has_images: bool = False,
    extra_vars: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[str]]:
    """
    跑批前补全 Prompt 必备要素（主要输入 = primary_content，可为文字/URL/内嵌图）。
    返回 (新 prompt, 自动修补说明列表)。
    """
    fm = finalize_field_mapping(dict(field_mapping or {}))
    primary = fm.get("primary_content") or ""
    core_cols = list_core_input_columns(fm)
    notes: List[str] = []
    out = (prompt or "").rstrip()
    if not primary and not core_cols:
        return out, notes

    if len(core_cols) > 1:
        missing_blocks: List[str] = []
        for col in core_cols:
            raw = str(row.get(col) or "").strip()
            if not raw:
                continue
            if raw in out or f"{{{{{col}}}}}" in (prompt or ""):
                continue
            missing_blocks.append(f"### {col}\n{raw[:4000]}")
        if missing_blocks:
            out = (
                "## 核心输入\n"
                + "\n\n".join(missing_blocks)
                + "\n\n"
                + out.lstrip()
            )
            notes.append("已自动补充核心输入列：" + "、".join(core_cols))

    if not primary:
        return out, notes

    val = str(row.get(primary) or "").strip()
    unreplaced = list_unreplaced_placeholders(out)
    if unreplaced:
        notes.append("未替换占位符：" + "、".join(unreplaced[:8]))

    if not prompt_references_primary_input(out, primary):
        out = _prepend_primary_input_body(out, primary, val, vision_has_images=vision_has_images)
        kind = "图片+说明" if vision_has_images else "正文"
        notes.append(f"已在 Prompt 文首自动补充「主要输入」{kind}（模板未引用该列）")
    elif not vision_has_images and val and val not in out and len(val) < 8000:
        templ = prompt or ""
        if not any(x in templ for x in ("{{ primary_input }}", "{{ student_answer }}", "{{" + primary + "}}")):
            out = _prepend_primary_input_body(out, primary, val, vision_has_images=False)
            notes.append("已在文首追加主要输入正文（占位符未展开）")

    context_blocks: List[str] = []
    for col in _auto_prompt_context_columns(row, fm):
        raw = str(row.get(col) or "").strip()
        if not raw:
            continue
        if raw in out or f"{{{{{col}}}}}" in (prompt or "") or f"{{{{ {col} }}}}" in (prompt or ""):
            continue
        context_blocks.append(f"### {col}\n{raw[:4000]}")
    if context_blocks:
        out = (
            "## 必要对比/参考输入\n"
            + "\n\n".join(context_blocks)
            + "\n\n"
            + out.lstrip()
        )
        notes.append("已自动补充对比/参考输入列：" + "、".join(b.split("\n", 1)[0].replace("### ", "") for b in context_blocks))

    rub_global = (extra_vars or {}).get("rubric_content")
    if rub_global and str(rub_global).strip() and str(rub_global).strip() not in out:
        if "rubric_content" in (prompt or "") or "评分标准" in out:
            out += f"\n\n## 评分标准（任务级文件）\n{str(rub_global).strip()[:12000]}"
            notes.append("已补充上传的评分标准全文")

    return out, notes


def render_row_prompt(
    prompt_template: str,
    row: Dict[str, str],
    field_mapping: Dict[str, Any],
    extra_vars: Optional[Dict[str, Any]] = None,
) -> str:
    """按 {{ 变量名 }} 替换行数据；不用 Jinja 解析整段 Prompt，避免列名含（）等字符报错。"""
    if not prompt_template:
        return ""
    vars_dict = build_template_vars(row, field_mapping, extra_vars=extra_vars)

    def _repl(m: re.Match[str]) -> str:
        key = m.group(1).strip()
        if key in vars_dict:
            val = vars_dict[key]
            return "" if val is None else str(val)
        return m.group(0)

    return _PLACEHOLDER_RE.sub(_repl, prompt_template)


USER_SUPPLEMENT_HEADER = "## 用户补充说明（确认时填写）"


def strip_user_supplement_block(prompt: str) -> str:
    """移除 Prompt 末尾由 apply_user_answers_to_prompt 写入的补充块，避免重复确认时叠加。"""
    if not prompt:
        return ""
    idx = prompt.find(USER_SUPPLEMENT_HEADER)
    if idx < 0:
        return prompt.rstrip()
    return prompt[:idx].rstrip()


def _answer_for_index(answers: Dict[str, Any], index: int) -> str:
    raw = answers.get(str(index), answers.get(index, ""))
    return (raw if raw is not None else "").strip()


def build_user_supplement_block(questions: List[str], answers: Dict[str, Any]) -> str:
    """将澄清问题与用户回答格式化为 Prompt 附录；无有效回答时返回空字符串。"""
    if not questions:
        return ""
    lines = [USER_SUPPLEMENT_HEADER, "以下回答须在逐行判断时遵守：", ""]
    n = 0
    for i, q in enumerate(questions):
        ans = _answer_for_index(answers, i)
        if not ans:
            continue
        n += 1
        lines.append(f"{n}. 问：{q}")
        lines.append(f"   答：{ans}")
        lines.append("")
    if n == 0:
        return ""
    return "\n".join(lines).rstrip()


def apply_user_answers_to_prompt(
    prompt: str,
    questions: List[str],
    answers: Dict[str, Any],
) -> tuple[str, bool, int]:
    """
    把用户补充说明写入 Prompt 末尾（先去掉旧块再写入）。
    返回 (新 Prompt, 是否写入, 写入条数)。
    """
    base = strip_user_supplement_block(prompt or "")
    block = build_user_supplement_block(questions, answers)
    if not block:
        return base, False, 0
    merged = base + "\n\n" + block if base else block
    return merged, True, block.count("问：")


def validate_prompt_renders(
    prompt_template: str,
    *,
    columns: List[str],
    rows: List[List[str]],
    field_mapping: Dict[str, Any],
    sample_n: int = 3,
    extra_vars: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """试渲染前 N 行；返回错误/警告信息列表（空=通过）。"""
    errors: List[str] = []
    if not (prompt_template or "").strip():
        errors.append("Prompt 模板为空")
        return errors
    fm = finalize_field_mapping(field_mapping)
    primary = fm.get("primary_content") or ""
    if primary and not prompt_references_primary_input(prompt_template, primary):
        errors.append(
            f"Prompt 未引用主要输入列「{primary}」。"
            f"请加入 {{{{ primary_input }}}}（或 {{{{ student_answer }}}}）或「## 主要输入」小节。"
        )
    for i, rv in enumerate(rows[:sample_n]):
        try:
            rd = row_to_dict(columns, rv)
            out = render_row_prompt(prompt_template, rd, field_mapping, extra_vars=extra_vars)
            if not out.strip():
                errors.append(f"第 {i + 1} 行渲染结果为空")
            for ph in list_unreplaced_placeholders(out):
                errors.append(f"第 {i + 1} 行未替换占位符：{{{{ {ph} }}}}")
        except Exception as e:
            errors.append(f"第 {i + 1} 行渲染失败：{e}")
    return errors


def backup_and_prepare(
    source: Path,
    backup_dir: Path,
    prepared_dir: Path,
    field_mapping: Dict[str, Any],
    *,
    columns: List[str],
    rows: List[List[str]],
) -> Dict[str, str]:
    """备份原文件，写出带 platform_* 列的 prepared 文件（CSV）。"""
    backup_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{source.stem}_{ts}{source.suffix}"
    shutil.copy2(source, backup_path)

    data = {"columns": columns, "rows": rows}
    columns = list(data["columns"])
    rows = data["rows"]

    primary = field_mapping.get("primary_content") or ""
    context_cols = field_mapping.get("context") or []
    if isinstance(context_cols, str):
        context_cols = [context_cols]
    fm = finalize_field_mapping(field_mapping)
    gt_cols = list_ground_truth_columns(fm)
    gt_col = fm.get("primary_ground_truth")

    extra_cols = ["platform_primary", "platform_reference", "platform_ground_truth"]
    out_cols = columns + [c for c in extra_cols if c not in columns]

    prepared_path = prepared_dir / f"{source.stem}_prepared.csv"
    with prepared_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(out_cols)
        for rv in rows:
            row = row_to_dict(columns, rv)
            ref_parts = [str(row.get(c, "") or "") for c in context_cols]
            row["platform_primary"] = row.get(primary, "") if primary else ""
            row["platform_reference"] = " | ".join(p for p in ref_parts if p)
            if gt_cols:
                blob = {c: row.get(c, "") for c in gt_cols}
                row["platform_ground_truth"] = json.dumps(blob, ensure_ascii=False)
            else:
                row["platform_ground_truth"] = row.get(gt_col, "") if gt_col else ""
            w.writerow([row.get(c, "") for c in out_cols])

    return {
        "backup_path": str(backup_path),
        "prepared_path": str(prepared_path),
        "prepared_filename": prepared_path.name,
        "row_count": str(len(rows)),
    }


# ── 评测结果评估（与人工标注 / ground_truth 比对）────────────────

_PREDICTION_KEYS = (
    "judge",
    "prediction",
    "label",
    "verdict",
    "result",
    "answer",
    "score",
    "is_correct",
    "is_error",
    "correct",
    "ai_judge",
    "grade",
    "verdict_label",
    "识别结果",
    "判定",
    "error",
    "wrong",
)


def extract_prediction(parsed: Dict[str, Any], raw_text: str = "") -> Any:
    """从模型 JSON 或原文中抽取可与标注列对比的预测值。"""
    if not isinstance(parsed, dict):
        parsed = {}
    use_parsed = parsed if not parsed.get("parse_error") else {}

    for key in _PREDICTION_KEYS:
        if key in use_parsed and use_parsed[key] is not None and str(use_parsed[key]).strip() != "":
            return use_parsed[key]

    for key in ("judges", "labels", "results", "points"):
        val = use_parsed.get(key)
        if isinstance(val, list) and val:
            return val[0] if len(val) == 1 else val

    data = use_parsed.get("data")
    if isinstance(data, dict):
        inner = extract_prediction(data, raw_text)
        if inner is not None:
            return inner

    text = (raw_text or "").strip()
    if text:
        for pat in (
            r'["\']?judge["\']?\s*[:=]\s*["\']([abcABC])["\']',
            r'["\']?prediction["\']?\s*[:=]\s*["\']([^"\']+)["\']',
            r'["\']?score["\']?\s*[:=]\s*([0-9.]+)',
            r'list\s*\[\s*["\']([^"\']+)["\']\s*\]',
            r'["\']?(?:是否可用|可用性|label|result)["\']?\s*[:=]\s*["\'](可用|勉强可用|不可用)["\']',
            r'["\']?(?:is_correct|is_error|识别结果)["\']?\s*[:=]\s*(true|false|0|1)',
            r'["\']?(?:is_correct|is_error|识别结果)["\']?\s*[:=]\s*["\']([^"\']+)["\']',
        ):
            m = re.search(pat, text, re.I)
            if m:
                return m.group(1)
        label_hits = re.findall(r"(?<!勉强)(不可用|勉强可用|可用)", text)
        if label_hits and len(text) < 200:
            return label_hits[-1]
        if len(text) < 200 and re.search(
            r"(is_correct|is_error|识别结果|judge)\s*[:=]", text, re.I
        ):
            bare = re.search(r"\b([01])\b", text)
            if bare:
                return bare.group(1)
    return None


def extract_all_predictions(
    parsed: Dict[str, Any],
    raw_text: str,
    gt_columns: List[str],
    field_mapping: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """从模型 JSON 中抽取各标注列对应的预测（多采分点任务）。"""
    use_parsed = parsed if isinstance(parsed, dict) and not parsed.get("parse_error") else {}
    fm = finalize_field_mapping(dict(field_mapping or {}))
    pkm = fm.get("prediction_key_mapping")
    if not isinstance(pkm, dict):
        pkm = {}
    out: Dict[str, Any] = {}

    for col in gt_columns:
        for key in _column_name_variants(col):
            if key in use_parsed and use_parsed[key] is not None:
                out[col] = unwrap_score_value(use_parsed[key])
                break

    for container in ("scores", "fields", "labels", "predictions", "judgments", "annotations", "points_detail"):
        sub = use_parsed.get(container)
        if isinstance(sub, dict):
            for col in gt_columns:
                if col in out:
                    continue
                for key in _column_name_variants(col):
                    if key in sub and sub[key] is not None:
                        out[col] = unwrap_score_value(sub[key])
                        break

    points = use_parsed.get("points") or use_parsed.get("采分点")
    if isinstance(points, list):
        for i, item in enumerate(points):
            col_guess = f"采分点{i + 1}"
            if col_guess in gt_columns and col_guess not in out:
                if isinstance(item, dict):
                    out[col_guess] = (
                        item.get("score")
                        or item.get("point_score")
                        or item.get("value")
                        or item.get("points")
                    )
                elif item is not None:
                    out[col_guess] = item
            elif i < len(gt_columns) and gt_columns[i] not in out:
                if isinstance(item, dict):
                    out[gt_columns[i]] = item.get("score") or item.get("value")
                else:
                    out[gt_columns[i]] = item

    if "总分" in gt_columns and "总分" not in out:
        for key in ("总分", "total_score", "total", "sum"):
            if key in use_parsed and use_parsed[key] is not None:
                out["总分"] = use_parsed[key]
                break
        if "总分" not in out:
            for container in _iter_score_dicts(use_parsed):
                for key in ("总分", "total_score", "total", "sum", "totalScore"):
                    if key in container and container[key] is not None:
                        out["总分"] = container[key]
                        break

    out = apply_ai_token_prediction_aliases(use_parsed, gt_columns, out)
    out = map_alias_keys_to_columns(use_parsed, gt_columns, out)
    out = apply_user_prediction_key_mapping(use_parsed, gt_columns, out, pkm)
    return {k: collapse_prediction_value(v) for k, v in out.items()}


_SCORE_DICT_KEYS = (
    "score",
    "point_score",
    "points",
    "value",
    "result",
    "label",
    "verdict",
    "判定",
    "得分",
)


def unwrap_score_value(val: Any) -> Any:
    """模型常按采分点返回 {\"score\": 2, \"reason\": \"...\"}，比对前抽出可比较的标量。"""
    if val is None:
        return None
    if isinstance(val, dict):
        for key in _SCORE_DICT_KEYS:
            if key in val and val[key] is not None:
                inner = unwrap_score_value(val[key])
                if inner is not None:
                    return inner
        return val
    if isinstance(val, list) and len(val) == 1:
        return unwrap_score_value(val[0])
    return val


def collapse_prediction_value(val: Any) -> Any:
    """展示/投票前把单元素列表压成标量，并丢弃纯选项占位符。"""
    val = unwrap_score_value(val)
    if isinstance(val, list):
        if len(val) == 1:
            return collapse_prediction_value(val[0])
        return [collapse_prediction_value(v) for v in val]
    if isinstance(val, str):
        s = val.strip()
        if _looks_like_label_option_placeholder(s):
            return None
    return val


def _looks_like_label_option_placeholder(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return False
    labels = ("可用", "勉强可用", "不可用")
    return (
        all(label in compact for label in labels)
        and any(sep in compact for sep in ("/", "、", "|", "或"))
        and len(compact) <= 30
    )


def apply_ai_token_prediction_aliases(
    use_parsed: Dict[str, Any],
    gt_columns: List[str],
    out: Dict[str, Any],
) -> Dict[str, Any]:
    """
    将模型键名中的 _ai_ 与表头对齐：如 biezi_ai_1 / cuozi_ai_2 → biezi_1 / cuozi_2。
    仅在表头存在对应列且尚未写入 out 时生效。
    """
    merged = dict(out)
    gt_set = {str(c).strip() for c in gt_columns if c and not str(c).startswith("_")}
    for container in _iter_score_dicts(use_parsed):
        if not isinstance(container, dict):
            continue
        for k, v in container.items():
            if v is None or k is None:
                continue
            sk = str(k).strip()
            if sk in merged:
                continue
            candidate = re.sub(r"_ai_", "_", sk, flags=re.I)
            if candidate != sk and candidate in gt_set and candidate not in merged:
                merged[candidate] = v
    return merged


def _sum_numeric(values: List[Any]) -> Optional[float]:
    total = 0.0
    n = 0
    for v in values:
        nv = normalize_value(v)
        if nv is None:
            continue
        try:
            total += float(nv)
            n += 1
        except ValueError:
            continue
    return total if n else None


def format_per_column_match_line(
    by_col: Dict[str, Any],
    ground_truth_columns: List[str],
    *,
    max_cols: int = 16,
) -> str:
    """逐列比对摘要（规则生成，供结果表「分列判断」列）：biezi_1✓ 或 judge_gt✗(4/6字)"""
    from comparison_spec import COMPARISON_UNIT_SEQUENCE_CHAR, format_sequence_match_line

    seq_cols = [
        c
        for c in ground_truth_columns[:max_cols]
        if not str(c).startswith("_")
        and (by_col.get(c) or {}).get("comparison_unit") == COMPARISON_UNIT_SEQUENCE_CHAR
    ]
    if seq_cols:
        primary = seq_cols[0]
        line = format_sequence_match_line(by_col, primary)
        if line:
            return line

    parts: List[str] = []
    for c in ground_truth_columns[:max_cols]:
        if str(c).startswith("_"):
            continue
        d = by_col.get(c) or {}
        if normalize_value(d.get("ground_truth")) is None:
            continue
        sym = "?"
        if d.get("correct") is True:
            sym = "✓"
        elif d.get("correct") is False:
            sym = "✗"
        elif d.get("predicted") is None:
            sym = "?"
        md = d.get("match_detail")
        parts.append(f"{c}{sym}" + (f"({md})" if md and sym == "✗" else ""))
    return " ".join(parts) if parts else ""


def format_row_prediction_summary(by_col: Dict[str, Dict[str, Any]], profile: Dict[str, Any]) -> str:
    """结果页展示用：多列预测摘要。"""
    point_cols = profile.get("scoring_point_columns") or []
    parts: List[str] = []
    for c in point_cols:
        d = by_col.get(c) or {}
        p = d.get("predicted")
        if p is not None:
            parts.append(f"{c}={p}")
    for tc in profile.get("scoring_total_columns") or []:
        d = by_col.get(tc) or {}
        p = d.get("predicted")
        if p is not None:
            parts.append(f"{tc}={p}")
    if not parts:
        for c, d in by_col.items():
            if str(c).startswith("_"):
                continue
            p = (d or {}).get("predicted")
            if p is not None:
                parts.append(f"{c}={p}")
    return " | ".join(parts) if parts else ""


def format_row_ground_truth_summary(
    row: Dict[str, str],
    field_mapping: Dict[str, Any],
    columns: Optional[List[str]] = None,
) -> str:
    gts = resolve_row_ground_truths(row, field_mapping, columns)
    parts = [f"{k}={v}" for k, v in gts.items() if normalize_value(v) is not None]
    return " | ".join(parts)


def normalize_value(val: Any) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, dict):
        val = unwrap_score_value(val)
        if isinstance(val, dict):
            return None
    if isinstance(val, bool):
        return "1" if val else "0"
    if isinstance(val, (int, float)):
        if val == int(val):
            return str(int(val))
        return str(round(float(val), 6))
    if isinstance(val, list):
        parts = [normalize_value(x) for x in val]
        if any(p is None for p in parts):
            return None
        if len(parts) == 1:
            return parts[0]
        return "|".join(parts)
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "null", "n/a", ""):
        return None
    if _looks_like_label_option_placeholder(s):
        return None
    sl = s.lower()
    if sl in ("1", "true", "yes", "y", "正确", "对", "right", "correct", "是"):
        return "1"
    if sl in ("0", "false", "no", "n", "错误", "错", "wrong", "incorrect", "否"):
        return "0"
    if sl in ("a", "b", "c"):
        return sl
    try:
        f = float(sl.replace(",", ""))
        if f == int(f):
            return str(int(f))
        return str(round(f, 6))
    except ValueError:
        pass
    return sl


def resolve_row_ground_truth(
    row: Dict[str, str],
    field_mapping: Dict[str, Any],
    columns: Optional[List[str]] = None,
    *,
    gt_column: Optional[str] = None,
) -> Any:
    """按用户映射读取行内标注；可指定 gt_column（多标注列之一）。"""
    fm = finalize_field_mapping(dict(field_mapping or {}))
    fm_gt = gt_column or fm.get("primary_ground_truth") or fm.get("ground_truth")
    if fm_gt and str(fm_gt).lower() in ("null", "none", ""):
        fm_gt = None

    candidates: List[Any] = []
    if fm_gt:
        candidates.append(row.get(fm_gt))
    if not gt_column and "platform_ground_truth" in (columns or row.keys()):
        raw = row.get("platform_ground_truth")
        if raw and str(raw).strip().startswith("{"):
            try:
                blob = json.loads(raw)
                if isinstance(blob, dict) and fm_gt and fm_gt in blob:
                    candidates.append(blob.get(fm_gt))
            except json.JSONDecodeError:
                pass
        candidates.append(raw)

    for v in candidates:
        if normalize_value(v) is not None:
            return v
    return None


def resolve_row_ground_truths(
    row: Dict[str, str],
    field_mapping: Dict[str, Any],
    columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """读取该行所有已映射标注列的值。"""
    out: Dict[str, Any] = {}
    for col in list_ground_truth_columns(field_mapping):
        out[col] = resolve_row_ground_truth(row, field_mapping, columns, gt_column=col)
    return out


def _column_name_variants(column_name: str) -> List[str]:
    cn = str(column_name).strip()
    variants = [cn, cn.replace("（", "(").replace("）", ")")]
    safe = re.sub(r"[^\w\u4e00-\u9fff]+", "_", cn).strip("_")
    if safe and safe not in variants:
        variants.append(safe)
    return variants


def extract_prediction_for_column(
    parsed: Dict[str, Any],
    raw_text: str,
    column_name: str,
    *,
    fallback_single: bool = True,
) -> Any:
    """从模型 JSON 中抽取与某一标注列对应的预测值。"""
    if not column_name:
        return None
    use_parsed = parsed if isinstance(parsed, dict) and not parsed.get("parse_error") else {}
    for key in _column_name_variants(column_name):
        if key in use_parsed and use_parsed[key] is not None:
            return unwrap_score_value(use_parsed[key])
    for container in ("labels", "predictions", "judgments", "results", "scores", "fields", "annotations"):
        sub = use_parsed.get(container)
        if isinstance(sub, dict):
            for key in _column_name_variants(column_name):
                if key in sub and sub[key] is not None:
                    return unwrap_score_value(sub[key])
    if fallback_single:
        return extract_prediction(use_parsed, raw_text)
    return None


def compare_row_ground_truths(
    parsed: Dict[str, Any],
    raw_text: str,
    row: Dict[str, str],
    field_mapping: Dict[str, Any],
    columns: Optional[List[str]] = None,
    *,
    default_predicted: Any = None,
) -> Dict[str, Any]:
    """逐标注列比对；多采分点时额外计算「整行全采分点一致」与总分一致。"""
    fm = finalize_field_mapping(dict(field_mapping or {}))
    gt_cols = list_ground_truth_columns(fm)
    profile = infer_scoring_profile(columns or list(row.keys()), fm)
    primary = fm.get("primary_ground_truth")

    preds_map = extract_all_predictions(parsed, raw_text, gt_cols, fm)
    by_col: Dict[str, Dict[str, Any]] = {}

    col_units = profile.get("column_comparison_units") or {}
    default_unit = profile.get("comparison_unit") or "scalar"

    for col in gt_cols:
        gt_val = resolve_row_ground_truth(row, fm, columns, gt_column=col)
        pred = preds_map.get(col)
        if pred is None:
            pred = extract_prediction_for_column(
                parsed, raw_text, col, fallback_single=True
            )
        if pred is None and default_predicted is not None and col == primary:
            pred = default_predicted
        if pred is None and len(gt_cols) == 1:
            pred = extract_prediction(parsed, raw_text)
        pred = collapse_prediction_value(pred)
        unit = col_units.get(col) or default_unit
        from comparison_spec import compare_column_values

        cmp = compare_column_values(pred, gt_val, unit)
        by_col[col] = {
            "ground_truth": cmp.get("ground_truth", gt_val),
            "predicted": cmp.get("predicted", pred),
            "correct": cmp.get("correct"),
            "strict_correct": cmp.get("strict_correct"),
            "micro_correct": cmp.get("micro_correct"),
            "micro_total": cmp.get("micro_total"),
            "micro_ratio": cmp.get("micro_ratio"),
            "match_detail": cmp.get("match_detail"),
            "comparison_unit": cmp.get("comparison_unit"),
        }

    from evaluation import apply_composite_judgment

    apply_composite_judgment(
        by_col,
        profile,
        preds_map=preds_map,
        sum_numeric_fn=_sum_numeric,
    )

    return by_col


def row_judgment_correct(by_col: Dict[str, Dict[str, Any]], field_mapping: Dict[str, Any]) -> Optional[bool]:
    """汇总行级判断正确率（由 evaluation_profile / evaluation_mode 决定规则）。"""
    from evaluation import row_level_correct

    profile = infer_scoring_profile([], field_mapping)
    return row_level_correct(by_col, profile)


def compare_prediction(predicted: Any, ground_truth: Any) -> Optional[bool]:
    gt_n = normalize_value(ground_truth)
    if gt_n is None:
        return None
    pred_n = normalize_value(predicted)
    if pred_n is None:
        return None
    return pred_n == gt_n


def metrics_for_responses(
    responses: List[Dict[str, Any]],
    *,
    ground_truth_column: Optional[str] = None,
    ground_truth_columns: Optional[List[str]] = None,
    field_mapping: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """按行汇总准确率等指标；支持多标注列（主列 + per_ground_truth）。"""
    profile = infer_scoring_profile([], field_mapping or {})
    total = len(responses)
    api_ok = sum(1 for r in responses if r.get("status") == "ok")
    with_gt = [
        r
        for r in responses
        if normalize_value(r.get("ground_truth")) is not None
    ]
    comparable = [r for r in with_gt if r.get("correct") is not None]
    correct = sum(1 for r in comparable if r.get("correct"))
    no_pred = len(with_gt) - len(comparable)

    per_gt_metrics: Dict[str, Dict[str, Any]] = {}
    gt_cols = ground_truth_columns or []
    for col in gt_cols:
        if str(col).startswith("_"):
            continue
        sub_rows = []
        for r in responses:
            detail = (r.get("ground_truth_by_column") or {}).get(col)
            if not detail:
                continue
            sub_rows.append({
                "ground_truth": detail.get("ground_truth"),
                "correct": detail.get("correct"),
                "status": r.get("status"),
            })
        if sub_rows:
            per_gt_metrics[col] = metrics_for_responses(sub_rows, ground_truth_column=col)

    per_class: Dict[str, Dict[str, Any]] = {}
    for r in comparable:
        cls = normalize_value(r.get("ground_truth")) or "?"
        bucket = per_class.setdefault(cls, {"count": 0, "correct": 0})
        bucket["count"] += 1
        if r.get("correct"):
            bucket["correct"] += 1
    for cls, b in per_class.items():
        b["accuracy"] = round(b["correct"] / b["count"], 4) if b["count"] else 0.0

    accuracy = round(correct / len(comparable), 4) if comparable else None
    msg = None
    gt_label = ground_truth_column or ""
    if not with_gt:
        if gt_label:
            msg = (
                f"已映射人工标注列「{gt_label}」，但数据行中未读到有效标注值。"
                "请确认该列在 Excel/CSV 中有内容（非空），并重新点击「确认并准备数据」。"
            )
        else:
            msg = (
                "未配置人工标注列（ground_truth），无法计算判断正确率。"
                "请在第二步「列映射」中选择人工标注列。"
            )
    elif with_gt and not comparable:
        rule = profile.get("row_correct_rule") or "primary"
        if rule != "primary":
            pts = profile.get("scoring_point_columns") or profile.get("ground_truth_columns") or []
            pts_s = "、".join(pts[:10])
            msg = (
                f"已读取 {len(with_gt)} 行人工标注（评测：{profile.get('evaluation_mode_label', rule)}），"
                "但未能从模型输出解析出可比对字段。"
                f"请在 Prompt 中要求 JSON 键名与列名一致（如 {pts_s}）。"
            )
        else:
            msg = (
                f"已读取 {len(with_gt)} 行人工标注（列「{gt_label}」），"
                "但未能从模型输出中解析出可比对字段。"
                "请在 Prompt 中要求 JSON 输出，包含 is_correct、prediction、judge 或 score 等字段（0/1 或 true/false）。"
            )
    elif comparable and no_pred:
        msg = f"有 {no_pred} 行有标注但未能从模型输出中解析出可比对字段（如 judge、score、prediction）。"

    result = {
        "rows_total": total,
        "rows_api_ok": api_ok,
        "rows_api_error": total - api_ok,
        "rows_with_ground_truth": len(with_gt),
        "rows_evaluated": len(comparable),
        "rows_correct": correct,
        "rows_wrong": len(comparable) - correct,
        "rows_no_prediction": no_pred,
        "accuracy": accuracy,
        "accuracy_pct": round(accuracy * 100, 2) if accuracy is not None else None,
        "per_class": per_class,
        "ground_truth_column": gt_label or None,
        "ground_truth_columns": gt_cols or None,
        "message": msg,
    }
    if per_gt_metrics:
        result["per_ground_truth"] = per_gt_metrics
    result["evaluation_mode"] = profile.get("evaluation_mode")
    result["evaluation_mode_label"] = profile.get("evaluation_mode_label")
    result["row_correct_rule"] = profile.get("row_correct_rule")
    if profile.get("judgment_metric_note"):
        result["judgment_metric_note"] = profile.get("judgment_metric_note")
    if profile.get("comparison_unit"):
        result["comparison_unit"] = profile.get("comparison_unit")
    if profile.get("row_headline"):
        result["row_headline"] = profile.get("row_headline")
    if profile.get("is_multi_point") or profile.get("is_multi_column"):
        result["is_multi_point"] = profile.get("is_multi_point")
        result["is_multi_column"] = profile.get("is_multi_column")
        result["scoring_point_columns"] = profile.get("scoring_point_columns")
    from evaluation import apply_headline_to_metrics

    return apply_headline_to_metrics(result, responses, profile)


def compute_column_stats(columns: List[str], rows: List[List[str]]) -> Dict[str, int]:
    """每列非空行数，供前端下拉标注「N 行有值」。"""
    stats: Dict[str, int] = {c: 0 for c in columns}
    for rv in rows:
        rd = row_to_dict(columns, rv)
        for c in columns:
            if normalize_value(rd.get(c)) is not None:
                stats[c] += 1
    return stats


def count_ground_truth_rows(
    columns: List[str],
    rows: List[List[str]],
    field_mapping: Dict[str, Any],
) -> Dict[str, Any]:
    """准备数据前/后：统计标注列有效行数（含多列）。"""
    fm = finalize_field_mapping(dict(field_mapping or {}))
    gt_cols = list_ground_truth_columns(fm)
    primary = fm.get("primary_ground_truth")
    if not gt_cols:
        return {
            "ground_truth_column": None,
            "ground_truth_columns": [],
            "filled_rows": 0,
            "total_rows": len(rows),
            "per_column": {},
        }
    per_column: Dict[str, int] = {c: 0 for c in gt_cols}
    filled_primary = 0
    for rv in rows:
        rd = row_to_dict(columns, rv)
        for c in gt_cols:
            if normalize_value(resolve_row_ground_truth(rd, fm, columns, gt_column=c)) is not None:
                per_column[c] += 1
        if primary and normalize_value(resolve_row_ground_truth(rd, fm, columns, gt_column=primary)) is not None:
            filled_primary += 1
    return {
        "ground_truth_column": primary,
        "ground_truth_columns": gt_cols,
        "filled_rows": filled_primary,
        "total_rows": len(rows),
        "per_column": per_column,
    }


_FALLBACK_CLARIFY_TEMPLATE = """你是自动化评测的任务理解助手。

## 用户的任务描述
{{ task_description }}

## 数据文件
- 路径：{{ file_path }}
- 列名：{{ columns_list }}

## 数据示例（前 {{ sample_count }} 行）
{{ sample_rows_json }}

请只输出 JSON：
{
  "understood_task": "...",
  "task_type": "verification|grading|extraction|classification",
  "output_format": "judge_abc|score_numeric|score_4level|per_point|custom",
  "field_mapping": {
    "primary_content": "列名",
    "context": ["列名"],
    "ground_truth": "列名或 null"
  },
  "suggested_prompt": "完整 prompt，用 {{ student_answer }} 等占位",
  "clarifying_questions": ["问题1"],
  "confidence": 0.85
}
"""


# ── 跑批策略（第二步配置，默认关闭）────────────────────────────

DEFAULT_STRATEGIES: Dict[str, Any] = {
    "enabled": False,
    "prompt_engineering": False,
    "json_output": False,
    "one_shot": False,
    "few_shot": False,
    "chain_of_thought": False,
    "multi_model": False,
    "repeat_runs": False,
    "repeat_count": 2,
    "confidence_filter": False,
    "confidence_threshold": 0.8,
}


def normalize_strategies(raw: Any) -> Dict[str, Any]:
    out = dict(DEFAULT_STRATEGIES)
    if not isinstance(raw, dict):
        return out
    enabled = bool(raw.get("enabled"))
    out["enabled"] = enabled
    if not enabled:
        return out
    out["prompt_engineering"] = bool(raw.get("prompt_engineering"))
    out["json_output"] = bool(raw.get("json_output") or raw.get("prompt_engineering"))
    out["one_shot"] = bool(raw.get("one_shot"))
    out["few_shot"] = bool(raw.get("few_shot"))
    out["chain_of_thought"] = bool(raw.get("chain_of_thought"))
    out["multi_model"] = bool(raw.get("multi_model"))
    out["repeat_runs"] = bool(raw.get("repeat_runs"))
    try:
        out["repeat_count"] = max(2, min(5, int(raw.get("repeat_count") or 2)))
    except (TypeError, ValueError):
        out["repeat_count"] = 2
    out["confidence_filter"] = bool(raw.get("confidence_filter"))
    try:
        out["confidence_threshold"] = float(raw.get("confidence_threshold") or 0.8)
    except (TypeError, ValueError):
        out["confidence_threshold"] = 0.8
    return out


_IMAGE_URL_RE = re.compile(r"https?://\S+|\.(?:png|jpg|jpeg|webp|gif)\b", re.I)


def append_vision_limit_note(
    prompt: str,
    row: Dict[str, str],
    field_mapping: Dict[str, Any],
) -> str:
    """主要内容列为图片 URL 时提示：当前为纯文本调用。"""
    primary = (field_mapping or {}).get("primary_content") or ""
    if not primary:
        return prompt
    val = str(row.get(primary) or "").strip()
    if not val or not _IMAGE_URL_RE.search(val):
        return prompt
    note = (
        "\n\n【系统说明】本题「主要内容」列为图片 URL，当前评测为纯文本接口，模型无法直接查看图片。"
        "请根据题干、标答、评分标准等文本字段判断；须按 JSON 输出各采分点数值得分，勿仅输出 is_correct。"
    )
    if "纯文本接口" in (prompt or ""):
        return prompt
    return (prompt or "").rstrip() + note


def preview_prompt_engineering_append(
    field_mapping: Optional[Dict[str, Any]] = None,
    strategies: Optional[Dict[str, Any]] = None,
) -> str:
    """返回启用策略时将要追加到 Prompt 末尾的文案（供 UI 预览）。"""
    fm = finalize_field_mapping(field_mapping or {}) if field_mapping else {}
    profile = infer_scoring_profile([], fm) if fm else {}
    from evaluation import prompt_output_hint

    gt_cols = list_ground_truth_columns(fm) if fm else []
    s = normalize_strategies(strategies or {"enabled": True, "json_output": True})
    blocks: List[str] = []
    if s.get("json_output"):
        hint = prompt_output_hint(profile, gt_cols).strip()
        if hint:
            blocks.append(hint)
    if s.get("one_shot"):
        blocks.append(
            "【One-shot 示例】\n"
            "示例输入：学生作答与人工标注含义一致的样例。\n"
            "示例输出：请严格按本任务字段输出 JSON，字段值使用 0/1、分数或文本标签，"
            "不要输出额外说明。"
        )
    if s.get("few_shot"):
        blocks.append(
            "【Few-shot 对齐】\n"
            "请在判断前参考 2 类边界样例：\n"
            "1. 明显正确/一致：输出正例标签或满分。\n"
            "2. 明显错误/不一致：输出负例标签或 0 分。\n"
            "若样本介于两者之间，优先按评分标准和人工标注列的含义对齐。"
        )
    if s.get("chain_of_thought"):
        blocks.append(
            "【逐步核对】\n"
            "请先在内部逐步核对：读主要输入、对照题干/标准/规则、再判断输出字段。"
            "不要输出完整推理链；如需要解释，仅输出一句简短 reason。"
        )
    if s.get("confidence_filter"):
        blocks.append(
            "【置信度】\n"
            "请在 JSON 中增加 confidence 字段，取值 0 到 1；"
            "低把握样本给较低分，便于平台后续抛出复核。"
        )
    return "\n\n".join(blocks).strip()


def apply_prompt_engineering(
    prompt: str,
    strategies: Dict[str, Any],
    field_mapping: Optional[Dict[str, Any]] = None,
) -> str:
    s = normalize_strategies(strategies)
    if not s.get("enabled"):
        return prompt
    append = preview_prompt_engineering_append(field_mapping, s)
    if not append:
        return prompt
    if append in (prompt or ""):
        return prompt
    return (prompt or "").rstrip() + "\n\n" + append


def describe_active_strategies(strategies: Dict[str, Any]) -> List[str]:
    """人类可读的策略说明，供结果页/校验展示。"""
    s = normalize_strategies(strategies)
    if not s.get("enabled"):
        return ["高级策略：未启用"]
    lines = ["高级策略：已启用"]
    if s.get("json_output"):
        lines.append("· JSON 输出：追加固定 JSON 字段与格式约束")
    if s.get("one_shot"):
        lines.append("· One-shot：追加单样例输出约束")
    if s.get("few_shot"):
        lines.append("· Few-shot：追加正/负边界样例说明")
    if s.get("chain_of_thought"):
        lines.append("· 逐步核对：要求模型内部逐步核对，仅输出简短依据")
    if not any(s.get(k) for k in ("json_output", "one_shot", "few_shot", "chain_of_thought", "confidence_filter")):
        lines.append("· Prompt 策略：未勾选（仅启用总开关不会改变 Prompt）")
    if s.get("repeat_runs"):
        lines.append(f"· 重复评测：每行调用 {s.get('repeat_count', 2)} 次，多数表决")
    if s.get("confidence_filter"):
        lines.append(f"· 置信度过滤：低于 {s.get('confidence_threshold', 0.8)} 的行不参与判对")
    if s.get("multi_model"):
        lines.append("· 多模型：请在第 3 步多选模型（本身不修改 Prompt）")
    return lines


def select_sample_rows(
    rows: List[List[str]],
    *,
    sample_percent: Any = None,
    max_rows: Any = None,
) -> List[List[str]]:
    n = len(rows)
    if n == 0:
        return []
    if sample_percent is not None:
        try:
            pct = float(sample_percent)
        except (TypeError, ValueError):
            pct = 100.0
        if pct >= 100:
            return list(rows)
        count = max(1, int(math.ceil(n * pct / 100.0)))
        return list(rows[:count])
    if max_rows is not None:
        try:
            lim = int(max_rows)
            if lim > 0:
                return list(rows[: min(lim, n)])
        except (TypeError, ValueError):
            pass
    return list(rows)


_CONFIDENCE_KEYS = ("confidence", "certainty", "prob", "probability")


def extract_confidence(parsed: Dict[str, Any], raw_text: str = "") -> Optional[float]:
    if isinstance(parsed, dict):
        for key in _CONFIDENCE_KEYS:
            if key in parsed and parsed[key] is not None:
                try:
                    v = float(parsed[key])
                    if 0 <= v <= 1:
                        return v
                    if v > 1:
                        return min(1.0, v / 100.0)
                except (TypeError, ValueError):
                    continue
    text = raw_text or ""
    m = re.search(r'["\']?confidence["\']?\s*[:=]\s*([0-9.]+)', text, re.I)
    if m:
        try:
            v = float(m.group(1))
            return v if v <= 1 else min(1.0, v / 100.0)
        except ValueError:
            pass
    return None


def majority_prediction(values: List[Any]) -> Any:
    from collections import Counter

    normed = [normalize_value(v) for v in values if normalize_value(v) is not None]
    if not normed:
        return None
    counts = Counter(normed).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return None
    return counts[0][0]
