# -*- coding: utf-8 -*-
"""比对规格单测（python test_comparison_spec.py）"""
from comparison_spec import (
    compare_sequence_char,
    infer_comparison_unit,
    build_comparison_spec,
    compare_column_values,
    COMPARISON_UNIT_SEQUENCE_CHAR,
    COMPARISON_UNIT_SCALAR,
)


def test_sequence_strict_and_micro():
    r = compare_sequence_char("aaaaaa", "aaaaac")
    assert r["strict_correct"] is False
    assert r["micro_correct"] == 5
    assert r["micro_total"] == 6
    assert abs(r["micro_ratio"] - 5 / 6) < 0.001


def test_infer_judge_gt():
    fm = {"ground_truths": ["judge_gt"], "primary_ground_truth": "judge_gt"}
    rows = [{"judge_gt": "aaab"}, {"judge_gt": "caaaa"}]
    assert infer_comparison_unit(fm, sample_rows=rows) == COMPARISON_UNIT_SEQUENCE_CHAR


def test_infer_poetry_scalar():
    fm = {
        "ground_truths": ["识别结果（0=错误）"],
        "primary_ground_truth": "识别结果（0=错误）",
    }
    rows = [{"识别结果（0=错误）": "0"}, {"识别结果（0=错误）": "1"}]
    assert infer_comparison_unit(fm, sample_rows=rows) == COMPARISON_UNIT_SCALAR


def test_scalar_binary():
    r = compare_column_values("0", "1", COMPARISON_UNIT_SCALAR)
    assert r["correct"] is False


if __name__ == "__main__":
    test_sequence_strict_and_micro()
    test_infer_judge_gt()
    test_infer_poetry_scalar()
    test_scalar_binary()
    spec = build_comparison_spec(
        {"ground_truths": ["judge_gt"], "primary_ground_truth": "judge_gt"},
        sample_rows=[{"judge_gt": "aaab"}],
    )
    assert spec["comparison_unit"] == COMPARISON_UNIT_SEQUENCE_CHAR
    print("ok", spec)
