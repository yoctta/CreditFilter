"""Training grader, passed to verl as its custom reward function.

The last \\boxed{...} of the response is compared with the reference answer after the normalisation of verl's math_dapo
utilities; data sources outside the math family, and answers on which that normalisation fails, are compared as
normalised strings."""

from __future__ import annotations

from typing import Any, Optional


def _normalize(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip().replace(" ", "").replace(",", "").replace("$", "")


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    **kwargs,
):
    """Returns {"score": +1 or -1, "acc": bool}."""
    ds = data_source or ""
    if any(m in ds for m in ("math", "MATH", "AIME", "AMC", "Olympiad", "numina", "Eurus")):
        try:
            from verl.utils.reward_score.math_dapo import (
                last_boxed_only_string, remove_boxed, normalize_final_answer,
            )
            boxed = last_boxed_only_string(solution_str)
            pred_ans = remove_boxed(boxed) if boxed else None
            pred_norm = normalize_final_answer(pred_ans) if pred_ans else ""
            gold_norm = normalize_final_answer(str(ground_truth))
            is_correct = bool(pred_norm) and pred_norm == gold_norm
            return {
                "score": 1.0 if is_correct else -1.0,
                "acc": is_correct,
            }
        except Exception:
            pass
    try:
        from verl.utils.reward_score.math_dapo import last_boxed_only_string
        pred = last_boxed_only_string(solution_str) or solution_str
        if pred.startswith("\\boxed{") and pred.endswith("}"):
            pred = pred[len("\\boxed{"):-1]
    except Exception:
        pred = solution_str
    is_correct = _normalize(pred) == _normalize(ground_truth)
    return {
        "score": 1.0 if is_correct else -1.0,
        "acc": is_correct,
    }
