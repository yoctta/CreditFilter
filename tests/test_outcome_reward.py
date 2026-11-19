"""CPU unit tests of the overlong penalty, the group standardisation (creditfilter/outcome_reward.py) and the grader."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from creditfilter.outcome_reward import (  # noqa: E402
    ADVANTAGE_CLIP, RolloutRecord, group_standardized_advantages, overlong_penalty,
)


def test_overlong_penalty():
    assert overlong_penalty(10, 64, 16, 1.0) == 0.0
    assert overlong_penalty(48, 64, 16, 1.0) == 0.0
    assert abs(overlong_penalty(60, 64, 16, 1.0) - (-0.75)) < 1e-12
    assert abs(overlong_penalty(64, 64, 16, 1.0) - (-1.0)) < 1e-12
    assert abs(overlong_penalty(60, 64, 16, 0.5) - (-0.375)) < 1e-12
    assert overlong_penalty(60, 64, 0, 1.0) == 0.0 and overlong_penalty(60, 8, 16, 1.0) == 0.0
    print("overlong OK")


def test_group_standardisation():
    recs = [RolloutRecord("A", True, 1.0), RolloutRecord("A", False, -1.0), RolloutRecord("A", False, -1.0),
            RolloutRecord("B", True, 1.5), RolloutRecord("B", True, 1.0), RolloutRecord("B", False, -1.0),
            RolloutRecord("C", False, -1.0), RolloutRecord("C", False, -1.0),
            RolloutRecord("D", True, 0.25)]
    a = group_standardized_advantages(recs)
    assert abs(sum(a[:3])) < 1e-9 and abs(sum(a[3:6])) < 1e-9, "z-scores sum to 0 within a group"
    assert a[6] == 0.0 and a[7] == 0.0, "homogeneous group -> 0"
    assert a[8] == 0.0, "single-row group -> 0"
    assert max(abs(x) for x in a) <= ADVANTAGE_CLIP
    assert a[0] > 0 > a[1] and a[3] > a[4] > a[5]
    r = RolloutRecord.from_dict(RolloutRecord("A", True, 1.5).to_dict())
    assert r.prompt_uid == "A" and r.answer_correct and r.terminal_reward == 1.5
    print("standardisation OK", [round(x, 3) for x in a])


def test_grader():
    from creditfilter.custom_score import compute_score
    out = compute_score("math_dapo", "so the answer is \\boxed{\\frac{1}{2}}", "\\frac{1}{2}")
    assert out["score"] == 1.0 and out["acc"] is True
    out = compute_score("math_dapo", "so the answer is \\boxed{3}", "4")
    assert out["score"] == -1.0 and out["acc"] is False
    out = compute_score("math_dapo", "no boxed here", "4")
    assert out["score"] == -1.0
    print("grader OK")


if __name__ == "__main__":
    test_overlong_penalty()
    test_group_standardisation()
    test_grader()
    print("ALL LOGIC TESTS OK")
