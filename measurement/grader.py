"""The grader of the instruments: the DAPO math grader of verl with strict boxed-answer matching.  A text is correct when the
boxed answer in its last 100 characters equals the reference string."""
import importlib.util
import os

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "verl", "utils", "reward_score", "math_dapo.py")
_SPEC = importlib.util.spec_from_file_location("math_dapo", _PATH)
_MATH_DAPO = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MATH_DAPO)


def is_correct(text, reference):
    try:
        return 1 if _MATH_DAPO.verify(text, reference, strict_box_verify=True)[0] else 0
    except Exception:
        return 0
