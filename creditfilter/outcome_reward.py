"""Outcome reward and group-standardised advantages.

Per rollout (OutcomeRewardManager.run_single): the boxed answer is graded (custom_score.py: +1 correct, -1 wrong) and
the DAPO soft overlong penalty is added when the response runs into the last CF_OVERLONG_BUFFER_LEN tokens of the
response budget.  The reward is returned with an `outcome_record` (prompt group, correctness, reward).

Per batch (group_standardized_advantages, called from compute_advantage in verl/trainer/ppo/ray_trainer.py):
A_i = clip((r_i - mean_G r) / (std_G r + 1e-6), -5, 5) within each prompt group G, broadcast to the response tokens.
"""
import inspect
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

ADVANTAGE_CLIP = 5.0


def overlong_penalty(valid_len: int, max_len: int, buffer_len: int, factor: float) -> float:
    """DAPO soft overlong penalty: 0 while the response ends before max_len - buffer_len, then linear down to -factor
    at max_len."""
    if max_len > buffer_len > 0:
        expected_len = max_len - buffer_len
        exceed = int(valid_len) - expected_len
        return min(-exceed / buffer_len * factor, 0.0)
    return 0.0


@dataclass
class RolloutRecord:
    """Outcome of one rollout, passed from the reward worker to the advantage step as JSON."""

    prompt_uid: str
    answer_correct: bool
    terminal_reward: float

    def to_dict(self) -> dict:
        return {"prompt_uid": str(self.prompt_uid), "answer_correct": bool(self.answer_correct),
                "terminal_reward": float(self.terminal_reward)}

    @classmethod
    def from_dict(cls, d: dict) -> "RolloutRecord":
        return cls(prompt_uid=str(d["prompt_uid"]), answer_correct=bool(d["answer_correct"]),
                   terminal_reward=float(d["terminal_reward"]))


def group_standardized_advantages(records: List[RolloutRecord], clip: Optional[float] = ADVANTAGE_CLIP,
                                  eps: float = 1e-6) -> List[float]:
    """A_i = clip((r_i - mean) / (std + eps)) within each prompt group, in input order."""
    pos_by_uid: Dict[str, List[int]] = {}
    for i, r in enumerate(records):
        pos_by_uid.setdefault(r.prompt_uid, []).append(i)
    out = [0.0] * len(records)
    for idxs in pos_by_uid.values():
        raw = [float(records[i].terminal_reward) for i in idxs]
        mu = sum(raw) / len(raw)
        sd = (sum((x - mu) ** 2 for x in raw) / len(raw)) ** 0.5
        adv = [(x - mu) / (sd + eps) for x in raw]
        if clip is not None:
            adv = [max(-clip, min(clip, a)) for a in adv]
        for gi, i in enumerate(idxs):
            out[i] = float(adv[gi])
    return out


@register("outcome")
class OutcomeRewardManager(RewardManagerBase):
    """Reward manager: the reward of a rollout (grade + overlong penalty) and its outcome record."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None, **kwargs):
        super().__init__(config, tokenizer, compute_score)
        from verl.utils.reward_score import default_compute_score
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.overlong_enable = os.environ.get("CF_OVERLONG_BUFFER_ENABLE", "0").strip() == "1"
        self.overlong_buffer_len = int(os.environ.get("CF_OVERLONG_BUFFER_LEN", "4096").strip() or "4096")
        self.overlong_factor = float(os.environ.get("CF_OVERLONG_PENALTY_FACTOR", "1.0").strip() or "1.0")

    async def run_single(self, data: DataProto) -> dict:
        """Reward of one rollout."""
        assert len(data) == 1, "run_single expects DataProto of length 1"
        data_item = data[0]

        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]          # the response budget
        valid_response_length = int(data_item.batch["attention_mask"][-response_length:].sum())
        valid_response_ids = response_ids[:valid_response_length]

        nt = data_item.non_tensor_batch
        data_source = nt.get("data_source") or "math"
        gt = nt["reward_model"]["ground_truth"] if "reward_model" in nt else ""
        extra_info = nt.get("extra_info") or {}
        if not isinstance(extra_info, dict):
            extra_info = {}

        response_str = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True),
        )

        if self.is_async_reward_score:
            outcome_result = await self.compute_score(
                data_source=data_source, solution_str=response_str, ground_truth=gt, extra_info=extra_info,
            )
        else:
            outcome_result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source, solution_str=response_str, ground_truth=gt, extra_info=extra_info,
                ),
            )
        if isinstance(outcome_result, dict):
            outcome_score = float(outcome_result.get("score", 0.0))
            outcome_acc = bool(outcome_result.get("acc", outcome_score > 0))
        else:
            outcome_score = float(outcome_result)
            outcome_acc = outcome_score > 0

        terminal_reward = float(outcome_score)

        if self.overlong_enable:
            terminal_reward = float(terminal_reward) + float(overlong_penalty(
                int(valid_response_length), int(response_length), self.overlong_buffer_len, self.overlong_factor))

        # prompt group: the per-prompt uid of verl, else the dataset index
        uid = nt.get("uid")
        if isinstance(uid, (bytes, bytearray)):
            uid = uid.decode("utf-8", errors="replace")
        prompt_uid = (uid if isinstance(uid, str) else "") or str(extra_info.get("index", "")) or "_missing_"

        rec = RolloutRecord(prompt_uid=prompt_uid, answer_correct=bool(outcome_acc), terminal_reward=float(terminal_reward))
        # the record travels as a JSON string, which the metric aggregation of verl skips
        return {
            "reward_score": float(terminal_reward),
            "reward_extra_info": {
                "score": float(terminal_reward),
                "acc": bool(outcome_acc),
                "outcome_record": json.dumps(rec.to_dict()),
            },
        }
