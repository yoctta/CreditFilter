# CreditFilter

Training and evaluation code of the paper *CreditFilter: Calibrated Token Weights over the Within-Response Entropy Rank for RLVR*.

CreditFilter weights the advantage of every response token by a window over the entropy rank of the token within its
response. The code is a copy of [verl](https://github.com/volcengine/verl) with the package `creditfilter/` on top.

## Layout

| path | content |
|---|---|
| `creditfilter/` | the coefficient (`credit_filter.py`), the compared rules (`baselines.py`, `token_signals.py`), the reward (`outcome_reward.py`, `custom_score.py`) |
| `creditfilter/scripts/`, `launchers/` | the training recipe and a launcher for every arm of the paper |
| `eval/` | the evaluation behind the paper's tables |
| `verl/` | verl 0.8.0.dev; the files changed for this work are listed in `NOTICE` |
| `data/` | the training prompts (DAPO-17k with boxed-answer prompts) and the three evaluation sets |
| `tests/` | CPU unit tests: `python -m pytest tests/` |
| `measurement/` | the forking-value instrument and the per-position records of its measurements, with its own `README.md` |

## Requirements

- A Ray cluster of 8-GPU nodes whose workers run Python 3.12 with torch 2.10.0, transformers 5.3.0, vllm 0.18.0 and
  ray 2.54.1. The complete package list is `environment/pip_freeze.txt`.
- The model `Qwen/Qwen3-8B-Base`.
- This tree and the model under the same paths on the submitting host and on the workers; the submitting host needs
  the `ray` package.

## Training

```
export RAY_HEAD=<head_ip:8265> GEN_PATH=/path/to/Qwen3-8B-Base
bash launchers/run_arm.sh creditfilter
```

| | arms (name in the paper where it differs) |
|---|---|
| methods | `creditfilter`, `creditfilter_d` (CreditFilter-D) |
| baselines | `grpo`, `dapo`, `entropy_mask_batch` and `entropy_mask_response` (80/20 rule with the batch and the per-response cut), `stare`, `scgrpo` |
| ablations | `hard_mask` (top 20% window), `wide_window`, `random_rank`, `reversed_rank`, `batch_rank`, `no_floor`, `rescaled_masses`, `no_fallback`, `no_damping`, `no_window` (damping only), `top_rank_gain` |

Options, as environment variables:

| variable | meaning |
|---|---|
| `DATA_SEED` | order of the training prompts; 201 by default, the paper also uses 300 and 400 (`top_rank_gain`: 300) |
| `ACTOR_NNODES` | number of 8-GPU nodes, 2 by default |
| `HW_PROFILE` | `h100` (default) or `b200`; throughput settings only |
| `SMOKE=1` | 3 steps on a small batch |
| `DRY_RUN=1` | writes the job to the run directory and submits nothing |

A run writes to `runs/<arm>_<timestamp>/`: the checkpoints `global_step_N/`, the metrics `metrics/metrics.jsonl`, and
`runtime_env.json` with every `CF_*` setting of the run. `launchers/arms.sh` defines the arms, and
`creditfilter/scripts/launch_train.sh` lists every setting with its meaning.

## Evaluation

```
python3 eval/make_plan.py --out plan.json --base /path/to/Qwen3-8B-Base --runs-root runs --steps 350 375 400
bash launchers/run_eval.sh plan.json late        # a Ray job when RAY_HEAD is set, else on this node
python3 eval/grade_mv_sidecar.py 'results/late/*/samples.jsonl'      # needs: pip install math-verify
```

Every checkpoint is sampled 32 times per problem at temperature 1.0 and top-p 0.7, with at most 20480 new tokens, on
MATH500, AIME2024 and AIME2025. `samples.jsonl` holds one line per sample, and `samples.mv.jsonl` the Math-Verify
verdict that the paper's tables use.

## License

Apache-2.0. `verl/` is a modified copy of verl, also under Apache-2.0; see `NOTICE`.
