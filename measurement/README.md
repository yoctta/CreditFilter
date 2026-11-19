# Forking-value instrument

The measurement behind the forking spectrum of the paper. At sampled positions of a response the next token is replaced by
each of several branches, every branch is continued several times with the training sampler, and the grades of the
continuations estimate the forking value (the variance of the success probability over the next token), the fork probability
and the fork consequence, their ratio.

| path | content |
|---|---|
| `four_branch/` | the four-branch instrument: the sampled token and its three most probable alternatives, 8 continuations each, 24 positions per response over ten bands of the entropy rank |
| `full_support/` | the full-support instrument: the most probable tokens up to 0.9 of the probability mass plus two draws from the rest, 48 continuations each, 16 rank strata with Horvitz-Thompson weights |
| `metrics/` | the per-position records of the paper's measurements |
| `prompts.py`, `grader.py` | the prompts (rows of `data/dapo17k_boxed.parquet`, listed in `metrics/prompts.jsonl`) and the grader of both instruments |

## Records

| files | measurement |
|---|---|
| `metrics/four_branch/qwen3-8b-base.{correct,wrong}.jsonl.gz` | Qwen3-8B-Base: 294 correct responses with 7,056 positions, 672 wrong responses with 16,128 positions |
| `metrics/four_branch/<model>.jsonl.gz` | seven further models, 120 responses each (60 for Qwen3-32B) |
| `metrics/full_support/{positions,responses}.jsonl.gz`, `design.json` | Qwen3-8B-Base: 13,129 positions of 800 responses |

One line is one measured position. The fields are described at the top of the two `analyze.py`.

## Tables of the paper

```
cd measurement
python3 four_branch/analyze.py base metrics/four_branch/qwen3-8b-base.correct.jsonl.gz metrics/four_branch/qwen3-8b-base.wrong.jsonl.gz
python3 four_branch/analyze.py models metrics/four_branch/qwen3-1.7b.jsonl.gz metrics/four_branch/qwen3-4b.jsonl.gz    # ... one file per model
python3 full_support/analyze.py metrics/full_support/positions.jsonl.gz metrics/full_support/responses.jsonl.gz metrics/full_support/design.json
python3 full_support/increments.py metrics/full_support/positions.jsonl.gz metrics/full_support/responses.jsonl.gz metrics/full_support/design.json
```

| command | output |
|---|---|
| `four_branch/analyze.py base` | the spectrum by response class, the ten-band table, cover and truncation, the continuation-horizon check |
| `four_branch/analyze.py models` | the model table (the row of Qwen3-32B uses `--replicates 400`) |
| `full_support/analyze.py` | the strata table, the consequence pooled over rank regions, the retention table |
| `full_support/increments.py` | the test of the look-ahead assumption, the value-factor comparison |

The commands need numpy only and reproduce the intervals of the paper, which are seeded.

## Running a measurement

A measurement needs the training environment (`environment/pip_freeze.txt`) and a Ray cluster with GPUs.

```
MODEL=/path/to/Qwen3-8B-Base TAG=base bash measurement/four_branch/run.sh
MODEL=/path/to/Qwen3-8B-Base TAG=base bash measurement/full_support/run.sh        # SMOKE=1: a small run
```

Both scripts write to `runs/<TAG>/` next to them and end with the tables of the new measurement. Another model family needs
its own special tokens in `FORMAT_TOKENS`. `python3 full_support/test_estimator.py` checks the estimator, the sampling
design and the analysis on CPU.
