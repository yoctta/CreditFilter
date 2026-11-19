"""CreditFilter: token weights over the within-response entropy rank for RL with verifiable rewards, on top of verl.

  credit_filter.py   the CreditFilter coefficient
  baselines.py       the compared token-weighting rules (80/20 rule, STARE, SC-GRPO)
  routing.py         CF_ROUTE: which weighting is applied to the advantages
  token_signals.py   the teacher KL that SC-GRPO needs
  outcome_reward.py  the reward manager and the group-standardised advantages
  custom_score.py    the training grader
"""
