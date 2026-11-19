#!/bin/bash
# The training arms of the paper (sourced by run_arm.sh). resolve_arm <arm> exports the settings of an arm;
# everything else is the shared recipe of creditfilter/scripts/launch_recipe.sh.
ARM_LIST="creditfilter creditfilter_d top_rank_gain hard_mask wide_window random_rank reversed_rank batch_rank no_floor rescaled_masses no_fallback no_damping no_window entropy_mask_batch entropy_mask_response grpo dapo stare scgrpo"

resolve_arm() {
  local A=${1:?arm}
  case " $ARM_LIST " in *" $A "*) ;; *) echo "[arms] unknown arm '$A' (valid: $ARM_LIST)" >&2; return 2 ;; esac
  case $A in
    grpo)   export ARM=grpo ;;
    dapo)   export ARM=dapo ;;
    stare)  export ARM=stare ;;
    scgrpo) export ARM=scgrpo ;;
    entropy_mask_batch)    export ARM=entropy_mask CF_MASK_SCOPE=batch ;;      # 80/20 rule, batch cut
    entropy_mask_response) export ARM=entropy_mask CF_MASK_SCOPE=response ;;   # 80/20 rule, per-response cut
    *)
      # CreditFilter: widths 0.10 / 0.14, floor 0.1 on positive-advantage responses, damping 1 / 0, fallback band 1.5 - 3.0 nats
      export ARM=creditfilter CF_ROUTE=creditfilter
      export CF_WIDTH_POS=0.10 CF_WIDTH_NEG=0.14 CF_FLOOR_POS=0.1 CF_FLOOR_NEG=0.0
      export CF_KERNEL=exp CF_HARD_TOP=0.2 CF_RANK_KEY=entropy CF_RANK_SCOPE=response CF_WIDTH_MODE=fixed
      export CF_DAMPING_POS=1.0 CF_DAMPING_NEG=0.0
      export CF_FALLBACK_H_LO=1.5 CF_FALLBACK_H_HI=3.0
      export CF_MASS_TARGET_POS=0 CF_MASS_TARGET_NEG=0 CF_MASS_EQUALIZE=0
      export CF_TOP_GAIN_Q=0 CF_TOP_GAIN=1.0
      case $A in
        creditfilter)    ;;
        creditfilter_d)  # CreditFilter-D: tracked widths starting at the measured 0.111 / 0.155, no floor, class masses equalised
                         export CF_WIDTH_MODE=tracked CF_WIDTH_POS=0.111 CF_WIDTH_NEG=0.155
                         export CF_TRACK_C_POS=1.0 CF_TRACK_C_NEG=1.0 CF_TRACK_ALPHA=0.1 CF_TRACK_MIN_ROWS=64
                         export CF_TRACK_MIN_POS=0.05 CF_TRACK_MAX_POS=0.30 CF_TRACK_MIN_NEG=0.07 CF_TRACK_MAX_NEG=0.40
                         export CF_FLOOR_POS=0 CF_FLOOR_NEG=0 CF_MASS_EQUALIZE=1 ;;
        top_rank_gain)   # weight of the top 0.25% of the rank times 3, class masses kept at CreditFilter's (run with DATA_SEED=300)
                         export CF_TOP_GAIN_Q=0.0025 CF_TOP_GAIN=3.0 CF_MASS_TARGET_POS=0.1964 CF_MASS_TARGET_NEG=0.2003 ;;
        hard_mask)       # top 20% window
                         export CF_KERNEL=hard CF_HARD_TOP=0.2 CF_FLOOR_POS=0 CF_FLOOR_NEG=0 ;;
        wide_window)     # double widths, class masses kept at CreditFilter's
                         export CF_WIDTH_POS=0.20 CF_WIDTH_NEG=0.28 CF_MASS_TARGET_POS=0.1964 CF_MASS_TARGET_NEG=0.2003 ;;
        random_rank)     export CF_RANK_KEY=random ;;
        reversed_rank)   export CF_RANK_KEY=reversed ;;
        batch_rank)      export CF_RANK_SCOPE=batch ;;
        no_floor)        export CF_FLOOR_POS=0 CF_FLOOR_NEG=0 ;;
        rescaled_masses) export CF_FLOOR_POS=0 CF_FLOOR_NEG=0 CF_MASS_EQUALIZE=1 ;;
        no_fallback)     export CF_FALLBACK_H_LO=1e6 CF_FALLBACK_H_HI=2e6 ;;
        no_damping)      export CF_DAMPING_POS=0.0 CF_DAMPING_NEG=0.0 ;;
        no_window)       # damping only: every token weight is 1
                         export CF_WIDTH_POS=9e9 CF_WIDTH_NEG=9e9 ;;
      esac ;;
  esac
  return 0
}
