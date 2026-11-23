#!/usr/bin/env bash
# Full experiment plan: 4 methods x 3 seeds x 2 datasets, cross-dataset test, ISBI ablations,
# evaluation, flagging and comparison tables. Run from the project root on a GPU machine.
# Each block can be run on its own; comment out what you have already done.
set -euo pipefail
SEEDS="0 1 2"
METHODS="dqn mappo coma coma_ind unet"

# 1) train ---------------------------------------------------------------
for ds in isbi aariz; do
  for m in $METHODS; do
    for s in $SEEDS; do
      python scripts/train.py --config configs/${ds}_${m}.yaml --seed $s
    done
  done
done
for ab in noshare randinit singlescale team_reward; do      # ablations: ISBI, on DQN (best and fastest); team-reward ablation stays on MAPPO (fairness vs COMA)
  for s in $SEEDS; do
    python scripts/train.py --config configs/ablation_${ab}.yaml --seed $s
  done
done

# 2) evaluate --------------------------------------------------------------
for run in runs/isbi_* runs/ablation_*; do
  for split in val test1 test2; do
    python scripts/evaluate.py --checkpoint $run/best.pt --split $split
  done
  # cross-dataset: ISBI-trained models on the Aariz test set (shared landmarks only)
  if [[ $run == runs/isbi_* ]]; then
    python scripts/evaluate.py --checkpoint $run/best.pt --data-root data/aariz --split test
  fi
done
for run in runs/aariz_*; do
  for split in val test; do
    python scripts/evaluate.py --checkpoint $run/best.pt --split $split
  done
done

# 3) flagging (scorer fit on val, applied to test) -----------------------------
for run in runs/isbi_* runs/ablation_*; do
  for split in test1 test2; do
    python scripts/flag.py --val $run/eval_isbi_val.npz --test $run/eval_isbi_${split}.npz
  done
done
for run in runs/aariz_*; do
  python scripts/flag.py --val $run/eval_aariz_val.npz --test $run/eval_aariz_test.npz
done
for sig in spread conf shape; do                         # single-signal ablation, ISBI DQN
  for run in runs/isbi_dqn_dqn_s*; do
    python scripts/flag.py --val $run/eval_isbi_val.npz --test $run/eval_isbi_test1.npz --signals $sig
  done
done

# 4) tables + significance -----------------------------------------------------
for split in isbi_test1 isbi_test2; do
  python scripts/compare.py --split $split \
    --method DQN "runs/isbi_dqn_dqn_s*" --method MAPPO "runs/isbi_mappo_mappo_s*" \
    --method COMA "runs/isbi_coma_coma_s*" --method "COMA indiv" "runs/isbi_coma_ind_coma_s*" --method U-Net "runs/isbi_unet_unet_s*" \
    --method "DQN no sharing" "runs/ablation_noshare_dqn_s*" \
    --method "DQN random init" "runs/ablation_randinit_dqn_s*" \
    --method "DQN single scale" "runs/ablation_singlescale_dqn_s*" \
    --method "MAPPO team reward" "runs/ablation_team_reward_mappo_s*"
done
python scripts/compare.py --split aariz_test \
  --method DQN "runs/aariz_dqn_dqn_s*" --method MAPPO "runs/aariz_mappo_mappo_s*" \
  --method COMA "runs/aariz_coma_coma_s*" --method "COMA indiv" "runs/aariz_coma_ind_coma_s*" --method U-Net "runs/aariz_unet_unet_s*"
python scripts/compare.py --split aariz_test --out results/cross_dataset \
  --method DQN "runs/isbi_dqn_dqn_s*" --method MAPPO "runs/isbi_mappo_mappo_s*" \
  --method COMA "runs/isbi_coma_coma_s*" --method U-Net "runs/isbi_unet_unet_s*"
echo "done: tables in results/"
