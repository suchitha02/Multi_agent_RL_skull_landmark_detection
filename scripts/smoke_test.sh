#!/usr/bin/env bash
# End-to-end check on synthetic data (CPU, a few minutes). Run this first on any new machine.
set -euo pipefail
python -m pytest -q tests
python scripts/prepare_data.py synthetic --like isbi --out data/syn_isbi
python scripts/prepare_data.py synthetic --like aariz --seed 1 --out data/syn_aariz
python scripts/analyze_dataset.py --data-root data/syn_isbi --compare data/syn_aariz --out analysis/smoke > /dev/null
for m in dqn mappo coma unet; do
  python scripts/train.py --config configs/synthetic_smoke.yaml --out runs/smoke_$m --set algo=$m data_root=data/syn_isbi
  python scripts/evaluate.py --checkpoint runs/smoke_$m/best.pt --split val
  python scripts/evaluate.py --checkpoint runs/smoke_$m/best.pt --split test
  python scripts/evaluate.py --checkpoint runs/smoke_$m/best.pt --data-root data/syn_aariz --split test
  python scripts/flag.py --val runs/smoke_$m/eval_syn_isbi_val.npz --test runs/smoke_$m/eval_syn_isbi_test.npz > /dev/null
done
python scripts/compare.py --split syn_isbi_test --out results/smoke \
  --method DQN runs/smoke_dqn --method MAPPO runs/smoke_mappo --method COMA runs/smoke_coma --method U-Net runs/smoke_unet
echo "smoke test passed"
