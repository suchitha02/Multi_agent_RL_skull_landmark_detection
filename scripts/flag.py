"""Fit the flag scorer on validation predictions, apply it to test predictions.

    python scripts/flag.py --val runs/X/eval_isbi_val.npz --test runs/X/eval_isbi_test1.npz
    python scripts/flag.py ... --signals shape          # single-signal ablation
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cephmarl.flagging import SIGNALS, FlagScorer, flag_metrics  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--val", required=True)
p.add_argument("--test", required=True)
p.add_argument("--signals", default=",".join(SIGNALS), help="comma list of spread,conf,shape")
p.add_argument("--threshold", type=float, default=2.0, help="error (mm) the scorer learns to detect")
p.add_argument("--budget", type=float, default=0.10, help="share of landmarks sent for review")
p.add_argument("--out")
a = p.parse_args()

val, test = np.load(a.val), np.load(a.test)
signals = a.signals.split(",")
# drop signals that are constant on validation (e.g. U-Net has no oscillation spread)
signals = [s for s in signals if np.nanstd(val["features"][..., SIGNALS.index(s)]) > 1e-8]
if not signals:
    sys.exit(f"skip {a.test}: every uncertainty signal is constant on validation (untrained model?)")
scorer = FlagScorer(signals, a.threshold).fit(val["features"], val["err"])
score = scorer.score(test["features"])
dis = test["disagreement"] if not np.isnan(test["disagreement"]).all() else None
res = {"signals": signals, "threshold_mm": a.threshold, "budget": a.budget,
       **flag_metrics(score, test["err"], a.budget, dis)}

out = a.out or a.test.replace(".npz", f"_flags_{'-'.join(signals)}.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
print(json.dumps({k: v for k, v in res.items() if k != "risk_coverage"}, indent=2))
print(f"-> {out}")
