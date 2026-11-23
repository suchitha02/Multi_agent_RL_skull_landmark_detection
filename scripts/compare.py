"""Results table over seeds + paired Wilcoxon tests (per-image MRE) with Holm correction.

    python scripts/compare.py --split isbi_test1 \
        --method DQN "runs/isbi_dqn_dqn_s*" --method MAPPO "runs/isbi_mappo_mappo_s*" ...

Each --method takes a name and a glob of run folders; files eval_<split>.npz are read from each.
"""
import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cephmarl.metrics import per_image_mre, summarize  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--split", required=True, help="e.g. isbi_test1, aariz_test")
p.add_argument("--method", nargs=2, action="append", metavar=("NAME", "GLOB"), required=True)
p.add_argument("--out", default="results")
a = p.parse_args()

rows, per_image = [], {}
for name, pattern in a.method:
    files = sorted(glob.glob(os.path.join(pattern, f"eval_{a.split}.npz")))
    if not files:
        print(f"skip {name}: no eval_{a.split}.npz under {pattern}")
        continue
    stats = [summarize(np.load(f)["err"]) for f in files]
    errs = np.stack([np.load(f)["err"] for f in files])
    per_image[name] = per_image_mre(np.nanmean(errs, 0))  # average seeds per image
    row = {"method": name, "seeds": len(files)}
    for k in ("MRE_mm", "SDR_2.0mm_%", "SDR_2.5mm_%", "SDR_3.0mm_%", "SDR_4.0mm_%"):
        v = [s[k] for s in stats]
        row[k] = f"{np.mean(v):.2f} ± {np.std(v):.2f}"
    rows.append(row)

tests = []
for x, y in itertools.combinations(per_image, 2):
    tests.append({"a": x, "b": y, "p": float(wilcoxon(per_image[x], per_image[y]).pvalue)})
order = np.argsort([t["p"] for t in tests])  # Holm-Bonferroni
running = 0.0
for rank, i in enumerate(order):
    running = max(running, min(1.0, tests[i]["p"] * (len(tests) - rank)))
    tests[i]["p_holm"] = running

os.makedirs(a.out, exist_ok=True)
cols = list(rows[0].keys()) if rows else []
lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
lines += ["", "| A | B | p (Wilcoxon) | p (Holm) |", "|---|---|---|---|"]
lines += [f"| {t['a']} | {t['b']} | {t['p']:.4g} | {t['p_holm']:.4g} |" for t in tests]
md = "\n".join(lines)
print(md)
with open(os.path.join(a.out, f"compare_{a.split}.md"), "w", encoding="utf-8") as f:
    f.write(md + "\n")
with open(os.path.join(a.out, f"compare_{a.split}.json"), "w", encoding="utf-8") as f:
    json.dump({"rows": rows, "tests": tests}, f, indent=2)
