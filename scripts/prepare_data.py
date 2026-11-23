"""Convert a raw dataset into the cache format used by training.

    python scripts/prepare_data.py isbi  --raw /path/to/ISBI2015 --out data/isbi
    python scripts/prepare_data.py aariz --raw /path/to/Aariz --out data/aariz --spacing-csv spacing.csv
    python scripts/prepare_data.py synthetic --out data/synthetic
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cephmarl.data import make_synthetic, prepare_aariz, prepare_isbi  # noqa: E402
from cephmarl.landmarks import AARIZ_LANDMARKS, ISBI_LANDMARKS  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("dataset", choices=["isbi", "aariz", "synthetic"])
p.add_argument("--raw", help="raw dataset folder")
p.add_argument("--out", required=True)
p.add_argument("--n-val", type=int, default=20, help="ISBI: training images held out for validation")
p.add_argument("--spacing-csv", help="Aariz: CSV with image id and pixel size in mm")
p.add_argument("--default-spacing", type=float, help="Aariz: mm/px used when an image is not in the CSV")
p.add_argument("--like", choices=["isbi", "aariz"], default="isbi",
               help="synthetic: use ISBI (19) or Aariz (29) landmark names, to test cross-dataset code")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

if a.dataset == "isbi":
    prepare_isbi(a.raw, a.out, n_val=a.n_val)
elif a.dataset == "aariz":
    prepare_aariz(a.raw, a.out, a.spacing_csv, a.default_spacing)
else:
    names = ISBI_LANDMARKS if a.like == "isbi" else AARIZ_LANDMARKS
    make_synthetic(a.out, num_landmarks=len(names), names=names, seed=a.seed)
