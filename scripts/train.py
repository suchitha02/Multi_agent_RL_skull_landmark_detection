"""Train one method from a config; saves the best-on-validation checkpoint.

    python scripts/train.py --config configs/isbi_mappo.yaml --seed 0
    python scripts/train.py --config configs/synthetic_smoke.yaml --set algo=dqn
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cephmarl import unet_baseline  # noqa: E402
from cephmarl.algos import coma, dqn, mappo  # noqa: E402
from cephmarl.data import CephDataset  # noqa: E402
from cephmarl.utils import Logger, get_device, load_config, set_seed  # noqa: E402

TRAINERS = {"dqn": dqn.train, "mappo": mappo.train, "coma": coma.train, "unet": unet_baseline.train}

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--seed", type=int)
p.add_argument("--out", help="run folder (default runs/<config>_<algo>_s<seed>)")
p.add_argument("--set", nargs="*", default=[], help="overrides, e.g. env.reward=team agent_steps=10000")
a = p.parse_args()

cfg = load_config(a.config, a.set)
if a.seed is not None:
    cfg["seed"] = a.seed
name = os.path.splitext(os.path.basename(a.config))[0]
out = a.out or os.path.join("runs", f"{name}_{cfg['algo']}_s{cfg['seed']}")
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)

set_seed(cfg["seed"])
device = get_device(cfg["device"])
train_ds = CephDataset(cfg["data_root"], cfg["train_split"])
val_ds = CephDataset(cfg["data_root"], cfg["val_split"])
print(f"{cfg['algo']} on {cfg['data_root']}: {len(train_ds)} train / {len(val_ds)} val images, "
      f"{train_ds.num_landmarks} landmarks, device {device}")

ckpt, best = TRAINERS[cfg["algo"]](cfg, train_ds, val_ds, Logger(out), device)
torch.save(ckpt, os.path.join(out, "best.pt"))
print(f"best val MRE {best:.3f} mm -> {out}/best.pt")
