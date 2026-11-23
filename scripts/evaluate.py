"""Evaluate a checkpoint on a split; works across datasets (ISBI model on Aariz, etc.).

    python scripts/evaluate.py --checkpoint runs/isbi_mappo_mappo_s0/best.pt --split test1
    python scripts/evaluate.py --checkpoint runs/isbi_mappo_mappo_s0/best.pt --data-root data/aariz --split test

Writes <out>.npz (predictions, errors, uncertainty features) and <out>.json (metrics).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cephmarl import unet_baseline  # noqa: E402
from cephmarl.algos import dqn, onpolicy  # noqa: E402
from cephmarl.data import CephDataset  # noqa: E402
from cephmarl.env import CephEnv  # noqa: E402
from cephmarl.flagging import ShapeModel, features  # noqa: E402
from cephmarl.landmarks import map_indices  # noqa: E402
from cephmarl.metrics import radial_errors, summarize  # noqa: E402
from cephmarl.models import Actor, DuelingQNet, build_unet  # noqa: E402
from cephmarl.rl_common import gather_gt, predict_dataset  # noqa: E402
from cephmarl.utils import get_device  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--data-root", help="default: the dataset the model was trained on")
p.add_argument("--split", required=True)
p.add_argument("--out", help="output prefix (default: next to the checkpoint)")
p.add_argument("--device", default="auto")
a = p.parse_args()

device = get_device(a.device)
ck = torch.load(a.checkpoint, map_location=device, weights_only=False)
cfg, names = ck["cfg"], ck["landmark_names"]
K = len(names)
root = a.data_root or cfg["data_root"]
ds = CephDataset(root, a.split)
gt_index = map_indices(names, ds.landmark_names)
print(f"{ck['algo']} ({K} landmarks) on {root}/{a.split}: {len(ds)} images, "
      f"{sum(i >= 0 for i in gt_index)} landmarks scored")

if ck["algo"] == "unet":
    model = build_unet(K, cfg["unet"]["encoder"], None).to(device)
    model.load_state_dict(ck["model"])
    out = unet_baseline.predict_dataset(model, ds, cfg, device, gt_index)
else:
    env = CephEnv(cfg["env"], K, ck["mean_shape"], device)
    if ck["algo"] == "dqn":
        net = DuelingQNet(K).to(device)
        net.load_state_dict(ck["net"])
        policy = dqn.make_policy(net.eval())
    else:
        actor = Actor(K).to(device)
        actor.load_state_dict(ck["actor"])
        policy = onpolicy.make_policy(actor.eval())
    out = predict_dataset(env, policy, ds, batch=cfg["batch_images"], gt_index=gt_index)

err = radial_errors(out["pred"], out["gt"])
feats = features(out, ShapeModel.from_state(ck["shape_model"]))
metrics = {"algo": ck["algo"], "checkpoint": a.checkpoint, "data": f"{root}/{a.split}",
           "model": summarize(err, names), "mean_steps_per_agent": float(out["steps"].mean())}

# Aariz: also report errors with the official CSV pixel sizes (what published Aariz results use).
# An error of e cached px is e * 0.1 / spacing original px, i.e. e * 0.1 * csv / spacing mm officially.
if all("csv_spacing_mm" in it for it in ds.items):
    factor = np.array([it["csv_spacing_mm"] / it["spacing_mm"] for it in ds.items])[:, None]
    metrics["model_official_csv_pixel_size"] = summarize(err * factor, names)
devices = [it.get("device") for it in ds.items]
if any(devices):
    metrics["per_device_MRE_mm"] = {d: float(np.nanmean(err[[x == d for x in devices]]))
                                    for d in sorted(set(devices))}

jun, sen = gather_gt(ds, "junior", gt_index), gather_gt(ds, "senior", gt_index)
disagreement = None
if jun is not None and sen is not None:
    disagreement = radial_errors(jun, sen)
    metrics["human_junior_vs_senior"] = summarize(disagreement, names)

prefix = a.out or os.path.join(os.path.dirname(a.checkpoint), f"eval_{os.path.basename(root)}_{a.split}")
np.savez(prefix + ".npz", pred=out["pred"], gt=out["gt"], err=err, features=feats,
         disagreement=disagreement if disagreement is not None else np.full(err.shape, np.nan),
         ids=np.array([it["id"] for it in ds.items]), landmark_names=np.array(names))
with open(prefix + ".json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
m = metrics["model"]
print(f"MRE {m['MRE_mm']:.3f} +- {m['SD_mm']:.3f} mm | " +
      " ".join(f"SDR{t}={m[f'SDR_{t}mm_%']:.1f}%" for t in (2.0, 2.5, 3.0, 4.0)))
print(f"-> {prefix}.json / .npz")
