"""Predict landmark coordinates on your own X-ray image(s).

    python scripts/predict.py --checkpoint runs/isbi_unet_unet_s0/best.pt --image my_xray.bmp
    python scripts/predict.py --checkpoint runs/isbi_dqn_dqn_s0/best.pt --image my_xray.png --spacing 0.1 \
        --flag-val runs/isbi_dqn_dqn_s0/eval_isbi_val.npz

--spacing is the image's pixel size in mm (ISBI images: 0.1). Aariz-trained RL models can also take
--device-name (e.g. "ART Plus") to start from that device's mean shape.

For each image, writes to --out (default predictions/):
  <name>.csv / .json   landmark, x and y in original image pixels, x and y in mm, review flag
  <name>.png           the image with numbered landmarks (red = flagged for review)
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cephmarl import unet_baseline  # noqa: E402
from cephmarl.algos import dqn, onpolicy  # noqa: E402
from cephmarl.data import MM_PER_PX, _load_resampled  # noqa: E402
from cephmarl.env import CephEnv  # noqa: E402
from cephmarl.flagging import FlagScorer, ShapeModel, SIGNALS, features  # noqa: E402
from cephmarl.models import Actor, DuelingQNet, build_unet  # noqa: E402
from cephmarl.rl_common import run_episode  # noqa: E402
from cephmarl.utils import get_device  # noqa: E402


def load_model(ck, device):
    cfg, K = ck["cfg"], len(ck["landmark_names"])
    if ck["algo"] == "unet":
        model = build_unet(K, cfg["unet"]["encoder"], None).to(device)
        model.load_state_dict(ck["model"])
        return lambda img, dev_name: _unet(model, img, cfg, device)
    env = CephEnv(cfg["env"], K, ck["mean_shape"], device)
    if ck["algo"] == "dqn":
        net = DuelingQNet(K).to(device)
        net.load_state_dict(ck["net"])
        policy = dqn.make_policy(net.eval())
    else:
        actor = Actor(K).to(device)
        actor.load_state_dict(ck["actor"])
        policy = onpolicy.make_policy(actor.eval())
    return lambda img, dev_name: run_episode(env, policy, [img], devices=[dev_name])


def _unet(model, img, cfg, device):
    pred, conf = unet_baseline.predict_images(model, [img], cfg, device)
    return {"pred": pred, "conf": conf, "spread": np.zeros(pred.shape[:2])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", nargs="+", required=True, help="one or more image files")
    p.add_argument("--spacing", type=float, default=0.1, help="pixel size of the image in mm (ISBI: 0.1)")
    p.add_argument("--device-name", help="imaging device, for Aariz-trained RL models")
    p.add_argument("--flag-val", help="eval_<data>_val.npz of the same run: enables calibrated review flags")
    p.add_argument("--budget", type=float, default=0.10, help="share of landmarks flagged (calibrated on val)")
    p.add_argument("--out", default="predictions")
    p.add_argument("--device", default="auto")
    a = p.parse_args()

    device = get_device(a.device)
    ck = torch.load(a.checkpoint, map_location=device, weights_only=False)
    names = ck["landmark_names"]
    predict = load_model(ck, device)
    shape_model = ShapeModel.from_state(ck["shape_model"])

    scorer, threshold = None, None
    if a.flag_val:
        val = np.load(a.flag_val)
        signals = [s for s in SIGNALS if np.nanstd(val["features"][..., SIGNALS.index(s)]) > 1e-8]
        scorer = FlagScorer(signals).fit(val["features"], val["err"])
        threshold = np.quantile(scorer.score(val["features"]), 1 - a.budget)
    os.makedirs(a.out, exist_ok=True)

    for path in a.image:
        img, scale = _load_resampled(path, a.spacing)          # same preprocessing as training
        res = predict(img, a.device_name)
        pred = res["pred"][0]                                   # (K,2) in 0.1 mm/px pixels
        shape_res = shape_model.residuals(pred)
        if scorer is not None:
            score = scorer.score(features(res, shape_model))[0]
            flagged = score > threshold
        else:                                                   # uncalibrated fallback: unusual shape
            score, flagged = shape_res, shape_res > 3.0

        rows = []
        for k, n in enumerate(names):
            x, y = pred[k]
            rows.append({"landmark": n, "x_px": round(float(x / scale), 1), "y_px": round(float(y / scale), 1),
                         "x_mm": round(float(x * MM_PER_PX), 2), "y_mm": round(float(y * MM_PER_PX), 2),
                         "uncertainty": round(float(score[k]), 3), "review": bool(flagged[k])})
        stem = os.path.join(a.out, os.path.splitext(os.path.basename(path))[0])
        with open(stem + ".csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        with open(stem + ".json", "w", encoding="utf-8") as f:
            json.dump({"image": path, "checkpoint": a.checkpoint, "algo": ck["algo"],
                       "pixel_spacing_mm": a.spacing, "landmarks": rows}, f, indent=2)

        im = Image.open(path).convert("RGB")
        dr = ImageDraw.Draw(im)
        r = max(4, im.size[0] // 200)
        for k, row in enumerate(rows):
            x, y, col = row["x_px"], row["y_px"], (255, 40, 40) if row["review"] else (40, 220, 40)
            dr.ellipse([x - r, y - r, x + r, y + r], outline=col, width=max(2, r // 2))
            dr.text((x + r + 2, y - r), str(k + 1), fill=col)
        im.save(stem + ".png")

        print(f"\n{path}  ({ck['algo']}, {len(names)} landmarks; coordinates in original image pixels)")
        for k, row in enumerate(rows):
            print(f"  {k + 1:2d} {row['landmark']:<26s} x={row['x_px']:8.1f}  y={row['y_px']:8.1f}"
                  + ("   <- review" if row["review"] else ""))
        print(f"-> {stem}.csv / .json / .png")


if __name__ == "__main__":
    main()
