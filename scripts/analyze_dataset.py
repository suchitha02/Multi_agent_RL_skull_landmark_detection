"""Dataset analysis before training. Works on any prepared dataset (ISBI, Aariz, synthetic).

    python scripts/analyze_dataset.py --data-root data/isbi  --out analysis/isbi
    python scripts/analyze_dataset.py --data-root data/aariz --out analysis/aariz
    python scripts/analyze_dataset.py --data-root data/isbi  --compare data/aariz --out analysis/isbi_vs_aariz
    python scripts/analyze_dataset.py --isbi-annotations /path/to/AnnotationsByMD --out analysis/isbi_ann
        (landmarks only, no images needed: image-based sections are skipped)

Writes <out>/report.md (tables + findings), <out>/summary.json and <out>/figures/*.png.

Sections
  1  Inventory          images per split, sizes, pixel spacing, devices, missing / out-of-image landmarks
  2  Intensity          brightness and contrast per split and device
  3  Landmark layout    where each landmark sits and how much it moves between patients
  4  Doctor agreement   junior vs senior distance and systematic bias per landmark
  5  Shape variation    Procrustes PCA, outlier skulls
  6  Mean-shape start   error of the RL agents' starting position, and whether the step budget reaches
  7  Augmentation       how often a random affine would push a landmark out of the image
  8  Split shift        train vs test differences (KS tests)
  9  Physical scale     Sella-Nasion distance in mm per split and device (catches wrong pixel spacing)
  10 Cross-dataset      (with --compare) landmark mapping, scale and starting error across datasets
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cephmarl.augment import random_affine  # noqa: E402
from cephmarl.data import CephDataset, _read_isbi_txt, isbi_annotations, isbi_split_ids  # noqa: E402
from cephmarl.landmarks import ISBI_LANDMARKS  # noqa: E402
from cephmarl.flagging import _align  # noqa: E402
from cephmarl.landmarks import map_indices  # noqa: E402
from cephmarl.utils import load_config  # noqa: E402

MM = 0.1  # every cached image is 0.1 mm/px
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- helpers
class AnnotationsOnly:
    """ISBI split built from the annotation files alone (images are 1935x2400 at 0.1 mm/px)."""

    def __init__(self, ann_root, ids):
        jun_dir, sen_dir = isbi_annotations(ann_root)
        self.landmark_names = ISBI_LANDMARKS
        self.items = []
        for i in ids:
            jun = _read_isbi_txt(os.path.join(jun_dir, f"{i}.txt"))
            sen = _read_isbi_txt(os.path.join(sen_dir, f"{i}.txt"))
            self.items.append({"id": i, "width": 1935, "height": 2400, "landmarks": (jun + sen) / 2,
                               "junior": jun, "senior": sen})

    def __len__(self):
        return len(self.items)

    @property
    def num_landmarks(self):
        return len(self.landmark_names)

    def landmarks(self, i, key="landmarks"):
        v = self.items[i].get(key)
        return None if v is None else np.asarray(v, dtype=np.float32)

    def all_landmarks(self, key="landmarks"):
        return np.stack([self.landmarks(i, key) for i in range(len(self))])


def has_images(ds):
    return not isinstance(ds, AnnotationsOnly)


def splits_of(root):
    return [s for s in ("train", "val", "test1", "test2", "test") if os.path.exists(os.path.join(root, s, "meta.json"))]


def norm_shapes(ds, key="landmarks"):
    """(N,K,2) landmarks divided by image (W,H)."""
    return np.stack([ds.landmarks(i, key) / [it["width"], it["height"]] for i, it in enumerate(ds.items)])


def fmt(x, d=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def thumb(ds, i, scale=8):
    img = ds.image(i)
    return np.asarray(Image.fromarray(img).resize((img.shape[1] // scale, img.shape[0] // scale), Image.BILINEAR))


class Report:
    def __init__(self, out):
        self.out, self.md, self.js = out, [], {}
        os.makedirs(os.path.join(out, "figures"), exist_ok=True)

    def h(self, text):
        self.md.append(f"\n## {text}\n")

    def p(self, text):
        self.md.append(text + "\n")

    def fig(self, name, caption):
        path = os.path.join(self.out, "figures", name)
        plt.tight_layout()
        plt.savefig(path, dpi=110)
        plt.close()
        self.md.append(f"![{caption}](figures/{name})\n\n*{caption}*\n")

    def save(self):
        with open(os.path.join(self.out, "report.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(self.md))
        with open(os.path.join(self.out, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(self.js, f, indent=2, default=float)


# ---------------------------------------------------------------- sections
def inventory(rep, root, data):
    rep.h("1. Inventory")
    rows, js = [], {}
    for s, ds in data.items():
        h = np.array([it["height"] for it in ds.items]) * MM
        w = np.array([it["width"] for it in ds.items]) * MM
        lm = ds.all_landmarks()
        missing = int(np.isnan(lm).any(-1).sum())
        size = np.array([[it["width"], it["height"]] for it in ds.items])[:, None]
        outside = int(((lm < 0) | (lm > size - 1)).any(-1).sum())
        spacing = sorted({it.get("spacing_mm", MM) for it in ds.items})
        has_both = all(it.get("junior") is not None and it.get("senior") is not None for it in ds.items)
        rows.append([s, len(ds), f"{h.min():.0f}–{h.max():.0f}", f"{w.min():.0f}–{w.max():.0f}",
                     ", ".join(f"{v:g}" for v in spacing), missing, outside, "yes" if has_both else "no"])
        js[s] = {"images": len(ds), "height_mm": [h.min(), h.max()], "width_mm": [w.min(), w.max()],
                 "spacings_mm": spacing, "missing_landmarks": missing, "outside_image": outside,
                 "junior_and_senior": has_both}
        devices = [it.get("device") for it in ds.items if it.get("device")]
        if devices:
            js[s]["devices"] = {d: devices.count(d) for d in sorted(set(devices))}
    names = next(iter(data.values())).landmark_names
    rep.p(f"Dataset `{root}`: {sum(len(d) for d in data.values())} images, {len(names)} landmarks.\n")
    rep.p(table(["Split", "Images", "Height (mm)", "Width (mm)", "Original mm/px", "Missing landmarks",
                 "Outside image", "Junior + senior"], rows))
    dev = {s: v["devices"] for s, v in js.items() if "devices" in v}
    if dev:
        all_d = sorted({d for v in dev.values() for d in v})
        rep.p("\nImages per device:\n")
        rep.p(table(["Device"] + list(dev), [[d] + [dev[s].get(d, 0) for s in dev] for d in all_d]))
    rep.js["inventory"] = js

    # sample thumbnails
    ds = data[next(iter(data))]
    if not has_images(ds):
        return
    n = min(6, len(ds))
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.4))
    for ax, i in zip(np.atleast_1d(axes), np.linspace(0, len(ds) - 1, n).astype(int)):
        ax.imshow(thumb(ds, i), cmap="gray")
        lm = ds.landmarks(i) / 8
        ax.scatter(lm[:, 0], lm[:, 1], s=6, c="red")
        ax.set_title(ds.items[i]["id"], fontsize=8)
        ax.axis("off")
    rep.fig("samples.png", f"Sample {next(iter(data))} images with ground-truth landmarks (red).")


def intensity(rep, data, max_images=60):
    rep.h("2. Intensity")
    if not has_images(next(iter(data.values()))):
        rep.p("Skipped: annotations-only run (no images).")
        return
    rows, groups = [], {}
    for s, ds in data.items():
        idx = np.linspace(0, len(ds) - 1, min(max_images, len(ds))).astype(int)
        means, stds = [], []
        for i in idx:
            t = thumb(ds, i, 4).astype(np.float32)
            means.append(t.mean())
            stds.append(t.std())
            groups.setdefault(ds.items[i].get("device") or s, []).append(t.ravel()[::7])
        rows.append([s, len(idx), fmt(np.mean(means), 1), fmt(np.std(means), 1), fmt(np.mean(stds), 1)])
        rep.js.setdefault("intensity", {})[s] = {"mean": np.mean(means), "mean_sd": np.std(means),
                                                 "contrast": np.mean(stds)}
    rep.p(table(["Split", "Images sampled", "Mean grey (0–255)", "SD between images", "Mean contrast (SD)"], rows))
    plt.figure(figsize=(7, 3.5))
    for g, vals in groups.items():
        plt.hist(np.concatenate(vals), bins=64, histtype="step", density=True, label=str(g))
    plt.xlabel("grey level")
    plt.ylabel("density")
    plt.legend(fontsize=7)
    rep.fig("intensity_hist.png", "Grey-level distribution per split (or per device when known).")


def landmark_layout(rep, train):
    rep.h("3. Landmark layout (train)")
    names = train.landmark_names
    lm = train.all_landmarks()
    nrm = norm_shapes(train)
    sd_mm = np.nanstd(lm, 0) * MM                                           # (K,2)
    rows = [[n, fmt(nrm[:, k, 0].mean(), 3), fmt(nrm[:, k, 1].mean(), 3), fmt(sd_mm[k, 0], 1), fmt(sd_mm[k, 1], 1)]
            for k, n in enumerate(names)]
    rep.p("Mean position as a fraction of image width/height, and spread between patients in mm. "
          "Large spread means the agent's starting point (mean shape) is far off more often.\n")
    rep.p(table(["Landmark", "Mean x (frac)", "Mean y (frac)", "SD x (mm)", "SD y (mm)"], rows))
    rep.js["landmark_spread_mm"] = {n: float(np.hypot(*sd_mm[k])) for k, n in enumerate(names)}

    fig, ax = plt.subplots(figsize=(6, 7))
    if has_images(train):
        ax.imshow(thumb(train, 0), cmap="gray", extent=(0, 1, 1, 0), alpha=0.5)
    for k in range(len(names)):
        ax.scatter(nrm[:, k, 0], nrm[:, k, 1], s=3, alpha=0.4)
        ax.annotate(str(k + 1), nrm[:, k].mean(0), fontsize=8, weight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect(train.items[0]["height"] / train.items[0]["width"])
    ax.set_title("All training landmarks (normalised coords); numbers = landmark index + 1")
    rep.fig("landmark_cloud.png", "Landmark positions across all training images.")


def doctor_agreement(rep, data):
    rep.h("4. Doctor agreement (junior vs senior)")
    ds_all = [ds for ds in data.values() if all(it.get("junior") is not None for it in ds.items)]
    if not ds_all:
        rep.p("No junior/senior annotations in this dataset.")
        return
    names = ds_all[0].landmark_names
    jun = np.concatenate([d.all_landmarks("junior") for d in ds_all])
    sen = np.concatenate([d.all_landmarks("senior") for d in ds_all])
    dist = np.linalg.norm(jun - sen, axis=-1) * MM                          # (N,K)
    bias = np.nanmean(sen - jun, 0) * MM                                    # (K,2)
    rows = [[n, fmt(np.nanmean(dist[:, k])), fmt(np.nanstd(dist[:, k])), fmt(np.nanmean(dist[:, k] > 2) * 100, 1),
             fmt(np.nanmean(dist[:, k] > 4) * 100, 1), f"({fmt(bias[k, 0])}, {fmt(bias[k, 1])})"]
            for k, n in enumerate(names)]
    order = np.argsort(-np.nanmean(dist, 0))
    rep.p(f"All splits pooled ({len(jun)} images). Overall junior–senior distance: "
          f"**{np.nanmean(dist):.2f} ± {np.nanstd(dist):.2f} mm**; {np.nanmean(dist > 2) * 100:.1f}% of points differ by "
          f"more than 2 mm. This is the human reference error; a model near it is at expert level. "
          f"Least consistent landmarks: {', '.join(names[k] for k in order[:3])}.\n")
    rep.p("Bias = mean (senior − junior) offset in mm (x, y). A bias far from zero means the two doctors use a "
          "different definition, so the averaged ground truth sits between two habits.\n")
    rep.p(table(["Landmark", "Mean (mm)", "SD (mm)", "> 2 mm (%)", "> 4 mm (%)", "Bias x, y (mm)"],
                [rows[k] for k in order]))
    ids = [it["id"] for d in ds_all for it in d.items]
    gross = np.argwhere(dist > 10)
    rep.p(f"\n**Gross disagreements (> 10 mm): {len(gross)} points** in {len({i for i, _ in gross})} images. "
          "Each is a mis-click by one annotator or a different definition of the landmark (look at which landmarks repeat); either way the averaged ground truth is unreliable there.\n")
    if len(gross):
        top = sorted(gross.tolist(), key=lambda ik: -dist[ik[0], ik[1]])[:12]
        rep.p(table(["Image", "Landmark", "Junior–senior (mm)"], [[ids[i], names[k], fmt(dist[i, k], 1)] for i, k in top]))
    rep.js["doctor_agreement"] = {"overall_mm": float(np.nanmean(dist)), "gross_points": int(len(gross)),
                                  "per_landmark_mm": {n: float(np.nanmean(dist[:, k])) for k, n in enumerate(names)}}
    plt.figure(figsize=(8, 3.5))
    plt.bar(range(len(names)), np.nanmean(dist, 0), yerr=np.nanstd(dist, 0), capsize=2)
    plt.axhline(2, ls="--", c="gray", lw=1)
    plt.xticks(range(len(names)), [str(k + 1) for k in range(len(names))])
    plt.xlabel("landmark index + 1")
    plt.ylabel("junior–senior distance (mm)")
    rep.fig("doctor_agreement.png", "Junior–senior distance per landmark (mean ± SD); dashed line = 2 mm.")


def shape_variation(rep, data):
    rep.h("5. Shape variation (Procrustes PCA on train)")
    train = data["train"]
    shapes = train.all_landmarks().astype(np.float64)
    ref = shapes[0]
    for _ in range(3):
        aligned = np.stack([_align(s, ref) for s in shapes])
        ref = aligned.mean(0)
    flat = aligned.reshape(len(aligned), -1)
    mu = flat.mean(0)
    u, sv, vt = np.linalg.svd(flat - mu, full_matrices=False)
    var = sv ** 2 / (sv ** 2).sum()
    k95 = int(np.searchsorted(np.cumsum(var), 0.95) + 1)
    rep.p(f"{k95} of {flat.shape[1]} possible shape modes explain 95% of the variation between skulls (first 3: "
          f"{', '.join(f'{v * 100:.1f}%' for v in var[:3])}). The larger the share of the first few modes, the more "
          f"one landmark's position predicts the others, which is what position sharing lets the agents exploit.\n")
    sd = sv[:k95] / np.sqrt(len(flat) - 1)

    rows = []
    for s, ds in data.items():
        sh = ds.all_landmarks().astype(np.float64)
        sh = sh[~np.isnan(sh).any((1, 2))]
        z = np.stack([((_align(x, ref).reshape(-1) - mu) @ vt[:k95].T) / sd for x in sh])
        d = np.sqrt((z ** 2).sum(1))
        thr = np.sqrt(k95) + 3 * np.sqrt(0.5)  # ~ chi distribution: mean sqrt(k), sd ~ 0.7
        top = np.argsort(-d)[:3]
        ids = [ds.items[i]["id"] for i in range(len(ds)) if not np.isnan(ds.landmarks(i)).any()]
        rows.append([s, fmt(np.median(d)), int((d > thr).sum()), ", ".join(f"{ids[i]} ({d[i]:.1f})" for i in top)])
    rep.p(f"Mahalanobis distance of each skull from the training mean in the {k95}-mode space; "
          "outliers are worth a visual check for annotation errors or unusual anatomy.\n")
    rep.p(table(["Split", "Median distance", "Outliers", "Most unusual (distance)"], rows))
    rep.js["shape"] = {"modes_95": k95, "variance_first10": var[:10].tolist()}
    plt.figure(figsize=(6, 3))
    plt.plot(np.arange(1, min(20, len(var)) + 1), np.cumsum(var)[:20] * 100, "o-")
    plt.axhline(95, ls="--", c="gray", lw=1)
    plt.xlabel("number of shape modes")
    plt.ylabel("variance explained (%)")
    rep.fig("shape_pca.png", "Cumulative shape variance explained by PCA modes.")


def mean_shape_start(rep, data, env_cfg):
    rep.h("6. Mean-shape start (the RL agents' starting point)")
    train = data["train"]
    mean = np.nanmean(norm_shapes(train), 0)
    steps = env_cfg["step_px"]
    caps = env_cfg["caps"]
    l0_reach = steps[0] * caps[0] * MM
    total_reach = sum(steps[l] * caps[l] for l in env_cfg["levels"]) * MM
    dev_mean = {}
    for dev in {it.get("device") for it in train.items} - {None}:
        sel = [i for i, it in enumerate(train.items) if it.get("device") == dev]
        dev_mean[dev] = np.nanmean(norm_shapes(train)[sel], 0)
    rows, per_lm = [], {}
    for s, ds in data.items():
        if s == "train":
            continue
        size = np.array([[it["width"], it["height"]] for it in ds.items])[:, None]
        err = np.linalg.norm(mean[None] * size - ds.all_landmarks(), axis=-1) * MM
        per_lm[s] = np.nanmean(err, 0)
        row = [s, fmt(np.nanmean(err)), fmt(np.nanpercentile(err, 95)), fmt(np.nanmax(err)),
               fmt(np.nanmean(err <= 4) * 100, 1), fmt(np.nanmean(err + env_cfg["jitter_mm"] > l0_reach) * 100, 1)]
        if dev_mean:
            start = np.stack([dev_mean.get(it.get("device"), mean) for it in ds.items])
            err_d = np.linalg.norm(start * size - ds.all_landmarks(), axis=-1) * MM
            row += [fmt(np.nanmean(err_d)), fmt(np.nanmax(err_d)),
                    fmt(np.nanmean(err_d + env_cfg["jitter_mm"] > l0_reach) * 100, 1)]
        rows.append(row)
    rep.p(f"Placing every landmark at the training mean shape (scaled to the image) is the trivial baseline any "
          f"method must beat, and is where agents start. Reach at the coarsest level: {l0_reach:.0f} mm; "
          f"total over all levels: {total_reach:.0f} mm.\n")
    headers = ["Split", "MRE (mm)", "95th pct (mm)", "Max (mm)", "SDR 4 mm (%)", "Beyond L0 reach incl. jitter (%)"]
    if dev_mean:
        headers += ["MRE, per-device mean (mm)", "Max, per-device mean (mm)", "Beyond reach, per-device (%)"]
        rep.p("Per-device columns: start from the mean shape of the image's own imaging device "
              "(`env.init: mean_device`), which removes differences in field of view between devices.\n")
    rep.p(table(headers, rows))
    rep.js["mean_shape_baseline"] = {r[0]: float(r[1]) for r in rows}
    if per_lm:
        names = train.landmark_names
        plt.figure(figsize=(8, 3.5))
        w = 0.8 / len(per_lm)
        for j, (s, v) in enumerate(per_lm.items()):
            plt.bar(np.arange(len(names)) + j * w, v, w, label=s)
        plt.xticks(range(len(names)), [str(k + 1) for k in range(len(names))])
        plt.xlabel("landmark index + 1")
        plt.ylabel("start error (mm)")
        plt.legend()
        rep.fig("mean_shape_error.png", "Error of the mean-shape start per landmark.")


def augmentation_check(rep, train, aug_cfg, trials=20):
    rep.h("7. Augmentation check")
    lm = torch.tensor(train.all_landmarks())
    sizes = torch.tensor([[it["width"], it["height"]] for it in train.items], dtype=torch.float32)
    kept = []
    for _ in range(trials):
        dummy = torch.zeros(len(lm), 1, 4, 4)  # landmark maths does not depend on the image
        _, new = random_affine(dummy, lm, sizes, aug_cfg["rot_deg"], tuple(aug_cfg["scale"]), aug_cfg["shift"])
        kept.append(float((new != lm).any(-1).any(-1).float().mean()))
    rate = 1 - float(np.mean(kept))
    edge = np.min(np.concatenate([lm.numpy(), sizes[:, None].numpy() - 1 - lm.numpy()], -1), -1) * MM
    names = train.landmark_names
    close = np.argsort(np.nanmean(edge, 0))[:3]
    rep.p(f"With rotation ±{aug_cfg['rot_deg']}°, scale {aug_cfg['scale']} and shift ±{aug_cfg['shift'] * 100:.0f}%, "
          f"**{rate * 100:.1f}%** of training images would push a landmark outside the image and are therefore left "
          f"unaugmented. Above ~20%, reduce rotation or shift. Landmarks closest to the border (mean distance): "
          + ", ".join(f"{names[k]} ({np.nanmean(edge[:, k]):.1f} mm)" for k in close) + ".")
    rep.js["augmentation_fallback_rate"] = rate


def split_shift(rep, data):
    rep.h("8. Split shift (train vs test)")
    train = data["train"]
    names = train.landmark_names
    tr = norm_shapes(train)
    rows = []
    for s, ds in data.items():
        if s == "train":
            continue
        te = norm_shapes(ds)
        alpha = 0.05 / (2 * len(names))  # Bonferroni over landmarks and axes
        shifted = []
        for k, n in enumerate(names):
            for ax, a in ((0, "x"), (1, "y")):
                x, y = tr[:, k, ax], te[:, k, ax]
                p = ks_2samp(x[~np.isnan(x)], y[~np.isnan(y)]).pvalue
                if p < alpha:
                    shifted.append(f"{n} {a}")
        p_int = "–"
        if has_images(ds):
            bright_tr = [thumb(train, i, 8).mean() for i in range(0, len(train), max(1, len(train) // 40))]
            bright_te = [thumb(ds, i, 8).mean() for i in range(0, len(ds), max(1, len(ds) // 40))]
            p_int = f"{ks_2samp(bright_tr, bright_te).pvalue:.3g}"
        rows.append([s, len(shifted), ", ".join(shifted[:6]) + (" …" if len(shifted) > 6 else ""), p_int])
    rep.p("Kolmogorov–Smirnov tests on normalised landmark coordinates (Bonferroni-corrected) and on mean "
          "brightness. Shifted landmarks explain why one test split is harder than another.\n")
    rep.p(table(["Split", "Shifted coords", "Which", "Brightness KS p"], rows))


def physical_scale(rep, data):
    rep.h("9. Physical scale (Sella–Nasion distance)")
    names = next(iter(data.values())).landmark_names
    try:
        si, ni = names.index("Sella"), names.index("Nasion")
    except ValueError:
        rep.p("Dataset has no Sella/Nasion landmarks; skipped.")
        return
    groups = {}
    for s, ds in data.items():
        for i, it in enumerate(ds.items):
            lm = ds.landmarks(i)
            d = np.linalg.norm(lm[si] - lm[ni]) * MM
            groups.setdefault(it.get("device") or s, []).append(d)
    allv = np.concatenate([np.array(v) for v in groups.values()])
    med = np.nanmedian(allv)
    rows = [[g, len(v), fmt(np.median(v), 1), fmt(np.percentile(v, 5), 1), fmt(np.percentile(v, 95), 1),
             "check spacing" if abs(np.median(v) / med - 1) > 0.1 else "ok"] for g, v in groups.items()]
    rep.p("S–N length varies little between patients compared with image scale errors, so it is used here as a "
          "relative check: a group whose median is more than 10% off the overall median most likely has a wrong "
          "pixel spacing (or a very different age mix).\n")
    rep.p(table(["Group", "Images", "Median S–N (mm)", "5th pct", "95th pct", "Verdict"], rows))
    rep.js["sella_nasion_mm"] = {g: float(np.median(v)) for g, v in groups.items()}


def cross_dataset(rep, data, other_root):
    rep.h(f"10. Cross-dataset: this dataset → `{other_root}`")
    train = data["train"]
    other = {s: CephDataset(other_root, s) for s in splits_of(other_root)}
    test_split = "test" if "test" in other else list(other)[-1]
    ods = other[test_split]
    idx = map_indices(train.landmark_names, ods.landmark_names)
    pairs = [(train.landmark_names[m], ods.landmark_names[d]) for m, d in enumerate(idx) if d >= 0]
    rep.p(f"{len(pairs)} of {len(idx)} landmarks have a counterpart in `{other_root}`.\n")
    mean = np.nanmean(norm_shapes(train), 0)
    size = np.array([[it["width"], it["height"]] for it in ods.items])[:, None]
    olm = ods.all_landmarks()
    rows = []
    for m, d in enumerate(idx):
        if d < 0:
            continue
        err = np.linalg.norm(mean[m] * size[:, 0] - olm[:, d], axis=-1) * MM
        own = np.nanmean(norm_shapes(other["train"])[:, d], 0) if "train" in other else None
        own_err = np.linalg.norm(own * size[:, 0] - olm[:, d], axis=-1) * MM if own is not None else np.nan
        rows.append([train.landmark_names[m], ods.landmark_names[d], fmt(np.nanmean(err)), fmt(np.nanmean(own_err))])
    rep.p("Start error on the other dataset's test images when the agents use *this* dataset's mean shape, versus "
          "that dataset's own mean shape. A large gap means the cross-dataset test starts far off and relies on the "
          "coarse level to recover.\n")
    rep.p(table(["Landmark", "Matched to", "Start error, this mean (mm)", "Start error, own mean (mm)"], rows))

    fig, ax = plt.subplots(figsize=(5, 6))
    on = np.nanmean(norm_shapes(ods), 0)
    for m, d in enumerate(idx):
        if d >= 0:
            ax.plot([mean[m, 0], on[d, 0]], [mean[m, 1], on[d, 1]], c="gray", lw=0.8)
    ax.scatter(mean[:, 0], mean[:, 1], c="tab:blue", label="this dataset (train mean)")
    ax.scatter(on[:, 0], on[:, 1], c="tab:orange", s=12, label="other dataset (test mean)")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.legend(fontsize=7)
    rep.fig("cross_mean_shapes.png", "Mean shapes of the two datasets in normalised coordinates (lines join matched landmarks).")


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", help="prepared dataset (scripts/prepare_data.py)")
    p.add_argument("--isbi-annotations", help="ISBI annotation folder only (contains 400_junior, 400_senior)")
    p.add_argument("--out", required=True)
    p.add_argument("--compare", help="second prepared dataset for the cross-dataset section")
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "base.yaml"))
    a = p.parse_args()

    cfg = load_config(a.config)
    if a.isbi_annotations:
        ids = isbi_split_ids()
        data = {s: AnnotationsOnly(a.isbi_annotations, ids[s]) for s in ("train", "val", "test1", "test2")}
        name = f"{a.isbi_annotations} (ISBI annotations only)"
    else:
        if not a.data_root:
            sys.exit("give --data-root or --isbi-annotations")
        data = {s: CephDataset(a.data_root, s) for s in splits_of(a.data_root)}
        name = a.data_root
    if "train" not in data:
        sys.exit(f"no train split under {name}")
    rep = Report(a.out)
    rep.p(f"# Dataset analysis: `{name}`")
    inventory(rep, name, data)
    intensity(rep, data)
    landmark_layout(rep, data["train"])
    doctor_agreement(rep, data)
    shape_variation(rep, data)
    mean_shape_start(rep, data, cfg["env"])
    augmentation_check(rep, data["train"], cfg["env"]["augment"])
    split_shift(rep, data)
    physical_scale(rep, data)
    if a.compare:
        cross_dataset(rep, data, a.compare)
    rep.save()
    print(f"report: {os.path.join(a.out, 'report.md')}")


if __name__ == "__main__":
    main()
