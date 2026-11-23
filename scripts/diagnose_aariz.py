"""Check Aariz pixel spacing per device, straight from the raw download.

    python scripts/diagnose_aariz.py --raw raw/Aariz --out diagnose_aariz

The dataset analysis found Sella-Nasion lengths of ~24-118 mm for four devices (others ~72 mm),
so the CSV pixel size does not fit those image files. This script prints, per device:
the CSV pixel sizes, the image sizes in pixels, Sella-Nasion in pixels and in mm, and the pixel
size that would make Sella-Nasion match the reliable devices. It also saves overlays of sample
and suspicious images. Send back the whole output folder.
"""
import argparse
import collections
import csv
import glob
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cephmarl.data import _find, _read_aariz_json  # noqa: E402
from cephmarl.landmarks import AARIZ_LANDMARKS  # noqa: E402

SELLA, NASION = AARIZ_LANDMARKS.index("Sella"), AARIZ_LANDMARKS.index("Nasion")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, help="unzipped Aariz folder (train/valid/test + CSV)")
    p.add_argument("--out", default="diagnose_aariz")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    root = a.raw if any(e.lower() == "train" for e in os.listdir(a.raw)) else _find(a.raw, "Aariz")
    lines = []

    def say(text=""):
        print(text)
        lines.append(str(text))

    # --- readme + raw csv ------------------------------------------------------------
    for name in ("Readme.txt", "readme.txt", "README.txt"):
        if os.path.exists(os.path.join(root, name)):
            say(f"===== {name} =====")
            with open(os.path.join(root, name), encoding="utf-8", errors="replace") as f:
                say(f.read()[:4000])
            break
    csv_path = _find(root, "cephalogram_machine_mappings.csv")
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    say("===== CSV header + first rows =====")
    say(", ".join(rows[0].keys()))
    for r in rows[:5]:
        say(", ".join(r.values()))
    meta = {os.path.splitext(os.path.basename(r["cephalogram_id"]))[0]: r for r in rows}

    # --- per image table -------------------------------------------------------------
    table = []
    for split in ("train", "valid", "test"):
        sd = _find(root, split)
        img_dir = _find(sd, "Cephalograms")
        ann = _find(_find(sd, "Annotations"), "Cephalometric Landmarks")
        sen_dir = _find(ann, "Senior Orthodontists")
        for path in sorted(glob.glob(os.path.join(img_dir, "*.*"))):
            iid = os.path.splitext(os.path.basename(path))[0]
            with Image.open(path) as im:
                w, h = im.size
                mode = im.mode
                dpi = im.info.get("dpi")
            lm = _read_aariz_json(os.path.join(sen_dir, f"{iid}.json"))
            r = meta.get(iid, {})
            table.append({"id": iid, "split": split, "device": r.get("machine", "?"),
                          "csv_pixel_size": float(r.get("pixel_size", "nan") or "nan"),
                          "width_px": w, "height_px": h, "mode": mode, "dpi": dpi, "file": path,
                          "sn_px": float(np.linalg.norm(lm[SELLA] - lm[NASION])),
                          "x_max": float(np.nanmax(lm[:, 0])), "y_max": float(np.nanmax(lm[:, 1]))})
    with open(os.path.join(a.out, "aariz_image_table.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        wr.writeheader()
        wr.writerows(table)

    # --- per device summary --------------------------------------------------------------
    by_dev = collections.defaultdict(list)
    for t in table:
        by_dev[t["device"]].append(t)
    med_mm = {d: np.median([t["sn_px"] * t["csv_pixel_size"] for t in ts]) for d, ts in by_dev.items()}
    reliable = [v for v in med_mm.values() if 60 <= v <= 85]
    ref = float(np.median(reliable)) if reliable else 72.0
    say(f"\n===== Per device (reference Sella-Nasion = {ref:.1f} mm, median of devices in 60-85 mm) =====")
    for d, ts in sorted(by_dev.items()):
        sizes = collections.Counter((t["width_px"], t["height_px"]) for t in ts)
        sn = np.array([t["sn_px"] for t in ts])
        sp = sorted({t["csv_pixel_size"] for t in ts})
        say(f"\n{d}: {len(ts)} images, CSV pixel size {sp}, modes {sorted({t['mode'] for t in ts})}, "
            f"dpi {sorted({str(t['dpi']) for t in ts})[:3]}")
        say(f"  image sizes (w x h: count): " + ", ".join(f"{w}x{h}: {c}" for (w, h), c in sizes.most_common(8)))
        say(f"  Sella-Nasion: median {np.median(sn):.0f} px (5-95%: {np.percentile(sn, 5):.0f}-{np.percentile(sn, 95):.0f})"
            f" -> {med_mm[d]:.1f} mm with CSV pixel size; pixel size that gives {ref:.0f} mm: {ref / np.median(sn):.4f}")
        for (w, h), c in sizes.most_common(4):
            grp = np.array([t["sn_px"] for t in ts if (t["width_px"], t["height_px"]) == (w, h)])
            say(f"    size {w}x{h} ({c} imgs): S-N median {np.median(grp):.0f} px -> implied pixel size "
                f"{ref / np.median(grp):.4f} mm")

    # --- overlays: 2 per device + the 6 smallest images ------------------------------------
    picks = []
    for d, ts in by_dev.items():
        picks += [ts[0], ts[len(ts) // 2]]
    picks += sorted(table, key=lambda t: t["width_px"] * t["height_px"])[:6]
    for t in picks:
        with Image.open(t["file"]) as im:
            im = im.convert("RGB")
        sd = _find(root, t["split"])
        ann = _find(_find(_find(sd, "Annotations"), "Cephalometric Landmarks"), "Senior Orthodontists")
        lm = _read_aariz_json(os.path.join(ann, f"{t['id']}.json"))
        dr = ImageDraw.Draw(im)
        r = max(2, im.size[0] // 200)
        for k, (x, y) in enumerate(lm):
            dr.ellipse([x - r, y - r, x + r, y + r], fill=(255, 0, 0) if k not in (SELLA, NASION) else (0, 255, 0))
        im.thumbnail((800, 800))
        im.save(os.path.join(a.out, f"{t['device'].replace(' ', '_')}_{t['width_px']}x{t['height_px']}_{t['id']}.png"))

    with open(os.path.join(a.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n-> {a.out}/ (summary.txt, aariz_image_table.csv, overlay PNGs). Zip and send this folder.")


if __name__ == "__main__":
    main()
