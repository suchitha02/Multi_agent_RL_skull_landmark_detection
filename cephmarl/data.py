"""Dataset preparation and loading.

Every dataset is converted once into the same cache format:

    <out_root>/<split>/meta.json      landmark names + one entry per image
    <out_root>/<split>/<id>.npy       uint8 grayscale image resampled to 0.1 mm/px

Landmarks are stored in pixel units of the 0.1 mm/px image (so 1 px = 0.1 mm).
"""
import csv
import glob
import json
import os
import random

import numpy as np
from PIL import Image

from .landmarks import AARIZ_LANDMARKS, ISBI_LANDMARKS

MM_PER_PX = 0.1  # resolution of every cached image


# ---------------------------------------------------------------- loading
class CephDataset:
    """Cached split. Images are loaded lazily (Aariz is too big to keep in RAM)."""

    def __init__(self, root, split, cache_images=False):
        self.dir = os.path.join(root, split)
        with open(os.path.join(self.dir, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        self.landmark_names = meta["landmark_names"]
        self.items = meta["items"]
        self.cache = {} if cache_images else None

    def __len__(self):
        return len(self.items)

    @property
    def num_landmarks(self):
        return len(self.landmark_names)

    def image(self, i):
        if self.cache is not None and i in self.cache:
            return self.cache[i]
        img = np.load(os.path.join(self.dir, self.items[i]["file"]))
        if self.cache is not None:
            self.cache[i] = img
        return img

    def landmarks(self, i, key="landmarks"):
        v = self.items[i].get(key)
        return None if v is None else np.asarray(v, dtype=np.float32)

    def all_landmarks(self, key="landmarks"):
        return np.stack([self.landmarks(i, key) for i in range(len(self))])


# ---------------------------------------------------------------- writing
def _write_split(out_dir, names, entries):
    """entries: list of (id, uint8 image at 0.1 mm/px, dict of landmark arrays + extras)."""
    os.makedirs(out_dir, exist_ok=True)
    items = []
    for img_id, img, extra in entries:
        np.save(os.path.join(out_dir, f"{img_id}.npy"), img)
        item = {"id": img_id, "file": f"{img_id}.npy", "height": int(img.shape[0]), "width": int(img.shape[1])}
        for k, v in extra.items():
            item[k] = v.tolist() if isinstance(v, np.ndarray) else v
        items.append(item)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"landmark_names": names, "items": items}, f)
    print(f"wrote {len(items)} images to {out_dir}")


def _to_gray8(img):
    """Any PIL image (8-bit, RGB, 16-bit) -> 8-bit grayscale. 16-bit data is rescaled, not clipped."""
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        a = np.asarray(img, dtype=np.float32)
        lo, hi = np.percentile(a, [0.5, 99.5])
        return Image.fromarray(np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8))
    return img.convert("L")


def normalize_intensity(img, background=8, low=1.0, high=99.0):
    """Stretch grey levels so the 1st-99th percentile of the non-black pixels spans 0-255.

    Devices differ a lot in brightness and contrast, and many images have black padding; ignoring
    near-black pixels keeps the padding from dominating the percentiles.
    """
    a = img.astype(np.float32)
    fg = a[a > background]
    if fg.size < 100:
        return img
    lo, hi = np.percentile(fg, [low, high])
    return np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)


def _load_resampled(path, spacing_mm):
    """Load as 8-bit grayscale, resample so that 1 px = 0.1 mm, normalise intensity. Returns (image, scale)."""
    img = _to_gray8(Image.open(path))
    scale = spacing_mm / MM_PER_PX
    if abs(scale - 1.0) > 1e-6:
        w, h = img.size
        img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    return normalize_intensity(np.asarray(img, dtype=np.uint8)), scale


# ---------------------------------------------------------------- ISBI 2015
def _read_isbi_txt(path):
    with open(path, encoding="utf-8") as f:
        rows = [line.strip() for line in f if line.strip()]
    pts = [list(map(float, r.replace(" ", "").split(",")[:2])) for r in rows[:19]]
    return np.array(pts, dtype=np.float32)


def find_dir(root, name):
    """First folder called `name` (any case) anywhere under root, or None."""
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if d.lower() == name.lower():
                return os.path.join(dirpath, d)
    return None


ISBI_IMAGE_FOLDERS = {"train": "TrainingData", "test1": "Test1Data", "test2": "Test2Data"}
ISBI_OFFICIAL_RANGES = {"train": range(1, 151), "test1": range(151, 301), "test2": range(301, 401)}


def isbi_split_ids(n_val=20, seed=0):
    """Image ids per split: official ranges, with n_val training images moved to 'val' (seeded)."""
    ids = {s: [f"{i:03d}" for i in r] for s, r in ISBI_OFFICIAL_RANGES.items()}
    random.Random(seed).shuffle(ids["train"])
    ids["val"], ids["train"] = sorted(ids["train"][:n_val]), sorted(ids["train"][n_val:])
    return ids


def isbi_annotations(raw_root):
    """(junior dir, senior dir), found anywhere under raw_root (with or without an AnnotationsByMD parent)."""
    jun, sen = find_dir(raw_root, "400_junior"), find_dir(raw_root, "400_senior")
    if not (jun and sen):
        raise FileNotFoundError(f"400_junior / 400_senior folders not found under {raw_root}")
    return jun, sen


def prepare_isbi(raw_root, out_root, n_val=20, seed=0):
    """Official release layout (folders are searched for, so extra nesting is fine):
        .../RawImage/{TrainingData,Test1Data,Test2Data}/001.bmp ...   (or all 400 images in one folder)
        .../400_junior/001.txt, .../400_senior/001.txt               (19 "x,y" lines, then class labels)
    Images are already 0.1 mm/px. Ground truth = mean of junior and senior.
    Splits follow the official numbering: 001-150 train (n_val moved to val), 151-300 test1, 301-400 test2.
    """
    jun_dir, sen_dir = isbi_annotations(raw_root)
    images = {}
    for folder in list(ISBI_IMAGE_FOLDERS.values()) + ["RawImage"]:
        d = find_dir(raw_root, folder)
        if d:
            for p in glob.glob(os.path.join(d, "*.*")):
                if p.lower().endswith((".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                    images[os.path.splitext(os.path.basename(p))[0]] = p
    if len(images) < 400:
        print(f"warning: found {len(images)} of 400 images; missing ids are skipped")
    for split, ids in isbi_split_ids(n_val, seed).items():
        entries = []
        for img_id in ids:
            if img_id not in images:
                continue
            img, _ = _load_resampled(images[img_id], MM_PER_PX)
            jun = _read_isbi_txt(os.path.join(jun_dir, f"{img_id}.txt"))
            sen = _read_isbi_txt(os.path.join(sen_dir, f"{img_id}.txt"))
            entries.append((img_id, img, {"landmarks": (jun + sen) / 2, "junior": jun, "senior": sen}))
        _write_split(os.path.join(out_root, split), ISBI_LANDMARKS, entries)


# ---------------------------------------------------------------- Aariz
def _read_aariz_json(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    pts = np.full((len(AARIZ_LANDMARKS), 2), np.nan, dtype=np.float32)
    for i, lm in enumerate(data["landmarks"]):
        title = lm.get("title")
        j = AARIZ_LANDMARKS.index(title) if title in AARIZ_LANDMARKS else i
        pts[j] = [lm["value"]["x"], lm["value"]["y"]]
    return pts


def _read_spacing_csv(path):
    """CSV with one row per image: id/name/file, pixel size (mm/px) and, if present, the device.

    The Aariz release ships such a CSV (device and pixel size per cephalogram).
    Returns {image id: (spacing_mm, device or None)}.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [{k.strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]
    cols = list(rows[0].keys())
    dev_col = next((c for c in cols if any(k in c.lower() for k in ("device", "machine", "source"))), None)
    sp_col = next(c for c in cols if any(k in c.lower() for k in ("pixel", "spacing", "size", "resolution")))
    id_col = next(c for c in cols if c not in (dev_col, sp_col)
                  and any(k in c.lower() for k in ("id", "name", "file")))
    print(f"spacing CSV columns: id={id_col!r} spacing={sp_col!r} device={dev_col!r}")
    return {os.path.splitext(os.path.basename(r[id_col]))[0]: (float(r[sp_col]), r[dev_col] if dev_col else None)
            for r in rows}


def _find(parent, name):
    """Sub-folder or file of `parent` named `name`, ignoring case (releases differ: train vs Train)."""
    for entry in os.listdir(parent):
        if entry.lower() == name.lower():
            return os.path.join(parent, entry)
    raise FileNotFoundError(f"no '{name}' in {parent} (found: {sorted(os.listdir(parent))})")


# How the pixel size of each Aariz device is set. Evidence (scripts/diagnose_aariz.py, Nov 2026):
#  - csv:           Sella-Nasion with the CSV pixel size is 71-73 mm, as expected -> CSV is right.
#  - native_height: images were downscaled by varying factors (heights 538-2234 px) but the physical
#                   image height stays ~215 mm, so the CSV pixel size holds at the largest (native)
#                   height and scales with it: spacing = csv * native_height / height.
#  - sella_nasion:  all images share one size and the CSV pixel size gives Sella-Nasion of 30, 45 or
#                   118 mm. No metadata explains it, so the pixel size is calibrated so that the device's
#                   median Sella-Nasion on the TRAIN split equals that of the 'csv' devices (an estimate,
#                   roughly +-10%: it assumes a similar age mix across devices).
AARIZ_SPACING_RULES = {
    "ART Plus": "csv", "ProMax 2D": "csv", "ProMax with ProTouch": "csv",
    "Hyperion X5": "native_height",
    "Smart3D": "sella_nasion", "Veraviewepocs 2D": "sella_nasion", "Rotograph EVO": "sella_nasion",
}


def _aariz_index(raw_root, spacing):
    """One record per image: paths, device, CSV spacing, pixel height and Sella-Nasion in raw pixels."""
    sella, nasion = AARIZ_LANDMARKS.index("Sella"), AARIZ_LANDMARKS.index("Nasion")
    recs = []
    for split, folder in {"train": "train", "val": "valid", "test": "test"}.items():
        split_dir = _find(raw_root, folder)
        img_dir = _find(split_dir, "Cephalograms")
        ann = _find(_find(split_dir, "Annotations"), "Cephalometric Landmarks")
        sen_dir, jun_dir = _find(ann, "Senior Orthodontists"), _find(ann, "Junior Orthodontists")
        for p in sorted(glob.glob(os.path.join(img_dir, "*.*"))):
            img_id = os.path.splitext(os.path.basename(p))[0]
            sp, device = spacing.get(img_id, (None, None))
            sen = _read_aariz_json(os.path.join(sen_dir, f"{img_id}.json"))
            jun = _read_aariz_json(os.path.join(jun_dir, f"{img_id}.json"))
            gt = (sen + jun) / 2
            with Image.open(p) as im:
                height = im.size[1]
            recs.append({"split": split, "id": img_id, "path": p, "device": device, "csv_spacing": sp,
                         "height": height, "sen": sen, "jun": jun,
                         "sn_px": float(np.linalg.norm(gt[sella] - gt[nasion]))})
    return recs


def _aariz_spacings(recs, rules):
    """Corrected pixel size per image (see AARIZ_SPACING_RULES). Calibration uses the train split only."""
    train = [r for r in recs if r["split"] == "train"]
    ref = [r["sn_px"] * r["csv_spacing"] for r in train if rules.get(r["device"], "csv") == "csv"]
    ref_mm = float(np.median(ref))
    out, report = {}, []
    for dev in sorted({r["device"] for r in recs}):
        rule = rules.get(dev, "csv")
        dev_recs = [r for r in recs if r["device"] == dev]
        if rule == "native_height":
            native = max(r["height"] for r in dev_recs)
            for r in dev_recs:
                out[r["id"]] = r["csv_spacing"] * native / r["height"]
            note = f"csv {dev_recs[0]['csv_spacing']} at {native} px height, scaled per image"
        elif rule == "sella_nasion":
            sp = ref_mm / float(np.median([r["sn_px"] for r in train if r["device"] == dev]))
            for r in dev_recs:
                out[r["id"]] = sp
            note = f"calibrated {sp:.4f} mm/px (csv said {dev_recs[0]['csv_spacing']})"
        else:
            for r in dev_recs:
                out[r["id"]] = r["csv_spacing"]
            note = f"csv {dev_recs[0]['csv_spacing']}"
        sn = np.median([r["sn_px"] * out[r["id"]] for r in dev_recs])
        report.append(f"  {dev:22s} {rule:13s} {note}  -> median Sella-Nasion {sn:.1f} mm")
    print(f"Aariz pixel sizes (reference Sella-Nasion {ref_mm:.1f} mm from the 'csv' devices, train split):")
    print("\n".join(report))
    return out


def prepare_aariz(raw_root, out_root, spacing_csv=None, default_spacing=None, rules=None, gross_mm=10.0):
    """Expected layout (figshare release; folder-name case does not matter):
        raw_root/{train,valid,test}/Cephalograms/<id>.<jpg|png|bmp>
        raw_root/{train,valid,test}/Annotations/Cephalometric Landmarks/{Senior,Junior} Orthodontists/<id>.json
        raw_root/cephalogram_machine_mappings.csv   (device + pixel size per image; found automatically)
    raw_root may also be the folder that contains 'Aariz/'.

    Pixel sizes are corrected per device (AARIZ_SPACING_RULES). Each item also stores the CSV pixel size
    as `csv_spacing_mm`, so errors can be reported both ways: corrected, and with the official CSV
    pixel sizes that published Aariz results use.

    Ground truth is the official mean of junior and senior marks. Aariz seniors reviewed and corrected
    the juniors' marks, so where the two differ by more than `gross_mm` the junior mark is a mistake
    (e.g. a click at the image edge) and averaging would keep half of it: in train and val the senior
    mark is used instead. Test keeps the official mean so results stay comparable with published work.
    """
    if not any(e.lower() == "train" for e in os.listdir(raw_root)):
        raw_root = _find(raw_root, "Aariz")
    if spacing_csv is None:
        try:
            spacing_csv = _find(raw_root, "cephalogram_machine_mappings.csv")
        except FileNotFoundError:
            pass
    spacing = _read_spacing_csv(spacing_csv) if spacing_csv else {}
    recs = _aariz_index(raw_root, spacing)
    for r in recs:
        if r["csv_spacing"] is None:
            if default_spacing is None:
                raise ValueError(f"no pixel spacing for {r['id']}: pass --spacing-csv or --default-spacing")
            r["csv_spacing"] = default_spacing
    corrected = _aariz_spacings(recs, AARIZ_SPACING_RULES if rules is None else rules)

    n_gross = {"train": 0, "val": 0, "test": 0}
    for split in ("train", "val", "test"):
        entries = []
        for r in (r for r in recs if r["split"] == split):
            sp = corrected[r["id"]]
            img, scale = _load_resampled(r["path"], sp)
            jun, sen = r["jun"] * scale, r["sen"] * scale
            gt = (jun + sen) / 2
            gross = np.linalg.norm(jun - sen, axis=-1) * MM_PER_PX > gross_mm
            if split != "test":
                gt[gross] = sen[gross]
            n_gross[split] += int(gross.sum())
            entries.append((r["id"], img, {"landmarks": gt, "junior": jun, "senior": sen,
                                           "spacing_mm": sp, "csv_spacing_mm": r["csv_spacing"],
                                           "device": r["device"], "gross_disagreement": np.where(gross)[0].tolist()}))
        _write_split(os.path.join(out_root, split), AARIZ_LANDMARKS, entries)
    print(f"junior/senior differ by > {gross_mm} mm at {n_gross} points; senior mark used in train/val, "
          f"official mean kept in test")


# ---------------------------------------------------------------- synthetic
def make_synthetic(out_root, num_landmarks=19, sizes=(40, 10, 10), shape=(640, 520), seed=0, names=None):
    """Small fake dataset in the cache format, for tests and smoke runs only.

    Each landmark sits at the centre of a ring whose radius depends on its index,
    on a noisy background; shapes vary around a fixed mean, like real skulls do.
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    mean = np.stack([rng.uniform(0.15 * w, 0.85 * w, num_landmarks),
                     rng.uniform(0.15 * h, 0.85 * h, num_landmarks)], 1)
    basis = rng.normal(size=(3, num_landmarks, 2)) * 15  # shared shape variation
    yy, xx = np.mgrid[0:h, 0:w]
    names = names or [f"L{i}" for i in range(num_landmarks)]
    for split, n in zip(("train", "val", "test"), sizes):
        entries = []
        for k in range(n):
            pts = mean + np.tensordot(rng.normal(size=3), basis, 1) + rng.normal(size=(num_landmarks, 2)) * 3
            pts += rng.uniform(-30, 30, size=2)  # whole-head offset
            pts = np.clip(pts, 10, [w - 10, h - 10])
            img = rng.normal(60, 12, size=shape)
            for i, (x, y) in enumerate(pts):
                r = np.hypot(xx - x, yy - y)
                img += 120 * np.exp(-((r - (6 + 2 * i)) ** 2) / 4) + 80 * np.exp(-(r ** 2) / 8)
            img = np.clip(img, 0, 255).astype(np.uint8)
            jun = pts + rng.normal(size=pts.shape) * 2
            sen = pts + rng.normal(size=pts.shape) * 2
            entries.append((f"{split}_{k:03d}", img, {"landmarks": pts.astype(np.float32),
                                                      "junior": jun, "senior": sen,
                                                      "spacing_mm": MM_PER_PX, "device": f"synthetic-{k % 3}"}))
        _write_split(os.path.join(out_root, split), names, entries)
