"""Save prepared images with their landmarks drawn (numbered), e.g. to inspect suspected annotation errors.

    python scripts/show_images.py --data-root data/aariz --ids cks2ip8g62b7q0yufcezt50ch cl76jfqyc0e2x07610em2fb9f
"""
import argparse
import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cephmarl.data import CephDataset  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data-root", required=True)
p.add_argument("--ids", nargs="+", required=True)
p.add_argument("--out", default="inspect")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
for split in ("train", "val", "test", "test1", "test2"):
    if not os.path.exists(os.path.join(a.data_root, split, "meta.json")):
        continue
    ds = CephDataset(a.data_root, split)
    for i, it in enumerate(ds.items):
        if it["id"] not in a.ids:
            continue
        im = Image.fromarray(ds.image(i)).convert("RGB")
        dr = ImageDraw.Draw(im)
        for key, col in (("junior", (0, 160, 255)), ("senior", (255, 160, 0)), ("landmarks", (255, 0, 0))):
            lm = ds.landmarks(i, key)
            if lm is None:
                continue
            for k, (x, y) in enumerate(lm):
                dr.ellipse([x - 8, y - 8, x + 8, y + 8], outline=col, width=3)
                if key == "landmarks":
                    dr.text((x + 10, y - 10), str(k + 1), fill=col)
        im.thumbnail((1200, 1200))
        path = os.path.join(a.out, f"{split}_{it['id']}.png")
        im.save(path)
        print(f"{path}  (red = ground truth with landmark number, blue = junior, orange = senior)")
