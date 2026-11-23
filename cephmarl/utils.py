import copy
import json
import os
import random
import time

import numpy as np
import torch
import yaml


def load_config(path, overrides=()):
    """YAML config with optional `base:` inheritance and key.sub=value overrides."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:
        base = load_config(os.path.join(os.path.dirname(path), cfg.pop("base")))
        cfg = deep_merge(base, cfg)
    for ov in overrides:
        key, val = ov.split("=", 1)
        node = cfg
        *parents, last = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(val)
    return cfg


def deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name="auto"):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


class Logger:
    """Prints and appends JSON lines to <out_dir>/log.jsonl."""

    def __init__(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, "log.jsonl")
        self.t0 = time.time()

    def log(self, **kv):
        kv["time_s"] = round(time.time() - self.t0, 1)
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in kv.items()), flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(kv) + "\n")
