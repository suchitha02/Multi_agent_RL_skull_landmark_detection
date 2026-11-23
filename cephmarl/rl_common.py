"""Pieces shared by DQN, MAPPO and COMA: batching, episodes, validation, checkpoints."""
import copy

import numpy as np
import torch

from .env import CephEnv
from .flagging import ShapeModel
from .metrics import radial_errors


def mean_shape(ds):
    """(K,2) mean landmark position normalised by image size."""
    shapes = [ds.landmarks(i) / np.array([it["width"], it["height"]]) for i, it in enumerate(ds.items)]
    return np.nanmean(np.stack(shapes), 0).astype(np.float32)


def mean_shapes(ds):
    """{device: mean normalised shape} for every device in the split, plus "__all__"."""
    out = {"__all__": mean_shape(ds)}
    for dev in {it.get("device") for it in ds.items} - {None}:
        shapes = [ds.landmarks(i) / np.array([it["width"], it["height"]])
                  for i, it in enumerate(ds.items) if it.get("device") == dev]
        out[dev] = np.nanmean(np.stack(shapes), 0).astype(np.float32)
    return out


def devices_of(ds, idx):
    return [ds.items[i].get("device") for i in idx]


def sample_batch(ds, batch, rng):
    idx = rng.choice(len(ds), size=batch, replace=len(ds) < batch)
    return [ds.image(i) for i in idx], np.stack([ds.landmarks(i) for i in idx]), devices_of(ds, idx)


def make_env(cfg, train_ds, device):
    return CephEnv(cfg["env"], train_ds.num_landmarks, mean_shapes(train_ds), device)


def obs_to(obs, device):
    return {k: v.to(device) for k, v in obs.items()}


@torch.no_grad()
def run_episode(env, policy, images, landmarks=None, train=False, devices=None):
    """policy(obs) -> (actions (B,K), confidence (B,K)). Returns predictions and uncertainty features."""
    obs = env.reset(images, landmarks, train=train, devices=devices)
    fine = len(env.levels) - 1
    conf_hist = torch.full((env.B, env.K, 8), float("nan"), device=env.dev)
    steps = torch.zeros(env.B, env.K, device=env.dev)
    while not env.done.all():
        actions, conf = policy(obs)
        at_fine = (env.lvl == fine) & ~env.done
        conf_hist = torch.where(at_fine[..., None], torch.cat([conf[..., None], conf_hist[..., :-1]], -1), conf_hist)
        steps += ~env.done
        obs, _, _, _ = env.step(actions)
    return {"pred": env.prediction().cpu().numpy(), "spread": env.spread_mm().cpu().numpy(),
            "conf": torch.nanmean(conf_hist, -1).cpu().numpy(), "steps": steps.cpu().numpy()}


def predict_dataset(env, policy, ds, batch=8, gt_index=None):
    """Greedy predictions for a whole split. gt_index maps model landmarks to dataset landmarks."""
    outs = {"pred": [], "spread": [], "conf": [], "steps": []}
    for s in range(0, len(ds), batch):
        idx = list(range(s, min(s + batch, len(ds))))
        res = run_episode(env, policy, [ds.image(i) for i in idx], devices=devices_of(ds, idx))
        for k in outs:
            outs[k].append(res[k])
    out = {k: np.concatenate(v) for k, v in outs.items()}
    out["gt"] = gather_gt(ds, "landmarks", gt_index)
    return out


def gather_gt(ds, key, gt_index=None):
    """(N,K,2) ground truth in model landmark order; NaN where the dataset lacks a landmark."""
    gts = []
    for i in range(len(ds)):
        lm = ds.landmarks(i, key)
        if lm is None:
            return None
        if gt_index is not None:
            full = np.full((len(gt_index), 2), np.nan, dtype=np.float32)
            for m, d in enumerate(gt_index):
                if d >= 0:
                    full[m] = lm[d]
            lm = full
        gts.append(lm)
    return np.stack(gts)


def validate(env, policy, val_ds):
    out = predict_dataset(env, policy, val_ds)
    return float(np.nanmean(radial_errors(out["pred"], out["gt"])))


def checkpoint(algo, cfg, train_ds, env, **state_dicts):
    shape = ShapeModel.fit(train_ds.all_landmarks())
    return {"algo": algo, "cfg": cfg, "landmark_names": train_ds.landmark_names,
            "mean_shape": {k: v.cpu().numpy() for k, v in env.mean_shapes.items()}, "shape_model": shape.state(),
            **{k: copy.deepcopy(v.state_dict()) for k, v in state_dicts.items()}}  # snapshot, not live refs


def checkpoint_unet(cfg, train_ds, model):
    shape = ShapeModel.fit(train_ds.all_landmarks())
    return {"algo": "unet", "cfg": cfg, "landmark_names": train_ds.landmark_names,
            "mean_shape": mean_shape(train_ds), "shape_model": shape.state(), "model": copy.deepcopy(model.state_dict())}
