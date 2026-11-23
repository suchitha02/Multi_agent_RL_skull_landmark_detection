"""U-Net heatmap baseline: same data, splits and augmentation as the agents."""
import numpy as np
import torch
import torch.nn.functional as F

from .augment import augment
from .metrics import radial_errors
from .models import build_unet
from .rl_common import gather_gt


def to_input(images, out_hw, device):
    """Resize each image to fit inside out_hw (keeping aspect ratio), pad. Returns (B,1,H,W), scales (B,)."""
    H, W = out_hw
    x = torch.zeros(len(images), 1, H, W, device=device)
    scales = []
    for b, im in enumerate(images):
        s = min(H / im.shape[0], W / im.shape[1])
        t = torch.from_numpy(im).to(device).float()[None, None] / 255
        t = F.interpolate(t, size=(int(im.shape[0] * s), int(im.shape[1] * s)), mode="area")
        x[b, :, : t.shape[2], : t.shape[3]] = t[0]
        scales.append(s)
    return x, torch.tensor(scales, device=device)


def heatmaps(pts, hw, sigma):
    """pts (B,K,2) in input pixels -> Gaussian heatmaps (B,K,H,W); NaN landmarks give zeros."""
    H, W = hw
    ys = torch.arange(H, device=pts.device).float()[None, None, :, None]
    xs = torch.arange(W, device=pts.device).float()[None, None, None, :]
    px, py = pts[..., 0, None, None], pts[..., 1, None, None]
    h = torch.exp(-((xs - px) ** 2 + (ys - py) ** 2) / (2 * sigma ** 2))
    return torch.nan_to_num(h)


def heatmap_loss(out, target):
    """Spatial cross-entropy: each landmark's map is a softmax over pixels, matched to the
    normalised Gaussian target. Unlike pixel-wise MSE (which, at 800x640, is minimised by
    predicting zero everywhere and leaves spurious peaks), probability mass on a wrong peak
    directly costs mass at the right one. In tests it localised in ~100 steps vs >300 for MSE."""
    B, K, H, W = out.shape
    t = target.reshape(B, K, -1)
    t = t / t.sum(-1, keepdim=True).clamp(min=1e-8)
    return -(t * F.log_softmax(out.reshape(B, K, -1), -1)).sum(-1).mean()


def probs(out):
    """Per-landmark softmax over pixels, shaped like the heatmaps."""
    B, K, H, W = out.shape
    return F.softmax(out.reshape(B, K, -1), -1).reshape(B, K, H, W)


def decode(heat, win=2):
    """Argmax, then soft-argmax in a (2*win+1)^2 window. Returns (B,K,2) coords and peak heights (B,K)."""
    B, K, H, W = heat.shape
    flat = heat.reshape(B, K, -1)
    peak, idx = flat.max(-1)
    cy, cx = idx // W, idx % W
    d = torch.arange(-win, win + 1, device=heat.device)
    yy = (cy[..., None, None] + d[:, None]).clamp(0, H - 1)
    xx = (cx[..., None, None] + d[None, :]).clamp(0, W - 1)
    bi = torch.arange(B, device=heat.device)[:, None, None, None]
    ki = torch.arange(K, device=heat.device)[None, :, None, None]
    w = heat[bi, ki, yy, xx].clamp(min=0) + 1e-8
    x = (w * xx).sum((-1, -2)) / w.sum((-1, -2))
    y = (w * yy).sum((-1, -2)) / w.sum((-1, -2))
    return torch.stack([x, y], -1), peak


@torch.no_grad()
def predict_dataset(model, ds, cfg, device, gt_index=None):
    c = cfg["unet"]
    model.eval()
    preds, confs = [], []
    for s in range(0, len(ds), c["batch"]):
        idx = list(range(s, min(s + c["batch"], len(ds))))
        x, scales = to_input([ds.image(i) for i in idx], c["input_hw"], device)
        pts, peak = decode(probs(model(x)))
        preds.append((pts / scales[:, None, None]).cpu().numpy())
        confs.append(peak.cpu().numpy())
    pred = np.concatenate(preds)
    return {"pred": pred, "conf": np.concatenate(confs), "spread": np.zeros(pred.shape[:2]),
            "steps": np.zeros(pred.shape[:2]), "gt": gather_gt(ds, "landmarks", gt_index)}


def train(cfg, train_ds, val_ds, logger, device):
    from .rl_common import checkpoint_unet
    c = cfg["unet"]
    K = train_ds.num_landmarks
    model = build_unet(K, c["encoder"], c["encoder_weights"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=c["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, c["epochs"])
    rng = np.random.default_rng(cfg["seed"])
    best, best_ckpt = float("inf"), None
    for epoch in range(1, c["epochs"] + 1):
        model.train()
        order = rng.permutation(len(train_ds))
        for s in range(0, len(order), c["batch"]):
            idx = order[s:s + c["batch"]]
            x, scales = to_input([train_ds.image(i) for i in idx], c["input_hw"], device)
            pts = torch.as_tensor(np.stack([train_ds.landmarks(i) for i in idx]), device=device) * scales[:, None, None]
            sizes = torch.tensor([[train_ds.items[i]["width"], train_ds.items[i]["height"]] for i in idx],
                                 dtype=torch.float32, device=device) * scales[:, None]
            if cfg["env"]["augment"]["enabled"]:
                x, pts = augment(x, pts, sizes, cfg["env"]["augment"])
            target = heatmaps(pts, c["input_hw"], c["sigma"])
            out = model(x)
            loss = heatmap_loss(out, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if epoch % c["eval_every"] == 0 or epoch == c["epochs"]:
            out = predict_dataset(model, val_ds, cfg, device)
            val = float(np.nanmean(radial_errors(out["pred"], out["gt"])))
            logger.log(epoch=epoch, loss=loss.item(), val_MRE_mm=val)
            if val < best:
                best = val
                best_ckpt = checkpoint_unet(cfg, train_ds, model)
    return best_ckpt, best
