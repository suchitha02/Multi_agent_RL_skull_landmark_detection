"""Random affine + intensity augmentation on GPU tensors, applied to images and landmarks together."""
import math

import torch
import torch.nn.functional as F


def random_affine(img, pts, sizes, rot_deg=10, scale=(0.9, 1.1), shift=0.05):
    """img: (B,1,H,W) in [0,1]; pts: (B,K,2) pixel (x,y); sizes: (B,2) valid (W,H).

    Each image is rotated/scaled about the centre of its valid area and shifted:
        p_out = c + s R (p_in - c) + t
    """
    b, _, h, w = img.shape
    dev = img.device
    ang = torch.empty(b, device=dev).uniform_(-rot_deg, rot_deg) * math.pi / 180
    s = torch.empty(b, device=dev).uniform_(*scale)
    t = (torch.rand(b, 2, device=dev) * 2 - 1) * shift * sizes
    c = sizes / 2
    cos, sin = torch.cos(ang), torch.sin(ang)
    fwd = torch.stack([torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2) * s[:, None, None]
    inv = torch.linalg.inv(fwd)  # (B,2,2)

    # Input pixel location for every output pixel: p_in = c + inv (p_out - c - t).
    # Convert to grid_sample's normalised coordinates (align_corners=False).
    scl = torch.tensor([w / 2, h / 2], device=dev)
    off = torch.tensor([w / 2 - 0.5, h / 2 - 0.5], device=dev)
    a = inv / scl[None, :, None] * scl[None, None, :]                        # S^-1 inv S
    bvec = c + torch.einsum("bij,bj->bi", inv, off - c - t) - off             # pixel offset
    theta = torch.cat([a, (bvec / scl)[..., None]], -1)
    grid = F.affine_grid(theta, img.shape, align_corners=False)
    out = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    new_pts = c[:, None] + torch.einsum("bij,bkj->bki", fwd, pts - c[:, None]) + t[:, None]
    # keep an image unaugmented if the transform would push any landmark out of it
    outside = ((new_pts < 0) | (new_pts > sizes[:, None] - 1)).any(-1).any(-1)
    out = torch.where(outside[:, None, None, None], img, out)
    new_pts = torch.where(outside[:, None, None], pts, new_pts)
    return out, new_pts


def random_intensity(img, gamma=(0.7, 1.5), contrast=0.2, brightness=0.1, noise=0.02):
    b = img.shape[0]
    dev = img.device
    g = torch.empty(b, 1, 1, 1, device=dev).uniform_(*gamma)
    ct = 1 + torch.empty(b, 1, 1, 1, device=dev).uniform_(-contrast, contrast)
    br = torch.empty(b, 1, 1, 1, device=dev).uniform_(-brightness, brightness)
    out = img.clamp(1e-6, 1) ** g
    out = (out - 0.5) * ct + 0.5 + br
    out = out + torch.randn_like(out) * noise
    return out.clamp(0, 1)


def augment(img, pts, sizes, cfg):
    img, pts = random_affine(img, pts, sizes, cfg["rot_deg"], tuple(cfg["scale"]), cfg["shift"])
    img = random_intensity(img, tuple(cfg["gamma"]), cfg["contrast"], cfg["brightness"], cfg["noise"])
    return img, pts
