"""Vectorised multi-agent landmark environment (B images x K agents, all on one device).

Positions are kept in full-resolution pixels (0.1 mm/px). Each agent moves on a
4-level pyramid and zooms in by rule (oscillation or step cap), never by action.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .augment import augment

# up, down, left, right as (dx, dy)
MOVES = torch.tensor([[0, -1], [0, 1], [-1, 0], [1, 0]], dtype=torch.float32)


class CephEnv:
    def __init__(self, cfg, num_landmarks, mean_shape, device="cpu"):
        """cfg: the 'env' section of the config.
        mean_shape: (K,2) in [0,1] (x/W, y/H), or a dict {device name: (K,2), "__all__": (K,2)}
        used when cfg["init"] == "mean_device" (falls back to "__all__" for unknown devices)."""
        self.cfg = cfg
        self.K = num_landmarks
        self.dev = torch.device(device)
        self.levels = list(cfg["levels"])                       # e.g. [0,1,2,3]
        self.factor = torch.tensor([cfg["factors"][l] for l in self.levels], device=self.dev)
        self.step_px = torch.tensor([cfg["step_px"][l] for l in self.levels], dtype=torch.float32, device=self.dev)
        self.cap = torch.tensor([cfg["caps"][l] for l in self.levels], device=self.dev)
        self.patch = cfg["patch"]
        shapes = mean_shape if isinstance(mean_shape, dict) else {"__all__": mean_shape}
        self.mean_shapes = {k: torch.as_tensor(v, dtype=torch.float32, device=self.dev) for k, v in shapes.items()}
        self.mean_shape = self.mean_shapes["__all__"]
        self.moves = MOVES.to(self.dev)
        self.max_steps = int(self.cap.sum().item())

    # ------------------------------------------------------------ setup
    def reset(self, images, landmarks=None, train=True, devices=None):
        """images: list of uint8 HxW arrays (sizes may differ). landmarks: (B,K,2) array or None.
        devices: imaging device per image (only used with init: mean_device)."""
        B, K, dev = len(images), self.K, self.dev
        H = max(im.shape[0] for im in images)
        W = max(im.shape[1] for im in images)
        img = torch.zeros(B, 1, H, W, device=dev)
        for b, im in enumerate(images):
            img[b, 0, : im.shape[0], : im.shape[1]] = torch.from_numpy(im).to(dev).float() / 255
        self.size = torch.tensor([[im.shape[1], im.shape[0]] for im in images], dtype=torch.float32, device=dev)
        gt = None if landmarks is None else torch.as_tensor(np.asarray(landmarks), dtype=torch.float32, device=dev)
        if train and self.cfg["augment"]["enabled"]:
            img, gt = augment(img, gt, self.size, self.cfg["augment"])
        self.gt = gt
        self.B = B

        # pyramid, each level padded so a patch can be cut anywhere inside the image
        half = self.patch // 2
        self.pyr = []
        for f in self.factor.tolist():
            lv = F.avg_pool2d(img, f, f, ceil_mode=True) if f > 1 else img
            self.pyr.append(F.pad(lv, (half, half, half, half)))

        # start positions
        if self.cfg["init"] == "random":
            pos = torch.rand(B, K, 2, device=dev) * self.size[:, None]
        else:
            if self.cfg["init"] == "mean_device" and devices is not None:
                start = torch.stack([self.mean_shapes.get(d, self.mean_shape) for d in devices])
            else:
                start = self.mean_shape[None].expand(B, -1, -1)
            pos = start * self.size[:, None]
            if train:
                j = self.cfg["jitter_mm"] * 10  # mm -> px at 0.1 mm/px
                pos = pos + (torch.rand(B, K, 2, device=dev) * 2 - 1) * j
        self.pos = self._clamp(pos.round())

        self.lvl = torch.zeros(B, K, dtype=torch.long, device=dev)      # index into self.levels
        self.steps_in_lvl = torch.zeros(B, K, dtype=torch.long, device=dev)
        self.hist = torch.full((B, K, 4, 2), float("nan"), device=dev)   # recent positions, this level
        self.hist[:, :, 0] = self.pos
        self.fine_hist = torch.full((B, K, 8, 2), float("nan"), device=dev)  # positions at the finest level
        self.last_actions = torch.full((B, K, 4), -1, dtype=torch.long, device=dev)
        self.done = torch.zeros(B, K, dtype=torch.bool, device=dev)
        self.t = 0
        self.dist = self._dist(self.pos)
        return self.obs()

    def _clamp(self, pos):
        return torch.minimum(torch.maximum(pos, torch.zeros_like(pos)), self.size[:, None] - 1)

    def _dist(self, pos):
        if self.gt is None:
            return torch.zeros(pos.shape[:2], device=self.dev)
        return torch.linalg.norm(pos - self.gt, dim=-1)

    # ------------------------------------------------------------ observation
    def obs(self):
        B, K, P, dev = self.B, self.K, self.patch, self.dev
        half = P // 2
        patches = torch.zeros(B, K, P, P, device=dev)
        d = torch.arange(P, device=dev) - half
        for li in range(len(self.levels)):
            m = self.lvl == li
            if not m.any():
                continue
            b_idx, k_idx = m.nonzero(as_tuple=True)
            c = torch.div(self.pos[b_idx, k_idx], self.factor[li], rounding_mode="floor").long() + half
            ys = (c[:, 1, None] + d)[:, :, None]
            xs = (c[:, 0, None] + d)[:, None, :]
            patches[b_idx, k_idx] = self.pyr[li][b_idx[:, None, None], 0, ys, xs]

        coords = (self.pos / self.size[:, None]).reshape(B, 1, 2 * K).expand(B, K, 2 * K).clone()
        if not self.cfg["share_positions"]:  # ablation: each agent only sees its own position
            own = torch.eye(K, device=dev).repeat_interleave(2, dim=1).bool()
            coords = coords * own[None]
        act = F.one_hot(self.last_actions.clamp(min=0), 4).float()
        act = act * (self.last_actions >= 0)[..., None]
        level = torch.tensor(self.levels, device=dev)[self.lvl]
        return {"patch": patches - 0.5, "level": level, "coords": coords, "actions": act.reshape(B, K, 16),
                "alive": ~self.done}

    # ------------------------------------------------------------ dynamics
    def step(self, actions):
        """actions: (B,K) long. Returns obs, reward (B,K), done (B,K), alive_before (B,K)."""
        alive = ~self.done
        step = self.step_px[self.lvl]
        new = self.pos + self.moves[actions] * step[..., None]
        out = ((new < 0) | (new > self.size[:, None] - 1)).any(-1)
        new = torch.where((out | ~alive)[..., None], self.pos, new)

        new_dist = self._dist(new)
        reward = torch.nan_to_num((self.dist - new_dist) / step)
        reward = torch.where(out, torch.full_like(reward, -1.0), reward)
        reward = reward * alive
        if self.cfg["reward"] == "team":  # every alive agent gets the mean reward of its image
            team = reward.sum(1, keepdim=True) / alive.sum(1, keepdim=True).clamp(min=1)
            reward = team * alive

        osc = ((self.hist == new[:, :, None]).all(-1)).any(-1) & alive
        self.pos, self.dist = new, new_dist
        self.last_actions = torch.cat([actions[..., None], self.last_actions[..., :-1]], -1)
        self.hist = torch.cat([new[:, :, None], self.hist[:, :, :-1]], 2)
        self.steps_in_lvl = self.steps_in_lvl + alive

        last = len(self.levels) - 1
        at_fine = (self.lvl == last) & alive
        self.fine_hist = torch.where(at_fine[..., None, None],
                                     torch.cat([new[:, :, None], self.fine_hist[:, :, :-1]], 2), self.fine_hist)
        finished = osc | (self.steps_in_lvl >= self.cap[self.lvl])
        self.done = self.done | (finished & at_fine)
        zoom = finished & alive & (self.lvl < last)
        self.lvl = self.lvl + zoom
        self.steps_in_lvl = torch.where(zoom, torch.zeros_like(self.steps_in_lvl), self.steps_in_lvl)
        fresh = torch.full_like(self.hist, float("nan"))
        fresh[:, :, 0] = self.pos
        self.hist = torch.where(zoom[..., None, None], fresh, self.hist)
        self.t += 1
        if self.t >= self.max_steps:
            self.done[:] = True
        return self.obs(), reward, self.done.clone(), alive

    # ------------------------------------------------------------ outputs
    def prediction(self):
        """(B,K,2) mean of the last 4 positions at the finest level (current position if none)."""
        last4 = self.fine_hist[:, :, :4]
        pred = torch.nanmean(last4, 2)
        return torch.where(torch.isnan(pred), self.pos, pred)

    def spread_mm(self):
        """(B,K) spread of the last 8 fine-level positions, in mm."""
        h = self.fine_hist
        n = (~torch.isnan(h[..., 0])).sum(-1).clamp(min=1)
        mu = torch.nanmean(h, 2, keepdim=True)
        var = torch.nan_to_num((h - mu) ** 2).sum((2, 3)) / n
        return var.sqrt() * 0.1
