"""Independent Double + Dueling DQN with one network shared by all agents."""
import copy

import numpy as np
import torch
import torch.nn.functional as F

from ..models import DuelingQNet
from ..rl_common import checkpoint, make_env, sample_batch, validate


class Replay:
    """Per-agent transitions. Patches stored as uint8 to save memory."""

    def __init__(self, size, K, patch, device):
        self.size, self.n, self.i, self.dev = size, 0, 0, device
        ctx = 2 * K + 16
        self.data = {
            "patch": torch.zeros(size, patch, patch, dtype=torch.uint8), "next_patch": torch.zeros(size, patch, patch, dtype=torch.uint8),
            "ctx": torch.zeros(size, ctx), "next_ctx": torch.zeros(size, ctx),
            "level": torch.zeros(size, dtype=torch.long), "next_level": torch.zeros(size, dtype=torch.long),
            "id": torch.zeros(size, dtype=torch.long), "action": torch.zeros(size, dtype=torch.long),
            "reward": torch.zeros(size), "done": torch.zeros(size),
        }

    def add(self, **batch):
        m = len(batch["action"])
        idx = (torch.arange(m) + self.i) % self.size
        for k, v in batch.items():
            self.data[k][idx] = v.cpu().to(self.data[k].dtype)
        self.i = (self.i + m) % self.size
        self.n = min(self.n + m, self.size)

    def sample(self, m):
        idx = torch.randint(0, self.n, (m,))
        return {k: v[idx].to(self.dev) for k, v in self.data.items()}


def flat_obs(obs, alive):
    """Select alive agents and flatten (B,K,...) -> (M,...)."""
    b, k = alive.nonzero(as_tuple=True)
    patch = ((obs["patch"][b, k] + 0.5) * 255).round().clamp(0, 255)
    ctx = torch.cat([obs["coords"][b, k], obs["actions"][b, k]], -1)
    return patch, ctx, obs["level"][b, k], k


def q_single(net, patch, ctx, level, ids):
    """Q-values (N,4) for individual stored agent samples."""
    return net.forward_flat(patch.float() / 255 - 0.5, ids, level, ctx)


def make_policy(net, eps=0.0):
    def policy(obs):
        q = net(obs)
        top2 = q.topk(2, -1).values
        actions = q.argmax(-1)
        if eps > 0:
            rand = torch.rand(actions.shape, device=q.device) < eps
            actions = torch.where(rand, torch.randint_like(actions, 4), actions)
        return actions, top2[..., 0] - top2[..., 1]  # confidence = Q gap
    return policy


def train(cfg, train_ds, val_ds, logger, device):
    c = cfg["dqn"]
    K = train_ds.num_landmarks
    env = make_env(cfg, train_ds, device)
    net = DuelingQNet(K).to(device)
    target = copy.deepcopy(net)
    opt = torch.optim.Adam(net.parameters(), lr=c["lr"])
    buf = Replay(c["buffer"], K, cfg["env"]["patch"], device)
    rng = np.random.default_rng(cfg["seed"])
    budget, steps, best, next_eval = cfg["agent_steps"], 0, float("inf"), cfg["eval_every"]
    best_ckpt = None

    while steps < budget:
        imgs, lms, devs = sample_batch(train_ds, cfg["batch_images"], rng)
        net.eval()
        obs = env.reset(imgs, lms, train=True, devices=devs)
        while not env.done.all() and steps < budget:
            eps = max(c["eps_end"], 1 - (1 - c["eps_end"]) * steps / (c["eps_frac"] * budget))
            with torch.no_grad():
                actions, _ = make_policy(net, eps)(obs)
            next_obs, reward, done, alive = env.step(actions)
            p, x, lv, ids = flat_obs(obs, alive)
            np_, nx, nlv, _ = flat_obs(next_obs, alive)
            b, k = alive.nonzero(as_tuple=True)
            buf.add(patch=p, ctx=x, level=lv, id=ids, action=actions[b, k], reward=reward[b, k],
                    next_patch=np_, next_ctx=nx, next_level=nlv, done=done[b, k].float())
            steps += int(alive.sum())
            obs = next_obs

            if buf.n >= c["warmup"]:
                net.train()
                for _ in range(c["updates_per_step"]):
                    s = buf.sample(c["batch"])
                    q = q_single(net, s["patch"], s["ctx"], s["level"], s["id"])
                    q = q.gather(1, s["action"][:, None]).squeeze(1)
                    with torch.no_grad():
                        a_next = q_single(net, s["next_patch"], s["next_ctx"], s["next_level"], s["id"]).argmax(1)
                        q_next = q_single(target, s["next_patch"], s["next_ctx"], s["next_level"], s["id"])
                        y = s["reward"] + cfg["gamma"] * (1 - s["done"]) * q_next.gather(1, a_next[:, None]).squeeze(1)
                    loss = F.smooth_l1_loss(q, y)
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
                    opt.step()
                    with torch.no_grad():
                        for pt, po in zip(target.parameters(), net.parameters()):
                            pt.lerp_(po, c["tau"])
                net.eval()

        if steps >= next_eval or steps >= budget:
            next_eval += cfg["eval_every"]
            val = validate(env, make_policy(net), val_ds)
            logger.log(step=steps, val_MRE_mm=val, eps=eps)
            if val < best:
                best = val
                best_ckpt = checkpoint("dqn", cfg, train_ds, env, net=net)
    return best_ckpt, best
