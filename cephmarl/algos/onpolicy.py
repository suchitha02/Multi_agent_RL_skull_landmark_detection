"""Rollout storage shared by MAPPO and COMA (one rollout = full episodes on B images)."""
import torch
from torch.distributions import Categorical


def make_policy(actor, greedy=True):
    def policy(obs):
        logits = actor(obs)
        dist = Categorical(logits=logits)
        actions = logits.argmax(-1) if greedy else dist.sample()
        return actions, -dist.entropy()  # confidence = negative entropy
    return policy


@torch.no_grad()
def collect(env, actor, images, landmarks, devices=None):
    """Run one training episode with the stochastic policy; return stacked tensors (T,B,K,...)."""
    obs = env.reset(images, landmarks, train=True, devices=devices)
    keys = ("patch", "level", "coords", "actions")
    store = {k: [] for k in keys + ("action", "logp", "reward", "alive")}
    while not env.done.all():
        dist = Categorical(logits=actor(obs))
        a = dist.sample()
        for k in keys:
            v = obs[k]
            store[k].append(((v + 0.5) * 255).round().to(torch.uint8) if k == "patch" else v)
        store["action"].append(a)
        store["logp"].append(dist.log_prob(a))
        obs, reward, _, alive = env.step(a)
        store["reward"].append(reward)
        store["alive"].append(alive.float())
    return {k: torch.stack(v) for k, v in store.items()}


def obs_at(roll, t_idx, b_idx):
    """Rebuild an obs dict for (t,b) pairs -> tensors (n,K,...)."""
    return {"patch": roll["patch"][t_idx, b_idx].float() / 255 - 0.5,
            "level": roll["level"][t_idx, b_idx], "coords": roll["coords"][t_idx, b_idx],
            "actions": roll["actions"][t_idx, b_idx]}


def next_alive(alive):
    """alive at t+1 (0 after the last step)."""
    return torch.cat([alive[1:], torch.zeros_like(alive[:1])])


def minibatches(T, B, size, device):
    perm = torch.randperm(T * B, device=device)
    for s in range(0, T * B, size):
        idx = perm[s:s + size]
        yield idx // B, idx % B


def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum().clamp(min=1)
