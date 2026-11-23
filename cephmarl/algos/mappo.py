"""MAPPO: shared actor, centralised critic V(s, i), clipped PPO objective, GAE."""
import numpy as np
import torch
from torch.distributions import Categorical

from ..models import Actor, CentralCritic
from ..rl_common import checkpoint, make_env, sample_batch, validate
from .onpolicy import collect, make_policy, masked_mean, minibatches, next_alive, obs_at


@torch.no_grad()
def values(critic, roll, chunk=16):
    T = roll["action"].shape[0]
    B = roll["action"].shape[1]
    out = []
    for s in range(0, T, chunk):
        t = torch.arange(s, min(s + chunk, T), device=roll["action"].device)
        tt, bb = t.repeat_interleave(B), t.new_tensor(range(B)).repeat(len(t))
        out.append(critic(obs_at(roll, tt, bb)).reshape(len(t), B, -1))
    return torch.cat(out)


def gae(reward, value, alive, gamma, lam):
    nxt_alive = next_alive(alive)
    nxt_value = torch.cat([value[1:], torch.zeros_like(value[:1])])
    adv = torch.zeros_like(reward)
    last = torch.zeros_like(reward[0])
    for t in reversed(range(reward.shape[0])):
        delta = reward[t] + gamma * nxt_value[t] * nxt_alive[t] - value[t]
        last = delta + gamma * lam * nxt_alive[t] * last
        adv[t] = last
    return adv * alive, (adv + value) * alive


def train(cfg, train_ds, val_ds, logger, device):
    c = cfg["mappo"]
    K = train_ds.num_landmarks
    env = make_env(cfg, train_ds, device)
    actor, critic = Actor(K).to(device), CentralCritic(K).to(device)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=c["lr"])
    rng = np.random.default_rng(cfg["seed"])
    steps, best, next_eval, best_ckpt = 0, float("inf"), cfg["eval_every"], None

    while steps < cfg["agent_steps"]:
        imgs, lms, devs = sample_batch(train_ds, cfg["batch_images"], rng)
        roll = collect(env, actor, imgs, lms, devs)
        alive = roll["alive"]
        steps += int(alive.sum())
        v = values(critic, roll)
        adv, ret = gae(roll["reward"], v, alive, cfg["gamma"], c["gae_lambda"])
        m = alive.bool()
        adv = (adv - adv[m].mean()) / (adv[m].std() + 1e-8)

        T, B = alive.shape[:2]
        for _ in range(c["epochs"]):
            for tt, bb in minibatches(T, B, c["minibatch"], device):
                o = obs_at(roll, tt, bb)
                mask = alive[tt, bb]
                dist = Categorical(logits=actor(o))
                logp = dist.log_prob(roll["action"][tt, bb])
                ratio = (logp - roll["logp"][tt, bb]).exp()
                a = adv[tt, bb]
                pg = -torch.min(ratio * a, ratio.clamp(1 - c["clip"], 1 + c["clip"]) * a)
                v_loss = (critic(o) - ret[tt, bb]) ** 2
                ent = dist.entropy()
                ent_last = float(masked_mean(ent, mask))
                loss = masked_mean(pg + c["value_coef"] * v_loss - c["entropy"] * ent, mask)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
                opt.step()

        if steps >= next_eval or steps >= cfg["agent_steps"]:
            next_eval += cfg["eval_every"]
            val = validate(env, make_policy(actor), val_ds)
            logger.log(step=steps, val_MRE_mm=val, train_reward=float(masked_mean(roll["reward"], alive)),
                       entropy=ent_last)
            if val < best:
                best = val
                best_ckpt = checkpoint("mappo", cfg, train_ds, env, actor=actor, critic=critic)
    return best_ckpt, best
