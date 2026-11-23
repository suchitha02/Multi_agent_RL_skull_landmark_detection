"""COMA: shared actor, counterfactual centralised critic Q(s, u_-i, .), team reward, TD(lambda) critic."""
import copy

import numpy as np
import torch
from torch.distributions import Categorical

from ..models import Actor, COMACritic
from ..rl_common import checkpoint, make_env, sample_batch, validate
from .onpolicy import collect, make_policy, masked_mean, minibatches, next_alive, obs_at


@torch.no_grad()
def taken_q(critic, roll, chunk=16):
    T, B = roll["action"].shape[:2]
    out = []
    for s in range(0, T, chunk):
        t = torch.arange(s, min(s + chunk, T), device=roll["action"].device)
        tt, bb = t.repeat_interleave(B), t.new_tensor(range(B)).repeat(len(t))
        a = roll["action"][tt, bb]
        q = critic(obs_at(roll, tt, bb), a).gather(-1, a[..., None]).squeeze(-1)
        out.append(q.reshape(len(t), B, -1))
    return torch.cat(out)


@torch.no_grad()
def counterfactual_adv(critic, actor, roll, chunk=16):
    """A_i = Q(s, u)[a_i] - sum_a pi_i(a) Q(s, (u_-i, a)), for every step of the rollout (T,B,K)."""
    T, B = roll["action"].shape[:2]
    out = []
    for s in range(0, T, chunk):
        t = torch.arange(s, min(s + chunk, T), device=roll["action"].device)
        tt, bb = t.repeat_interleave(B), t.new_tensor(range(B)).repeat(len(t))
        o, a = obs_at(roll, tt, bb), roll["action"][tt, bb]
        q = critic(o, a)
        pi = torch.softmax(actor(o), -1)
        adv = q.gather(-1, a[..., None]).squeeze(-1) - (pi * q).sum(-1)
        out.append(adv.reshape(len(t), B, -1))
    return torch.cat(out)


def td_lambda(reward, q, alive, gamma, lam):
    nxt_alive = next_alive(alive)
    nxt_q = torch.cat([q[1:], torch.zeros_like(q[:1])])
    ret = torch.zeros_like(reward)
    g = torch.zeros_like(reward[0])
    for t in reversed(range(reward.shape[0])):
        g = reward[t] + gamma * nxt_alive[t] * ((1 - lam) * nxt_q[t] + lam * g)
        ret[t] = g
    return ret * alive


def train(cfg, train_ds, val_ds, logger, device):
    c = cfg["coma"]
    K = train_ds.num_landmarks
    env = make_env(cfg, train_ds, device)
    actor, critic = Actor(K).to(device), COMACritic(K).to(device)
    target = copy.deepcopy(critic)
    opt_a = torch.optim.Adam(actor.parameters(), lr=c["lr_actor"])
    opt_c = torch.optim.Adam(critic.parameters(), lr=c["lr_critic"])
    rng = np.random.default_rng(cfg["seed"])
    steps, best, next_eval, best_ckpt, it = 0, float("inf"), cfg["eval_every"], None, 0

    while steps < cfg["agent_steps"]:
        imgs, lms, devs = sample_batch(train_ds, cfg["batch_images"], rng)
        roll = collect(env, actor, imgs, lms, devs)
        alive = roll["alive"]
        steps += int(alive.sum())
        ret = td_lambda(roll["reward"], taken_q(target, roll), alive, cfg["gamma"], c["td_lambda"])
        T, B = alive.shape[:2]

        for _ in range(c["critic_epochs"]):
            for tt, bb in minibatches(T, B, c["minibatch"], device):
                a = roll["action"][tt, bb]
                q = critic(obs_at(roll, tt, bb), a).gather(-1, a[..., None]).squeeze(-1)
                loss = masked_mean((q - ret[tt, bb]) ** 2, alive[tt, bb])
                opt_c.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 10)
                opt_c.step()

        # Actor: COMA's counterfactual advantage, optimised with PPO's clipped objective.
        # The plain (unclipped) COMA step collapsed the policy to one action on ISBI (entropy
        # 1.05 -> 1e-24 right after warm-up). Clipping bounds how far one rollout can move it.
        # A critic-only warm-up comes first, because a random critic gives random advantages.
        ent_sum, ent_n = 0.0, 0
        if it >= c["critic_warmup"]:
            adv = counterfactual_adv(critic, actor, roll)
            m = alive.bool()
            adv = (adv - adv[m].mean()) / (adv[m].std() + 1e-8)
            for _ in range(c["actor_epochs"]):
                for tt, bb in minibatches(T, B, c["minibatch"], device):
                    o, a, mask = obs_at(roll, tt, bb), roll["action"][tt, bb], alive[tt, bb]
                    dist = Categorical(logits=actor(o))
                    ratio = (dist.log_prob(a) - roll["logp"][tt, bb]).exp()
                    A = adv[tt, bb]
                    pg = -torch.min(ratio * A, ratio.clamp(1 - c["clip"], 1 + c["clip"]) * A)
                    ent = dist.entropy()
                    ent_sum += float((ent.detach() * mask).sum())
                    ent_n += float(mask.sum())
                    loss = masked_mean(pg - c["entropy"] * ent, mask)
                    opt_a.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                    opt_a.step()

        it += 1
        if it % c["target_every"] == 0:
            target.load_state_dict(critic.state_dict())

        if steps >= next_eval or steps >= cfg["agent_steps"]:
            next_eval += cfg["eval_every"]
            val = validate(env, make_policy(actor), val_ds)
            logger.log(step=steps, val_MRE_mm=val, train_reward=float(masked_mean(roll["reward"], alive)),
                       entropy=ent_sum / max(ent_n, 1))
            if val < best:
                best = val
                best_ckpt = checkpoint("coma", cfg, train_ds, env, actor=actor, critic=critic)
    return best_ckpt, best
