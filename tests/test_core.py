"""Fast correctness checks:  python -m pytest -q tests"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cephmarl.augment import random_affine  # noqa: E402
from cephmarl.data import CephDataset, make_synthetic  # noqa: E402
from cephmarl.env import CephEnv  # noqa: E402
from cephmarl.flagging import ShapeModel, flag_metrics  # noqa: E402
from cephmarl.landmarks import AARIZ_LANDMARKS, ISBI_LANDMARKS, map_indices  # noqa: E402
from cephmarl.metrics import radial_errors, summarize  # noqa: E402
from cephmarl.rl_common import mean_shape  # noqa: E402
from cephmarl.utils import load_config  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    out = tmp_path_factory.mktemp("syn")
    make_synthetic(str(out), num_landmarks=7, sizes=(12, 4, 4), shape=(400, 360))
    return CephDataset(str(out), "train")


def oracle(env):
    def policy(obs):
        d = env.gt - env.pos
        horiz = d[..., 0].abs() >= d[..., 1].abs()
        a = torch.where(horiz, torch.where(d[..., 0] > 0, 3, 2), torch.where(d[..., 1] > 0, 1, 0))
        return a, torch.zeros_like(d[..., 0])
    return policy


@pytest.mark.parametrize("augment", [False, True])
def test_oracle_reaches_target(ds, augment):
    cfg = load_config(os.path.join(ROOT, "configs", "base.yaml"))["env"]
    cfg["augment"]["enabled"] = augment
    env = CephEnv(cfg, ds.num_landmarks, mean_shape(ds))
    imgs = [ds.image(i) for i in range(4)]
    lms = np.stack([ds.landmarks(i) for i in range(4)])
    env.reset(imgs, lms, train=True)
    pol, total = oracle(env), 0.0
    while not env.done.all():
        a, _ = pol(None)
        _, r, _, _ = env.step(a)
        total += float(r.sum())
    err = radial_errors(env.prediction().numpy(), env.gt.numpy())
    assert err.max() < 0.1, err.max()          # within 0.1 mm of every target
    assert total > 0                            # moving toward targets is rewarded
    assert env.t <= env.max_steps


def test_rewards_are_bounded_and_signed(ds):
    cfg = load_config(os.path.join(ROOT, "configs", "base.yaml"))["env"]
    env = CephEnv(cfg, ds.num_landmarks, mean_shape(ds))
    env.reset([ds.image(0)], ds.landmarks(0)[None], train=False)
    env.lvl[:] = 3  # 1-px steps, so a step toward the target can't overshoot
    good, _ = oracle(env)(None)
    _, r, _, _ = env.step(good)
    assert (r >= -1).all() and (r <= 1).all() and (r > 0).all()
    env.reset([ds.image(0)], ds.landmarks(0)[None], train=False)
    env.lvl[:] = 3
    _, r_bad, _, _ = env.step(good ^ 1)  # opposite move from the same start: up<->down, left<->right
    assert (r_bad < 0).all()


def test_patch_is_centred_on_agent(ds):
    cfg = load_config(os.path.join(ROOT, "configs", "base.yaml"))["env"]
    cfg["augment"]["enabled"] = False
    env = CephEnv(cfg, ds.num_landmarks, mean_shape(ds))
    img = np.zeros((300, 300), np.uint8)
    img[150, 120] = 255
    obs = env.reset([img], np.zeros((1, ds.num_landmarks, 2)), train=False)
    env.pos[:] = torch.tensor([120.0, 150.0])
    env.lvl[:] = 3
    obs = env.obs()
    p = obs["patch"][0, 0].numpy()
    assert np.unravel_index(p.argmax(), p.shape) == (32, 32)


def test_affine_moves_landmarks_with_image():
    img = torch.zeros(1, 1, 200, 240)
    pts = torch.tensor([[[60.0, 50.0], [180.0, 140.0]]])
    for x, y in pts[0].long():
        img[0, 0, y - 1:y + 2, x - 1:x + 2] = 1
    sizes = torch.tensor([[240.0, 200.0]])
    torch.manual_seed(0)
    out, new = random_affine(img, pts, sizes, rot_deg=10, scale=(0.9, 1.1), shift=0.05)
    for x, y in new[0]:
        patch = out[0, 0, int(y) - 3:int(y) + 4, int(x) - 3:int(x) + 4]
        assert patch.max() > 0.3


def test_shape_residual_flags_displaced_landmark(ds):
    shapes = ds.all_landmarks()
    sm = ShapeModel.fit(shapes)
    s = shapes[0].copy()
    base = sm.residuals(s)
    s[3] += [80, -60]
    moved = sm.residuals(s)
    assert moved.argmax() == 3 and moved[3] > base[3] * 3


def test_metrics_and_flags():
    gt = np.zeros((2, 3, 2))
    pred = gt.copy()
    pred[0, 0] = [30, 40]  # 50 px = 5 mm
    err = radial_errors(pred, gt)
    m = summarize(err)
    assert np.isclose(m["MRE_mm"], 5 / 6) and np.isclose(m["SDR_4.0mm_%"], 100 * 5 / 6)
    f = flag_metrics(err.copy(), err, budget=1 / 6)
    assert f["catch_rate_err>4mm_at_16%"] == 1.0


def test_landmark_mapping_covers_all_isbi():
    idx = map_indices(ISBI_LANDMARKS, AARIZ_LANDMARKS)
    assert len(idx) == 19 and min(idx) >= 0 and len(set(idx)) == 19


def test_policy_gradient_sign(ds):
    """One PPO-style step on a rollout must raise the log-prob of positive-advantage actions."""
    from torch.distributions import Categorical
    from cephmarl.algos.onpolicy import collect, obs_at
    from cephmarl.models import Actor
    torch.manual_seed(0)
    cfg = load_config(os.path.join(ROOT, "configs", "base.yaml"))["env"]
    env = CephEnv(cfg, ds.num_landmarks, mean_shape(ds))
    actor = Actor(ds.num_landmarks)
    roll = collect(env, actor, [ds.image(i) for i in range(2)], np.stack([ds.landmarks(i) for i in range(2)]))
    T, B = roll["alive"].shape[:2]
    tt, bb = torch.arange(T).repeat_interleave(B), torch.arange(B).repeat(T)
    o, a, m = obs_at(roll, tt, bb), roll["action"][tt, bb], roll["alive"][tt, bb]
    adv = torch.sign(roll["reward"][tt, bb]) * m
    opt = torch.optim.SGD(actor.parameters(), lr=0.5)
    before = Categorical(logits=actor(o)).log_prob(a)
    loss = -(before * adv).sum() / m.sum()
    opt.zero_grad(); loss.backward(); opt.step()
    after = Categorical(logits=actor(o)).log_prob(a)
    pos, neg = (adv > 0), (adv < 0)
    assert (after - before)[pos].mean() > 0 > (after - before)[neg].mean()
