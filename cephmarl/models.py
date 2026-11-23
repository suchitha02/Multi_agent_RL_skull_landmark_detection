"""Shared agent trunk, the per-algorithm heads, and the U-Net baseline."""
import torch
import torch.nn as nn


def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.GroupNorm(8, cout), nn.ReLU(inplace=True))


class Trunk(nn.Module):
    """Patch encoder + context encoder -> 256-d feature per agent. Shared by all K agents."""

    def __init__(self, num_landmarks, num_levels=4):
        super().__init__()
        K = num_landmarks
        self.K = K
        self.encoder = nn.Sequential(conv_block(1, 32), conv_block(32, 64), conv_block(64, 128),
                                     conv_block(128, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.id_emb = nn.Embedding(K, 16)
        self.level_emb = nn.Embedding(num_levels, 4)
        self.context = nn.Sequential(nn.Linear(16 + 4 + 2 * K + 16, 64), nn.ReLU(inplace=True))
        self.fusion = nn.Sequential(nn.Linear(128 + 64, 256), nn.ReLU(inplace=True),
                                    nn.Linear(256, 256), nn.ReLU(inplace=True))

    def forward(self, obs):
        """obs tensors shaped (B,K,...). Returns (B,K,256)."""
        B, K, P, _ = obs["patch"].shape
        ids = torch.arange(K, device=obs["patch"].device).expand(B, K)
        rest = torch.cat([obs["coords"], obs["actions"]], -1)
        h = self.forward_flat(obs["patch"].reshape(B * K, P, P), ids.reshape(-1),
                              obs["level"].reshape(-1), rest.reshape(B * K, -1))
        return h.reshape(B, K, -1)

    def forward_flat(self, patch, ids, level, rest):
        """Single agent samples: patch (N,P,P), ids (N,), level (N,), rest = coords + actions (N, 2K+16)."""
        f_img = self.encoder(patch[:, None])
        ctx = torch.cat([self.id_emb(ids), self.level_emb(level), rest], -1)
        return self.fusion(torch.cat([f_img, self.context(ctx)], -1))


class DuelingQNet(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.trunk = Trunk(K)
        self.value = nn.Linear(256, 1)
        self.adv = nn.Linear(256, 4)

    def forward(self, obs):
        return self.head(self.trunk(obs))  # (B,K,4)

    def forward_flat(self, patch, ids, level, rest):
        return self.head(self.trunk.forward_flat(patch, ids, level, rest))  # (N,4)

    def head(self, h):
        a = self.adv(h)
        return self.value(h) + a - a.mean(-1, keepdim=True)


class Actor(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.trunk = Trunk(K)
        self.head = nn.Linear(256, 4)

    def forward(self, obs):
        return self.head(self.trunk(obs))  # logits (B,K,4)


class CentralCritic(nn.Module):
    """MAPPO critic V(s, i): agent i's features + mean of all agents' features."""

    def __init__(self, K):
        super().__init__()
        self.trunk = Trunk(K)
        self.head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 1))

    def forward(self, obs):
        h = self.trunk(obs)
        g = h.mean(1, keepdim=True).expand_as(h)
        return self.head(torch.cat([h, g], -1)).squeeze(-1)  # (B,K)


class COMACritic(nn.Module):
    """COMA critic Q(s, u_-i, .) for agent i: global features + the other agents' actions -> 4 Q-values."""

    def __init__(self, K):
        super().__init__()
        self.K = K
        self.trunk = Trunk(K)
        self.head = nn.Sequential(nn.Linear(512 + 4 * K, 256), nn.ReLU(inplace=True), nn.Linear(256, 4))

    def forward(self, obs, actions):
        h = self.trunk(obs)
        B, K, _ = h.shape
        g = h.mean(1, keepdim=True).expand_as(h)
        u = nn.functional.one_hot(actions, 4).float().reshape(B, 1, K * 4).expand(B, K, K * 4)
        own = 1 - torch.eye(K, device=h.device).repeat_interleave(4, dim=1)  # hide each agent's own action
        return self.head(torch.cat([h, g, u * own[None]], -1))  # (B,K,4)


def build_unet(K, encoder="resnet34", weights="imagenet"):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights=weights, in_channels=1, classes=K)
