"""PyTorch Dataset over the precomputed PushT features.

Each training example is:
    obs    : ResNet tokens for `obs_horizon` image frames -> (obs_h * n_img_tokens, 512)
    action : the (normalized) action(s) to predict          -> (pred_h * action_dim,)

For the "most basic" setup we use obs_h = 1 and pred_h = 1: one image in, one
2-D action out. The code is written so you can raise either horizon later
(more frames simply concatenate into one larger token set that the VisionStem's
cross-attention happily pools down again).

Episodes are respected: we never stitch frames across the boundary between two
demonstrations. The train/validation split is done at the EPISODE level (in
train.py) so frames from the same trajectory never leak across splits.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def episode_starts(episode_ends: np.ndarray) -> np.ndarray:
    """Convert exclusive end indices into start indices.

    episode_ends = [161, 279, ...]  means episode 0 is frames 0..160,
    episode 1 is frames 161..278, etc. So starts = [0, 161, 279, ...].
    """
    return np.concatenate([[0], episode_ends[:-1]])


def build_sample_indices(
    episode_ends: np.ndarray, obs_horizon: int, pred_horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (indices, episode_ids): every valid sample start timestep `t` and
    the episode each one belongs to.

    A timestep t is valid when both the observation window
    [t-obs_h+1 .. t] and the prediction window [t .. t+pred_h-1] fit fully
    inside a single episode.
    """
    starts = episode_starts(episode_ends)
    idx, eps = [], []
    for ei, (s, e) in enumerate(zip(starts, episode_ends)):
        lo = s + obs_horizon - 1          # earliest t whose obs window fits
        hi = e - pred_horizon             # latest t whose pred window fits (inclusive)
        if hi >= lo:
            for t in range(int(lo), int(hi) + 1):
                idx.append(t)
                eps.append(ei)
    return np.array(idx, dtype=np.int64), np.array(eps, dtype=np.int64)


class PushTDataset(Dataset):
    """Serves (observation tokens, normalized action) pairs for one split."""

    def __init__(
        self,
        features: np.ndarray,          # (N, n_img_tokens, 512) float16/float32
        actions: np.ndarray,           # (N, action_dim) float32
        sample_indices: np.ndarray,    # valid timestep starts for THIS split
        action_mean: np.ndarray,       # (action_dim,)
        action_std: np.ndarray,        # (action_dim,)
        obs_horizon: int = 1,
        pred_horizon: int = 1,
    ):
        self.features = features
        self.actions = actions
        self.indices = sample_indices
        self.obs_h = obs_horizon
        self.pred_h = pred_horizon
        self.action_dim = actions.shape[1]
        self.action_mean = torch.as_tensor(action_mean, dtype=torch.float32)
        self.action_std = torch.as_tensor(action_std, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        t = int(self.indices[i])
        # observation: obs_h frames stacked into one token set
        obs = self.features[t - self.obs_h + 1 : t + 1]       # (obs_h, n_img_tokens, 512)
        obs = obs.reshape(-1, obs.shape[-1])                  # (obs_h * n_img_tokens, 512)
        # target: pred_h actions flattened
        act = self.actions[t : t + self.pred_h]               # (pred_h, action_dim)
        act = torch.as_tensor(act, dtype=torch.float32).reshape(-1)   # (pred_h * action_dim,)
        # ! normalize so the network targets a well-scaled (~unit variance) signal
        mean = self.action_mean.repeat(self.pred_h)           # (pred_h * action_dim,)
        std = self.action_std.repeat(self.pred_h)
        act = (act - mean) / std
        return torch.as_tensor(obs, dtype=torch.float32), act
