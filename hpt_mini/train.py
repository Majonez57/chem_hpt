"""Train the minimal HPT policy on PushT (a fast smoke test).

Run from the project root:
    python -m hpt_mini.train
  or
    python hpt_mini/train.py

What it does:
  1. load the precomputed ResNet features (run data_prep.py first)
  2. split episodes into train / validation
  3. train HPTMini with AdamW + MSE on NORMALIZED actions
  4. print train/val loss each epoch and save the best checkpoint

The defaults are deliberately tiny ("very small trunk") so a full run finishes
in well under a minute on a modest GPU. Tweak the Config dataclass to scale up.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from torch.utils.data import DataLoader

# support both "python -m hpt_mini.train" and "python hpt_mini/train.py"
try:
    from .dataset import PushTDataset, build_sample_indices
    from .model import HPTMini
except ImportError:
    from dataset import PushTDataset, build_sample_indices
    from model import HPTMini

FEAT_PATH = "data/pusht_resnet18_tokens.npy"
ZARR_PATH = "data/pusht_raw_dl/pusht_cchi_v7_replay.zarr"


@dataclass
class Config:
    # data
    obs_horizon: int = 1
    pred_horizon: int = 1
    val_episode_ratio: float = 0.1     # fraction of episodes held out for validation
    seed: int = 0
    # model (the "very small" setting)
    embed_dim: int = 128
    num_query_tokens: int = 4
    num_heads: int = 4
    num_trunk_blocks: int = 2
    dropout: float = 0.0
    # training
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    num_workers: int = 2
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    out_dir: str = "outputs/hpt_mini_pusht"


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load cached features, raw actions and episode boundaries."""
    features = np.load(FEAT_PATH)                                   # (N, n_img_tokens, 512)
    root = zarr.open(ZARR_PATH, mode="r")
    actions = np.asarray(root["data/action"][:], dtype=np.float32)  # (N, 2)
    ep_ends = np.asarray(root["meta/episode_ends"][:])             # (n_episodes,)
    return features, actions, ep_ends


def main() -> None:
    cfg = Config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    features, actions, ep_ends = load_data()
    print(f"features {features.shape} | actions {actions.shape} | episodes {len(ep_ends)}")

    # * episode-level train/val split (no frame leakage between splits)
    all_idx, ep_ids = build_sample_indices(ep_ends, cfg.obs_horizon, cfg.pred_horizon)
    n_eps = len(ep_ends)
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n_eps)
    n_val = max(1, int(n_eps * cfg.val_episode_ratio))
    val_eps = set(perm[:n_val].tolist())
    is_val = np.array([ei in val_eps for ei in ep_ids])
    train_idx, val_idx = all_idx[~is_val], all_idx[is_val]

    # * normalize actions using TRAINING data only
    train_acts = actions[train_idx]
    action_mean = train_acts.mean(axis=0)
    action_std = train_acts.std(axis=0) + 1e-6
    print(f"action mean {action_mean} | action std {action_std}")

    train_set = PushTDataset(features, actions, train_idx, action_mean, action_std,
                             cfg.obs_horizon, cfg.pred_horizon)
    val_set = PushTDataset(features, actions, val_idx, action_mean, action_std,
                           cfg.obs_horizon, cfg.pred_horizon)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    print(f"train samples {len(train_set)} | val samples {len(val_set)}")

    action_dim = actions.shape[1] * cfg.pred_horizon
    model = HPTMini(
        feat_dim=features.shape[-1],
        embed_dim=cfg.embed_dim,
        num_query_tokens=cfg.num_query_tokens,
        num_heads=cfg.num_heads,
        num_trunk_blocks=cfg.num_trunk_blocks,
        action_dim=action_dim,
        dropout=cfg.dropout,
    ).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"HPTMini trainable params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    ckpt_path = os.path.join(cfg.out_dir, "best.pt")
    for epoch in range(1, cfg.epochs + 1):
        # train
        model.train()
        running = 0.0
        for obs, act in train_loader:
            obs = obs.to(cfg.device, non_blocking=True)
            act = act.to(cfg.device, non_blocking=True)
            pred = model(obs)
            loss = F.mse_loss(pred, act)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * obs.size(0)
        train_loss = running / len(train_set)

        # validate
        model.eval()
        vrun = 0.0
        with torch.no_grad():
            for obs, act in val_loader:
                obs = obs.to(cfg.device, non_blocking=True)
                act = act.to(cfg.device, non_blocking=True)
                vrun += F.mse_loss(model(obs), act).item() * obs.size(0)
        val_loss = vrun / len(val_set)

        print(f"epoch {epoch:3d}/{cfg.epochs}  train_mse {train_loss:.4f}  val_mse {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"model": model.state_dict(), "action_mean": action_mean,
                 "action_std": action_std, "cfg": cfg.__dict__},
                ckpt_path,
            )

    print(f"done. best val_mse {best_val:.4f} -> {ckpt_path}")


if __name__ == "__main__":
    main()
