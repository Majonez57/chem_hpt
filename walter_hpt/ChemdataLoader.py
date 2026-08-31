import h5py
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List

# * Map output keys to the h5 datasets they come from (gt keys share sources with obs keys)
OBS_SOURCES = {
    "arm_joints": "arm_joints",
    "arm_pose": "arm_pose",
    "cam_exo": "exo_rgb_feats",
    "cam_wrist": "wrist_rgb_feats",
}
GT_SOURCES = {
    "pred_joint": "arm_joints",
    "pred_pose": "arm_pose",
}
# * gt keys mapped to the obs key of the same source (for baselines)
GT_TO_OBS = {gt: obs for gt, src in GT_SOURCES.items() for obs, osrc in OBS_SOURCES.items() if src == osrc}


class ChemdataLoader(Dataset):
    """
    A class to load samples from saved chemdatasets
    currently single-domain #TODO multidomain

    Inputs and targets are z-score normalised per dimension, with stats computed over the
    full dataset (obs and gt share stats when sourced from the same h5 dataset).
    Use norm()/denorm() to apply/undo this outside of the loader (e.g. inference)
    Args:
        split: Type of split, defines the stride of episodes
        ep_ids: List of episode indexes. Can be None for all episodes in file
    """
    def __init__(self, data_path: str, horizon_steps: int, action_steps: int, split:str, ep_ids: Optional[List[int]] = None, stats: Optional[dict] = None, seed: int = 57, device: str = 'cuda'):
        self.path = data_path
        self.horizon = horizon_steps
        self.future_steps = action_steps
        self.device = device

        with h5py.File(data_path, 'r') as f:
            self.episode_idxs = range(len(f)) if ep_ids is None else ep_ids

            # Find the number of episodes and their lengths for this split
            # Use these to find all possible (episode_n, starts)

            self.samples  = []
            self.episodes = []
            for idx, ep in enumerate(f.keys()):
                if idx in self.episode_idxs: 
                    episode_len = f[ep]["arm_joints"].shape[0]

                    self.episodes.append((ep, episode_len))

                    # All posible samples
                    # Note that for validation and testing, the stride is the size of the window length to avoid correlating errors
                    self.samples += [(ep, start) for start in range(0, episode_len-horizon_steps-action_steps + 1, 1 if split == "train" else action_steps)]

            print(len(self.samples))

            # Val loader should use the stats of the training data!
            if stats == None:
                # * Per-dimension mean/std of every source dataset # TODO cache to disk
                source_stats = {
                    h5_key: self._dataset_stats(f, h5_key)
                    for h5_key in set(OBS_SOURCES.values()) | set(GT_SOURCES.values())
                }
                self.stats = {
                    key: (mean.to(self.device), std.to(self.device))
                    for key, (mean, std) in
                    {key: source_stats[h5_key] for key, h5_key in {**OBS_SOURCES, **GT_SOURCES}.items()}.items()
                }
            else: self.stats = stats

        


        self._file: h5py.File | None = None

    def _dataset_stats(self, f: h5py.File, h5_key: str, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-dimension mean/std of a source dataset, accumulated over every episode"""
        n = 0
        s = None
        sq = None
        for idx, ep in enumerate(f.keys()):
            if idx in self.episode_idxs:
                ds = f[ep][h5_key]
                arr = ds[:].astype(np.float64).reshape(-1, ds.shape[-1])
                n += arr.shape[0]
                s = arr.sum(0) if s is None else s + arr.sum(0)
                sq = np.square(arr).sum(0) if sq is None else sq + np.square(arr).sum(0)

        mean = s / n
        std = np.sqrt(np.maximum(sq / n - mean**2, eps))
        return torch.from_numpy(mean).float(), torch.from_numpy(std).float()

    def __len__(self): return len(self.samples)

    def norm(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        z-score a dict of raw tensors with the dataset stats (moved to self.device)
        2D (t D) inputs are reshaped to a single token per timestep (t 1 D)
        """
        out = {}
        for key, x in data.items():
            mean, std = self.stats[key]
            if x.dim() == 2:
                x = x.reshape(-1, 1, x.shape[-1])
            x = x.float().to(self.device)
            out[key] = (x - mean) / std
        return out

    def denorm(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Undo the z-score normalisation on a dict of tensors (e.g. model predictions)
        """
        out = {}
        for key, x in data.items():
            mean, std = self.stats[key]
            out[key] = x * std + mean
        return out

    def __getitem__(self, index:int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Returns:
            obs_dict: modality-keyed observation dict
            gt_dict: modality-keyed ground truth data
        """
        if self._file is None:
            self._file = h5py.File(self.path, 'r')

        ep_name, obs_start = self.samples[index]

        obs_end = obs_start+self.horizon

        future_start = obs_end
        future_end   = obs_end+self.future_steps

        ep = self._file[ep_name]

        raw_obs = {key: torch.from_numpy(ep[h5_key][obs_start:obs_end]) for key, h5_key in OBS_SOURCES.items()}
        raw_gt = {key: torch.from_numpy(ep[h5_key][future_start:future_end]) for key, h5_key in GT_SOURCES.items()}

        return self.norm(raw_obs), self.norm(raw_gt)


def cv_predict(obs: dict[str, torch.Tensor], key: str, steps: int) -> torch.Tensor:
    """
    Constant-velocity extrapolation for a target key, from its matching obs modality.
    Falls back to hold-last (zero velocity) when the obs horizon is a single step.
    Linear extrapolation is exact in normalised space, so obs/gt stay comparable

    Returns a prediction shaped like the gt: [B steps 1 D]
    """
    x = obs[GT_TO_OBS[key]]  # [B t 1 D]

    x_last = x[:, -1]                                                        # [B 1 D]
    v = x[:, -1] - x[:, -2] if x.shape[1] >= 2 else torch.zeros_like(x_last) # [B 1 D]

    k = torch.arange(1, steps + 1, device=x.device, dtype=x.dtype).view(1, steps, 1, 1)
    return x_last.unsqueeze(1) + v.unsqueeze(1) * k                          # [B steps 1 D]


def baseline_losses(loader: DataLoader, loss_fn) -> dict[str, dict[str, float]]:
    """
    Zero (predict the dataset mean) and constant-velocity baselines over a loader,
    averaged like the train script's val loss (mean of per-batch losses).
    The CV baseline uses no more information than the model: hold-last when horizon==1
    """
    zero: dict[str, float] = {}
    cv: dict[str, float] = {}
    with torch.no_grad():
        for obs, gt in loader:
            for key, target in gt.items():
                zero_loss = loss_fn(torch.zeros_like(target), target)
                cv_loss = loss_fn(cv_predict(obs, key, target.shape[1]), target)

                zero[key] = zero.get(key, 0.0) + zero_loss.item()
                cv[key] = cv.get(key, 0.0) + cv_loss.item()

    for losses in (zero, cv):
        for key in losses:
            losses[key] /= len(loader)
        losses["total"] = sum(losses.values())
    return {"zero": zero, "cv": cv}


if __name__ == "__main__":
    dataset = ChemdataLoader(
        data_path="/home/majonez57/Documents/chem_hpt/chemdata/opaque_v2.hdf5",
        horizon_steps=8,
        action_steps=32,
        split="train",
        device='cuda'
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=0,
    )

    batch = next(iter(loader))

    obs, gt = batch

    for key, value in obs.items():
        print(f"{key:20s} {tuple(value.shape)}")

    for key, value in gt.items():
            print(f"{key:20s} {tuple(value.shape)}")
