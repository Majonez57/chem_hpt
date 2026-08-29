import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

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


class ChemdataLoader(Dataset):
    """
    A class to load samples from saved chemdatasets
    currently single-domain #TODO multidomain

    Inputs and targets are z-score normalised per dimension, with stats computed over the
    full dataset (obs and gt share stats when sourced from the same h5 dataset).
    Use norm()/denorm() to apply/undo this outside of the loader (e.g. inference)
    """

    def __init__(self, data_path: str, horizon_steps: int, action_steps: int, device:str):
        self.path = data_path
        self.horizon = horizon_steps
        self.future_steps = action_steps
        self.device = device

        with h5py.File(data_path, 'r') as f:
            # Find the number of episodes and their lengths
            self.episodes = [
                (ep, f[ep]["arm_joints"].shape[0]) for ep in f.keys()
            ]

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

        self._file: h5py.File | None = None

    @staticmethod
    def _dataset_stats(f: h5py.File, h5_key: str, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-dimension mean/std of a source dataset, accumulated over every episode"""
        n = 0
        s = None
        sq = None
        for ep in f.keys():
            ds = f[ep][h5_key]
            arr = ds[:].astype(np.float64).reshape(-1, ds.shape[-1])
            n += arr.shape[0]
            s = arr.sum(0) if s is None else s + arr.sum(0)
            sq = np.square(arr).sum(0) if sq is None else sq + np.square(arr).sum(0)

        mean = s / n
        std = np.sqrt(np.maximum(sq / n - mean**2, eps))
        return torch.from_numpy(mean).float(), torch.from_numpy(std).float()

    def __len__(self): return 300 #jank

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

        ep_name, N = self.episodes[torch.randint(len(self.episodes), (1,)).item()]

        total = self.horizon + self.future_steps

        # Random episode starting points
        obs_start = torch.randint(
            0,
            N-total+1,
            (1,)
        ).item()
        obs_end = obs_start+self.horizon

        future_start = obs_end
        future_end   = obs_end+self.future_steps

        ep = self._file[ep_name]

        raw_obs = {key: torch.from_numpy(ep[h5_key][obs_start:obs_end]) for key, h5_key in OBS_SOURCES.items()}
        raw_gt = {key: torch.from_numpy(ep[h5_key][future_start:future_end]) for key, h5_key in GT_SOURCES.items()}

        return self.norm(raw_obs), self.norm(raw_gt)


if __name__ == "__main__":
    dataset = ChemdataLoader(
        data_path="/home/majonez57/Documents/chem_hpt/chemdata/opaque_15.hdf5",
        horizon_steps=8,
        action_steps=32,
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
