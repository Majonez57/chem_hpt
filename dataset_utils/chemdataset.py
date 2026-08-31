""" Pytorch Datset interface for a custom hdf5 dataset files

Takes the path to a chemdata folder, and uses the first 
.json it finds as a description of the contained data.

Each training example contains:
    obs    : dictionary over observation modalities
    action : dictionary over actions to predict



"""
from __future__ import annotations
from typing import Tuple
import numpy as no
import torch
from torch.utils.data import Dataset


class chemDataset(Dataset):

    def __init__(
            self,
            dataset_path,
            obs_modality_names: list[str],
            act_modality_names: list[str],

            obs_horizon: int = 1,
            act_horizon: int = 1
    ):
        self.dataset_path = dataset_path

        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon

    def __len__(self) -> int:
        return 0 #TODO

    def __getitem__(self, i: int):
        obs = None



        return obs, action

    