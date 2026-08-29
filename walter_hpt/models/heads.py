import torch
import torchvision
import torch.nn as nn
from typing import Optional
from einops import rearrange

class M_Head_MLP(nn.Module):
    """
    Basic HPT Head
    Processes fixed M tokens from the trunk into a given modality
    This version assumes that M represents the action chunk length
    """
    def __init__(self, feat_dim:int, hidden_dims: list[int], out_dim: int, dropout: float = 0.0):
        super().__init__()

        self.net = [nn.Linear(feat_dim, hidden_dims[0]), nn.GELU(), nn.Dropout(dropout)]
        for i in range(len(hidden_dims) - 1):
            self.net.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.net.append(nn.GELU())
            self.net.append(nn.Dropout(dropout))

        self.net.append(nn.Linear(hidden_dims[-1], out_dim))
        self.net = nn.Sequential(*self.net)


    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of a basic MLP head
        Args:
            m : Feature vector [B M D]
        Returns:
            Output tensor [B M out_dim] matching the shape of the desired output
        """
        return self.net(m)