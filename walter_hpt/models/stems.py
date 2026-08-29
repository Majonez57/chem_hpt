import torch
import torchvision
import torch.nn as nn
from einops import rearrange

class Stem(nn.Module):
    """ 
    Basic HPT Stem
    Processes a sequence of input features into a fixed set of latent tokens
    
    """
    def __init__(self, feat_dim:int, embed_dim: int, out_dim:int, num_heads:int, t_horizon:int, dropout: float = 0.0):
        super().__init__()
        self.t_horizon = t_horizon

        self.input_proj = nn.Linear(feat_dim, embed_dim) # Project features to trunk embed dim

        self.query_tokens = nn.Parameter(torch.randn(1, out_dim, embed_dim) * 0.02) # Initialize our learnable query tokens
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the vision stem
        Args:
            x : Image tensor with shape [B t T feat_dim]
                B - batch size
                t - time (horizon) 
                T - number of tokens

        Returns:
            Output tensor with latent tokens shape [B, out_dim ,embed_dim]
        """
        B = x.shape[0]
        q = self.query_tokens.expand(B, -1, -1) # Same query tokens for every batch

        x = rearrange(x, "B t T D -> B (t T) D") # Concatenate horizon images time-wise (t*T -> N)
        proj_x = self.input_proj(x) # [B N feat_dim] -> [B N embed_dim]

        latent, _ = self.cross_attn(q, proj_x, proj_x, need_weights=False) # -> [B out_dim embed_dim]

        # ? Maybe add a residual connection: latent+q

        return self.norm(latent)