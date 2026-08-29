import torch
import torchvision
import torch.nn as nn
from einops import rearrange


def get_sinusoid_encoding_table(position_start: int, position_end: int, d_hid: int) -> torch.Tensor:
    """Sinusoid position encoding table"""

    d_vec = (1. / torch.pow(10000, 2 * (torch.arange(d_hid) / 2).floor_() / d_hid)).unsqueeze(0).float()

    sinusoid_table = torch.arange(position_start, position_end).unsqueeze(1) * d_vec

    sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return sinusoid_table.unsqueeze(0)


class Stem(nn.Module):
    """ 
    Basic HPT Stem
    Processes a sequence of input features into a fixed set of latent tokens,
    slicing to the data horizon and applying sinusoidal position encodings first
    """
    def __init__(self, feat_dim:int, embed_dim: int, out_dim:int, num_heads:int, t_horizon:int, num_tokens:int, dropout: float = 0.0):
        super().__init__()
        self.t_horizon = t_horizon
        self.num_tokens = num_tokens

        self.input_proj = nn.Linear(feat_dim, embed_dim) # Project features to trunk embed dim

        # Sinusoidal positions over the flattened [t T] input tokens
        table = get_sinusoid_encoding_table(0, t_horizon * num_tokens, feat_dim)
        self.register_buffer("pos_emb", table.view(1, t_horizon, num_tokens, feat_dim))

        self.query_tokens = nn.Parameter(torch.randn(1, out_dim, embed_dim) * 0.02) # Initialize our learnable query tokens
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the vision stem
        Args:
            x : Image tensor with shape [B t_full T feat_dim]
                B - batch size
                t_full - time, only the last t_horizon steps are used
                T - number of inputtokens

        Returns:
            Output tensor with latent tokens shape [B, out_dim ,embed_dim]
        """
        B = x.shape[0]
        q = self.query_tokens.expand(B, -1, -1) # Same query tokens for every batch

        x = x[:, -self.t_horizon:]               # [B t T feat_dim]
        x = x + self.pos_emb                     # Sinusoidal positions on the input tokens
        x = rearrange(x, "B t T D -> B (t T) D") # Concatenate horizon images time-wise (t*T -> N)
        proj_x = self.input_proj(x) # [B N feat_dim] -> [B N embed_dim]

        latent, _ = self.cross_attn(q, proj_x, proj_x, need_weights=False) # -> [B out_dim embed_dim]

        # ? Maybe add a residual connection: latent+q

        return self.norm(latent)