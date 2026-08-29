import torch
import torchvision
import torch.nn as nn
from einops import rearrange

# TODO The below uses M input tokens representing the action chunk length as in Zhu et al
# TODO I want to also try masking the missing stem inputs

class TransformerBlock(nn.Module):
    """
    Standard single layer of transformer decoder
    """


    def __init__(self, embed_dim: int, num_heads:int, mlp_ratio: int =4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim*mlp_ratio, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.norm1(x)
        x1, _ = self.attn(xn, xn, xn, need_weights=False)
        x1 += x # residual 
        xn2 = self.norm2(x1)
        xn2 = self.mlp(xn2)
        xn2 += x1 # residual

        return xn2

class StandardTrunk(nn.Module):
    """
    The main decoder-transformer trunk of a HPT model.
    Processes concatenated stem features as input, as well as M pre-pended learnable tokens (as in Zhu et al)
    
    """
    def __init__(self, embed_dim:int, num_heads: int, num_blocks: int, m_token_dim: int, dropout: float = 0.0):
        super().__init__()
        self.m_token_dim = m_token_dim
        self.m_tokens = nn.Parameter(torch.randn(1, m_token_dim, embed_dim) *0.02)



        self.blocks = nn.Sequential(*[TransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(embed_dim)


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        """
        Forward pass of the trunk
        Input: 
            x : concatenated feature tensor shape [B N D]
        Returns
            Output tensor of shape [B M D], where M is the number of learnable prepended tokens
            Output tensor of shape [B N D], with the 'rest' of the output
        """
        B = x.shape[0]
        m = self.m_tokens.expand(B, -1, -1)
        mx = torch.concat([m,x], dim=1)

        out_mx = self.blocks(mx)
        out_mx = self.norm(out_mx)

        out_m = out_mx[:, :self.m_token_dim, :]
        out_x = out_mx[:, self.m_token_dim:, :]

        return out_m, out_x