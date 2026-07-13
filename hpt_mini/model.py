"""Minimal Heterogeneous Pre-trained Transformer (HPT) for learning purposes.

This file builds the smallest possible HPT-style policy:

        [Vision Stem] -> [Shared Trunk] -> [Action Head]

Every component is written from scratch so each piece is readable. The pieces
mirror the design of the original HPT package (hpt/models/...) but are stripped
down to the essentials:

  VisionStem  ~  hpt PolicyStem + CrossAttention   (turn an image into a FIXED
                                                    number of latent tokens via
                                                    cross-attention)
  SmallTrunk  ~  hpt SimpleTransformer             (the shared transformer body)
  ActionHead  ~  hpt PolicyHead (MLP)              (latent tokens -> action)

The whole network predicts a 2-D action from a single camera image. The image
has ALREADY been converted into a small set of ResNet feature tokens by
data_prep.py, so this model never sees raw pixels.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """The self-attention operation used inside every transformer block.

    Intuition: each token decides how much to "listen" to every other token.
    For every token we produce a Query (Q), Key (K) and Value (V) vector.
    The dot product Q.K measures relevance; softmax turns it into weights; the
    output is a weighted sum of the Value vectors.

    "Multi-head" splits the embedding into several smaller groups (heads) that
    each run attention independently, letting the model focus on different
    relationships in parallel.

    Shapes (B = batch, N = tokens, D = embed_dim, H = num_heads):
        input  x : (B, N, D)
        output   : (B, N, D)
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # * scale keeps the Q.K dot products from growing large, which would
        #   squash the softmax into near-one-hot attention and kill gradients
        self.scale = self.head_dim ** -0.5

        # one projection produces Q, K and V together (3 * embed_dim outputs)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x)                                   # (B, N, 3*D)
        # split heads: (B, N, 3, H, head_dim) -> (3, B, H, N, head_dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                    # each (B, H, N, head_dim)

        # similarity of every query to every key: (B, H, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)                         # weights sum to 1 over keys
        attn = self.dropout(attn)

        out = attn @ v                                      # (B, H, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)          # merge heads back
        return self.dropout(self.proj(out))


class CrossAttention(nn.Module):
    """Cross-attention: a set of *query* tokens gather information from a
    separate set of *context* tokens.

    This is the mechanism HPT uses in its stems. We keep a fixed number of
    learnable query tokens; they attend to the (possibly many) image/ResNet
    tokens and each one summarizes whatever it finds relevant. The result is a
    FIXED-size set of latent tokens regardless of how many inputs there were.
    That fixed-size property is exactly what lets HPT handle heterogeneous,
    variable-length inputs with one shared trunk.

    Shapes:
        query   : (B, M, D)   our learnable latent tokens
        context : (B, N, D)   e.g. the 9 ResNet tokens of an image
        output  : (B, M, D)
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        # queries come from one projection, keys+values from another
        self.to_q = nn.Linear(embed_dim, embed_dim)
        self.to_kv = nn.Linear(embed_dim, 2 * embed_dim)
        self.to_out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, M, D = query.shape
        N = context.shape[1]
        q = self.to_q(query).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)   # (B,H,M,hd)
        kv = self.to_kv(context).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]                                                                  # (B,H,N,hd)

        attn = (q @ k.transpose(-2, -1)) * self.scale        # (B,H,M,N): each query attends to all keys
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, M, D)    # (B,M,D)
        return self.dropout(self.to_out(out))


class TransformerBlock(nn.Module):
    """One layer of a transformer, "pre-norm" style (as in GPT / ViT).

    Residual connections (the `x + ...`) let gradients flow straight through
    the network and let each layer learn a small correction on top of its
    input. Pre-norm applies LayerNorm BEFORE the sublayer, which trains more
    stably than post-norm and is the modern default.

        x = x + SelfAttention(LayerNorm(x))    # mix information across tokens
        x = x + MLP(LayerNorm(x))              # transform each token on its own
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = embed_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionStem(nn.Module):
    """The input side of HPT: turn a variable number of ResNet image tokens into
    a FIXED set of latent tokens.

    Steps:
      1. project the 512-d ResNet features down to the trunk's embed_dim
      2. take the learnable query tokens (num_query_tokens of them)
      3. cross-attention: each query token reads from the image tokens
      4. layer-norm the result
    """

    def __init__(self, feat_dim: int, embed_dim: int, num_query_tokens: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, embed_dim)
        # ! learnable latent tokens: the core of the Perceiver / HPT idea. These
        #   are parameters the network learns; their number fixes the trunk size.
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, embed_dim) * 0.02)
        self.cross_attn = CrossAttention(embed_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, image_tokens: torch.Tensor) -> torch.Tensor:
        # image_tokens: (B, N, feat_dim)  e.g. (B, 9, 512)
        B = image_tokens.shape[0]
        ctx = self.input_proj(image_tokens)            # (B, N, embed_dim)
        q = self.query_tokens.expand(B, -1, -1)        # (B, num_query, embed_dim)
        latent = self.cross_attn(q, ctx)               # (B, num_query, embed_dim)
        return self.norm(latent)


class SmallTrunk(nn.Module):
    """The shared transformer body. Real HPT trunks are large (16+ blocks); here
    we use just 2 so it trains in seconds and is easy to read.

    A learnable positional embedding is added so the tokens know "where" they
    are, because attention by itself has no notion of order.
    """

    def __init__(self, embed_dim: int, num_blocks: int, num_heads: int, num_tokens: int, dropout: float = 0.0):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, embed_dim) * 0.02)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed
        x = self.blocks(x)
        return self.norm(x)


class ActionHead(nn.Module):
    """Map the trunk's latent tokens to a robot action.

    We average the tokens into one vector (mean pooling) then a small MLP
    regresses the action components. The target action is in NORMALIZED space
    (see dataset.py), so no final activation is applied.
    """

    def __init__(self, embed_dim: int, action_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim // 2
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        pooled = tokens.mean(dim=1)        # (B, embed_dim)
        return self.mlp(pooled)            # (B, action_dim)


class HPTMini(nn.Module):
    """The full minimal policy:  VisionStem -> SmallTrunk -> ActionHead.

        input  : image_tokens (B, N, 512)   precomputed ResNet features per frame
        output : action       (B, action_dim)
    """

    def __init__(
        self,
        feat_dim: int = 512,
        embed_dim: int = 128,
        num_query_tokens: int = 4,
        num_heads: int = 4,
        num_trunk_blocks: int = 2,
        action_dim: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.stem = VisionStem(feat_dim, embed_dim, num_query_tokens, num_heads, dropout)
        self.trunk = SmallTrunk(embed_dim, num_trunk_blocks, num_heads, num_query_tokens, dropout)
        self.head = ActionHead(embed_dim, action_dim)

    def forward(self, image_tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.stem(image_tokens)    # (B, num_query, embed_dim)
        tokens = self.trunk(tokens)         # (B, num_query, embed_dim)
        return self.head(tokens)            # (B, action_dim)
