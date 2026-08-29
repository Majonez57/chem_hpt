import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional

from .stems import Stem
from .trunk import StandardTrunk
from .heads import M_Head_MLP


@dataclass
class StemSpec:
    """Hyperparameters of the stem for one input modality"""
    feat_dim: int        # Raw feature dim of the modality
    num_tokens: int      # T, tokens per timestep
    out_dim: int         # Number of latent tokens the stem emits
    num_heads: int = 4   # Cross-attn heads inside the stem
    dropout: float = 0.0

@dataclass
class HeadSpec:
    """Hyperparameters of the head for one output modality"""
    out_dim: int  # Output dim per M token
    hidden_dims: list[int] = field(default_factory=lambda: [256]) # Needs to be a function to create a new mutable list
    dropout: float = 0.0

@dataclass
class DomainSpec:
    """Modalities (stems/heads) active for one domain. Domains may share modalities"""
    stems: list[str]
    heads: list[str]


class BasicHPTModel(nn.Module):
    """
    A basic HPT model
    Holds one stem and one head per modality, and maps each domain onto a subset of
    these, so stems and heads are shared between domains that use the same modality.
    Each batch comes from a single domain, which decides which stems/heads run.
    """
    def __init__(self, stem_specs: dict[str, StemSpec], head_specs: dict[str, HeadSpec],
                 domain_specs: dict[str, DomainSpec], embed_dim: int, num_heads: int,
                 num_blocks: int, t_horizon: int, m_token_dim: int, dropout: float = 0.0):
        super().__init__()

        self.stem_specs = stem_specs
        self.head_specs = head_specs
        self.domain_specs = domain_specs
        self.active_domain: Optional[str] = None

        self.t_horizon = t_horizon      # Only use the last n timesteps of each input
        self.m_token_dim = m_token_dim  # M, action chunk length

        # ! Domains may only reference modalities we actually build
        for d, dspec in domain_specs.items():
            unknown = (set(dspec.stems) - stem_specs.keys()) | (set(dspec.heads) - head_specs.keys())
            if unknown:
                raise ValueError(f"Domain '{d}' references unknown modalities: {sorted(unknown)}")

        self.stems = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        self.modality_embs = nn.ParameterDict()  # Modality token added to each stem's latents

        for mod, spec in stem_specs.items():
            self.stems[mod] = Stem(spec.feat_dim, embed_dim, spec.out_dim, spec.num_heads, t_horizon, spec.num_tokens, spec.dropout)
            self.modality_embs[mod] = nn.Parameter(torch.randn(embed_dim) * 0.02)

        for mod, spec in head_specs.items():
            self.heads[mod] = M_Head_MLP(embed_dim, spec.hidden_dims, spec.out_dim, spec.dropout)

        self.trunk = StandardTrunk(embed_dim, num_heads, num_blocks, m_token_dim, dropout)

    def set_active_domain(self, domain: str) -> None:
        """
        Make `domain` the active one and freeze every stem, head and modality token
        it does not use. Trunk and m-tokens stay trainable, as they are shared.
        """
        if domain not in self.domain_specs:
            raise ValueError(f"Unknown domain '{domain}', expected one of {list(self.domain_specs)}")

        self.active_domain = domain
        dspec = self.domain_specs[domain]

        for mod, stem in self.stems.items():
            stem.requires_grad_(mod in dspec.stems)
        for mod, head in self.heads.items():
            head.requires_grad_(mod in dspec.heads)
        for mod, emb in self.modality_embs.items():
            emb.requires_grad_(mod in dspec.stems)

    def forward(self, data: dict[str, torch.Tensor], domain: Optional[str] = None) -> dict[str, torch.Tensor]:
        """
        Forward pass of the basic HPT model
        Args:
            data : dict of input tensors, modality -> [B t_full T feat_dim]
                   only the last t_horizon timesteps of each input are used
            domain : domain of this batch, falls back to self.active_domain
        Returns:
            dict of head outputs, modality -> [B M out_dim]
        """
        domain = domain if domain is not None else self.active_domain
        if domain is None:
            raise ValueError("No domain given and no active domain set (see set_active_domain)")

        dspec = self.domain_specs[domain]

        # * Run active stems (they handle horizon slicing + positional embeddings), tag latents with a modality token
        latents = []
        for mod in dspec.stems:
            latent = self.stems[mod](data[mod])         # [B out_dim embed_dim]
            latents.append(latent + self.modality_embs[mod])

        trunk_in = torch.concat(latents, dim=1)     # [B N embed_dim]

        # * The first M trunk outputs feed every active head
        out_m, _ = self.trunk(trunk_in)             # [B M embed_dim]

        return {mod: self.heads[mod](out_m) for mod in dspec.heads}
