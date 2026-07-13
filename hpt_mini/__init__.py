"""hpt_mini: a minimal, standalone Heterogeneous Pre-trained Transformer.

Everything is written from scratch (no hidden machinery from the `hpt` package)
so that each piece is readable. Submodules:

  model     - the network:  VisionStem -> SmallTrunk -> ActionHead
  data_prep - precompute frozen ResNet18 features from PushT images
  dataset   - a PyTorch Dataset over those precomputed features
  train     - the training loop (a fast smoke test on PushT)
"""
from .model import HPTMini, VisionStem, SmallTrunk, ActionHead

__all__ = ["HPTMini", "VisionStem", "SmallTrunk", "ActionHead"]
