# hpt_mini — a minimal Heterogeneous Pre-trained Transformer

A tiny, readable HPT-style policy trained from scratch on the **PushT** dataset.
The goal is pedagogical: every component is written from scratch and commented
so that, once understood, you can re-point the same architecture at any other
robot (or human) dataset.

```
 PushT image (96x96x3)
       │
 ┌─────────────────┐
 │  frozen ResNet18 │   ◄── run ONCE, offline, by data_prep.py
 │   (no classifier) │       (kept frozen; not part of training)
 └─────────────────┘
       │
 9 tokens x 512         ◄── cached to disk as data/pusht_resnet18_tokens.npy
       │
 ┌─────────────────┐
 │   VisionStem    │   project 512→128, then CROSS-ATTENTION:
 │                 │   4 learnable query tokens read from the 9 image
 │                 │   tokens → a FIXED 4 tokens x 128
 └─────────────────┘
       │
 4 tokens x 128
       │
 ┌─────────────────┐
 │   SmallTrunk    │   2 transformer blocks (self-attention + MLP),
 │                 │   plus a learnable positional embedding
 └─────────────────┘
       │
 4 tokens x 128
       │
 ┌─────────────────┐
 │   ActionHead    │   mean-pool the 4 tokens → small MLP → 2-D action
 └─────────────────┘
       │
 predicted action (x, y)
```

## Why this architecture is "HPT"

HPT has three parts and the **stem → trunk → head** split is the whole point:

- **Stems** convert *heterogeneous* raw inputs (images, proprioception, points…)
  into one common currency: a fixed number of latent tokens. Cross-attention is
  what makes the count fixed regardless of input size.
- **Trunk** is a single shared transformer that all modalities/domains flow
  through. It is the part you would pre-train once and reuse.
- **Heads** map the shared latent back out to whatever this particular task
  needs (here: a 2-D action).

This repo implements the smallest faithful version of that idea: **one vision
stem, a 2-block trunk, one action head.**

## Files

| file | role |
|------|------|
| `hpt_mini/model.py`    | the network: `MultiHeadSelfAttention`, `CrossAttention`, `TransformerBlock`, `VisionStem`, `SmallTrunk`, `ActionHead`, `HPTMini` |
| `hpt_mini/data_prep.py`| precompute frozen ResNet18 features from PushT images → `data/pusht_resnet18_tokens.npy` |
| `hpt_mini/dataset.py`  | `PushTDataset`: episode-aware sampling + action normalization |
| `hpt_mini/train.py`    | training loop (episode-level train/val split, AdamW, MSE) |

## How to run

From the project root (with `.venv` activated):

```bash
# 1. get PushT (already done in this workspace; reproducible source below)
#    HuggingFace dataset: cadene/pusht_raw  -> data/pusht_raw_dl/pusht_cchi_v7_replay.zarr

# 2. precompute ResNet18 features (run once; ~seconds on GPU)
python -m hpt_mini.data_prep

# 3. train (smoke test, ~under a minute)
python -m hpt_mini.train
```

Checkpoint + normalization stats are written to `outputs/hpt_mini_pusht/best.pt`.

## How to adapt to another dataset

Only **two things** are dataset-specific:

1. **`data_prep.py`** — replace the PushT loading block so it produces an
   `(N, H, W, 3)` image array (values 0–255). Everything after that (normalize,
   ResNet, flatten to tokens) is generic.
2. **`train.py` → `load_data()`** — return your cached features, your actions
   array `(N, action_dim)`, and an `episode_ends` array marking where each
   demonstration finishes.

The model itself (`HPTMini`) does not change. To add a *second* input modality
(e.g. proprioception) later, add a second stem and concatenate its tokens with
the vision tokens before the trunk — that is the full HPT recipe.
