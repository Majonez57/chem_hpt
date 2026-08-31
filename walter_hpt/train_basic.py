"""
Training a basic HPT model using only robot chemdata
"""
import h5py
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional

from ChemdataLoader import ChemdataLoader, baseline_losses
from models.BasicModel import StemSpec, HeadSpec, DomainSpec, BasicHPTModel

EMBED_DIM  = 512
PRED_ACTION_STEPS = NUM_M_TOKENS = 32 # In this implementation these are the same!
OBS_HORIZON = 1 

DATA_PATH = "/home/majonez57/Documents/chem_hpt/chemdata/opaque_v2.hdf5"
BATCH_SIZE = 16
N_EPOCHS   = 100
VAL_RATE   = 10 #Val every N epochs

TRAIN_SPLIT = 0.85
VAL_SPLIT   = 0.15


### TRUNK
NUM_TRUNK_HEADS = 8
NUM_TRANSFORMER_BLOCKS = 8

def main():

    with h5py.File(DATA_PATH, 'r') as f:
        random.seed(57)
        val_idx  = random.sample(range(len(f)), k=round(len(f)*VAL_SPLIT))
        train_idx = set(range(len(f))) - set(val_idx)

    # Read, load, and normalise data
    train_dataset = ChemdataLoader(
        data_path=DATA_PATH,
        horizon_steps=OBS_HORIZON,
        action_steps=PRED_ACTION_STEPS,
        split="train",
        ep_ids=train_idx
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_dataset = ChemdataLoader(
        data_path=DATA_PATH,
        horizon_steps=OBS_HORIZON,
        action_steps=PRED_ACTION_STEPS,
        split="val",
        ep_ids=val_idx,
        stats=train_dataset.stats
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        drop_last=True
    )

    stem_specs = {
        "cam_exo": StemSpec(feat_dim=512, num_tokens=49, out_dim=16, num_heads=8),
        "cam_wrist": StemSpec(feat_dim=512, num_tokens=49, out_dim=16, num_heads=8),
        "arm_joints": StemSpec(feat_dim=6, num_tokens=1, out_dim=16, num_heads=4),
        "arm_pose": StemSpec(feat_dim=6, num_tokens=1, out_dim=16, num_heads=4),
    }

    head_specs = {
        "pred_joint": HeadSpec(out_dim=6, hidden_dims=[256, 128]),
        "pred_pose": HeadSpec(out_dim=6, hidden_dims=[256, 128])
    }

    domain_specs = {
        "robot": DomainSpec(stems=["cam_exo", "cam_wrist", "arm_joints", "arm_pose"],
                            heads=["pred_joint", "pred_pose"])
    }

    model = BasicHPTModel(
        stem_specs, head_specs, domain_specs, EMBED_DIM, 
        num_trunk_heads=NUM_TRUNK_HEADS, num_blocks=NUM_TRANSFORMER_BLOCKS,
        t_horizon=OBS_HORIZON, m_token_dim=NUM_M_TOKENS
        ).to("cuda")

    model.set_active_domain("robot")

    loss_fn = nn.HuberLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # * Reference numbers for the val loss (hold-last while OBS_HORIZON == 1)
    for name, losses in baseline_losses(val_loader, loss_fn).items():
        print(f"VAL BASELINE {name:4s} | " + " ".join(f"{k} {v:.4f}" for k, v in losses.items()))

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"This model has :{(params/1000000):.2f} M params")

    model.eval()
    with torch.no_grad():
        val_loss = 0
        for obs, gt in val_loader:
            pred = model(obs, "robot")

            loss = 0
            for mod in gt.keys():
                loss += loss_fn(pred[mod].unsqueeze(2), gt[mod])

            val_loss += loss.item()
        val_loss /= len(val_loader)

    print(f"PRE-TRAIN | VAL_LOSS: {round(val_loss,3)}")

    for epoch in range(N_EPOCHS):
        epoch_loss = 0.0
        model.train() # set model to training mode
        for obs, gt in train_loader:

            optimiser.zero_grad()

            pred = model(obs, "robot")

            loss = 0
            for mod in gt.keys():
                loss += loss_fn(pred[mod].unsqueeze(2), gt[mod])

            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()

        epoch_loss/=len(train_loader)

        print(f"EPOCH: {epoch+1}/{N_EPOCHS} | TRAIN_LOSS: {round(epoch_loss,3)}")
        # Run validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            mod_losses = {}

            for mod in gt.keys(): mod_losses[mod] = 0

            for obs, gt in val_loader:
                pred = model(obs, "robot")

                loss = 0
                for mod in gt.keys():
                    current = loss_fn(pred[mod].unsqueeze(2), gt[mod])
                    loss += current
                    mod_losses[mod] += current.item()

                val_loss += loss.item()
            val_loss /= len(val_loader)
            for mod in mod_losses.keys(): mod_losses[mod] /= len(val_loader)

        print(f"VAL_LOSS: {round(val_loss,3)}")
        for mod in mod_losses.keys(): print(f"VAL_LOSS_{mod}: {round(mod_losses[mod],3)}")


if __name__ == "__main__":
    main()
