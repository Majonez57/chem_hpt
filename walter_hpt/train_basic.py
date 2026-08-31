"""
Training a basic HPT model using only robot chemdata
"""
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional

from ChemdataLoader import ChemdataLoader
from models.BasicModel import StemSpec, HeadSpec, DomainSpec, BasicHPTModel

EMBED_DIM  = 512
PRED_ACTION_STEPS = NUM_M_TOKENS = 32 # In this implementation these are the same!
OBS_HORIZON = 1 

DATA_PATH = "/home/majonez57/Documents/chem_hpt/chemdata/opaque_15.hdf5"
BATCH_SIZE = 16
N_EPOCHS   = 100
VAL_RATE   = 10 #Val every N epochs


### TRUNK
NUM_TRUNK_HEADS = 8
NUM_TRANSFORMER_BLOCKS = 16

def main():
    # Read, load, and normalise data
    train_dataset = ChemdataLoader(
        data_path=DATA_PATH,
        horizon_steps=OBS_HORIZON,
        action_steps=PRED_ACTION_STEPS,
        split="train",
        proportion=0.85
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0
    )

    val_dataset = ChemdataLoader(
        data_path=DATA_PATH,
        horizon_steps=OBS_HORIZON,
        action_steps=PRED_ACTION_STEPS,
        split="val",
        proportion=0.15,
        stats=train_dataset.stats
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0
    )

    stem_specs = {
        "cam_exo": StemSpec(feat_dim=512, num_tokens=64, out_dim=16, num_heads=8),
        "cam_wrist": StemSpec(feat_dim=512, num_tokens=64, out_dim=16, num_heads=8),
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

    loader_iter = iter(train_loader)
    model.set_active_domain("robot")
    model.train() # set model to training mode

    loss_fn = nn.HuberLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"This model has :{(params/1000000):.2f} M params")

    # ADD VAL LOSS CHECK
    for epoch in range(N_EPOCHS):
        epoch_loss = 0.0

        for obs, gt in loader_iter:

            optimiser.zero_grad()

            pred = model(obs, "robot")

            loss = 0
            for mod in gt.keys():
                loss += loss_fn(pred[mod].unsqueeze(2), gt[mod])

            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()

        epoch_loss/=len(loader_iter)

        if epoch % VAL_RATE == 0:
            # Run validation
            for obs, gt in val_loader:
                pred = model(obs, "robot")

                loss = 0
                for mod in gt.keys():
                    loss += loss_fn(pred[mod].unsqueeze(2), gt[mod])

            print(f"EPOCH: {epoch} | VAL_LOSS: {loss/len(val_loader)}")


if __name__ == "__main__":
    main()
