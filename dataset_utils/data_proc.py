""" Process a given chemdata folder.

- Combines data into a single hdf5
- Splits soarms__data into proprio and pose (Only keeping the follower data)
- Crops (ZED RGB,D), downsamples(Z RGB,D, W RGB) and runs ResNet on RGB to obtain features

- Saves everything at 15hz with LOCF alignment

"""
import os
import h5py 
import numpy as np
import cv2
import torch
import torch.nn as nn
import pandas as pd
import json
import time


SOURCE_CHEMDATA_FOLDER_PATH = "/home/majonez57/Documents/chem_hpt/chemdata_raw/raw_aug26/opaque"
TARGET_CHEMDATA_PATH = "/home/majonez57/Documents/chem_hpt/chemdata"
DATASET_NAME = "opaque_v2"
HZ_PER_SOURCE = 15


ZED_CROP_RANGE_W = (280,-93) # Cropping the Zed image (if not already 480)
ZED_CROP_RANGE_H = (0, None)
RESNET_BATCH = 512
# Imagenet norm params 
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resnet18_backbone(device: str = "cuda") -> nn.Module:
    import torchvision

    resnet = torchvision.models.resnet18(weights="DEFAULT")
    backbone = nn.Sequential(*list(resnet.children())[:-2])
    backbone.eval().to(device)
    for p in backbone.parameters(): p.requires_grad = False
    return backbone

def parse_soarms(datapoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    #Parse follower json strings into (joints + gripper, xyzrpy pose)
    arm_pose = []
    arm_joints = []
    for datapoint in datapoints:
        arm_dict = json.loads(datapoint)  # json inside a h5, I know...
        follower_dict = arm_dict["follower"]
        pose = follower_dict["pose"]
        arm_pose.append([round(x, 5) for x in (
            pose["x"],
            pose["y"],
            pose["z"],
            pose["roll"],
            pose["pitch"],
            pose["yaw"],
        )])
        joints = follower_dict["joints"] + [follower_dict["gripper"]]
        arm_joints.append([round(x, 5) for x in joints])
    return np.asarray(arm_joints), np.asarray(arm_pose)

@torch.no_grad()
def precompute_resnet(data: np.array, device: str = 'cuda') -> np.array:
    # precomputes resnet features of a image array
    resnet = resnet18_backbone(device)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    

    resized = np.array([cv2.resize(image,(224,224),interpolation=cv2.INTER_AREA) for image in data])
    n = resized.shape[0]

    features = []
    for start in range(0, n, RESNET_BATCH):
        end = min(start + RESNET_BATCH, n)
        batch = torch.from_numpy(np.asarray(resized[start:end])).float().to(device)
        batch /= 255.0                    # Makes pixels 0..1
        batch = batch.permute(0, 3, 1, 2) # channel first (b, 3, 244, 244)
        batch = (batch - mean) /std
        print(f"[INFO]: extracting resnet features from {batch.shape[0]} images...")
        feature_map = resnet(batch) # (b, 512, h, w)
        _, _, h,w = feature_map.shape
        flat = feature_map.flatten(2).transpose(1,2) # (b, h*w, 512)
        features.append(flat.cpu().numpy().astype(np.float16))

    features = np.concatenate(features, axis=0) # ? since we append them batch-wise (N, h*w, 512)
    return features
# Create new dataset file

episode_n = 0
with h5py.File(f"{TARGET_CHEMDATA_PATH}/{DATASET_NAME}.hdf5", "w") as dataset:
    files = sorted(f for f in os.listdir(SOURCE_CHEMDATA_FOLDER_PATH) if f.endswith(".h5"))

    for file in files:
        timer_start = time.time()
        old_data = h5py.File(f"{SOURCE_CHEMDATA_FOLDER_PATH}/{file}", 'r')

        episode_group = dataset.create_group(f"eps_{episode_n:04d}") # Create group for episode

        old_obs = old_data["observations"]
        old_stamps = old_data["timestamps"]
        v1: np.array = old_stamps["soarms__data"][:]

        # we use the timestamps to ensure good time-alingment. 

        start_time = round(v1.min() * 1000) #round to nearest ms
        end_time   = round(v1.max() * 1000) #round to nearest ms
        grid_timestamps_ms = np.arange(start_time, end_time, int(1000/HZ_PER_SOURCE))

        keep_indices_dict = {}
        for source in old_stamps:
            # here we produce an array of indices for each source, ensuring good time-alignment
            # we use LOCF (last observation carried forward) for images

            nearest_ms: np.array = (old_stamps[source][:] * 1000).round() #timestamps are in s, we want to work in ms ints
            keep_indices = np.searchsorted(nearest_ms, grid_timestamps_ms, side='right') -  1
            keep_indices[keep_indices < 0] = 0 #solves potential edge cases
            keep_indices_dict[source] = keep_indices

        for source in old_obs:
            match source:
                case "soarms__data":
                    # Data from the two arms.
                    # We only want the follower data!
                    n = old_obs[source].shape[0] 

                    data = old_obs[source][:]
                    data = data[keep_indices_dict[source]] # ? fixes increasing order bug
                    arm_joints, arm_pose = parse_soarms(data)

                    dts = episode_group.create_dataset("arm_pose", data=np.array(arm_pose))
                    dts.attrs['labels'] = "x,y,z,roll,pitch,yaw"
                    dts.attrs['mode'] = "robot"

                    dts = episode_group.create_dataset("arm_joints", data=np.array(arm_joints))
                    dts.attrs['mode'] = "robot"

                    print(f"[INFO]: ep {episode_n}. t: {n} -> {len(arm_pose)}. {end_time-start_time}ms")
                case "wrist_cam__image_raw":
                    # RGB Wrist cam
                    # We downsample the image, normalize, and pass it though resnet
                    # This saves the need to run resnet many times.
                    data = old_obs[source][:]
                    data = data[keep_indices_dict[source]]
                    wrist_features = precompute_resnet(data)
                    episode_group.create_dataset("wrist_rgb_feats", data=wrist_features)

                case "zed__zed_node__rgb__color__rect__image":
                    # RBG ZED cam. as above, but with added crop to remove redudant parts of the image
                    data = old_obs[source][:]
                    data = data[keep_indices_dict[source]]

                    # crop for old data (new data will crop at record time.)
                    if data.shape[2] == 853: data = data[:, ZED_CROP_RANGE_H[0]:ZED_CROP_RANGE_H[1],ZED_CROP_RANGE_W[0]:ZED_CROP_RANGE_W[1],:]

                    zed_features = precompute_resnet(data)
                    dts = episode_group.create_dataset("exo_rgb_feats", data=zed_features)
                    dts.attrs['mode'] = "robot"

                case "zed__zed_node__depth__depth_registered":
                    # DEPTH ZED
                    pass
                case _:
                    print(f"[WARN]: Unknown Source '{source}' in {file}. Skipping...")

        episode_n += 1
        print(f"episode {episode_n} processed in {round(time.time()-timer_start,1)}s")

