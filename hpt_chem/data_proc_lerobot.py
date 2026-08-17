"""Convert raw chemdata recordings into a LeRobot dataset (v3.0 format).

Mirrors data_proc.py closely (same 15hz timestamp grid + LOCF alignment,
same per-source handling), but writes a LeRobotDataset instead of a combined
hdf5 and keeps the raw RGB frames as video features (no resnet features).

Dataset keys written:
    observation.state           (6,) float32       5 joints + gripper (follower)
    observation.pose            (6,) float32       x, y, z, roll, pitch, yaw
    action                      (6,) float32       same as state (absolute, teleop replay)
    observation.images.exo      video (3, 480, 480)  ZED rgb, cropped like data_proc.py
    observation.images.wrist    video (3, 480, 640)  wrist cam rgb

! requires `lerobot` to actually write the dataset. Run once with
! DRY_RUN = True first (no lerobot needed) to sanity-check the alignment
! and parsing, then flip to False.
"""
import json
import os
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

# * config (mirrors data_proc.py)

SOURCE_CHEMDATA_FOLDER_PATH = "/home/majonez57/Documents/chem_hpt/chemdata_raw/aug7/opaque"
TARGET_CHEMDATA_PATH = "/home/majonez57/Documents/chem_hpt/lerobot_data"
DATASET_NAME = "opaque_screwdriver"
REPO_ID = f"local/{DATASET_NAME}"

HZ_PER_SOURCE = 15
TASK = "pick and place screwdriver"
VAL_EPISODES = (17, 18, 19)  # ? last 3 of 20, sorted by filename

ZED_CROP_RANGE_W = (280, -93)  # width crop for legacy 853px ZED frames
ZED_CROP_RANGE_H = (0, None)

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
POSE_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")

DRY_RUN = True  # ? flip to False once lerobot is installed

FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (6,), "names": {"joints": list(JOINT_NAMES)}},
    "observation.pose": {"dtype": "float32", "shape": (6,), "names": {"pose": list(POSE_NAMES)}},
    "action": {"dtype": "float32", "shape": (6,), "names": {"joints": list(JOINT_NAMES)}},
    "observation.images.exo": {"dtype": "video", "shape": (3, 480, 480), "names": ["channels", "height", "width"]},
    "observation.images.wrist": {"dtype": "video", "shape": (3, 480, 640), "names": ["channels", "height", "width"]},
}


def locf_align(source_stamps: np.ndarray, grid_ms: np.ndarray) -> np.ndarray:
    """LOCF-align one source onto the uniform ms grid (same math as data_proc.py)."""
    nearest_ms = (source_stamps * 1000).round()  # timestamps are in s, work in int ms
    keep_indices = np.searchsorted(nearest_ms, grid_ms, side="right") - 1
    keep_indices[keep_indices < 0] = 0  # solves potential edge cases
    return keep_indices


def parse_soarms(datapoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Parse follower json strings into (joints + gripper, xyzrpy pose)."""
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


def crop_zed(data: np.ndarray) -> np.ndarray:
    """Crop legacy 853px-wide ZED frames (newer data crops at record time)."""
    if data.shape[2] == 853:
        data = data[:, ZED_CROP_RANGE_H[0]:ZED_CROP_RANGE_H[1], ZED_CROP_RANGE_W[0]:ZED_CROP_RANGE_W[1], :]
    return data


def write_val_split(dataset_root: Path) -> None:
    """Assign train/val splits in meta/episodes.jsonl (post-hoc pass).

    ? lerobot v3.0 keeps the per-episode split field in episodes.jsonl;
    ? verify this still holds for the installed lerobot version.
    """
    episodes_path = Path(dataset_root) / REPO_ID / "meta" / "episodes.jsonl"
    entries = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    for entry in entries:
        entry["split"] = "val" if entry["episode_index"] in VAL_EPISODES else "train"
    episodes_path.write_text("".join(json.dumps(e) + "\n" for e in entries))
    print(f"[INFO]: split written ({len(VAL_EPISODES)} val / {len(entries) - len(VAL_EPISODES)} train) -> {episodes_path}")


def process_episode(file_path: str, episode_n: int, dataset: Any) -> int:
    """Align all sources of one raw episode and add its frames (unless dry-running)."""
    arm_joints: np.ndarray | None = None
    arm_pose: np.ndarray | None = None
    exo_frames: np.ndarray | None = None
    wrist_frames: np.ndarray | None = None
    source_stats: dict[str, tuple[int, int]] = {}

    with h5py.File(file_path, "r") as old_data:
        old_obs = old_data["observations"]
        old_stamps = old_data["timestamps"]
        v1: np.ndarray = old_stamps["soarms__data"][:]

        start_time = round(v1.min() * 1000)  # round to nearest ms
        end_time = round(v1.max() * 1000)
        # ! int(1000/15) = 66ms grid step, identical to data_proc.py for frame parity
        grid_timestamps_ms = np.arange(start_time, end_time, int(1000 / HZ_PER_SOURCE))

        keep_indices_dict = {}
        for source in old_stamps:
            keep_indices_dict[source] = locf_align(old_stamps[source][:], grid_timestamps_ms)

        for source in old_obs:
            match source:
                case "soarms__data":
                    # data from the two arms, we only keep the follower
                    data = old_obs[source][:]
                    data = data[keep_indices_dict[source]]  # ? fixes increasing order bug (h5py needs sorted idx)
                    arm_joints, arm_pose = parse_soarms(data)
                case "wrist_cam__image_raw":
                    # rgb wrist cam, kept raw (no resnet here)
                    data = old_obs[source][:]
                    wrist_frames = data[keep_indices_dict[source]]
                case "zed__zed_node__rgb__color__rect__image":
                    # rgb ZED cam, kept raw with the legacy crop applied
                    data = old_obs[source][:]
                    data = data[keep_indices_dict[source]]
                    exo_frames = crop_zed(data)
                case "zed__zed_node__depth__depth_registered":
                    pass  # depth skipped, same as data_proc.py
                case _:
                    print(f"[WARN]: unknown source '{source}' in {file_path}. Skipping...")

        for source, keep in keep_indices_dict.items():
            source_stats[source] = (len(keep), len(np.unique(keep)))

    n = len(grid_timestamps_ms)
    print(f"[INFO]: ep {episode_n}: {end_time - start_time}ms -> {n} frames @ {HZ_PER_SOURCE}hz | "
          f"joints {arm_joints.shape}, exo {exo_frames.shape}, wrist {wrist_frames.shape}")
    for source, (kept, unique) in source_stats.items():
        print(f"        {source}: kept {kept} (unique {unique})")
    print(f"        gripper range: [{arm_joints[:, -1].min():.3f}, {arm_joints[:, -1].max():.3f}]")

    if dataset is None:
        return n

    for t in range(n):
        # ! frames are passed through as recorded (assumed RGB, same convention as data_proc.py)
        dataset.add_frame({
            "observation.state": arm_joints[t].astype(np.float32),
            "observation.pose": arm_pose[t].astype(np.float32),
            "action": arm_joints[t].astype(np.float32),  # teleop replay: action == executed state
            "observation.images.exo": exo_frames[t],
            "observation.images.wrist": wrist_frames[t],
        })
    dataset.save_episode(task=TASK)
    return n


def main() -> None:
    dataset: Any = None
    if not DRY_RUN:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # ! needs lerobot installed

        dataset = LeRobotDataset.create(
            repo_id=REPO_ID,
            root=TARGET_CHEMDATA_PATH,
            fps=HZ_PER_SOURCE,
            features=FEATURES,
        )

    files = sorted(f for f in os.listdir(SOURCE_CHEMDATA_FOLDER_PATH) if f.endswith(".h5"))
    print(f"[INFO]: {len(files)} raw episodes found in {SOURCE_CHEMDATA_FOLDER_PATH} (DRY_RUN={DRY_RUN})")

    total_frames = 0
    for episode_n, file in enumerate(files):
        timer_start = time.time()
        total_frames += process_episode(os.path.join(SOURCE_CHEMDATA_FOLDER_PATH, file), episode_n, dataset)
        print(f"[INFO]: episode {episode_n} ({file}) processed in {round(time.time() - timer_start, 1)}s\n")

    print(f"[INFO]: total frames: {total_frames}")

    if dataset is not None:
        write_val_split(Path(TARGET_CHEMDATA_PATH))
        print("[INFO]: done. train with e.g.:")
        print(f"[INFO]: lerobot-train policy.type=act dataset.repo_id={REPO_ID} "
              f"dataset.root={TARGET_CHEMDATA_PATH} policy.chunk_size=50")


if __name__ == "__main__":
    main()
