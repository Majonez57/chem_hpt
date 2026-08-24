
## `dataset_utils`

This folder contains useful scripts relating to the use of chemdata datasets:

- `data_proc.py` converts a raw chemdata folder produced by WALTER into a structured dataset with preprocessing (such as resnet18 image features)
- `data_proc_lerobot.py` converts a raw chemdata folder produced by WALTER into a lerobot compatible dataset.

---
### Useful commands

Training ACT from saved lerobot dataset:
```python
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=local/opaque_15 \
  --dataset.root=chemdata_lerobot/opaque_15 \
  --dataset.eval_split=0.15 \
  --policy.device=cuda \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/opaque15_act \
  --job_name=opaque15_act \
  --batch_size=8 \
  --steps=50000 \
  --eval_steps=500 \
  --save_freq=5000 \
  --log_freq=100
```