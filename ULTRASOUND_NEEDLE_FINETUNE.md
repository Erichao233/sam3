# SAM3 Ultrasound Needle Finetuning (Project Context)

This document summarizes the current ultrasound-needle finetuning workflow in this repo, including dataset prep, training configs, Slurm usage, evaluation/visualization scripts, and the key code changes made to support stable training on typical HPC clusters.

It is written to provide enough context for another agent to reproduce and extend the work without reading the full repository history.

---

## 1) Goal & Training Variants

We are finetuning SAM3 to obtain initial ultrasound needle segmentation capability on a dataset with:

- `pork/<sample_id>/<frame>.png`: grayscale ultrasound frames (stored as PNG; loaded as RGB)
- `gt/<sample_id>/<frame>.png`: binary masks (values in `{0,1}`), often visually “black” in preview because foreground is value `1`
- `train.txt / val.txt / test.txt`: list of sample IDs for split

Two **segmentation** finetuning variants are compared:

1) **Segmentation, text-only prompting**
   - Input prompt: text `"ultrasound needle"`
   - No box prompt provided to the model
   - Trains segmentation loss (mask + dice) + detection losses

2) **Segmentation, text + box prompting**
   - Input prompt: text `"ultrasound needle"` + box prompt
   - Box prompt is produced from GT during training (optionally noised)
   - Trains segmentation loss (mask + dice) + detection losses

Important: A separate earlier config named “text_only” was **detection-only** and is not suitable for comparing segmentation behavior. The current comparison is between:

- `pork_needle_seg_text_only_train.yaml` (text-only segmentation)
- `pork_needle_seg_boxprompt_train.yaml` (text+box segmentation)

---

## 2) Repository Entry Points & Layout

Key files/folders:

- `sam3/train/train.py`
  - CLI entry point for training/eval (Hydra config).
  - Supports local run (`--use-cluster 0`) and Submitit/Slurm mode (`--use-cluster 1`).
  - Modified to accept Hydra overrides via `parse_known_args()` so `key=value` overrides work.

- `sam3/train/trainer.py`
  - Main Trainer implementation (DDP, checkpointing, val/train loops).
  - Updated to:
    - set a safer Triton cache directory per-rank (done in `sam3/train/train.py`)
    - optionally force all forward outputs into the loss graph to avoid certain DDP reduction edge cases
    - load checkpoints in eval-only runs without requiring optimizer state

- Training configs (Hydra YAML) under:
  - `sam3/train/configs/ultrasound/`

- Dataset conversion:
  - `scripts/pork_dataset_to_coco.py` converts the `pork_dataset` folder into COCO JSONs for SAM3’s training API.

- Visualization / “human-check”:
  - `scripts/infer_boxprompt_gtbbox_overlay.py` runs inference and saves overlay images (GT vs prediction) without using the Trainer.
  - `scripts/overlay_coco_predictions.py` overlays COCO-style prediction files onto images (older utility; useful when you already have dumped predictions).

---

## 3) Dataset Conversion (pork_dataset → COCO)

SAM3 training expects images + COCO JSON annotations. We convert the dataset with:

### 3.1 Basic conversion

`scripts/pork_dataset_to_coco.py` reads:

- `train.txt / val.txt / test.txt` (sample IDs)
- For each sample ID, enumerates `pork/<id>/*.png` frames
- Loads corresponding `gt/<id>/<frame>.png` and:
  - binarizes mask via `(mask > 0)`
  - if non-empty: computes bbox + RLE segmentation

Outputs:

- `<dataset_root>/annotations/train.json`
- `<dataset_root>/annotations/val.json`
- `<dataset_root>/annotations/test.json`

### 3.2 Recommended conversion for segmentation training: skip empty frames

Segmentation training is much more stable if you exclude frames with empty masks (to avoid “No find queries” retries).

Use:

```bash
python scripts/pork_dataset_to_coco.py \
  --dataset-root /path/to/pork_dataset \
  --skip-empty \
  --out-dir annotations_nonempty
```

This creates:

- `<dataset_root>/annotations_nonempty/train.json`
- `<dataset_root>/annotations_nonempty/val.json`
- `<dataset_root>/annotations_nonempty/test.json`

The segmentation configs in `sam3/train/configs/ultrasound/` are written to use the `annotations_nonempty/*` paths by default.

---

## 4) Training Configs (Ultrasound)

Configs live in `sam3/train/configs/ultrasound/`:

### 4.1 Text-only segmentation training

- `sam3/train/configs/ultrasound/pork_needle_seg_text_only_train.yaml`

Properties:

- `scratch.enable_segmentation: True`
- `loss_fns_find` includes `Masks` loss (mask focal + dice)
- Prompt fixed to a single category prompt:
  - `ultrasound_train.prompts: '[{"id": 1, "name": "ultrasound needle"}]'`
- No transform that injects input boxes; queries are text-only.
- Uses `annotations_nonempty/train.json` and `annotations_nonempty/val.json`.

Epoch count:

- Controlled by `scratch.max_data_epochs` (mapped to `trainer.max_epochs`).

### 4.2 Text + box segmentation training

- `sam3/train/configs/ultrasound/pork_needle_seg_boxprompt_train.yaml`

Properties:

- Same segmentation losses as above
- Additionally uses the transform:
  - `sam3.train.transforms.filter_query_transforms.TextQueryToVisual(probability=1.0, keep_text_queries=true)`
    - This injects a GT-based `input_bbox` into the query (a “box prompt”).
  - Optionally `RandomizeInputBbox` adds noise to the prompt box.

---

## 5) Model Freezing Strategy (Encoder Frozen, Train Detector/Head)

In both segmentation configs, we keep the heavy encoders frozen to reduce required compute and stabilize finetuning:

- Vision backbone frozen
- Language backbone frozen
- Detector + mask head remain trainable

Implementation:

- `sam3/model_builder.py` was extended to accept:
  - `freeze_vision_backbone: bool`
  - `freeze_language_backbone: bool`
  and sets `requires_grad=False` for those submodules.

Checkpoint size:

- Training checkpoints are small (e.g., ~370MB) because configs set `checkpoint.skip_saving_parameters` to exclude frozen backbone weights.
- For inference, you must load:
  1) the base SAM3 checkpoint (`sam3.pt` ~3GB)
  2) then overlay the finetuned checkpoint weights (small checkpoint) with `strict=False`.

---

## 6) Slurm Usage Pattern (Important)

If you already submit a job via `sbatch`, you should run SAM3 training with:

- `--use-cluster 0`

Reason:

- `--use-cluster 1` triggers Submitit to submit *another* Slurm job from inside your Slurm job (nested submission), which often fails or causes confusion.

Typical sbatch template:

- allocate GPUs with `#SBATCH --gres=gpu:4`
- set `--use-cluster 0 --num-gpus 4`

---

## 7) Cluster Reliability Fixes (What Was Changed & Why)

This project needed several practical fixes to run reliably on shared HPC systems:

### 7.1 Triton cache failures

Symptoms:

- Triton JIT compilation errors or `.so` load errors under `~/.triton/cache` or `/tmp`.

Fixes:

- `sam3/train/train.py` now sets a per-rank `TRITON_CACHE_DIR` under `SLURM_TMPDIR` (or `/tmp/$USER`) if unset.
- `sam3/train/loss/loss_fns.py` supports disabling Triton sigmoid focal loss via:
  - `SAM3_DISABLE_TRITON=1`
  and will also fall back to PyTorch implementation if Triton fails.

### 7.2 Optional OpenCV dependency

`sam3/train/transforms/point_sampling.py` used to hard-import `cv2`.

- We made `cv2` optional so transforms not requiring distance transforms can run without installing OpenCV.

### 7.3 Box prompt shape mismatch

`RandomizeInputBbox` assumed `(N,4)` boxes. Some transforms provide `(4,)`.

- Updated `RandomizeInputBbox` to reshape `(4,) → (1,4)`.

### 7.4 DDP reduction errors for conditional outputs

In some segmentation workflows, some forward outputs do not always contribute to loss on every rank.

- Added optional guard controlled by:
  - `SAM3_DDP_FORCE_OUTPUTS_IN_LOSS=1`
  which adds a `0.0 * sum(all_forward_tensors)` term to loss to keep DDP reduction stable.

### 7.5 Eval-only checkpoint resume without optimizer

When using Trainer for eval-only runs with `optim: null`, resuming a training checkpoint failed.

- Updated `Trainer._load_resuming_checkpoint` to load optimizer/loss state only if they exist.

---

## 8) Inference & Visualization (No Trainer)

To quickly inspect segmentation quality visually without Trainer/eval plumbing:

### 8.1 GT-mask vs prediction overlay (supports text-only prompting)

Script:

- `scripts/infer_boxprompt_gtbbox_overlay.py`

Features:

- Loads base `sam3.pt`
- Loads finetuned checkpoint (`checkpoint_*.pt`) with `strict=False`
- Reads `annotations/test.json`
- Overlays:
  - GT mask (blue)
  - Pred mask (red)
- Optionally visualizes overlap/FN/FP regions distinctly
- Can run with:
  - GT bbox prompt
  - OR text-only prompt via `--no-box-prompt`
- Can filter to sample IDs and preserve folder structure via:
  - `--sample-ids 1,2,3`
  - `--sample-txt /path/to/test.txt`
  - `--preserve-relpath` to save into `out_dir/<sample_id>/<frame>.png`

This is designed for exporting frames per sample for later video creation.

---

## 9) Practical “How To” Checklists

### 9.1 One-time dataset prep

```bash
python scripts/pork_dataset_to_coco.py --dataset-root /path/to/pork_dataset
python scripts/pork_dataset_to_coco.py --dataset-root /path/to/pork_dataset --skip-empty --out-dir annotations_nonempty
```

### 9.2 Run training on Slurm (4×A100)

Use `--use-cluster 0 --num-gpus 4` inside `sbatch` job.

### 9.3 Resume training

Trainer automatically resumes from:

- `${experiment_log_dir}/checkpoints/checkpoint.pt`

as long as you keep the same `experiment_log_dir`.

### 9.4 Export frames for video

Use `scripts/infer_boxprompt_gtbbox_overlay.py` with:

- `--preserve-relpath`
- `--sample-ids` or `--sample-txt`

Then you can `rsync` the output folder to local and run `ffmpeg` to create videos per sample.

---

## 10) Files Added/Modified (Quick Index)

Added:

- `sam3/train/configs/ultrasound/pork_needle_seg_text_only_train.yaml`
- `scripts/pork_dataset_to_coco.py`
- `scripts/infer_boxprompt_gtbbox_overlay.py`
- `scripts/overlay_coco_predictions.py`
- `sam3/train/configs/ultrasound/pork_needle_seg_boxprompt_eval.yaml`
- `sam3/train/configs/ultrasound/pork_needle_text_only_eval.yaml`

Modified (high-level):

- `sam3/model_builder.py` (freeze vision/text backbones)
- `sam3/train/train.py` (Hydra overrides + Triton cache directory)
- `sam3/train/loss/loss_fns.py` (disable Triton focal loss + fallback)
- `sam3/train/loss/sam3_loss.py` (compute `indices` if missing)
- `sam3/train/transforms/point_sampling.py` (optional cv2 + bbox shape support)
- `sam3/train/trainer.py` (DDP stability option + eval resume without optimizer)

---

## 11) Common Pitfalls

- “Text-only checkpoint produces no masks”
  - If a model was trained with `enable_segmentation=False`, it will not learn mask prediction.
  - For segmentation comparison, use the two segmentation configs.

- “No find queries”
  - Usually caused by empty-mask frames. Use `--skip-empty` conversion and point configs at `annotations_nonempty/*`.

- Triton compilation/load issues
  - Set `SAM3_DISABLE_TRITON=1` and rely on PyTorch focal loss.
  - Ensure `TRITON_CACHE_DIR` points to node-local storage (handled automatically by `sam3/train/train.py` for training runs).

---

## 12) Minimal Commands (Fill Paths)

Train text-only segmentation:

```bash
python sam3/train/train.py -c configs/ultrasound/pork_needle_seg_text_only_train.yaml --use-cluster 0 --num-gpus 4
```

Train text+box segmentation:

```bash
python sam3/train/train.py -c configs/ultrasound/pork_needle_seg_boxprompt_train.yaml --use-cluster 0 --num-gpus 4
```

Visualize text-only prompting from a finetuned checkpoint (export per-sample frames):

```bash
python scripts/infer_boxprompt_gtbbox_overlay.py \
  --images-root /path/to/pork_dataset/pork \
  --coco-json /path/to/pork_dataset/annotations/test.json \
  --base-sam3-pt /path/to/sam3.pt \
  --finetune-ckpt /path/to/checkpoint_18.pt \
  --bpe-path /path/to/bpe_simple_vocab_16e6.txt.gz \
  --out-dir /path/to/out_frames \
  --prompt "ultrasound needle" \
  --no-box-prompt \
  --preserve-relpath \
  --sample-ids 1,2,3 \
  --draw-fnfp
```

Video tracking inference (treat each sample as a video) with **three init modes**:

- `gtmask`: initialize tracker from GT mask (first non-empty GT frame by default)
- `gtbbox_text`: initialize from GT bbox (derived from GT mask) + text
- `text_only`: initialize from text prompt only

```bash
python scripts/infer_video_tracking_ultrasound.py \
  --frames-root /path/to/pork_dataset/pork \
  --gt-root /path/to/pork_dataset/gt \
  --sample-ids 1,2,3 \
  --out-dir /path/to/out_tracking \
  --base-sam3-pt /path/to/sam3.pt \
  --finetune-ckpt /path/to/checkpoint_18.pt \
  --bpe-path /path/to/bpe_simple_vocab_16e6.txt.gz \
  --prompt "ultrasound needle" \
  --modes gtmask,gtbbox_text,text_only \
  --draw-fnfp \
  --pred-color 0,255,0 \
  --fn-color 255,0,0 \
  --fp-color 255,255,0 \
  --alpha-gt 0.15 --alpha-pred 0.35 --alpha-fnfp 0.65
```

Output layout:

- `<out-dir>/<mode>/<sample_id>/<frame>.png`
