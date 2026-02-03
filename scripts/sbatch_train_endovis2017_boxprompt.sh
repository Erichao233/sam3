#!/bin/bash
#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -J sam3_endovis2017_ft_boxprompt
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.err
#SBATCH --exclude=hpc-n968

set -euo pipefail

source /home2020/home/icube/kunyuan/anaconda3/etc/profile.d/conda.sh
conda activate sam3

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=16
export HYDRA_FULL_ERROR=1
export SAM3_DISABLE_TRITON=1

REPO=/home2020/home/icube/kunyuan/SurgBench/SAM/sam3
# Canonical processed dataset root (created from the official release).
DATA=/home2020/home/icube/kunyuan/SurgBench/surgicaltool/endovis2017
# Optional: official EndoVis2017 raw release root.
ENDOVIS_SRC="${ENDOVIS_SRC:-/home2020/home/icube/kunyuan/SurgBench/surgicaltool/Endovis2017}"
# If set to 1, run official->canonical preprocessing before COCO conversion/training.
# If set to "auto" (default), it runs preprocessing only when "$DATA/train/image" is missing.
PREPARE_DATA="${PREPARE_DATA:-auto}"  # 0 | 1 | auto
CAMERA="${CAMERA:-left}"  # left | right
CATEGORY_NAME_MODE="${CATEGORY_NAME_MODE:-generic}"  # generic | official

cd "$REPO"

echo "CATEGORY_NAME_MODE=$CATEGORY_NAME_MODE"

if [[ "$PREPARE_DATA" == "1" || ( "$PREPARE_DATA" == "auto" && ! -d "$DATA/train/image" ) ]]; then
  echo "=== [0/2] Prepare official EndoVis2017 -> canonical layout ==="
  if [[ ! -d "$ENDOVIS_SRC" ]]; then
    echo "[error] ENDOVIS_SRC not found: $ENDOVIS_SRC"
    echo "Set ENDOVIS_SRC to the official EndoVis2017 root (contains instrument_* folders)."
    exit 1
  fi
  python -u scripts/prepare_endovis2017_official.py \
    --src-root "$ENDOVIS_SRC" \
    --out-root "$DATA" \
    --camera "$CAMERA" \
    --overwrite
fi

if [[ ! -d "$DATA/train/image" ]]; then
  echo "[error] Missing processed dataset folder: $DATA/train/image"
  echo "Either run: sbatch scripts/sbatch_prepare_endovis2017_official.sh"
  echo "Or set: PREPARE_DATA=1 ENDOVIS_SRC=/path/to/Endovis2017 CAMERA=left"
  exit 1
fi

echo "=== [1/2] Convert EndoVis2017 -> COCO (train seq 1..7, val seq 8) ==="
python scripts/endovis2017_to_coco.py \
  --dataset-root "$DATA" \
  --out-dir annotations_endovis2017 \
  --category-name-mode "$CATEGORY_NAME_MODE"

echo "=== [2/2] Finetune (boxprompt segmentation) ==="
# NOTE: Config path is relative to sam3/train/ (Hydra search path)
python sam3/train/train.py \
  -c configs/endovis2017/endovis2017_tool_seg_boxprompt_train.yaml \
  --use-cluster 0 \
  --num-gpus 4 \
  paths.dataset_root="$DATA"

echo "=== Done ==="
