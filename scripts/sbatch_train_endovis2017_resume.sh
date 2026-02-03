#!/bin/bash
#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -J sam3_endovis2017_ft_resume
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.err
#SBATCH --exclude=hpc-n968

# Resume training from checkpoint_5 (settings already in config YAML)
# Config has:
#   paths.resume_from_ckpt = checkpoint_5.pt
#   scratch.max_data_epochs = 20
#   trainer.checkpoint.resume_from = ${paths.resume_from_ckpt}

set -euo pipefail

source /home2020/home/icube/kunyuan/anaconda3/etc/profile.d/conda.sh
conda activate sam3

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=16
export HYDRA_FULL_ERROR=1
export SAM3_DISABLE_TRITON=1

REPO=/home2020/home/icube/kunyuan/SurgBench/SAM/sam3
DATA=/home2020/home/icube/kunyuan/SurgBench/surgicaltool/endovis2017
ENDOVIS_SRC="${ENDOVIS_SRC:-/home2020/home/icube/kunyuan/SurgBench/surgicaltool/Endovis2017}"
PREPARE_DATA="${PREPARE_DATA:-auto}"  # 0 | 1 | auto
CAMERA="${CAMERA:-left}"  # left | right

cd "$REPO"

if [[ "$PREPARE_DATA" == "1" || ( "$PREPARE_DATA" == "auto" && ! -d "$DATA/train/image" ) ]]; then
  echo "=== [0/1] Prepare official EndoVis2017 -> canonical layout ==="
  if [[ ! -d "$ENDOVIS_SRC" ]]; then
    echo "[error] ENDOVIS_SRC not found: $ENDOVIS_SRC"
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
  exit 1
fi

echo "=== Resume EndoVis finetuning from checkpoint_5 ==="
echo "Config: max_data_epochs=20, resume_from=checkpoint_5.pt"

# Skip COCO conversion (already done in first run)
# Just run training directly
python sam3/train/train.py \
  -c endovis2017/endovis2017_tool_seg_boxprompt_train.yaml \
  --use-cluster 0 \
  --num-gpus 4 \
  paths.dataset_root="$DATA"

echo "=== Done ==="
