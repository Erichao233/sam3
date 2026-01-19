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
DATA=/home2020/home/icube/kunyuan/SurgBench/surgicaltool/endovis2017

cd "$REPO"

echo "=== [1/2] Convert EndoVis2017 -> COCO (train seq 1..7, val seq 8) ==="
python scripts/endovis2017_to_coco.py \
  --dataset-root "$DATA" \
  --out-dir annotations_endovis2017

echo "=== [2/2] Finetune (boxprompt segmentation) ==="
# NOTE: Config path is relative to sam3/train/ (Hydra search path)
python sam3/train/train.py \
  -c endovis2017/endovis2017_tool_seg_boxprompt_train.yaml \
  --use-cluster 0 \
  --num-gpus 4 \
  paths.dataset_root="$DATA"

echo "=== Done ==="
