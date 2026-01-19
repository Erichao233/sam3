#!/bin/bash
#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -J sam3_us_needle_insertion_train
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.err
#SBATCH --exclude=hpc-n968

# Fine-tune SAM3 on insertion-only ultrasound data
# This starts from checkpoint_18.pt and trains for 5 more epochs

set -euo pipefail

source /home2020/home/icube/kunyuan/anaconda3/etc/profile.d/conda.sh
conda activate sam3

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=16
export HYDRA_FULL_ERROR=1
export SAM3_DISABLE_TRITON=1

REPO=/home2020/home/icube/kunyuan/SurgBench/SAM/sam3

cd "$REPO"

echo "=== Starting insertion-only fine-tuning ==="
echo "Config: configs/ultrasound/pork_needle_seg_boxprompt_insertion_train.yaml"
echo "Starting from: checkpoint_18.pt"
echo "Training for: 10 epochs on 4 GPUs"

python sam3/train/train.py \
  -c configs/ultrasound/pork_needle_seg_boxprompt_insertion_train.yaml \
  --use-cluster 0 \
  --num-gpus 4

echo "=== Training complete ==="
echo "Checkpoints saved to: /home2020/home/icube/kunyuan/SurgBench/SAM/outputs/sam3_us_needle_insertion_only/checkpoints"
