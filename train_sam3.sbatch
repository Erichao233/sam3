#!/bin/bash
#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -J sam3_us_needle_det
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/logs/%x-%j.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/logs/%x-%j.err
#SBATCH --exclude=hpc-n968

source /home2020/home/icube/kunyuan/anaconda3/etc/profile.d/conda.sh
conda activate sam3

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=16
export HYDRA_FULL_ERROR=1

REPO=/home2020/home/icube/kunyuan/SurgBench/SAM/sam3
cd "$REPO"

python sam3/train/train.py \
  -c configs/ultrasound/pork_needle_text_only_train.yaml \
  --use-cluster 0 \
  --num-gpus 4
