#!/bin/bash
#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -J sam3_gen_anns
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/logs/%x-%j.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/logs/%x-%j.err
#SBATCH --exclude=hpc-n968

source /home2020/home/icube/kunyuan/anaconda3/etc/profile.d/conda.sh
conda activate sam3

REPO=/home2020/home/icube/kunyuan/SurgBench/SAM/sam3
DATA=/home2020/home/icube/kunyuan/SurgBench/Ultrasound/pork_dataset

cd "$REPO"
#python -m pip install -e ".[train]"
python scripts/pork_dataset_to_coco.py --dataset-root "$DATA"
