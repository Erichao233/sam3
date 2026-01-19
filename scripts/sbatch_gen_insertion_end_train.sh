#!/bin/bash
#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -J sam3_gen_insertion_end_train
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.err

# This script generates insertion_end.txt for the training set
# AND creates a filtered train_insertion.json annotation file.

set -euo pipefail

source /home2020/home/icube/kunyuan/anaconda3/etc/profile.d/conda.sh
conda activate sam3

REPO=/home2020/home/icube/kunyuan/SurgBench/SAM/sam3
DATA=/home2020/home/icube/kunyuan/SurgBench/Ultrasound/pork_dataset

cd "$REPO"

echo "=== Step 1: Generate insertion_end.txt for training set ==="
python scripts/find_insertion_end.py \
  --gt-root "$DATA/gt" \
  --sample-txt "$DATA/train.txt" \
  --output "$DATA/insertion_end_train.txt" \
  --window-size 10 \
  --decrease-ratio 0.15

echo "=== Step 2: Filter training annotations to insertion-only ==="
python scripts/filter_annotations_insertion_only.py \
  --input-json "$DATA/annotations/train.json" \
  --output-json "$DATA/annotations/train_insertion.json" \
  --insertion-end-txt "$DATA/insertion_end_train.txt" \
  --frames-root "$DATA/pork"

echo "=== Done. Files created: ==="
echo "  $DATA/insertion_end_train.txt"
echo "  $DATA/annotations/train_insertion.json"
