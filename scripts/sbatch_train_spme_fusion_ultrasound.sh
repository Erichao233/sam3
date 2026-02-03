#!/bin/bash
#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_us_train_spme_fusion_ckpt40
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.err

set -euo pipefail

source /home2020/home/icube/kunyuan/anaconda3/etc/profile.d/conda.sh
conda activate sam3

export OMP_NUM_THREADS=8
export HYDRA_FULL_ERROR=1
export SAM3_DISABLE_TRITON=1

REPO=/home2020/home/icube/kunyuan/SurgBench/SAM/sam3
DATA=/home2020/home/icube/kunyuan/SurgBench/Ultrasound/pork_dataset
SAM3_PT=$REPO/sam3.pt
BPE=$REPO/sam3/assets/bpe_simple_vocab_16e6.txt.gz

# Detector checkpoint (ckpt40 insertion-only fine-tune).
CKPT=/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/sam3_us_needle_insertion_only/checkpoints/checkpoint_40.pt

# Insertion boundaries for training clips.
INSERTION_END_TRAIN=$DATA/insertion_end_train.txt

# Output root (overrideable).
OUT_ROOT_DEFAULT=/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/phase2_spme_fusion_train_v1_ckpt40
OUT_ROOT="${OUT_ROOT_OVERRIDE:-$OUT_ROOT_DEFAULT}"

# Optional: resume from a previous checkpoint (spme_fusion_latest.pt or spme_fusion_step*.pt).
RESUME="${RESUME:-}"

# Optional: fast pilot (set FAST=1 via sbatch --export).
FAST="${FAST:-0}"

cd "$REPO"

mkdir -p "$OUT_ROOT"
echo "OUT_ROOT=$OUT_ROOT"
echo "CKPT=$CKPT"
echo "FAST=$FAST"

if [[ "$FAST" == "1" ]]; then
  # ~3 hour quick retrain: 2500 steps should be enough for SPME convergence.
  python scripts/train_spme_fusion_ultrasound.py \
    --frames-root "$DATA/pork" \
    --gt-root "$DATA/gt" \
    --insertion-end-train-txt "$INSERTION_END_TRAIN" \
    --out-dir "$OUT_ROOT/quick_retrain" \
    --base-sam3-pt "$SAM3_PT" \
    --finetune-ckpt "$CKPT" \
    --bpe-path "$BPE" \
    --prompt "ultrasound needle" \
    --clip-len 4 \
    --max-steps 2500 \
    --lr 1e-3 \
    --query-pool topk_weighted \
    --query-topk 5 \
    --anchor-det-thr 0.3 \
    --fusion-mode film \
    --fusion-alpha 0.05 \
    --fusion-alpha-obj 0.005 \
    --fusion-det-thr 0.3 \
    --fusion-qcos-thr 0.5 \
    --fusion-qcos-temp 20 \
    --fusion-qcos-gate sigmoid \
    --fusion-use-presence 1 \
    --log-every 20 \
    --save-every 200 \
    ${RESUME:+--resume "$RESUME"}
else
  # Main run: stop by wallclock to match SLURM time (12h job -> use ~11.5h here).
  python scripts/train_spme_fusion_ultrasound.py \
    --frames-root "$DATA/pork" \
    --gt-root "$DATA/gt" \
    --insertion-end-train-txt "$INSERTION_END_TRAIN" \
    --out-dir "$OUT_ROOT/main" \
    --base-sam3-pt "$SAM3_PT" \
    --finetune-ckpt "$CKPT" \
    --bpe-path "$BPE" \
    --prompt "ultrasound needle" \
    --clip-len 4 \
    --max-hours 11.5 \
    --lr 1e-3 \
    --query-pool topk_weighted \
    --query-topk 5 \
    --anchor-det-thr 0.3 \
    --fusion-mode film \
    --fusion-alpha 0.05 \
    --fusion-alpha-obj 0.005 \
    --fusion-det-thr 0.3 \
    --fusion-qcos-thr 0.5 \
    --fusion-qcos-temp 20 \
    --fusion-qcos-gate sigmoid \
    --fusion-use-presence 1 \
    --log-every 20 \
    --save-every 500 \
    ${RESUME:+--resume "$RESUME"}
fi

echo "=== Done. Training outputs in: $OUT_ROOT ==="
