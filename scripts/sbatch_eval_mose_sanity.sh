#!/bin/bash
#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -J sam3_mose_sanity
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.err

set -euo pipefail

# === ENVIRONMENT SETUP ===
source /home2020/home/icube/kunyuan/anaconda3/etc/profile.d/conda.sh
conda activate sam3
export OMP_NUM_THREADS=8
export HYDRA_FULL_ERROR=1
export SAM3_DISABLE_TRITON=1
export TORCH_CUDNN_V8_API_ENABLED=1

# === PATHS ===
REPO=/home2020/home/icube/kunyuan/SurgBench/SAM/sam3
# Adjust this to where you extracted MOSE on the server
# Assuming structure: $MOSE_ROOT/train/JPEGImages and $MOSE_ROOT/train/Annotations
MOSE_ROOT=/home2020/home/icube/kunyuan/public_datasets/MOSE_release
SAM3_PT=$REPO/sam3.pt
BPE=$REPO/sam3/assets/bpe_simple_vocab_16e6.txt.gz

# Use the base sam3.pt as the "finetuned" checkpoint too (Zero-Shot on MOSE)
CKPT=$SAM3_PT

# Output dir
OUT_ROOT=/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/mose_sanity_check
mkdir -p "$OUT_ROOT"

cd "$REPO"

# === CONFIGURATION ===
# Use actual MOSE train video IDs (8-char hex codes).
# Select 5 representative videos from the train set.
SEQS="001ca3cb 002b4dce 00501424 006e6b64 009acc1a"
# OR if you want to run on whatever is there, leave empty and let script scan
# SEQS=""

MAX_FRAMES="${MAX_FRAMES:-200}"
# Optional: also run on MOSE v1 `valid` split to generate submission-style masks.
# Note: MOSE v1 valid provides GT only for the first frame, so metrics on valid are not meaningful.
RUN_VALID="${RUN_VALID:-0}"

echo "Running MOSE Sanity Check..."
echo "Data: $MOSE_ROOT"
echo "Output: $OUT_ROOT"
echo "SEQS=${SEQS:-<auto>} MAX_FRAMES=$MAX_FRAMES RUN_VALID=$RUN_VALID"

# 1. Baseline (Standard SAM3 Tracker)
echo "=== Baseline ==="
unset SAM3_SPME_FUSION SAM3_SPME_PER_OBJECT
python scripts/eval_mose.py \
  --mose-root "$MOSE_ROOT" \
  --split train \
  --seqs "$SEQS" \
  --out-dir "$OUT_ROOT/baseline" \
  --base-sam3-pt "$SAM3_PT" \
  --finetune-ckpt "$CKPT" \
  --bpe-path "$BPE" \
  --auto-prompt-mode "gt_click" \
  --max-frames "$MAX_FRAMES"

# 2. Untrained Fusion (Zero-Shot Logic Validation)
# Tests if "High Confidence = Strengthen Memory" logic works when semantics are clear.
# Uses init=0 identity MLP, so no training needed.
echo "=== Untrained Fusion (Soft Gate) ==="
export SAM3_SPME_FUSION=1
export SAM3_SPME_FUSION_MODE=film
export SAM3_SPME_FUSION_ALPHA=0.05
export SAM3_SPME_FUSION_ALPHA_OBJ=0.005
export SAM3_SPME_FUSION_DET_THR=0.3
export SAM3_SPME_FUSION_QCOS_THR=0.5
export SAM3_SPME_FUSION_QCOS_TEMP=20
export SAM3_SPME_FUSION_QCOS_GATE=sigmoid
export SAM3_SPME_FUSION_USE_PRESENCE=1
export SAM3_SPME_ANCHOR_DET_THR=0.6
export SAM3_SPME_QUERY_POOL=topk_weighted
export SAM3_SPME_QUERY_TOPK=5
# MOSE is multi-object, so enable per-object
export SAM3_SPME_PER_OBJECT=1

python scripts/eval_mose.py \
  --mose-root "$MOSE_ROOT" \
  --split train \
  --seqs "$SEQS" \
  --out-dir "$OUT_ROOT/fusion_untrained_softgate" \
  --base-sam3-pt "$SAM3_PT" \
  --finetune-ckpt "$CKPT" \
  --bpe-path "$BPE" \
  --auto-prompt-mode "gt_click" \
  --max-frames "$MAX_FRAMES"

if [[ "$RUN_VALID" == "1" ]]; then
  echo "=== VALID submission-style masks (no GT eval after frame 0 in MOSE v1) ==="
  unset SAM3_SPME_FUSION SAM3_SPME_PER_OBJECT
  python scripts/eval_mose.py \
    --mose-root "$MOSE_ROOT" \
    --split valid \
    --seqs "$SEQS" \
    --out-dir "$OUT_ROOT/valid_pred_masks" \
    --base-sam3-pt "$SAM3_PT" \
    --finetune-ckpt "$CKPT" \
    --bpe-path "$BPE" \
    --auto-prompt-mode "gt_click" \
    --max-frames "$MAX_FRAMES" \
    --save-pred-masks
  echo "Saved masks under: $OUT_ROOT/valid_pred_masks/pred_masks (zip this folder for submission)."
fi

echo "Done. Compare results in $OUT_ROOT"
