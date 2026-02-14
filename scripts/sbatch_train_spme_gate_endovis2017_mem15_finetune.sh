#!/bin/bash
# EndoVis2017: finetune learned gate for the mem15 (memory-expanded) tracker.
#
# Why:
# - `SAM3_TRACKER_NUM_MASKMEM=15` shifts the tracker/memory dynamics; a gate trained under mem7 can regress.
# - This finetunes the *same* gate architecture under mem15 so E (learned gate only) stays competitive.
#
# Usage (server):
#   sbatch scripts/sbatch_train_spme_gate_endovis2017_mem15_finetune.sh
#
# Notes:
# - Requires syncing `/home2020/home/icube/kunyuan/SurgBench/SAM/sam3/sam3/model_builder.py` to the server
#   (it provides training-free temporal-pos-enc interpolation for mem expansion).
# - By default we warmstart from an existing gate checkpoint (mem7 sweep best) via `--spme-init-ckpt`,
#   and keep fusion params frozen (`FREEZE_FUSION=1`) for stability.

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -J sam3_endovis2017_train_spme_gate_mem15
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.err

set -euo pipefail

source /home2020/home/icube/kunyuan/anaconda3/etc/profile.d/conda.sh
conda activate sam3

export OMP_NUM_THREADS=8
export HYDRA_FULL_ERROR=1
export SAM3_DISABLE_TRITON=1
export PYTHONUNBUFFERED=1

# Memory expansion (the change we are adapting the gate to).
export SAM3_TRACKER_NUM_MASKMEM="15"

# Gate MLP width (must match the checkpoint). Read at model construction time.
export SAM3_SPME_GATE_HIDDEN="${SAM3_SPME_GATE_HIDDEN:-64}"

REPO=/home2020/home/icube/kunyuan/SurgBench/SAM/sam3
DATA=/home2020/home/icube/kunyuan/SurgBench/surgicaltool/endovis2017
SAM3_PT=$REPO/sam3.pt
BPE=$REPO/sam3/assets/bpe_simple_vocab_16e6.txt.gz

# Warmstart: best ABSENT_W sweep checkpoint (mem7). Override if you want.
SPME_INIT_CKPT="${SPME_INIT_CKPT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_gate_train_sweep_absent/job_16100471/aw0.1/main/checkpoints/spme_gate_latest.pt}"

# Output folder (job-scoped for easy download).
OUT_ROOT="${OUT_ROOT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_gate_train_mem15_ft/job_${SLURM_JOB_ID:-local}}"

# Data selection.
TRAIN_SEQS="${TRAIN_SEQS:-1 2 3 4 5 6 7}"
ALLOWED_CLASSES="${ALLOWED_CLASSES:-}"
MIN_INIT_AREA="${MIN_INIT_AREA:-50}"

# Prompting (visual-only for EndoVis).
PROMPT_MODE="${PROMPT_MODE:-visual}"
PROMPT="${PROMPT:-surgical instrument}"

# Unroll (match the current "15-frame" setting: clip_len=16, gate_frame=14, supervise_frame=15).
CLIP_LEN="${CLIP_LEN:-16}"
GATE_FRAME="${GATE_FRAME:-14}"
SUPERVISE_FRAME="${SUPERVISE_FRAME:-15}"
RANDOMIZE_GATE_FRAME="${RANDOMIZE_GATE_FRAME:-0}"   # 0|1
GATE_FRAME_MIN="${GATE_FRAME_MIN:-1}"
GATE_FRAME_MAX="${GATE_FRAME_MAX:--1}"
SUPERVISE_GAP_MIN="${SUPERVISE_GAP_MIN:-1}"
SUPERVISE_GAP_MAX="${SUPERVISE_GAP_MAX:-1}"
INIT_PROMPT="${INIT_PROMPT:-mask}"
IMAGE_SIZE="${IMAGE_SIZE:-1008}"

# Stop by wallclock (leave a small buffer).
MAX_HOURS="${MAX_HOURS:-5.8}"

# Optim / loss.
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
GRAD_CLIP="${GRAD_CLIP:-0.1}"
SEED="${SEED:-123}"
BCE_W="${BCE_W:-1.0}"
DICE_W="${DICE_W:-1.0}"
ABSENT_W="${ABSENT_W:-0.1}"
PREFER_ABSENT_SUPERVISE="${PREFER_ABSENT_SUPERVISE:-0.7}"
FREEZE_FUSION="${FREEZE_FUSION:-1}"

# SPME signals (keep aligned with eval).
QUERY_POOL="${QUERY_POOL:-top1}"
QUERY_TOPK="${QUERY_TOPK:-5}"
ANCHOR_DET_THR="${ANCHOR_DET_THR:-0.0}"
POINTER_MODE="${POINTER_MODE:-hybrid}"
MATCH_IOU_THR="${MATCH_IOU_THR:-0.1}"
MATCH_TOPK="${MATCH_TOPK:-20}"
SCORE_THR_DET="${SCORE_THR_DET:-0.2}"

# Gate settings (must match warmstart ckpt).
GATE_USE_DECAY="${GATE_USE_DECAY:-1}"
GATE_OCC_NORM="${GATE_OCC_NORM:-10.0}"
GATE_INPUTS="${GATE_INPUTS:-full}"
FUSION_HEAD="${FUSION_HEAD:-1}"
DET_PRESENT_THR="${DET_PRESENT_THR:-0.3}"

# Keep fusion injection small during gate finetune (optional, default on but small).
FUSION_MODE="${FUSION_MODE:-film}"
FUSION_ALPHA="${FUSION_ALPHA:-0.01}"
FUSION_ALPHA_OBJ="${FUSION_ALPHA_OBJ:-0.001}"

LOG_EVERY="${LOG_EVERY:-50}"
SAVE_EVERY="${SAVE_EVERY:-2000}"

cd "$REPO"
mkdir -p "$OUT_ROOT"

echo "OUT_ROOT=$OUT_ROOT"
echo "DATA=$DATA"
echo "SAM3_TRACKER_NUM_MASKMEM=$SAM3_TRACKER_NUM_MASKMEM"
echo "SPME_INIT_CKPT=$SPME_INIT_CKPT"
echo "PROMPT_MODE=$PROMPT_MODE"
echo "GATE_INPUTS=$GATE_INPUTS FUSION_HEAD=$FUSION_HEAD ABSENT_W=$ABSENT_W PREFER_ABSENT_SUPERVISE=$PREFER_ABSENT_SUPERVISE FREEZE_FUSION=$FREEZE_FUSION"

EXTRA_FRAME_ARGS=()
if [[ "$RANDOMIZE_GATE_FRAME" == "1" ]]; then
  EXTRA_FRAME_ARGS+=(--randomize-gate-frame)
  EXTRA_FRAME_ARGS+=(--gate-frame-min "$GATE_FRAME_MIN")
  EXTRA_FRAME_ARGS+=(--gate-frame-max "$GATE_FRAME_MAX")
  EXTRA_FRAME_ARGS+=(--supervise-gap-min "$SUPERVISE_GAP_MIN")
  EXTRA_FRAME_ARGS+=(--supervise-gap-max "$SUPERVISE_GAP_MAX")
fi

python -u scripts/train_spme_gate_endovis2017.py \
  --data-root "$DATA" \
  --train-seqs $TRAIN_SEQS \
  ${ALLOWED_CLASSES:+--allowed-classes $ALLOWED_CLASSES} \
  --min-init-area "$MIN_INIT_AREA" \
  --prompt-mode "$PROMPT_MODE" \
  --prompt "$PROMPT" \
  --base-sam3-pt "$SAM3_PT" \
  --bpe-path "$BPE" \
  --spme-init-ckpt "$SPME_INIT_CKPT" \
  --out-dir "$OUT_ROOT/main" \
  --seed "$SEED" \
  --image-size "$IMAGE_SIZE" \
  --clip-len "$CLIP_LEN" \
  --init-prompt "$INIT_PROMPT" \
  --gate-frame "$GATE_FRAME" \
  --supervise-frame "$SUPERVISE_FRAME" \
  "${EXTRA_FRAME_ARGS[@]}" \
  --max-hours "$MAX_HOURS" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --grad-clip "$GRAD_CLIP" \
  --bce-weight "$BCE_W" \
  --dice-weight "$DICE_W" \
  --absent-weight "$ABSENT_W" \
  --prefer-absent-supervise "$PREFER_ABSENT_SUPERVISE" \
  --query-pool "$QUERY_POOL" \
  --query-topk "$QUERY_TOPK" \
  --anchor-det-thr "$ANCHOR_DET_THR" \
  --pointer-mode "$POINTER_MODE" \
  --match-iou-thr "$MATCH_IOU_THR" \
  --match-topk "$MATCH_TOPK" \
  --score-thr-detection "$SCORE_THR_DET" \
  --occ-norm "$GATE_OCC_NORM" \
  --use-decay "$GATE_USE_DECAY" \
  --gate-inputs "$GATE_INPUTS" \
  --fusion-head "$FUSION_HEAD" \
  --det-present-thr "$DET_PRESENT_THR" \
  --fusion-mode "$FUSION_MODE" \
  --fusion-alpha "$FUSION_ALPHA" \
  --fusion-alpha-obj "$FUSION_ALPHA_OBJ" \
  --freeze-fusion "$FREEZE_FUSION" \
  --log-every "$LOG_EVERY" \
  --save-every "$SAVE_EVERY"

tar -czf "${OUT_ROOT}.tar.gz" -C "$(dirname "$OUT_ROOT")" "$(basename "$OUT_ROOT")"
echo "Packed: ${OUT_ROOT}.tar.gz"
