#!/bin/bash
# EndoVis2018: joint finetune learned gate + fusion (clip16 + mem15).
#
# This runs `scripts/train_spme_gate_endovis2018.py` with fusion params unfrozen
# (FREEZE_FUSION=0) and initializes from an existing SPME checkpoint that already
# contains both gate + fusion weights (typically the output of gate-only training).
#
# Usage (server):
#   sbatch scripts/sbatch_train_spme_joint_endovis2018_clip16_mem15.sh
#
# Outputs:
#   /home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2018_spme_joint_train_clip16_mem15/job_${SLURM_JOB_ID}/...

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 08:00:00
#SBATCH -J sam3_endovis2018_train_spme_joint
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.err

set -euo pipefail

source /home2020/home/icube/kunyuan/anaconda3/etc/profile.d/conda.sh
conda activate sam3

export OMP_NUM_THREADS=8
export HYDRA_FULL_ERROR=1
export SAM3_DISABLE_TRITON=1
export PYTHONUNBUFFERED=1

# Memory expansion (must match eval + checkpoint).
export SAM3_TRACKER_NUM_MASKMEM="${SAM3_TRACKER_NUM_MASKMEM:-15}"
# Gate MLP width (read at model construction time).
export SAM3_SPME_GATE_HIDDEN="${SAM3_SPME_GATE_HIDDEN:-64}"

REPO=/home2020/home/icube/kunyuan/SurgBench/SAM/sam3
DATA="${DATA:-/home2020/home/icube/kunyuan/SurgBench/surgicaltool/endovis2018_train}"
ENDOVIS_SRC="${ENDOVIS_SRC:-/home2020/home/icube/kunyuan/SurgBench/surgicaltool/Endovis2018}"
PREPARE_DATA="${PREPARE_DATA:-auto}"  # 0 | 1 | auto
CAMERA="${CAMERA:-left}"  # left | right

SAM3_PT=$REPO/sam3.pt
BPE=$REPO/sam3/assets/bpe_simple_vocab_16e6.txt.gz

OVERLAY_CKPT="${OVERLAY_CKPT:-}"

# Initialize from a gate-only (or prior joint) checkpoint that already contains gate+fusion weights.
SPME_INIT_CKPT="${SPME_INIT_CKPT:-}"
DEFAULT_CKPT_ROOT="/home2020/home/icube/kunyuan/SurgBench/SAM/outputs"
GATE_CKPT_DIR="${GATE_CKPT_DIR:-$DEFAULT_CKPT_ROOT/endovis2018_spme_gate_train_clip16_mem15}"
if [[ -z "$SPME_INIT_CKPT" ]]; then
  SPME_INIT_CKPT="$(ls -t "$GATE_CKPT_DIR"/job_*/main/checkpoints/spme_gate_latest.pt 2>/dev/null | head -n1 || true)"
fi

OUT_ROOT="${OUT_ROOT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2018_spme_joint_train_clip16_mem15/job_${SLURM_JOB_ID:-local}}"
RESUME="${RESUME:-}"

TRAIN_SEQS="${TRAIN_SEQS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16}"
ALLOWED_CLASSES="${ALLOWED_CLASSES:-}" # e.g. "1 2 3 6 9 11"
MIN_INIT_AREA="${MIN_INIT_AREA:-50}"

# Prompting: keep visual-only (MICCAI-safe).
PROMPT_MODE="${PROMPT_MODE:-visual}"   # class | generic | visual
PROMPT="${PROMPT:-surgical instrument}" # only used when PROMPT_MODE=generic

# Unroll (init at t=0).
CLIP_LEN="${CLIP_LEN:-16}"
GATE_FRAME="${GATE_FRAME:-14}"
SUPERVISE_FRAME="${SUPERVISE_FRAME:-15}"

# Joint finetune: randomize gate/supervise positions to avoid overfitting to a single temporal offset.
RANDOMIZE_GATE_FRAME="${RANDOMIZE_GATE_FRAME:-1}"  # 0|1
GATE_FRAME_MIN="${GATE_FRAME_MIN:-10}"
GATE_FRAME_MAX="${GATE_FRAME_MAX:-14}"
SUPERVISE_GAP_MIN="${SUPERVISE_GAP_MIN:-1}"
SUPERVISE_GAP_MAX="${SUPERVISE_GAP_MAX:-1}"

INIT_PROMPT="${INIT_PROMPT:-mask}"  # mask | box
IMAGE_SIZE="${IMAGE_SIZE:-1008}"    # must match SAM3 internal size (avoid RoPE mismatch)

MAX_HOURS="${MAX_HOURS:-7.6}"

# Joint finetune: slightly smaller LR is usually safer.
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
GRAD_CLIP="${GRAD_CLIP:-0.1}"
SEED="${SEED:-123}"

BCE_W="${BCE_W:-1.0}"
DICE_W="${DICE_W:-1.0}"
ABSENT_W="${ABSENT_W:-0.1}"
PREFER_ABSENT_SUPERVISE="${PREFER_ABSENT_SUPERVISE:-0.7}"
FREEZE_FUSION="${FREEZE_FUSION:-0}"

QUERY_POOL="${QUERY_POOL:-top1}"
QUERY_TOPK="${QUERY_TOPK:-5}"
ANCHOR_DET_THR="${ANCHOR_DET_THR:-0.0}"
POINTER_MODE="${POINTER_MODE:-hybrid}" # top1 | per_object | hybrid
MATCH_IOU_THR="${MATCH_IOU_THR:-0.1}"
MATCH_TOPK="${MATCH_TOPK:-20}"
SCORE_THR_DET="${SCORE_THR_DET:-0.2}"

GATE_USE_DECAY="${GATE_USE_DECAY:-1}"
GATE_OCC_NORM="${GATE_OCC_NORM:-10.0}"
GATE_INPUTS="${GATE_INPUTS:-full}"
FUSION_HEAD="${FUSION_HEAD:-1}"
DET_PRESENT_THR="${DET_PRESENT_THR:-0.3}"

FUSION_MODE="${FUSION_MODE:-film}"
FUSION_ALPHA="${FUSION_ALPHA:-0.01}"
FUSION_ALPHA_OBJ="${FUSION_ALPHA_OBJ:-0.001}"

LOG_EVERY="${LOG_EVERY:-50}"
SAVE_EVERY="${SAVE_EVERY:-2000}"

cd "$REPO"
mkdir -p "$OUT_ROOT"

if [[ -z "$SPME_INIT_CKPT" || ! -f "$SPME_INIT_CKPT" ]]; then
  echo "[error] SPME_INIT_CKPT not found: ${SPME_INIT_CKPT:-<empty>}"
  echo "        Provide SPME_INIT_CKPT explicitly or make sure gate-only training finished under:"
  echo "        $GATE_CKPT_DIR/job_*/main/checkpoints/spme_gate_latest.pt"
  exit 1
fi

EXTRA_FRAME_ARGS=()
if [[ "$RANDOMIZE_GATE_FRAME" == "1" ]]; then
  EXTRA_FRAME_ARGS+=(--randomize-gate-frame)
  EXTRA_FRAME_ARGS+=(--gate-frame-min "$GATE_FRAME_MIN")
  EXTRA_FRAME_ARGS+=(--gate-frame-max "$GATE_FRAME_MAX")
  EXTRA_FRAME_ARGS+=(--supervise-gap-min "$SUPERVISE_GAP_MIN")
  EXTRA_FRAME_ARGS+=(--supervise-gap-max "$SUPERVISE_GAP_MAX")
fi

if [[ "$PREPARE_DATA" == "1" || ( "$PREPARE_DATA" == "auto" && ! -d "$DATA/train/image" ) ]]; then
  echo "=== [0/1] Prepare EndoVis2018 training releases -> canonical layout ==="
  if [[ ! -d "$ENDOVIS_SRC" ]]; then
    echo "[error] ENDOVIS_SRC not found: $ENDOVIS_SRC"
    exit 1
  fi
  python -u scripts/prepare_endovis2018_official.py \
    --src-root "$ENDOVIS_SRC" \
    --out-root "$DATA" \
    --camera "$CAMERA" \
    --symlink-images \
    --groups miccai_challenge_2018_release_1 miccai_challenge_release_2 miccai_challenge_release_3 miccai_challenge_release_4 \
    --overwrite
fi
if [[ ! -d "$DATA/train/image" ]]; then
  echo "[error] Missing processed dataset folder: $DATA/train/image"
  exit 1
fi

python -u scripts/train_spme_gate_endovis2018.py \
  --data-root "$DATA" \
  --train-seqs $TRAIN_SEQS \
  ${ALLOWED_CLASSES:+--allowed-classes $ALLOWED_CLASSES} \
  --min-init-area "$MIN_INIT_AREA" \
  --prompt-mode "$PROMPT_MODE" \
  --prompt "$PROMPT" \
  --base-sam3-pt "$SAM3_PT" \
  --bpe-path "$BPE" \
  ${OVERLAY_CKPT:+--overlay-ckpt "$OVERLAY_CKPT"} \
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
  --save-every "$SAVE_EVERY" \
  ${RESUME:+--resume "$RESUME"}

tar -czf "${OUT_ROOT}.tar.gz" -C "$(dirname "$OUT_ROOT")" "$(basename "$OUT_ROOT")"
echo "Packed: ${OUT_ROOT}.tar.gz"

