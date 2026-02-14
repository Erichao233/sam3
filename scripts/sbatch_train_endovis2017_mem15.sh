#!/bin/bash
# sbatch_train_endovis2017_mem15.sh
#
# Plan: Retrain SPME-Gate on a 15-frame memory horizon.
#
# Logic:
# - Scale: Gate MLP is small, 1 GPU is sufficient (unlike det finetuning).
# - Memory: Clip=16 is IO/VRAM heavy, requesting 128GB and long runtime.
# - Env: Must export SAM3_SPME_GATE_HIDDEN (used by model builder).
# - Paths: Executed from REPO root.

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 48:00:00
#SBATCH -J spme_train_mem15
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A.err

set -euo pipefail

# --- Environment Setup ---
# 1. Initialize Conda (Crucial!)
source /home2020/home/icube/kunyuan/anaconda3/etc/profile.d/conda.sh
conda activate sam3

# 2. Key Exports
export OMP_NUM_THREADS=8
export HYDRA_FULL_ERROR=1
export SAM3_DISABLE_TRITON=1
export PYTHONUNBUFFERED=1

# 3. Model Config (Must match eval script)
export SAM3_SPME_GATE_HIDDEN="${SAM3_SPME_GATE_HIDDEN:-64}"
export SAM3_TRACKER_NUM_MASKMEM="15"  # Validates logic + triggers interpolation

# --- Paths ---
REPO="/home2020/home/icube/kunyuan/SurgBench/SAM/sam3"
cd "$REPO"

DATA="/home2020/home/icube/kunyuan/SurgBench/surgicaltool/endovis2017"
# Official EndoVis2017 release (raw). Used only when PREPARE_DATA is enabled.
ENDOVIS_SRC="${ENDOVIS_SRC:-/home2020/home/icube/kunyuan/SurgBench/surgicaltool/Endovis2017}"
PREPARE_DATA="${PREPARE_DATA:-auto}"  # 0 | 1 | auto
CAMERA="${CAMERA:-left}"  # left | right

SAM3_PT="$REPO/sam3.pt"
BPE_PATH="$REPO/sam3/assets/bpe_simple_vocab_16e6.txt.gz"

# Optional: detector/domain-adaptation checkpoint (Hydra trainer ckpt with ['model']).
# Default: empty (train SPME modules without EndoVis detector adaptation unless explicitly provided).
OVERLAY_CKPT="${OVERLAY_CKPT:-}"

# Optional: initialize spme_* from a fusion checkpoint before learning the gate.
SPME_INIT_CKPT="${SPME_INIT_CKPT:-}"

# Output folder (job-scoped for easy download).
OUT_DIR="${OUT_ROOT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_gate_train_mem15/job_${SLURM_JOB_ID:-local}}"

# Optional: resume from an earlier gate run (spme_gate_latest.pt).
RESUME="${RESUME:-}"

# Train data selection.
TRAIN_SEQS="${TRAIN_SEQS:-1 2 3 4 5 6 7}"
ALLOWED_CLASSES="${ALLOWED_CLASSES:-}" # e.g. "3 6"
MIN_INIT_AREA="${MIN_INIT_AREA:-50}"

# Prompting (important for EndoVis: instrument names are often brittle).
PROMPT_MODE="${PROMPT_MODE:-visual}"   # class | generic | visual
PROMPT="${PROMPT:-surgical instrument}" # only used when PROMPT_MODE=generic


# --- Training Hyperparameters ---
# Unroll (init at t=0). You can increase the gap (supervise_frame - gate_frame) to make occlusion learning easier.
# ME: Overridden for 15-frame memory experiment
CLIP_LEN="${CLIP_LEN:-16}"
GATE_FRAME="${GATE_FRAME:-14}"
SUPERVISE_FRAME="${SUPERVISE_FRAME:-15}"
INIT_PROMPT="${INIT_PROMPT:-mask}"  # mask | box
IMAGE_SIZE="${IMAGE_SIZE:-1008}"    # must match SAM3 internal size (avoid RoPE mismatch)

# Stop by wallclock to match SLURM (leave a small buffer for packing logs/ckpts).
MAX_HOURS="${MAX_HOURS:-47.8}"

# Optim.
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
GRAD_CLIP="${GRAD_CLIP:-0.1}"
SEED="${SEED:-123}"

# Loss.
BCE_W="${BCE_W:-1.0}"
DICE_W="${DICE_W:-1.0}"
# If the learned gate collapses to its init (do-nothing), increase ABSENT_W and/or oversample absent supervise frames.
ABSENT_W="${ABSENT_W:-0.1}"
PREFER_ABSENT_SUPERVISE="${PREFER_ABSENT_SUPERVISE:-0.7}"  # reject present-at-supervise clips with prob p
FREEZE_FUSION="${FREEZE_FUSION:-1}"                         # 1 => train gate heads first (recommended)

# SPME signals.
QUERY_POOL="${QUERY_POOL:-topk_weighted}"
QUERY_TOPK="${QUERY_TOPK:-5}"
ANCHOR_DET_THR="${ANCHOR_DET_THR:-0.0}"
POINTER_MODE="${POINTER_MODE:-hybrid}" # top1 | per_object | hybrid
MATCH_IOU_THR="${MATCH_IOU_THR:-0.1}"
MATCH_TOPK="${MATCH_TOPK:-20}"
SCORE_THR_DET="${SCORE_THR_DET:-0.2}"

# Gate settings.
GATE_USE_DECAY="${GATE_USE_DECAY:-1}"
GATE_OCC_NORM="${GATE_OCC_NORM:-10.0}"
GATE_INPUTS="${GATE_INPUTS:-full}"          # full | det3 | det4 (must match eval)
FUSION_HEAD="${FUSION_HEAD:-1}"            # 0 | 1 (decouple fusion strength)
DET_PRESENT_THR="${DET_PRESENT_THR:-0.3}"  # only relevant for det4

# (Optional) fusion injection during gate training (kept small).
FUSION_MODE="${FUSION_MODE:-film}"
FUSION_ALPHA="${FUSION_ALPHA:-0.01}"
FUSION_ALPHA_OBJ="${FUSION_ALPHA_OBJ:-0.001}"

LOG_EVERY="${LOG_EVERY:-50}"
SAVE_EVERY="${SAVE_EVERY:-2000}"
MAX_STEPS="${MAX_STEPS:-50000}"  # ME: Added explicitly for long training

mkdir -p "$OUT_DIR"

# Auto-pick latest fusion checkpoint for initialization (optional).
# Override SPME_INIT_CKPT explicitly if you want to pin a specific run.
DEFAULT_CKPT_ROOT="/home2020/home/icube/kunyuan/SurgBench/SAM/outputs"
FUSION_CKPT_DIR="${FUSION_CKPT_DIR:-$DEFAULT_CKPT_ROOT/endovis2017_spme_fusion_train_v1}"
if [[ -z "$SPME_INIT_CKPT" ]]; then
  SPME_INIT_CKPT="$(ls -t "$FUSION_CKPT_DIR"/job_*/main/checkpoints/spme_fusion_latest.pt 2>/dev/null | head -n1 || true)"
fi

echo "OUT_DIR=$OUT_DIR"
echo "DATA=$DATA"
echo "OVERLAY_CKPT=$OVERLAY_CKPT"
echo "SPME_INIT_CKPT=${SPME_INIT_CKPT:-<none>}"
echo "TRAIN_SEQS=$TRAIN_SEQS ALLOWED_CLASSES=${ALLOWED_CLASSES:-<all>}"
echo "PROMPT_MODE=$PROMPT_MODE PROMPT=$PROMPT"
echo "GATE_INPUTS=$GATE_INPUTS FUSION_HEAD=$FUSION_HEAD DET_PRESENT_THR=$DET_PRESENT_THR PREFER_ABSENT_SUPERVISE=$PREFER_ABSENT_SUPERVISE FREEZE_FUSION=$FREEZE_FUSION"

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

python -u sam3/scripts/train_spme_gate_endovis2017.py \
  --data-root "$DATA" \
  --train-seqs $TRAIN_SEQS \
  ${ALLOWED_CLASSES:+--allowed-classes $ALLOWED_CLASSES} \
  --min-init-area "$MIN_INIT_AREA" \
  --prompt-mode "$PROMPT_MODE" \
  --prompt "$PROMPT" \
  --base-sam3-pt "$SAM3_PT" \
  --bpe-path "$BPE_PATH" \
  ${OVERLAY_CKPT:+--overlay-ckpt "$OVERLAY_CKPT"} \
  ${SPME_INIT_CKPT:+--spme-init-ckpt "$SPME_INIT_CKPT"} \
  --out-dir "$OUT_DIR/main" \
  --seed "$SEED" \
  --image-size "$IMAGE_SIZE" \
  --clip-len "$CLIP_LEN" \
  --init-prompt "$INIT_PROMPT" \
  --gate-frame "$GATE_FRAME" \
  --supervise-frame "$SUPERVISE_FRAME" \
  --max-hours "$MAX_HOURS" \
  --max-steps "$MAX_STEPS" \
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

tar -czf "${OUT_DIR}.tar.gz" -C "$(dirname "$OUT_DIR")" "$(basename "$OUT_DIR")"
echo "Packed: ${OUT_DIR}.tar.gz"
