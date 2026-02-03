#!/bin/bash
#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_endovis2017_train_spme_fusion
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.err

set -euo pipefail

source /home2020/home/icube/kunyuan/anaconda3/etc/profile.d/conda.sh
conda activate sam3

export OMP_NUM_THREADS=8
export HYDRA_FULL_ERROR=1
export SAM3_DISABLE_TRITON=1
export PYTHONUNBUFFERED=1

REPO=/home2020/home/icube/kunyuan/SurgBench/SAM/sam3
DATA=/home2020/home/icube/kunyuan/SurgBench/surgicaltool/endovis2017
# Official EndoVis2017 release (raw). Used only when PREPARE_DATA is enabled.
ENDOVIS_SRC="${ENDOVIS_SRC:-/home2020/home/icube/kunyuan/SurgBench/surgicaltool/Endovis2017}"
PREPARE_DATA="${PREPARE_DATA:-auto}"  # 0 | 1 | auto
CAMERA="${CAMERA:-left}"  # left | right

SAM3_PT=$REPO/sam3.pt
BPE=$REPO/sam3/assets/bpe_simple_vocab_16e6.txt.gz

# Optional: detector/domain-adaptation checkpoint (Hydra trainer ckpt with ['model']).
OVERLAY_CKPT="${OVERLAY_CKPT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/sam3_endovis2017_ft_boxprompt/checkpoints/checkpoint_40.pt}"

# Output folder (job-scoped for easy download).
OUT_ROOT="${OUT_ROOT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_fusion_train_v1/job_${SLURM_JOB_ID:-local}}"

# Optional resume from a previous SPME-Fusion checkpoint (spme_fusion_latest.pt).
RESUME="${RESUME:-}"

# Train data selection.
TRAIN_SEQS="${TRAIN_SEQS:-1 2 3 4 5 6 7}"
ALLOWED_CLASSES="${ALLOWED_CLASSES:-}" # e.g. "3 6"
MIN_INIT_AREA="${MIN_INIT_AREA:-50}"
CLIP_LEN="${CLIP_LEN:-4}"
INIT_PROMPT="${INIT_PROMPT:-mask}"  # mask | box

# Prompting (important for EndoVis: instrument names are often brittle).
PROMPT_MODE="${PROMPT_MODE:-visual}"   # class | generic | visual
PROMPT="${PROMPT:-surgical instrument}" # only used when PROMPT_MODE=generic

# Stop by wallclock to match SLURM.
MAX_HOURS="${MAX_HOURS:-11.5}"

# Optim.
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
GRAD_CLIP="${GRAD_CLIP:-0.1}"
SEED="${SEED:-123}"

# SPME signal knobs.
QUERY_POOL="${QUERY_POOL:-top1}"
QUERY_TOPK="${QUERY_TOPK:-5}"
ANCHOR_DET_THR="${ANCHOR_DET_THR:-0.0}"

# Fusion knobs (applied manually inside the trainer).
FUSION_MODE="${FUSION_MODE:-film}"
FUSION_ALPHA="${FUSION_ALPHA:-0.05}"
FUSION_ALPHA_OBJ="${FUSION_ALPHA_OBJ:-0.005}"
FUSION_DET_THR="${FUSION_DET_THR:-0.3}"
FUSION_QCOS_THR="${FUSION_QCOS_THR:-0.5}"
FUSION_QCOS_TEMP="${FUSION_QCOS_TEMP:-20.0}"
FUSION_QCOS_GATE="${FUSION_QCOS_GATE:-sigmoid}"
FUSION_USE_PRESENCE="${FUSION_USE_PRESENCE:-1}"

LOG_EVERY="${LOG_EVERY:-20}"
SAVE_EVERY="${SAVE_EVERY:-500}"

cd "$REPO"
mkdir -p "$OUT_ROOT"

echo "OUT_ROOT=$OUT_ROOT"
echo "DATA=$DATA"
echo "OVERLAY_CKPT=$OVERLAY_CKPT"
echo "TRAIN_SEQS=$TRAIN_SEQS ALLOWED_CLASSES=${ALLOWED_CLASSES:-<all>}"
echo "PROMPT_MODE=$PROMPT_MODE PROMPT=$PROMPT"

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

python -u scripts/train_spme_fusion_endovis2017.py \
  --data-root "$DATA" \
  --train-seqs $TRAIN_SEQS \
  ${ALLOWED_CLASSES:+--allowed-classes $ALLOWED_CLASSES} \
  --min-init-area "$MIN_INIT_AREA" \
  --prompt-mode "$PROMPT_MODE" \
  --prompt "$PROMPT" \
  --base-sam3-pt "$SAM3_PT" \
  --bpe-path "$BPE" \
  ${OVERLAY_CKPT:+--overlay-ckpt "$OVERLAY_CKPT"} \
  --out-dir "$OUT_ROOT/main" \
  --seed "$SEED" \
  --clip-len "$CLIP_LEN" \
  --init-prompt "$INIT_PROMPT" \
  --max-hours "$MAX_HOURS" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --grad-clip "$GRAD_CLIP" \
  --query-pool "$QUERY_POOL" \
  --query-topk "$QUERY_TOPK" \
  --anchor-det-thr "$ANCHOR_DET_THR" \
  --fusion-mode "$FUSION_MODE" \
  --fusion-alpha "$FUSION_ALPHA" \
  --fusion-alpha-obj "$FUSION_ALPHA_OBJ" \
  --fusion-det-thr "$FUSION_DET_THR" \
  --fusion-qcos-thr "$FUSION_QCOS_THR" \
  --fusion-qcos-temp "$FUSION_QCOS_TEMP" \
  --fusion-qcos-gate "$FUSION_QCOS_GATE" \
  --fusion-use-presence "$FUSION_USE_PRESENCE" \
  --log-every "$LOG_EVERY" \
  --save-every "$SAVE_EVERY" \
  ${RESUME:+--resume "$RESUME"}

tar -czf "${OUT_ROOT}.tar.gz" -C "$(dirname "$OUT_ROOT")" "$(basename "$OUT_ROOT")"
echo "Packed: ${OUT_ROOT}.tar.gz"
