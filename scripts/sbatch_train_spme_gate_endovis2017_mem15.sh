#!/bin/bash
# EndoVis2017: train learned gate on expanded Tracker memory (num_maskmem=15).
#
# Motivation (numbers-first + scientific):
# - Training the gate with mem7 but evaluating with mem15 can regress (seen in memexp ablation).
# - mem15 changes temporal conditioning; the gate may need to learn a different write/decay regime.
#
# This script is identical to the standard gate trainer except:
# - `SAM3_TRACKER_NUM_MASKMEM=15`
#
# Usage (server):
#   sbatch scripts/sbatch_train_spme_gate_endovis2017_mem15.sh
#
# Output:
#   /home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_gate_train_mem15/job_${SLURM_JOB_ID}/main/checkpoints/spme_gate_latest.pt
#
# Note:
# - Keep OVERLAY_CKPT empty (visual-only protocol).

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_endovis2017_gate_train_mem15
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
SAM3_PT=$REPO/sam3.pt

OUT_ROOT="${OUT_ROOT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_gate_train_mem15/job_${SLURM_JOB_ID:-local}/main}"

# Training protocol: visual-only + mask init.
export OVERLAY_CKPT=""
export PROMPT_MODE="${PROMPT_MODE:-visual}"
export INIT_PROMPT="${INIT_PROMPT:-mask}"
export INIT_FRAME_MODE="${INIT_FRAME_MODE:-first_present}"

# Memory expansion (critical difference vs standard trainer).
export SAM3_TRACKER_NUM_MASKMEM="15"

# Gate architecture (must match your intended evaluation).
export GATE_INPUTS="${GATE_INPUTS:-full}"
export GATE_FUSION_HEAD="${GATE_FUSION_HEAD:-1}"
export GATE_DET_PRESENT_THR="${GATE_DET_PRESENT_THR:-0.3}"
export GATE_USE_DECAY="${GATE_USE_DECAY:-1}"
export GATE_OCC_NORM="${GATE_OCC_NORM:-10.0}"

# Optional: warm-start from an existing gate checkpoint (recommended).
# Leave empty to train from scratch.
SPME_INIT_CKPT="${SPME_INIT_CKPT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_gate_train_sweep_absent/job_16100471/aw0.1/main/checkpoints/spme_gate_latest.pt}"

cd "$REPO"
mkdir -p "$OUT_ROOT"

echo "[train_gate_mem15] OUT_ROOT=$OUT_ROOT"
echo "[train_gate_mem15] SAM3_TRACKER_NUM_MASKMEM=$SAM3_TRACKER_NUM_MASKMEM"
echo "[train_gate_mem15] SPME_INIT_CKPT=${SPME_INIT_CKPT:-<empty>}"
echo "[train_gate_mem15] PROMPT_MODE=$PROMPT_MODE INIT_PROMPT=$INIT_PROMPT INIT_FRAME_MODE=$INIT_FRAME_MODE"
echo "[train_gate_mem15] GATE_INPUTS=$GATE_INPUTS GATE_FUSION_HEAD=$GATE_FUSION_HEAD GATE_USE_DECAY=$GATE_USE_DECAY"

python -u scripts/train_spme_gate_endovis2017.py \
  --data-root "$DATA" \
  --out-root "$OUT_ROOT" \
  --base-sam3-pt "$SAM3_PT" \
  --prompt-mode "$PROMPT_MODE" \
  --init-prompt "$INIT_PROMPT" \
  --init-frame-mode "$INIT_FRAME_MODE" \
  ${SPME_INIT_CKPT:+--spme-init-ckpt "$SPME_INIT_CKPT"} \
  --gate-inputs "$GATE_INPUTS" \
  --fusion-head "$GATE_FUSION_HEAD" \
  --det-present-thr "$GATE_DET_PRESENT_THR"

