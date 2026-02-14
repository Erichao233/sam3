#!/bin/bash
# EndoVis2017: evaluate joint-finetuned SPME (gate+fusion) under mem15 + eff_iou fusion weights.
#
# What it runs:
# - A: baseline (no SPME)
# - D: fusion-only (from the fusion ckpt)
# - F: learned gate + fusion (from the joint ckpt; loads spme_* from GATE_CKPT)
#
# Usage (server):
#   sbatch scripts/sbatch_eval_endovis2017_joint_clip16_mem15.sh
#
# Outputs:
#   /home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_joint_clip16_mem15/job_${SLURM_JOB_ID}/...
#
# Notes:
# - MICCAI-safe protocol: PROMPT=visual, INIT_PROMPT=mask, INIT_FRAME_MODE=first_present, PROPAGATION_MODE=vg.
# - Recondition disabled to avoid confounds.

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_endovis2017_eval_joint_clip16_mem15
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Fixed protocol + memory expansion.
# ---------------------------------------------------------------------------
export OVERLAY_CKPT=""
export PROMPT="visual"
export PROPAGATION_MODE="vg"
export INIT_PROMPT="mask"
export INIT_FRAME_MODE="first_present"
export INIT_SELECT_MODE="prob"
export HOTSTART_DELAY="0"
export DEBUG_SPME_SUMMARY="1"
export SAM3_DISABLE_RECONDITION="1"

export SAM3_TRACKER_NUM_MASKMEM="${SAM3_TRACKER_NUM_MASKMEM:-15}"
export SAM3_SPME_GATE_HIDDEN="${SAM3_SPME_GATE_HIDDEN:-64}"

# Best fusion weight mode on our runs.
export ENDOVIS_FUSION_WEIGHT_MODE="${ENDOVIS_FUSION_WEIGHT_MODE:-eff_iou}"

# ---------------------------------------------------------------------------
# Checkpoints.
# ---------------------------------------------------------------------------
export FUSION_CKPT="${FUSION_CKPT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_fusion_train_clip16_mem15/job_16149223/main/checkpoints/spme_fusion_latest.pt}"

# Default: auto-pick latest joint ckpt unless pinned.
export GATE_CKPT="${GATE_CKPT:-}"
JOINT_ROOT="${JOINT_ROOT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_joint_train_clip16_mem15}"
if [[ -z "$GATE_CKPT" ]]; then
  GATE_CKPT="$(ls -t "$JOINT_ROOT"/job_*/main/checkpoints/spme_gate_latest.pt 2>/dev/null | head -n1 || true)"
fi
export GATE_CKPT

if [[ -z "$GATE_CKPT" || ! -f "$GATE_CKPT" ]]; then
  echo "[error] Could not find a joint checkpoint spme_gate_latest.pt under: $JOINT_ROOT"
  echo "        Set GATE_CKPT explicitly (to a joint run) and resubmit."
  exit 1
fi

# ---------------------------------------------------------------------------
# Learned-gate safety envelope (stable, avoids over-suppression on false dets).
# ---------------------------------------------------------------------------
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET="1"
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_USE_QCOS="1"
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_POW="4.0"

# Keep calibration strengths at identity unless explicitly overridden.
export SAM3_SPME_LEARNED_GATE_OFFSET_STRENGTH="${SAM3_SPME_LEARNED_GATE_OFFSET_STRENGTH:-1.0}"
export SAM3_SPME_LEARNED_GATE_WRITE_STRENGTH="${SAM3_SPME_LEARNED_GATE_WRITE_STRENGTH:-1.0}"
export SAM3_SPME_LEARNED_GATE_DECAY_STRENGTH="${SAM3_SPME_LEARNED_GATE_DECAY_STRENGTH:-1.0}"

# Overseer-style fusion injection (prevents regressions).
export SAM3_SPME_FUSION_EVENT_DRIVEN="${SAM3_SPME_FUSION_EVENT_DRIVEN:-1}"
export SAM3_SPME_FUSION_TRACKER_THR="${SAM3_SPME_FUSION_TRACKER_THR:-0.8}"

# Gate config (must match how the checkpoint was trained).
export GATE_INPUTS="${GATE_INPUTS:-full}"
export GATE_FUSION_HEAD="${GATE_FUSION_HEAD:-1}"
export GATE_USE_DECAY="${GATE_USE_DECAY:-1}"
export GATE_OCC_NORM="${GATE_OCC_NORM:-10.0}"
export GATE_DET_PRESENT_THR="${GATE_DET_PRESENT_THR:-0.3}"

# Run A/D/F only.
export RUN_CONFIGS="${RUN_CONFIGS:-A D F}"

export OUT_ROOT="${OUT_ROOT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_joint_clip16_mem15/job_${SLURM_JOB_ID:-local}}"

echo "[eval_joint] OUT_ROOT=$OUT_ROOT"
echo "[eval_joint] RUN_CONFIGS=$RUN_CONFIGS"
echo "[eval_joint] SAM3_TRACKER_NUM_MASKMEM=$SAM3_TRACKER_NUM_MASKMEM ENDOVIS_FUSION_WEIGHT_MODE=$ENDOVIS_FUSION_WEIGHT_MODE"
echo "[eval_joint] FUSION_CKPT=$FUSION_CKPT"
echo "[eval_joint] GATE_CKPT=$GATE_CKPT"

bash scripts/sbatch_eval_endovis2017.sh
