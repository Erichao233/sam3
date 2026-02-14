#!/bin/bash
# EndoVis2017: metric-driven best-numbers check (memory expansion × fusion-weight).
#
# Goal:
# - Identify the best achievable mcIoU (fused multi-class IoU) under a reviewer-safe protocol,
#   before deciding whether we must retrain the learned gate for the mem15 setting.
#
# Protocol (MICCAI-safe):
# - PROMPT=visual, INIT_PROMPT=mask, INIT_FRAME_MODE=first_present, PROPAGATION_MODE=vg
# - no detector finetune overlay
# - recondition disabled (avoid confounds)
#
# Tasks (array):
#   0: mem7  + fusion-weight=tracker_prob  -> run A + E
#   1: mem7  + fusion-weight=eff_iou       -> run A + E
#   2: mem15 + fusion-weight=tracker_prob  -> run A + E
#   3: mem15 + fusion-weight=eff_iou       -> run A + E
#
# Usage (server):
#   sbatch scripts/sbatch_eval_endovis2017_best_numbers_memexp_eff_iou.sh
#
# Outputs:
#   /home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_best_numbers/job_${SLURM_JOB_ID}/mem${MEM}_w${WEIGHT}/...
#
# Notes:
# - `SAM3_TRACKER_NUM_MASKMEM=15` requires syncing `/home2020/home/icube/kunyuan/SurgBench/SAM/sam3/sam3/model_builder.py`
#   to the server (it implements training-free memory expansion via temporal-pos-enc interpolation).

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_endovis2017_eval_best_numbers
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.err
#SBATCH --array=0-3

set -euo pipefail

# ---------------------------------------------------------------------------
# Fixed eval protocol (MICCAI-safe).
# ---------------------------------------------------------------------------
export OVERLAY_CKPT=""
export PROMPT="visual"
export PROPAGATION_MODE="vg"
export INIT_PROMPT="mask"
export INIT_FRAME_MODE="first_present"
export INIT_SELECT_MODE="prob"
export HOTSTART_DELAY="0"
export DEBUG_SPME_SUMMARY="1"

# Disable SAM3 periodic recondition (detector→tracker overwrite).
export SAM3_DISABLE_RECONDITION="1"

# ---------------------------------------------------------------------------
# Checkpoints (pin for reproducibility; override if you want).
# ---------------------------------------------------------------------------
export GATE_CKPT="${GATE_CKPT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_gate_train_sweep_absent/job_16100471/aw0.1/main/checkpoints/spme_gate_latest.pt}"

# Use the current best learned-gate safety envelope by default.
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET="1"
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_USE_QCOS="1"
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_POW="4.0"

# Gate architecture knobs (must match the checkpoint).
export GATE_INPUTS="${GATE_INPUTS:-full}"
export GATE_FUSION_HEAD="${GATE_FUSION_HEAD:-1}"
export GATE_USE_DECAY="${GATE_USE_DECAY:-1}"
export GATE_OCC_NORM="${GATE_OCC_NORM:-10.0}"
export GATE_DET_PRESENT_THR="${GATE_DET_PRESENT_THR:-0.3}"

# Keep SPME-Fusion injection OFF for this best-numbers check (we focus on learned gate + evaluation protocol).
export FUSION_CKPT=""
export RUN_CONFIGS="A E"

TASK="${SLURM_ARRAY_TASK_ID}"
case "$TASK" in
  0) MEM="7";  WEIGHT="tracker_prob" ;;
  1) MEM="7";  WEIGHT="eff_iou" ;;
  2) MEM="15"; WEIGHT="tracker_prob" ;;
  3) MEM="15"; WEIGHT="eff_iou" ;;
  *) echo "[error] unexpected SLURM_ARRAY_TASK_ID=$TASK"; exit 1 ;;
esac

export SAM3_TRACKER_NUM_MASKMEM="${MEM}"
export ENDOVIS_FUSION_WEIGHT_MODE="${WEIGHT}"

export OUT_ROOT="/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_best_numbers/job_${SLURM_JOB_ID}/mem${MEM}_w${WEIGHT}"

echo "[best_numbers] task=$TASK mem=$MEM weight=$WEIGHT"
echo "[best_numbers] RUN_CONFIGS=$RUN_CONFIGS"
echo "[best_numbers] OUT_ROOT=$OUT_ROOT"
echo "[best_numbers] SAM3_TRACKER_NUM_MASKMEM=$SAM3_TRACKER_NUM_MASKMEM"
echo "[best_numbers] ENDOVIS_FUSION_WEIGHT_MODE=$ENDOVIS_FUSION_WEIGHT_MODE"
echo "[best_numbers] GATE_CKPT=$GATE_CKPT"

bash scripts/sbatch_eval_endovis2017.sh

