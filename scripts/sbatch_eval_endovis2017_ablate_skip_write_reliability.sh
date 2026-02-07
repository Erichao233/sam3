#!/bin/bash
# EndoVis2017: ablate "Selective Memory Write" reliability gating (mismatch-only vs reliable-overseer mismatch).
#
# Motivation (paper-safe):
# - Skip-write v2 (freeze maskmem + obj_ptr) fixes the class-competition bug from v1, but can still
#   increase misses on GT-present frames because det↔trk IoU can be low due to *unreliable detections*.
# - A more principled memory-hygiene rule is: skip writes only when the overseer (detector) is both
#   confident and identity-consistent, *and* it strongly disagrees with the tracker (mismatch).
#
# Protocol (MICCAI-safe):
# - PROMPT=visual, INIT_PROMPT=mask, INIT_FRAME_MODE=first_present, PROPAGATION_MODE=vg
# - no detector finetune overlay
# - recondition disabled (avoid confounds)
#
# Tasks:
#   0: Learned gate only (E), skip-write OFF (reference)
#   1: Learned gate only (E), skip-write ON, mode=full, mismatch-only trigger
#   2: Learned gate only (E), skip-write ON, mode=full, require det_score+qcos (recommended)
#
# Usage (server):
#   sbatch scripts/sbatch_eval_endovis2017_ablate_skip_write_reliability.sh
#
# Outputs:
#   /home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_ablate_skip_write_reliability/job_${SLURM_JOB_ID}/task${TASK}/...

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_endovis2017_eval_ablate_skip_write_reliability
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.err
#SBATCH --array=0-2

set -euo pipefail

# ---------------------------------------------------------------------------
# Fixed eval protocol (MICCAI-safe): visual-only + mask init at first visible frame.
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

# ---------------------------------------------------------------------------
# Best learned-gate safety envelope so far (inference-only).
# ---------------------------------------------------------------------------
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET="1"
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_USE_QCOS="1"
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_POW="4.0"

# Keep calibration strengths at identity.
export SAM3_SPME_LEARNED_GATE_OFFSET_STRENGTH="${SAM3_SPME_LEARNED_GATE_OFFSET_STRENGTH:-1.0}"
export SAM3_SPME_LEARNED_GATE_WRITE_STRENGTH="${SAM3_SPME_LEARNED_GATE_WRITE_STRENGTH:-1.0}"
export SAM3_SPME_LEARNED_GATE_DECAY_STRENGTH="${SAM3_SPME_LEARNED_GATE_DECAY_STRENGTH:-1.0}"

# Gate settings (must match ckpt).
export GATE_INPUTS="${GATE_INPUTS:-full}"
export GATE_FUSION_HEAD="${GATE_FUSION_HEAD:-1}"
export GATE_OCC_NORM="${GATE_OCC_NORM:-10.0}"
export GATE_DET_PRESENT_THR="${GATE_DET_PRESENT_THR:-0.3}"
export GATE_USE_DECAY="${GATE_USE_DECAY:-1}"

# ---------------------------------------------------------------------------
# Selective Memory Write (v2):
# - trigger: det↔trk IoU < thr (after matching)
# - mode=full freezes both maskmem_features and obj_ptr
# ---------------------------------------------------------------------------
export SAM3_SPME_SKIP_WRITE_MISMATCH_IOU_THR="${SAM3_SPME_SKIP_WRITE_MISMATCH_IOU_THR:-0.2}"
export SAM3_SPME_SKIP_WRITE_MODE="full"

# Reliability gating defaults (used only when SAM3_SPME_SKIP_WRITE_USE_DET_QCOS=1).
export SAM3_SPME_SKIP_WRITE_DET_THR="${SAM3_SPME_SKIP_WRITE_DET_THR:-0.7}"
export SAM3_SPME_SKIP_WRITE_QCOS_THR="${SAM3_SPME_SKIP_WRITE_QCOS_THR:-0.7}"

TASK="${SLURM_ARRAY_TASK_ID}"
case "$TASK" in
  0)
    export RUN_CONFIGS="E"
    export SAM3_SPME_SKIP_WRITE_ON_MISMATCH="0"
    export SAM3_SPME_SKIP_WRITE_USE_DET_QCOS="0"
    ;;
  1)
    export RUN_CONFIGS="E"
    export SAM3_SPME_SKIP_WRITE_ON_MISMATCH="1"
    export SAM3_SPME_SKIP_WRITE_USE_DET_QCOS="0"
    ;;
  2)
    export RUN_CONFIGS="E"
    export SAM3_SPME_SKIP_WRITE_ON_MISMATCH="1"
    export SAM3_SPME_SKIP_WRITE_USE_DET_QCOS="1"
    ;;
  *)
    echo "[error] unexpected SLURM_ARRAY_TASK_ID=$TASK"
    exit 1
    ;;
esac

export OUT_ROOT="/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_ablate_skip_write_reliability/job_${SLURM_JOB_ID}/task${TASK}"

echo "[ablate_skip_write_reliability] task=$TASK"
echo "[ablate_skip_write_reliability] RUN_CONFIGS=$RUN_CONFIGS"
echo "[ablate_skip_write_reliability] OUT_ROOT=$OUT_ROOT"
echo "[ablate_skip_write_reliability] GATE_CKPT=$GATE_CKPT"
echo "[ablate_skip_write_reliability] SAM3_DISABLE_RECONDITION=$SAM3_DISABLE_RECONDITION"
echo "[ablate_skip_write_reliability] SAM3_SPME_SKIP_WRITE_ON_MISMATCH=$SAM3_SPME_SKIP_WRITE_ON_MISMATCH"
echo "[ablate_skip_write_reliability] SAM3_SPME_SKIP_WRITE_MODE=$SAM3_SPME_SKIP_WRITE_MODE"
echo "[ablate_skip_write_reliability] SAM3_SPME_SKIP_WRITE_USE_DET_QCOS=$SAM3_SPME_SKIP_WRITE_USE_DET_QCOS"
echo "[ablate_skip_write_reliability] SAM3_SPME_SKIP_WRITE_MISMATCH_IOU_THR=$SAM3_SPME_SKIP_WRITE_MISMATCH_IOU_THR"
echo "[ablate_skip_write_reliability] SAM3_SPME_SKIP_WRITE_DET_THR=$SAM3_SPME_SKIP_WRITE_DET_THR"
echo "[ablate_skip_write_reliability] SAM3_SPME_SKIP_WRITE_QCOS_THR=$SAM3_SPME_SKIP_WRITE_QCOS_THR"

bash scripts/sbatch_eval_endovis2017.sh

