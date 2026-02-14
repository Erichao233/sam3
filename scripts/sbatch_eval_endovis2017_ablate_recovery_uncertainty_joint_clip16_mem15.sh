#!/bin/bash
# EndoVis2017: ablate "uncertainty-triggered" recovery actions on top of the joint-finetuned mem15 model.
#
# Motivation:
# - det↔trk IoU mismatch can be unreliable under detector domain shift (can be low even when tracker is correct).
# - Instead of using mismatch to drive recovery, we test *uncertainty-triggered* recovery:
#   only intervene (skip-write / re-init) when tracker confidence is low, and an overseer signal is reliable.
#
# Protocol (MICCAI-safe):
# - PROMPT=visual, INIT_PROMPT=mask, INIT_FRAME_MODE=first_present, PROPAGATION_MODE=vg
# - no detector finetune overlay
# - recondition disabled (avoid confounds)
# - mem expansion: SAM3_TRACKER_NUM_MASKMEM=15
#
# Tasks:
#   0: A (baseline), reference
#   1: F (joint gate+fusion), reference
#   2: F + re-init (safe branch only; mismatch trigger OFF) + optional ReID validation
#   3: F + skip-write (triggered by tracker_score < thr; NOT by mismatch)
#   4: F + re-init (safe) + skip-write (tracker_score trigger)
#
# Usage (server):
#   sbatch scripts/sbatch_eval_endovis2017_ablate_recovery_uncertainty_joint_clip16_mem15.sh
#
# Outputs:
#   /home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_ablate_recovery_uncertainty/job_${SLURM_JOB_ID}/task${TASK}/...

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_endovis2017_eval_ablate_recovery_uncertainty
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.err
#SBATCH --exclude=hpc-n968
#SBATCH --array=0-4

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
# Stable defaults: learned-gate safety envelope + overseer-style fusion.
# ---------------------------------------------------------------------------
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET="1"
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_USE_QCOS="1"
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_POW="4.0"

export SAM3_SPME_LEARNED_GATE_OFFSET_STRENGTH="${SAM3_SPME_LEARNED_GATE_OFFSET_STRENGTH:-1.0}"
export SAM3_SPME_LEARNED_GATE_WRITE_STRENGTH="${SAM3_SPME_LEARNED_GATE_WRITE_STRENGTH:-1.0}"
export SAM3_SPME_LEARNED_GATE_DECAY_STRENGTH="${SAM3_SPME_LEARNED_GATE_DECAY_STRENGTH:-1.0}"

export SAM3_SPME_FUSION_EVENT_DRIVEN="${SAM3_SPME_FUSION_EVENT_DRIVEN:-1}"
export SAM3_SPME_FUSION_TRACKER_THR="${SAM3_SPME_FUSION_TRACKER_THR:-0.8}"
# Keep fusion event mode unset => tracker-score trigger (avoid det↔trk IoU mismatch).
unset SAM3_SPME_FUSION_EVENT_MODE SAM3_SPME_FUSION_MISMATCH_IOU_THR SAM3_SPME_PER_OBJECT_ANCHOR_MATCH

export GATE_INPUTS="${GATE_INPUTS:-full}"
export GATE_FUSION_HEAD="${GATE_FUSION_HEAD:-1}"
export GATE_USE_DECAY="${GATE_USE_DECAY:-1}"
export GATE_OCC_NORM="${GATE_OCC_NORM:-10.0}"
export GATE_DET_PRESENT_THR="${GATE_DET_PRESENT_THR:-0.3}"

TASK="${SLURM_ARRAY_TASK_ID}"

# Reset recovery toggles (enabled per-task).
unset SAM3_SPME_REINIT SAM3_SPME_REINIT_ENABLE_MISMATCH SAM3_SPME_REINIT_ALLOW_EMPTY
unset SAM3_SPME_REID
unset SAM3_SPME_SKIP_WRITE_ON_MISMATCH SAM3_SPME_SKIP_WRITE_EVENT_MODE SAM3_SPME_SKIP_WRITE_TRACKER_THR
unset SAM3_SPME_SKIP_WRITE_MODE SAM3_SPME_SKIP_WRITE_USE_DET_QCOS SAM3_SPME_SKIP_WRITE_DET_THR SAM3_SPME_SKIP_WRITE_QCOS_THR

case "$TASK" in
  0)
    export RUN_CONFIGS="A"
    ;;
  1)
    export RUN_CONFIGS="F"
    ;;
  2)
    export RUN_CONFIGS="F"
    # Safe re-init only (no mismatch trigger).
    export SAM3_SPME_REINIT="1"
    export SAM3_SPME_REINIT_ENABLE_MISMATCH="0"
    export SAM3_SPME_REINIT_ALLOW_EMPTY="0"
    # Optional: validate detector candidate identity with a lightweight ReID bank.
    export SAM3_SPME_REID="1"
    ;;
  3)
    export RUN_CONFIGS="F"
    # Skip-write triggered by tracker uncertainty (NOT by det↔trk mismatch).
    export SAM3_SPME_SKIP_WRITE_ON_MISMATCH="1"  # legacy enable flag
    export SAM3_SPME_SKIP_WRITE_EVENT_MODE="tracker_score"
    export SAM3_SPME_SKIP_WRITE_TRACKER_THR="${SAM3_SPME_SKIP_WRITE_TRACKER_THR:-0.8}"
    export SAM3_SPME_SKIP_WRITE_MODE="full"
    # Require a reliable overseer signal before skipping writes.
    export SAM3_SPME_SKIP_WRITE_USE_DET_QCOS="1"
    export SAM3_SPME_SKIP_WRITE_DET_THR="${SAM3_SPME_SKIP_WRITE_DET_THR:-0.7}"
    export SAM3_SPME_SKIP_WRITE_QCOS_THR="${SAM3_SPME_SKIP_WRITE_QCOS_THR:-0.7}"
    ;;
  4)
    export RUN_CONFIGS="F"
    export SAM3_SPME_REINIT="1"
    export SAM3_SPME_REINIT_ENABLE_MISMATCH="0"
    export SAM3_SPME_REINIT_ALLOW_EMPTY="0"
    export SAM3_SPME_REID="1"
    export SAM3_SPME_SKIP_WRITE_ON_MISMATCH="1"
    export SAM3_SPME_SKIP_WRITE_EVENT_MODE="tracker_score"
    export SAM3_SPME_SKIP_WRITE_TRACKER_THR="${SAM3_SPME_SKIP_WRITE_TRACKER_THR:-0.8}"
    export SAM3_SPME_SKIP_WRITE_MODE="full"
    export SAM3_SPME_SKIP_WRITE_USE_DET_QCOS="1"
    export SAM3_SPME_SKIP_WRITE_DET_THR="${SAM3_SPME_SKIP_WRITE_DET_THR:-0.7}"
    export SAM3_SPME_SKIP_WRITE_QCOS_THR="${SAM3_SPME_SKIP_WRITE_QCOS_THR:-0.7}"
    ;;
  *)
    echo "[error] unexpected SLURM_ARRAY_TASK_ID=$TASK"
    exit 1
    ;;
esac

export OUT_ROOT="/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_ablate_recovery_uncertainty/job_${SLURM_JOB_ID}/task${TASK}"

echo "[ablate_recovery_uncertainty] task=$TASK"
echo "[ablate_recovery_uncertainty] RUN_CONFIGS=$RUN_CONFIGS"
echo "[ablate_recovery_uncertainty] OUT_ROOT=$OUT_ROOT"
echo "[ablate_recovery_uncertainty] FUSION_CKPT=$FUSION_CKPT"
echo "[ablate_recovery_uncertainty] GATE_CKPT=$GATE_CKPT"
echo "[ablate_recovery_uncertainty] SAM3_SPME_REINIT=${SAM3_SPME_REINIT:-0} SAM3_SPME_SKIP_WRITE_EVENT_MODE=${SAM3_SPME_SKIP_WRITE_EVENT_MODE:-none}"

bash scripts/sbatch_eval_endovis2017.sh

