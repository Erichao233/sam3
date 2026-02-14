#!/bin/bash
# EndoVis2017: ablate "recovery stack" on top of the joint-finetuned mem15 model.
#
# Motivation:
# - Our joint gate+fusion improves drift metrics but can be near-identity and only helps a subset of sequences.
# - This script evaluates two *architecture-level* recovery mechanisms that should improve robustness
#   without blind tuning:
#   (1) Selective memory write (skip-write) on mismatch frames.
#   (2) Overseer re-init on tracker failure, optionally using feature-based ReID for identity safety.
#
# What it runs (MICCAI-safe protocol):
# - PROMPT=visual, INIT_PROMPT=mask, INIT_FRAME_MODE=first_present, PROPAGATION_MODE=vg
# - no detector finetune overlay
# - recondition disabled (avoid confounds)
# - mem expansion: SAM3_TRACKER_NUM_MASKMEM=15
#
# Tasks:
#   0: A + D (baseline + fusion-only), reference
#   1: F (joint gate+fusion), reference
#   2: F + mismatch-driven fusion event mode (more targeted than tracker-score mode)
#   3: F + mismatch-driven fusion + skip-write (freeze updates on drift frames)
#   4: F + mismatch-driven fusion + re-init (mismatch-trigger ON) + ReID (identity-safe recovery)
#   5: F + mismatch-driven fusion + skip-write + re-init + ReID (full recovery stack)
#
# Usage (server):
#   sbatch scripts/sbatch_eval_endovis2017_ablate_recovery_joint_clip16_mem15.sh
#
# Outputs:
#   /home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_ablate_recovery_joint_clip16_mem15/job_${SLURM_JOB_ID}/task${TASK}_...
#
# Notes:
# - Use `spme_debug_summary.json` to verify mechanisms actually fire:
#   - reinit_on_frac, skip_write_on_frac, reid_accept, fusion_on_frac.
#

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_endovis2017_eval_ablate_recovery_joint_clip16_mem15
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.err
#SBATCH --exclude=hpc-n968
#SBATCH --array=0-5

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
# Learned gate safety envelope (stable).
# ---------------------------------------------------------------------------
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET="1"
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_USE_QCOS="1"
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_POW="4.0"

# Keep calibration strengths at identity unless explicitly overridden.
export SAM3_SPME_LEARNED_GATE_OFFSET_STRENGTH="${SAM3_SPME_LEARNED_GATE_OFFSET_STRENGTH:-1.0}"
export SAM3_SPME_LEARNED_GATE_WRITE_STRENGTH="${SAM3_SPME_LEARNED_GATE_WRITE_STRENGTH:-1.0}"
export SAM3_SPME_LEARNED_GATE_DECAY_STRENGTH="${SAM3_SPME_LEARNED_GATE_DECAY_STRENGTH:-1.0}"

# Overseer-style fusion injection (stable). Note: event mode is set per-task below.
export SAM3_SPME_FUSION_EVENT_DRIVEN="${SAM3_SPME_FUSION_EVENT_DRIVEN:-1}"
export SAM3_SPME_FUSION_TRACKER_THR="${SAM3_SPME_FUSION_TRACKER_THR:-0.8}"

# Gate config (must match how the joint checkpoint was trained).
export GATE_INPUTS="${GATE_INPUTS:-full}"
export GATE_FUSION_HEAD="${GATE_FUSION_HEAD:-1}"
export GATE_USE_DECAY="${GATE_USE_DECAY:-1}"
export GATE_OCC_NORM="${GATE_OCC_NORM:-10.0}"
export GATE_DET_PRESENT_THR="${GATE_DET_PRESENT_THR:-0.3}"

# ---------------------------------------------------------------------------
# Recovery defaults (kept minimal; override if needed).
# ---------------------------------------------------------------------------
# Re-init v2.
export SAM3_SPME_REINIT_MISS_THR="${SAM3_SPME_REINIT_MISS_THR:-3}"
export SAM3_SPME_REINIT_CONFIRM_THR="${SAM3_SPME_REINIT_CONFIRM_THR:-2}"
export SAM3_SPME_REINIT_COOLDOWN="${SAM3_SPME_REINIT_COOLDOWN:-8}"
export SAM3_SPME_REINIT_DET_THR="${SAM3_SPME_REINIT_DET_THR:-0.7}"
export SAM3_SPME_REINIT_QCOS_THR="${SAM3_SPME_REINIT_QCOS_THR:-0.7}"
export SAM3_SPME_REINIT_TRACKER_THR="${SAM3_SPME_REINIT_TRACKER_THR:-0.8}"
export SAM3_SPME_REINIT_CONFIRM_IOU_THR="${SAM3_SPME_REINIT_CONFIRM_IOU_THR:-0.2}"
export SAM3_SPME_REINIT_IOU_THR="${SAM3_SPME_REINIT_IOU_THR:-0.2}"
export SAM3_SPME_REINIT_MISMATCH_THR="${SAM3_SPME_REINIT_MISMATCH_THR:-2}"
export SAM3_SPME_REINIT_ALLOW_EMPTY="${SAM3_SPME_REINIT_ALLOW_EMPTY:-0}"

# ReID (feature-bank identity validation for re-init). Only enabled in task 4/5.
export SAM3_SPME_REID_BANK_SIZE="${SAM3_SPME_REID_BANK_SIZE:-20}"
export SAM3_SPME_REID_MARGIN="${SAM3_SPME_REID_MARGIN:-0.01}"
export SAM3_SPME_REID_SELF_THR="${SAM3_SPME_REID_SELF_THR:-0.0}"
export SAM3_SPME_REID_USE_OTHER_BANKS="${SAM3_SPME_REID_USE_OTHER_BANKS:-1}"
export SAM3_SPME_REID_UPDATE="${SAM3_SPME_REID_UPDATE:-1}"

# Skip-write on mismatch (selective memory write). Only enabled in task 3/5.
export SAM3_SPME_SKIP_WRITE_MISMATCH_IOU_THR="${SAM3_SPME_SKIP_WRITE_MISMATCH_IOU_THR:-0.2}"
export SAM3_SPME_SKIP_WRITE_USE_DET_QCOS="${SAM3_SPME_SKIP_WRITE_USE_DET_QCOS:-1}"
export SAM3_SPME_SKIP_WRITE_DET_THR="${SAM3_SPME_SKIP_WRITE_DET_THR:-0.7}"
export SAM3_SPME_SKIP_WRITE_QCOS_THR="${SAM3_SPME_SKIP_WRITE_QCOS_THR:-0.7}"

TASK="${SLURM_ARRAY_TASK_ID}"

# Reset toggles that we turn on per-task.
unset SAM3_SPME_FUSION_EVENT_MODE SAM3_SPME_FUSION_MISMATCH_IOU_THR SAM3_SPME_PER_OBJECT_ANCHOR_MATCH
unset SAM3_SPME_SKIP_WRITE_ON_MISMATCH SAM3_SPME_SKIP_WRITE_MODE
unset SAM3_SPME_REINIT SAM3_SPME_REINIT_ENABLE_MISMATCH
unset SAM3_SPME_REID

case "$TASK" in
  0)
    export RUN_CONFIGS="A D"
    ;;
  1)
    export RUN_CONFIGS="F"
    ;;
  2)
    export RUN_CONFIGS="F"
    export SAM3_SPME_FUSION_EVENT_MODE="mismatch"
    export SAM3_SPME_FUSION_MISMATCH_IOU_THR="${SAM3_SPME_FUSION_MISMATCH_IOU_THR:-0.2}"
    export SAM3_SPME_PER_OBJECT_ANCHOR_MATCH="${SAM3_SPME_PER_OBJECT_ANCHOR_MATCH:-1}"
    ;;
  3)
    export RUN_CONFIGS="F"
    export SAM3_SPME_FUSION_EVENT_MODE="mismatch"
    export SAM3_SPME_FUSION_MISMATCH_IOU_THR="${SAM3_SPME_FUSION_MISMATCH_IOU_THR:-0.2}"
    export SAM3_SPME_PER_OBJECT_ANCHOR_MATCH="${SAM3_SPME_PER_OBJECT_ANCHOR_MATCH:-1}"
    export SAM3_SPME_SKIP_WRITE_ON_MISMATCH="1"
    export SAM3_SPME_SKIP_WRITE_MODE="${SAM3_SPME_SKIP_WRITE_MODE:-full}"
    ;;
  4)
    export RUN_CONFIGS="F"
    export SAM3_SPME_FUSION_EVENT_MODE="mismatch"
    export SAM3_SPME_FUSION_MISMATCH_IOU_THR="${SAM3_SPME_FUSION_MISMATCH_IOU_THR:-0.2}"
    export SAM3_SPME_PER_OBJECT_ANCHOR_MATCH="${SAM3_SPME_PER_OBJECT_ANCHOR_MATCH:-1}"
    export SAM3_SPME_REINIT="1"
    export SAM3_SPME_REINIT_ENABLE_MISMATCH="1"
    export SAM3_SPME_REID="1"
    ;;
  5)
    export RUN_CONFIGS="F"
    export SAM3_SPME_FUSION_EVENT_MODE="mismatch"
    export SAM3_SPME_FUSION_MISMATCH_IOU_THR="${SAM3_SPME_FUSION_MISMATCH_IOU_THR:-0.2}"
    export SAM3_SPME_PER_OBJECT_ANCHOR_MATCH="${SAM3_SPME_PER_OBJECT_ANCHOR_MATCH:-1}"
    export SAM3_SPME_SKIP_WRITE_ON_MISMATCH="1"
    export SAM3_SPME_SKIP_WRITE_MODE="${SAM3_SPME_SKIP_WRITE_MODE:-full}"
    export SAM3_SPME_REINIT="1"
    export SAM3_SPME_REINIT_ENABLE_MISMATCH="1"
    export SAM3_SPME_REID="1"
    ;;
  *)
    echo "[error] unexpected SLURM_ARRAY_TASK_ID=$TASK"
    exit 1
    ;;
esac

export OUT_ROOT="/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_ablate_recovery_joint_clip16_mem15/job_${SLURM_JOB_ID}/task${TASK}"

echo "[ablate_recovery_joint] task=$TASK"
echo "[ablate_recovery_joint] RUN_CONFIGS=$RUN_CONFIGS"
echo "[ablate_recovery_joint] OUT_ROOT=$OUT_ROOT"
echo "[ablate_recovery_joint] SAM3_TRACKER_NUM_MASKMEM=$SAM3_TRACKER_NUM_MASKMEM ENDOVIS_FUSION_WEIGHT_MODE=$ENDOVIS_FUSION_WEIGHT_MODE"
echo "[ablate_recovery_joint] FUSION_CKPT=$FUSION_CKPT"
echo "[ablate_recovery_joint] GATE_CKPT=$GATE_CKPT"
echo "[ablate_recovery_joint] SAM3_SPME_FUSION_EVENT_DRIVEN=$SAM3_SPME_FUSION_EVENT_DRIVEN SAM3_SPME_FUSION_EVENT_MODE=${SAM3_SPME_FUSION_EVENT_MODE:-<default>}"
echo "[ablate_recovery_joint] SAM3_SPME_SKIP_WRITE_ON_MISMATCH=${SAM3_SPME_SKIP_WRITE_ON_MISMATCH:-0} SAM3_SPME_SKIP_WRITE_MODE=${SAM3_SPME_SKIP_WRITE_MODE:-<unset>}"
echo "[ablate_recovery_joint] SAM3_SPME_REINIT=${SAM3_SPME_REINIT:-0} SAM3_SPME_REINIT_ENABLE_MISMATCH=${SAM3_SPME_REINIT_ENABLE_MISMATCH:-0}"
echo "[ablate_recovery_joint] SAM3_SPME_REID=${SAM3_SPME_REID:-0}"

bash scripts/sbatch_eval_endovis2017.sh
