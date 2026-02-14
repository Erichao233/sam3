#!/bin/bash
# EndoVis2017: ablate an uncertainty-aware (tracker-score) modulation envelope for the learned gate.
#
# Why:
# - Our joint (gate+fusion) model improves some hard sequences (e.g., occlusion-heavy), but can slightly
#   regress on stable sequences where the detector is under-confident/miscalibrated.
# - This ablation tests a *do-no-harm* inference-time envelope:
#     When tracker_score is high => push learned gate towards identity (scale→1, offset→0, decay→0)
#     When tracker_score is low  => allow learned gate to intervene.
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
#   2: F + tracker-modulated learned gate (SAM3_SPME_LEARNED_GATE_MODULATE_BY_TRK=1)
#
# Usage (server):
#   sbatch scripts/sbatch_eval_endovis2017_ablate_gate_trk_modulation_joint_clip16_mem15.sh
#
# Outputs:
#   /home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_ablate_gate_trk_mod/job_${SLURM_JOB_ID}/task${TASK}/...

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_endovis2017_eval_ablate_gate_trk_mod
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.err
#SBATCH --exclude=hpc-n968
#SBATCH --array=0-2

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
# Learned-gate safety envelope + overseer-style fusion (stable defaults).
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

# Reset tracker-modulation toggles (enabled per-task).
unset SAM3_SPME_LEARNED_GATE_MODULATE_BY_TRK SAM3_SPME_LEARNED_GATE_MODULATE_BY_TRK_POW

case "$TASK" in
  0)
    export RUN_CONFIGS="A"
    ;;
  1)
    export RUN_CONFIGS="F"
    ;;
  2)
    export RUN_CONFIGS="F"
    export SAM3_SPME_LEARNED_GATE_MODULATE_BY_TRK="1"
    export SAM3_SPME_LEARNED_GATE_MODULATE_BY_TRK_POW="${SAM3_SPME_LEARNED_GATE_MODULATE_BY_TRK_POW:-1.0}"
    ;;
  *)
    echo "[error] unexpected SLURM_ARRAY_TASK_ID=$TASK"
    exit 1
    ;;
esac

export OUT_ROOT="/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_ablate_gate_trk_mod/job_${SLURM_JOB_ID}/task${TASK}"

echo "[ablate_gate_trk_mod] task=$TASK"
echo "[ablate_gate_trk_mod] RUN_CONFIGS=$RUN_CONFIGS"
echo "[ablate_gate_trk_mod] OUT_ROOT=$OUT_ROOT"
echo "[ablate_gate_trk_mod] FUSION_CKPT=$FUSION_CKPT"
echo "[ablate_gate_trk_mod] GATE_CKPT=$GATE_CKPT"
echo "[ablate_gate_trk_mod] SAM3_SPME_LEARNED_GATE_MODULATE_BY_TRK=${SAM3_SPME_LEARNED_GATE_MODULATE_BY_TRK:-0}"

bash scripts/sbatch_eval_endovis2017.sh

