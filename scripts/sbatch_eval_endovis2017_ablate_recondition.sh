#!/bin/bash
# EndoVis2017: ablate SAM3 "recondition" (periodic detector→tracker overwrite) ON/OFF.
#
#
# Why this matters (reviewer-safe):
# - Recondition can mask drift/occlusion failures by forcibly snapping the tracker to detector masks.
# - To claim improvements from SPME (learned gate / fusion), we must show gains persist when recondition is disabled.
#
# What it runs (MICCAI-safe protocol):
# - PROMPT=visual, INIT_PROMPT=mask, INIT_FRAME_MODE=first_present, PROPAGATION_MODE=vg
# - no detector finetune overlay
#
# Tasks:
#   0: Baseline (A), recondition ON
#   1: Learned gate only (E), recondition ON
#   2: Learned gate + fusion (F), recondition ON
#   3: Baseline (A), recondition OFF
#   4: Learned gate only (E), recondition OFF
#   5: Learned gate + fusion (F), recondition OFF
#
# Usage (server):
#   sbatch scripts/sbatch_eval_endovis2017_ablate_recondition.sh
#
# Outputs:
#   /home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_ablate_recondition/job_${SLURM_JOB_ID}/task${TASK}_recond${RECOND_ON}/...

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_endovis2017_eval_ablate_recondition
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.err
#SBATCH --array=0-5

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

# ---------------------------------------------------------------------------
# Checkpoints (pin for reproducibility; override if you want).
# ---------------------------------------------------------------------------
export FUSION_CKPT="${FUSION_CKPT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_fusion_train_v1/job_16015880/main/checkpoints/spme_fusion_latest.pt}"
export GATE_CKPT="${GATE_CKPT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_gate_train_sweep_absent/job_16100471/aw0.1/main/checkpoints/spme_gate_latest.pt}"

# ---------------------------------------------------------------------------
# Best learned-gate safety envelope so far: det×qcos modulation with pow=4.
# ---------------------------------------------------------------------------
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET="1"
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_USE_QCOS="1"
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_POW="4.0"

# Keep calibration strengths at identity unless explicitly overridden.
export SAM3_SPME_LEARNED_GATE_OFFSET_STRENGTH="${SAM3_SPME_LEARNED_GATE_OFFSET_STRENGTH:-1.0}"
export SAM3_SPME_LEARNED_GATE_WRITE_STRENGTH="${SAM3_SPME_LEARNED_GATE_WRITE_STRENGTH:-1.0}"
export SAM3_SPME_LEARNED_GATE_DECAY_STRENGTH="${SAM3_SPME_LEARNED_GATE_DECAY_STRENGTH:-1.0}"

# ---------------------------------------------------------------------------
# Make fusion "overseer-style" by default (drift-aware; avoids absent-frame ghosts on EndoVis).
# ---------------------------------------------------------------------------
export SAM3_SPME_FUSION_EVENT_DRIVEN="${SAM3_SPME_FUSION_EVENT_DRIVEN:-1}"
export SAM3_SPME_FUSION_EVENT_MODE="${SAM3_SPME_FUSION_EVENT_MODE:-mismatch}"
export SAM3_SPME_FUSION_MISMATCH_IOU_THR="${SAM3_SPME_FUSION_MISMATCH_IOU_THR:-0.2}"
# Needed for mismatch-driven fusion on drift frames (det↔trk IoU matching can be empty during drift).
export SAM3_SPME_PER_OBJECT_ANCHOR_MATCH="${SAM3_SPME_PER_OBJECT_ANCHOR_MATCH:-1}"

TASK="${SLURM_ARRAY_TASK_ID}"

case "$TASK" in
  0)
    export RUN_CONFIGS="A"
    export SAM3_DISABLE_RECONDITION="0"
    RECOND_ON="1"
    ;;
  1)
    export RUN_CONFIGS="E"
    export SAM3_DISABLE_RECONDITION="0"
    RECOND_ON="1"
    ;;
  2)
    export RUN_CONFIGS="F"
    export SAM3_DISABLE_RECONDITION="0"
    RECOND_ON="1"
    ;;
  3)
    export RUN_CONFIGS="A"
    export SAM3_DISABLE_RECONDITION="1"
    RECOND_ON="0"
    ;;
  4)
    export RUN_CONFIGS="E"
    export SAM3_DISABLE_RECONDITION="1"
    RECOND_ON="0"
    ;;
  5)
    export RUN_CONFIGS="F"
    export SAM3_DISABLE_RECONDITION="1"
    RECOND_ON="0"
    ;;
  *)
    echo "[error] unexpected SLURM_ARRAY_TASK_ID=$TASK"
    exit 1
    ;;
esac

export OUT_ROOT="/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_ablate_recondition/job_${SLURM_JOB_ID}/task${TASK}_recond${RECOND_ON}"

echo "[ablate_recondition] task=$TASK"
echo "[ablate_recondition] RUN_CONFIGS=$RUN_CONFIGS"
echo "[ablate_recondition] OUT_ROOT=$OUT_ROOT"
echo "[ablate_recondition] SAM3_DISABLE_RECONDITION=$SAM3_DISABLE_RECONDITION"
echo "[ablate_recondition] GATE_CKPT=$GATE_CKPT"
echo "[ablate_recondition] FUSION_CKPT=$FUSION_CKPT"
echo "[ablate_recondition] SAM3_SPME_FUSION_EVENT_DRIVEN=$SAM3_SPME_FUSION_EVENT_DRIVEN"
echo "[ablate_recondition] SAM3_SPME_FUSION_EVENT_MODE=$SAM3_SPME_FUSION_EVENT_MODE"
echo "[ablate_recondition] SAM3_SPME_FUSION_MISMATCH_IOU_THR=$SAM3_SPME_FUSION_MISMATCH_IOU_THR"
echo "[ablate_recondition] SAM3_SPME_PER_OBJECT_ANCHOR_MATCH=$SAM3_SPME_PER_OBJECT_ANCHOR_MATCH"

# Run the main eval script (as bash; SBATCH headers inside are ignored here).
bash scripts/sbatch_eval_endovis2017.sh
