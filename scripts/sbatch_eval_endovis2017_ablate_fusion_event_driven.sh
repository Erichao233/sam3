#!/bin/bash
# EndoVis2017: ablate event-driven fusion (overseer only when tracker is uncertain).
#
# Rationale:
# - Current SPME-F fusion is applied in ~70% frames and hurts mean IoU on EndoVis.
# - A more principled design is to inject detector guidance only when the tracker is uncertain
#   (drift signature), i.e., around occlusions/drift.
#
# This script compares:
# - Task 0: existing behavior (fusion not event-driven)
# - Task 1: event-driven fusion enabled (skip fusion when det↔trk IoU >= 0.2)
#
# Both tasks run configs A/D/E/F under the MICCAI-safe protocol:
# - PROMPT=visual, INIT_PROMPT=mask, INIT_FRAME_MODE=first_present, PROPAGATION_MODE=vg
# - no detector finetune overlay
#
# Usage (server):
#   sbatch scripts/sbatch_eval_endovis2017_ablate_fusion_event_driven.sh
#
# Outputs:
#   /home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_ablate_fusion_event/job_${SLURM_JOB_ID}/event${SAM3_SPME_FUSION_EVENT_DRIVEN}/...

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_endovis2017_eval_ablate_fusion_event
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.err
#SBATCH --array=0-1

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

# Keep calibration strengths at identity.
export SAM3_SPME_LEARNED_GATE_OFFSET_STRENGTH="${SAM3_SPME_LEARNED_GATE_OFFSET_STRENGTH:-1.0}"
export SAM3_SPME_LEARNED_GATE_WRITE_STRENGTH="${SAM3_SPME_LEARNED_GATE_WRITE_STRENGTH:-1.0}"
export SAM3_SPME_LEARNED_GATE_DECAY_STRENGTH="${SAM3_SPME_LEARNED_GATE_DECAY_STRENGTH:-1.0}"

# ---------------------------------------------------------------------------
# Event-driven fusion toggle.
# ---------------------------------------------------------------------------
TASK="${SLURM_ARRAY_TASK_ID}"
case "$TASK" in
  0)
    export SAM3_SPME_FUSION_EVENT_DRIVEN="0"
    ;;
  1)
    export SAM3_SPME_FUSION_EVENT_DRIVEN="1"
    export SAM3_SPME_FUSION_EVENT_MODE="mismatch"
    export SAM3_SPME_FUSION_MISMATCH_IOU_THR="${SAM3_SPME_FUSION_MISMATCH_IOU_THR:-0.2}"
    # Needed for mismatch-driven fusion on drift frames (det↔trk IoU matching can be empty during drift).
    export SAM3_SPME_PER_OBJECT_ANCHOR_MATCH="${SAM3_SPME_PER_OBJECT_ANCHOR_MATCH:-1}"
    ;;
  *)
    echo "[error] unexpected SLURM_ARRAY_TASK_ID=$TASK"
    exit 1
    ;;
esac

# Run all key configs in both tasks.
export RUN_CONFIGS="A D E F"

export OUT_ROOT="/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_ablate_fusion_event/job_${SLURM_JOB_ID}/event${SAM3_SPME_FUSION_EVENT_DRIVEN}"

echo "[ablate_fusion_event] task=$TASK"
echo "[ablate_fusion_event] RUN_CONFIGS=$RUN_CONFIGS"
echo "[ablate_fusion_event] OUT_ROOT=$OUT_ROOT"
echo "[ablate_fusion_event] FUSION_CKPT=$FUSION_CKPT"
echo "[ablate_fusion_event] GATE_CKPT=$GATE_CKPT"
echo "[ablate_fusion_event] SAM3_SPME_FUSION_EVENT_DRIVEN=$SAM3_SPME_FUSION_EVENT_DRIVEN"
echo "[ablate_fusion_event] SAM3_SPME_FUSION_EVENT_MODE=${SAM3_SPME_FUSION_EVENT_MODE:-<default>}"
echo "[ablate_fusion_event] SAM3_SPME_FUSION_MISMATCH_IOU_THR=${SAM3_SPME_FUSION_MISMATCH_IOU_THR:-<default>}"
echo "[ablate_fusion_event] SAM3_SPME_PER_OBJECT_ANCHOR_MATCH=${SAM3_SPME_PER_OBJECT_ANCHOR_MATCH:-<default>}"
echo "[ablate_fusion_event] SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_USE_QCOS=$SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_USE_QCOS"
echo "[ablate_fusion_event] SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_POW=$SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_POW"

# Run the main eval script (as bash; SBATCH headers inside are ignored here).
bash scripts/sbatch_eval_endovis2017.sh
