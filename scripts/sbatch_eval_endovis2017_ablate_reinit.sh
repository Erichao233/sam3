#!/bin/bash
# EndoVis2017: ablate detector-guided "re-init" (identity-safe burst refresh) ON/OFF.
#
# Why this matters (reviewer-safe):
# - Learned-gate reduces ghosts, but some remaining failures are "long occlusion / drift recovery".
# - This re-init is an overseer-style mechanism: when the tracker is uncertain for K frames and the
#   detector is confident + identity-consistent (qcos), we refresh the memory write mask using the
#   detector mask on that frame.
#
# Protocol (MICCAI-safe):
# - PROMPT=visual, INIT_PROMPT=mask, INIT_FRAME_MODE=first_present, PROPAGATION_MODE=vg
# - no detector finetune overlay
# - recondition disabled (avoid confounds)
#
# Tasks:
#   0: Baseline (A), re-init OFF
#   1: Learned gate only (E), re-init OFF
#   2: Learned gate only (E), re-init ON
#   3: Learned gate + fusion (F), re-init OFF
#   4: Learned gate + fusion (F), re-init ON
#
# Usage (server):
#   sbatch scripts/sbatch_eval_endovis2017_ablate_reinit.sh
#
# Outputs:
#   /home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_ablate_reinit/job_${SLURM_JOB_ID}/task${TASK}_reinit${SAM3_SPME_REINIT}/...

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_endovis2017_eval_ablate_reinit
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.err
#SBATCH --array=0-4

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
export FUSION_CKPT="${FUSION_CKPT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_fusion_train_v1/job_16015880/main/checkpoints/spme_fusion_latest.pt}"
export GATE_CKPT="${GATE_CKPT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_gate_train_sweep_absent/job_16100471/aw0.1/main/checkpoints/spme_gate_latest.pt}"

# ---------------------------------------------------------------------------
# Best learned-gate safety envelope so far: det×qcos modulation (pow=4) + overseer-style fusion.
# ---------------------------------------------------------------------------
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET="1"
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_USE_QCOS="1"
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_POW="4.0"

export SAM3_SPME_FUSION_EVENT_DRIVEN="${SAM3_SPME_FUSION_EVENT_DRIVEN:-1}"
export SAM3_SPME_FUSION_EVENT_MODE="${SAM3_SPME_FUSION_EVENT_MODE:-mismatch}"
export SAM3_SPME_FUSION_MISMATCH_IOU_THR="${SAM3_SPME_FUSION_MISMATCH_IOU_THR:-0.2}"
# Needed for mismatch-driven fusion on drift frames (det↔trk IoU matching can be empty during drift).
export SAM3_SPME_PER_OBJECT_ANCHOR_MATCH="${SAM3_SPME_PER_OBJECT_ANCHOR_MATCH:-1}"

# ---------------------------------------------------------------------------
# Re-init defaults (keep minimal; tune only if the mechanism triggers too rarely/often).
# ---------------------------------------------------------------------------
export SAM3_SPME_REINIT_MISS_THR="${SAM3_SPME_REINIT_MISS_THR:-3}"
export SAM3_SPME_REINIT_CONFIRM_THR="${SAM3_SPME_REINIT_CONFIRM_THR:-2}"
export SAM3_SPME_REINIT_COOLDOWN="${SAM3_SPME_REINIT_COOLDOWN:-8}"
export SAM3_SPME_REINIT_DET_THR="${SAM3_SPME_REINIT_DET_THR:-0.7}"
export SAM3_SPME_REINIT_QCOS_THR="${SAM3_SPME_REINIT_QCOS_THR:-0.7}"
export SAM3_SPME_REINIT_TRACKER_THR="${SAM3_SPME_REINIT_TRACKER_THR:-0.8}"
# Only refresh when det↔trk overlap is sane (unless tracker is empty).
export SAM3_SPME_REINIT_CONFIRM_IOU_THR="${SAM3_SPME_REINIT_CONFIRM_IOU_THR:-0.2}"
# Re-detection style re-init (tracker empty) is high-risk for false positives; keep OFF by default.
export SAM3_SPME_REINIT_ALLOW_EMPTY="${SAM3_SPME_REINIT_ALLOW_EMPTY:-0}"
# Drift-based re-init is riskier; keep OFF by default (evaluate as ablation).
export SAM3_SPME_REINIT_ENABLE_MISMATCH="${SAM3_SPME_REINIT_ENABLE_MISMATCH:-0}"
export SAM3_SPME_REINIT_IOU_THR="${SAM3_SPME_REINIT_IOU_THR:-0.2}"
export SAM3_SPME_REINIT_MISMATCH_THR="${SAM3_SPME_REINIT_MISMATCH_THR:-2}"

TASK="${SLURM_ARRAY_TASK_ID}"

case "$TASK" in
  0)
    export RUN_CONFIGS="A"
    export SAM3_SPME_REINIT="0"
    ;;
  1)
    export RUN_CONFIGS="E"
    export SAM3_SPME_REINIT="0"
    ;;
  2)
    export RUN_CONFIGS="E"
    export SAM3_SPME_REINIT="1"
    ;;
  3)
    export RUN_CONFIGS="F"
    export SAM3_SPME_REINIT="0"
    ;;
  4)
    export RUN_CONFIGS="F"
    export SAM3_SPME_REINIT="1"
    ;;
  *)
    echo "[error] unexpected SLURM_ARRAY_TASK_ID=$TASK"
    exit 1
    ;;
esac

export OUT_ROOT="/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_ablate_reinit/job_${SLURM_JOB_ID}/task${TASK}_reinit${SAM3_SPME_REINIT}"

echo "[ablate_reinit] task=$TASK"
echo "[ablate_reinit] RUN_CONFIGS=$RUN_CONFIGS"
echo "[ablate_reinit] SAM3_SPME_REINIT=$SAM3_SPME_REINIT"
echo "[ablate_reinit] OUT_ROOT=$OUT_ROOT"
echo "[ablate_reinit] GATE_CKPT=$GATE_CKPT"
echo "[ablate_reinit] FUSION_CKPT=$FUSION_CKPT"
echo "[ablate_reinit] SAM3_DISABLE_RECONDITION=$SAM3_DISABLE_RECONDITION"

# Run the main eval script (as bash; SBATCH headers inside are ignored here).
bash scripts/sbatch_eval_endovis2017.sh
