#!/bin/bash
# EndoVis2017: ablate mismatch-triggered event-fusion *severity scaling* (linear vs hard).
#
# Why this matters (paper-safe):
# - Our drift-aware event trigger (det↔trk IoU < thr) already avoids GT-absent interventions.
# - However, we additionally scale fusion by mismatch severity: r *= (thr - miou)/thr.
#   On EndoVis this can make fusion too weak to matter.
# - This ablation tests whether disabling the within-event severity scaling ("hard") helps,
#   without changing any training.
#
# Protocol (MICCAI-safe):
# - PROMPT=visual, INIT_PROMPT=mask, INIT_FRAME_MODE=first_present, PROPAGATION_MODE=vg
# - no detector finetune overlay
# - recondition disabled (avoid confounds)
#
# Tasks:
#   0: Learned gate only (E) as reference
#   1: Learned gate + fusion (F), mismatch event, severity scaling = linear (default)
#   2: Learned gate + fusion (F), mismatch event, severity scaling = hard (no extra scaling)
#
# Usage (server):
#   sbatch scripts/sbatch_eval_endovis2017_ablate_fusion_mismatch_scale.sh
#
# Outputs:
#   /home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_ablate_fusion_mismatch_scale/job_${SLURM_JOB_ID}/task${TASK}_scale${SAM3_SPME_FUSION_MISMATCH_SCALE}/...

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_endovis2017_eval_ablate_fusion_mismatch_scale
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.err
#SBATCH --array=0-2

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
export FUSION_CKPT="${FUSION_CKPT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_fusion_train_v1/job_16015880/main/checkpoints/spme_fusion_latest.pt}"
export GATE_CKPT="${GATE_CKPT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_gate_train_sweep_absent/job_16100471/aw0.1/main/checkpoints/spme_gate_latest.pt}"

# ---------------------------------------------------------------------------
# Best learned-gate safety envelope so far (kept fixed).
# ---------------------------------------------------------------------------
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET="1"
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_USE_QCOS="1"
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_POW="4.0"

# Enable SPME per-object association by anchor qcos (needed for det↔trk mismatch on drift frames).
export SAM3_SPME_PER_OBJECT_ANCHOR_MATCH="1"

# Keep calibration strengths at identity.
export SAM3_SPME_LEARNED_GATE_OFFSET_STRENGTH="${SAM3_SPME_LEARNED_GATE_OFFSET_STRENGTH:-1.0}"
export SAM3_SPME_LEARNED_GATE_WRITE_STRENGTH="${SAM3_SPME_LEARNED_GATE_WRITE_STRENGTH:-1.0}"
export SAM3_SPME_LEARNED_GATE_DECAY_STRENGTH="${SAM3_SPME_LEARNED_GATE_DECAY_STRENGTH:-1.0}"

# ---------------------------------------------------------------------------
# Fusion: event-driven by mismatch (det↔trk IoU < thr).
# ---------------------------------------------------------------------------
export SAM3_SPME_FUSION_EVENT_DRIVEN="1"
export SAM3_SPME_FUSION_EVENT_MODE="mismatch"
export SAM3_SPME_FUSION_MISMATCH_IOU_THR="${SAM3_SPME_FUSION_MISMATCH_IOU_THR:-0.2}"

TASK="${SLURM_ARRAY_TASK_ID}"
case "$TASK" in
  0)
    export RUN_CONFIGS="E"
    export SAM3_SPME_FUSION_MISMATCH_SCALE="linear"  # unused
    ;;
  1)
    export RUN_CONFIGS="F"
    export SAM3_SPME_FUSION_MISMATCH_SCALE="linear"
    ;;
  2)
    export RUN_CONFIGS="F"
    export SAM3_SPME_FUSION_MISMATCH_SCALE="hard"
    ;;
  *)
    echo "[error] unexpected SLURM_ARRAY_TASK_ID=$TASK"
    exit 1
    ;;
esac

export OUT_ROOT="/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_ablate_fusion_mismatch_scale/job_${SLURM_JOB_ID}/task${TASK}_scale${SAM3_SPME_FUSION_MISMATCH_SCALE}"

echo "[ablate_fusion_mismatch_scale] task=$TASK"
echo "[ablate_fusion_mismatch_scale] RUN_CONFIGS=$RUN_CONFIGS"
echo "[ablate_fusion_mismatch_scale] OUT_ROOT=$OUT_ROOT"
echo "[ablate_fusion_mismatch_scale] FUSION_CKPT=$FUSION_CKPT"
echo "[ablate_fusion_mismatch_scale] GATE_CKPT=$GATE_CKPT"
echo "[ablate_fusion_mismatch_scale] SAM3_DISABLE_RECONDITION=$SAM3_DISABLE_RECONDITION"
echo "[ablate_fusion_mismatch_scale] SAM3_SPME_FUSION_EVENT_DRIVEN=$SAM3_SPME_FUSION_EVENT_DRIVEN"
echo "[ablate_fusion_mismatch_scale] SAM3_SPME_FUSION_EVENT_MODE=$SAM3_SPME_FUSION_EVENT_MODE"
echo "[ablate_fusion_mismatch_scale] SAM3_SPME_FUSION_MISMATCH_IOU_THR=$SAM3_SPME_FUSION_MISMATCH_IOU_THR"
echo "[ablate_fusion_mismatch_scale] SAM3_SPME_FUSION_MISMATCH_SCALE=$SAM3_SPME_FUSION_MISMATCH_SCALE"
echo "[ablate_fusion_mismatch_scale] SAM3_SPME_PER_OBJECT_ANCHOR_MATCH=$SAM3_SPME_PER_OBJECT_ANCHOR_MATCH"

# Run the main eval script (as bash; SBATCH headers inside are ignored here).
bash scripts/sbatch_eval_endovis2017.sh

