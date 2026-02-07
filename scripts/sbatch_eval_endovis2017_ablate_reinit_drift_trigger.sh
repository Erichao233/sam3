#!/bin/bash
# EndoVis2017: ablate mismatch-based "drift trigger" for re-init ON/OFF.
#
# Motivation (paper-safe):
# - Our v2 re-init is designed as an overseer re-prompt *on tracker failure* (uncertainty-triggered + overlap-confirmed).
# - A mismatch-based drift trigger can be riskier (may fire on GT-absent false detections or overconfident drift).
# - This ablation isolates whether the mismatch trigger helps or harms, without changing any training.
#
# Protocol (MICCAI-safe):
# - PROMPT=visual, INIT_PROMPT=mask, INIT_FRAME_MODE=first_present, PROPAGATION_MODE=vg
# - no detector finetune overlay
# - recondition disabled (avoid confounds)
#
# Tasks:
#   0: Learned gate only (E), re-init ON, mismatch trigger OFF  (recommended default)
#   1: Learned gate only (E), re-init ON, mismatch trigger ON   (riskier ablation)
#
# Usage (server):
#   sbatch scripts/sbatch_eval_endovis2017_ablate_reinit_drift_trigger.sh
#
# Outputs:
#   /home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_ablate_reinit_drift/job_${SLURM_JOB_ID}/task${TASK}_mismatch${SAM3_SPME_REINIT_ENABLE_MISMATCH}/...

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_endovis2017_eval_ablate_reinit_drift
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.err
#SBATCH --array=0-1

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
# Best learned-gate safety envelope so far.
# ---------------------------------------------------------------------------
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET="1"
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_USE_QCOS="1"
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_POW="4.0"

# ---------------------------------------------------------------------------
# Re-init v2 defaults.
# ---------------------------------------------------------------------------
export SAM3_SPME_REINIT="1"
export SAM3_SPME_REINIT_MISS_THR="${SAM3_SPME_REINIT_MISS_THR:-3}"
export SAM3_SPME_REINIT_CONFIRM_THR="${SAM3_SPME_REINIT_CONFIRM_THR:-2}"
export SAM3_SPME_REINIT_COOLDOWN="${SAM3_SPME_REINIT_COOLDOWN:-8}"
export SAM3_SPME_REINIT_DET_THR="${SAM3_SPME_REINIT_DET_THR:-0.7}"
export SAM3_SPME_REINIT_QCOS_THR="${SAM3_SPME_REINIT_QCOS_THR:-0.7}"
export SAM3_SPME_REINIT_TRACKER_THR="${SAM3_SPME_REINIT_TRACKER_THR:-0.8}"
export SAM3_SPME_REINIT_CONFIRM_IOU_THR="${SAM3_SPME_REINIT_CONFIRM_IOU_THR:-0.2}"
# Re-detection style re-init (tracker empty) is high-risk for false positives; keep OFF by default.
export SAM3_SPME_REINIT_ALLOW_EMPTY="${SAM3_SPME_REINIT_ALLOW_EMPTY:-0}"
export SAM3_SPME_REINIT_IOU_THR="${SAM3_SPME_REINIT_IOU_THR:-0.2}"
export SAM3_SPME_REINIT_MISMATCH_THR="${SAM3_SPME_REINIT_MISMATCH_THR:-2}"

TASK="${SLURM_ARRAY_TASK_ID}"

case "$TASK" in
  0)
    export RUN_CONFIGS="E"
    export SAM3_SPME_REINIT_ENABLE_MISMATCH="0"
    ;;
  1)
    export RUN_CONFIGS="E"
    export SAM3_SPME_REINIT_ENABLE_MISMATCH="1"
    ;;
  *)
    echo "[error] unexpected SLURM_ARRAY_TASK_ID=$TASK"
    exit 1
    ;;
esac

export OUT_ROOT="/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_ablate_reinit_drift/job_${SLURM_JOB_ID}/task${TASK}_mismatch${SAM3_SPME_REINIT_ENABLE_MISMATCH}"

echo "[ablate_reinit_drift] task=$TASK"
echo "[ablate_reinit_drift] RUN_CONFIGS=$RUN_CONFIGS"
echo "[ablate_reinit_drift] OUT_ROOT=$OUT_ROOT"
echo "[ablate_reinit_drift] GATE_CKPT=$GATE_CKPT"
echo "[ablate_reinit_drift] SAM3_DISABLE_RECONDITION=$SAM3_DISABLE_RECONDITION"
echo "[ablate_reinit_drift] SAM3_SPME_REINIT_ENABLE_MISMATCH=$SAM3_SPME_REINIT_ENABLE_MISMATCH"

# Run the main eval script (as bash; SBATCH headers inside are ignored here).
bash scripts/sbatch_eval_endovis2017.sh
