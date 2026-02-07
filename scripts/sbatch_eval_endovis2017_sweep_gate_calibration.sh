#!/bin/bash
# EndoVis2017: sweep inference-time learned-gate calibration knobs.
#
# Motivation:
# - Our current learned-gate checkpoints strongly suppress ghosts, but can reduce IoU on GT-present frames.
# - `spme_debug_summary.json` shows a large, near-constant `mem_offset` term for some ckpts.
# - This script sweeps `SAM3_SPME_LEARNED_GATE_OFFSET_STRENGTH` to test whether offset is the culprit,
#   without retraining.
#
# What it runs (per task):
# - Task 0: baseline (A) + learned gate only (E) + learned gate+fusion (F)
# - Task 1..: learned gate only (E) + learned gate+fusion (F)
#
# Usage (server):
#   sbatch scripts/sbatch_eval_endovis2017_sweep_gate_calibration.sh
#
# Outputs:
#   /home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_gate_calibration/job_${SLURM_JOB_ID}/off${OFFSET_STRENGTH}/...

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_endovis2017_eval_gate_calibration
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.err
#SBATCH --array=0-3

set -euo pipefail

# ---------------------------------------------------------------------------
# Fixed eval protocol (MICCAI-safe): visual-only + mask init at first visible frame.
# ---------------------------------------------------------------------------
export OVERLAY_CKPT=""            # keep empty unless you explicitly want detector finetune
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
# Default: best-ish gate ckpt from ABSENT_W sweep (aw0.1).
export GATE_CKPT="${GATE_CKPT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_gate_train_sweep_absent/job_16100471/aw0.1/main/checkpoints/spme_gate_latest.pt}"

# ---------------------------------------------------------------------------
# SPME / fusion defaults (match the current evaluation defaults).
# ---------------------------------------------------------------------------
export ANCHOR_DET_THR="${ANCHOR_DET_THR:-0.0}"
export QUERY_POOL="${QUERY_POOL:-top1}"
export QUERY_TOPK="${QUERY_TOPK:-5}"
export FUSION_DET_THR="${FUSION_DET_THR:-0.3}"
export FUSION_QCOS_THR="${FUSION_QCOS_THR:-0.7}"
export FUSION_QCOS_TEMP="${FUSION_QCOS_TEMP:-20.0}"
export FUSION_QCOS_GATE="${FUSION_QCOS_GATE:-sigmoid}"
export FUSION_USE_PRESENCE="${FUSION_USE_PRESENCE:-1}"
export FUSION_ALPHA="${FUSION_ALPHA:-0.01}"
export FUSION_ALPHA_OBJ="${FUSION_ALPHA_OBJ:-0.001}"

# Gate settings (must match the ckpt).
export GATE_INPUTS="${GATE_INPUTS:-full}"
export GATE_FUSION_HEAD="${GATE_FUSION_HEAD:-1}"
export GATE_OCC_NORM="${GATE_OCC_NORM:-10.0}"
export GATE_DET_PRESENT_THR="${GATE_DET_PRESENT_THR:-0.3}"
export GATE_USE_DECAY="${GATE_USE_DECAY:-1}"

# ---------------------------------------------------------------------------
# Sweep values.
# ---------------------------------------------------------------------------
OFFSET_STRENGTHS=(0.0 0.25 0.5 1.0)
TASK="${SLURM_ARRAY_TASK_ID}"
OFFSET_STRENGTH="${OFFSET_STRENGTHS[$TASK]}"

export SAM3_SPME_LEARNED_GATE_OFFSET_STRENGTH="${OFFSET_STRENGTH}"
# Keep other calibration knobs as identity unless you are explicitly sweeping them.
export SAM3_SPME_LEARNED_GATE_WRITE_STRENGTH="${SAM3_SPME_LEARNED_GATE_WRITE_STRENGTH:-1.0}"
export SAM3_SPME_LEARNED_GATE_DECAY_STRENGTH="${SAM3_SPME_LEARNED_GATE_DECAY_STRENGTH:-1.0}"

if [[ "$TASK" == "0" ]]; then
  export RUN_CONFIGS="A E F"
else
  export RUN_CONFIGS="E F"
fi

export OUT_ROOT="/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_gate_calibration/job_${SLURM_JOB_ID}/off${OFFSET_STRENGTH}"

echo "[gate_calib] task=$TASK OFFSET_STRENGTH=$OFFSET_STRENGTH"
echo "[gate_calib] RUN_CONFIGS=$RUN_CONFIGS"
echo "[gate_calib] OUT_ROOT=$OUT_ROOT"
echo "[gate_calib] GATE_CKPT=$GATE_CKPT"
echo "[gate_calib] FUSION_CKPT=$FUSION_CKPT"
echo "[gate_calib] SAM3_SPME_LEARNED_GATE_OFFSET_STRENGTH=$SAM3_SPME_LEARNED_GATE_OFFSET_STRENGTH"
echo "[gate_calib] SAM3_SPME_LEARNED_GATE_WRITE_STRENGTH=$SAM3_SPME_LEARNED_GATE_WRITE_STRENGTH"
echo "[gate_calib] SAM3_SPME_LEARNED_GATE_DECAY_STRENGTH=$SAM3_SPME_LEARNED_GATE_DECAY_STRENGTH"

# Run the main eval script (as bash; SBATCH headers inside are ignored here).
bash scripts/sbatch_eval_endovis2017.sh

