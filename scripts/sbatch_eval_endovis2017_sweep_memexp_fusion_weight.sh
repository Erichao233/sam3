#!/bin/bash
# EndoVis2017: sweep (memory expansion) × (output-level fusion weight) for A vs E.
#
# Goal (numbers-first, paper-safe):
# - `SAM3_TRACKER_NUM_MASKMEM=15` can reduce ghosting / improve long occlusions (training-free).
# - Output-level quality-weighted fusion can massively affect semantic mcIoU (class competition),
#   without changing per-class tracking predictions.
#
# This script evaluates the combined effect, so we can decide whether:
# - we should keep mem15 for main results,
# - and whether learned gate (E) still helps under mem15 + best fusion weighting,
# - before spending time retraining gate for mem15.
#
# Protocol (MICCAI-safe):
# - PROMPT=visual, INIT_PROMPT=mask, INIT_FRAME_MODE=first_present, PROPAGATION_MODE=vg
# - no detector finetune overlay
# - recondition disabled (avoid confounds)
#
# Grid (8 tasks):
#   mem ∈ {7,15} × fusion_weight ∈ {tracker_prob,eff_iou} × config ∈ {A,E}
#
# Usage (server):
#   sbatch scripts/sbatch_eval_endovis2017_sweep_memexp_fusion_weight.sh
#
# Outputs:
#   /home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_sweep_memexp_fusion_weight/job_${SLURM_JOB_ID}/mem${MEM}_w${WEIGHT}_cfg${CFG}/...
#
# Notes:
# - This is NOT feature-injection fusion (FUSION_CKPT is kept empty).

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_endovis2017_eval_sweep_memexp_fusion_weight
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.err
#SBATCH --array=0-7

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

# Keep feature-injection fusion off for this sweep (output-level fusion only).
export FUSION_CKPT=""

# ---------------------------------------------------------------------------
# Checkpoints (pin for reproducibility; override if you want).
# ---------------------------------------------------------------------------
export GATE_CKPT="${GATE_CKPT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_gate_train_sweep_absent/job_16100471/aw0.1/main/checkpoints/spme_gate_latest.pt}"

# Best learned-gate safety envelope so far (inference-only; only affects E).
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET="1"
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_USE_QCOS="1"
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_POW="4.0"

# Gate architecture knobs (must match the checkpoint).
export GATE_INPUTS="${GATE_INPUTS:-full}"
export GATE_FUSION_HEAD="${GATE_FUSION_HEAD:-1}"
export GATE_USE_DECAY="${GATE_USE_DECAY:-1}"
export GATE_OCC_NORM="${GATE_OCC_NORM:-10.0}"
export GATE_DET_PRESENT_THR="${GATE_DET_PRESENT_THR:-0.3}"

# ---------------------------------------------------------------------------
# Grid definition.
# ---------------------------------------------------------------------------
MEMS=(7 15)
WEIGHTS=("tracker_prob" "eff_iou")
CFGS=("A" "E")

TASK="${SLURM_ARRAY_TASK_ID}"

mem_idx=$(( TASK / 4 ))
rem=$(( TASK % 4 ))
w_idx=$(( rem / 2 ))
cfg_idx=$(( rem % 2 ))

MEM="${MEMS[$mem_idx]}"
WEIGHT="${WEIGHTS[$w_idx]}"
CFG="${CFGS[$cfg_idx]}"

export SAM3_TRACKER_NUM_MASKMEM="${MEM}"
export ENDOVIS_FUSION_WEIGHT_MODE="${WEIGHT}"

if [[ "$CFG" == "A" ]]; then
  export RUN_CONFIGS="A"
else
  export RUN_CONFIGS="E"
fi

export OUT_ROOT="/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_sweep_memexp_fusion_weight/job_${SLURM_JOB_ID}/mem${MEM}_w${WEIGHT}_cfg${CFG}"

echo "[sweep_memexp_fusion_weight] task=$TASK mem=$MEM weight=$WEIGHT cfg=$CFG"
echo "[sweep_memexp_fusion_weight] RUN_CONFIGS=$RUN_CONFIGS"
echo "[sweep_memexp_fusion_weight] OUT_ROOT=$OUT_ROOT"
echo "[sweep_memexp_fusion_weight] SAM3_TRACKER_NUM_MASKMEM=$SAM3_TRACKER_NUM_MASKMEM"
echo "[sweep_memexp_fusion_weight] ENDOVIS_FUSION_WEIGHT_MODE=$ENDOVIS_FUSION_WEIGHT_MODE"
echo "[sweep_memexp_fusion_weight] GATE_CKPT=$GATE_CKPT"

bash scripts/sbatch_eval_endovis2017.sh

