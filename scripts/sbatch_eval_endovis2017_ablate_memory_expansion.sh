#!/bin/bash
# EndoVis2017: ablate Tracker memory expansion (num_maskmem=7 vs 15) with tpos interpolation.
#
# Motivation (paper-safe):
# - Surgical videos have long occlusions; SAM3 default memory length (7) can be too short.
# - Increasing `num_maskmem` naively introduces random untrained temporal positional encodings.
# - We support a training-free upgrade by interpolating `tracker.maskmem_tpos_enc` at checkpoint load time.
#
# Protocol (MICCAI-safe):
# - PROMPT=visual, INIT_PROMPT=mask, INIT_FRAME_MODE=first_present, PROPAGATION_MODE=vg
# - no detector finetune overlay
# - recondition disabled (avoid confounds)
#
# Tasks:
#   0: Baseline (A), num_maskmem=7
#   1: Learned gate only (E), num_maskmem=7
#   2: Baseline (A), num_maskmem=15
#   3: Learned gate only (E), num_maskmem=15
#
# Usage (server):
#   sbatch scripts/sbatch_eval_endovis2017_ablate_memory_expansion.sh
#
# Outputs:
#   /home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_ablate_memexp/job_${SLURM_JOB_ID}/task${TASK}_mem${SAM3_TRACKER_NUM_MASKMEM}/...

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_endovis2017_eval_ablate_memexp
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.err
#SBATCH --array=0-3

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

# Best learned-gate safety envelope so far (optional; only affects E runs).
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET="1"
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_USE_QCOS="1"
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_POW="4.0"

# Gate architecture knobs (must match the checkpoint).
export GATE_INPUTS="${GATE_INPUTS:-full}"
export GATE_FUSION_HEAD="${GATE_FUSION_HEAD:-1}"
export GATE_USE_DECAY="${GATE_USE_DECAY:-1}"
export GATE_OCC_NORM="${GATE_OCC_NORM:-10.0}"
export GATE_DET_PRESENT_THR="${GATE_DET_PRESENT_THR:-0.3}"

TASK="${SLURM_ARRAY_TASK_ID}"

case "$TASK" in
  0)
    export RUN_CONFIGS="A"
    export SAM3_TRACKER_NUM_MASKMEM="7"
    ;;
  1)
    export RUN_CONFIGS="E"
    export SAM3_TRACKER_NUM_MASKMEM="7"
    ;;
  2)
    export RUN_CONFIGS="A"
    export SAM3_TRACKER_NUM_MASKMEM="15"
    ;;
  3)
    export RUN_CONFIGS="E"
    export SAM3_TRACKER_NUM_MASKMEM="15"
    ;;
  *)
    echo "[error] unexpected SLURM_ARRAY_TASK_ID=$TASK"
    exit 1
    ;;
esac

export OUT_ROOT="/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_ablate_memexp/job_${SLURM_JOB_ID}/task${TASK}_mem${SAM3_TRACKER_NUM_MASKMEM}"

echo "[ablate_memexp] task=$TASK"
echo "[ablate_memexp] RUN_CONFIGS=$RUN_CONFIGS"
echo "[ablate_memexp] OUT_ROOT=$OUT_ROOT"
echo "[ablate_memexp] SAM3_TRACKER_NUM_MASKMEM=$SAM3_TRACKER_NUM_MASKMEM"
echo "[ablate_memexp] SAM3_DISABLE_RECONDITION=$SAM3_DISABLE_RECONDITION"
echo "[ablate_memexp] GATE_CKPT=$GATE_CKPT"

# Run the main eval script (as bash; SBATCH headers inside are ignored here).
bash scripts/sbatch_eval_endovis2017.sh
