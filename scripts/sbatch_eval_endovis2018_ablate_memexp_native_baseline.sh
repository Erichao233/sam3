#!/bin/bash
# EndoVis2018: ablate "native SAM3" vs memory expansion (mem7 vs mem15) under the same eval protocol.
#
# What it runs (official test_data split):
# - Task 0: Baseline (A), mem7  (closest to native SAM3)
# - Task 1: Baseline (A), mem15 (memory expansion only)
# - Task 2: Joint model (F), mem15 (your trained ckpt)
#
# Usage (server):
#   sbatch scripts/sbatch_eval_endovis2018_ablate_memexp_native_baseline.sh
#
# Outputs:
#   /home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2018_spme_eval_ablate_memexp/job_${SLURM_JOB_ID}/task${TASK}/...

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_endovis2018_eval_ablate_memexp
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.err
#SBATCH --exclude=hpc-n968
#SBATCH --array=0-2

set -euo pipefail

# Raw official root + prepared test root.
export ENDOVIS_SRC="${ENDOVIS_SRC:-/home2020/home/icube/kunyuan/SurgBench/surgicaltool/Endovis2018}"
export DATA="${DATA:-/home2020/home/icube/kunyuan/SurgBench/surgicaltool/endovis2018_test}"

# If DATA is not prepared yet, auto-prepare from test_data only.
export PREPARE_DATA="${PREPARE_DATA:-auto}"
export ENDOVIS_GROUPS="${ENDOVIS_GROUPS:-test_data}"

# Fixed eval protocol (MICCAI-safe).
export OVERLAY_CKPT=""
export PROMPT="visual"
export PROPAGATION_MODE="vg"
export INIT_PROMPT="mask"
export INIT_FRAME_MODE="first_present"
export INIT_SELECT_MODE="prob"
export HOTSTART_DELAY="0"
export DEBUG_SPME_SUMMARY="1"
export SAM3_DISABLE_RECONDITION="1"

# Gate MLP width (only matters when loading a gate ckpt).
export SAM3_SPME_GATE_HIDDEN="${SAM3_SPME_GATE_HIDDEN:-64}"
export ENDOVIS_FUSION_WEIGHT_MODE="${ENDOVIS_FUSION_WEIGHT_MODE:-eff_iou}"

# Safety envelope (only affects task 2 / learned-gate configs).
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET="${SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET:-1}"
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_USE_QCOS="${SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_USE_QCOS:-1}"
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_POW="${SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_POW:-4.0}"

export SAM3_SPME_LEARNED_GATE_OFFSET_STRENGTH="${SAM3_SPME_LEARNED_GATE_OFFSET_STRENGTH:-1.0}"
export SAM3_SPME_LEARNED_GATE_WRITE_STRENGTH="${SAM3_SPME_LEARNED_GATE_WRITE_STRENGTH:-1.0}"
export SAM3_SPME_LEARNED_GATE_DECAY_STRENGTH="${SAM3_SPME_LEARNED_GATE_DECAY_STRENGTH:-1.0}"

export SAM3_SPME_FUSION_EVENT_DRIVEN="${SAM3_SPME_FUSION_EVENT_DRIVEN:-1}"
export SAM3_SPME_FUSION_TRACKER_THR="${SAM3_SPME_FUSION_TRACKER_THR:-0.8}"

TASK="${SLURM_ARRAY_TASK_ID}"

# Default joint checkpoint (override if needed).
JOINT_CKPT_DEFAULT="/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2018_spme_joint_train_clip16_mem15/job_16165365/main/checkpoints/spme_gate_latest.pt"

case "$TASK" in
  0)
    export RUN_CONFIGS="A"
    export SAM3_TRACKER_NUM_MASKMEM="7"
    ;;
  1)
    export RUN_CONFIGS="A"
    export SAM3_TRACKER_NUM_MASKMEM="15"
    ;;
  2)
    export RUN_CONFIGS="F"
    export SAM3_TRACKER_NUM_MASKMEM="15"
    export GATE_CKPT="${GATE_CKPT:-$JOINT_CKPT_DEFAULT}"
    export FUSION_CKPT="${FUSION_CKPT:-$GATE_CKPT}"
    ;;
  *)
    echo "[error] unexpected SLURM_ARRAY_TASK_ID=$TASK"
    exit 1
    ;;
esac

export OUT_ROOT="/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2018_spme_eval_ablate_memexp/job_${SLURM_JOB_ID}/task${TASK}"

echo "[ablate_memexp] task=$TASK"
echo "[ablate_memexp] RUN_CONFIGS=$RUN_CONFIGS"
echo "[ablate_memexp] SAM3_TRACKER_NUM_MASKMEM=$SAM3_TRACKER_NUM_MASKMEM"
echo "[ablate_memexp] DATA=$DATA ENDOVIS_SRC=$ENDOVIS_SRC ENDOVIS_GROUPS=$ENDOVIS_GROUPS"
echo "[ablate_memexp] OUT_ROOT=$OUT_ROOT"
echo "[ablate_memexp] GATE_CKPT=${GATE_CKPT:-<none>} FUSION_CKPT=${FUSION_CKPT:-<none>}"

bash scripts/sbatch_eval_endovis2018.sh
