#!/bin/bash
# EndoVis2017: Ablate learned-gate decay on/off (best ABSENT_W checkpoint).
#
# What it runs:
# - Task 0: baseline (A) + fusion-only (D) as references
# - Task 1: learned gate only (E) with decay ON
# - Task 2: learned gate only (E) with decay OFF
# - Task 3: learned gate + fusion (F) with decay ON
# - Task 4: learned gate + fusion (F) with decay OFF
#
# Usage (server):
#   sbatch scripts/sbatch_eval_endovis2017_ablate_gate_decay.sh
#
# Outputs (per task):
#   /home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_ablate_gate_decay/job_${SLURM_JOB_ID}/task${SLURM_ARRAY_TASK_ID}_decay${GATE_USE_DECAY}/...

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_endovis2017_eval_ablate_gate_decay
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.err
#SBATCH --array=0-4

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
# Best gate ckpt from ABSENT_W sweep (aw0.1).
export GATE_CKPT="${GATE_CKPT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_gate_train_sweep_absent/job_16100471/aw0.1/main/checkpoints/spme_gate_latest.pt}"

# ---------------------------------------------------------------------------
# SPME / fusion defaults (match training/eval assumptions).
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

# Optional confound ablation (set to 1 to disable SAM3 periodic recondition).
export SAM3_DISABLE_RECONDITION="${SAM3_DISABLE_RECONDITION:-0}"

TASK="${SLURM_ARRAY_TASK_ID}"

case "$TASK" in
  0)
    export RUN_CONFIGS="A D"
    export GATE_USE_DECAY="1"  # unused here
    ;;
  1)
    export RUN_CONFIGS="E"
    export GATE_USE_DECAY="1"
    ;;
  2)
    export RUN_CONFIGS="E"
    export GATE_USE_DECAY="0"
    ;;
  3)
    export RUN_CONFIGS="F"
    export GATE_USE_DECAY="1"
    ;;
  4)
    export RUN_CONFIGS="F"
    export GATE_USE_DECAY="0"
    ;;
  *)
    echo "[error] unexpected SLURM_ARRAY_TASK_ID=$TASK"
    exit 1
    ;;
esac

export OUT_ROOT="/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_ablate_gate_decay/job_${SLURM_JOB_ID}/task${TASK}_decay${GATE_USE_DECAY}"

echo "[ablate_gate_decay] task=$TASK"
echo "[ablate_gate_decay] RUN_CONFIGS=$RUN_CONFIGS"
echo "[ablate_gate_decay] GATE_USE_DECAY=$GATE_USE_DECAY"
echo "[ablate_gate_decay] OUT_ROOT=$OUT_ROOT"
echo "[ablate_gate_decay] GATE_CKPT=$GATE_CKPT"
echo "[ablate_gate_decay] FUSION_CKPT=$FUSION_CKPT"

# Run the main eval script (as bash; SBATCH headers inside are ignored here).
bash scripts/sbatch_eval_endovis2017.sh

