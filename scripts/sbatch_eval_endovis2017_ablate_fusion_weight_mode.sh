#!/bin/bash
# EndoVis2017: ablate quality-weighted fusion weight mode (paper alignment).
#
# Motivation (paper-safe):
# - arXiv:2512.16880v1 uses quality-weighted mask fusion (Sec 3.3.4).
# - In this repo, fusion weights affect the final multi-class label map and thus mcIoU.
#
# Protocol (MICCAI-safe):
# - PROMPT=visual, INIT_PROMPT=mask, INIT_FRAME_MODE=first_present, PROPAGATION_MODE=vg
# - no detector finetune overlay
# - recondition disabled (avoid confounds)
#
# Tasks:
#   0: baseline (A) + learned gate only (E), weight=tracker_prob (reference)
#   1: learned gate only (E), weight=det_prob
#   2: learned gate only (E), weight=eff_iou
#   3: learned gate only (E), weight=reliability (st*ct proxy; closer to ReMeDI wording)
#
# Usage (server):
#   sbatch scripts/sbatch_eval_endovis2017_ablate_fusion_weight_mode.sh
#
# Outputs:
#   /home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_ablate_fusion_weight/job_${SLURM_JOB_ID}/w${WEIGHT_MODE}/...

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_endovis2017_eval_ablate_fusion_weight
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

# Use the current best learned-gate safety envelope by default.
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET="1"
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_USE_QCOS="1"
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_POW="4.0"

# Gate architecture knobs (must match the checkpoint).
export GATE_INPUTS="${GATE_INPUTS:-full}"
export GATE_FUSION_HEAD="${GATE_FUSION_HEAD:-1}"
export GATE_USE_DECAY="${GATE_USE_DECAY:-1}"
export GATE_OCC_NORM="${GATE_OCC_NORM:-10.0}"
export GATE_DET_PRESENT_THR="${GATE_DET_PRESENT_THR:-0.3}"

# Keep SPME-Fusion injection off for this ablation (we only want output-level fusion weights).
export FUSION_CKPT=""
export RUN_CONFIGS="E"

TASK="${SLURM_ARRAY_TASK_ID}"
case "$TASK" in
  0) WEIGHT_MODE="tracker_prob"; export RUN_CONFIGS="A E" ;;
  1) WEIGHT_MODE="det_prob" ;;
  2) WEIGHT_MODE="eff_iou" ;;
  3) WEIGHT_MODE="reliability" ;;
  *) echo "[error] unexpected SLURM_ARRAY_TASK_ID=$TASK"; exit 1 ;;
esac

export ENDOVIS_FUSION_WEIGHT_MODE="${WEIGHT_MODE}"

export OUT_ROOT="/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_ablate_fusion_weight/job_${SLURM_JOB_ID}/w${WEIGHT_MODE}"

echo "[ablate_fusion_weight] task=$TASK weight=$WEIGHT_MODE"
echo "[ablate_fusion_weight] RUN_CONFIGS=$RUN_CONFIGS"
echo "[ablate_fusion_weight] OUT_ROOT=$OUT_ROOT"
echo "[ablate_fusion_weight] ENDOVIS_FUSION_WEIGHT_MODE=$ENDOVIS_FUSION_WEIGHT_MODE"
echo "[ablate_fusion_weight] GATE_CKPT=$GATE_CKPT"

# Run the main eval script (as bash; SBATCH headers inside are ignored here).
bash scripts/sbatch_eval_endovis2017.sh
