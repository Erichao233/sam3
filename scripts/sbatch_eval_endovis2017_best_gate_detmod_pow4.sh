#!/bin/bash
# EndoVis2017: paper-ready eval with the current best learned-gate calibration.
#
# Protocol (MICCAI-safe):
# - visual-only prompting
# - mask init at the first visible GT frame (semi-supervised tracking-with-init)
# - VG propagation (so SPME edits are applied)
# - no detector finetune overlay
#
# Key setting:
# - enable detector-modulated learned gate with a sharp exponent (DET_POW=4.0)
#
# What it runs:
# - A: baseline
# - D: fusion-only (SPME-F)
# - E: learned gate only (SPME-G)
# - F: learned gate + fusion
#
# Usage (server):
#   sbatch scripts/sbatch_eval_endovis2017_best_gate_detmod_pow4.sh
#
# Outputs:
#   /home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_best_gate_detmod_pow4/job_${SLURM_JOB_ID}/...

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_endovis2017_eval_best_gate_detmod_pow4
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Fixed eval protocol
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
# Checkpoints (override if needed)
# ---------------------------------------------------------------------------
export FUSION_CKPT="${FUSION_CKPT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_fusion_train_v1/job_16015880/main/checkpoints/spme_fusion_latest.pt}"
export GATE_CKPT="${GATE_CKPT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_gate_train_sweep_absent/job_16100471/aw0.1/main/checkpoints/spme_gate_latest.pt}"

# ---------------------------------------------------------------------------
# Learned gate: detector-modulated constraint (best so far)
# ---------------------------------------------------------------------------
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET="1"
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_POW="4.0"

# Keep calibration strengths at identity (no extra scaling).
export SAM3_SPME_LEARNED_GATE_OFFSET_STRENGTH="${SAM3_SPME_LEARNED_GATE_OFFSET_STRENGTH:-1.0}"
export SAM3_SPME_LEARNED_GATE_WRITE_STRENGTH="${SAM3_SPME_LEARNED_GATE_WRITE_STRENGTH:-1.0}"
export SAM3_SPME_LEARNED_GATE_DECAY_STRENGTH="${SAM3_SPME_LEARNED_GATE_DECAY_STRENGTH:-1.0}"

# ---------------------------------------------------------------------------
# Which configs to run
# ---------------------------------------------------------------------------
export RUN_CONFIGS="${RUN_CONFIGS:-A D E F}"

export OUT_ROOT="/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_best_gate_detmod_pow4/job_${SLURM_JOB_ID}"

echo "[best_detmod_pow4] RUN_CONFIGS=${RUN_CONFIGS}"
echo "[best_detmod_pow4] OUT_ROOT=${OUT_ROOT}"
echo "[best_detmod_pow4] FUSION_CKPT=${FUSION_CKPT}"
echo "[best_detmod_pow4] GATE_CKPT=${GATE_CKPT}"
echo "[best_detmod_pow4] SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET=${SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET}"
echo "[best_detmod_pow4] SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_POW=${SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_POW}"

# Run the main eval script (as bash; SBATCH headers inside are ignored here).
bash scripts/sbatch_eval_endovis2017.sh

