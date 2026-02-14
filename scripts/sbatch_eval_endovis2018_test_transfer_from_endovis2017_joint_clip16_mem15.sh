#!/bin/bash
# EndoVis2018: evaluate existing EndoVis2017-trained SPME modules on the *official test_data* split.
#
# What it does:
# - Prepares EndoVis2018 test_data -> canonical layout under `endovis2018_test` (if missing)
# - Runs eval for configs A (baseline), D (fusion-only), F (learned gate + fusion)
#
# Usage (server):
#   sbatch scripts/sbatch_eval_endovis2018_test_transfer_from_endovis2017_joint_clip16_mem15.sh
#
# Outputs:
#   /home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2018_spme_eval_test_transfer/job_${SLURM_JOB_ID}/...

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_endovis2018_eval_test_transfer
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.err
#SBATCH --exclude=hpc-n968

set -euo pipefail

# Canonical roots.
export ENDOVIS_SRC="${ENDOVIS_SRC:-/home2020/home/icube/kunyuan/SurgBench/surgicaltool/Endovis2018}"
export DATA="${DATA:-/home2020/home/icube/kunyuan/SurgBench/surgicaltool/endovis2018_test}"

# Make sure prepare stage uses only test_data (no leakage).
export PREPARE_DATA="${PREPARE_DATA:-auto}"
export ENDOVIS_GROUPS="test_data"

# MICCAI-safe eval defaults (match our EndoVis2017 protocol).
export OVERLAY_CKPT=""
export PROMPT="visual"
export PROPAGATION_MODE="vg"
export INIT_PROMPT="mask"
export INIT_FRAME_MODE="first_present"
export INIT_SELECT_MODE="prob"
export HOTSTART_DELAY="0"
export DEBUG_SPME_SUMMARY="1"
export SAM3_DISABLE_RECONDITION="1"

# Match joint clip16/mem15 models.
export SAM3_TRACKER_NUM_MASKMEM="${SAM3_TRACKER_NUM_MASKMEM:-15}"
export SAM3_SPME_GATE_HIDDEN="${SAM3_SPME_GATE_HIDDEN:-64}"
export ENDOVIS_FUSION_WEIGHT_MODE="${ENDOVIS_FUSION_WEIGHT_MODE:-eff_iou}"

# ---------------------------------------------------------------------------
# Learned-gate safety envelope (do-no-harm).
# ---------------------------------------------------------------------------
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET="${SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET:-1}"
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_USE_QCOS="${SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_USE_QCOS:-1}"
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_POW="${SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_POW:-4.0}"

export SAM3_SPME_LEARNED_GATE_OFFSET_STRENGTH="${SAM3_SPME_LEARNED_GATE_OFFSET_STRENGTH:-1.0}"
export SAM3_SPME_LEARNED_GATE_WRITE_STRENGTH="${SAM3_SPME_LEARNED_GATE_WRITE_STRENGTH:-1.0}"
export SAM3_SPME_LEARNED_GATE_DECAY_STRENGTH="${SAM3_SPME_LEARNED_GATE_DECAY_STRENGTH:-1.0}"

export SAM3_SPME_FUSION_EVENT_DRIVEN="${SAM3_SPME_FUSION_EVENT_DRIVEN:-1}"
export SAM3_SPME_FUSION_TRACKER_THR="${SAM3_SPME_FUSION_TRACKER_THR:-0.8}"

# ---------------------------------------------------------------------------
# Checkpoints: EndoVis2017-trained SPME modules (transfer to EndoVis2018).
# ---------------------------------------------------------------------------
# We intentionally do NOT hardcode a single job id here; instead we restrict the glob so
# `scripts/sbatch_eval_endovis2018.sh` will auto-pick the latest matching checkpoint.
export FUSION_CKPT_GLOB="${FUSION_CKPT_GLOB:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_fusion_train_clip16_mem15/job_*/main/checkpoints/spme_fusion_latest.pt}"
export GATE_CKPT_GLOB="${GATE_CKPT_GLOB:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_joint_train_clip16_mem15/job_*/main/checkpoints/spme_gate_latest.pt}"

# Which configs to run.
export RUN_CONFIGS="${RUN_CONFIGS:-A D F}"

export OUT_ROOT="${OUT_ROOT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2018_spme_eval_test_transfer/job_${SLURM_JOB_ID}}"

echo "[endovis2018_test_transfer] ENDOVIS_SRC=$ENDOVIS_SRC"
echo "[endovis2018_test_transfer] DATA=$DATA"
echo "[endovis2018_test_transfer] RUN_CONFIGS=$RUN_CONFIGS"
echo "[endovis2018_test_transfer] OUT_ROOT=$OUT_ROOT"

# Reuse the generic EndoVis2018 eval entrypoint (as bash; SBATCH headers inside are ignored here).
bash scripts/sbatch_eval_endovis2018.sh
