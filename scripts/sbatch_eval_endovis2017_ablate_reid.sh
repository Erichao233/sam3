#!/bin/bash
# EndoVis2017: ablate ReID-assisted re-init (ReMeDI-style) ON/OFF.
#
# Motivation (paper-safe):
# - Our re-init triggers are conservative; without identity validation, allowing re-detection (tracker empty)
#   can create ghosts.
# - arXiv:2512.16880v1 Sec 3.3.3 proposes a lightweight feature-bank re-identification to validate recovery
#   after occlusions. We implement an analogous "masked-mean descriptor bank" to decide whether a detector
#   candidate is identity-consistent before re-initializing memory.
#
# Protocol (MICCAI-safe):
# - PROMPT=visual, INIT_PROMPT=mask, INIT_FRAME_MODE=first_present, PROPAGATION_MODE=vg
# - no detector finetune overlay
# - recondition disabled (avoid confounds)
#
# Tasks:
#   0: Learned gate only (E), re-init OFF (baseline reference)
#   1: Learned gate only (E), re-init ON, mismatch trigger ON, allow_empty OFF (safe)
#   2: Learned gate only (E), re-init ON, mismatch trigger ON, allow_empty ON,  ReID OFF (risky)
#   3: Learned gate only (E), re-init ON, mismatch trigger ON, allow_empty ON,  ReID ON  (proposed)
#
# Usage (server):
#   sbatch scripts/sbatch_eval_endovis2017_ablate_reid.sh
#
# Outputs:
#   /home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_ablate_reid/job_${SLURM_JOB_ID}/task${TASK}_reid${SAM3_SPME_REID}_empty${SAM3_SPME_REINIT_ALLOW_EMPTY}/...

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_endovis2017_eval_ablate_reid
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

# ---------------------------------------------------------------------------
# Best learned-gate safety envelope so far (keep fixed while ablating re-id).
# ---------------------------------------------------------------------------
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
# Re-init defaults.
# ---------------------------------------------------------------------------
export SAM3_SPME_REINIT_MISS_THR="${SAM3_SPME_REINIT_MISS_THR:-3}"
export SAM3_SPME_REINIT_CONFIRM_THR="${SAM3_SPME_REINIT_CONFIRM_THR:-2}"
export SAM3_SPME_REINIT_COOLDOWN="${SAM3_SPME_REINIT_COOLDOWN:-8}"
export SAM3_SPME_REINIT_DET_THR="${SAM3_SPME_REINIT_DET_THR:-0.7}"
export SAM3_SPME_REINIT_QCOS_THR="${SAM3_SPME_REINIT_QCOS_THR:-0.7}"
export SAM3_SPME_REINIT_TRACKER_THR="${SAM3_SPME_REINIT_TRACKER_THR:-0.8}"
export SAM3_SPME_REINIT_CONFIRM_IOU_THR="${SAM3_SPME_REINIT_CONFIRM_IOU_THR:-0.2}"
export SAM3_SPME_REINIT_IOU_THR="${SAM3_SPME_REINIT_IOU_THR:-0.2}"
export SAM3_SPME_REINIT_MISMATCH_THR="${SAM3_SPME_REINIT_MISMATCH_THR:-2}"

# ---------------------------------------------------------------------------
# ReID settings (ReMeDI-style, training-free).
# ---------------------------------------------------------------------------
export SAM3_SPME_REID_BANK_SIZE="${SAM3_SPME_REID_BANK_SIZE:-20}"
export SAM3_SPME_REID_MARGIN="${SAM3_SPME_REID_MARGIN:-0.01}"
# NOTE: In single-object tracking (our EndoVis per-class loop), cross-class banks are unavailable,
# so ReID falls back to an absolute self-sim threshold. Keep it >0 to make ReID meaningful.
export SAM3_SPME_REID_SELF_THR="${SAM3_SPME_REID_SELF_THR:-0.3}"
export SAM3_SPME_REID_USE_OTHER_BANKS="${SAM3_SPME_REID_USE_OTHER_BANKS:-1}"
export SAM3_SPME_REID_UPDATE="${SAM3_SPME_REID_UPDATE:-1}"

TASK="${SLURM_ARRAY_TASK_ID}"

case "$TASK" in
  0)
    export RUN_CONFIGS="E"
    export SAM3_SPME_REINIT="0"
    export SAM3_SPME_REINIT_ENABLE_MISMATCH="0"
    export SAM3_SPME_REINIT_ALLOW_EMPTY="0"
    export SAM3_SPME_REID="0"
    ;;
  1)
    export RUN_CONFIGS="E"
    export SAM3_SPME_REINIT="1"
    export SAM3_SPME_REINIT_ENABLE_MISMATCH="1"
    export SAM3_SPME_REINIT_ALLOW_EMPTY="0"
    export SAM3_SPME_REID="0"
    ;;
  2)
    export RUN_CONFIGS="E"
    export SAM3_SPME_REINIT="1"
    export SAM3_SPME_REINIT_ENABLE_MISMATCH="1"
    export SAM3_SPME_REINIT_ALLOW_EMPTY="1"
    export SAM3_SPME_REID="0"
    ;;
  3)
    export RUN_CONFIGS="E"
    export SAM3_SPME_REINIT="1"
    export SAM3_SPME_REINIT_ENABLE_MISMATCH="1"
    export SAM3_SPME_REINIT_ALLOW_EMPTY="1"
    export SAM3_SPME_REID="1"
    ;;
  *)
    echo "[error] unexpected SLURM_ARRAY_TASK_ID=$TASK"
    exit 1
    ;;
esac

export OUT_ROOT="/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_ablate_reid/job_${SLURM_JOB_ID}/task${TASK}_reid${SAM3_SPME_REID}_empty${SAM3_SPME_REINIT_ALLOW_EMPTY}"

echo "[ablate_reid] task=$TASK"
echo "[ablate_reid] RUN_CONFIGS=$RUN_CONFIGS"
echo "[ablate_reid] OUT_ROOT=$OUT_ROOT"
echo "[ablate_reid] GATE_CKPT=$GATE_CKPT"
echo "[ablate_reid] SAM3_SPME_REINIT=$SAM3_SPME_REINIT"
echo "[ablate_reid] SAM3_SPME_REINIT_ENABLE_MISMATCH=$SAM3_SPME_REINIT_ENABLE_MISMATCH"
echo "[ablate_reid] SAM3_SPME_REINIT_ALLOW_EMPTY=$SAM3_SPME_REINIT_ALLOW_EMPTY"
echo "[ablate_reid] SAM3_SPME_REID=$SAM3_SPME_REID"
echo "[ablate_reid] SAM3_DISABLE_RECONDITION=$SAM3_DISABLE_RECONDITION"

# Run the main eval script (as bash; SBATCH headers inside are ignored here).
bash scripts/sbatch_eval_endovis2017.sh
