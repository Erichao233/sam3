#!/bin/bash
# Compare gate checkpoints from the ABSENT_W sweep on EndoVis2017.
#
# This script runs EndoVis eval for (E) learned gate only and (F) learned gate + fusion
# for each gate checkpoint from the sweep. For convenience, task 0 also runs baseline (A)
# and fusion-only (D) once as a reference.
#
# Usage (server):
#   sbatch scripts/sbatch_eval_endovis2017_sweep_gate_absent.sh
#
# Outputs:
#   /home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_gate_sweep_absent/job_${SLURM_JOB_ID}/aw${ABSENT_W}/...
#
# Notes:
# - No OVERLAY_CKPT is used (pure visual init).
# - Uses INIT_PROMPT=mask and INIT_FRAME_MODE=first_present (no future leakage).

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_endovis2017_eval_gate_sweep_absent
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.err
#SBATCH --array=0-2

set -euo pipefail

# Gate checkpoints from your sweep (aw0.05 / aw0.1 / aw0.2).
GATE_CKPTS=(
  "/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_gate_train_sweep_absent/job_16100470/aw0.05/main/checkpoints/spme_gate_latest.pt"
  "/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_gate_train_sweep_absent/job_16100471/aw0.1/main/checkpoints/spme_gate_latest.pt"
  "/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_gate_train_sweep_absent/job_16100469/aw0.2/main/checkpoints/spme_gate_latest.pt"
)
ABSENT_WS=(0.05 0.1 0.2)

IDX="${SLURM_ARRAY_TASK_ID}"
GATE_CKPT="${GATE_CKPTS[$IDX]}"
ABSENT_W="${ABSENT_WS[$IDX]}"

export OVERLAY_CKPT=""           # do NOT use detector finetune for this sweep
export PROMPT="visual"
export PROPAGATION_MODE="vg"
export INIT_PROMPT="mask"
export INIT_FRAME_MODE="first_present"
export DEBUG_SPME_SUMMARY="1"

# ---------------------------------------------------------------------------
# Evaluate all sweep checkpoints under the *current best* inference envelope,
# so the comparison reflects the paper-facing pipeline (not an unsafe default).
# ---------------------------------------------------------------------------
# Disable SAM3 periodic recondition (detector→tracker overwrite) to avoid confounds.
export SAM3_DISABLE_RECONDITION="1"

# Best learned-gate safety envelope so far: det×qcos modulation with pow=4.
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET="1"
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_USE_QCOS="1"
export SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_POW="4.0"

# Make fusion overseer-style by default (prevents regressions on EndoVis).
export SAM3_SPME_FUSION_EVENT_DRIVEN="${SAM3_SPME_FUSION_EVENT_DRIVEN:-1}"
export SAM3_SPME_FUSION_EVENT_MODE="${SAM3_SPME_FUSION_EVENT_MODE:-mismatch}"
export SAM3_SPME_FUSION_MISMATCH_IOU_THR="${SAM3_SPME_FUSION_MISMATCH_IOU_THR:-0.2}"
# Needed for mismatch-driven fusion on drift frames (det↔trk IoU matching can be empty during drift).
export SAM3_SPME_PER_OBJECT_ANCHOR_MATCH="${SAM3_SPME_PER_OBJECT_ANCHOR_MATCH:-1}"

# For each gate checkpoint we compare E/F. Task 0 also runs A/D once for reference.
if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
  export RUN_CONFIGS="A D E F"
else
  export RUN_CONFIGS="E F"
fi

export GATE_CKPT

export OUT_ROOT="/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval_gate_sweep_absent/job_${SLURM_JOB_ID}/aw${ABSENT_W}"

echo "[gate_sweep] task=${SLURM_ARRAY_TASK_ID} ABSENT_W=${ABSENT_W}"
echo "[gate_sweep] GATE_CKPT=${GATE_CKPT}"
echo "[gate_sweep] RUN_CONFIGS=${RUN_CONFIGS}"
echo "[gate_sweep] OUT_ROOT=${OUT_ROOT}"

# Run the main eval script (as bash; SBATCH headers inside are ignored here).
bash scripts/sbatch_eval_endovis2017.sh
