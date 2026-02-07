#!/bin/bash
# Sweep ABSENT_W to avoid gate collapse (EndoVis2017).
#
# Runs 3 array tasks with ABSENT_W in {0.05, 0.1, 0.2}, reusing
# `scripts/sbatch_train_spme_gate_endovis2017.sh` (which already sets
# reasonable defaults for prompt/mode/freeze_fusion/etc).
#
# Usage (server):
#   sbatch scripts/sbatch_train_spme_gate_endovis2017_sweep_absent.sh
#
# Outputs:
#   /home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_gate_train_sweep_absent/job_${SLURM_JOB_ID}/aw${ABSENT_W}/...

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -J sam3_endovis2017_gate_sweep_absent
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%A_%a.err
#SBATCH --array=0-2

set -euo pipefail

ABSENT_WS=(0.05 0.1 0.2)
ABSENT_W="${ABSENT_WS[${SLURM_ARRAY_TASK_ID}]}"

export ABSENT_W
export OUT_ROOT="/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_gate_train_sweep_absent/job_${SLURM_JOB_ID}/aw${ABSENT_W}"

echo "[sweep] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} ABSENT_W=${ABSENT_W}"
echo "[sweep] OUT_ROOT=${OUT_ROOT}"

# Run the main gate training sbatch script as a normal bash script (SBATCH headers are ignored here).
bash scripts/sbatch_train_spme_gate_endovis2017.sh
