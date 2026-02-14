#!/bin/bash
# Prepare EndoVis2018 *test_data* -> canonical layout (keeps test separate from training releases).
#
# Usage (server):
#   sbatch scripts/sbatch_prepare_endovis2018_test_official.sh
#
# Outputs:
#   /home2020/home/icube/kunyuan/SurgBench/surgicaltool/endovis2018_test/val*/...
#   /home2020/home/icube/kunyuan/SurgBench/surgicaltool/endovis2018_test/endovis2018_prepared_meta.json
#   /home2020/home/icube/kunyuan/SurgBench/surgicaltool/endovis2018_test/prepare_stats.json
#
# Notes:
# - The official EndoVis2018 root contains `test_data/seq_1..4` which is different from the training releases.
# - We keep it in a separate canonical root (`endovis2018_test`) to avoid leakage into training.

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -J prep_endovis2018_test_official
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.err

set -euo pipefail

source /home2020/home/icube/kunyuan/anaconda3/etc/profile.d/conda.sh
conda activate sam3

export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

REPO="${REPO:-/home2020/home/icube/kunyuan/SurgBench/SAM/sam3}"
ENDOVIS_SRC="${ENDOVIS_SRC:-/home2020/home/icube/kunyuan/SurgBench/surgicaltool/Endovis2018}"
ENDOVIS_OUT="${ENDOVIS_OUT:-/home2020/home/icube/kunyuan/SurgBench/surgicaltool/endovis2018_test}"

CAMERA="${CAMERA:-left}"
SYMLINK_IMAGES="${SYMLINK_IMAGES:-1}"

cd "$REPO"

echo "REPO=$REPO"
echo "ENDOVIS_SRC=$ENDOVIS_SRC"
echo "ENDOVIS_OUT=$ENDOVIS_OUT"
echo "CAMERA=$CAMERA"
echo "SYMLINK_IMAGES=$SYMLINK_IMAGES"

ARGS=(
  --src-root "$ENDOVIS_SRC"
  --out-root "$ENDOVIS_OUT"
  --camera "$CAMERA"
  --groups test_data
  --no-train-view
  --overwrite
)
if [ "$SYMLINK_IMAGES" = "1" ]; then
  ARGS+=(--symlink-images)
fi

python -u scripts/prepare_endovis2018_official.py "${ARGS[@]}"

echo "=== Done ==="
echo "Meta: $ENDOVIS_OUT/endovis2018_prepared_meta.json"
