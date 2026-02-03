#!/bin/bash
#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -J prep_endovis2017_official
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.err

set -euo pipefail

source /home2020/home/icube/kunyuan/anaconda3/etc/profile.d/conda.sh
conda activate sam3

export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

REPO="${REPO:-/home2020/home/icube/kunyuan/SurgBench/SAM/sam3}"

# Official EndoVis2017 release (raw)
ENDOVIS_SRC="${ENDOVIS_SRC:-/home2020/home/icube/kunyuan/SurgBench/surgicaltool/Endovis2017}"

# Canonical processed dataset used by our training/eval scripts
# (keep lowercase to distinguish from raw official folder).
ENDOVIS_OUT="${ENDOVIS_OUT:-/home2020/home/icube/kunyuan/SurgBench/surgicaltool/endovis2017}"

CAMERA="${CAMERA:-left}"            # left | right
SYMLINK_IMAGES="${SYMLINK_IMAGES:-0}"  # 1 -> symlink images, 0 -> copy

cd "$REPO"

echo "REPO=$REPO"
echo "ENDOVIS_SRC=$ENDOVIS_SRC"
echo "ENDOVIS_OUT=$ENDOVIS_OUT"
echo "CAMERA=$CAMERA"
echo "SYMLINK_IMAGES=$SYMLINK_IMAGES"

SYMLINK_FLAG=""
if [ "$SYMLINK_IMAGES" = "1" ]; then
  SYMLINK_FLAG="--symlink-images"
fi

python -u scripts/prepare_endovis2017_official.py \
  --src-root "$ENDOVIS_SRC" \
  --out-root "$ENDOVIS_OUT" \
  --camera "$CAMERA" \
  $SYMLINK_FLAG \
  --overwrite

echo "=== Done ==="
echo "Meta: $ENDOVIS_OUT/endovis2017_prepared_meta.json"

