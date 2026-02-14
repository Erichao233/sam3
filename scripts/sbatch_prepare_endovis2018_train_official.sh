#!/bin/bash
# Prepare EndoVis2018 *training* releases -> canonical layout (train root only; excludes test_data).
#
# Usage (server):
#   sbatch scripts/sbatch_prepare_endovis2018_train_official.sh
#
# Outputs:
#   /home2020/home/icube/kunyuan/SurgBench/surgicaltool/endovis2018_train/{train,val*}/...

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -J prep_endovis2018_train_official
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.err

set -euo pipefail

source /home2020/home/icube/kunyuan/anaconda3/etc/profile.d/conda.sh
conda activate sam3

export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

REPO="${REPO:-/home2020/home/icube/kunyuan/SurgBench/SAM/sam3}"
ENDOVIS_SRC="${ENDOVIS_SRC:-/home2020/home/icube/kunyuan/SurgBench/surgicaltool/Endovis2018}"
ENDOVIS_OUT="${ENDOVIS_OUT:-/home2020/home/icube/kunyuan/SurgBench/surgicaltool/endovis2018_train}"

CAMERA="${CAMERA:-left}"
SYMLINK_IMAGES="${SYMLINK_IMAGES:-1}"

# Training releases (with labels).
#
# NOTE: Do NOT use the name `GROUPS` here — in bash it is a special array containing UNIX group IDs,
# which would silently break `--groups` filtering.
ENDOVIS_GROUPS="${ENDOVIS_GROUPS:-miccai_challenge_2018_release_1 miccai_challenge_release_2 miccai_challenge_release_3 miccai_challenge_release_4}"

cd "$REPO"

echo "REPO=$REPO"
echo "ENDOVIS_SRC=$ENDOVIS_SRC"
echo "ENDOVIS_OUT=$ENDOVIS_OUT"
echo "CAMERA=$CAMERA"
echo "SYMLINK_IMAGES=$SYMLINK_IMAGES"
echo "ENDOVIS_GROUPS=$ENDOVIS_GROUPS"

ARGS=(
  --src-root "$ENDOVIS_SRC"
  --out-root "$ENDOVIS_OUT"
  --camera "$CAMERA"
  --groups $ENDOVIS_GROUPS
  --overwrite
)
if [ "$SYMLINK_IMAGES" = "1" ]; then
  ARGS+=(--symlink-images)
fi

python -u scripts/prepare_endovis2018_official.py "${ARGS[@]}"

echo "=== Done ==="
echo "Meta: $ENDOVIS_OUT/endovis2018_prepared_meta.json"
