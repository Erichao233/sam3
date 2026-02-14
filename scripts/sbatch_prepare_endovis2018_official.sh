#!/bin/bash
# Prepare official EndoVis2018 -> canonical layout used by this repo.
#
# Usage (server):
#   sbatch scripts/sbatch_prepare_endovis2018_official.sh
#
# Outputs:
#   $ENDOVIS_OUT/{train,val*}/...
#   $ENDOVIS_OUT/endovis2018_prepared_meta.json

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -J prep_endovis2018_official
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.err

set -euo pipefail

source /home2020/home/icube/kunyuan/anaconda3/etc/profile.d/conda.sh
conda activate sam3

export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

REPO="${REPO:-/home2020/home/icube/kunyuan/SurgBench/SAM/sam3}"

# Official EndoVis2018 release (raw)
ENDOVIS_SRC="${ENDOVIS_SRC:-/home2020/home/icube/kunyuan/SurgBench/surgicaltool/Endovis2018}"

# Canonical processed dataset used by our training/eval scripts
# (keep lowercase to distinguish from raw official folder).
ENDOVIS_OUT="${ENDOVIS_OUT:-/home2020/home/icube/kunyuan/SurgBench/surgicaltool/endovis2018}"

CAMERA="${CAMERA:-left}"              # left | right
SYMLINK_IMAGES="${SYMLINK_IMAGES:-1}" # 1 -> symlink images, 0 -> copy
INCLUDE_TEST_DATA="${INCLUDE_TEST_DATA:-0}" # 1 -> also include top-level test_data/ if present
# Optional: restrict to specific group folders (space-separated)
#
# NOTE: Do NOT use the name `GROUPS` here — in bash it is a special array containing UNIX group IDs.
ENDOVIS_GROUPS="${ENDOVIS_GROUPS:-}"
SEQ_IDS="${SEQ_IDS:-}"               # Optional: restrict to seq ids (space-separated ints)
LABELS_JSON="${LABELS_JSON:-}"       # Optional: explicit labels.json path
MAX_FRAMES_PER_SEQ="${MAX_FRAMES_PER_SEQ:-}"  # Optional: debug limit per sequence

cd "$REPO"

echo "REPO=$REPO"
echo "ENDOVIS_SRC=$ENDOVIS_SRC"
echo "ENDOVIS_OUT=$ENDOVIS_OUT"
echo "CAMERA=$CAMERA"
echo "SYMLINK_IMAGES=$SYMLINK_IMAGES"
echo "INCLUDE_TEST_DATA=$INCLUDE_TEST_DATA"
echo "ENDOVIS_GROUPS=${ENDOVIS_GROUPS:-<all>}"
echo "SEQ_IDS=${SEQ_IDS:-<all>}"
echo "LABELS_JSON=${LABELS_JSON:-<auto>}"
echo "MAX_FRAMES_PER_SEQ=${MAX_FRAMES_PER_SEQ:-<all>}"

ARGS=(
  --src-root "$ENDOVIS_SRC"
  --out-root "$ENDOVIS_OUT"
  --camera "$CAMERA"
  --overwrite
)
if [ "$SYMLINK_IMAGES" = "1" ]; then
  ARGS+=(--symlink-images)
fi
if [ "$INCLUDE_TEST_DATA" = "1" ]; then
  ARGS+=(--include-test-data)
fi
if [[ -n "${ENDOVIS_GROUPS:-}" ]]; then
  ARGS+=(--groups $ENDOVIS_GROUPS)
fi
if [[ -n "${SEQ_IDS:-}" ]]; then
  ARGS+=(--seq-ids $SEQ_IDS)
fi
if [[ -n "${LABELS_JSON:-}" ]]; then
  ARGS+=(--labels-json "$LABELS_JSON")
fi
if [[ -n "${MAX_FRAMES_PER_SEQ:-}" ]]; then
  ARGS+=(--max-frames-per-seq "$MAX_FRAMES_PER_SEQ")
fi

python -u scripts/prepare_endovis2018_official.py "${ARGS[@]}"

echo "=== Done ==="
echo "Meta: $ENDOVIS_OUT/endovis2018_prepared_meta.json"
