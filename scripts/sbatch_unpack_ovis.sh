#!/bin/bash
#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -J unpack_ovis
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail

# Usage (server):
#   export OVIS_ROOT=/path/to/OVIS
#   sbatch scripts/sbatch_unpack_ovis.sh
#
# Expected inputs:
#   $OVIS_ROOT/Images/train/train.z01 ... train.zNN + train.zip
#   $OVIS_ROOT/Images/valid.zip
#   $OVIS_ROOT/Images/test.zip

OVIS_ROOT="${OVIS_ROOT:-/home2020/home/icube/kunyuan/SurgBench/Ultrasound/OVIS}"

echo "OVIS_ROOT=$OVIS_ROOT"
echo "==== Checking archives ===="
ls -lh "$OVIS_ROOT/Images/train"/train.z* "$OVIS_ROOT/Images/train/train.zip" || true
ls -lh "$OVIS_ROOT/Images/valid.zip" "$OVIS_ROOT/Images/test.zip" || true

echo "==== Unsplit + unzip train ===="
cd "$OVIS_ROOT/Images/train"

# Prefer `zip -s 0` which properly merges split archives (train.z01 + ... + train.zip).
if [ ! -f train_full.zip ]; then
  echo "[train] merging split archive -> train_full.zip"
  if ! zip -s 0 train.zip --out train_full.zip; then
    echo "[train] zip -s failed; trying repair with zip -FF -> train_full.zip"
    rm -f train_full.zip
    zip -FF train.zip --out train_full.zip
  fi
fi

echo "[train] testing merged zip"
if ! unzip -tq train_full.zip >/dev/null; then
  echo "[train] zip test failed; trying repair with zip -FF -> train_full.zip"
  rm -f train_full.zip
  zip -FF train.zip --out train_full.zip
  unzip -tq train_full.zip >/dev/null
fi

echo "[train] extracting"
mkdir -p "$OVIS_ROOT/Images/train_extracted"
unzip -q train_full.zip -d "$OVIS_ROOT/Images/train_extracted"
if [ -d "$OVIS_ROOT/Images/train_extracted/train" ]; then
  # Flatten one level: train_extracted/train/<vid>/img_*.jpg -> train_extracted/<vid>/img_*.jpg
  echo "[train] flattening train_extracted/train/* -> train_extracted/"
  shopt -s dotglob nullglob
  mv "$OVIS_ROOT/Images/train_extracted/train/"* "$OVIS_ROOT/Images/train_extracted/" || true
  rmdir "$OVIS_ROOT/Images/train_extracted/train" || true
fi

echo "==== Unzip valid/test ===="
cd "$OVIS_ROOT/Images"
mkdir -p "$OVIS_ROOT/Images/valid_extracted" "$OVIS_ROOT/Images/test_extracted"
unzip -q valid.zip -d "$OVIS_ROOT/Images/valid_extracted"
unzip -q test.zip -d "$OVIS_ROOT/Images/test_extracted"
if [ -d "$OVIS_ROOT/Images/valid_extracted/valid" ]; then
  echo "[valid] flattening valid_extracted/valid/* -> valid_extracted/"
  shopt -s dotglob nullglob
  mv "$OVIS_ROOT/Images/valid_extracted/valid/"* "$OVIS_ROOT/Images/valid_extracted/" || true
  rmdir "$OVIS_ROOT/Images/valid_extracted/valid" || true
fi
if [ -d "$OVIS_ROOT/Images/test_extracted/test" ]; then
  echo "[test] flattening test_extracted/test/* -> test_extracted/"
  shopt -s dotglob nullglob
  mv "$OVIS_ROOT/Images/test_extracted/test/"* "$OVIS_ROOT/Images/test_extracted/" || true
  rmdir "$OVIS_ROOT/Images/test_extracted/test" || true
fi

echo "==== Done ===="
echo "train -> $OVIS_ROOT/Images/train_extracted"
echo "valid -> $OVIS_ROOT/Images/valid_extracted"
echo "test  -> $OVIS_ROOT/Images/test_extracted"
