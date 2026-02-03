#!/bin/bash
#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_spme_fusion_ovis_train
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.err

set -euo pipefail

source /home2020/home/icube/kunyuan/anaconda3/etc/profile.d/conda.sh
conda activate sam3

export OMP_NUM_THREADS=8
export HYDRA_FULL_ERROR=1
export SAM3_DISABLE_TRITON=1
export PYTHONUNBUFFERED=1

# If pycocotools is missing in this env, install it once.
python - <<'PY' || pip install -q pycocotools
import importlib.util
assert importlib.util.find_spec("pycocotools") is not None
print("pycocotools OK")
PY

REPO=/home2020/home/icube/kunyuan/SurgBench/SAM/sam3
# Default to your extracted OVIS location (train has GT; valid/test have no GT).
OVIS=${OVIS_ROOT:-/home2020/home/icube/kunyuan/SurgBench/Ultrasound/OVIS}

SAM3_PT=$REPO/sam3.pt
BPE=$REPO/sam3/assets/bpe_simple_vocab_16e6.txt.gz

# Splits from scripts/ovis_make_splits.py
SPLIT_DIR=${SPLIT_DIR:-$OVIS/splits_seed0}
TRAIN_SPLIT=$SPLIT_DIR/train.txt

# Frames root (directory containing per-video folders, e.g. train/<video_id>/img_*.jpg).
# Your extracted layout appears to be: $OVIS/train/<video_id>/img_*.jpg
IMAGES_ROOT=${IMAGES_ROOT:-$OVIS/train}
ANN=$OVIS/annotations_train.json

OUT_DIR=${OUT_DIR:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/spme_fusion_ovis_train_v1/main}

cd "$REPO"

mkdir -p "$SPLIT_DIR"
if [ ! -s "$TRAIN_SPLIT" ]; then
  echo "[OVIS] split files not found; generating at $SPLIT_DIR"
  python scripts/ovis_make_splits.py \
    --ann "$ANN" \
    --out-dir "$SPLIT_DIR" \
    --seed 0 \
    --train-frac 0.8 \
    --val-frac 0.1
fi

python -u scripts/train_spme_fusion_ovis.py \
  --images-root "$IMAGES_ROOT" \
  --ann "$ANN" \
  --split-txt "$TRAIN_SPLIT" \
  --out-dir "$OUT_DIR" \
  --base-sam3-pt "$SAM3_PT" \
  --bpe-path "$BPE" \
  --seed 0 \
  --query-pool top1 \
  --query-topk 5 \
  --anchor-det-thr 0.0 \
  --pointer-mode hybrid \
  --match-iou-thr 0.0 \
  --score-thr-detection 0.0 \
  --clip-len 16 \
  --max-steps 20000 \
  --lr 5e-4 \
  --weight-decay 0.0 \
  --fusion-mode film \
  --fusion-alpha 0.05 \
  --fusion-alpha-obj 0.005 \
  --fusion-det-thr 0.0 \
  --fusion-det-score-floor 1e-4 \
  --fusion-qcos-thr 0.5 \
  --fusion-qcos-temp 20.0 \
  --fusion-qcos-gate sigmoid \
  --fusion-use-presence 1 \
  --debug-first-n 10 \
  --log-every 50 \
  --save-every 2000

echo "Done. Latest ckpt: $OUT_DIR/checkpoints/spme_fusion_latest.pt"
