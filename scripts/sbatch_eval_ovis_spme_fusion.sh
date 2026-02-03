#!/bin/bash
#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -J sam3_spme_fusion_ovis_eval
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.err

set -euo pipefail

source /home2020/home/icube/kunyuan/anaconda3/etc/profile.d/conda.sh
conda activate sam3

export OMP_NUM_THREADS=8
export HYDRA_FULL_ERROR=1
export SAM3_DISABLE_TRITON=1
export PYTHONUNBUFFERED=1

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

SPLIT_DIR=${SPLIT_DIR:-$OVIS/splits_seed0}
VAL_SPLIT=$SPLIT_DIR/val.txt
TEST_SPLIT=$SPLIT_DIR/test.txt

# Frames root (directory containing per-video folders, e.g. train/<video_id>/img_*.jpg).
# Your extracted layout appears to be: $OVIS/train/<video_id>/img_*.jpg
IMAGES_ROOT=${IMAGES_ROOT:-$OVIS/train}
ANN=$OVIS/annotations_train.json

# Trained fusion (optional). Point to your trained ckpt.
FUSION_CKPT=${FUSION_CKPT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/spme_fusion_ovis_train_v1/main/checkpoints/spme_fusion_latest.pt}

OUT_ROOT=${OUT_ROOT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/spme_fusion_ovis_eval_v1}

MAX_INSTANCES=${MAX_INSTANCES:-200}
MAX_FRAMES=${MAX_FRAMES:-128}
SAVE_SIGNALS=${SAVE_SIGNALS:-0}
SAVE_SIGNALS_FOR=${SAVE_SIGNALS_FOR:-}
ONLY_PAIRS=${ONLY_PAIRS:-}
RUN_VAL=${RUN_VAL:-1}
RUN_TEST=${RUN_TEST:-1}

# Fusion eval knobs (override via env without editing this file).
FUSION_TAG=${FUSION_TAG:-spme_fusion_trained}
FUSION_MODE=${FUSION_MODE:-film}
FUSION_ALPHA=${FUSION_ALPHA:-0.05}
FUSION_ALPHA_OBJ=${FUSION_ALPHA_OBJ:-0.005}
FUSION_DET_THR=${FUSION_DET_THR:-0.0}
FUSION_USE_PRESENCE=${FUSION_USE_PRESENCE:-1}
FUSION_QCOS_THR=${FUSION_QCOS_THR:-0.5}
FUSION_QCOS_TEMP=${FUSION_QCOS_TEMP:-20.0}
FUSION_QCOS_GATE=${FUSION_QCOS_GATE:-sigmoid}

EXTRA_SIGNALS_ARGS=()
if [ -n "$SAVE_SIGNALS_FOR" ]; then
  EXTRA_SIGNALS_ARGS+=(--save-signals-for "$SAVE_SIGNALS_FOR")
  SAVE_SIGNALS=1
fi
if [ -n "$ONLY_PAIRS" ]; then
  EXTRA_SIGNALS_ARGS+=(--only-pairs "$ONLY_PAIRS")
fi

cd "$REPO"

mkdir -p "$SPLIT_DIR"
if [ ! -s "$VAL_SPLIT" ] || [ ! -s "$TEST_SPLIT" ]; then
  echo "[OVIS] split files not found; generating at $SPLIT_DIR"
  python scripts/ovis_make_splits.py \
    --ann "$ANN" \
    --out-dir "$SPLIT_DIR" \
    --seed 0 \
    --train-frac 0.8 \
    --val-frac 0.1
fi

unset SAM3_SPME_FUSION SAM3_SPME_FUSION_MODE SAM3_SPME_FUSION_ALPHA SAM3_SPME_FUSION_ALPHA_OBJ SAM3_SPME_FUSION_DET_THR SAM3_SPME_FUSION_USE_PRESENCE SAM3_SPME_FUSION_QCOS_THR SAM3_SPME_FUSION_QCOS_TEMP SAM3_SPME_FUSION_QCOS_GATE SAM3_SPME_QUERY_POOL SAM3_SPME_QUERY_TOPK SAM3_SPME_ANCHOR_DET_THR SAM3_SPME_PER_OBJECT SAM3_SPME_PER_OBJECT_FALLBACK SAM3_SPME_KEEP_QUERIES
unset SAM3_SPME_WRITE_GATE SAM3_SPME_WRITE_GATE_MODE SAM3_SPME_DET_THR SAM3_SPME_QCOS_THR SAM3_SPME_USE_DET_SCORE SAM3_SPME_WRITE_GATE_APPLY SAM3_SPME_DEBUG_SIGNALS

echo "=== [OVIS] baseline (no fusion ckpt) | val ==="
if [ "$RUN_VAL" = "1" ]; then
  python -u scripts/phase0_eval_ovis_signals.py \
    --images-root "$IMAGES_ROOT" \
    --ann "$ANN" \
    --split-txt "$VAL_SPLIT" \
    --out-dir "$OUT_ROOT/val/baseline" \
    --base-sam3-pt "$SAM3_PT" \
    --bpe-path "$BPE" \
    --max-videos 0 \
    --max-instances "$MAX_INSTANCES" \
    --max-frames "$MAX_FRAMES" \
    --save-signals "$SAVE_SIGNALS" \
    "${EXTRA_SIGNALS_ARGS[@]}"
else
  echo "[OVIS] RUN_VAL=0 -> skip val baseline"
fi

export SAM3_SPME_FUSION=1
export SAM3_SPME_QUERY_POOL=top1
export SAM3_SPME_QUERY_TOPK=5
export SAM3_SPME_ANCHOR_DET_THR=0.0
export SAM3_SPME_FUSION_MODE="$FUSION_MODE"
export SAM3_SPME_FUSION_ALPHA="$FUSION_ALPHA"
export SAM3_SPME_FUSION_ALPHA_OBJ="$FUSION_ALPHA_OBJ"
export SAM3_SPME_FUSION_DET_THR="$FUSION_DET_THR"
export SAM3_SPME_FUSION_USE_PRESENCE="$FUSION_USE_PRESENCE"
export SAM3_SPME_FUSION_QCOS_THR="$FUSION_QCOS_THR"
export SAM3_SPME_FUSION_QCOS_TEMP="$FUSION_QCOS_TEMP"
export SAM3_SPME_FUSION_QCOS_GATE="$FUSION_QCOS_GATE"
export SAM3_SPME_PER_OBJECT=1
export SAM3_SPME_PER_OBJECT_FALLBACK=0
export SAM3_SPME_KEEP_QUERIES=1

echo "=== [OVIS] fusion ($FUSION_TAG) | val ==="
if [ "$RUN_VAL" = "1" ]; then
  python -u scripts/phase0_eval_ovis_signals.py \
    --images-root "$IMAGES_ROOT" \
    --ann "$ANN" \
    --split-txt "$VAL_SPLIT" \
    --out-dir "$OUT_ROOT/val/$FUSION_TAG" \
    --base-sam3-pt "$SAM3_PT" \
    --spme-fusion-ckpt "$FUSION_CKPT" \
    --bpe-path "$BPE" \
    --max-videos 0 \
    --max-instances "$MAX_INSTANCES" \
    --max-frames "$MAX_FRAMES" \
    --save-signals "$SAVE_SIGNALS" \
    "${EXTRA_SIGNALS_ARGS[@]}"
else
  echo "[OVIS] RUN_VAL=0 -> skip val fusion"
fi

unset SAM3_SPME_FUSION SAM3_SPME_FUSION_MODE SAM3_SPME_FUSION_ALPHA SAM3_SPME_FUSION_ALPHA_OBJ SAM3_SPME_FUSION_DET_THR SAM3_SPME_FUSION_USE_PRESENCE SAM3_SPME_FUSION_QCOS_THR SAM3_SPME_FUSION_QCOS_TEMP SAM3_SPME_FUSION_QCOS_GATE SAM3_SPME_QUERY_POOL SAM3_SPME_QUERY_TOPK SAM3_SPME_ANCHOR_DET_THR SAM3_SPME_PER_OBJECT SAM3_SPME_PER_OBJECT_FALLBACK SAM3_SPME_KEEP_QUERIES
unset SAM3_SPME_WRITE_GATE SAM3_SPME_WRITE_GATE_MODE SAM3_SPME_DET_THR SAM3_SPME_QCOS_THR SAM3_SPME_USE_DET_SCORE SAM3_SPME_WRITE_GATE_APPLY SAM3_SPME_DEBUG_SIGNALS

echo "=== [OVIS] baseline (no fusion ckpt) | test ==="
if [ "$RUN_TEST" = "1" ]; then
  python -u scripts/phase0_eval_ovis_signals.py \
    --images-root "$IMAGES_ROOT" \
    --ann "$ANN" \
    --split-txt "$TEST_SPLIT" \
    --out-dir "$OUT_ROOT/test/baseline" \
    --base-sam3-pt "$SAM3_PT" \
    --bpe-path "$BPE" \
    --max-videos 0 \
    --max-instances "$MAX_INSTANCES" \
    --max-frames "$MAX_FRAMES" \
    --save-signals "$SAVE_SIGNALS" \
    "${EXTRA_SIGNALS_ARGS[@]}"
else
  echo "[OVIS] RUN_TEST=0 -> skip test baseline"
fi

export SAM3_SPME_FUSION=1
export SAM3_SPME_QUERY_POOL=top1
export SAM3_SPME_QUERY_TOPK=5
export SAM3_SPME_ANCHOR_DET_THR=0.0
export SAM3_SPME_FUSION_MODE="$FUSION_MODE"
export SAM3_SPME_FUSION_ALPHA="$FUSION_ALPHA"
export SAM3_SPME_FUSION_ALPHA_OBJ="$FUSION_ALPHA_OBJ"
export SAM3_SPME_FUSION_DET_THR="$FUSION_DET_THR"
export SAM3_SPME_FUSION_USE_PRESENCE="$FUSION_USE_PRESENCE"
export SAM3_SPME_FUSION_QCOS_THR="$FUSION_QCOS_THR"
export SAM3_SPME_FUSION_QCOS_TEMP="$FUSION_QCOS_TEMP"
export SAM3_SPME_FUSION_QCOS_GATE="$FUSION_QCOS_GATE"
export SAM3_SPME_PER_OBJECT=1
export SAM3_SPME_PER_OBJECT_FALLBACK=0
export SAM3_SPME_KEEP_QUERIES=1

echo "=== [OVIS] fusion ($FUSION_TAG) | test ==="
if [ "$RUN_TEST" = "1" ]; then
  python -u scripts/phase0_eval_ovis_signals.py \
    --images-root "$IMAGES_ROOT" \
    --ann "$ANN" \
    --split-txt "$TEST_SPLIT" \
    --out-dir "$OUT_ROOT/test/$FUSION_TAG" \
    --base-sam3-pt "$SAM3_PT" \
    --spme-fusion-ckpt "$FUSION_CKPT" \
    --bpe-path "$BPE" \
    --max-videos 0 \
    --max-instances "$MAX_INSTANCES" \
    --max-frames "$MAX_FRAMES" \
    --save-signals "$SAVE_SIGNALS" \
    "${EXTRA_SIGNALS_ARGS[@]}"
else
  echo "[OVIS] RUN_TEST=0 -> skip test fusion"
fi

echo "=== Done. Results saved to: $OUT_ROOT ==="
