#!/bin/bash
# Sweep inference-time SPME-Fusion knobs on OVIS val split.
#
# Goal: find a safer config than the default (reduce regressions / identity swaps / FP-absent)
# without retraining, by sweeping a small set of alphas / thresholds.
#
# Usage (server):
#   sbatch scripts/sbatch_eval_ovis_spme_fusion_sweep.sh
#
# Optional env overrides:
#   OVIS_ROOT=...
#   SPLIT_DIR=...
#   IMAGES_ROOT=...
#   OUT_ROOT=...
#   FUSION_CKPT=...
#   MAX_INSTANCES=500
#   MAX_FRAMES=128

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -J sam3_spme_fusion_ovis_sweep
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
OVIS=${OVIS_ROOT:-/home2020/home/icube/kunyuan/SurgBench/Ultrasound/OVIS}

SAM3_PT=$REPO/sam3.pt
BPE=$REPO/sam3/assets/bpe_simple_vocab_16e6.txt.gz

SPLIT_DIR=${SPLIT_DIR:-$OVIS/splits_seed0}
VAL_SPLIT=$SPLIT_DIR/val.txt

IMAGES_ROOT=${IMAGES_ROOT:-$OVIS/train}
ANN=$OVIS/annotations_train.json

FUSION_CKPT=${FUSION_CKPT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/spme_fusion_ovis_train_v1/main/checkpoints/spme_fusion_latest.pt}
OUT_ROOT=${OUT_ROOT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/spme_fusion_ovis_sweep_v1}

MAX_INSTANCES=${MAX_INSTANCES:-500}
MAX_FRAMES=${MAX_FRAMES:-128}

cd "$REPO"

mkdir -p "$SPLIT_DIR"
if [ ! -s "$VAL_SPLIT" ]; then
  echo "[OVIS] val split file not found; generating at $SPLIT_DIR"
  python scripts/ovis_make_splits.py \
    --ann "$ANN" \
    --out-dir "$SPLIT_DIR" \
    --seed 0 \
    --train-frac 0.8 \
    --val-frac 0.1
fi

unset SAM3_SPME_FUSION SAM3_SPME_FUSION_MODE SAM3_SPME_FUSION_ALPHA SAM3_SPME_FUSION_ALPHA_OBJ SAM3_SPME_FUSION_DET_THR SAM3_SPME_FUSION_USE_PRESENCE SAM3_SPME_FUSION_QCOS_THR SAM3_SPME_FUSION_QCOS_TEMP SAM3_SPME_FUSION_QCOS_GATE SAM3_SPME_QUERY_POOL SAM3_SPME_QUERY_TOPK SAM3_SPME_ANCHOR_DET_THR SAM3_SPME_PER_OBJECT SAM3_SPME_KEEP_QUERIES

echo "=== [OVIS sweep] baseline | val ==="
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
  --save-signals 0

export SAM3_SPME_FUSION=1
export SAM3_SPME_QUERY_POOL=top1
export SAM3_SPME_QUERY_TOPK=5
export SAM3_SPME_ANCHOR_DET_THR=0.0
export SAM3_SPME_FUSION_MODE=film
export SAM3_SPME_FUSION_USE_PRESENCE=1
export SAM3_SPME_FUSION_QCOS_TEMP=20.0
export SAM3_SPME_FUSION_QCOS_GATE=sigmoid
export SAM3_SPME_PER_OBJECT=1
export SAM3_SPME_KEEP_QUERIES=1

run_one () {
  local tag="$1"
  local alpha="$2"
  local alpha_obj="$3"
  local det_thr="$4"
  local qcos_thr="$5"

  export SAM3_SPME_FUSION_ALPHA="$alpha"
  export SAM3_SPME_FUSION_ALPHA_OBJ="$alpha_obj"
  export SAM3_SPME_FUSION_DET_THR="$det_thr"
  export SAM3_SPME_FUSION_QCOS_THR="$qcos_thr"

  echo "=== [OVIS sweep] $tag | val (alpha=$alpha obj=$alpha_obj det_thr=$det_thr qcos_thr=$qcos_thr) ==="
  python -u scripts/phase0_eval_ovis_signals.py \
    --images-root "$IMAGES_ROOT" \
    --ann "$ANN" \
    --split-txt "$VAL_SPLIT" \
    --out-dir "$OUT_ROOT/val/$tag" \
    --base-sam3-pt "$SAM3_PT" \
    --spme-fusion-ckpt "$FUSION_CKPT" \
    --bpe-path "$BPE" \
    --max-videos 0 \
    --max-instances "$MAX_INSTANCES" \
    --max-frames "$MAX_FRAMES" \
    --save-signals 0

  python -u scripts/analyze_ovis_eval.py \
    --baseline-dir "$OUT_ROOT/val/baseline" \
    --run-dir "$OUT_ROOT/val/$tag" \
    --metric mean_iou_track_present \
    --top-k 5 \
    --bootstrap 5000 \
    --seed 0
}

# Conservative variants (often reduce regressions)
run_one "film_a0p02_obj0_det0_q0p5"  0.02  0.0   0.0  0.5
run_one "film_a0p02_obj0_det0p2_q0p5" 0.02  0.0   0.2  0.5
run_one "film_a0p02_obj0_det0_q0p6"  0.02  0.0   0.0  0.6

# Baseline-ish (your current default) + a couple neighbors
run_one "film_a0p05_obj0_det0_q0p5"  0.05  0.0   0.0  0.5
run_one "film_a0p05_obj0p005_det0_q0p5" 0.05 0.005 0.0  0.5
run_one "film_a0p05_obj0p005_det0_q0p6" 0.05 0.005 0.0  0.6
run_one "film_a0p05_obj0p005_det0p2_q0p5" 0.05 0.005 0.2  0.5

echo "=== Done. Sweep results saved to: $OUT_ROOT ==="

