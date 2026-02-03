#!/bin/bash
#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_spme_gate_ovis_train
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.err

set -euo pipefail

source /home2020/home/icube/kunyuan/anaconda3/etc/profile.d/conda.sh
conda activate sam3

export OMP_NUM_THREADS=8
export HYDRA_FULL_ERROR=1
export SAM3_DISABLE_TRITON=1
export PYTHONUNBUFFERED=1

# Optional: gate MLP width (read by sam3/model/sam3_video_base.py at model construction time).
export SAM3_SPME_GATE_HIDDEN=${SAM3_SPME_GATE_HIDDEN:-64}

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
TRAIN_SPLIT=$SPLIT_DIR/train.txt

IMAGES_ROOT=${IMAGES_ROOT:-$OVIS/train}
ANN=$OVIS/annotations_train.json

OUT_DIR=${OUT_DIR:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/spme_gate_ovis_train_v1/main}
GATE_INPUTS="${GATE_INPUTS:-full}"          # full | det3 | det4
FUSION_HEAD="${FUSION_HEAD:-0}"            # 0 | 1 (decouple fusion strength)
DET_PRESENT_THR="${DET_PRESENT_THR:-0.3}"  # only relevant for det4

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

python -u scripts/train_spme_gate.py \
  --images-root "$IMAGES_ROOT" \
  --ann "$ANN" \
  --split-txt "$TRAIN_SPLIT" \
  --out-dir "$OUT_DIR" \
  --base-sam3-pt "$SAM3_PT" \
  --bpe-path "$BPE" \
  --seed 0 \
  --gate-inputs "$GATE_INPUTS" \
  --fusion-head "$FUSION_HEAD" \
  --det-present-thr "$DET_PRESENT_THR" \
  --query-pool top1 \
  --query-topk 5 \
  --anchor-det-thr 0.0 \
  --pointer-mode hybrid \
  --match-iou-thr 0.1 \
  --match-topk 20 \
  --score-thr-detection 0.05 \
  --clip-len 3 \
  --gate-frame 1 \
  --supervise-frame 2 \
  --use-decay 1 \
  --occ-norm 10.0 \
  --fusion-mode film \
  --fusion-alpha 0.05 \
  --fusion-alpha-obj 0.005 \
  --fusion-det-thr 0.0 \
  --absent-weight 0.25 \
  --max-steps 20000 \
  --lr 5e-4 \
  --weight-decay 0.0 \
  --log-every 50 \
  --save-every 2000

echo "Done. Checkpoints: $OUT_DIR/checkpoints/"
