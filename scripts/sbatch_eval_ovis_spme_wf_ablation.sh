#!/bin/bash
#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -J sam3_ovis_spme_wf_ablate
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
TEST_SPLIT=$SPLIT_DIR/test.txt

IMAGES_ROOT=${IMAGES_ROOT:-$OVIS/train}
ANN=$OVIS/annotations_train.json

# Trained SPME-Fusion module checkpoint (contains only spme_state_dict)
FUSION_CKPT=${FUSION_CKPT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/spme_fusion_ovis_train_v1/main/checkpoints/spme_fusion_latest.pt}

# Output root (download this folder after job finishes)
OUT_ROOT=${OUT_ROOT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/ovis_spme_wf_ablation_v1}

# Full-test caps (0 = no limit)
MAX_INSTANCES_FULL=${MAX_INSTANCES_FULL:-0}
MAX_FRAMES_FULL=${MAX_FRAMES_FULL:-0}

# ---- Config A: Fusion only (best so far) ----
F_TAG=${F_TAG:-film_a0p02_obj0_det0_q0p5}
F_MODE=${F_MODE:-film}
F_ALPHA=${F_ALPHA:-0.02}
F_ALPHA_OBJ=${F_ALPHA_OBJ:-0.0}
F_DET_THR=${F_DET_THR:-0.0}
F_USE_PRESENCE=${F_USE_PRESENCE:-1}
F_QCOS_THR=${F_QCOS_THR:-0.5}
F_QCOS_TEMP=${F_QCOS_TEMP:-20.0}
F_QCOS_GATE=${F_QCOS_GATE:-sigmoid}

# ---- Config B: Write-gate only (SPME-W) ----
# Start conservative: block memory writes unless qcos is very high.
W_TAG=${W_TAG:-spme_w_hard_q0p8_blendmem}
W_MODE=${W_MODE:-hard}
W_QCOS_THR=${W_QCOS_THR:-0.8}
W_DET_THR=${W_DET_THR:-0.0}
W_USE_DET_SCORE=${W_USE_DET_SCORE:-1}
W_APPLY=${W_APPLY:-blend_mem}

cd "$REPO"

mkdir -p "$SPLIT_DIR"
if [ ! -s "$TEST_SPLIT" ]; then
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

echo "=== [OVIS] FULL test baseline ==="
python -u scripts/phase0_eval_ovis_signals.py \
  --images-root "$IMAGES_ROOT" \
  --ann "$ANN" \
  --split-txt "$TEST_SPLIT" \
  --out-dir "$OUT_ROOT/test/full/baseline" \
  --base-sam3-pt "$SAM3_PT" \
  --bpe-path "$BPE" \
  --max-videos 0 \
  --max-instances "$MAX_INSTANCES_FULL" \
  --max-frames "$MAX_FRAMES_FULL" \
  --save-signals 0

echo "=== [OVIS] FULL test write-gate only ($W_TAG) ==="
export SAM3_SPME_WRITE_GATE=1
export SAM3_SPME_WRITE_GATE_MODE="$W_MODE"
export SAM3_SPME_QCOS_THR="$W_QCOS_THR"
export SAM3_SPME_DET_THR="$W_DET_THR"
export SAM3_SPME_USE_DET_SCORE="$W_USE_DET_SCORE"
export SAM3_SPME_WRITE_GATE_APPLY="$W_APPLY"

# Enable per-object context so W is per-object on multi-instance videos.
export SAM3_SPME_PER_OBJECT=1
export SAM3_SPME_PER_OBJECT_FALLBACK=0
export SAM3_SPME_KEEP_QUERIES=1

python -u scripts/phase0_eval_ovis_signals.py \
  --images-root "$IMAGES_ROOT" \
  --ann "$ANN" \
  --split-txt "$TEST_SPLIT" \
  --out-dir "$OUT_ROOT/test/full/$W_TAG" \
  --base-sam3-pt "$SAM3_PT" \
  --bpe-path "$BPE" \
  --max-videos 0 \
  --max-instances "$MAX_INSTANCES_FULL" \
  --max-frames "$MAX_FRAMES_FULL" \
  --save-signals 0

echo "=== [OVIS] FULL test fusion only ($F_TAG) ==="
unset SAM3_SPME_WRITE_GATE SAM3_SPME_WRITE_GATE_MODE SAM3_SPME_DET_THR SAM3_SPME_QCOS_THR SAM3_SPME_USE_DET_SCORE SAM3_SPME_WRITE_GATE_APPLY SAM3_SPME_DEBUG_SIGNALS

export SAM3_SPME_FUSION=1
export SAM3_SPME_QUERY_POOL=top1
export SAM3_SPME_QUERY_TOPK=5
export SAM3_SPME_ANCHOR_DET_THR=0.0
export SAM3_SPME_FUSION_MODE="$F_MODE"
export SAM3_SPME_FUSION_ALPHA="$F_ALPHA"
export SAM3_SPME_FUSION_ALPHA_OBJ="$F_ALPHA_OBJ"
export SAM3_SPME_FUSION_DET_THR="$F_DET_THR"
export SAM3_SPME_FUSION_USE_PRESENCE="$F_USE_PRESENCE"
export SAM3_SPME_FUSION_QCOS_THR="$F_QCOS_THR"
export SAM3_SPME_FUSION_QCOS_TEMP="$F_QCOS_TEMP"
export SAM3_SPME_FUSION_QCOS_GATE="$F_QCOS_GATE"
export SAM3_SPME_PER_OBJECT=1
export SAM3_SPME_PER_OBJECT_FALLBACK=0
export SAM3_SPME_KEEP_QUERIES=1

python -u scripts/phase0_eval_ovis_signals.py \
  --images-root "$IMAGES_ROOT" \
  --ann "$ANN" \
  --split-txt "$TEST_SPLIT" \
  --out-dir "$OUT_ROOT/test/full/$F_TAG" \
  --base-sam3-pt "$SAM3_PT" \
  --spme-fusion-ckpt "$FUSION_CKPT" \
  --bpe-path "$BPE" \
  --max-videos 0 \
  --max-instances "$MAX_INSTANCES_FULL" \
  --max-frames "$MAX_FRAMES_FULL" \
  --save-signals 0

echo "=== [OVIS] FULL test fusion + write-gate (W+F) ==="
export SAM3_SPME_WRITE_GATE=1
export SAM3_SPME_WRITE_GATE_MODE="$W_MODE"
export SAM3_SPME_QCOS_THR="$W_QCOS_THR"
export SAM3_SPME_DET_THR="$W_DET_THR"
export SAM3_SPME_USE_DET_SCORE="$W_USE_DET_SCORE"
export SAM3_SPME_WRITE_GATE_APPLY="$W_APPLY"

WF_TAG=${WF_TAG:-${F_TAG}__${W_TAG}}
python -u scripts/phase0_eval_ovis_signals.py \
  --images-root "$IMAGES_ROOT" \
  --ann "$ANN" \
  --split-txt "$TEST_SPLIT" \
  --out-dir "$OUT_ROOT/test/full/$WF_TAG" \
  --base-sam3-pt "$SAM3_PT" \
  --spme-fusion-ckpt "$FUSION_CKPT" \
  --bpe-path "$BPE" \
  --max-videos 0 \
  --max-instances "$MAX_INSTANCES_FULL" \
  --max-frames "$MAX_FRAMES_FULL" \
  --save-signals 0

echo "=== Done. Results saved to: $OUT_ROOT ==="
