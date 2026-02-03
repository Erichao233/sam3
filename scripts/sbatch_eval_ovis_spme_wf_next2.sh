#!/bin/bash
#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -J sam3_ovis_spme_wf_next2
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

# (Optional) Pair list for ghost/FP debugging (each line: "<video_id> <ann_id>")
FP_PAIRS=${FP_PAIRS:-$REPO/tmp_top_regressed_fp.txt}

# Output root (download this folder after job finishes)
OUT_ROOT=${OUT_ROOT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/ovis_spme_wf_next2_v1}

# Full-test caps (0 = no limit)
MAX_INSTANCES_FULL=${MAX_INSTANCES_FULL:-0}
MAX_FRAMES_FULL=${MAX_FRAMES_FULL:-0}

# ONLY-PAIRS caps (0 = full from init)
MAX_FRAMES_PAIRS=${MAX_FRAMES_PAIRS:-0}

# ---- Fusion best config (from v2/v1 experiments) ----
F_TAG=${F_TAG:-film_a0p02_obj0_det0_q0p5}
F_MODE=${F_MODE:-film}
F_ALPHA=${F_ALPHA:-0.02}
F_ALPHA_OBJ=${F_ALPHA_OBJ:-0.0}
F_DET_THR=${F_DET_THR:-0.0}
F_USE_PRESENCE=${F_USE_PRESENCE:-1}
F_QCOS_THR=${F_QCOS_THR:-0.5}
F_QCOS_TEMP=${F_QCOS_TEMP:-20.0}
F_QCOS_GATE=${F_QCOS_GATE:-sigmoid}

# ---- Write-gate (SPME-W) sanity configs (mask_logits; avoid stale-memory blend) ----
W1_TAG=${W1_TAG:-spme_w_masklogits_hard_det0p5_q0p8}
W1_MODE=${W1_MODE:-hard}
W1_DET_THR=${W1_DET_THR:-0.5}
W1_QCOS_THR=${W1_QCOS_THR:-0.8}
W1_APPLY=${W1_APPLY:-mask_logits}

W2_TAG=${W2_TAG:-spme_w_masklogits_soft_det0p5_q0p7}
W2_MODE=${W2_MODE:-soft}
W2_DET_THR=${W2_DET_THR:-0.5}
W2_QCOS_THR=${W2_QCOS_THR:-0.7}
W2_APPLY=${W2_APPLY:-mask_logits}

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

unset SAM3_SPME_FUSION SAM3_SPME_FUSION_MODE SAM3_SPME_FUSION_ALPHA SAM3_SPME_FUSION_ALPHA_OBJ SAM3_SPME_FUSION_DET_THR SAM3_SPME_FUSION_USE_PRESENCE SAM3_SPME_FUSION_QCOS_THR SAM3_SPME_FUSION_QCOS_TEMP SAM3_SPME_FUSION_QCOS_GATE SAM3_SPME_FUSION_USE_WRITE_GATE
unset SAM3_SPME_QUERY_POOL SAM3_SPME_QUERY_TOPK SAM3_SPME_ANCHOR_DET_THR SAM3_SPME_PER_OBJECT SAM3_SPME_PER_OBJECT_FALLBACK SAM3_SPME_KEEP_QUERIES
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

echo "=== [OVIS] FULL test fusion only ($F_TAG) ==="
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

echo "=== [OVIS] FULL test W-only ($W1_TAG) ==="
unset SAM3_SPME_FUSION SAM3_SPME_FUSION_MODE SAM3_SPME_FUSION_ALPHA SAM3_SPME_FUSION_ALPHA_OBJ SAM3_SPME_FUSION_DET_THR SAM3_SPME_FUSION_USE_PRESENCE SAM3_SPME_FUSION_QCOS_THR SAM3_SPME_FUSION_QCOS_TEMP SAM3_SPME_FUSION_QCOS_GATE SAM3_SPME_FUSION_USE_WRITE_GATE

export SAM3_SPME_WRITE_GATE=1
export SAM3_SPME_WRITE_GATE_MODE="$W1_MODE"
export SAM3_SPME_DET_THR="$W1_DET_THR"
export SAM3_SPME_QCOS_THR="$W1_QCOS_THR"
export SAM3_SPME_USE_DET_SCORE=1
export SAM3_SPME_WRITE_GATE_APPLY="$W1_APPLY"
export SAM3_SPME_PER_OBJECT=1
export SAM3_SPME_PER_OBJECT_FALLBACK=0
export SAM3_SPME_KEEP_QUERIES=1

python -u scripts/phase0_eval_ovis_signals.py \
  --images-root "$IMAGES_ROOT" \
  --ann "$ANN" \
  --split-txt "$TEST_SPLIT" \
  --out-dir "$OUT_ROOT/test/full/$W1_TAG" \
  --base-sam3-pt "$SAM3_PT" \
  --bpe-path "$BPE" \
  --max-videos 0 \
  --max-instances "$MAX_INSTANCES_FULL" \
  --max-frames "$MAX_FRAMES_FULL" \
  --save-signals 0

echo "=== [OVIS] FULL test W1 + Fusion (coupled) ==="
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
export SAM3_SPME_FUSION_USE_WRITE_GATE=1

WF1_TAG=${WF1_TAG:-${F_TAG}__${W1_TAG}}
python -u scripts/phase0_eval_ovis_signals.py \
  --images-root "$IMAGES_ROOT" \
  --ann "$ANN" \
  --split-txt "$TEST_SPLIT" \
  --out-dir "$OUT_ROOT/test/full/$WF1_TAG" \
  --base-sam3-pt "$SAM3_PT" \
  --spme-fusion-ckpt "$FUSION_CKPT" \
  --bpe-path "$BPE" \
  --max-videos 0 \
  --max-instances "$MAX_INSTANCES_FULL" \
  --max-frames "$MAX_FRAMES_FULL" \
  --save-signals 0

if [ -s "$FP_PAIRS" ]; then
  echo "=== [OVIS] ONLY_PAIRS (FP regressed) baseline signals ==="
  unset SAM3_SPME_FUSION SAM3_SPME_FUSION_MODE SAM3_SPME_FUSION_ALPHA SAM3_SPME_FUSION_ALPHA_OBJ SAM3_SPME_FUSION_DET_THR SAM3_SPME_FUSION_USE_PRESENCE SAM3_SPME_FUSION_QCOS_THR SAM3_SPME_FUSION_QCOS_TEMP SAM3_SPME_FUSION_QCOS_GATE SAM3_SPME_FUSION_USE_WRITE_GATE
  unset SAM3_SPME_WRITE_GATE SAM3_SPME_WRITE_GATE_MODE SAM3_SPME_DET_THR SAM3_SPME_QCOS_THR SAM3_SPME_USE_DET_SCORE SAM3_SPME_WRITE_GATE_APPLY SAM3_SPME_DEBUG_SIGNALS
  python -u scripts/phase0_eval_ovis_signals.py \
    --images-root "$IMAGES_ROOT" \
    --ann "$ANN" \
    --split-txt "$TEST_SPLIT" \
    --out-dir "$OUT_ROOT/test/signals_fp/baseline" \
    --base-sam3-pt "$SAM3_PT" \
    --bpe-path "$BPE" \
    --max-videos 0 \
    --max-instances 0 \
    --max-frames "$MAX_FRAMES_PAIRS" \
    --save-signals 1 \
    --save-signals-for "$FP_PAIRS" \
    --only-pairs "$FP_PAIRS"

  echo "=== [OVIS] ONLY_PAIRS fusion signals ($F_TAG) ==="
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
  unset SAM3_SPME_WRITE_GATE SAM3_SPME_WRITE_GATE_MODE SAM3_SPME_DET_THR SAM3_SPME_QCOS_THR SAM3_SPME_USE_DET_SCORE SAM3_SPME_WRITE_GATE_APPLY SAM3_SPME_DEBUG_SIGNALS

  python -u scripts/phase0_eval_ovis_signals.py \
    --images-root "$IMAGES_ROOT" \
    --ann "$ANN" \
    --split-txt "$TEST_SPLIT" \
    --out-dir "$OUT_ROOT/test/signals_fp/$F_TAG" \
    --base-sam3-pt "$SAM3_PT" \
    --spme-fusion-ckpt "$FUSION_CKPT" \
    --bpe-path "$BPE" \
    --max-videos 0 \
    --max-instances 0 \
    --max-frames "$MAX_FRAMES_PAIRS" \
    --save-signals 1 \
    --save-signals-for "$FP_PAIRS" \
    --only-pairs "$FP_PAIRS"

  echo "=== [OVIS] ONLY_PAIRS W1 signals ($W1_TAG) ==="
  unset SAM3_SPME_FUSION SAM3_SPME_FUSION_MODE SAM3_SPME_FUSION_ALPHA SAM3_SPME_FUSION_ALPHA_OBJ SAM3_SPME_FUSION_DET_THR SAM3_SPME_FUSION_USE_PRESENCE SAM3_SPME_FUSION_QCOS_THR SAM3_SPME_FUSION_QCOS_TEMP SAM3_SPME_FUSION_QCOS_GATE SAM3_SPME_FUSION_USE_WRITE_GATE
  export SAM3_SPME_WRITE_GATE=1
  export SAM3_SPME_WRITE_GATE_MODE="$W1_MODE"
  export SAM3_SPME_DET_THR="$W1_DET_THR"
  export SAM3_SPME_QCOS_THR="$W1_QCOS_THR"
  export SAM3_SPME_USE_DET_SCORE=1
  export SAM3_SPME_WRITE_GATE_APPLY="$W1_APPLY"
  export SAM3_SPME_PER_OBJECT=1
  export SAM3_SPME_PER_OBJECT_FALLBACK=0
  export SAM3_SPME_KEEP_QUERIES=1

  python -u scripts/phase0_eval_ovis_signals.py \
    --images-root "$IMAGES_ROOT" \
    --ann "$ANN" \
    --split-txt "$TEST_SPLIT" \
    --out-dir "$OUT_ROOT/test/signals_fp/$W1_TAG" \
    --base-sam3-pt "$SAM3_PT" \
    --bpe-path "$BPE" \
    --max-videos 0 \
    --max-instances 0 \
    --max-frames "$MAX_FRAMES_PAIRS" \
    --save-signals 1 \
    --save-signals-for "$FP_PAIRS" \
    --only-pairs "$FP_PAIRS"

  echo "=== [OVIS] ONLY_PAIRS W2 signals ($W2_TAG) ==="
  export SAM3_SPME_WRITE_GATE=1
  export SAM3_SPME_WRITE_GATE_MODE="$W2_MODE"
  export SAM3_SPME_DET_THR="$W2_DET_THR"
  export SAM3_SPME_QCOS_THR="$W2_QCOS_THR"
  export SAM3_SPME_USE_DET_SCORE=1
  export SAM3_SPME_WRITE_GATE_APPLY="$W2_APPLY"
  export SAM3_SPME_PER_OBJECT=1
  export SAM3_SPME_PER_OBJECT_FALLBACK=0
  export SAM3_SPME_KEEP_QUERIES=1

  python -u scripts/phase0_eval_ovis_signals.py \
    --images-root "$IMAGES_ROOT" \
    --ann "$ANN" \
    --split-txt "$TEST_SPLIT" \
    --out-dir "$OUT_ROOT/test/signals_fp/$W2_TAG" \
    --base-sam3-pt "$SAM3_PT" \
    --bpe-path "$BPE" \
    --max-videos 0 \
    --max-instances 0 \
    --max-frames "$MAX_FRAMES_PAIRS" \
    --save-signals 1 \
    --save-signals-for "$FP_PAIRS" \
    --only-pairs "$FP_PAIRS"

  echo "=== [OVIS] ONLY_PAIRS W1 + Fusion signals ($WF1_TAG) ==="
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
  export SAM3_SPME_FUSION_USE_WRITE_GATE=1
  export SAM3_SPME_WRITE_GATE=1
  export SAM3_SPME_WRITE_GATE_MODE="$W1_MODE"
  export SAM3_SPME_DET_THR="$W1_DET_THR"
  export SAM3_SPME_QCOS_THR="$W1_QCOS_THR"
  export SAM3_SPME_USE_DET_SCORE=1
  export SAM3_SPME_WRITE_GATE_APPLY="$W1_APPLY"
  export SAM3_SPME_PER_OBJECT=1
  export SAM3_SPME_PER_OBJECT_FALLBACK=0
  export SAM3_SPME_KEEP_QUERIES=1

  python -u scripts/phase0_eval_ovis_signals.py \
    --images-root "$IMAGES_ROOT" \
    --ann "$ANN" \
    --split-txt "$TEST_SPLIT" \
    --out-dir "$OUT_ROOT/test/signals_fp/$WF1_TAG" \
    --base-sam3-pt "$SAM3_PT" \
    --spme-fusion-ckpt "$FUSION_CKPT" \
    --bpe-path "$BPE" \
    --max-videos 0 \
    --max-instances 0 \
    --max-frames "$MAX_FRAMES_PAIRS" \
    --save-signals 1 \
    --save-signals-for "$FP_PAIRS" \
    --only-pairs "$FP_PAIRS"
else
  echo "[OVIS] FP_PAIRS missing/empty; skipping ONLY_PAIRS signals. Path: $FP_PAIRS"
fi

echo "=== Done. Results saved to: $OUT_ROOT ==="
