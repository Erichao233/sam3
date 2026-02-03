#!/bin/bash
#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -J sam3_ovis_spme_wf_next3
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
OUT_ROOT=${OUT_ROOT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/ovis_spme_wf_next3_v1}

# Full-test caps (0 = no limit)
MAX_INSTANCES_FULL=${MAX_INSTANCES_FULL:-0}
MAX_FRAMES_FULL=${MAX_FRAMES_FULL:-0}

# ONLY-PAIRS caps (0 = full from init)
MAX_FRAMES_PAIRS=${MAX_FRAMES_PAIRS:-0}

# ---- Fusion (FiLM) base config ----
F_MODE=${F_MODE:-film}
F_ALPHA=${F_ALPHA:-0.02}
F_ALPHA_OBJ=${F_ALPHA_OBJ:-0.0}
F_USE_PRESENCE=${F_USE_PRESENCE:-1}
F_QCOS_THR=${F_QCOS_THR:-0.5}
F_QCOS_TEMP=${F_QCOS_TEMP:-20.0}
F_QCOS_GATE=${F_QCOS_GATE:-sigmoid}

# Det-threshold sweep (main hypothesis: suppress noisy edits to reduce GT-absent FP regressions)
F0_DET_THR=${F0_DET_THR:-0.0}
F3_DET_THR=${F3_DET_THR:-0.3}
F5_DET_THR=${F5_DET_THR:-0.5}

F0_TAG=${F0_TAG:-film_a0p02_obj0_det0_q0p5}
F3_TAG=${F3_TAG:-film_a0p02_obj0_det0p3_q0p5}
F5_TAG=${F5_TAG:-film_a0p02_obj0_det0p5_q0p5}

# ---- Write-gate (SPME-W) sanity sweep (blend_mem + soft, low qcos thr) ----
# Note: `mask_logits` mode was found to collapse tracking on OVIS (writes empty/no-object memory).
W1_TAG=${W1_TAG:-spme_w_blendmem_soft_det0_q0p5}
W1_MODE=${W1_MODE:-soft}
W1_DET_THR=${W1_DET_THR:-0.0}
W1_QCOS_THR=${W1_QCOS_THR:-0.5}
W1_APPLY=${W1_APPLY:-blend_mem}

W2_TAG=${W2_TAG:-spme_w_blendmem_soft_det0_q0p6}
W2_MODE=${W2_MODE:-soft}
W2_DET_THR=${W2_DET_THR:-0.0}
W2_QCOS_THR=${W2_QCOS_THR:-0.6}
W2_APPLY=${W2_APPLY:-blend_mem}

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

reset_spme_env() {
  unset SAM3_SPME_FUSION SAM3_SPME_FUSION_MODE SAM3_SPME_FUSION_ALPHA SAM3_SPME_FUSION_ALPHA_OBJ \
    SAM3_SPME_FUSION_DET_THR SAM3_SPME_FUSION_USE_PRESENCE SAM3_SPME_FUSION_QCOS_THR \
    SAM3_SPME_FUSION_QCOS_TEMP SAM3_SPME_FUSION_QCOS_GATE SAM3_SPME_FUSION_USE_WRITE_GATE
  unset SAM3_SPME_QUERY_POOL SAM3_SPME_QUERY_TOPK SAM3_SPME_QUERY_POOL_TEMP SAM3_SPME_ANCHOR_DET_THR \
    SAM3_SPME_PER_OBJECT SAM3_SPME_PER_OBJECT_FALLBACK SAM3_SPME_KEEP_QUERIES
  unset SAM3_SPME_WRITE_GATE SAM3_SPME_WRITE_GATE_MODE SAM3_SPME_DET_THR SAM3_SPME_QCOS_THR \
    SAM3_SPME_USE_DET_SCORE SAM3_SPME_WRITE_GATE_APPLY SAM3_SPME_DEBUG_SIGNALS
}

run_full_eval() {
  local out_dir="$1"
  shift
  python -u scripts/phase0_eval_ovis_signals.py \
    --images-root "$IMAGES_ROOT" \
    --ann "$ANN" \
    --split-txt "$TEST_SPLIT" \
    --out-dir "$out_dir" \
    --base-sam3-pt "$SAM3_PT" \
    --bpe-path "$BPE" \
    --max-videos 0 \
    --max-instances "$MAX_INSTANCES_FULL" \
    --max-frames "$MAX_FRAMES_FULL" \
    --save-signals 0 \
    "$@"
}

run_pairs_eval() {
  local out_dir="$1"
  shift
  python -u scripts/phase0_eval_ovis_signals.py \
    --images-root "$IMAGES_ROOT" \
    --ann "$ANN" \
    --split-txt "$TEST_SPLIT" \
    --out-dir "$out_dir" \
    --base-sam3-pt "$SAM3_PT" \
    --bpe-path "$BPE" \
    --max-videos 0 \
    --max-instances 0 \
    --max-frames "$MAX_FRAMES_PAIRS" \
    --save-signals 1 \
    --save-signals-for "$FP_PAIRS" \
    --only-pairs "$FP_PAIRS" \
    "$@"
}

echo "=== [OVIS] FULL test baseline ==="
reset_spme_env
run_full_eval "$OUT_ROOT/test/full/baseline"

run_fusion_full() {
  local tag="$1"
  local det_thr="$2"
  echo "=== [OVIS] FULL test fusion ($tag) det_thr=$det_thr ==="
  reset_spme_env
  export SAM3_SPME_FUSION=1
  export SAM3_SPME_QUERY_POOL=top1
  export SAM3_SPME_QUERY_TOPK=5
  export SAM3_SPME_ANCHOR_DET_THR=0.0
  export SAM3_SPME_FUSION_MODE="$F_MODE"
  export SAM3_SPME_FUSION_ALPHA="$F_ALPHA"
  export SAM3_SPME_FUSION_ALPHA_OBJ="$F_ALPHA_OBJ"
  export SAM3_SPME_FUSION_DET_THR="$det_thr"
  export SAM3_SPME_FUSION_USE_PRESENCE="$F_USE_PRESENCE"
  export SAM3_SPME_FUSION_QCOS_THR="$F_QCOS_THR"
  export SAM3_SPME_FUSION_QCOS_TEMP="$F_QCOS_TEMP"
  export SAM3_SPME_FUSION_QCOS_GATE="$F_QCOS_GATE"
  export SAM3_SPME_PER_OBJECT=1
  export SAM3_SPME_PER_OBJECT_FALLBACK=0
  export SAM3_SPME_KEEP_QUERIES=1

  run_full_eval "$OUT_ROOT/test/full/$tag" --spme-fusion-ckpt "$FUSION_CKPT"
}

run_fusion_full "$F0_TAG" "$F0_DET_THR"
run_fusion_full "$F3_TAG" "$F3_DET_THR"
run_fusion_full "$F5_TAG" "$F5_DET_THR"

if [ -s "$FP_PAIRS" ]; then
  echo "=== [OVIS] ONLY_PAIRS (FP regressed) baseline signals ==="
  reset_spme_env
  run_pairs_eval "$OUT_ROOT/test/signals_fp/baseline"

  run_fusion_pairs() {
    local tag="$1"
    local det_thr="$2"
    echo "=== [OVIS] ONLY_PAIRS fusion signals ($tag) det_thr=$det_thr ==="
    reset_spme_env
    export SAM3_SPME_FUSION=1
    export SAM3_SPME_QUERY_POOL=top1
    export SAM3_SPME_QUERY_TOPK=5
    export SAM3_SPME_ANCHOR_DET_THR=0.0
    export SAM3_SPME_FUSION_MODE="$F_MODE"
    export SAM3_SPME_FUSION_ALPHA="$F_ALPHA"
    export SAM3_SPME_FUSION_ALPHA_OBJ="$F_ALPHA_OBJ"
    export SAM3_SPME_FUSION_DET_THR="$det_thr"
    export SAM3_SPME_FUSION_USE_PRESENCE="$F_USE_PRESENCE"
    export SAM3_SPME_FUSION_QCOS_THR="$F_QCOS_THR"
    export SAM3_SPME_FUSION_QCOS_TEMP="$F_QCOS_TEMP"
    export SAM3_SPME_FUSION_QCOS_GATE="$F_QCOS_GATE"
    export SAM3_SPME_PER_OBJECT=1
    export SAM3_SPME_PER_OBJECT_FALLBACK=0
    export SAM3_SPME_KEEP_QUERIES=1
    unset SAM3_SPME_WRITE_GATE SAM3_SPME_WRITE_GATE_MODE SAM3_SPME_DET_THR SAM3_SPME_QCOS_THR \
      SAM3_SPME_USE_DET_SCORE SAM3_SPME_WRITE_GATE_APPLY SAM3_SPME_DEBUG_SIGNALS

    run_pairs_eval "$OUT_ROOT/test/signals_fp/$tag" --spme-fusion-ckpt "$FUSION_CKPT"
  }

  run_fusion_pairs "$F0_TAG" "$F0_DET_THR"
  run_fusion_pairs "$F3_TAG" "$F3_DET_THR"
  run_fusion_pairs "$F5_TAG" "$F5_DET_THR"

  run_w_pairs() {
    local tag="$1"
    local mode="$2"
    local det_thr="$3"
    local q_thr="$4"
    local apply="$5"
    echo "=== [OVIS] ONLY_PAIRS W signals ($tag) mode=$mode det_thr=$det_thr qcos_thr=$q_thr apply=$apply ==="
    reset_spme_env
    export SAM3_SPME_WRITE_GATE=1
    export SAM3_SPME_WRITE_GATE_MODE="$mode"
    export SAM3_SPME_DET_THR="$det_thr"
    export SAM3_SPME_QCOS_THR="$q_thr"
    export SAM3_SPME_USE_DET_SCORE=1
    export SAM3_SPME_WRITE_GATE_APPLY="$apply"
    export SAM3_SPME_PER_OBJECT=1
    export SAM3_SPME_PER_OBJECT_FALLBACK=0
    export SAM3_SPME_KEEP_QUERIES=1
    run_pairs_eval "$OUT_ROOT/test/signals_fp/$tag"
  }

  run_w_pairs "$W1_TAG" "$W1_MODE" "$W1_DET_THR" "$W1_QCOS_THR" "$W1_APPLY"
  run_w_pairs "$W2_TAG" "$W2_MODE" "$W2_DET_THR" "$W2_QCOS_THR" "$W2_APPLY"
else
  echo "[OVIS] FP_PAIRS missing/empty; skipping ONLY_PAIRS signals. Path: $FP_PAIRS"
fi

echo "=== Done. Results saved to: $OUT_ROOT ==="
