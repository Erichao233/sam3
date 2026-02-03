#!/bin/bash
#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_spme_gate_ovis_eval
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
TEST_SPLIT=$SPLIT_DIR/test.txt

IMAGES_ROOT=${IMAGES_ROOT:-$OVIS/train}
ANN=$OVIS/annotations_train.json

# Your trained gate+fusion checkpoint (contains only spme_state_dict).
GATE_CKPT=${GATE_CKPT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/spme_gate_ovis_train_v1/main/checkpoints/spme_gate_step18000.pt}

# Optional detector fine-tune checkpoint (trainer-style ckpt with ['model']).
FINETUNE_CKPT=${FINETUNE_CKPT:-}

OUT_ROOT=${OUT_ROOT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/spme_gate_ovis_eval_v1/job_${SLURM_JOB_ID:-local}}

# 0 = no limit (full eval). Override via env if you want a quick smoke test.
MAX_VIDEOS=${MAX_VIDEOS:-0}
MAX_INSTANCES=${MAX_INSTANCES:-0}
MAX_FRAMES=${MAX_FRAMES:-0}
TAU=${TAU:-0.5}

RUN_VAL=${RUN_VAL:-0}
RUN_TEST=${RUN_TEST:-1}

# SPME knobs (override via env)
QUERY_POOL=${QUERY_POOL:-top1}
QUERY_TOPK=${QUERY_TOPK:-5}
ANCHOR_DET_THR=${ANCHOR_DET_THR:-0.0}

FUSION_MODE=${FUSION_MODE:-film}
FUSION_ALPHA=${FUSION_ALPHA:-0.05}
FUSION_ALPHA_OBJ=${FUSION_ALPHA_OBJ:-0.005}
FUSION_DET_THR=${FUSION_DET_THR:-0.0}
FUSION_USE_PRESENCE=${FUSION_USE_PRESENCE:-1}
FUSION_QCOS_THR=${FUSION_QCOS_THR:-0.5}
FUSION_QCOS_TEMP=${FUSION_QCOS_TEMP:-20.0}
FUSION_QCOS_GATE=${FUSION_QCOS_GATE:-sigmoid}

GATE_USE_DECAY=${GATE_USE_DECAY:-1}
GATE_OCC_NORM=${GATE_OCC_NORM:-10.0}
GATE_HIDDEN=${SAM3_SPME_GATE_HIDDEN:-64}

ANALYZE=${ANALYZE:-1}
BOOTSTRAP=${BOOTSTRAP:-20000}
TOPK=${TOPK:-50}

AUTO_SAVE_SIGNALS=${AUTO_SAVE_SIGNALS:-0}
MAX_FRAMES_PAIRS=${MAX_FRAMES_PAIRS:-0}

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

COMMON_ARGS=(
  --images-root "$IMAGES_ROOT"
  --ann "$ANN"
  --base-sam3-pt "$SAM3_PT"
  --bpe-path "$BPE"
  --max-videos "$MAX_VIDEOS"
  --max-instances "$MAX_INSTANCES"
  --tau "$TAU"
)
if [ -n "$FINETUNE_CKPT" ]; then
  COMMON_ARGS+=(--finetune-ckpt "$FINETUNE_CKPT")
fi

unset_all_spme() {
  unset SAM3_SPME_FUSION SAM3_SPME_FUSION_MODE SAM3_SPME_FUSION_ALPHA SAM3_SPME_FUSION_ALPHA_OBJ SAM3_SPME_FUSION_DET_THR SAM3_SPME_FUSION_USE_PRESENCE SAM3_SPME_FUSION_QCOS_THR SAM3_SPME_FUSION_QCOS_TEMP SAM3_SPME_FUSION_QCOS_GATE SAM3_SPME_FUSION_USE_WRITE_GATE
  unset SAM3_SPME_QUERY_POOL SAM3_SPME_QUERY_TOPK SAM3_SPME_ANCHOR_DET_THR SAM3_SPME_PER_OBJECT SAM3_SPME_PER_OBJECT_FALLBACK SAM3_SPME_KEEP_QUERIES
  unset SAM3_SPME_WRITE_GATE SAM3_SPME_WRITE_GATE_MODE SAM3_SPME_DET_THR SAM3_SPME_QCOS_THR SAM3_SPME_USE_DET_SCORE SAM3_SPME_WRITE_GATE_APPLY SAM3_SPME_DEBUG_SIGNALS
  unset SAM3_SPME_LEARNED_GATE SAM3_SPME_LEARNED_GATE_USE_DECAY SAM3_SPME_LEARNED_GATE_OCCLUDED_NORM SAM3_SPME_LEARNED_GATE_LOG SAM3_SPME_GATE_HIDDEN
}

set_common_spme_signal_env() {
  export SAM3_SPME_QUERY_POOL="$QUERY_POOL"
  export SAM3_SPME_QUERY_TOPK="$QUERY_TOPK"
  export SAM3_SPME_ANCHOR_DET_THR="$ANCHOR_DET_THR"
  export SAM3_SPME_PER_OBJECT=1
  export SAM3_SPME_PER_OBJECT_FALLBACK=0
  export SAM3_SPME_KEEP_QUERIES=1
}

set_fusion_env() {
  export SAM3_SPME_FUSION=1
  export SAM3_SPME_FUSION_MODE="$FUSION_MODE"
  export SAM3_SPME_FUSION_ALPHA="$FUSION_ALPHA"
  export SAM3_SPME_FUSION_ALPHA_OBJ="$FUSION_ALPHA_OBJ"
  export SAM3_SPME_FUSION_DET_THR="$FUSION_DET_THR"
  export SAM3_SPME_FUSION_USE_PRESENCE="$FUSION_USE_PRESENCE"
  export SAM3_SPME_FUSION_QCOS_THR="$FUSION_QCOS_THR"
  export SAM3_SPME_FUSION_QCOS_TEMP="$FUSION_QCOS_TEMP"
  export SAM3_SPME_FUSION_QCOS_GATE="$FUSION_QCOS_GATE"
}

set_gate_env() {
  export SAM3_SPME_LEARNED_GATE=1
  export SAM3_SPME_LEARNED_GATE_USE_DECAY="$GATE_USE_DECAY"
  export SAM3_SPME_LEARNED_GATE_OCCLUDED_NORM="$GATE_OCC_NORM"
  export SAM3_SPME_GATE_HIDDEN="$GATE_HIDDEN"
}

eval_one_split() {
  local split_name="$1"
  local split_txt="$2"

  echo "=== [OVIS][$split_name] baseline ==="
  unset_all_spme
  python -u scripts/phase0_eval_ovis_signals.py \
    --split-txt "$split_txt" \
    --out-dir "$OUT_ROOT/$split_name/baseline" \
    --max-frames "$MAX_FRAMES" \
    "${COMMON_ARGS[@]}"

  echo "=== [OVIS][$split_name] fusion_only (load spme ckpt; gate disabled) ==="
  unset_all_spme
  set_common_spme_signal_env
  set_fusion_env
  export SAM3_SPME_LEARNED_GATE=0
  python -u scripts/phase0_eval_ovis_signals.py \
    --split-txt "$split_txt" \
    --out-dir "$OUT_ROOT/$split_name/fusion_only" \
    --max-frames "$MAX_FRAMES" \
    --spme-fusion-ckpt "$GATE_CKPT" \
    "${COMMON_ARGS[@]}"

  echo "=== [OVIS][$split_name] gate_only (fusion alpha=0; gate enabled) ==="
  unset_all_spme
  set_common_spme_signal_env
  # Keep SPME_FUSION=1 so detector attaches spme signals, but disable injection.
  export SAM3_SPME_FUSION=1
  export SAM3_SPME_FUSION_ALPHA=0.0
  export SAM3_SPME_FUSION_ALPHA_OBJ=0.0
  set_gate_env
  python -u scripts/phase0_eval_ovis_signals.py \
    --split-txt "$split_txt" \
    --out-dir "$OUT_ROOT/$split_name/gate_only" \
    --max-frames "$MAX_FRAMES" \
    --spme-fusion-ckpt "$GATE_CKPT" \
    "${COMMON_ARGS[@]}"

  echo "=== [OVIS][$split_name] fusion_gate (fusion + gate enabled) ==="
  unset_all_spme
  set_common_spme_signal_env
  set_fusion_env
  set_gate_env
  python -u scripts/phase0_eval_ovis_signals.py \
    --split-txt "$split_txt" \
    --out-dir "$OUT_ROOT/$split_name/fusion_gate" \
    --max-frames "$MAX_FRAMES" \
    --spme-fusion-ckpt "$GATE_CKPT" \
    "${COMMON_ARGS[@]}"

  if [ "$ANALYZE" = "1" ]; then
    echo "=== [OVIS][$split_name] analyze vs baseline ==="
    ANALYSIS_DIR="$OUT_ROOT/$split_name/analysis"
    mkdir -p "$ANALYSIS_DIR"

    python -u scripts/analyze_ovis_eval.py \
      --baseline-dir "$OUT_ROOT/$split_name/baseline" \
      --run-dir "$OUT_ROOT/$split_name/fusion_only" \
      --run-dir "$OUT_ROOT/$split_name/gate_only" \
      --run-dir "$OUT_ROOT/$split_name/fusion_gate" \
      --metric mean_iou_track_present \
      --subset all \
      --top-k "$TOPK" \
      --bootstrap "$BOOTSTRAP" \
      --seed 0 \
      --write-signal-lists "$ANALYSIS_DIR/pairs_mean_iou_track_present" \
      > "$ANALYSIS_DIR/compare_mean_iou_track_present.txt"

    python -u scripts/analyze_ovis_eval.py \
      --baseline-dir "$OUT_ROOT/$split_name/baseline" \
      --run-dir "$OUT_ROOT/$split_name/fusion_only" \
      --run-dir "$OUT_ROOT/$split_name/gate_only" \
      --run-dir "$OUT_ROOT/$split_name/fusion_gate" \
      --metric fp_absent_rate_track \
      --subset baseline_gt_absent_gt0 \
      --top-k "$TOPK" \
      --bootstrap "$BOOTSTRAP" \
      --seed 0 \
      --write-signal-lists "$ANALYSIS_DIR/pairs_fp_absent_rate_track" \
      > "$ANALYSIS_DIR/compare_fp_absent_rate_track.txt"

    python -u scripts/analyze_ovis_eval.py \
      --baseline-dir "$OUT_ROOT/$split_name/baseline" \
      --run-dir "$OUT_ROOT/$split_name/fusion_only" \
      --run-dir "$OUT_ROOT/$split_name/gate_only" \
      --run-dir "$OUT_ROOT/$split_name/fusion_gate" \
      --metric identity_swap_frac_tau \
      --subset baseline_swap_gt0 \
      --top-k "$TOPK" \
      --bootstrap "$BOOTSTRAP" \
      --seed 0 \
      > "$ANALYSIS_DIR/compare_identity_swap_frac_tau.txt"

    python -u scripts/analyze_ovis_eval.py \
      --baseline-dir "$OUT_ROOT/$split_name/baseline" \
      --run-dir "$OUT_ROOT/$split_name/fusion_only" \
      --run-dir "$OUT_ROOT/$split_name/gate_only" \
      --run-dir "$OUT_ROOT/$split_name/fusion_gate" \
      --metric fail_frac_tau \
      --subset baseline_fail_gt0 \
      --top-k "$TOPK" \
      --bootstrap "$BOOTSTRAP" \
      --seed 0 \
      > "$ANALYSIS_DIR/compare_fail_frac_tau.txt"

    if [ "$AUTO_SAVE_SIGNALS" = "1" ]; then
      echo "=== [OVIS][$split_name] save signals for top regressions (mean_iou_track_present) ==="
      PAIRS_REG="$ANALYSIS_DIR/pairs_mean_iou_track_present/top_regressed_pairs.txt"
      if [ -s "$PAIRS_REG" ]; then
        # Baseline signals
        unset_all_spme
        python -u scripts/phase0_eval_ovis_signals.py \
          --split-txt "$split_txt" \
          --out-dir "$OUT_ROOT/$split_name/signals_iou_regressed/baseline" \
          --only-pairs "$PAIRS_REG" \
          --save-signals 1 \
          --max-frames "$MAX_FRAMES_PAIRS" \
          "${COMMON_ARGS[@]}"

        # Fusion+Gate signals
        unset_all_spme
        set_common_spme_signal_env
        set_fusion_env
        set_gate_env
        python -u scripts/phase0_eval_ovis_signals.py \
          --split-txt "$split_txt" \
          --out-dir "$OUT_ROOT/$split_name/signals_iou_regressed/fusion_gate" \
          --only-pairs "$PAIRS_REG" \
          --save-signals 1 \
          --max-frames "$MAX_FRAMES_PAIRS" \
          --spme-fusion-ckpt "$GATE_CKPT" \
          "${COMMON_ARGS[@]}"
      else
        echo "[warn] missing pairs list: $PAIRS_REG"
      fi
    fi
  fi
}

mkdir -p "$OUT_ROOT"
(echo "job_id=${SLURM_JOB_ID:-none}"; echo "gate_ckpt=$GATE_CKPT") > "$OUT_ROOT/README_run.txt"

if [ "$RUN_VAL" = "1" ]; then
  eval_one_split "val" "$VAL_SPLIT"
else
  echo "[OVIS] RUN_VAL=0 -> skip val"
fi

if [ "$RUN_TEST" = "1" ]; then
  eval_one_split "test" "$TEST_SPLIT"
else
  echo "[OVIS] RUN_TEST=0 -> skip test"
fi

echo "=== Done. Results saved to: $OUT_ROOT ==="
