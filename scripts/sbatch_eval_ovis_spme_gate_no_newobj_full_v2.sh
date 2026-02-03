#!/bin/bash
# Full OVIS eval (identity-safe) for selecting the final method.
#
# Key design choice:
# - Disable spawning NEW objects from text prompts (identity takeover prevention).
#   Detections are still computed for matching/confirmation + SPME signals.
#
# Runs (test split by default):
# - baseline
# - fusion_only (trained fusion projector; no learned gate)
# - gate_only (learned gate; fusion injection disabled via alpha=0)
# - gate_only_nodecay (learned gate; decay disabled)
#
# Output:
# - Per-run metrics.json/csv
# - analysis/compare_*.txt (bootstrap CIs + top improved/regressed)
# - Packs a tar.gz for download
#
# Usage (server):
#   sbatch scripts/sbatch_eval_ovis_spme_gate_no_newobj_full_v2.sh
#
# Optional overrides:
#   export GATE_CKPT=/path/to/spme_gate_stepXXXXX.pt
#   export OUT_ROOT=/path/to/outputs/job_${SLURM_JOB_ID}
#   export RUN_VAL=1 RUN_TEST=1
#   export FUSION_ALPHA=0.01 FUSION_ALPHA_OBJ=0.001
#   export MAX_VIDEOS=0 MAX_INSTANCES=0 MAX_FRAMES=0
#
# Notes:
# - This script is intended for server SLURM runs; do not run locally.

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -J sam3_spme_gate_ovis_eval_no_newobj_v2
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.err

set -euo pipefail

source /home2020/home/icube/kunyuan/anaconda3/etc/profile.d/conda.sh
conda activate sam3

export OMP_NUM_THREADS=8
export HYDRA_FULL_ERROR=1
export SAM3_DISABLE_TRITON=1
export PYTHONUNBUFFERED=1

# Identity-safe: do NOT spawn new objects from text prompts.
export SAM3_ALLOW_NEW_DETECTIONS=1
export SAM3_ALLOW_NEW_DETECTIONS_WITH_TEXT=0

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

GATE_CKPT=${GATE_CKPT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/spme_gate_ovis_train_v1/main/checkpoints/spme_gate_step18000.pt}

OUT_ROOT=${OUT_ROOT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/spme_gate_ovis_eval_no_newobj_v2/job_${SLURM_JOB_ID:-local}}

# 0 = full eval (no limit)
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
FUSION_ALPHA=${FUSION_ALPHA:-0.01}
FUSION_ALPHA_OBJ=${FUSION_ALPHA_OBJ:-0.001}
FUSION_DET_THR=${FUSION_DET_THR:-0.0}
FUSION_USE_PRESENCE=${FUSION_USE_PRESENCE:-1}

GATE_OCC_NORM=${GATE_OCC_NORM:-10.0}
GATE_HIDDEN=${SAM3_SPME_GATE_HIDDEN:-64}

ANALYZE=${ANALYZE:-1}
BOOTSTRAP=${BOOTSTRAP:-20000}
TOPK=${TOPK:-50}

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
}

set_gate_env() {
  local use_decay="$1"
  export SAM3_SPME_LEARNED_GATE=1
  export SAM3_SPME_LEARNED_GATE_USE_DECAY="$use_decay"
  export SAM3_SPME_LEARNED_GATE_OCCLUDED_NORM="$GATE_OCC_NORM"
  export SAM3_SPME_GATE_HIDDEN="$GATE_HIDDEN"
}

analyze_metric() {
  local baseline_dir="$1"
  local analysis_dir="$2"
  local metric="$3"
  local subset="$4"

  python -u scripts/analyze_ovis_eval.py \
    --baseline-dir "$baseline_dir" \
    --run-dir "$analysis_dir/../fusion_only" \
    --run-dir "$analysis_dir/../gate_only" \
    --run-dir "$analysis_dir/../gate_only_nodecay" \
    --metric "$metric" \
    --subset "$subset" \
    --top-k "$TOPK" \
    --bootstrap "$BOOTSTRAP" \
    --seed 0 \
    > "$analysis_dir/compare_${metric}_${subset}.txt"
}

eval_one_split() {
  local split_name="$1"
  local split_txt="$2"
  local out_dir="$OUT_ROOT/$split_name"
  mkdir -p "$out_dir"

  echo "=== [OVIS][$split_name] baseline ==="
  unset_all_spme
  python -u scripts/phase0_eval_ovis_signals.py \
    --split-txt "$split_txt" \
    --out-dir "$out_dir/baseline" \
    --max-frames "$MAX_FRAMES" \
    "${COMMON_ARGS[@]}"

  echo "=== [OVIS][$split_name] fusion_only (alpha=$FUSION_ALPHA; gate off) ==="
  unset_all_spme
  set_common_spme_signal_env
  set_fusion_env
  export SAM3_SPME_LEARNED_GATE=0
  python -u scripts/phase0_eval_ovis_signals.py \
    --split-txt "$split_txt" \
    --out-dir "$out_dir/fusion_only" \
    --max-frames "$MAX_FRAMES" \
    --spme-fusion-ckpt "$GATE_CKPT" \
    "${COMMON_ARGS[@]}"

  echo "=== [OVIS][$split_name] gate_only (decay=1; fusion alpha=0) ==="
  unset_all_spme
  set_common_spme_signal_env
  export SAM3_SPME_FUSION=1
  export SAM3_SPME_FUSION_MODE="$FUSION_MODE"
  export SAM3_SPME_FUSION_ALPHA=0.0
  export SAM3_SPME_FUSION_ALPHA_OBJ=0.0
  export SAM3_SPME_FUSION_DET_THR="$FUSION_DET_THR"
  export SAM3_SPME_FUSION_USE_PRESENCE="$FUSION_USE_PRESENCE"
  set_gate_env 1
  python -u scripts/phase0_eval_ovis_signals.py \
    --split-txt "$split_txt" \
    --out-dir "$out_dir/gate_only" \
    --max-frames "$MAX_FRAMES" \
    --spme-fusion-ckpt "$GATE_CKPT" \
    "${COMMON_ARGS[@]}"

  echo "=== [OVIS][$split_name] gate_only_nodecay (decay=0; fusion alpha=0) ==="
  unset_all_spme
  set_common_spme_signal_env
  export SAM3_SPME_FUSION=1
  export SAM3_SPME_FUSION_MODE="$FUSION_MODE"
  export SAM3_SPME_FUSION_ALPHA=0.0
  export SAM3_SPME_FUSION_ALPHA_OBJ=0.0
  export SAM3_SPME_FUSION_DET_THR="$FUSION_DET_THR"
  export SAM3_SPME_FUSION_USE_PRESENCE="$FUSION_USE_PRESENCE"
  set_gate_env 0
  python -u scripts/phase0_eval_ovis_signals.py \
    --split-txt "$split_txt" \
    --out-dir "$out_dir/gate_only_nodecay" \
    --max-frames "$MAX_FRAMES" \
    --spme-fusion-ckpt "$GATE_CKPT" \
    "${COMMON_ARGS[@]}"

  if [ "$ANALYZE" = "1" ]; then
    echo "=== [OVIS][$split_name] analysis ==="
    local analysis_dir="$out_dir/analysis"
    mkdir -p "$analysis_dir"

    analyze_metric "$out_dir/baseline" "$analysis_dir" mean_iou_track_present all
    analyze_metric "$out_dir/baseline" "$analysis_dir" fp_absent_rate_track baseline_gt_absent_gt0
    analyze_metric "$out_dir/baseline" "$analysis_dir" fp_tail_run_track baseline_fp_tail_gt0
    analyze_metric "$out_dir/baseline" "$analysis_dir" fail_frac_tau all
    analyze_metric "$out_dir/baseline" "$analysis_dir" identity_swap_frac_tau all
    analyze_metric "$out_dir/baseline" "$analysis_dir" persistence_until_fail_tau all
  fi
}

mkdir -p "$OUT_ROOT"
{
  echo "job_id=${SLURM_JOB_ID:-none}"
  echo "repo=$REPO"
  echo "git_rev=$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
  echo "gate_ckpt=$GATE_CKPT"
  echo "no_newobj_from_text=1"
  echo "fusion_alpha=$FUSION_ALPHA fusion_alpha_obj=$FUSION_ALPHA_OBJ fusion_mode=$FUSION_MODE"
  echo "gate_occ_norm=$GATE_OCC_NORM gate_hidden=$GATE_HIDDEN"
  echo "max_videos=$MAX_VIDEOS max_instances=$MAX_INSTANCES max_frames=$MAX_FRAMES tau=$TAU"
} > "$OUT_ROOT/README_run.txt"

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

if command -v tar >/dev/null 2>&1; then
  echo "=== Packaging results into tar.gz ==="
  TAR_PATH="${OUT_ROOT}.tar.gz"
  tar -czf "$TAR_PATH" -C "$(dirname "$OUT_ROOT")" "$(basename "$OUT_ROOT")"
  echo "Tarball: $TAR_PATH"
fi

