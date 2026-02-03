#!/bin/bash
# Outlier-focused OVIS eval (identity-safe, no new objects from text prompts).
#
# Goal:
# - Diagnose *why* fusion-only / learned-gate have rare catastrophic regressions on the full test split.
# - Collect per-frame signals + learned-gate logs for a curated set of problematic (video_id,ann_id) pairs.
# - Sweep fusion strength for:
#     (a) fusion_only (gate off) and
#     (b) fusion_gate (gate on; fusion is modulated by gate_fusion internally)
#
# Output:
# - Per-run metrics.json/csv + per-instance summary.json
# - Per-instance signals.csv + signals.png (for the curated pairs)
# - analysis/compare_*.txt comparing all runs to baseline
# - A tar.gz of the whole folder for download
#
# Usage (server):
#   sbatch scripts/sbatch_eval_ovis_spme_gate_outliers_no_newobj_v1_sweep_alpha.sh
#
# Optional overrides:
#   export GATE_CKPT=/path/to/spme_gate_stepXXXXX.pt
#   export OUT_ROOT=/path/to/outputs/job_${SLURM_JOB_ID}
#   export PAIRS_TXT=/path/to/pairs.txt
#   export MAX_FRAMES=0   # 0 = full from init frame
#   export TAU=0.5
#   export SWEEP_ALPHAS="0.01 0.005 0.002"
#   export SWEEP_ALPHA_OBJS="0.001 0.0005 0.0002"
#   export FUSION_MODE=film
#   export FUSION_DET_THR=0.0
#   export GATE_OCC_NORM=10.0
#
# Notes:
# - This script is intended for SLURM servers; do not run locally.
# - Identity-safe setting: SAM3_ALLOW_NEW_DETECTIONS_WITH_TEXT=0.

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -J sam3_spme_gate_outliers_no_newobj_v1
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
# Detections are still computed and used for matching/confirmation + SPME signals.
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
TEST_SPLIT=$SPLIT_DIR/test.txt

IMAGES_ROOT=${IMAGES_ROOT:-$OVIS/train}
ANN=$OVIS/annotations_train.json

GATE_CKPT=${GATE_CKPT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/spme_gate_ovis_train_v1/main/checkpoints/spme_gate_step18000.pt}

OUT_ROOT=${OUT_ROOT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/spme_gate_ovis_outliers_no_newobj_v1/job_${SLURM_JOB_ID:-local}}

# Optional caps for faster debugging (0 = full from init frame)
MAX_FRAMES=${MAX_FRAMES:-0}
TAU=${TAU:-0.5}

# SPME knobs (override via env if needed)
QUERY_POOL=${QUERY_POOL:-top1}
QUERY_TOPK=${QUERY_TOPK:-5}
ANCHOR_DET_THR=${ANCHOR_DET_THR:-0.0}

FUSION_MODE=${FUSION_MODE:-film}
FUSION_DET_THR=${FUSION_DET_THR:-0.0}
FUSION_USE_PRESENCE=${FUSION_USE_PRESENCE:-1}

GATE_OCC_NORM=${GATE_OCC_NORM:-10.0}
GATE_HIDDEN=${SAM3_SPME_GATE_HIDDEN:-64}

ANALYZE=${ANALYZE:-1}
BOOTSTRAP=${BOOTSTRAP:-20000}
TOPK=${TOPK:-50}

SWEEP_ALPHAS=${SWEEP_ALPHAS:-"0.01 0.005 0.002"}
SWEEP_ALPHA_OBJS=${SWEEP_ALPHA_OBJS:-"0.001 0.0005 0.0002"}

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

mkdir -p "$OUT_ROOT/pairs"

PAIRS_TXT=${PAIRS_TXT:-$OUT_ROOT/pairs/outliers_job_15908744.txt}
if [ ! -s "$PAIRS_TXT" ]; then
  # Curated from job_15908744 (full test):
  # - fusion_only catastrophic IoU regression: 0423/002506
  # - gate_only catastrophic IoU regressions: 0198/001337, 0606/003570, 0542/003230
  # - gate_only ghost regressions: 0156/001121, 0144/001039, 0066/000474, 0198/001339, 0221/001475, 0379/002248
  # - include a few improvements for contrast
  cat > "$PAIRS_TXT" <<'EOF'
423 2506
198 1337
606 3570
542 3230

156 1121
144 1039
66 474
198 1339
221 1475
379 2248

150 1068
144 1038
198 1340
420 2491
156 1118
409 2435
EOF
fi

COMMON_ARGS=(
  --images-root "$IMAGES_ROOT"
  --ann "$ANN"
  --split-txt "$TEST_SPLIT"
  --base-sam3-pt "$SAM3_PT"
  --bpe-path "$BPE"
  --max-videos 0
  --max-instances 0
  --max-frames "$MAX_FRAMES"
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
  local alpha="$1"
  local alpha_obj="$2"
  export SAM3_SPME_FUSION=1
  export SAM3_SPME_FUSION_MODE="$FUSION_MODE"
  export SAM3_SPME_FUSION_ALPHA="$alpha"
  export SAM3_SPME_FUSION_ALPHA_OBJ="$alpha_obj"
  export SAM3_SPME_FUSION_DET_THR="$FUSION_DET_THR"
  export SAM3_SPME_FUSION_USE_PRESENCE="$FUSION_USE_PRESENCE"
}

set_gate_env() {
  local use_decay="$1"
  export SAM3_SPME_LEARNED_GATE=1
  export SAM3_SPME_LEARNED_GATE_USE_DECAY="$use_decay"
  export SAM3_SPME_LEARNED_GATE_OCCLUDED_NORM="$GATE_OCC_NORM"
  export SAM3_SPME_LEARNED_GATE_LOG=1
  export SAM3_SPME_GATE_HIDDEN="$GATE_HIDDEN"
}

alpha_tag() {
  # 0.005 -> 0p005 (safe for dir names)
  echo "$1" | sed 's/\\./p/g'
}

run_eval() {
  local out_dir="$1"
  local spme_ckpt="$2"
  shift 2
  python -u scripts/phase0_eval_ovis_signals.py \
    --out-dir "$out_dir" \
    --only-pairs "$PAIRS_TXT" \
    --save-signals 1 \
    ${spme_ckpt:+--spme-fusion-ckpt "$spme_ckpt"} \
    "${COMMON_ARGS[@]}" \
    "$@"
}

mkdir -p "$OUT_ROOT"
{
  echo "job_id=${SLURM_JOB_ID:-none}"
  echo "repo=$REPO"
  echo "git_rev=$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
  echo "gate_ckpt=$GATE_CKPT"
  echo "no_newobj_from_text=1"
  echo "pairs_txt=$PAIRS_TXT"
  echo "max_frames=$MAX_FRAMES tau=$TAU"
  echo "sweep_alphas=$SWEEP_ALPHAS"
  echo "sweep_alpha_objs=$SWEEP_ALPHA_OBJS"
  echo "fusion_mode=$FUSION_MODE det_thr=$FUSION_DET_THR use_presence=$FUSION_USE_PRESENCE"
  echo "gate_occ_norm=$GATE_OCC_NORM gate_hidden=$GATE_HIDDEN"
} > "$OUT_ROOT/README_run.txt"

echo "=== [outliers] baseline ==="
unset_all_spme
run_eval "$OUT_ROOT/baseline" ""

echo "=== [outliers] gate_only (decay=1; alpha=0) ==="
unset_all_spme
set_common_spme_signal_env
set_fusion_env 0.0 0.0
set_gate_env 1
run_eval "$OUT_ROOT/gate_only" "$GATE_CKPT"

echo "=== [outliers] gate_only_nodecay (decay=0; alpha=0) ==="
unset_all_spme
set_common_spme_signal_env
set_fusion_env 0.0 0.0
set_gate_env 0
run_eval "$OUT_ROOT/gate_only_nodecay" "$GATE_CKPT"

echo "=== [outliers] sweep fusion_only + fusion_gate ==="
mkdir -p "$OUT_ROOT/sweep"

local_alphas=($SWEEP_ALPHAS)
local_alpha_objs=($SWEEP_ALPHA_OBJS)
if [ "${#local_alphas[@]}" -ne "${#local_alpha_objs[@]}" ]; then
  echo "ERROR: SWEEP_ALPHAS and SWEEP_ALPHA_OBJS must have same length."
  echo "  SWEEP_ALPHAS=($SWEEP_ALPHAS)"
  echo "  SWEEP_ALPHA_OBJS=($SWEEP_ALPHA_OBJS)"
  exit 1
fi

for i in "${!local_alphas[@]}"; do
  a="${local_alphas[$i]}"
  ao="${local_alpha_objs[$i]}"
  a_tag="$(alpha_tag "$a")"

  echo "=== [outliers] fusion_only_a${a} (alpha_obj=${ao}; gate off) ==="
  unset_all_spme
  set_common_spme_signal_env
  set_fusion_env "$a" "$ao"
  export SAM3_SPME_LEARNED_GATE=0
  run_eval "$OUT_ROOT/sweep/fusion_only_a${a_tag}" "$GATE_CKPT"

  echo "=== [outliers] fusion_gate_a${a} (alpha_obj=${ao}; gate on) ==="
  unset_all_spme
  set_common_spme_signal_env
  set_fusion_env "$a" "$ao"
  set_gate_env 1
  run_eval "$OUT_ROOT/sweep/fusion_gate_a${a_tag}" "$GATE_CKPT"
done

analyze_metric() {
  local metric="$1"
  local subset="$2"
  local out_path="$OUT_ROOT/analysis/compare_${metric}_${subset}.txt"

  mkdir -p "$OUT_ROOT/analysis"

  # Build run-dir list.
  run_dirs=(
    "$OUT_ROOT/gate_only"
    "$OUT_ROOT/gate_only_nodecay"
  )
  for i in "${!local_alphas[@]}"; do
    a="${local_alphas[$i]}"
    a_tag="$(alpha_tag "$a")"
    run_dirs+=("$OUT_ROOT/sweep/fusion_only_a${a_tag}")
    run_dirs+=("$OUT_ROOT/sweep/fusion_gate_a${a_tag}")
  done

  args=(--baseline-dir "$OUT_ROOT/baseline" --metric "$metric" --subset "$subset" --top-k "$TOPK" --bootstrap "$BOOTSTRAP" --seed 0)
  for rd in "${run_dirs[@]}"; do
    args+=(--run-dir "$rd")
  done

  python -u scripts/analyze_ovis_eval.py "${args[@]}" > "$out_path"
}

if [ "$ANALYZE" = "1" ]; then
  echo "=== [outliers] analysis ==="
  analyze_metric mean_iou_track_present all
  analyze_metric fp_absent_rate_track baseline_gt_absent_gt0
  analyze_metric fp_tail_run_track baseline_fp_tail_gt0
  analyze_metric identity_swap_frac_tau all
  analyze_metric fail_frac_tau all
  analyze_metric persistence_until_fail_tau all
fi

echo "=== Done. Results saved to: $OUT_ROOT ==="

if command -v tar >/dev/null 2>&1; then
  echo "=== Packaging results into tar.gz ==="
  TAR_PATH="${OUT_ROOT}.tar.gz"
  tar -czf "$TAR_PATH" -C "$(dirname "$OUT_ROOT")" "$(basename "$OUT_ROOT")"
  echo "Tarball: $TAR_PATH"
fi

