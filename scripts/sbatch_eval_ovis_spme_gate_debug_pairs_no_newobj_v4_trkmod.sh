#!/bin/bash
# Debug-pair eval to validate a robustness tweak:
#   SAM3_SPME_LEARNED_GATE_SCALE_BY_TRK=1
# which scales learned memory write gates by tracker_score to avoid overwriting memory when masks are low-quality.
#
# It runs both with and without the tweak, writes comprehensive metrics, and packages a tar.gz for download.
#
# Usage (server):
#   sbatch scripts/sbatch_eval_ovis_spme_gate_debug_pairs_no_newobj_v4_trkmod.sh
#
# Optional overrides:
#   export GATE_CKPT=/path/to/spme_gate_stepXXXXX.pt
#   export OUT_ROOT=/path/to/outputs/job_${SLURM_JOB_ID}
#   export FUSION_ALPHA=0.01
#   export FUSION_ALPHA_OBJ=0.001
#
# Notes:
# - Identity-safe: disables spawning NEW objects from text prompts (detections still computed for matching/signals).

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_spme_gate_debug_no_newobj_v4_trkmod
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.err

set -euo pipefail

source /home2020/home/icube/kunyuan/anaconda3/etc/profile.d/conda.sh
conda activate sam3

export OMP_NUM_THREADS=8
export HYDRA_FULL_ERROR=1
export SAM3_DISABLE_TRITON=1
export PYTHONUNBUFFERED=1

# Disable spawning NEW objects from text prompts.
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

OUT_ROOT=${OUT_ROOT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/spme_gate_ovis_debug_pairs_no_newobj_v4_trkmod/job_${SLURM_JOB_ID:-local}}

# Optional caps for faster debugging (0 = full from init frame)
MAX_FRAMES_PAIRS=${MAX_FRAMES_PAIRS:-0}
TAU=${TAU:-0.5}

# SPME knobs (override via env if needed)
QUERY_POOL=${QUERY_POOL:-top1}
QUERY_TOPK=${QUERY_TOPK:-5}
ANCHOR_DET_THR=${ANCHOR_DET_THR:-0.0}

FUSION_MODE=${FUSION_MODE:-film}
FUSION_ALPHA=${FUSION_ALPHA:-0.01}
FUSION_ALPHA_OBJ=${FUSION_ALPHA_OBJ:-0.001}
FUSION_DET_THR=${FUSION_DET_THR:-0.0}
FUSION_USE_PRESENCE=${FUSION_USE_PRESENCE:-1}

GATE_USE_DECAY=${GATE_USE_DECAY:-1}
GATE_OCC_NORM=${GATE_OCC_NORM:-10.0}
GATE_HIDDEN=${SAM3_SPME_GATE_HIDDEN:-64}

ANALYZE=${ANALYZE:-1}
BOOTSTRAP=${BOOTSTRAP:-20000}
TOPK=${TOPK:-50}

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

PAIRS_REGRESSED_IOU=${PAIRS_REGRESSED_IOU:-$OUT_ROOT/pairs/top_regressed_iou.txt}
if [ ! -s "$PAIRS_REGRESSED_IOU" ]; then
  cat > "$PAIRS_REGRESSED_IOU" <<'EOF'
28 207
129 905
84 586
150 1068
542 3230
67 481
505 3014
103 704
156 1119
538 3198
54 339
505 3007
103 706
151 1082
505 3008
66 471
48 304
379 2248
308 1935
129 898
149 1059
379 2251
40 270
104 710
156 1113
423 2513
129 891
469 2832
505 3011
104 709
67 479
156 1116
67 478
379 2247
559 3320
402 2389
40 269
156 1115
590 3497
129 892
100 690
86 609
526 3131
89 627
395 2353
86 610
198 1342
144 1033
156 1114
156 1125
EOF
fi

PAIRS_GHOST_IMPROVED=${PAIRS_GHOST_IMPROVED:-$OUT_ROOT/pairs/top_improved_fp_absent_rate_track.txt}
if [ ! -s "$PAIRS_GHOST_IMPROVED" ]; then
  cat > "$PAIRS_GHOST_IMPROVED" <<'EOF'
129 892
420 2491
156 1114
198 1342
28 207
198 1340
156 1118
28 208
84 585
420 2490
109 744
198 1341
66 473
103 707
129 899
373 2224
570 3372
66 466
100 692
100 693
28 206
40 271
48 303
48 304
48 305
54 337
54 339
66 465
66 470
66 471
66 474
66 475
67 476
67 478
67 483
67 484
84 584
84 586
84 587
84 588
84 589
84 590
84 591
84 592
84 593
84 594
84 595
86 610
86 611
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
  --max-frames "$MAX_FRAMES_PAIRS"
  --tau "$TAU"
)

unset_all_spme() {
  unset SAM3_SPME_FUSION SAM3_SPME_FUSION_MODE SAM3_SPME_FUSION_ALPHA SAM3_SPME_FUSION_ALPHA_OBJ SAM3_SPME_FUSION_DET_THR SAM3_SPME_FUSION_USE_PRESENCE SAM3_SPME_FUSION_QCOS_THR SAM3_SPME_FUSION_QCOS_TEMP SAM3_SPME_FUSION_QCOS_GATE SAM3_SPME_FUSION_USE_WRITE_GATE
  unset SAM3_SPME_QUERY_POOL SAM3_SPME_QUERY_TOPK SAM3_SPME_ANCHOR_DET_THR SAM3_SPME_PER_OBJECT SAM3_SPME_PER_OBJECT_FALLBACK SAM3_SPME_KEEP_QUERIES
  unset SAM3_SPME_WRITE_GATE SAM3_SPME_WRITE_GATE_MODE SAM3_SPME_DET_THR SAM3_SPME_QCOS_THR SAM3_SPME_USE_DET_SCORE SAM3_SPME_WRITE_GATE_APPLY SAM3_SPME_DEBUG_SIGNALS
  unset SAM3_SPME_LEARNED_GATE SAM3_SPME_LEARNED_GATE_USE_DECAY SAM3_SPME_LEARNED_GATE_OCCLUDED_NORM SAM3_SPME_LEARNED_GATE_LOG SAM3_SPME_GATE_HIDDEN SAM3_SPME_LEARNED_GATE_SCALE_BY_TRK
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
  export SAM3_SPME_LEARNED_GATE=1
  export SAM3_SPME_LEARNED_GATE_USE_DECAY="$GATE_USE_DECAY"
  export SAM3_SPME_LEARNED_GATE_OCCLUDED_NORM="$GATE_OCC_NORM"
  export SAM3_SPME_LEARNED_GATE_LOG=1
  export SAM3_SPME_GATE_HIDDEN="$GATE_HIDDEN"
}

analyze_metric() {
  local baseline_dir="$1"
  local out_dir="$2"
  local metric="$3"
  local subset="$4"

  python -u scripts/analyze_ovis_eval.py \
    --baseline-dir "$baseline_dir" \
    --run-dir "$out_dir/fusion_only" \
    --run-dir "$out_dir/gate_only" \
    --run-dir "$out_dir/gate_only_trkmod" \
    --run-dir "$out_dir/fusion_gate" \
    --run-dir "$out_dir/fusion_gate_trkmod" \
    --metric "$metric" \
    --subset "$subset" \
    --top-k "$TOPK" \
    --bootstrap "$BOOTSTRAP" \
    --seed 0 \
    --write-signal-lists "$out_dir/analysis/pairs_${metric}_${subset}" \
    > "$out_dir/analysis/compare_${metric}_${subset}.txt"
}

run_suite() {
  local tag="$1"
  local pairs="$2"
  local out_dir="$OUT_ROOT/$tag"
  mkdir -p "$out_dir"

  echo "=== [$tag] baseline ==="
  unset_all_spme
  python -u scripts/phase0_eval_ovis_signals.py \
    --out-dir "$out_dir/baseline" \
    --only-pairs "$pairs" \
    --save-signals 1 \
    "${COMMON_ARGS[@]}"

  echo "=== [$tag] fusion_only (load ckpt; learned gate disabled) ==="
  unset_all_spme
  set_common_spme_signal_env
  set_fusion_env
  export SAM3_SPME_LEARNED_GATE=0
  python -u scripts/phase0_eval_ovis_signals.py \
    --out-dir "$out_dir/fusion_only" \
    --only-pairs "$pairs" \
    --save-signals 1 \
    --spme-fusion-ckpt "$GATE_CKPT" \
    "${COMMON_ARGS[@]}"

  echo "=== [$tag] gate_only (alpha=0; learned gate on) ==="
  unset_all_spme
  set_common_spme_signal_env
  export SAM3_SPME_FUSION=1
  export SAM3_SPME_FUSION_ALPHA=0.0
  export SAM3_SPME_FUSION_ALPHA_OBJ=0.0
  set_gate_env
  python -u scripts/phase0_eval_ovis_signals.py \
    --out-dir "$out_dir/gate_only" \
    --only-pairs "$pairs" \
    --save-signals 1 \
    --spme-fusion-ckpt "$GATE_CKPT" \
    "${COMMON_ARGS[@]}"

  echo "=== [$tag] gate_only_trkmod (alpha=0; learned gate + trkmod) ==="
  unset_all_spme
  set_common_spme_signal_env
  export SAM3_SPME_FUSION=1
  export SAM3_SPME_FUSION_ALPHA=0.0
  export SAM3_SPME_FUSION_ALPHA_OBJ=0.0
  set_gate_env
  export SAM3_SPME_LEARNED_GATE_SCALE_BY_TRK=1
  python -u scripts/phase0_eval_ovis_signals.py \
    --out-dir "$out_dir/gate_only_trkmod" \
    --only-pairs "$pairs" \
    --save-signals 1 \
    --spme-fusion-ckpt "$GATE_CKPT" \
    "${COMMON_ARGS[@]}"

  echo "=== [$tag] fusion_gate (fusion + learned gate) ==="
  unset_all_spme
  set_common_spme_signal_env
  set_fusion_env
  set_gate_env
  python -u scripts/phase0_eval_ovis_signals.py \
    --out-dir "$out_dir/fusion_gate" \
    --only-pairs "$pairs" \
    --save-signals 1 \
    --spme-fusion-ckpt "$GATE_CKPT" \
    "${COMMON_ARGS[@]}"

  echo "=== [$tag] fusion_gate_trkmod (fusion + learned gate + trkmod) ==="
  unset_all_spme
  set_common_spme_signal_env
  set_fusion_env
  set_gate_env
  export SAM3_SPME_LEARNED_GATE_SCALE_BY_TRK=1
  python -u scripts/phase0_eval_ovis_signals.py \
    --out-dir "$out_dir/fusion_gate_trkmod" \
    --only-pairs "$pairs" \
    --save-signals 1 \
    --spme-fusion-ckpt "$GATE_CKPT" \
    "${COMMON_ARGS[@]}"

  if [ "$ANALYZE" = "1" ]; then
    echo "=== [$tag] analysis ==="
    mkdir -p "$out_dir/analysis"
    analyze_metric "$out_dir/baseline" "$out_dir" mean_iou_track_present all
    analyze_metric "$out_dir/baseline" "$out_dir" fp_absent_rate_track baseline_gt_absent_gt0
    analyze_metric "$out_dir/baseline" "$out_dir" fp_tail_run_track baseline_fp_tail_gt0
    analyze_metric "$out_dir/baseline" "$out_dir" fail_frac_tau all
    analyze_metric "$out_dir/baseline" "$out_dir" identity_swap_frac_tau all
    analyze_metric "$out_dir/baseline" "$out_dir" persistence_until_fail_tau all
  fi
}

echo "job_id=${SLURM_JOB_ID:-none}" > "$OUT_ROOT/README_run.txt"
echo "repo=$REPO" >> "$OUT_ROOT/README_run.txt"
echo "git_rev=$(git rev-parse HEAD 2>/dev/null || echo 'unknown')" >> "$OUT_ROOT/README_run.txt"
echo "gate_ckpt=$GATE_CKPT" >> "$OUT_ROOT/README_run.txt"
echo "max_frames_pairs=$MAX_FRAMES_PAIRS tau=$TAU" >> "$OUT_ROOT/README_run.txt"
echo "fusion_alpha=$FUSION_ALPHA fusion_alpha_obj=$FUSION_ALPHA_OBJ" >> "$OUT_ROOT/README_run.txt"
echo "fusion_mode=$FUSION_MODE det_thr=$FUSION_DET_THR use_presence=$FUSION_USE_PRESENCE" >> "$OUT_ROOT/README_run.txt"
echo "gate_use_decay=$GATE_USE_DECAY gate_occ_norm=$GATE_OCC_NORM gate_hidden=$GATE_HIDDEN" >> "$OUT_ROOT/README_run.txt"

run_suite "debug_regressed_iou" "$PAIRS_REGRESSED_IOU"
run_suite "debug_ghost_improved" "$PAIRS_GHOST_IMPROVED"

echo "=== Done. Results saved to: $OUT_ROOT ==="

if command -v tar >/dev/null 2>&1; then
  echo "=== Packaging results into tar.gz ==="
  TAR_PATH="${OUT_ROOT}.tar.gz"
  tar -czf "$TAR_PATH" -C "$(dirname "$OUT_ROOT")" "$(basename "$OUT_ROOT")"
  echo "Tarball: $TAR_PATH"
fi

