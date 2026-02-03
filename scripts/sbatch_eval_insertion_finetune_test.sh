#!/bin/bash
#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_us_insertion_ft_eval_ckpt28_modules
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.err

set -euo pipefail

source /home2020/home/icube/kunyuan/anaconda3/etc/profile.d/conda.sh
conda activate sam3

export OMP_NUM_THREADS=8
export HYDRA_FULL_ERROR=1
export SAM3_DISABLE_TRITON=1

REPO=/home2020/home/icube/kunyuan/SurgBench/SAM/sam3
DATA=/home2020/home/icube/kunyuan/SurgBench/Ultrasound/pork_dataset
SAM3_PT=$REPO/sam3.pt
BPE=$REPO/sam3/assets/bpe_simple_vocab_16e6.txt.gz

SAMPLE_TXT=$DATA/test.txt
INSERTION_END_TXT=$DATA/insertion_end.txt

# Baseline: your previous best detector fine-tune.
CKPT_BASE=/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/sam3_us_needle_detection_boxprompt/checkpoints/checkpoint_18.pt

# New: insertion-only fine-tune output dir (update if you changed experiment_log_dir).
CKPT_NEW_DIR=/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/sam3_us_needle_insertion_only/checkpoints
# For this run, we want to specifically test checkpoint_28.pt.
# You can override this from the CLI:
#   sbatch --export=ALL,CKPT_NEW_OVERRIDE=/path/to/checkpoint_X.pt scripts/sbatch_eval_insertion_finetune_test.sh
CKPT_NEW_OVERRIDE="${CKPT_NEW_OVERRIDE:-$CKPT_NEW_DIR/checkpoint_28.pt}"
if [[ -n "$CKPT_NEW_OVERRIDE" && -f "$CKPT_NEW_OVERRIDE" ]]; then
  CKPT_NEW="$CKPT_NEW_OVERRIDE"
else
  CKPT_NEW="$(ls -t "$CKPT_NEW_DIR"/checkpoint_*.pt 2>/dev/null | head -n 1 || true)"
  if [[ -z "$CKPT_NEW" ]]; then
    echo "[error] No checkpoint found in: $CKPT_NEW_DIR"
    exit 1
  fi
fi

# Quick ablation default: 5 representative samples.
# To run full test set, set SAMPLE_IDS="" (empty) via --export and it will use test.txt.
SAMPLE_IDS="${SAMPLE_IDS:-18,159,172,107,129}"

OUT_ROOT_DEFAULT=/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/phase1_spme_w_v7_ckpt28_modules
OUT_ROOT="${OUT_ROOT_OVERRIDE:-$OUT_ROOT_DEFAULT}"

cd "$REPO"

PROMPT_POS="ultrasound needle"

run_eval () {
  local tag="$1"
  local ckpt="$2"
  local scope="$3"  # full_video | insertion_only
  shift 3
  local -a extra_env=("$@")

  echo "=== [$scope] $tag ==="
  unset SAM3_SPME_WRITE_GATE SAM3_SPME_WRITE_GATE_MODE SAM3_SPME_DET_THR SAM3_SPME_QCOS_THR SAM3_SPME_USE_DET_SCORE SAM3_SPME_WRITE_GATE_APPLY SAM3_SPME_DEBUG_SIGNALS SAM3_SPME_ANCHOR_DET_THR
  unset SAM3_SPME_FUSION SAM3_SPME_FUSION_ALPHA SAM3_SPME_FUSION_DET_THR

  # Apply env overrides (as KEY=VALUE strings).
  for kv in "${extra_env[@]}"; do
    export "$kv"
  done

  local -a sample_args=()
  if [[ -n "$SAMPLE_IDS" ]]; then
    sample_args=(--sample-ids "$SAMPLE_IDS")
  else
    sample_args=(--sample-txt "$SAMPLE_TXT")
  fi

  if [[ "$scope" == "insertion_only" ]]; then
    python scripts/phase0_eval_ultrasound_signals.py \
      --frames-root "$DATA/pork" \
      --gt-root "$DATA/gt" \
      "${sample_args[@]}" \
      --insertion-end-txt "$INSERTION_END_TXT" \
      --out-dir "$OUT_ROOT/$scope/$tag" \
      --base-sam3-pt "$SAM3_PT" \
      --finetune-ckpt "$ckpt" \
      --bpe-path "$BPE" \
      --prompt "$PROMPT_POS" \
      --modes gtbbox_text
  else
    python scripts/phase0_eval_ultrasound_signals.py \
      --frames-root "$DATA/pork" \
      --gt-root "$DATA/gt" \
      "${sample_args[@]}" \
      --out-dir "$OUT_ROOT/$scope/$tag" \
      --base-sam3-pt "$SAM3_PT" \
      --finetune-ckpt "$ckpt" \
      --bpe-path "$BPE" \
      --prompt "$PROMPT_POS" \
      --modes gtbbox_text
  fi
}

echo "Using CKPT_BASE=$CKPT_BASE"
echo "Using CKPT_NEW=$CKPT_NEW"
echo "Using SAMPLE_IDS=${SAMPLE_IDS:-<test.txt>}"
echo "Using OUT_ROOT=$OUT_ROOT"

# 1) Baseline comparison: ckpt18 vs ckpt28.
run_eval "baseline_ckpt18" "$CKPT_BASE" "insertion_only"
run_eval "baseline_ckpt28" "$CKPT_NEW" "insertion_only"

if [[ "${RUN_MODULES:-1}" == "1" ]]; then
  # 2) Module ablations on ckpt28 (fast: insertion_only).
  # SPME-W (soft balanced) config:
  #   gate = clamp((qcos - 0.6)/(1-0.6), 0, 1) * det_score, then blend_mem.
  run_eval "ckpt28_spme_w_soft_q0p6_det0p3_blend" "$CKPT_NEW" "insertion_only" \
    "SAM3_SPME_WRITE_GATE=1" \
    "SAM3_SPME_WRITE_GATE_MODE=soft" \
    "SAM3_SPME_QCOS_THR=0.6" \
    "SAM3_SPME_DET_THR=0.3" \
    "SAM3_SPME_USE_DET_SCORE=1" \
    "SAM3_SPME_WRITE_GATE_APPLY=blend_mem" \
    "SAM3_SPME_ANCHOR_DET_THR=0.3"

  # SPME-F only (feature fusion / reinforcement): residual injection into maskmem_features.
  run_eval "ckpt28_spme_f_a0p05_det0p3" "$CKPT_NEW" "insertion_only" \
    "SAM3_SPME_FUSION=1" \
    "SAM3_SPME_FUSION_ALPHA=0.05" \
    "SAM3_SPME_FUSION_DET_THR=0.3" \
    "SAM3_SPME_ANCHOR_DET_THR=0.3"

  # SPME-W + SPME-F (expected best starting point).
  run_eval "ckpt28_spme_w_soft_q0p6_det0p3_blend__spme_f_a0p05" "$CKPT_NEW" "insertion_only" \
    "SAM3_SPME_WRITE_GATE=1" \
    "SAM3_SPME_WRITE_GATE_MODE=soft" \
    "SAM3_SPME_QCOS_THR=0.6" \
    "SAM3_SPME_DET_THR=0.3" \
    "SAM3_SPME_USE_DET_SCORE=1" \
    "SAM3_SPME_WRITE_GATE_APPLY=blend_mem" \
    "SAM3_SPME_FUSION=1" \
    "SAM3_SPME_FUSION_ALPHA=0.05" \
    "SAM3_SPME_FUSION_DET_THR=0.3" \
    "SAM3_SPME_ANCHOR_DET_THR=0.3"
fi

# 3) If the insertion_only results show improvement, re-run only the best tags on full_video.
# (Uncomment as needed; this is slower.)
# run_eval "baseline_ckpt28" "$CKPT_NEW" "full_video"
# run_eval "ckpt28_spme_w_soft_q0p6_det0p3_blend__spme_f_a0p05" "$CKPT_NEW" "full_video" \
#   "SAM3_SPME_WRITE_GATE=1" \
#   "SAM3_SPME_WRITE_GATE_MODE=soft" \
#   "SAM3_SPME_QCOS_THR=0.6" \
#   "SAM3_SPME_DET_THR=0.3" \
#   "SAM3_SPME_USE_DET_SCORE=1" \
#   "SAM3_SPME_WRITE_GATE_APPLY=blend_mem" \
#   "SAM3_SPME_FUSION=1" \
#   "SAM3_SPME_FUSION_ALPHA=0.05" \
#   "SAM3_SPME_FUSION_DET_THR=0.3" \
#   "SAM3_SPME_ANCHOR_DET_THR=0.3"

echo "=== Done. Results saved to: $OUT_ROOT ==="
