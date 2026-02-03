#!/bin/bash
#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_us_phase2_spme_fusion
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

# Detector checkpoint (using the new checkpoint_40 per user request).
CKPT=/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/sam3_us_needle_insertion_only/checkpoints/checkpoint_40.pt

# Optional override: run a quick subset (comma-separated). Leave empty for full test.txt.
SAMPLE_IDS="${SAMPLE_IDS:-}"
# Fixed quick sweep subset (kept independent from SAMPLE_IDS so a single sbatch can do both quick + full).
SAMPLE_IDS_QUICK="${SAMPLE_IDS_QUICK:-18,159,172,107,129}"

# Output folder (overrideable).
OUT_ROOT_DEFAULT=/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/phase2_spme_fusion_v2_ckpt40_eval_softgate
OUT_ROOT="${OUT_ROOT_OVERRIDE:-$OUT_ROOT_DEFAULT}"

# Trained SPME-Fusion checkpoint (spme_* only). Override with `--export=SPME_FUSION_CKPT=...`.
SPME_FUSION_CKPT_DEFAULT=/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/phase2_spme_fusion_train_v1_ckpt40/quick_retrain/checkpoints/spme_fusion_latest.pt
SPME_FUSION_CKPT="${SPME_FUSION_CKPT:-$SPME_FUSION_CKPT_DEFAULT}"

# Run toggles (set to 0 to skip sections).
RUN_QUICK="${RUN_QUICK:-1}"
RUN_FULL="${RUN_FULL:-1}"

PROMPT_POS="ultrasound needle"

cd "$REPO"

run_eval () {
  local tag="$1"
  local scope="$2"  # insertion_only | full_video
  local split="$3"  # quick | full
  local fusion_ckpt="$4"  # optional path (loads spme_* only); empty => no loading
  shift 2
  shift 2
  local -a extra_env=("$@")

  echo "=== [$scope/$split] $tag ==="

  # Reset SPME knobs.
  unset SAM3_SPME_WRITE_GATE SAM3_SPME_WRITE_GATE_MODE SAM3_SPME_DET_THR SAM3_SPME_QCOS_THR SAM3_SPME_USE_DET_SCORE SAM3_SPME_WRITE_GATE_APPLY SAM3_SPME_DEBUG_SIGNALS SAM3_SPME_ANCHOR_DET_THR
  unset SAM3_SPME_QUERY_POOL SAM3_SPME_QUERY_POOL_TEMP SAM3_SPME_QUERY_TOPK
  unset SAM3_SPME_FUSION SAM3_SPME_FUSION_MODE SAM3_SPME_FUSION_ALPHA SAM3_SPME_FUSION_ALPHA_OBJ SAM3_SPME_FUSION_DET_THR SAM3_SPME_FUSION_USE_PRESENCE SAM3_SPME_FUSION_QCOS_THR SAM3_SPME_FUSION_QCOS_TEMP SAM3_SPME_FUSION_QCOS_GATE SAM3_SPME_FUSION_USE_WRITE_GATE
  unset SAM3_SPME_PER_OBJECT

  for kv in "${extra_env[@]}"; do
    export "$kv"
  done

  local -a sample_args=()
  if [[ "$split" == "quick" ]]; then
    sample_args=(--sample-ids "$SAMPLE_IDS_QUICK")
  else
    if [[ -n "$SAMPLE_IDS" ]]; then
      sample_args=(--sample-ids "$SAMPLE_IDS")
    else
      sample_args=(--sample-txt "$SAMPLE_TXT")
    fi
  fi

  if [[ "$scope" == "insertion_only" ]]; then
    python scripts/phase0_eval_ultrasound_signals.py \
      --frames-root "$DATA/pork" \
      --gt-root "$DATA/gt" \
      "${sample_args[@]}" \
      --insertion-end-txt "$INSERTION_END_TXT" \
      --out-dir "$OUT_ROOT/$scope/$tag" \
      --base-sam3-pt "$SAM3_PT" \
      --finetune-ckpt "$CKPT" \
      ${fusion_ckpt:+--spme-fusion-ckpt "$fusion_ckpt"} \
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
      --finetune-ckpt "$CKPT" \
      ${fusion_ckpt:+--spme-fusion-ckpt "$fusion_ckpt"} \
      --bpe-path "$BPE" \
      --prompt "$PROMPT_POS" \
      --modes gtbbox_text
  fi
}

echo "Using CKPT=$CKPT"
echo "Using SAMPLE_IDS=${SAMPLE_IDS:-<test.txt>}"
echo "Using SAMPLE_IDS_QUICK=$SAMPLE_IDS_QUICK"
echo "Using OUT_ROOT=$OUT_ROOT"
echo "Using SPME_FUSION_CKPT=$SPME_FUSION_CKPT"
echo "RUN_QUICK=$RUN_QUICK RUN_FULL=$RUN_FULL"

# ------------------------------------------------------------------
# Phase-2: SPME-Fusion (trainable FiLM memory editing) ablations
# ------------------------------------------------------------------

if [[ "$RUN_QUICK" == "1" ]]; then
  echo ""
  echo "======================"
  echo "=== QUICK SWEEP (5 samples, insertion_only) ==="
  echo "======================"
  echo ""

  # Baseline (ckpt40, no SPME modules).
  run_eval "baseline" "insertion_only" "quick" ""

  # Sanity: enable fusion but DO NOT load trained weights (should be ~baseline due to zero-init).
  run_eval "sanity__spme_fusion_untrained_film_topk5" "insertion_only" "quick" "" \
    "SAM3_SPME_FUSION=1" \
    "SAM3_SPME_FUSION_MODE=film" \
    "SAM3_SPME_FUSION_ALPHA=0.05" \
    "SAM3_SPME_FUSION_ALPHA_OBJ=0.005" \
    "SAM3_SPME_FUSION_DET_THR=0.3" \
    "SAM3_SPME_FUSION_QCOS_THR=0.5" \
    "SAM3_SPME_FUSION_QCOS_TEMP=20" \
    "SAM3_SPME_FUSION_QCOS_GATE=sigmoid" \
    "SAM3_SPME_FUSION_USE_PRESENCE=1" \
    "SAM3_SPME_ANCHOR_DET_THR=0.3" \
    "SAM3_SPME_QUERY_POOL=topk_weighted" \
    "SAM3_SPME_QUERY_TOPK=5"

  # Main trained config (match training hyperparams).
  run_eval "spme_fusion_trained_film_topk5_a0p05_obj0p005" "insertion_only" "quick" "$SPME_FUSION_CKPT" \
    "SAM3_SPME_FUSION=1" \
    "SAM3_SPME_FUSION_MODE=film" \
    "SAM3_SPME_FUSION_ALPHA=0.05" \
    "SAM3_SPME_FUSION_ALPHA_OBJ=0.005" \
    "SAM3_SPME_FUSION_DET_THR=0.3" \
    "SAM3_SPME_FUSION_QCOS_THR=0.5" \
    "SAM3_SPME_FUSION_QCOS_TEMP=20" \
    "SAM3_SPME_FUSION_QCOS_GATE=sigmoid" \
    "SAM3_SPME_FUSION_USE_PRESENCE=1" \
    "SAM3_SPME_ANCHOR_DET_THR=0.3" \
    "SAM3_SPME_QUERY_POOL=topk_weighted" \
    "SAM3_SPME_QUERY_TOPK=5"

  # Soft-gate sweep: slightly lower qcos center (lets more borderline frames through).
  run_eval "spme_fusion_trained_film_topk5_a0p05_obj0p005_q0p45" "insertion_only" "quick" "$SPME_FUSION_CKPT" \
    "SAM3_SPME_FUSION=1" \
    "SAM3_SPME_FUSION_MODE=film" \
    "SAM3_SPME_FUSION_ALPHA=0.05" \
    "SAM3_SPME_FUSION_ALPHA_OBJ=0.005" \
    "SAM3_SPME_FUSION_DET_THR=0.3" \
    "SAM3_SPME_FUSION_QCOS_THR=0.45" \
    "SAM3_SPME_FUSION_QCOS_TEMP=20" \
    "SAM3_SPME_FUSION_QCOS_GATE=sigmoid" \
    "SAM3_SPME_FUSION_USE_PRESENCE=1" \
    "SAM3_SPME_ANCHOR_DET_THR=0.3" \
    "SAM3_SPME_QUERY_POOL=topk_weighted" \
    "SAM3_SPME_QUERY_TOPK=5"

  # Ablation: Top-1 pointer (vs Top-K).
  run_eval "spme_fusion_trained_film_top1_a0p05_obj0p005" "insertion_only" "quick" "$SPME_FUSION_CKPT" \
    "SAM3_SPME_FUSION=1" \
    "SAM3_SPME_FUSION_MODE=film" \
    "SAM3_SPME_FUSION_ALPHA=0.05" \
    "SAM3_SPME_FUSION_ALPHA_OBJ=0.005" \
    "SAM3_SPME_FUSION_DET_THR=0.3" \
    "SAM3_SPME_FUSION_QCOS_THR=0.5" \
    "SAM3_SPME_FUSION_QCOS_TEMP=20" \
    "SAM3_SPME_FUSION_QCOS_GATE=sigmoid" \
    "SAM3_SPME_FUSION_USE_PRESENCE=1" \
    "SAM3_SPME_ANCHOR_DET_THR=0.3" \
    "SAM3_SPME_QUERY_POOL=top1"

  # Ablation: mem-only (no obj_ptr edit).
  run_eval "spme_fusion_trained_film_topk5_a0p05_obj0" "insertion_only" "quick" "$SPME_FUSION_CKPT" \
    "SAM3_SPME_FUSION=1" \
    "SAM3_SPME_FUSION_MODE=film" \
    "SAM3_SPME_FUSION_ALPHA=0.05" \
    "SAM3_SPME_FUSION_ALPHA_OBJ=0.0" \
    "SAM3_SPME_FUSION_DET_THR=0.3" \
    "SAM3_SPME_FUSION_QCOS_THR=0.5" \
    "SAM3_SPME_FUSION_QCOS_TEMP=20" \
    "SAM3_SPME_FUSION_QCOS_GATE=sigmoid" \
    "SAM3_SPME_FUSION_USE_PRESENCE=1" \
    "SAM3_SPME_ANCHOR_DET_THR=0.3" \
    "SAM3_SPME_QUERY_POOL=topk_weighted" \
    "SAM3_SPME_QUERY_TOPK=5"

  # Ablation: presence off (should be ~no-op on ultrasound but keep for parity with EndoVis).
  run_eval "spme_fusion_trained_film_topk5_presence0" "insertion_only" "quick" "$SPME_FUSION_CKPT" \
    "SAM3_SPME_FUSION=1" \
    "SAM3_SPME_FUSION_MODE=film" \
    "SAM3_SPME_FUSION_ALPHA=0.05" \
    "SAM3_SPME_FUSION_ALPHA_OBJ=0.005" \
    "SAM3_SPME_FUSION_DET_THR=0.3" \
    "SAM3_SPME_FUSION_QCOS_THR=0.5" \
    "SAM3_SPME_FUSION_QCOS_TEMP=20" \
    "SAM3_SPME_FUSION_QCOS_GATE=sigmoid" \
    "SAM3_SPME_FUSION_USE_PRESENCE=0" \
    "SAM3_SPME_ANCHOR_DET_THR=0.3" \
    "SAM3_SPME_QUERY_POOL=topk_weighted" \
    "SAM3_SPME_QUERY_TOPK=5"

  # Ablation: disable qcos soft-gate (lets fusion trigger whenever det_score passes).
  run_eval "spme_fusion_trained_film_topk5_qcos0" "insertion_only" "quick" "$SPME_FUSION_CKPT" \
    "SAM3_SPME_FUSION=1" \
    "SAM3_SPME_FUSION_MODE=film" \
    "SAM3_SPME_FUSION_ALPHA=0.05" \
    "SAM3_SPME_FUSION_ALPHA_OBJ=0.005" \
    "SAM3_SPME_FUSION_DET_THR=0.3" \
    "SAM3_SPME_FUSION_QCOS_THR=0.0" \
    "SAM3_SPME_FUSION_QCOS_TEMP=20" \
    "SAM3_SPME_FUSION_QCOS_GATE=sigmoid" \
    "SAM3_SPME_FUSION_USE_PRESENCE=1" \
    "SAM3_SPME_ANCHOR_DET_THR=0.3" \
    "SAM3_SPME_QUERY_POOL=topk_weighted" \
    "SAM3_SPME_QUERY_TOPK=5"

  # Sanity: per-object pointer ON (should be equivalent for single-object ultrasound).
  run_eval "spme_fusion_trained_film_topk5_per_object1" "insertion_only" "quick" "$SPME_FUSION_CKPT" \
    "SAM3_SPME_PER_OBJECT=1" \
    "SAM3_SPME_FUSION=1" \
    "SAM3_SPME_FUSION_MODE=film" \
    "SAM3_SPME_FUSION_ALPHA=0.05" \
    "SAM3_SPME_FUSION_ALPHA_OBJ=0.005" \
    "SAM3_SPME_FUSION_DET_THR=0.3" \
    "SAM3_SPME_FUSION_QCOS_THR=0.5" \
    "SAM3_SPME_FUSION_QCOS_TEMP=20" \
    "SAM3_SPME_FUSION_QCOS_GATE=sigmoid" \
    "SAM3_SPME_FUSION_USE_PRESENCE=1" \
    "SAM3_SPME_ANCHOR_DET_THR=0.3" \
    "SAM3_SPME_QUERY_POOL=topk_weighted" \
    "SAM3_SPME_QUERY_TOPK=5"

  # Optional: combine with SPME-W (kept as a quick ablation only).
  run_eval "spme_w_soft_blend__spme_fusion_trained_film_topk5" "insertion_only" "quick" "$SPME_FUSION_CKPT" \
    "SAM3_SPME_WRITE_GATE=1" \
    "SAM3_SPME_WRITE_GATE_MODE=soft" \
    "SAM3_SPME_QCOS_THR=0.5" \
    "SAM3_SPME_DET_THR=0.3" \
    "SAM3_SPME_USE_DET_SCORE=1" \
    "SAM3_SPME_WRITE_GATE_APPLY=blend_mem" \
    "SAM3_SPME_FUSION=1" \
    "SAM3_SPME_FUSION_MODE=film" \
    "SAM3_SPME_FUSION_ALPHA=0.05" \
    "SAM3_SPME_FUSION_ALPHA_OBJ=0.005" \
    "SAM3_SPME_FUSION_DET_THR=0.3" \
    "SAM3_SPME_FUSION_QCOS_THR=0.5" \
    "SAM3_SPME_FUSION_QCOS_TEMP=20" \
    "SAM3_SPME_FUSION_QCOS_GATE=sigmoid" \
    "SAM3_SPME_FUSION_USE_PRESENCE=1" \
    "SAM3_SPME_ANCHOR_DET_THR=0.3" \
    "SAM3_SPME_QUERY_POOL=topk_weighted" \
    "SAM3_SPME_QUERY_TOPK=5"
fi

if [[ "$RUN_FULL" == "1" ]]; then
  echo ""
  echo "======================"
  echo "=== FULL EVAL (insertion_only, full test set) ==="
  echo "======================"
  echo ""

  run_eval "baseline" "insertion_only" "full" ""

  run_eval "spme_fusion_trained_film_topk5_a0p05_obj0p005" "insertion_only" "full" "$SPME_FUSION_CKPT" \
    "SAM3_SPME_FUSION=1" \
    "SAM3_SPME_FUSION_MODE=film" \
    "SAM3_SPME_FUSION_ALPHA=0.05" \
    "SAM3_SPME_FUSION_ALPHA_OBJ=0.005" \
    "SAM3_SPME_FUSION_DET_THR=0.3" \
    "SAM3_SPME_FUSION_QCOS_THR=0.5" \
    "SAM3_SPME_FUSION_QCOS_TEMP=20" \
    "SAM3_SPME_FUSION_QCOS_GATE=sigmoid" \
    "SAM3_SPME_FUSION_USE_PRESENCE=1" \
    "SAM3_SPME_ANCHOR_DET_THR=0.3" \
    "SAM3_SPME_QUERY_POOL=topk_weighted" \
    "SAM3_SPME_QUERY_TOPK=5"

  # Extra sanity: per-object pointer ON on the full set (should be equivalent for single-object ultrasound).
  run_eval "spme_fusion_trained_film_topk5_a0p05_obj0p005__per_object1" "insertion_only" "full" "$SPME_FUSION_CKPT" \
    "SAM3_SPME_PER_OBJECT=1" \
    "SAM3_SPME_FUSION=1" \
    "SAM3_SPME_FUSION_MODE=film" \
    "SAM3_SPME_FUSION_ALPHA=0.05" \
    "SAM3_SPME_FUSION_ALPHA_OBJ=0.005" \
    "SAM3_SPME_FUSION_DET_THR=0.3" \
    "SAM3_SPME_FUSION_QCOS_THR=0.5" \
    "SAM3_SPME_FUSION_QCOS_TEMP=20" \
    "SAM3_SPME_FUSION_QCOS_GATE=sigmoid" \
    "SAM3_SPME_FUSION_USE_PRESENCE=1" \
    "SAM3_SPME_ANCHOR_DET_THR=0.3" \
    "SAM3_SPME_QUERY_POOL=topk_weighted" \
    "SAM3_SPME_QUERY_TOPK=5"

  # Optional (expensive): full video eval (uncomment if needed for appendix).
  # run_eval "baseline" "full_video" "full" ""
  # run_eval "spme_fusion_trained_film_topk5_a0p05_obj0p005" "full_video" "full" "$SPME_FUSION_CKPT" \
  #   "SAM3_SPME_FUSION=1" \
  #   "SAM3_SPME_FUSION_MODE=film" \
  #   "SAM3_SPME_FUSION_ALPHA=0.05" \
  #   "SAM3_SPME_FUSION_ALPHA_OBJ=0.005" \
  #   "SAM3_SPME_FUSION_DET_THR=0.3" \
  #   "SAM3_SPME_FUSION_QCOS_THR=0.6" \
  #   "SAM3_SPME_FUSION_USE_PRESENCE=1" \
  #   "SAM3_SPME_ANCHOR_DET_THR=0.3" \
  #   "SAM3_SPME_QUERY_POOL=topk_weighted" \
  #   "SAM3_SPME_QUERY_TOPK=5"
fi

echo "=== Done. Results saved to: $OUT_ROOT ==="
