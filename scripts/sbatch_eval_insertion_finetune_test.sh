#!/bin/bash
#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_us_insertion_ft_eval
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
CKPT_NEW="$(ls -t "$CKPT_NEW_DIR"/checkpoint_*.pt 2>/dev/null | head -n 1 || true)"
if [[ -z "$CKPT_NEW" ]]; then
  echo "[error] No checkpoint found in: $CKPT_NEW_DIR"
  exit 1
fi

OUT_ROOT=/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/phase1_spme_w_v6_detector_finetune_eval

cd "$REPO"

PROMPT_POS="ultrasound needle"

run_eval () {
  local tag="$1"
  local ckpt="$2"
  local scope="$3"  # full_video | insertion_only

  echo "=== [$scope] $tag ==="
  unset SAM3_SPME_WRITE_GATE SAM3_SPME_WRITE_GATE_MODE SAM3_SPME_DET_THR SAM3_SPME_QCOS_THR SAM3_SPME_USE_DET_SCORE SAM3_SPME_WRITE_GATE_APPLY SAM3_SPME_DEBUG_SIGNALS

  if [[ "$scope" == "insertion_only" ]]; then
    python scripts/phase0_eval_ultrasound_signals.py \
      --frames-root "$DATA/pork" \
      --gt-root "$DATA/gt" \
      --sample-txt "$SAMPLE_TXT" \
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
      --sample-txt "$SAMPLE_TXT" \
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

run_eval "baseline_ckpt18" "$CKPT_BASE" "full_video"
run_eval "insertion_ft_latest" "$CKPT_NEW" "full_video"
run_eval "baseline_ckpt18" "$CKPT_BASE" "insertion_only"
run_eval "insertion_ft_latest" "$CKPT_NEW" "insertion_only"

echo "=== Done. Results saved to: $OUT_ROOT ==="
