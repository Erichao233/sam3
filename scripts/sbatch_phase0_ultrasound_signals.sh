#!/bin/bash
#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -J sam3_us_phase0_query
#SBATCH -o /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.out
#SBATCH -e /home2020/home/icube/kunyuan/SurgBench/SAM/logs/%x-%j.err
#SBATCH --exclude=hpc-n968

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
CKPT=/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/sam3_us_needle_detection_boxprompt/checkpoints/checkpoint_18.pt

# New output folder with query_cos support
OUT=/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/phase0_signals_query_cos

# Use test.txt for all test samples (adjust path if needed)
SAMPLE_TXT=$DATA/test.txt

cd "$REPO"

echo "=== Running with CORRECT prompt: ultrasound needle ==="
python scripts/phase0_eval_ultrasound_signals.py \
  --frames-root "$DATA/pork" \
  --gt-root "$DATA/gt" \
  --sample-txt "$SAMPLE_TXT" \
  --out-dir "$OUT/needle_prompt" \
  --base-sam3-pt "$SAM3_PT" \
  --finetune-ckpt "$CKPT" \
  --bpe-path "$BPE" \
  --prompt "ultrasound needle" \
  --modes gtbbox_text

echo "=== Running with WRONG prompt: ultrasound vessel ==="
python scripts/phase0_eval_ultrasound_signals.py \
  --frames-root "$DATA/pork" \
  --gt-root "$DATA/gt" \
  --sample-txt "$SAMPLE_TXT" \
  --out-dir "$OUT/vessel_prompt" \
  --base-sam3-pt "$SAM3_PT" \
  --finetune-ckpt "$CKPT" \
  --bpe-path "$BPE" \
  --prompt "ultrasound vessel" \
  --modes gtbbox_text

echo "=== Done. Results saved to: $OUT ==="

