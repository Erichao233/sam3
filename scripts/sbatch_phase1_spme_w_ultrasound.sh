#!/bin/bash
#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -J sam3_us_phase1_spme_w
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
CKPT=/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/sam3_us_needle_detection_boxprompt/checkpoints/checkpoint_18.pt
SAMPLE_TXT=$DATA/test.txt
SAMPLE_IDS=18,159,172,107,129

# New output folder for Phase-1 (SPME-W)
OUT_ROOT=/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/phase1_spme_w_v4

cd "$REPO"

echo "=== [Baseline] needle prompt ==="
unset SAM3_SPME_WRITE_GATE SAM3_SPME_WRITE_GATE_MODE SAM3_SPME_DET_THR SAM3_SPME_QCOS_THR SAM3_SPME_USE_DET_SCORE SAM3_SPME_WRITE_GATE_APPLY SAM3_SPME_DEBUG_SIGNALS
python scripts/phase0_eval_ultrasound_signals.py \
  --frames-root "$DATA/pork" \
  --gt-root "$DATA/gt" \
  --sample-ids "$SAMPLE_IDS" \
  --sample-txt "$SAMPLE_TXT" \
  --out-dir "$OUT_ROOT/baseline/needle_prompt" \
  --base-sam3-pt "$SAM3_PT" \
  --finetune-ckpt "$CKPT" \
  --bpe-path "$BPE" \
  --prompt "ultrasound needle" \
  --modes gtbbox_text

echo "=== [Baseline] vessel prompt (negative control) ==="
python scripts/phase0_eval_ultrasound_signals.py \
  --frames-root "$DATA/pork" \
  --gt-root "$DATA/gt" \
  --sample-ids "$SAMPLE_IDS" \
  --sample-txt "$SAMPLE_TXT" \
  --out-dir "$OUT_ROOT/baseline/vessel_prompt" \
  --base-sam3-pt "$SAM3_PT" \
  --finetune-ckpt "$CKPT" \
  --bpe-path "$BPE" \
  --prompt "ultrasound vessel" \
  --modes gtbbox_text

echo "=== [SPME-W soft + blend_mem] qcos ramp from 0.5, * det_score (B) ==="
export SAM3_SPME_WRITE_GATE=1
export SAM3_SPME_WRITE_GATE_MODE=soft
export SAM3_SPME_QCOS_THR=0.5
unset SAM3_SPME_DET_THR
export SAM3_SPME_USE_DET_SCORE=1
export SAM3_SPME_WRITE_GATE_APPLY=blend_mem

echo "=== [SPME-W soft + blend_mem] needle prompt (B) ==="
python scripts/phase0_eval_ultrasound_signals.py \
  --frames-root "$DATA/pork" \
  --gt-root "$DATA/gt" \
  --sample-ids "$SAMPLE_IDS" \
  --sample-txt "$SAMPLE_TXT" \
  --out-dir "$OUT_ROOT/spme_w_soft_blend_q0p5/needle_prompt" \
  --base-sam3-pt "$SAM3_PT" \
  --finetune-ckpt "$CKPT" \
  --bpe-path "$BPE" \
  --prompt "ultrasound needle" \
  --modes gtbbox_text

echo "=== [SPME-W soft + blend_mem] vessel prompt (B) ==="
python scripts/phase0_eval_ultrasound_signals.py \
  --frames-root "$DATA/pork" \
  --gt-root "$DATA/gt" \
  --sample-ids "$SAMPLE_IDS" \
  --sample-txt "$SAMPLE_TXT" \
  --out-dir "$OUT_ROOT/spme_w_soft_blend_q0p5/vessel_prompt" \
  --base-sam3-pt "$SAM3_PT" \
  --finetune-ckpt "$CKPT" \
  --bpe-path "$BPE" \
  --prompt "ultrasound vessel" \
  --modes gtbbox_text

echo "=== [SPME-W soft + blend_mem] qcos_thr=0.5 + det_thr=0.5 (v4 patch) ==="
export SAM3_SPME_WRITE_GATE=1
export SAM3_SPME_WRITE_GATE_MODE=soft
export SAM3_SPME_QCOS_THR=0.5
export SAM3_SPME_DET_THR=0.5
export SAM3_SPME_USE_DET_SCORE=1
export SAM3_SPME_WRITE_GATE_APPLY=blend_mem

echo "=== [SPME-W soft + blend_mem] needle prompt (v4 patch) ==="
python scripts/phase0_eval_ultrasound_signals.py \
  --frames-root "$DATA/pork" \
  --gt-root "$DATA/gt" \
  --sample-ids "$SAMPLE_IDS" \
  --sample-txt "$SAMPLE_TXT" \
  --out-dir "$OUT_ROOT/spme_w_soft_blend_det0p5_q0p5/needle_prompt" \
  --base-sam3-pt "$SAM3_PT" \
  --finetune-ckpt "$CKPT" \
  --bpe-path "$BPE" \
  --prompt "ultrasound needle" \
  --modes gtbbox_text

echo "=== [SPME-W soft + blend_mem] vessel prompt (v4 patch) ==="
python scripts/phase0_eval_ultrasound_signals.py \
  --frames-root "$DATA/pork" \
  --gt-root "$DATA/gt" \
  --sample-ids "$SAMPLE_IDS" \
  --sample-txt "$SAMPLE_TXT" \
  --out-dir "$OUT_ROOT/spme_w_soft_blend_det0p5_q0p5/vessel_prompt" \
  --base-sam3-pt "$SAM3_PT" \
  --finetune-ckpt "$CKPT" \
  --bpe-path "$BPE" \
  --prompt "ultrasound vessel" \
  --modes gtbbox_text

echo "=== Done. Results saved to: $OUT_ROOT ==="
