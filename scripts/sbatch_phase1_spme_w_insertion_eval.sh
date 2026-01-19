#!/bin/bash
#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_us_phase1_spme_w_insert_eval
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
INSERTION_END_TXT=$DATA/insertion_end.txt

# New output folder for Phase-1 (SPME-W) full-test eval.
# This job writes BOTH:
# - full_video: metrics on the entire clip
# - insertion_only: metrics truncated by insertion_end.txt (inclusive)
OUT_ROOT=/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/phase1_spme_w_v5_insert_eval

cd "$REPO"

PROMPT_POS="ultrasound needle"
PROMPT_NEG="ultrasound vessel"

# Optional: enable semantic feature fusion (SPME-F).
# This injects a small residual projected from the detector query into maskmem_features.
# Example:
#   export SAM3_SPME_FUSION=1
#   export SAM3_SPME_FUSION_ALPHA=0.05
#   export SAM3_SPME_FUSION_DET_THR=0.3
#
# Optional: delay anchor selection until the detector is confident:
#   export SAM3_SPME_ANCHOR_DET_THR=0.3

# Optional extra ablation (costs ~1 extra config):
# - B' removes det_score scaling from soft gate (may improve low-det videos like sample 107).
RUN_B_PRIME=1

echo "=== [FULL VIDEO] Baseline (needle prompt) ==="
unset SAM3_SPME_WRITE_GATE SAM3_SPME_WRITE_GATE_MODE SAM3_SPME_DET_THR SAM3_SPME_QCOS_THR SAM3_SPME_USE_DET_SCORE SAM3_SPME_WRITE_GATE_APPLY SAM3_SPME_DEBUG_SIGNALS
python scripts/phase0_eval_ultrasound_signals.py \
  --frames-root "$DATA/pork" \
  --gt-root "$DATA/gt" \
  --sample-txt "$SAMPLE_TXT" \
  --out-dir "$OUT_ROOT/full_video/baseline/needle_prompt" \
  --base-sam3-pt "$SAM3_PT" \
  --finetune-ckpt "$CKPT" \
  --bpe-path "$BPE" \
  --prompt "$PROMPT_POS" \
  --modes gtbbox_text

echo "=== [FULL VIDEO] SPME-W (B: soft+blend_mem q0.5 * det_score) (needle prompt) ==="
export SAM3_SPME_WRITE_GATE=1
export SAM3_SPME_WRITE_GATE_MODE=soft
export SAM3_SPME_QCOS_THR=0.5
unset SAM3_SPME_DET_THR
export SAM3_SPME_USE_DET_SCORE=1
export SAM3_SPME_WRITE_GATE_APPLY=blend_mem
python scripts/phase0_eval_ultrasound_signals.py \
  --frames-root "$DATA/pork" \
  --gt-root "$DATA/gt" \
  --sample-txt "$SAMPLE_TXT" \
  --out-dir "$OUT_ROOT/full_video/spme_w_soft_blend_q0p5/needle_prompt" \
  --base-sam3-pt "$SAM3_PT" \
  --finetune-ckpt "$CKPT" \
  --bpe-path "$BPE" \
  --prompt "$PROMPT_POS" \
  --modes gtbbox_text

if [[ "$RUN_B_PRIME" == "1" ]]; then
  echo "=== [FULL VIDEO] SPME-W (B': soft+blend_mem q0.5, no det_score scaling) (needle prompt) ==="
  export SAM3_SPME_WRITE_GATE=1
  export SAM3_SPME_WRITE_GATE_MODE=soft
  export SAM3_SPME_QCOS_THR=0.5
  unset SAM3_SPME_DET_THR
  export SAM3_SPME_USE_DET_SCORE=0
  export SAM3_SPME_WRITE_GATE_APPLY=blend_mem
  python scripts/phase0_eval_ultrasound_signals.py \
    --frames-root "$DATA/pork" \
    --gt-root "$DATA/gt" \
    --sample-txt "$SAMPLE_TXT" \
    --out-dir "$OUT_ROOT/full_video/spme_w_soft_blend_q0p5_nodet/needle_prompt" \
    --base-sam3-pt "$SAM3_PT" \
    --finetune-ckpt "$CKPT" \
    --bpe-path "$BPE" \
    --prompt "$PROMPT_POS" \
    --modes gtbbox_text
fi

echo "=== [INSERTION ONLY] Baseline (needle prompt) ==="
unset SAM3_SPME_WRITE_GATE SAM3_SPME_WRITE_GATE_MODE SAM3_SPME_DET_THR SAM3_SPME_QCOS_THR SAM3_SPME_USE_DET_SCORE SAM3_SPME_WRITE_GATE_APPLY SAM3_SPME_DEBUG_SIGNALS
python scripts/phase0_eval_ultrasound_signals.py \
  --frames-root "$DATA/pork" \
  --gt-root "$DATA/gt" \
  --sample-txt "$SAMPLE_TXT" \
  --insertion-end-txt "$INSERTION_END_TXT" \
  --out-dir "$OUT_ROOT/insertion_only/baseline/needle_prompt" \
  --base-sam3-pt "$SAM3_PT" \
  --finetune-ckpt "$CKPT" \
  --bpe-path "$BPE" \
  --prompt "$PROMPT_POS" \
  --modes gtbbox_text

echo "=== [INSERTION ONLY] SPME-W (B: soft+blend_mem q0.5 * det_score) (needle prompt) ==="
export SAM3_SPME_WRITE_GATE=1
export SAM3_SPME_WRITE_GATE_MODE=soft
export SAM3_SPME_QCOS_THR=0.5
unset SAM3_SPME_DET_THR
export SAM3_SPME_USE_DET_SCORE=1
export SAM3_SPME_WRITE_GATE_APPLY=blend_mem
python scripts/phase0_eval_ultrasound_signals.py \
  --frames-root "$DATA/pork" \
  --gt-root "$DATA/gt" \
  --sample-txt "$SAMPLE_TXT" \
  --insertion-end-txt "$INSERTION_END_TXT" \
  --out-dir "$OUT_ROOT/insertion_only/spme_w_soft_blend_q0p5/needle_prompt" \
  --base-sam3-pt "$SAM3_PT" \
  --finetune-ckpt "$CKPT" \
  --bpe-path "$BPE" \
  --prompt "$PROMPT_POS" \
  --modes gtbbox_text

if [[ "$RUN_B_PRIME" == "1" ]]; then
  echo "=== [INSERTION ONLY] SPME-W (B': soft+blend_mem q0.5, no det_score scaling) (needle prompt) ==="
  export SAM3_SPME_WRITE_GATE=1
  export SAM3_SPME_WRITE_GATE_MODE=soft
  export SAM3_SPME_QCOS_THR=0.5
  unset SAM3_SPME_DET_THR
  export SAM3_SPME_USE_DET_SCORE=0
  export SAM3_SPME_WRITE_GATE_APPLY=blend_mem
  python scripts/phase0_eval_ultrasound_signals.py \
    --frames-root "$DATA/pork" \
    --gt-root "$DATA/gt" \
    --sample-txt "$SAMPLE_TXT" \
    --insertion-end-txt "$INSERTION_END_TXT" \
    --out-dir "$OUT_ROOT/insertion_only/spme_w_soft_blend_q0p5_nodet/needle_prompt" \
    --base-sam3-pt "$SAM3_PT" \
    --finetune-ckpt "$CKPT" \
    --bpe-path "$BPE" \
    --prompt "$PROMPT_POS" \
    --modes gtbbox_text
fi

# Optional negative control (uncomment if time allows)
#
# echo "=== [FULL VIDEO] Baseline (vessel prompt, negative control) ==="
# unset SAM3_SPME_WRITE_GATE SAM3_SPME_WRITE_GATE_MODE SAM3_SPME_DET_THR SAM3_SPME_QCOS_THR SAM3_SPME_USE_DET_SCORE SAM3_SPME_WRITE_GATE_APPLY SAM3_SPME_DEBUG_SIGNALS
# python scripts/phase0_eval_ultrasound_signals.py \
#   --frames-root "$DATA/pork" \
#   --gt-root "$DATA/gt" \
#   --sample-txt "$SAMPLE_TXT" \
#   --out-dir "$OUT_ROOT/full_video/baseline/vessel_prompt" \
#   --base-sam3-pt "$SAM3_PT" \
#   --finetune-ckpt "$CKPT" \
#   --bpe-path "$BPE" \
#   --prompt "$PROMPT_NEG" \
#   --modes gtbbox_text
#
# echo "=== [FULL VIDEO] SPME-W (B) (vessel prompt, negative control) ==="
# export SAM3_SPME_WRITE_GATE=1
# export SAM3_SPME_WRITE_GATE_MODE=soft
# export SAM3_SPME_QCOS_THR=0.5
# unset SAM3_SPME_DET_THR
# export SAM3_SPME_USE_DET_SCORE=1
# export SAM3_SPME_WRITE_GATE_APPLY=blend_mem
# python scripts/phase0_eval_ultrasound_signals.py \
#   --frames-root "$DATA/pork" \
#   --gt-root "$DATA/gt" \
#   --sample-txt "$SAMPLE_TXT" \
#   --out-dir "$OUT_ROOT/full_video/spme_w_soft_blend_q0p5/vessel_prompt" \
#   --base-sam3-pt "$SAM3_PT" \
#   --finetune-ckpt "$CKPT" \
#   --bpe-path "$BPE" \
#   --prompt "$PROMPT_NEG" \
#   --modes gtbbox_text
#
# echo "=== [INSERTION ONLY] Baseline (vessel prompt, negative control) ==="
# unset SAM3_SPME_WRITE_GATE SAM3_SPME_WRITE_GATE_MODE SAM3_SPME_DET_THR SAM3_SPME_QCOS_THR SAM3_SPME_USE_DET_SCORE SAM3_SPME_WRITE_GATE_APPLY SAM3_SPME_DEBUG_SIGNALS
# python scripts/phase0_eval_ultrasound_signals.py \
#   --frames-root "$DATA/pork" \
#   --gt-root "$DATA/gt" \
#   --sample-txt "$SAMPLE_TXT" \
#   --insertion-end-txt "$INSERTION_END_TXT" \
#   --out-dir "$OUT_ROOT/insertion_only/baseline/vessel_prompt" \
#   --base-sam3-pt "$SAM3_PT" \
#   --finetune-ckpt "$CKPT" \
#   --bpe-path "$BPE" \
#   --prompt "$PROMPT_NEG" \
#   --modes gtbbox_text
#
# echo "=== [INSERTION ONLY] SPME-W (B) (vessel prompt, negative control) ==="
# export SAM3_SPME_WRITE_GATE=1
# export SAM3_SPME_WRITE_GATE_MODE=soft
# export SAM3_SPME_QCOS_THR=0.5
# unset SAM3_SPME_DET_THR
# export SAM3_SPME_USE_DET_SCORE=1
# export SAM3_SPME_WRITE_GATE_APPLY=blend_mem
# python scripts/phase0_eval_ultrasound_signals.py \
#   --frames-root "$DATA/pork" \
#   --gt-root "$DATA/gt" \
#   --sample-txt "$SAMPLE_TXT" \
#   --insertion-end-txt "$INSERTION_END_TXT" \
#   --out-dir "$OUT_ROOT/insertion_only/spme_w_soft_blend_q0p5/vessel_prompt" \
#   --base-sam3-pt "$SAM3_PT" \
#   --finetune-ckpt "$CKPT" \
#   --bpe-path "$BPE" \
#   --prompt "$PROMPT_NEG" \
#   --modes gtbbox_text

echo "=== Done. Results saved to: $OUT_ROOT ==="
