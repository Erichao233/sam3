#!/bin/bash
#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_endovis2017_eval
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
DATA=/home2020/home/icube/kunyuan/SurgBench/surgicaltool/endovis2017
SAM3_PT=$REPO/sam3.pt
OVERLAY_CKPT="${OVERLAY_CKPT:-}"  # e.g. /path/to/checkpoint_*.pt (optional finetuned overlay)

OUT_ROOT=/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval

cd "$REPO"

# Sequences to evaluate (val1-val10 are the test sequences)
SEQS="val1 val2 val3 val4 val5 val6 val7 val8 val9 val10"

# Generic prompt for all instruments
PROMPT="surgical instrument"

# EndoVis often has domain shift; if you see "IoU=0.000 everywhere", lower thresholds.
NEW_DET_THR="${NEW_DET_THR:-0.3}"
SCORE_THR_DET="${SCORE_THR_DET:-0.5}"
USE_CLASS_PROMPTS="${USE_CLASS_PROMPTS:-1}" # 1 => use per-class prompts (recommended for EndoVis)
INIT_FRAME_MODE="${INIT_FRAME_MODE:-first_present}"  # first_present | first_min_area | max_area
INIT_MIN_AREA="${INIT_MIN_AREA:-0}"
INIT_BOX_PAD="${INIT_BOX_PAD:-0}"

COMMON_ARGS=(
  --new-det-thr "$NEW_DET_THR"
  --score-thr-detection "$SCORE_THR_DET"
  --init-frame-mode "$INIT_FRAME_MODE"
  --init-min-area "$INIT_MIN_AREA"
  --init-box-pad "$INIT_BOX_PAD"
)
if [[ "$USE_CLASS_PROMPTS" == "1" ]]; then
  COMMON_ARGS+=(--class-specific-prompts)
fi
if [[ -n "$OVERLAY_CKPT" ]]; then
  COMMON_ARGS+=(--overlay-ckpt "$OVERLAY_CKPT")
fi

# Which configs to run (teacher focus: baseline + SPME-W + SPME-F => "A B D")
# Options: A (baseline), B (SPME-W), C (SPME-W+SPME-F), D (SPME-F only)
RUN_CONFIGS="${RUN_CONFIGS:-"A B D"}"
echo "RUN_CONFIGS=$RUN_CONFIGS"
echo "NEW_DET_THR=$NEW_DET_THR SCORE_THR_DET=$SCORE_THR_DET USE_CLASS_PROMPTS=$USE_CLASS_PROMPTS INIT_FRAME_MODE=$INIT_FRAME_MODE INIT_MIN_AREA=$INIT_MIN_AREA INIT_BOX_PAD=$INIT_BOX_PAD OVERLAY_CKPT=$OVERLAY_CKPT"

# =============================================================================
# Configuration A: Baseline (no SPME)
# =============================================================================
if [[ " $RUN_CONFIGS " == *" A "* ]]; then
  echo "=== [A] Baseline (no SPME) ==="
  unset SAM3_SPME_WRITE_GATE SAM3_SPME_WRITE_GATE_MODE SAM3_SPME_DET_THR SAM3_SPME_QCOS_THR
  unset SAM3_SPME_USE_DET_SCORE SAM3_SPME_WRITE_GATE_APPLY SAM3_SPME_DEBUG_SIGNALS
  unset SAM3_SPME_FUSION SAM3_SPME_FUSION_ALPHA SAM3_SPME_FUSION_DET_THR SAM3_SPME_ANCHOR_DET_THR

  python scripts/eval_endovis2017.py \
    --data-root "$DATA" \
    --sequences $SEQS \
    --out-dir "$OUT_ROOT/baseline" \
    --base-sam3-pt "$SAM3_PT" \
    --prompt "$PROMPT" \
    "${COMMON_ARGS[@]}"
fi

# =============================================================================
# Configuration B: SPME-W (soft gate + blend_mem)
# =============================================================================
if [[ " $RUN_CONFIGS " == *" B "* ]]; then
  echo "=== [B] SPME-W (soft q0.5 * det_score, blend_mem) ==="
  export SAM3_SPME_WRITE_GATE=1
  export SAM3_SPME_WRITE_GATE_MODE=soft
  export SAM3_SPME_QCOS_THR=0.5
  unset SAM3_SPME_DET_THR
  export SAM3_SPME_USE_DET_SCORE=1
  export SAM3_SPME_WRITE_GATE_APPLY=blend_mem
  unset SAM3_SPME_FUSION SAM3_SPME_FUSION_ALPHA SAM3_SPME_FUSION_DET_THR SAM3_SPME_ANCHOR_DET_THR

  python scripts/eval_endovis2017.py \
    --data-root "$DATA" \
    --sequences $SEQS \
    --out-dir "$OUT_ROOT/spme_w_soft_blend" \
    --base-sam3-pt "$SAM3_PT" \
    --prompt "$PROMPT" \
    "${COMMON_ARGS[@]}"
fi

# =============================================================================
# Configuration C: SPME-W + SPME-F (gate + fusion)
# =============================================================================
if [[ " $RUN_CONFIGS " == *" C "* ]]; then
  echo "=== [C] SPME-W + SPME-F (soft gate + blend_mem + fusion α=0.05) ==="
  export SAM3_SPME_WRITE_GATE=1
  export SAM3_SPME_WRITE_GATE_MODE=soft
  export SAM3_SPME_QCOS_THR=0.5
  unset SAM3_SPME_DET_THR
  export SAM3_SPME_USE_DET_SCORE=1
  export SAM3_SPME_WRITE_GATE_APPLY=blend_mem
  # Enable SPME-F fusion
  export SAM3_SPME_FUSION=1
  export SAM3_SPME_FUSION_ALPHA=0.05
  export SAM3_SPME_FUSION_DET_THR=0.3
  export SAM3_SPME_ANCHOR_DET_THR=0.3

  python scripts/eval_endovis2017.py \
    --data-root "$DATA" \
    --sequences $SEQS \
    --out-dir "$OUT_ROOT/spme_wf_fusion" \
    --base-sam3-pt "$SAM3_PT" \
    --prompt "$PROMPT" \
    "${COMMON_ARGS[@]}"
fi

# =============================================================================
# Configuration D: SPME-F only (fusion without gate, for ablation)
# =============================================================================
if [[ " $RUN_CONFIGS " == *" D "* ]]; then
  echo "=== [D] SPME-F only (fusion α=0.05, no gate) ==="
  unset SAM3_SPME_WRITE_GATE SAM3_SPME_WRITE_GATE_MODE SAM3_SPME_DET_THR SAM3_SPME_QCOS_THR
  unset SAM3_SPME_USE_DET_SCORE SAM3_SPME_WRITE_GATE_APPLY
  export SAM3_SPME_FUSION=1
  export SAM3_SPME_FUSION_ALPHA=0.05
  export SAM3_SPME_FUSION_DET_THR=0.3
  export SAM3_SPME_ANCHOR_DET_THR=0.3

  python scripts/eval_endovis2017.py \
    --data-root "$DATA" \
    --sequences $SEQS \
    --out-dir "$OUT_ROOT/spme_f_only" \
    --base-sam3-pt "$SAM3_PT" \
    --prompt "$PROMPT" \
    "${COMMON_ARGS[@]}"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=== All configurations complete ==="
echo "Results saved to: $OUT_ROOT"
echo ""
echo "Configurations requested: $RUN_CONFIGS"
