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
export PYTHONUNBUFFERED=1

# Gate MLP width (must match the checkpoint). Read at model construction time.
export SAM3_SPME_GATE_HIDDEN="${SAM3_SPME_GATE_HIDDEN:-64}"

REPO=/home2020/home/icube/kunyuan/SurgBench/SAM/sam3
DATA=/home2020/home/icube/kunyuan/SurgBench/surgicaltool/endovis2017
# Official EndoVis2017 release (raw). Used only when PREPARE_DATA is enabled.
ENDOVIS_SRC="${ENDOVIS_SRC:-/home2020/home/icube/kunyuan/SurgBench/surgicaltool/Endovis2017}"
PREPARE_DATA="${PREPARE_DATA:-auto}"  # 0 | 1 | auto
CAMERA="${CAMERA:-left}"  # left | right
SAM3_PT=$REPO/sam3.pt
# Optional: set OVERLAY_CKPT to a finetuned checkpoint (or set it to empty to disable overlay)
OVERLAY_CKPT="${OVERLAY_CKPT-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/sam3_endovis2017_ft_boxprompt/checkpoints/checkpoint_40.pt}"
# Optional: trained SPME checkpoints (load only spme_* keys).
# - FUSION_CKPT: spme_fusion_latest.pt
# - GATE_CKPT: spme_gate_latest.pt (contains spme_gate_mlp + fusion projectors)
FUSION_CKPT="${FUSION_CKPT:-}"
GATE_CKPT="${GATE_CKPT:-}"

OUT_ROOT="${OUT_ROOT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2017_spme_eval/job_${SLURM_JOB_ID:-local}}"

cd "$REPO"

if [[ "$PREPARE_DATA" == "1" || ( "$PREPARE_DATA" == "auto" && ! -d "$DATA/val1/image" ) ]]; then
  echo "=== [0/1] Prepare official EndoVis2017 -> canonical layout ==="
  if [[ ! -d "$ENDOVIS_SRC" ]]; then
    echo "[error] ENDOVIS_SRC not found: $ENDOVIS_SRC"
    exit 1
  fi
  python -u scripts/prepare_endovis2017_official.py \
    --src-root "$ENDOVIS_SRC" \
    --out-root "$DATA" \
    --camera "$CAMERA" \
    --overwrite
fi
if [[ ! -d "$DATA/val1/image" ]]; then
  echo "[error] Missing processed dataset folder: $DATA/val1/image"
  exit 1
fi

# Sequences to evaluate (val1-val10 are the test sequences).
# NOTE: val9 is known to have dataset issues in some releases; exclude by default.
SEQS="${SEQS:-"val1 val2 val3 val4 val5 val6 val7 val8 val10"}"

# Prompting:
# - For EndoVis, class-name text prompts (e.g., "bipolar forceps") often fail and can yield IoU=0 due to init failure.
# - Recommended: use `PROMPT=visual` (no language) and keep `USE_CLASS_PROMPTS=0`.
PROMPT="${PROMPT:-"visual"}"

# EndoVis often has domain shift; if you see "IoU=0.000 everywhere", lower thresholds.
NEW_DET_THR="${NEW_DET_THR:-0.3}"
SCORE_THR_DET="${SCORE_THR_DET:-0.5}"
USE_CLASS_PROMPTS="${USE_CLASS_PROMPTS:-0}" # 1 => use per-class prompts (often brittle on EndoVis)
INIT_PROMPT="${INIT_PROMPT:-mask}"  # mask | box (mask aligns with "first visible mask" init; uses GT mask in eval)
INIT_FRAME_MODE="${INIT_FRAME_MODE:-first_present}"  # first_present | first_min_area | max_area
INIT_SELECT_MODE="${INIT_SELECT_MODE:-prob}"  # prob | bbox_iou | gt_iou (debug only)
INIT_MIN_AREA="${INIT_MIN_AREA:-0}"
INIT_BOX_PAD="${INIT_BOX_PAD:-0}"
CLASSES="${CLASSES:-}"          # e.g. "3" or "1 2 3"
DEBUG_INIT="${DEBUG_INIT:-0}"  # 1 => print init candidate stats
HOTSTART_DELAY="${HOTSTART_DELAY:-}"  # e.g. 0 to disable hotstart buffering/filtering
WRITE_FUSED_MASKS="${WRITE_FUSED_MASKS:-0}"  # 1 => also dump fused label masks as PNGs (large)

# Identity-safe tracking: do NOT allow new objects to spawn from text after init box.
export SAM3_ALLOW_NEW_DETECTIONS="${SAM3_ALLOW_NEW_DETECTIONS:-1}"
export SAM3_ALLOW_NEW_DETECTIONS_WITH_TEXT="${SAM3_ALLOW_NEW_DETECTIONS_WITH_TEXT:-0}"

# Recondition ablation (optional):
# - export SAM3_DISABLE_RECONDITION=1 to disable periodic detector→tracker overwrite
# - or export SAM3_RECONDITION_EVERY_NTH_FRAME=N to override frequency
export SAM3_DISABLE_RECONDITION="${SAM3_DISABLE_RECONDITION:-0}"
export SAM3_RECONDITION_EVERY_NTH_FRAME="${SAM3_RECONDITION_EVERY_NTH_FRAME:-}"

# Learned gate knobs (used for configs E/F).
GATE_USE_DECAY="${GATE_USE_DECAY:-1}"
GATE_OCC_NORM="${GATE_OCC_NORM:-10.0}"
GATE_INPUTS="${GATE_INPUTS:-full}"                 # full | det3 | det4 (must match gate ckpt)
GATE_FUSION_HEAD="${GATE_FUSION_HEAD:-0}"          # 0 | 1 (must match gate ckpt)
GATE_DET_PRESENT_THR="${GATE_DET_PRESENT_THR:-0.3}" # only relevant for det4

COMMON_ARGS=(
  --new-det-thr "$NEW_DET_THR"
  --score-thr-detection "$SCORE_THR_DET"
  --init-prompt "$INIT_PROMPT"
  --init-frame-mode "$INIT_FRAME_MODE"
  --init-select-mode "$INIT_SELECT_MODE"
  --init-min-area "$INIT_MIN_AREA"
  --init-box-pad "$INIT_BOX_PAD"
)
if [[ "$USE_CLASS_PROMPTS" == "1" ]]; then
  COMMON_ARGS+=(--class-specific-prompts)
fi
if [[ -n "$CLASSES" ]]; then
  COMMON_ARGS+=(--classes $CLASSES)
fi
if [[ "$DEBUG_INIT" == "1" ]]; then
  COMMON_ARGS+=(--debug-init)
fi
if [[ "$WRITE_FUSED_MASKS" == "1" ]]; then
  COMMON_ARGS+=(--write-fused-masks)
fi
if [[ -n "$HOTSTART_DELAY" ]]; then
  COMMON_ARGS+=(--hotstart-delay "$HOTSTART_DELAY")
fi
if [[ -n "$OVERLAY_CKPT" ]]; then
  COMMON_ARGS+=(--overlay-ckpt "$OVERLAY_CKPT")
fi

# Which configs to run.
# Options:
# - A: baseline
# - B: SPME-W (heuristic qcos gate)
# - C: SPME-W + SPME-F (heuristic gate + fusion)
# - D: SPME-F only (fusion)
# - E: Learned gate only (no fusion injection; still uses SAM3_SPME_FUSION=1 for signals)
# - F: Learned gate + fusion injection
RUN_CONFIGS="${RUN_CONFIGS:-"A B D E F"}"
echo "RUN_CONFIGS=$RUN_CONFIGS"
echo "SEQS=$SEQS"
echo "NEW_DET_THR=$NEW_DET_THR SCORE_THR_DET=$SCORE_THR_DET USE_CLASS_PROMPTS=$USE_CLASS_PROMPTS INIT_PROMPT=$INIT_PROMPT INIT_FRAME_MODE=$INIT_FRAME_MODE INIT_SELECT_MODE=$INIT_SELECT_MODE INIT_MIN_AREA=$INIT_MIN_AREA INIT_BOX_PAD=$INIT_BOX_PAD HOTSTART_DELAY=$HOTSTART_DELAY CLASSES=$CLASSES DEBUG_INIT=$DEBUG_INIT OVERLAY_CKPT=$OVERLAY_CKPT"
echo "FUSION_CKPT=${FUSION_CKPT:-<none>} GATE_CKPT=${GATE_CKPT:-<none>} SAM3_SPME_GATE_HIDDEN=$SAM3_SPME_GATE_HIDDEN"
echo "SAM3_DISABLE_RECONDITION=$SAM3_DISABLE_RECONDITION SAM3_RECONDITION_EVERY_NTH_FRAME=${SAM3_RECONDITION_EVERY_NTH_FRAME:-<default>}"
echo "GATE_INPUTS=$GATE_INPUTS GATE_FUSION_HEAD=$GATE_FUSION_HEAD GATE_DET_PRESENT_THR=$GATE_DET_PRESENT_THR"

# =============================================================================
# Configuration A: Baseline (no SPME)
# =============================================================================
if [[ " $RUN_CONFIGS " == *" A "* ]]; then
  echo "=== [A] Baseline (no SPME) ==="
  unset SAM3_SPME_WRITE_GATE SAM3_SPME_WRITE_GATE_MODE SAM3_SPME_DET_THR SAM3_SPME_QCOS_THR
  unset SAM3_SPME_USE_DET_SCORE SAM3_SPME_WRITE_GATE_APPLY SAM3_SPME_DEBUG_SIGNALS
  unset SAM3_SPME_FUSION SAM3_SPME_FUSION_ALPHA SAM3_SPME_FUSION_DET_THR SAM3_SPME_ANCHOR_DET_THR
  unset SAM3_SPME_PER_OBJECT
  unset SAM3_SPME_KEEP_QUERIES
  unset SAM3_SPME_LEARNED_GATE SAM3_SPME_LEARNED_GATE_USE_DECAY SAM3_SPME_LEARNED_GATE_OCCLUDED_NORM SAM3_SPME_LEARNED_GATE_LOG

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
  unset SAM3_SPME_PER_OBJECT
  unset SAM3_SPME_KEEP_QUERIES
  unset SAM3_SPME_LEARNED_GATE SAM3_SPME_LEARNED_GATE_USE_DECAY SAM3_SPME_LEARNED_GATE_OCCLUDED_NORM SAM3_SPME_LEARNED_GATE_LOG

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
  export SAM3_SPME_PER_OBJECT=1
  export SAM3_SPME_KEEP_QUERIES=1
  export SAM3_SPME_FUSION=1
  export SAM3_SPME_FUSION_ALPHA=0.05
  export SAM3_SPME_FUSION_DET_THR=0.3
  export SAM3_SPME_ANCHOR_DET_THR=0.3
  unset SAM3_SPME_LEARNED_GATE SAM3_SPME_LEARNED_GATE_USE_DECAY SAM3_SPME_LEARNED_GATE_OCCLUDED_NORM SAM3_SPME_LEARNED_GATE_LOG

  python scripts/eval_endovis2017.py \
    --data-root "$DATA" \
    --sequences $SEQS \
    --out-dir "$OUT_ROOT/spme_wf_fusion" \
    --base-sam3-pt "$SAM3_PT" \
    ${FUSION_CKPT:+--spme-fusion-ckpt "$FUSION_CKPT"} \
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
  export SAM3_SPME_PER_OBJECT=1
  export SAM3_SPME_KEEP_QUERIES=1
  export SAM3_SPME_FUSION=1
  export SAM3_SPME_FUSION_ALPHA=0.05
  export SAM3_SPME_FUSION_DET_THR=0.3
  export SAM3_SPME_ANCHOR_DET_THR=0.3
  unset SAM3_SPME_LEARNED_GATE SAM3_SPME_LEARNED_GATE_USE_DECAY SAM3_SPME_LEARNED_GATE_OCCLUDED_NORM SAM3_SPME_LEARNED_GATE_LOG

  python scripts/eval_endovis2017.py \
    --data-root "$DATA" \
    --sequences $SEQS \
    --out-dir "$OUT_ROOT/spme_f_only" \
    --base-sam3-pt "$SAM3_PT" \
    ${FUSION_CKPT:+--spme-fusion-ckpt "$FUSION_CKPT"} \
    --prompt "$PROMPT" \
    "${COMMON_ARGS[@]}"
fi

# =============================================================================
# Configuration E: Learned Gate only (no fusion injection)
# =============================================================================
if [[ " $RUN_CONFIGS " == *" E "* ]]; then
  if [[ -z "$GATE_CKPT" ]]; then
    echo "[E] Skipping learned gate eval because GATE_CKPT is empty."
  else
    echo "=== [E] Learned gate only (decay=$GATE_USE_DECAY occ_norm=$GATE_OCC_NORM) ==="
    unset SAM3_SPME_WRITE_GATE SAM3_SPME_WRITE_GATE_MODE SAM3_SPME_DET_THR SAM3_SPME_QCOS_THR
    unset SAM3_SPME_USE_DET_SCORE SAM3_SPME_WRITE_GATE_APPLY
    export SAM3_SPME_PER_OBJECT=1
    export SAM3_SPME_KEEP_QUERIES=1
    export SAM3_SPME_FUSION=1
    export SAM3_SPME_FUSION_ALPHA=0.0
    export SAM3_SPME_FUSION_ALPHA_OBJ=0.0
    export SAM3_SPME_ANCHOR_DET_THR=0.3

    export SAM3_SPME_LEARNED_GATE=1
    export SAM3_SPME_LEARNED_GATE_USE_DECAY="$GATE_USE_DECAY"
    export SAM3_SPME_LEARNED_GATE_OCCLUDED_NORM="$GATE_OCC_NORM"
    export SAM3_SPME_LEARNED_GATE_INPUTS="$GATE_INPUTS"
    export SAM3_SPME_LEARNED_GATE_FUSION_HEAD="$GATE_FUSION_HEAD"
    export SAM3_SPME_LEARNED_GATE_DET_PRESENT_THR="$GATE_DET_PRESENT_THR"

    python scripts/eval_endovis2017.py \
      --data-root "$DATA" \
      --sequences $SEQS \
      --out-dir "$OUT_ROOT/learned_gate_only" \
      --base-sam3-pt "$SAM3_PT" \
      --spme-fusion-ckpt "$GATE_CKPT" \
      --prompt "$PROMPT" \
      "${COMMON_ARGS[@]}"
  fi
fi

# =============================================================================
# Configuration F: Learned Gate + Fusion injection
# =============================================================================
if [[ " $RUN_CONFIGS " == *" F "* ]]; then
  if [[ -z "$GATE_CKPT" ]]; then
    echo "[F] Skipping learned gate+fusion eval because GATE_CKPT is empty."
  else
    echo "=== [F] Learned gate + fusion (alpha=0.05) ==="
    unset SAM3_SPME_WRITE_GATE SAM3_SPME_WRITE_GATE_MODE SAM3_SPME_DET_THR SAM3_SPME_QCOS_THR
    unset SAM3_SPME_USE_DET_SCORE SAM3_SPME_WRITE_GATE_APPLY
    export SAM3_SPME_PER_OBJECT=1
    export SAM3_SPME_KEEP_QUERIES=1
    export SAM3_SPME_FUSION=1
    export SAM3_SPME_FUSION_MODE="${SAM3_SPME_FUSION_MODE:-film}"
    export SAM3_SPME_FUSION_ALPHA=0.05
    export SAM3_SPME_FUSION_ALPHA_OBJ=0.005
    export SAM3_SPME_FUSION_DET_THR=0.3
    export SAM3_SPME_ANCHOR_DET_THR=0.3

    export SAM3_SPME_LEARNED_GATE=1
    export SAM3_SPME_LEARNED_GATE_USE_DECAY="$GATE_USE_DECAY"
    export SAM3_SPME_LEARNED_GATE_OCCLUDED_NORM="$GATE_OCC_NORM"
    export SAM3_SPME_LEARNED_GATE_INPUTS="$GATE_INPUTS"
    export SAM3_SPME_LEARNED_GATE_FUSION_HEAD="$GATE_FUSION_HEAD"
    export SAM3_SPME_LEARNED_GATE_DET_PRESENT_THR="$GATE_DET_PRESENT_THR"

    python scripts/eval_endovis2017.py \
      --data-root "$DATA" \
      --sequences $SEQS \
      --out-dir "$OUT_ROOT/learned_gate_fusion" \
      --base-sam3-pt "$SAM3_PT" \
      --spme-fusion-ckpt "$GATE_CKPT" \
      --prompt "$PROMPT" \
      "${COMMON_ARGS[@]}"
  fi
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=== All configurations complete ==="
echo "Results saved to: $OUT_ROOT"
echo ""
echo "Configurations requested: $RUN_CONFIGS"

tar -czf "${OUT_ROOT}.tar.gz" -C "$(dirname "$OUT_ROOT")" "$(basename "$OUT_ROOT")"
echo "Packed: ${OUT_ROOT}.tar.gz"
