#!/bin/bash
# EndoVis2018 evaluation entrypoint (mirrors sbatch_eval_endovis2017.sh).
#
# Notes:
# - Defaults to a MICCAI-safe protocol: PROMPT=visual, INIT_PROMPT=mask, INIT_FRAME_MODE=first_present.
# - No OVERLAY_CKPT by default.
# - If your EndoVis2018 root contains a top-level `test_data/` folder, it is excluded by default to avoid
#   seq-id collisions. Set INCLUDE_TEST_DATA=1 to include it in preprocessing.

#SBATCH -A qoscammagpu2
#SBATCH -p pri2021gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J sam3_endovis2018_eval
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

# Match the joint clip16/mem15 models by default (override if needed).
export SAM3_TRACKER_NUM_MASKMEM="${SAM3_TRACKER_NUM_MASKMEM:-15}"

# Gate MLP width (must match the checkpoint). Read at model construction time.
export SAM3_SPME_GATE_HIDDEN="${SAM3_SPME_GATE_HIDDEN:-64}"
# Multi-class fusion weight mode (used when fusing per-class runs into a single label mask).
export ENDOVIS_FUSION_WEIGHT_MODE="${ENDOVIS_FUSION_WEIGHT_MODE:-eff_iou}"
# Optional: log SPME debug signals (det_score/qcos/fusion scales) into per-frame JSON outputs.
LOG_SIGNALS="${LOG_SIGNALS:-1}"
export SAM3_SPME_LOG_SIGNALS="$LOG_SIGNALS"

REPO=/home2020/home/icube/kunyuan/SurgBench/SAM/sam3
# Canonical processed dataset root (lowercase).
DATA="${DATA:-/home2020/home/icube/kunyuan/SurgBench/surgicaltool/endovis2018}"
# Official EndoVis2018 raw release root.
ENDOVIS_SRC="${ENDOVIS_SRC:-/home2020/home/icube/kunyuan/SurgBench/surgicaltool/Endovis2018}"
PREPARE_DATA="${PREPARE_DATA:-auto}"  # 0 | 1 | auto
INCLUDE_TEST_DATA="${INCLUDE_TEST_DATA:-0}"
# Optional: restrict to specific group folders (space-separated)
#
# NOTE: Do NOT use the name `GROUPS` here — in bash it is a special array containing UNIX group IDs.
ENDOVIS_GROUPS="${ENDOVIS_GROUPS:-}"
SEQ_IDS="${SEQ_IDS:-}" # Optional: restrict to seq ids (space-separated ints)
LABELS_JSON="${LABELS_JSON:-}" # Optional: explicit labels.json path
CAMERA="${CAMERA:-left}"  # left | right

SAM3_PT=$REPO/sam3.pt

# Optional: set OVERLAY_CKPT to a finetuned checkpoint (or set it to empty to disable overlay).
OVERLAY_CKPT="${OVERLAY_CKPT:-}"

# Optional: trained SPME checkpoints (load only spme_* keys).
FUSION_CKPT="${FUSION_CKPT:-}"
GATE_CKPT="${GATE_CKPT:-}"

OUT_ROOT="${OUT_ROOT:-/home2020/home/icube/kunyuan/SurgBench/SAM/outputs/endovis2018_spme_eval/job_${SLURM_JOB_ID:-local}}"

cd "$REPO"

# Determine whether the canonical dataset is already prepared. For eval we only need `val*/image`,
# not the optional flat `train/` view (which may be disabled for official test splits).
HAS_VAL_DIRS="0"
if compgen -G "$DATA/val*/image" > /dev/null; then
  HAS_VAL_DIRS="1"
fi

if [[ "$PREPARE_DATA" == "1" || ( "$PREPARE_DATA" == "auto" && "$HAS_VAL_DIRS" == "0" ) ]]; then
  echo "=== [0/1] Prepare official EndoVis2018 -> canonical layout ==="
  if [[ ! -d "$ENDOVIS_SRC" ]]; then
    echo "[error] ENDOVIS_SRC not found: $ENDOVIS_SRC"
    exit 1
  fi

  PREP_ARGS=(
    --src-root "$ENDOVIS_SRC"
    --out-root "$DATA"
    --camera "$CAMERA"
    --symlink-images
    --overwrite
  )
  if [[ "$INCLUDE_TEST_DATA" == "1" ]]; then
    PREP_ARGS+=(--include-test-data)
  fi
  if [[ -n "${ENDOVIS_GROUPS:-}" ]]; then
    PREP_ARGS+=(--groups $ENDOVIS_GROUPS)
  fi
  if [[ -n "${SEQ_IDS:-}" ]]; then
    PREP_ARGS+=(--seq-ids $SEQ_IDS)
  fi
  if [[ -n "${LABELS_JSON:-}" ]]; then
    PREP_ARGS+=(--labels-json "$LABELS_JSON")
  fi

  python -u scripts/prepare_endovis2018_official.py "${PREP_ARGS[@]}"
fi

if ! compgen -G "$DATA/val*/image" > /dev/null; then
  echo "[error] Missing processed dataset folders under: $DATA/val*/image"
  exit 1
fi

# Sequences to evaluate. If not provided, auto-detect val* folders.
if [[ -z "${SEQS:-}" ]]; then
  if [[ -d "$DATA" ]]; then
    SEQS="$(cd "$DATA" && ls -d val* 2>/dev/null | sort -V | tr '\n' ' ')"
  fi
fi
SEQS="${SEQS:-val1}"

# Prompting defaults: visual-only (no language).
PROMPT="${PROMPT:-visual}"

# Domain shift knobs (keep conservative defaults).
NEW_DET_THR="${NEW_DET_THR:-0.3}"
SCORE_THR_DET="${SCORE_THR_DET:-0.2}"
ANCHOR_DET_THR="${ANCHOR_DET_THR:-0.0}"
QUERY_POOL="${QUERY_POOL:-top1}"
QUERY_TOPK="${QUERY_TOPK:-5}"
FUSION_DET_THR="${FUSION_DET_THR:-0.3}"
FUSION_QCOS_THR="${FUSION_QCOS_THR:-0.7}"
FUSION_QCOS_TEMP="${FUSION_QCOS_TEMP:-20.0}"
FUSION_QCOS_GATE="${FUSION_QCOS_GATE:-sigmoid}"
FUSION_USE_PRESENCE="${FUSION_USE_PRESENCE:-1}"
FUSION_ALPHA="${FUSION_ALPHA:-0.01}"
FUSION_ALPHA_OBJ="${FUSION_ALPHA_OBJ:-0.001}"

USE_CLASS_PROMPTS="${USE_CLASS_PROMPTS:-0}"
INIT_PROMPT="${INIT_PROMPT:-mask}"
INIT_FRAME_MODE="${INIT_FRAME_MODE:-first_present}"
INIT_SELECT_MODE="${INIT_SELECT_MODE:-prob}"
INIT_MIN_AREA="${INIT_MIN_AREA:-0}"
INIT_BOX_PAD="${INIT_BOX_PAD:-0}"
PROPAGATION_MODE="${PROPAGATION_MODE:-vg}"
CLASSES="${CLASSES:-}"
DEBUG_INIT="${DEBUG_INIT:-0}"
DEBUG_SPME_SUMMARY="${DEBUG_SPME_SUMMARY:-1}"
HOTSTART_DELAY="${HOTSTART_DELAY:-0}"
WRITE_FUSED_MASKS="${WRITE_FUSED_MASKS:-0}"

export SAM3_ALLOW_NEW_DETECTIONS="${SAM3_ALLOW_NEW_DETECTIONS:-1}"
export SAM3_ALLOW_NEW_DETECTIONS_WITH_TEXT="${SAM3_ALLOW_NEW_DETECTIONS_WITH_TEXT:-0}"

export SAM3_DISABLE_RECONDITION="${SAM3_DISABLE_RECONDITION:-1}"
export SAM3_RECONDITION_EVERY_NTH_FRAME="${SAM3_RECONDITION_EVERY_NTH_FRAME:-}"

GATE_USE_DECAY="${GATE_USE_DECAY:-1}"
GATE_OCC_NORM="${GATE_OCC_NORM:-10.0}"
GATE_INPUTS="${GATE_INPUTS:-full}"
GATE_FUSION_HEAD="${GATE_FUSION_HEAD:-1}"
GATE_DET_PRESENT_THR="${GATE_DET_PRESENT_THR:-0.3}"

COMMON_ARGS=(
  --new-det-thr "$NEW_DET_THR"
  --score-thr-detection "$SCORE_THR_DET"
  --init-prompt "$INIT_PROMPT"
  --init-frame-mode "$INIT_FRAME_MODE"
  --init-select-mode "$INIT_SELECT_MODE"
  --init-min-area "$INIT_MIN_AREA"
  --init-box-pad "$INIT_BOX_PAD"
  --propagation-mode "$PROPAGATION_MODE"
)
if [[ "$DEBUG_SPME_SUMMARY" == "1" ]]; then
  COMMON_ARGS+=(--debug-spme-summary)
fi
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

RUN_CONFIGS="${RUN_CONFIGS:-A D F}"

# Auto-pick latest checkpoints (optional). Override explicitly if you want to pin ckpts.
DEFAULT_CKPT_ROOT="/home2020/home/icube/kunyuan/SurgBench/SAM/outputs"
FUSION_CKPT_GLOB="${FUSION_CKPT_GLOB:-$DEFAULT_CKPT_ROOT/endovis2017_spme_fusion_train*/job_*/main/checkpoints/spme_fusion_latest.pt}"
GATE_CKPT_GLOB="${GATE_CKPT_GLOB:-$DEFAULT_CKPT_ROOT/endovis2017_spme_joint_train*/job_*/main/checkpoints/spme_gate_latest.pt}"
if [[ -z "$FUSION_CKPT" ]]; then
  FUSION_CKPT="$(ls -t $FUSION_CKPT_GLOB 2>/dev/null | head -n1 || true)"
fi
if [[ -z "$GATE_CKPT" ]]; then
  GATE_CKPT="$(ls -t $GATE_CKPT_GLOB 2>/dev/null | head -n1 || true)"
fi

echo "RUN_CONFIGS=$RUN_CONFIGS"
echo "SEQS=$SEQS"
echo "DATA=$DATA ENDOVIS_SRC=$ENDOVIS_SRC CAMERA=$CAMERA"
echo "NEW_DET_THR=$NEW_DET_THR SCORE_THR_DET=$SCORE_THR_DET ANCHOR_DET_THR=$ANCHOR_DET_THR QUERY_POOL=$QUERY_POOL QUERY_TOPK=$QUERY_TOPK"
echo "FUSION_DET_THR=$FUSION_DET_THR FUSION_QCOS_THR=$FUSION_QCOS_THR FUSION_ALPHA=$FUSION_ALPHA FUSION_ALPHA_OBJ=$FUSION_ALPHA_OBJ"
echo "INIT_PROMPT=$INIT_PROMPT INIT_FRAME_MODE=$INIT_FRAME_MODE INIT_SELECT_MODE=$INIT_SELECT_MODE PROPAGATION_MODE=$PROPAGATION_MODE HOTSTART_DELAY=$HOTSTART_DELAY"
echo "FUSION_CKPT=${FUSION_CKPT:-<none>} GATE_CKPT=${GATE_CKPT:-<none>} SAM3_SPME_GATE_HIDDEN=$SAM3_SPME_GATE_HIDDEN"

# =============================================================================
# Configuration A: Baseline (no SPME)
# =============================================================================
if [[ " $RUN_CONFIGS " == *" A "* ]]; then
  echo "=== [A] Baseline (no SPME) ==="
  unset SAM3_SPME_WRITE_GATE SAM3_SPME_WRITE_GATE_MODE SAM3_SPME_DET_THR SAM3_SPME_QCOS_THR
  unset SAM3_SPME_USE_DET_SCORE SAM3_SPME_WRITE_GATE_APPLY SAM3_SPME_DEBUG_SIGNALS
  unset SAM3_SPME_FUSION SAM3_SPME_FUSION_ALPHA SAM3_SPME_FUSION_DET_THR SAM3_SPME_ANCHOR_DET_THR
  unset SAM3_SPME_QUERY_POOL SAM3_SPME_QUERY_TOPK
  unset SAM3_SPME_PER_OBJECT
  unset SAM3_SPME_KEEP_QUERIES
  unset SAM3_SPME_LEARNED_GATE SAM3_SPME_LEARNED_GATE_USE_DECAY SAM3_SPME_LEARNED_GATE_OCCLUDED_NORM SAM3_SPME_LEARNED_GATE_LOG

  python scripts/eval_endovis2018.py \
    --data-root "$DATA" \
    --sequences $SEQS \
    --out-dir "$OUT_ROOT/baseline" \
    --base-sam3-pt "$SAM3_PT" \
    --prompt "$PROMPT" \
    "${COMMON_ARGS[@]}"
fi

# =============================================================================
# Configuration D: SPME-F only (fusion)
# =============================================================================
if [[ " $RUN_CONFIGS " == *" D "* ]]; then
  echo "=== [D] Fusion only ==="
  unset SAM3_SPME_WRITE_GATE SAM3_SPME_WRITE_GATE_MODE SAM3_SPME_DET_THR SAM3_SPME_QCOS_THR
  unset SAM3_SPME_USE_DET_SCORE SAM3_SPME_WRITE_GATE_APPLY SAM3_SPME_DEBUG_SIGNALS
  unset SAM3_SPME_LEARNED_GATE SAM3_SPME_LEARNED_GATE_USE_DECAY SAM3_SPME_LEARNED_GATE_OCCLUDED_NORM SAM3_SPME_LEARNED_GATE_LOG
  export SAM3_SPME_FUSION=1
  export SAM3_SPME_FUSION_MODE="${SAM3_SPME_FUSION_MODE:-film}"
  export SAM3_SPME_FUSION_ALPHA="$FUSION_ALPHA"
  export SAM3_SPME_FUSION_ALPHA_OBJ="$FUSION_ALPHA_OBJ"
  export SAM3_SPME_FUSION_DET_THR="$FUSION_DET_THR"
  export SAM3_SPME_ANCHOR_DET_THR="$ANCHOR_DET_THR"
  export SAM3_SPME_QUERY_POOL="$QUERY_POOL"
  export SAM3_SPME_QUERY_TOPK="$QUERY_TOPK"
  # Per-object pointer matching requires full per-query embeddings from the detector buffer.
  export SAM3_SPME_KEEP_QUERIES="${SAM3_SPME_KEEP_QUERIES:-1}"
  export SAM3_SPME_PER_OBJECT="${SAM3_SPME_PER_OBJECT:-1}"

  export SAM3_SPME_FUSION_QCOS_THR="$FUSION_QCOS_THR"
  export SAM3_SPME_FUSION_QCOS_TEMP="$FUSION_QCOS_TEMP"
  export SAM3_SPME_FUSION_QCOS_GATE="$FUSION_QCOS_GATE"
  export SAM3_SPME_FUSION_USE_PRESENCE="$FUSION_USE_PRESENCE"

  if [[ -z "$FUSION_CKPT" ]]; then
    echo "[error] FUSION_CKPT is empty but RUN_CONFIGS includes D."
    exit 1
  fi

  python scripts/eval_endovis2018.py \
    --data-root "$DATA" \
    --sequences $SEQS \
    --out-dir "$OUT_ROOT/fusion_only" \
    --base-sam3-pt "$SAM3_PT" \
    --prompt "$PROMPT" \
    --spme-ckpt "$FUSION_CKPT" \
    "${COMMON_ARGS[@]}"
fi

# =============================================================================
# Configuration E: Learned gate only (no fusion injection)
# =============================================================================
if [[ " $RUN_CONFIGS " == *" E "* ]]; then
  echo "=== [E] Learned gate only ==="
  export SAM3_SPME_LEARNED_GATE=1
  export SAM3_SPME_LEARNED_GATE_USE_DECAY="$GATE_USE_DECAY"
  export SAM3_SPME_LEARNED_GATE_OCCLUDED_NORM="$GATE_OCC_NORM"
  export SAM3_SPME_LEARNED_GATE_INPUTS="$GATE_INPUTS"
  export SAM3_SPME_LEARNED_GATE_FUSION_HEAD="$GATE_FUSION_HEAD"
  export SAM3_SPME_LEARNED_GATE_DET_PRESENT_THR="$GATE_DET_PRESENT_THR"
  # Dump per-frame gate stats into outputs for debugging (small JSON; turn off if needed).
  export SAM3_SPME_LEARNED_GATE_LOG="${SAM3_SPME_LEARNED_GATE_LOG:-1}"

  # Keep fusion enabled to build per-object semantic pointers, but disable injection.
  export SAM3_SPME_FUSION=1
  export SAM3_SPME_FUSION_MODE="${SAM3_SPME_FUSION_MODE:-film}"
  export SAM3_SPME_FUSION_ALPHA="0.0"
  export SAM3_SPME_FUSION_ALPHA_OBJ="0.0"
  export SAM3_SPME_ANCHOR_DET_THR="$ANCHOR_DET_THR"
  export SAM3_SPME_QUERY_POOL="$QUERY_POOL"
  export SAM3_SPME_QUERY_TOPK="$QUERY_TOPK"
  export SAM3_SPME_PER_OBJECT="${SAM3_SPME_PER_OBJECT:-1}"
  # Per-object pointer matching requires full per-query embeddings from the detector buffer.
  export SAM3_SPME_KEEP_QUERIES="${SAM3_SPME_KEEP_QUERIES:-1}"

  export SAM3_SPME_FUSION_QCOS_THR="$FUSION_QCOS_THR"
  export SAM3_SPME_FUSION_QCOS_TEMP="$FUSION_QCOS_TEMP"
  export SAM3_SPME_FUSION_QCOS_GATE="$FUSION_QCOS_GATE"
  export SAM3_SPME_FUSION_USE_PRESENCE="$FUSION_USE_PRESENCE"

  if [[ -z "$GATE_CKPT" ]]; then
    echo "[error] GATE_CKPT is empty but RUN_CONFIGS includes E."
    exit 1
  fi

  python scripts/eval_endovis2018.py \
    --data-root "$DATA" \
    --sequences $SEQS \
    --out-dir "$OUT_ROOT/learned_gate_only" \
    --base-sam3-pt "$SAM3_PT" \
    --prompt "$PROMPT" \
    --spme-ckpt "$GATE_CKPT" \
    "${COMMON_ARGS[@]}"
fi

# =============================================================================
# Configuration F: Learned gate + fusion injection
# =============================================================================
if [[ " $RUN_CONFIGS " == *" F "* ]]; then
  echo "=== [F] Learned gate + fusion ==="
  export SAM3_SPME_LEARNED_GATE=1
  export SAM3_SPME_LEARNED_GATE_USE_DECAY="$GATE_USE_DECAY"
  export SAM3_SPME_LEARNED_GATE_OCCLUDED_NORM="$GATE_OCC_NORM"
  export SAM3_SPME_LEARNED_GATE_INPUTS="$GATE_INPUTS"
  export SAM3_SPME_LEARNED_GATE_FUSION_HEAD="$GATE_FUSION_HEAD"
  export SAM3_SPME_LEARNED_GATE_DET_PRESENT_THR="$GATE_DET_PRESENT_THR"
  # Dump per-frame gate stats into outputs for debugging (small JSON; turn off if needed).
  export SAM3_SPME_LEARNED_GATE_LOG="${SAM3_SPME_LEARNED_GATE_LOG:-1}"

  export SAM3_SPME_FUSION=1
  export SAM3_SPME_FUSION_MODE="${SAM3_SPME_FUSION_MODE:-film}"
  export SAM3_SPME_FUSION_ALPHA="$FUSION_ALPHA"
  export SAM3_SPME_FUSION_ALPHA_OBJ="$FUSION_ALPHA_OBJ"
  export SAM3_SPME_FUSION_DET_THR="$FUSION_DET_THR"
  export SAM3_SPME_ANCHOR_DET_THR="$ANCHOR_DET_THR"
  export SAM3_SPME_QUERY_POOL="$QUERY_POOL"
  export SAM3_SPME_QUERY_TOPK="$QUERY_TOPK"
  export SAM3_SPME_PER_OBJECT="${SAM3_SPME_PER_OBJECT:-1}"
  # Per-object pointer matching requires full per-query embeddings from the detector buffer.
  export SAM3_SPME_KEEP_QUERIES="${SAM3_SPME_KEEP_QUERIES:-1}"

  export SAM3_SPME_FUSION_QCOS_THR="$FUSION_QCOS_THR"
  export SAM3_SPME_FUSION_QCOS_TEMP="$FUSION_QCOS_TEMP"
  export SAM3_SPME_FUSION_QCOS_GATE="$FUSION_QCOS_GATE"
  export SAM3_SPME_FUSION_USE_PRESENCE="$FUSION_USE_PRESENCE"

  if [[ -z "$FUSION_CKPT" || -z "$GATE_CKPT" ]]; then
    echo "[error] FUSION_CKPT/GATE_CKPT is empty but RUN_CONFIGS includes F."
    exit 1
  fi

  python scripts/eval_endovis2018.py \
    --data-root "$DATA" \
    --sequences $SEQS \
    --out-dir "$OUT_ROOT/learned_gate_plus_fusion" \
    --base-sam3-pt "$SAM3_PT" \
    --prompt "$PROMPT" \
    --spme-ckpt "$FUSION_CKPT" \
    --spme-ckpt "$GATE_CKPT" \
    "${COMMON_ARGS[@]}"
fi

echo "=== Done ==="
echo "OUT_ROOT=$OUT_ROOT"
