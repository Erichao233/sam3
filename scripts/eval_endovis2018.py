#!/usr/bin/env python3
"""
EndoVis 2018 Instrument Segmentation Evaluation with SAM3.

Key features:
- Tracks each instrument class as a separate single-object problem
- Computes IoU, Dice, persistence, recovery metrics
- Supports SPME-W (write gate) and SPME-F (fusion) via environment variables

Usage:
    python scripts/eval_endovis2018.py \
        --data-root /path/to/endovis2018 \
        --sequences val1 val2 val3 \
        --out-dir /path/to/output \
        --base-sam3-pt /path/to/sam3.pt \
        --prompt "surgical instrument"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

# Add repo to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sam3.model_builder import build_sam3_video_model
from sam3.model.sam3_video_inference import Sam3VideoInference


# ============================================================================
# Data loading
# ============================================================================

INSTRUMENT_CLASSES = {
    0: "background-tissue",
    1: "instrument-shaft",
    2: "instrument-clasper",
    3: "instrument-wrist",
    4: "kidney-parenchyma",
    5: "covered-kidney",
    6: "thread",
    7: "clamps",
    8: "suturing-needle",
    9: "suction-instrument",
    10: "small-intestine",
    11: "ultrasound-probe",
}


def load_bmp_mask(path: Path) -> np.ndarray:
    """Load mask where pixel value = class ID (BMP/PNG/JPG supported)."""
    with Image.open(path) as m:
        arr = np.array(m)
    if arr.ndim == 3:
        arr = arr[:, :, 0]  # Take first channel if RGB
    return arr.astype(np.uint8)


def get_sequence_frames(data_root: Path, seq_name: str):
    """Get sorted list of (frame_idx, image_path, label_path) for a sequence."""
    seq_dir = data_root / seq_name
    image_dir = seq_dir / "image"
    label_dir = seq_dir / "label"

    if not image_dir.is_dir() or not label_dir.is_dir():
        print(f"[warn] Missing image/label dirs for sequence {seq_name}: {seq_dir}")
        return []
    
    frames = []
    allowed_exts = [".png", ".bmp", ".jpg", ".jpeg"]
    label_paths = sorted(
        p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() in set(allowed_exts)
    )
    for label_path in label_paths:
        # Parse frame index from filename: seq_X_frameYYY.bmp
        stem = label_path.stem
        parts = stem.split("_")
        frame_idx = int(parts[-1].replace("frame", ""))
        
        # Find corresponding image (prefer same extension, fallback to other common extensions)
        image_path = image_dir / label_path.name
        if not image_path.exists():
            image_path = None
        if image_path is None:
            for ext in allowed_exts:
                candidate = image_dir / f"{stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
        
        if image_path is None:
            print(f"[warn] No image found for {label_path}")
            continue
        
        frames.append((frame_idx, image_path, label_path))
    
    # Sort by frame index
    frames.sort(key=lambda x: x[0])
    return frames


def get_classes_in_mask(mask: np.ndarray) -> list[int]:
    """Get list of class IDs present in mask (excluding background=0)."""
    unique = np.unique(mask)
    return [int(c) for c in unique if c > 0]


def bbox_from_mask(mask: np.ndarray) -> Optional[tuple]:
    """Get XYXY bbox from binary mask. Returns None if mask is empty."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice between two binary masks."""
    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(2 * intersection) / float(total)


def fuse_class_masks_quality_weighted(
    class_ids: list[int],
    pred_masks_by_class: dict[int, dict[int, np.ndarray]],
    pred_weights_by_class: dict[int, dict[int, float]],
    num_frames: int,
    h: int,
    w: int,
) -> list[np.ndarray]:
    """
    Fuse per-class binary masks into a single label map per frame using a scalar
    quality weight per class per frame.

    Returns:
        fused_labels: list of length num_frames, each uint8 (H,W) with 0=background.
    """
    fused: list[np.ndarray] = []
    for t in range(int(num_frames)):
        out = np.zeros((h, w), dtype=np.uint8)
        best = np.full((h, w), -1e9, dtype=np.float32)
        for cid in class_ids:
            m = pred_masks_by_class.get(int(cid), {}).get(int(t), None)
            if m is None:
                continue
            if m.dtype != bool:
                m = m.astype(bool)
            if not m.any():
                continue
            wt = float(pred_weights_by_class.get(int(cid), {}).get(int(t), 0.0))
            upd = m & (wt > best)
            if upd.any():
                out[upd] = np.uint8(cid)
                best[upd] = np.float32(wt)
        fused.append(out)
    return fused


def compute_fused_metrics(
    gt_labels: list[np.ndarray],
    fused_labels: list[np.ndarray],
    class_ids: list[int],
) -> dict:
    """
    Compute multi-class metrics on fused label maps.

    - challenge_iou: average IoU over (frame, class) pairs where GT class is present
    - per_class_iou_present: mean IoU over GT-present frames for each class
    - mean_class_iou_present: mean over classes of per_class_iou_present (only classes that appear)
    - per_class_iou: IoU over the whole sequence for each class (penalizes false positives on GT-absent frames)
    - mean_class_iou: mean over classes of per_class_iou (paper-style mcIoU)
    - binary_iou: IoU of foreground (instrument vs background)
    """
    assert len(gt_labels) == len(fused_labels)
    num_frames = len(gt_labels)

    # Per-class accumulation (GT-present frames only)
    per_class_ious: dict[int, list[float]] = {int(cid): [] for cid in class_ids}
    all_present_ious: list[float] = []

    # Per-class intersection/union over the whole sequence (mcIoU-style).
    per_class_inter: dict[int, int] = {int(cid): 0 for cid in class_ids}
    per_class_union: dict[int, int] = {int(cid): 0 for cid in class_ids}

    # Binary foreground IoU per frame
    fg_ious: list[float] = []

    for t in range(num_frames):
        gt = gt_labels[t].astype(np.uint8)
        pred = fused_labels[t].astype(np.uint8)

        # Binary (any instrument)
        fg_ious.append(compute_iou(pred > 0, gt > 0))

        # Whole-sequence IoU accumulation (intersection/union sums).
        for cid in class_ids:
            cid_i = int(cid)
            gt_c = gt == cid_i
            pred_c = pred == cid_i
            per_class_inter[cid_i] += int(np.logical_and(pred_c, gt_c).sum())
            per_class_union[cid_i] += int(np.logical_or(pred_c, gt_c).sum())

        present = [int(c) for c in np.unique(gt) if int(c) in set(class_ids) and int(c) > 0]
        for cid in present:
            iou = compute_iou(pred == cid, gt == cid)
            per_class_ious[int(cid)].append(float(iou))
            all_present_ious.append(float(iou))

    per_class_iou_present = {
        int(cid): (float(np.mean(vals)) if len(vals) > 0 else None)
        for cid, vals in per_class_ious.items()
    }
    present_class_means = [v for v in per_class_iou_present.values() if v is not None]

    per_class_iou = {
        int(cid): (
            float(per_class_inter[int(cid)]) / float(per_class_union[int(cid)])
            if int(per_class_union[int(cid)]) > 0
            else None
        )
        for cid in class_ids
    }
    class_means = [v for v in per_class_iou.values() if v is not None]

    return {
        "num_frames": int(num_frames),
        "challenge_iou": float(np.mean(all_present_ious)) if all_present_ious else 0.0,
        "mean_class_iou_present": float(np.mean(present_class_means)) if present_class_means else 0.0,
        "per_class_iou_present": per_class_iou_present,
        "mean_class_iou": float(np.mean(class_means)) if class_means else 0.0,
        "per_class_iou": per_class_iou,
        "binary_iou": float(np.mean(fg_ious)) if fg_ious else 0.0,
    }


@dataclass
class FrameResult:
    frame_idx: int
    gt_present: bool
    pred_present: bool
    iou: float
    dice: float
    det_score: Optional[float] = None
    presence_prob: Optional[float] = None
    query_cos: Optional[float] = None
    det_trk_iou: Optional[float] = None
    tracker_score: Optional[float] = None
    fusion_scale_mem: Optional[float] = None
    fusion_scale_obj: Optional[float] = None
    gate_mem_scale_mean: Optional[float] = None
    gate_mem_offset_mean_abs: Optional[float] = None
    gate_decay: Optional[float] = None
    gate_fusion: Optional[float] = None
    reinit: Optional[float] = None
    reid_best_sim: Optional[float] = None
    reid_trk_sim: Optional[float] = None
    reid_det_vs_trk_margin: Optional[float] = None
    reid_margin: Optional[float] = None
    reid_accept: Optional[float] = None
    reid_bank_size: Optional[float] = None
    skip_write: Optional[float] = None


@dataclass  
class SequenceResult:
    sequence: str
    class_id: int
    class_name: str
    num_frames: int
    mean_iou: float
    mean_dice: float
    # Persistence: GT absent after last GT present, but pred still present
    persistence_length: Optional[int] = None
    # Drift
    num_drift_events: int = 0
    mean_drift_length: float = 0.0
    max_drift_length: int = 0
    # Recovery
    recovery_attempts: int = 0
    recovery_successes: int = 0
    recovery_rate: float = 0.0
    # Raw frame results
    frames: list = field(default_factory=list)


# ============================================================================
# SAM3 tracking
# ============================================================================

def run_sam3_tracking(
    model,
    frames: list,
    class_id: int,
    prompt: str,
    init_prompt: str = "box",
    init_frame_mode: str = "first_present",
    init_select_mode: str = "prob",
    init_min_area: int = 0,
    init_box_pad: int = 0,
    debug_init: bool = False,
    propagation_mode: str = "vg",
    fusion_weight_mode: str = "tracker_prob",
) -> tuple[list[FrameResult], dict[int, np.ndarray], dict[int, float]]:
    """
    Run SAM3 video tracking for a single class.
    
    Args:
        model: SAM3 video model (Sam3VideoInferenceWithInstanceInteractivity)
        frames: List of (frame_idx, image_path, label_path)
        class_id: Which class to track
        prompt: Text prompt
    
    Returns:
        (frame_results, pred_masks_by_local_frame, pred_weights_by_local_frame)
        - pred_masks_by_local_frame: {local_frame_idx -> bool(H,W)}
        - pred_weights_by_local_frame: {local_frame_idx -> float} (quality proxy)
    """
    results = []
    init_prompt = str(init_prompt).strip().lower()
    if init_prompt not in {"box", "mask"}:
        raise ValueError(f"Unknown init_prompt={init_prompt!r}. Use 'box' or 'mask'.")

    fusion_weight_mode = str(fusion_weight_mode).strip().lower()
    if fusion_weight_mode not in {"tracker_prob", "det_prob", "eff_iou", "reliability"}:
        raise ValueError(
            f"Unknown fusion_weight_mode={fusion_weight_mode!r}. "
            "Use one of: tracker_prob | det_prob | eff_iou | reliability."
        )
    
    # Load all images (PIL) and masks
    images_pil: list[Image.Image] = []
    gt_masks = []
    for frame_idx, image_path, label_path in frames:
        with Image.open(image_path) as im:
            images_pil.append(im.convert("RGB").copy())  # .copy() ensures data persists
        mask = load_bmp_mask(label_path)
        gt_binary = (mask == class_id).astype(bool)
        gt_masks.append(gt_binary)
    
    # Select init frame (EndoVis masks can be irregular/partially occluded early on; area-based init can be stabler)
    areas = [int(gt.sum()) for gt in gt_masks]
    init_frame_idx = None
    if init_frame_mode == "max_area":
        if max(areas, default=0) > 0:
            init_frame_idx = int(np.argmax(areas))
    elif init_frame_mode == "first_min_area":
        thr = max(0, int(init_min_area))
        for i, a in enumerate(areas):
            if a >= thr and a > 0:
                init_frame_idx = i
                break
    else:
        # default: first frame where GT is present
        for i, a in enumerate(areas):
            if a > 0:
                init_frame_idx = i
                break

    init_bbox = None
    if init_frame_idx is not None:
        init_bbox = bbox_from_mask(gt_masks[init_frame_idx])
    
    if init_frame_idx is None:
        # Class never appears, return empty results
        for i, (frame_idx, _, _) in enumerate(frames):
            results.append(FrameResult(
                frame_idx=frame_idx,
                gt_present=bool(gt_masks[i].any()),
                pred_present=False,
                iou=0.0,
                dice=0.0,
            ))
        return results, {}, {}

    spme_debug_by_local_frame: dict[int, dict[str, float]] = {}

    # Initialize inference state (use a list of PIL images to preserve the frame ordering)
    with torch.inference_mode():
        state = model.init_state(resource_path=images_pil, offload_video_to_cpu=False)

        # Store predicted segments as {local_frame_idx -> binary_mask(H,W)}.
        # NOTE: model frame indices are 0..num_frames-1 (not the dataset frame numbers).
        video_segments: dict[int, np.ndarray] = {}
        video_weights: dict[int, float] = {}

        w, h = images_pil[0].size
        if init_prompt == "mask":
            # Paper-aligned protocol: initialize from the first visible mask.
            # This is a semi-supervised setting (GT given at init), not fully automatic.
            init_mask = gt_masks[init_frame_idx].astype(bool)
            if init_mask.shape != (h, w):
                # If label resolution differs, fall back to box init.
                init_prompt = "box"
            else:
                # Match the semantics of `Sam3VideoInference.add_prompt(text_str="visual")`
                # without resetting state / running detection-based init.
                if prompt is not None and str(prompt).strip().lower() != "visual":
                    state["text_prompt"] = str(prompt)
                    state["input_batch"].find_text_batch[0] = str(prompt)
                    text_id = model.TEXT_ID_FOR_TEXT
                else:
                    state["text_prompt"] = None
                    state["input_batch"].find_text_batch[0] = "<text placeholder>"
                    text_id = model.TEXT_ID_FOR_VISUAL
                for t in range(state["num_frames"]):
                    state["input_batch"].find_inputs[t].text_ids[...] = text_id

                model.add_tracker_new_mask(
                    state,
                    frame_idx=init_frame_idx,
                    obj_id=0,
                    mask=torch.from_numpy(init_mask),
                    add_mask_to_memory=True,
                )
                target_obj_id = 0
                video_segments[int(init_frame_idx)] = init_mask
                video_weights[int(init_frame_idx)] = 1.0

        if init_prompt == "box":
            # Add prompt on the init frame using the GT bbox (normalized xywh in [0,1])
            x0, y0, x1, y1 = init_bbox
            if init_box_pad > 0:
                pad = int(init_box_pad)
                x0 = max(0, int(x0) - pad)
                y0 = max(0, int(y0) - pad)
                x1 = min(int(w), int(x1) + pad)
                y1 = min(int(h), int(y1) + pad)
            boxes_xywh = [[x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h]]
            box_labels = [1]  # 1=positive box
            if debug_init:
                print(
                    f"[init] class_id={class_id} init_prompt={init_prompt} init_frame_mode={init_frame_mode} "
                    f"init_min_area={init_min_area} init_box_pad={init_box_pad} "
                    f"init_frame_idx={init_frame_idx} init_area={areas[init_frame_idx]}"
                )
            _, out0 = model.add_prompt(
                state,
                frame_idx=init_frame_idx,
                text_str=prompt,
                boxes_xywh=boxes_xywh,
                box_labels=box_labels,
            )

        if init_prompt == "box":
            # Pick a single object to track (if multiple are produced).
            #
            # IMPORTANT (paper hygiene):
            # - Default selection uses model confidence (`out_probs`) or the *provided* init bbox (no GT leakage).
            # - `gt_iou` is kept only for debugging; it uses GT mask and should not be used for reporting.
            target_obj_id = None
            out0_obj_ids = out0.get("out_obj_ids", np.zeros(0, dtype=np.int64))
            out0_masks = out0.get("out_binary_masks", np.zeros((0, h, w), dtype=bool))
            out0_probs = out0.get("out_probs", np.zeros(0, dtype=np.float32))
            out0_boxes = out0.get("out_boxes_xywh", np.zeros((0, 4), dtype=np.float32))
            if len(out0_obj_ids) == 0:
                print(
                    f"[warn] No objects returned on init frame for class_id={class_id} "
                    f"(seq local frame index={init_frame_idx}). Prompt='{prompt}'. "
                    "Tracking will output empty masks."
                )
            if len(out0_obj_ids) > 0 and len(out0_masks) == len(out0_obj_ids):
                select_mode = str(init_select_mode).strip().lower()
                if select_mode not in {"prob", "bbox_iou", "gt_iou"}:
                    select_mode = "prob"

                cand_scores: list[tuple[float, int]] = []
                best_score = -1.0

                if select_mode == "prob" and len(out0_probs) == len(out0_obj_ids):
                    for p, oid in zip(out0_probs, out0_obj_ids):
                        s = float(p)
                        cand_scores.append((s, int(oid)))
                        if s > best_score:
                            best_score = s
                            target_obj_id = int(oid)
                else:
                    if select_mode == "bbox_iou":
                        box_mask = np.zeros((h, w), dtype=bool)
                        bx0, by0, bx1, by1 = init_bbox
                        box_mask[int(by0):int(by1), int(bx0):int(bx1)] = True
                        ref = box_mask
                    else:
                        # Debug-only: uses GT mask (leaky).
                        ref = gt_masks[init_frame_idx]
                    for oid, mask in zip(out0_obj_ids, out0_masks):
                        s = compute_iou(mask.astype(bool), ref)
                        cand_scores.append((float(s), int(oid)))
                        if s > best_score:
                            best_score = float(s)
                            target_obj_id = int(oid)

                cand_scores.sort(reverse=True, key=lambda x: x[0])
                if debug_init or best_score <= 0.0:
                    top = cand_scores[: min(5, len(cand_scores))]
                    print(
                        f"[init-debug] class_id={class_id} prompt='{prompt}' "
                        f"init_bbox_xyxy={init_bbox} num_cands={len(cand_scores)} "
                        f"select_mode={select_mode} best_score={best_score:.4f} "
                        f"best_obj_id={target_obj_id} top5={top}"
                    )
                    if len(out0_probs) == len(out0_obj_ids):
                        probs_top = sorted(
                            [(float(p), int(oid)) for p, oid in zip(out0_probs, out0_obj_ids)],
                            reverse=True,
                            key=lambda x: x[0],
                        )[:5]
                        print(f"[init-debug] top5 by out_probs: {probs_top}")
                    if len(out0_boxes) == len(out0_obj_ids):
                        boxes_preview = [
                            (int(oid), [float(x) for x in box])
                            for oid, box in list(zip(out0_obj_ids, out0_boxes))[:5]
                        ]
                        print(f"[init-debug] boxes_preview (obj_id, xywh_norm): {boxes_preview}")

        # Save the init-frame prediction (from add_prompt) before propagation.
        # This avoids reporting empty for the init frame if hotstart filtering hides early outputs.
        if init_prompt == "box":
            if target_obj_id is not None and len(out0_obj_ids) > 0 and len(out0_masks) == len(out0_obj_ids):
                match0 = np.where(out0_obj_ids.astype(np.int64) == np.int64(target_obj_id))[0]
                if len(match0) > 0:
                    video_segments[int(init_frame_idx)] = out0_masks[int(match0[0])].astype(bool)
                    out0_tracker_probs = out0.get(
                        "out_tracker_probs", np.zeros(len(out0_obj_ids), dtype=np.float32)
                    )
                    w_trk = (
                        float(out0_tracker_probs[int(match0[0])])
                        if len(out0_tracker_probs) == len(out0_obj_ids)
                        else 0.0
                    )
                    w_det = (
                        float(out0_probs[int(match0[0])])
                        if len(out0_probs) == len(out0_obj_ids)
                        else 0.0
                    )
                    if fusion_weight_mode == "det_prob":
                        video_weights[int(init_frame_idx)] = w_det if w_det > 0.0 else w_trk
                    else:
                        video_weights[int(init_frame_idx)] = w_trk if w_trk > 0.0 else w_det

        # Propagate through video and store predicted mask for the chosen object id.
        # If init_frame_idx is not the first GT-present frame, run backward propagation too.
        def _update_segments(out_frame_idx, out):
            if init_prompt == "mask" and int(out_frame_idx) == int(init_frame_idx):
                # Keep the provided init mask for the init frame (semi-supervised protocol).
                return
            out_obj_ids = out.get("out_obj_ids", np.zeros(0, dtype=np.int64))
            out_masks = out.get("out_binary_masks", np.zeros((0, h, w), dtype=bool))
            out_tracker_probs = out.get("out_tracker_probs", np.zeros(0, dtype=np.float32))
            out_probs = out.get("out_probs", np.zeros(0, dtype=np.float32))

            pred_mask = np.zeros_like(gt_masks[0], dtype=bool)
            weight = 0.0
            if target_obj_id is not None and len(out_obj_ids) > 0:
                match = np.where(out_obj_ids.astype(np.int64) == np.int64(target_obj_id))[0]
                if len(match) > 0:
                    mi = int(match[0])
                    pred_mask = out_masks[mi].astype(bool)
                    w_trk = float(out_tracker_probs[mi]) if len(out_tracker_probs) == len(out_obj_ids) else 0.0
                    w_det = float(out_probs[mi]) if len(out_probs) == len(out_obj_ids) else 0.0
                    if fusion_weight_mode == "det_prob":
                        weight = w_det if w_det > 0.0 else w_trk
                    else:
                        weight = w_trk if w_trk > 0.0 else w_det
            video_segments[int(out_frame_idx)] = pred_mask
            video_weights[int(out_frame_idx)] = weight

        if str(propagation_mode).strip().lower() == "vg":
            # IMPORTANT: This forces SAM3's detector+tracker propagation (VG path),
            # which is where SPME edits are applied in `Sam3VideoBase._tracker_update_memories`.
            # The instance-interactivity propagation runs Tracker-only with `run_mem_encoder=True`,
            # which bypasses SPME entirely.
            def propagate_fn(*args, **kwargs):
                return Sam3VideoInference.propagate_in_video(model, *args, **kwargs)
        else:
            propagate_fn = model.propagate_in_video

        for out_frame_idx, out in propagate_fn(
            state,
            start_frame_idx=init_frame_idx,
            max_frame_num_to_track=len(images_pil),
            reverse=False,
        ):
            _update_segments(out_frame_idx, out)

        has_gt_present_before = any(a > 0 for a in areas[:init_frame_idx])
        if has_gt_present_before:
            for out_frame_idx, out in propagate_fn(
                state,
                start_frame_idx=init_frame_idx,
                max_frame_num_to_track=len(images_pil),
                reverse=True,
            ):
                _update_segments(out_frame_idx, out)

        # Extract SPME debug stats (if enabled) from the tracker state's output_dict.
        # - Learned-gate keys are populated when `SAM3_SPME_LEARNED_GATE_LOG=1`.
        # - Additional SPME signal keys are populated when `SAM3_SPME_LOG_SIGNALS=1` (or learned-gate log is on).
        try:
            if target_obj_id is not None:
                for tracker_state in state.get("tracker_inference_states", []):
                    obj_ids = [int(x) for x in tracker_state.get("obj_ids", [])]
                    if int(target_obj_id) not in obj_ids:
                        continue
                    row = int(obj_ids.index(int(target_obj_id)))
                    out_dict = tracker_state.get("output_dict", {})
                    non_cond = out_dict.get("non_cond_frame_outputs", {})
                    cond = out_dict.get("cond_frame_outputs", {})
                    for local_fidx in range(int(state.get("num_frames", len(images_pil)))):
                        entry = None
                        if isinstance(non_cond, dict) and local_fidx in non_cond:
                            entry = non_cond.get(local_fidx)
                        if entry is None and isinstance(cond, dict) and local_fidx in cond:
                            entry = cond.get(local_fidx)
                        if not isinstance(entry, dict):
                            continue

                        def _row_scalar(key: str) -> float | None:
                            t = entry.get(key, None)
                            if not isinstance(t, torch.Tensor) or t.numel() == 0:
                                return None
                            if t.ndim == 0:
                                return float(t.detach().float().item())
                            if row >= int(t.shape[0]):
                                return None
                            # Reduce over any trailing dimensions.
                            return float(t[row].detach().float().mean().item())

                        det_score = _row_scalar("spme_det_score_raw")
                        presence_prob = _row_scalar("spme_presence_prob")
                        query_cos = _row_scalar("spme_query_cos")
                        det_trk_iou = _row_scalar("spme_det_trk_iou")
                        eff_iou_score = _row_scalar("eff_iou_score")
                        obj_score_logit = _row_scalar("object_score_logits")
                        iou_score = _row_scalar("iou_score")
                        fusion_scale_mem = _row_scalar("spme_fusion_scale_mem")
                        fusion_scale_obj = _row_scalar("spme_fusion_scale_obj")
                        mem_scale_mean = _row_scalar("spme_gate_mem_scale_mean")
                        mem_offset_mean_abs = _row_scalar("spme_gate_mem_offset_mean_abs")
                        decay = _row_scalar("spme_gate_decay")
                        fusion = _row_scalar("spme_gate_fusion")
                        reinit = _row_scalar("spme_reinit")
                        reid_best_sim = _row_scalar("spme_reid_best_sim")
                        reid_trk_sim = _row_scalar("spme_reid_trk_sim")
                        reid_det_vs_trk_margin = _row_scalar("spme_reid_det_vs_trk_margin")
                        reid_margin = _row_scalar("spme_reid_margin")
                        reid_accept = _row_scalar("spme_reid_accept")
                        reid_bank_size = _row_scalar("spme_reid_bank_size")
                        skip_write = _row_scalar("spme_skip_write")

                        # Optional: compute a more paper-aligned fusion weight from intrinsic tracker signals.
                        # This affects ONLY the multi-class fused segmentation (label competition), not the
                        # per-class tracking IoU computed above.
                        if fusion_weight_mode in {"eff_iou", "reliability"}:
                            w_new: float | None = None
                            if fusion_weight_mode == "eff_iou":
                                w_new = eff_iou_score
                            else:
                                if obj_score_logit is not None and iou_score is not None:
                                    # Match SAM3's `cal_mem_score` objectness normalization:
                                    # object_score_norm = sigmoid(logit) * 2 - 1 if logit > 0 else 0
                                    if float(obj_score_logit) > 0.0:
                                        st = float(1.0 / (1.0 + np.exp(-float(obj_score_logit))))
                                        st = max(0.0, min(1.0, st * 2.0 - 1.0))
                                    else:
                                        st = 0.0
                                    ct = max(0.0, min(1.0, float(iou_score)))
                                    w_new = float(st * ct)
                            if w_new is not None:
                                video_weights[int(local_fidx)] = float(w_new)

                        if (
                            det_score is not None
                            or query_cos is not None
                            or det_trk_iou is not None
                            or eff_iou_score is not None
                            or fusion_scale_mem is not None
                            or fusion_scale_obj is not None
                            or mem_scale_mean is not None
                            or mem_offset_mean_abs is not None
                            or decay is not None
                            or fusion is not None
                            or reinit is not None
                            or reid_best_sim is not None
                            or reid_trk_sim is not None
                            or reid_det_vs_trk_margin is not None
                            or reid_margin is not None
                            or reid_accept is not None
                            or reid_bank_size is not None
                            or skip_write is not None
                        ):
                            spme_debug_by_local_frame[int(local_fidx)] = {}
                            if det_score is not None:
                                spme_debug_by_local_frame[int(local_fidx)]["det_score"] = float(det_score)
                            if presence_prob is not None:
                                spme_debug_by_local_frame[int(local_fidx)]["presence_prob"] = float(presence_prob)
                            if query_cos is not None:
                                spme_debug_by_local_frame[int(local_fidx)]["query_cos"] = float(query_cos)
                            if det_trk_iou is not None:
                                spme_debug_by_local_frame[int(local_fidx)]["det_trk_iou"] = float(det_trk_iou)
                            if eff_iou_score is not None:
                                spme_debug_by_local_frame[int(local_fidx)]["eff_iou_score"] = float(
                                    eff_iou_score
                                )
                            if fusion_scale_mem is not None:
                                spme_debug_by_local_frame[int(local_fidx)]["fusion_scale_mem"] = float(
                                    fusion_scale_mem
                                )
                            if fusion_scale_obj is not None:
                                spme_debug_by_local_frame[int(local_fidx)]["fusion_scale_obj"] = float(
                                    fusion_scale_obj
                                )
                            if mem_scale_mean is not None:
                                spme_debug_by_local_frame[int(local_fidx)][
                                    "gate_mem_scale_mean"
                                ] = float(mem_scale_mean)
                            if mem_offset_mean_abs is not None:
                                spme_debug_by_local_frame[int(local_fidx)][
                                    "gate_mem_offset_mean_abs"
                                ] = float(mem_offset_mean_abs)
                            if decay is not None:
                                spme_debug_by_local_frame[int(local_fidx)]["gate_decay"] = float(decay)
                            if fusion is not None:
                                spme_debug_by_local_frame[int(local_fidx)]["gate_fusion"] = float(fusion)
                            if reinit is not None:
                                spme_debug_by_local_frame[int(local_fidx)]["reinit"] = float(reinit)
                            if reid_best_sim is not None:
                                spme_debug_by_local_frame[int(local_fidx)]["reid_best_sim"] = float(
                                    reid_best_sim
                                )
                            if reid_trk_sim is not None:
                                spme_debug_by_local_frame[int(local_fidx)]["reid_trk_sim"] = float(
                                    reid_trk_sim
                                )
                            if reid_det_vs_trk_margin is not None:
                                spme_debug_by_local_frame[int(local_fidx)][
                                    "reid_det_vs_trk_margin"
                                ] = float(reid_det_vs_trk_margin)
                            if reid_margin is not None:
                                spme_debug_by_local_frame[int(local_fidx)]["reid_margin"] = float(
                                    reid_margin
                                )
                            if reid_accept is not None:
                                spme_debug_by_local_frame[int(local_fidx)]["reid_accept"] = float(
                                    reid_accept
                                )
                            if reid_bank_size is not None:
                                spme_debug_by_local_frame[int(local_fidx)]["reid_bank_size"] = float(
                                    reid_bank_size
                                )
                            if skip_write is not None:
                                spme_debug_by_local_frame[int(local_fidx)]["skip_write"] = float(skip_write)
                    break
        except Exception:
            # Debug stats are best-effort; never fail evaluation.
            pass
    
    # Compute metrics for each frame
    for i, (frame_idx, _, _) in enumerate(frames):
        gt = gt_masks[i]
        pred = video_segments.get(i, np.zeros_like(gt, dtype=bool))
        
        gt_present = gt.any()
        pred_present = pred.any()

        if gt_present:
            iou = compute_iou(pred, gt)
            dice = compute_dice(pred, gt)
        else:
            iou = 0.0 if pred_present else 1.0
            dice = 0.0 if pred_present else 1.0
        
        results.append(FrameResult(
            frame_idx=frame_idx,
            gt_present=bool(gt_present),
            pred_present=bool(pred_present),
            iou=iou,
            dice=dice,
            tracker_score=float(video_weights.get(i, 0.0)) if video_weights else 0.0,
            det_score=spme_debug_by_local_frame.get(i, {}).get("det_score", None),
            presence_prob=spme_debug_by_local_frame.get(i, {}).get("presence_prob", None),
            query_cos=spme_debug_by_local_frame.get(i, {}).get("query_cos", None),
            det_trk_iou=spme_debug_by_local_frame.get(i, {}).get("det_trk_iou", None),
            fusion_scale_mem=spme_debug_by_local_frame.get(i, {}).get("fusion_scale_mem", None),
            fusion_scale_obj=spme_debug_by_local_frame.get(i, {}).get("fusion_scale_obj", None),
            gate_mem_scale_mean=spme_debug_by_local_frame.get(i, {}).get("gate_mem_scale_mean", None),
            gate_mem_offset_mean_abs=spme_debug_by_local_frame.get(i, {}).get(
                "gate_mem_offset_mean_abs", None
            ),
            gate_decay=spme_debug_by_local_frame.get(i, {}).get("gate_decay", None),
            gate_fusion=spme_debug_by_local_frame.get(i, {}).get("gate_fusion", None),
            reinit=spme_debug_by_local_frame.get(i, {}).get("reinit", None),
            reid_best_sim=spme_debug_by_local_frame.get(i, {}).get("reid_best_sim", None),
            reid_trk_sim=spme_debug_by_local_frame.get(i, {}).get("reid_trk_sim", None),
            reid_det_vs_trk_margin=spme_debug_by_local_frame.get(i, {}).get(
                "reid_det_vs_trk_margin", None
            ),
            reid_margin=spme_debug_by_local_frame.get(i, {}).get("reid_margin", None),
            reid_accept=spme_debug_by_local_frame.get(i, {}).get("reid_accept", None),
            reid_bank_size=spme_debug_by_local_frame.get(i, {}).get("reid_bank_size", None),
            skip_write=spme_debug_by_local_frame.get(i, {}).get("skip_write", None),
        ))
    
    return results, video_segments, video_weights


def compute_sequence_metrics(
    results: list[FrameResult],
    sequence: str,
    class_id: int,
    drift_thr: float = 0.3,
) -> SequenceResult:
    """Aggregate frame results into sequence-level metrics."""
    
    class_name = INSTRUMENT_CLASSES.get(class_id, f"Class_{class_id}")
    
    # Filter to frames where GT is present for IoU/Dice
    present_frames = [r for r in results if r.gt_present]
    absent_frames = [r for r in results if not r.gt_present]
    
    mean_iou = float(np.mean([r.iou for r in present_frames])) if present_frames else 0.0
    mean_dice = float(np.mean([r.dice for r in present_frames])) if present_frames else 0.0
    
    # Persistence: after last GT present, how long does pred continue?
    persistence_length = None
    if present_frames and absent_frames:
        last_gt_idx = max(i for i, r in enumerate(results) if r.gt_present)
        if last_gt_idx < len(results) - 1:
            after_last = results[last_gt_idx + 1:]
            persistence_length = sum(1 for r in after_last if r.pred_present)
    
    # Drift events: consecutive frames where IoU < threshold
    drift_events = []
    current_drift = 0
    for r in results:
        if r.gt_present and r.iou < drift_thr:
            current_drift += 1
        else:
            if current_drift > 0:
                drift_events.append(current_drift)
            current_drift = 0
    if current_drift > 0:
        drift_events.append(current_drift)
    
    num_drift_events = len(drift_events)
    mean_drift_length = float(np.mean(drift_events)) if drift_events else 0.0
    max_drift_length = max(drift_events) if drift_events else 0
    
    # Recovery: after a gap (GT absent), can we recover?
    # Find gaps and check if IoU recovers after
    recovery_attempts = 0
    recovery_successes = 0
    in_gap = False
    for i, r in enumerate(results):
        if not r.gt_present:
            in_gap = True
        elif in_gap:
            # GT just reappeared after gap
            recovery_attempts += 1
            # Check if IoU >= drift_thr within next 5 frames
            for j in range(i, min(i + 5, len(results))):
                if results[j].gt_present and results[j].iou >= drift_thr:
                    recovery_successes += 1
                    break
            in_gap = False
    
    recovery_rate = (
        float(recovery_successes) / float(recovery_attempts) if recovery_attempts > 0 else 1.0
    )
    
    return SequenceResult(
        sequence=sequence,
        class_id=class_id,
        class_name=class_name,
        num_frames=len(results),
        mean_iou=mean_iou,
        mean_dice=mean_dice,
        persistence_length=persistence_length,
        num_drift_events=num_drift_events,
        mean_drift_length=mean_drift_length,
        max_drift_length=max_drift_length,
        recovery_attempts=recovery_attempts,
        recovery_successes=recovery_successes,
        recovery_rate=recovery_rate,
        frames=[asdict(r) for r in results],
    )


# ============================================================================
# Main
# ============================================================================

def _apply_spme_fusion_ckpt(video_model, spme_ckpt: Path) -> None:
    """
    Load only `spme_*` parameters from a lightweight SPME training checkpoint.
    Expected format: either a raw state_dict with spme_* keys, or a dict containing
    {"spme_state_dict": {...}}.
    """
    ckpt = torch.load(str(spme_ckpt), map_location="cpu")
    if (
        isinstance(ckpt, dict)
        and "spme_state_dict" in ckpt
        and isinstance(ckpt["spme_state_dict"], dict)
    ):
        sd = ckpt["spme_state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise TypeError(f"Unsupported SPME checkpoint format: {spme_ckpt}")

    model_sd = video_model.state_dict()
    loaded = 0
    skipped_missing = 0
    skipped_shape = 0
    for k, v in sd.items():
        if not k.startswith("spme_"):
            continue
        if k not in model_sd:
            skipped_missing += 1
            continue
        if not torch.is_tensor(v) or model_sd[k].shape != v.shape:
            skipped_shape += 1
            continue
        model_sd[k].copy_(v)
        loaded += 1
    print(
        f"Loaded SPME params: loaded={loaded} skipped_missing={skipped_missing} "
        f"skipped_shape={skipped_shape} from {spme_ckpt}"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAM3 on EndoVis 2018")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to endovis2018 directory (canonical layout)")
    parser.add_argument("--sequences", type=str, nargs="+", default=["val1"],
                        help="Sequences to evaluate (e.g., val1 val2 val3)")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--base-sam3-pt", type=str, required=True,
                        help="Path to sam3.pt base model")
    parser.add_argument("--overlay-ckpt", type=str, default=None,
                        help="Optional finetuned checkpoint to overlay on top of base-sam3-pt (loaded with strict=False).")
    parser.add_argument(
        "--spme-ckpt",
        type=str,
        action="append",
        default=None,
        help="Optional: path to a trained SPME checkpoint (loads only spme_* params). "
        "Can be specified multiple times; later checkpoints override earlier ones. "
        "If provided, this takes precedence over --spme-fusion-ckpt.",
    )
    parser.add_argument(
        "--spme-fusion-ckpt",
        type=str,
        default=None,
        help="Optional (deprecated): path to a trained SPME checkpoint (loads only spme_* params). "
        "Prefer --spme-ckpt (repeatable) for composing fusion+gate checkpoints.",
    )
    parser.add_argument("--prompt", type=str, default="surgical instrument",
                        help="Text prompt for all classes")
    parser.add_argument(
        "--init-prompt",
        type=str,
        default="box",
        choices=["box", "mask"],
        help="Initialization prompt. 'mask' uses the GT first-visible mask (paper-aligned); "
             "'box' uses a box derived from that mask.",
    )
    parser.add_argument("--class-specific-prompts", action="store_true",
                        help="Use class-specific prompts instead of generic")
    parser.add_argument("--classes", type=int, nargs="*", default=None,
                        help="Specific classes to track (e.g., 3 6). Default: all present.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--drift-thr", type=float, default=0.3,
                        help="IoU threshold for drift detection")
    parser.add_argument("--hotstart-delay", type=int, default=None,
                        help="Override SAM3 video hotstart_delay (default: model default, often 15). "
                             "Setting 0 disables hotstart buffering/filtering and can reduce all-zero outputs.")
    parser.add_argument(
        "--init-frame-mode",
        type=str,
        default="first_present",
        choices=["first_present", "first_min_area", "max_area"],
        help="How to pick the init frame from GT: first_present (default), first_min_area, max_area. "
             "Note: max_area peeks into future GT frames and should be used only for debugging.",
    )
    parser.add_argument(
        "--init-select-mode",
        type=str,
        default="prob",
        choices=["prob", "bbox_iou", "gt_iou"],
        help="How to pick target obj_id on the init frame when multiple masks are returned. "
             "prob: highest model confidence (recommended). "
             "bbox_iou: best overlap with the provided init bbox (no GT leakage). "
             "gt_iou: debug-only (uses GT mask; do not report).",
    )
    parser.add_argument(
        "--init-min-area",
        type=int,
        default=0,
        help="Used only when --init-frame-mode=first_min_area (mask pixel count).",
    )
    parser.add_argument(
        "--init-box-pad",
        type=int,
        default=0,
        help="Pad GT bbox by this many pixels (clamped to image bounds).",
    )
    parser.add_argument("--new-det-thr", type=float, default=None,
                        help="Override SAM3 new_det_thresh (default: model default, often 0.7). "
                             "If you see all-zero outputs on EndoVis, try 0.3 or 0.0.")
    parser.add_argument("--score-thr-detection", type=float, default=None,
                        help="Override SAM3 score_threshold_detection (default: model default, often 0.5). "
                             "Lowering can help domain-shift datasets.")
    parser.add_argument("--debug-init", action="store_true",
                        help="Print init-frame candidate stats when selecting target obj_id.")
    parser.add_argument(
        "--propagation-mode",
        type=str,
        default="vg",
        choices=["vg", "tracker"],
        help="How to propagate after initialization. "
             "vg (recommended): SAM3 detector+tracker propagation (applies SPME edits). "
             "tracker: Tracker-only propagation from the instance-interactivity path (bypasses SPME).",
    )
    parser.add_argument(
        "--write-fused-masks",
        action="store_true",
        help="Also write fused multi-class label masks as PNGs (can be large).",
    )
    parser.add_argument(
        "--fusion-weight-mode",
        type=str,
        default=os.getenv("ENDOVIS_FUSION_WEIGHT_MODE", "tracker_prob"),
        choices=["tracker_prob", "det_prob", "eff_iou", "reliability"],
        help="Quality weight used for multi-class fusion. "
             "tracker_prob: per-frame tracker confidence (default). "
             "det_prob: per-frame detector probability. "
             "eff_iou: Tracker eff_iou_score (intrinsic reliability proxy). "
             "reliability: st*ct computed from (object_score_logits, iou_score) to match SAM3 cal_mem_score.",
    )
    parser.add_argument(
        "--debug-spme-summary",
        action="store_true",
        help="Print + write an aggregated SPME debug summary (gate/fusion stats, ghost/miss) to out_dir.",
    )
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Build predictor
    print("Loading SAM3 model...")
    model = build_sam3_video_model(
        checkpoint_path=args.base_sam3_pt,
        load_from_HF=False,
        device=args.device,
    )
    if args.overlay_ckpt:
        print(f"Loading overlay checkpoint (strict=False): {args.overlay_ckpt}")
        ckpt_obj = torch.load(args.overlay_ckpt, map_location="cpu", weights_only=True)
        if (
            isinstance(ckpt_obj, dict)
            and "model" in ckpt_obj
            and isinstance(ckpt_obj["model"], dict)
        ):
            ckpt_state = ckpt_obj["model"]
        elif isinstance(ckpt_obj, dict):
            ckpt_state = ckpt_obj
        else:
            raise TypeError(
                f"Unsupported checkpoint format: type={type(ckpt_obj)} at {args.overlay_ckpt}"
            )

        # EndoVis finetuning uses `build_sam3_image_model`, whose state_dict keys do NOT have
        # the "detector." prefix. The video model expects "detector.*" keys. Map if needed.
        if any(k.startswith("detector.") for k in ckpt_state.keys()):
            overlay_state = ckpt_state  # already video-style
        else:
            overlay_state = {f"detector.{k}": v for k, v in ckpt_state.items()}

        model_sd_keys = set(model.state_dict().keys())
        raw_ckpt_keys = len(ckpt_state)
        overlay_keys_before_filter = len(overlay_state)
        overlay_state = {k: v for k, v in overlay_state.items() if k in model_sd_keys}
        overlay_keys_after_filter = len(overlay_state)
        missing_keys, unexpected_keys = model.load_state_dict(overlay_state, strict=False)
        loaded_keys = len(overlay_state) - len(unexpected_keys)
        print(
            f"Overlay loaded. raw_ckpt_keys={raw_ckpt_keys} mapped_keys={overlay_keys_before_filter} "
            f"matched_keys={overlay_keys_after_filter} loaded_keys={loaded_keys} "
            f"missing_keys={len(missing_keys)} unexpected_keys={len(unexpected_keys)}"
        )

    spme_ckpts: list[Path] = []
    if args.spme_ckpt:
        spme_ckpts = [Path(p) for p in args.spme_ckpt if p]
    elif args.spme_fusion_ckpt:
        spme_ckpts = [Path(args.spme_fusion_ckpt)]

    for ckpt_path in spme_ckpts:
        _apply_spme_fusion_ckpt(model, ckpt_path)
    if args.new_det_thr is not None:
        model.new_det_thresh = float(args.new_det_thr)
        print(f"[config] new_det_thresh={model.new_det_thresh}")
    if args.score_thr_detection is not None:
        model.score_threshold_detection = float(args.score_thr_detection)
        print(f"[config] score_threshold_detection={model.score_threshold_detection}")
    if args.hotstart_delay is not None:
        model.hotstart_delay = int(args.hotstart_delay)
        print(f"[config] hotstart_delay={model.hotstart_delay}")
    elif str(args.init_prompt).strip().lower() == "mask" and str(args.propagation_mode).strip().lower() == "vg":
        # For mask-init tracking, hotstart heuristics (meant to suppress spurious *new* detections)
        # can accidentally suppress the init object if detector↔tracker matching is weak early on.
        model.hotstart_delay = 0
        print(f"[config] hotstart_delay={model.hotstart_delay} (auto for mask-init + vg)")
    
    all_results = []
    fused_results = []
    
    for seq_name in args.sequences:
        print(f"\n=== Processing {seq_name} ===")
        
        frames = get_sequence_frames(data_root, seq_name)
        if not frames:
            print(f"  No frames found, skipping")
            continue
        
        print(f"  Found {len(frames)} frames")
        
        # Find classes present in first frame
        first_mask = load_bmp_mask(frames[0][2])
        classes_in_first = get_classes_in_mask(first_mask)
        
        # Also check all frames to find classes that appear later
        all_classes = set()
        for _, _, label_path in frames:
            mask = load_bmp_mask(label_path)
            all_classes.update(get_classes_in_mask(mask))
        
        print(f"  Classes in first frame: {classes_in_first}")
        print(f"  Classes in sequence: {sorted(all_classes)}")

        # Load GT labels once (for multi-class fusion eval).
        gt_labels_seq: list[np.ndarray] = []
        for _, _, label_path in frames:
            gt_labels_seq.append(load_bmp_mask(label_path))
        if not gt_labels_seq:
            print("  [warn] No GT labels loaded, skipping fusion metrics")
            gt_h = gt_w = None
        else:
            gt_h, gt_w = gt_labels_seq[0].shape[:2]
        
        # Filter classes if specified
        if args.classes:
            classes_to_track = [c for c in args.classes if c in all_classes]
        else:
            classes_to_track = sorted(all_classes)

        pred_masks_by_class: dict[int, dict[int, np.ndarray]] = {}
        pred_weights_by_class: dict[int, dict[int, float]] = {}
        
        for class_id in classes_to_track:
            class_name = INSTRUMENT_CLASSES.get(class_id, f"Class_{class_id}")
            print(f"\n  Tracking class {class_id}: {class_name}")
            
            # Determine prompt
            if args.class_specific_prompts:
                prompt = class_name.lower()
            else:
                prompt = args.prompt
            
            print(f"    Prompt: '{prompt}'")
            
            # Run tracking
            frame_results, pred_masks, pred_weights = run_sam3_tracking(
                model=model,
                frames=frames,
                class_id=class_id,
                prompt=prompt,
                init_prompt=args.init_prompt,
                init_frame_mode=args.init_frame_mode,
                init_select_mode=args.init_select_mode,
                init_min_area=args.init_min_area,
                init_box_pad=args.init_box_pad,
                debug_init=args.debug_init,
                propagation_mode=args.propagation_mode,
                fusion_weight_mode=args.fusion_weight_mode,
            )
            pred_masks_by_class[int(class_id)] = pred_masks
            pred_weights_by_class[int(class_id)] = pred_weights
            
            # Compute metrics
            seq_result = compute_sequence_metrics(
                results=frame_results,
                sequence=seq_name,
                class_id=class_id,
                drift_thr=args.drift_thr,
            )
            
            # Print summary
            print(f"    IoU: {seq_result.mean_iou:.3f}, Dice: {seq_result.mean_dice:.3f}")
            print(f"    Drift events: {seq_result.num_drift_events}, max len: {seq_result.max_drift_length}")
            if seq_result.persistence_length is not None:
                print(f"    Persistence: {seq_result.persistence_length} frames")
            print(f"    Recovery rate: {seq_result.recovery_rate:.2f} ({seq_result.recovery_successes}/{seq_result.recovery_attempts})")
            
            all_results.append(seq_result)
            
            # Save per-class results
            class_out_dir = out_dir / seq_name / f"class_{class_id}"
            class_out_dir.mkdir(parents=True, exist_ok=True)
            
            with open(class_out_dir / "results.json", "w") as f:
                json.dump(asdict(seq_result), f, indent=2)

        # ---------------------------------------------------------------------
        # Multi-class fused evaluation (paper-aligned: quality-weighted mask fusion)
        # ---------------------------------------------------------------------
        if gt_labels_seq and gt_h is not None and gt_w is not None and classes_to_track:
            fused_labels = fuse_class_masks_quality_weighted(
                class_ids=classes_to_track,
                pred_masks_by_class=pred_masks_by_class,
                pred_weights_by_class=pred_weights_by_class,
                num_frames=len(frames),
                h=int(gt_h),
                w=int(gt_w),
            )
            fused_metrics = compute_fused_metrics(
                gt_labels=gt_labels_seq,
                fused_labels=fused_labels,
                class_ids=classes_to_track,
            )
            fused_metrics.update(
                {
                    "sequence": seq_name,
                    "classes": [int(c) for c in classes_to_track],
                    "prompt": args.prompt,
                    "class_specific_prompts": bool(args.class_specific_prompts),
                    "init_frame_mode": str(args.init_frame_mode),
                    "init_select_mode": str(args.init_select_mode),
                    "init_min_area": int(args.init_min_area),
                    "init_box_pad": int(args.init_box_pad),
                }
            )
            fused_dir = out_dir / seq_name / "fused"
            fused_dir.mkdir(parents=True, exist_ok=True)
            with open(fused_dir / "summary.json", "w") as f:
                json.dump(fused_metrics, f, indent=2)
            fused_results.append(fused_metrics)
            print(
                f"\n  [fused] challenge_iou={fused_metrics['challenge_iou']:.3f} "
                f"mcIoU_present={fused_metrics['mean_class_iou_present']:.3f} "
                f"mcIoU_all={fused_metrics.get('mean_class_iou', 0.0):.3f} "
                f"binary_iou={fused_metrics['binary_iou']:.3f}"
            )

            if args.write_fused_masks:
                masks_dir = fused_dir / "masks"
                masks_dir.mkdir(parents=True, exist_ok=True)
                for (frame_idx, _, _), fused in zip(frames, fused_labels):
                    Image.fromarray(fused).save(masks_dir / f"frame{int(frame_idx):04d}.png")
    
    # Save summary
    summary = {
        "sequences": args.sequences,
        "prompt": args.prompt,
        "num_results": len(all_results),
        "mean_iou": float(np.mean([r.mean_iou for r in all_results])) if all_results else 0.0,
        "mean_dice": float(np.mean([r.mean_dice for r in all_results])) if all_results else 0.0,
        "mean_drift_events": float(np.mean([r.num_drift_events for r in all_results])) if all_results else 0.0,
        "mean_max_drift": float(np.mean([r.max_drift_length for r in all_results])) if all_results else 0.0,
        "mean_recovery_rate": float(np.mean([r.recovery_rate for r in all_results])) if all_results else 0.0,
        "persistence_lengths": [r.persistence_length for r in all_results if r.persistence_length is not None],
        "fused_mean_challenge_iou": float(np.mean([r["challenge_iou"] for r in fused_results])) if fused_results else 0.0,
        "fused_mean_class_iou_present": float(np.mean([r["mean_class_iou_present"] for r in fused_results])) if fused_results else 0.0,
        "fused_mean_class_iou": float(np.mean([r.get("mean_class_iou", 0.0) for r in fused_results])) if fused_results else 0.0,
        "fused_mean_binary_iou": float(np.mean([r["binary_iou"] for r in fused_results])) if fused_results else 0.0,
    }
    
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== Summary ===")
    print(f"Mean IoU: {summary['mean_iou']:.3f}")
    print(f"Mean Dice: {summary['mean_dice']:.3f}")
    print(f"Mean drift events: {summary['mean_drift_events']:.1f}")
    print(f"Mean max drift: {summary['mean_max_drift']:.1f}")
    print(f"Mean recovery rate: {summary['mean_recovery_rate']:.2f}")

    if args.debug_spme_summary and all_results:
        # Aggregate per-frame SPME debug signals across all tracked objects.
        all_frames = []
        for r in all_results:
            if getattr(r, "frames", None):
                all_frames.extend(r.frames)

        def _get(fr, key: str, default=None):
            # `TrackResult.frames` may contain either dataclass objects (attribute access)
            # or plain dicts (JSON-ready). Support both.
            if isinstance(fr, dict):
                return fr.get(key, default)
            return getattr(fr, key, default)

        def _num_list(attr: str):
            vals = []
            for fr in all_frames:
                v = _get(fr, attr, None)
                if v is None:
                    continue
                try:
                    vals.append(float(v))
                except Exception:
                    continue
            return vals

        def _stats(vals: list[float]):
            if not vals:
                return None
            arr = np.asarray(vals, dtype=np.float64)
            nan_mask = np.isnan(arr)
            valid = arr[~nan_mask]
            if valid.size == 0:
                # Keep JSON standard-compliant (avoid NaN literals).
                return {
                    "n": int(arr.size),
                    "nan": int(nan_mask.sum()),
                    "nan_frac": float(nan_mask.mean()) if arr.size else 0.0,
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "unique": 0,
                }
            return {
                "n": int(arr.size),
                "nan": int(nan_mask.sum()),
                "nan_frac": float(nan_mask.mean()) if arr.size else 0.0,
                "mean": float(valid.mean()),
                "std": float(valid.std()),
                "min": float(valid.min()),
                "max": float(valid.max()),
                "unique": int(np.unique(valid).size),
            }

        # Presence/absence error modes from the same `gt_present/pred_present` fields used in results.json.
        ghost = 0
        miss = 0
        reinit_on_gt_present = 0
        reinit_on_gt_absent = 0
        fusion_on_gt_present = 0
        fusion_on_gt_absent = 0
        fusion_attr_present = False
        skip_on_gt_present = 0
        skip_on_gt_absent = 0
        skip_attr_present = False
        tot = 0
        gt_present_frames = 0
        for fr in all_frames:
            gt = bool(_get(fr, "gt_present", False))
            pred = bool(_get(fr, "pred_present", False))
            reinit_v = _get(fr, "reinit", 0.0)
            fusion_v = _get(fr, "fusion_scale_mem", None)
            skip_v = _get(fr, "skip_write", None)
            if fusion_v is not None:
                fusion_attr_present = True
            if skip_v is not None:
                skip_attr_present = True
            try:
                reinit_on = float(reinit_v) > 0.0
            except Exception:
                reinit_on = False
            try:
                fusion_on = float(fusion_v) > 0.0 if fusion_v is not None else False
            except Exception:
                fusion_on = False
            try:
                skip_on = float(skip_v) > 0.0 if skip_v is not None else False
            except Exception:
                skip_on = False
            tot += 1
            if gt:
                gt_present_frames += 1
                if reinit_on:
                    reinit_on_gt_present += 1
                if fusion_on:
                    fusion_on_gt_present += 1
                if skip_on:
                    skip_on_gt_present += 1
            else:
                if reinit_on:
                    reinit_on_gt_absent += 1
                if fusion_on:
                    fusion_on_gt_absent += 1
                if skip_on:
                    skip_on_gt_absent += 1
            if (not gt) and pred:
                ghost += 1
            if gt and (not pred):
                miss += 1

        fusion_scale_mem = _num_list("fusion_scale_mem")
        fusion_on_frac = (
            float(np.mean(np.asarray(fusion_scale_mem) > 0.0)) if fusion_scale_mem else None
        )
        gt_absent_frames = tot - gt_present_frames
        fusion_on_frac_gt_present = (
            float(fusion_on_gt_present / gt_present_frames)
            if fusion_attr_present and gt_present_frames
            else None
        )
        fusion_on_frac_gt_absent = (
            float(fusion_on_gt_absent / gt_absent_frames)
            if fusion_attr_present and gt_absent_frames > 0
            else None
        )
        reinit_vals = _num_list("reinit")
        reinit_on_frac = (
            float(np.mean(np.asarray(reinit_vals) > 0.0)) if reinit_vals else None
        )
        reinit_on_frac_gt_present = (
            float(reinit_on_gt_present / gt_present_frames) if gt_present_frames else None
        )
        reinit_on_frac_gt_absent = (
            float(reinit_on_gt_absent / gt_absent_frames) if gt_absent_frames > 0 else None
        )
        skip_write_vals = _num_list("skip_write")
        skip_write_on_frac = (
            float(np.mean(np.asarray(skip_write_vals) > 0.0)) if skip_write_vals else None
        )
        skip_write_on_frac_gt_present = (
            float(skip_on_gt_present / gt_present_frames)
            if skip_attr_present and gt_present_frames
            else None
        )
        skip_write_on_frac_gt_absent = (
            float(skip_on_gt_absent / gt_absent_frames)
            if skip_attr_present and gt_absent_frames > 0
            else None
        )

        spme_debug = {
            "gate_mem_scale_mean": _stats(_num_list("gate_mem_scale_mean")),
            "gate_mem_offset_mean_abs": _stats(_num_list("gate_mem_offset_mean_abs")),
            "gate_decay": _stats(_num_list("gate_decay")),
            "gate_fusion": _stats(_num_list("gate_fusion")),
            "det_trk_iou": _stats(_num_list("det_trk_iou")),
            "reid_best_sim": _stats(_num_list("reid_best_sim")),
            "reid_trk_sim": _stats(_num_list("reid_trk_sim")),
            "reid_det_vs_trk_margin": _stats(_num_list("reid_det_vs_trk_margin")),
            "reid_margin": _stats(_num_list("reid_margin")),
            "reid_accept": _stats(_num_list("reid_accept")),
            "reid_bank_size": _stats(_num_list("reid_bank_size")),
            "fusion_scale_mem": _stats(fusion_scale_mem),
            "fusion_scale_obj": _stats(_num_list("fusion_scale_obj")),
            "fusion_on_frac": fusion_on_frac,
            "fusion_on_frac_gt_present": fusion_on_frac_gt_present,
            "fusion_on_frac_gt_absent": fusion_on_frac_gt_absent,
            "reinit": _stats(reinit_vals),
            "reinit_on_frac": reinit_on_frac,
            "reinit_on_frac_gt_present": reinit_on_frac_gt_present,
            "reinit_on_frac_gt_absent": reinit_on_frac_gt_absent,
            "reinit_on_gt_present_frames": int(reinit_on_gt_present),
            "reinit_on_gt_absent_frames": int(reinit_on_gt_absent),
            "skip_write": _stats(skip_write_vals),
            "skip_write_on_frac": skip_write_on_frac,
            "skip_write_on_frac_gt_present": skip_write_on_frac_gt_present,
            "skip_write_on_frac_gt_absent": skip_write_on_frac_gt_absent,
            "skip_write_on_gt_present_frames": int(skip_on_gt_present),
            "skip_write_on_gt_absent_frames": int(skip_on_gt_absent),
            "ghost_frames": int(ghost),
            "ghost_rate_all": float(ghost / tot) if tot else 0.0,
            "miss_frames": int(miss),
            "miss_rate_gt_present": float(miss / gt_present_frames) if gt_present_frames else 0.0,
            "frames_total": int(tot),
            "frames_gt_present": int(gt_present_frames),
        }

        with open(out_dir / "spme_debug_summary.json", "w") as f:
            json.dump(spme_debug, f, indent=2)

        print("\n=== SPME Debug Summary ===")
        print(json.dumps(spme_debug, indent=2))

    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
