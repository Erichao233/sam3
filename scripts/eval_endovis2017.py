#!/usr/bin/env python3
"""
EndoVis 2017 Instrument Segmentation Evaluation with SAM3.

Key features:
- Tracks each instrument class as a separate single-object problem
- Computes IoU, Dice, persistence, recovery metrics
- Supports SPME-W (write gate) and SPME-F (fusion) via environment variables

Usage:
    python scripts/eval_endovis2017.py \
        --data-root /path/to/endovis2017 \
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


# ============================================================================
# Data loading
# ============================================================================

INSTRUMENT_CLASSES = {
    1: "Bipolar Forceps",
    2: "Prograsp Forceps",
    3: "Large Needle Driver",
    4: "Vessel Sealer",
    5: "Grasping Retractor",
    6: "Monopolar Curved Scissors",
    7: "Other",
}


def load_bmp_mask(path: Path) -> np.ndarray:
    """Load BMP mask where pixel value = class ID."""
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
    
    frames = []
    for label_path in sorted(label_dir.glob("*.bmp")):
        # Parse frame index from filename: seq_X_frameYYY.bmp
        stem = label_path.stem
        parts = stem.split("_")
        frame_idx = int(parts[-1].replace("frame", ""))
        
        # Find corresponding image (may be .jpg or .png)
        image_path = None
        for ext in [".jpg", ".png", ".bmp"]:
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


@dataclass
class FrameResult:
    frame_idx: int
    gt_present: bool
    pred_present: bool
    iou: float
    dice: float
    det_score: Optional[float] = None
    query_cos: Optional[float] = None
    tracker_score: Optional[float] = None


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
    init_frame_mode: str = "first_present",
    init_min_area: int = 0,
    init_box_pad: int = 0,
    debug_init: bool = False,
) -> list[FrameResult]:
    """
    Run SAM3 video tracking for a single class.
    
    Args:
        model: SAM3 video model (Sam3VideoInferenceWithInstanceInteractivity)
        frames: List of (frame_idx, image_path, label_path)
        class_id: Which class to track
        prompt: Text prompt
    
    Returns:
        List of FrameResult for each frame
    """
    results = []
    
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
        return results

    # Initialize inference state (use a list of PIL images to preserve the frame ordering)
    with torch.inference_mode():
        state = model.init_state(resource_path=images_pil, offload_video_to_cpu=False)

        # Add prompt on the init frame using the GT bbox (normalized xywh in [0,1])
        w, h = images_pil[0].size
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
                f"[init] class_id={class_id} init_frame_mode={init_frame_mode} "
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

        # Pick a single object to track (if multiple are produced): best IoU on init frame
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
            gt0 = gt_masks[init_frame_idx]
            best_iou = -1.0
            cand_ious = []
            for obj_id, mask in zip(out0_obj_ids, out0_masks):
                iou = compute_iou(mask.astype(bool), gt0)
                cand_ious.append((float(iou), int(obj_id)))
                if iou > best_iou:
                    best_iou = iou
                    target_obj_id = int(obj_id)
            cand_ious.sort(reverse=True, key=lambda x: x[0])
            if debug_init or best_iou <= 0.0:
                top = cand_ious[: min(5, len(cand_ious))]
                print(
                    f"[init-debug] class_id={class_id} prompt='{prompt}' "
                    f"init_bbox_xyxy={init_bbox} num_cands={len(cand_ious)} "
                    f"best_iou={best_iou:.4f} best_obj_id={target_obj_id} top5={top}"
                )
                if len(out0_probs) == len(out0_obj_ids):
                    probs_top = sorted(
                        [(float(p), int(oid)) for p, oid in zip(out0_probs, out0_obj_ids)],
                        reverse=True,
                        key=lambda x: x[0],
                    )[:5]
                    print(f"[init-debug] top5 by out_probs: {probs_top}")
                if len(out0_boxes) == len(out0_obj_ids):
                    # print first few boxes for sanity (normalized xywh)
                    boxes_preview = [
                        (int(oid), [float(x) for x in box])
                        for oid, box in list(zip(out0_obj_ids, out0_boxes))[:5]
                    ]
                    print(f"[init-debug] boxes_preview (obj_id, xywh_norm): {boxes_preview}")

        # Propagate through video and store predicted mask for the chosen object id.
        # If init_frame_idx is not the first GT-present frame, run backward propagation too.
        video_segments: dict[int, np.ndarray] = {}
        def _update_segments(out_frame_idx, out):
            out_obj_ids = out.get("out_obj_ids", np.zeros(0, dtype=np.int64))
            out_masks = out.get("out_binary_masks", np.zeros((0, h, w), dtype=bool))

            pred_mask = np.zeros_like(gt_masks[0], dtype=bool)
            if target_obj_id is not None and len(out_obj_ids) > 0:
                match = np.where(out_obj_ids.astype(np.int64) == np.int64(target_obj_id))[0]
                if len(match) > 0:
                    pred_mask = out_masks[int(match[0])].astype(bool)
            video_segments[int(out_frame_idx)] = pred_mask

        for out_frame_idx, out in model.propagate_in_video(
            state,
            start_frame_idx=init_frame_idx,
            max_frame_num_to_track=len(images_pil),
            reverse=False,
        ):
            _update_segments(out_frame_idx, out)

        has_gt_present_before = any(a > 0 for a in areas[:init_frame_idx])
        if has_gt_present_before:
            for out_frame_idx, out in model.propagate_in_video(
                state,
                start_frame_idx=init_frame_idx,
                max_frame_num_to_track=len(images_pil),
                reverse=True,
            ):
                _update_segments(out_frame_idx, out)
    
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
        ))
    
    return results


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

def main():
    parser = argparse.ArgumentParser(description="Evaluate SAM3 on EndoVis 2017")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to endovis2017 directory")
    parser.add_argument("--sequences", type=str, nargs="+", default=["val1"],
                        help="Sequences to evaluate (e.g., val1 val2 val3)")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--base-sam3-pt", type=str, required=True,
                        help="Path to sam3.pt base model")
    parser.add_argument("--overlay-ckpt", type=str, default=None,
                        help="Optional finetuned checkpoint to overlay on top of base-sam3-pt (loaded with strict=False).")
    parser.add_argument("--prompt", type=str, default="surgical instrument",
                        help="Text prompt for all classes")
    parser.add_argument("--class-specific-prompts", action="store_true",
                        help="Use class-specific prompts instead of generic")
    parser.add_argument("--classes", type=int, nargs="*", default=None,
                        help="Specific classes to track (e.g., 3 6). Default: all present.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--drift-thr", type=float, default=0.3,
                        help="IoU threshold for drift detection")
    parser.add_argument(
        "--init-frame-mode",
        type=str,
        default="first_present",
        choices=["first_present", "first_min_area", "max_area"],
        help="How to pick the init frame from GT: first_present (default), first_min_area, max_area. "
             "For irregular/occluded tools, max_area is often more stable.",
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
        ckpt = torch.load(args.overlay_ckpt, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
        print(f"Overlay loaded. missing_keys={len(missing_keys)} unexpected_keys={len(unexpected_keys)}")
    if args.new_det_thr is not None:
        model.new_det_thresh = float(args.new_det_thr)
        print(f"[config] new_det_thresh={model.new_det_thresh}")
    if args.score_thr_detection is not None:
        model.score_threshold_detection = float(args.score_thr_detection)
        print(f"[config] score_threshold_detection={model.score_threshold_detection}")
    
    all_results = []
    
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
        
        # Filter classes if specified
        if args.classes:
            classes_to_track = [c for c in args.classes if c in all_classes]
        else:
            classes_to_track = sorted(all_classes)
        
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
            frame_results = run_sam3_tracking(
                model=model,
                frames=frames,
                class_id=class_id,
                prompt=prompt,
                init_frame_mode=args.init_frame_mode,
                init_min_area=args.init_min_area,
                init_box_pad=args.init_box_pad,
                debug_init=args.debug_init,
            )
            
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
    }
    
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== Summary ===")
    print(f"Mean IoU: {summary['mean_iou']:.3f}")
    print(f"Mean Dice: {summary['mean_dice']:.3f}")
    print(f"Mean drift events: {summary['mean_drift_events']:.1f}")
    print(f"Mean max drift: {summary['mean_max_drift']:.1f}")
    print(f"Mean recovery rate: {summary['mean_recovery_rate']:.2f}")
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
