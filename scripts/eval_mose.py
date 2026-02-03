from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sam3.model_builder import build_sam3_video_model


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as m:
        return np.array(m)


def _get_obj_ids(mask: np.ndarray) -> list[int]:
    ids = np.unique(mask)
    return [int(i) for i in ids.tolist() if int(i) > 0]


def _mask_to_box_xyxy(mask: np.ndarray, obj_id: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask == int(obj_id))
    if ys.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2, y2


def _mask_to_click_xy(mask: np.ndarray, obj_id: int) -> tuple[float, float] | None:
    ys, xs = np.where(mask == int(obj_id))
    if ys.size == 0:
        return None
    # Median is more robust than mean and more likely to land inside the object.
    x = float(np.median(xs))
    y = float(np.median(ys))
    return x, y


def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0  # both empty => perfect
    return float(inter / union)


def _load_mose_palette(mose_root: Path) -> list[int] | None:
    # MOSE provides a DAVIS-style palette; sample submission masks include it.
    sample = mose_root / "sample_submission_valid_all"
    if not sample.exists():
        return None
    first = next(sample.glob("*/00000.png"), None)
    if first is None:
        return None
    with Image.open(first) as m:
        pal = m.getpalette()
    return pal


def _save_palette_mask(path: Path, mask: np.ndarray, palette: list[int] | None) -> None:
    out = Image.fromarray(mask.astype(np.uint8), mode="P")
    if palette:
        out.putpalette(palette)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mose-root", required=True, type=str)
    ap.add_argument("--split", default="train", choices=["train", "valid"])
    ap.add_argument("--seqs", default="", type=str, help="space-separated video IDs; empty => auto-scan")
    ap.add_argument("--out-dir", required=True, type=str)
    ap.add_argument("--base-sam3-pt", required=True, type=str)
    ap.add_argument("--finetune-ckpt", default=None, type=str)
    ap.add_argument("--bpe-path", required=True, type=str)
    ap.add_argument("--auto-prompt-mode", default="gt_click", choices=["gt_click", "gt_box"])
    ap.add_argument("--max-frames", default=None, type=int)
    ap.add_argument("--save-pred-masks", action="store_true", help="write DAVIS-style PNG masks per frame")
    ap.add_argument("--device", default="cuda", type=str)
    args = ap.parse_args()

    mose_root = Path(args.mose_root)
    split_dir = mose_root / args.split
    jpeg_dir = split_dir / "JPEGImages"
    anno_dir = split_dir / "Annotations"

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Select sequences.
    if args.seqs:
        seq_names = [s.strip() for s in args.seqs.split(" ") if s.strip()]
    else:
        seq_names = sorted([d.name for d in jpeg_dir.iterdir() if d.is_dir()])
        if len(seq_names) > 5:
            print(
                f"No seqs specified, running on first 5 of {len(seq_names)}: {seq_names[:5]}"
            )
            seq_names = seq_names[:5]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading model... base={args.base_sam3_pt} device={device}")
    model = build_sam3_video_model(
        checkpoint_path=args.base_sam3_pt,
        load_from_HF=False,
        bpe_path=args.bpe_path,
        device=str(device),
        compile=False,
    )
    if args.finetune_ckpt and args.finetune_ckpt != args.base_sam3_pt:
        print(f"Applying finetune ckpt: {args.finetune_ckpt}")
        ckpt = torch.load(args.finetune_ckpt, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]
        msg = model.load_state_dict(ckpt, strict=False)
        print(f"Finetune load msg: {msg}")

    model.to(device)
    model.eval()

    palette = _load_mose_palette(mose_root)
    results: dict[str, Any] = {}
    global_iou_sum = 0.0
    global_frames = 0

    for seq_name in tqdm(seq_names, desc="MOSE"):
        img_folder = jpeg_dir / seq_name
        anno_folder = anno_dir / seq_name
        frame_names = sorted([f.name for f in img_folder.glob("*.jpg")])
        if not frame_names:
            print(f"[WARN] Skipping {seq_name}: no frames found.")
            continue

        max_frames = int(args.max_frames) if args.max_frames else None
        if max_frames is not None:
            frame_names = frame_names[:max_frames]

        # Init state (video inference API expects `resource_path`).
        inference_state = model.init_state(
            resource_path=str(img_folder),
            offload_video_to_cpu=True,
            async_loading_frames=False,
            video_loader_type="cv2",
        )
        model.reset_state(inference_state)

        # Use visual prompt token (no language supervision in MOSE v1).
        inference_state["text_prompt"] = None
        inference_state["input_batch"].find_text_batch[0] = "<text placeholder>"
        for t in range(inference_state["num_frames"]):
            inference_state["input_batch"].find_inputs[t].text_ids[...] = model.TEXT_ID_FOR_VISUAL

        # First-frame annotation provides the objects to track.
        gt0_path = anno_folder / frame_names[0].replace(".jpg", ".png")
        if not gt0_path.exists():
            print(f"[WARN] Skipping {seq_name}: missing first-frame annotation {gt0_path}.")
            continue
        gt0 = _load_mask(gt0_path)
        seq_palette = palette
        if seq_palette is None:
            with Image.open(gt0_path) as m:
                seq_palette = m.getpalette()
        obj_ids = _get_obj_ids(gt0)
        if not obj_ids:
            print(f"[WARN] Skipping {seq_name}: no objects in first-frame annotation.")
            continue

        H0, W0 = gt0.shape[:2]

        # Add per-object prompts on frame 0 (keeps GT ids stable across propagation).
        for obj_id in obj_ids:
            if args.auto_prompt_mode == "gt_box":
                box = _mask_to_box_xyxy(gt0, obj_id)
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                pts = np.array([[x1 / W0, y1 / H0], [x2 / W0, y2 / H0]], dtype=np.float32)
                labels = np.array([2, 3], dtype=np.int32)
            else:
                click = _mask_to_click_xy(gt0, obj_id)
                if click is None:
                    continue
                x, y = click
                pts = np.array([[x / W0, y / H0]], dtype=np.float32)
                labels = np.array([1], dtype=np.int32)

            _ = model.add_prompt(
                inference_state,
                frame_idx=0,
                points=pts,
                point_labels=labels,
                obj_id=int(obj_id),
                rel_coordinates=True,
            )

        seq_iou_sum = 0.0
        seq_eval_frames = 0
        seq_pred_dir = out_root / "pred_masks" / seq_name

        # Propagate and evaluate.
        max_track = (len(frame_names) - 1) if max_frames is not None else None
        for frame_idx, out in model.propagate_in_video(
            inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=max_track,
            reverse=False,
        ):
            if out is None:
                continue

            out_obj_ids = [int(x) for x in out["out_obj_ids"].tolist()]
            out_masks = out["out_binary_masks"]  # (N, H, W) bool
            H, W = out_masks.shape[-2:]

            # Optional: write submission-style mask.
            if args.save_pred_masks:
                label_map = np.zeros((H, W), dtype=np.uint8)
                for i, obj_id in enumerate(out_obj_ids):
                    if obj_id <= 0 or obj_id > 255:
                        continue
                    label_map[out_masks[i]] = np.uint8(obj_id)
                _save_palette_mask(
                    seq_pred_dir / f"{frame_idx:05d}.png", label_map, seq_palette
                )

            # Metrics require per-frame GT masks; MOSE v1 valid only has the first frame annotated.
            gt_path = anno_folder / frame_names[frame_idx].replace(".jpg", ".png")
            if not gt_path.exists():
                continue
            gt = _load_mask(gt_path)

            # Per-frame mean IoU across objects present in the first frame.
            frame_ious = []
            for obj_id in obj_ids:
                gt_bin = gt == int(obj_id)
                try:
                    pred_idx = out_obj_ids.index(int(obj_id))
                    pred_bin = out_masks[pred_idx]
                except ValueError:
                    pred_bin = np.zeros_like(gt_bin, dtype=bool)
                frame_ious.append(_iou(pred_bin, gt_bin))

            if frame_ious:
                frame_miou = float(np.mean(frame_ious))
                seq_iou_sum += frame_miou
                seq_eval_frames += 1

        model.reset_state(inference_state)

        seq_mean_iou = float(seq_iou_sum / max(seq_eval_frames, 1))
        results[seq_name] = {"mean_iou": seq_mean_iou, "eval_frames": int(seq_eval_frames)}
        global_iou_sum += seq_iou_sum
        global_frames += seq_eval_frames

        print(f"{seq_name}: mIoU={seq_mean_iou:.4f} eval_frames={seq_eval_frames}")

    global_mean = float(global_iou_sum / max(global_frames, 1))
    summary = {
        "split": args.split,
        "num_seqs": int(len(results)),
        "global_mean_iou": global_mean,
        "global_eval_frames": int(global_frames),
        "seq_results": results,
    }
    print(f"\nGlobal Mean IoU: {global_mean:.4f} (eval_frames={global_frames})")
    (out_root / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
