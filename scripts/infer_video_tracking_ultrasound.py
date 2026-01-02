#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_video_model


def _parse_rgb(s: str):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError("Expected color as 'R,G,B'")
    rgb = tuple(int(p) for p in parts)
    if any(v < 0 or v > 255 for v in rgb):
        raise ValueError("RGB values must be in [0,255]")
    return rgb


def _overlay_mask(rgb: np.ndarray, mask: np.ndarray, color, alpha: float) -> np.ndarray:
    if mask is None:
        return rgb
    out = rgb.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32)[None, None, :]
    out[mask] = alpha * color_arr + (1 - alpha) * out[mask]
    return out.clip(0, 255).astype(np.uint8)


def _read_binary_mask_png(path: Path) -> np.ndarray:
    with Image.open(path) as m:
        arr = np.array(m.convert("L"), dtype=np.uint8)
    return (arr > 0).astype(bool)


def _bbox_xyxy_from_mask(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max()) + 1.0
    y2 = float(ys.max()) + 1.0
    return x1, y1, x2, y2


def _xyxy_to_xywh_norm(x1, y1, x2, y2, w, h):
    x = x1 / w
    y = y1 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return [float(x), float(y), float(bw), float(bh)]


def _numeric_png_sort_key(p: Path):
    try:
        return int(p.stem)
    except ValueError:
        return p.stem


def _find_first_nonempty_gt_frame(frame_paths: list[Path], gt_dir: Path) -> tuple[int, np.ndarray] | tuple[None, None]:
    for idx, fp in enumerate(frame_paths):
        gt_path = gt_dir / fp.name
        if not gt_path.exists():
            continue
        m = _read_binary_mask_png(gt_path)
        if m.any():
            return idx, m
    return None, None


def _load_trainer_checkpoint_state_dict(path: Path) -> dict:
    ckpt = torch.load(str(path), map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError(f"Unsupported checkpoint format: {path}")


def _apply_detector_finetune(video_model, finetune_ckpt: Path) -> None:
    sd = _load_trainer_checkpoint_state_dict(finetune_ckpt)
    target = getattr(video_model, "detector", None)
    if target is None:
        target = video_model
    missing, unexpected = target.load_state_dict(sd, strict=False)
    print(
        f"Loaded finetune into {'detector' if getattr(video_model,'detector',None) is not None else 'model'}: "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )


def _set_single_object_tracker_metadata(
    model, inference_state, obj_id: int, first_frame_idx: int
) -> None:
    tracker_metadata = model._initialize_metadata() if hasattr(model, "_initialize_metadata") else {}
    world_size = int(getattr(model, "world_size", 1))
    obj_ids = np.array([int(obj_id)], dtype=np.int64)
    obj_ids_per_gpu = [np.array([], dtype=np.int64) for _ in range(world_size)]
    obj_ids_per_gpu[0] = obj_ids

    tracker_metadata.update(
        {
            "obj_ids_per_gpu": obj_ids_per_gpu,
            "obj_ids_all_gpu": obj_ids,
            "num_obj_per_gpu": np.array([len(x) for x in obj_ids_per_gpu], dtype=np.int64),
            "obj_id_to_score": {int(obj_id): 1.0},
            "max_obj_id": int(obj_id) + 1,
        }
    )
    rank0_md = tracker_metadata.get("rank0_metadata", None)
    if isinstance(rank0_md, dict):
        obj_first = rank0_md.get("obj_first_frame_idx", None)
        if isinstance(obj_first, dict):
            obj_first[int(obj_id)] = int(first_frame_idx)
    inference_state["tracker_metadata"] = tracker_metadata


@torch.inference_mode()
def run_one_mode(
    model,
    frames_dir: Path,
    gt_dir: Path,
    out_dir: Path,
    prompt: str,
    init_mode: str,
    mask_select: str,
    draw_fnfp: bool,
    alpha_gt: float,
    alpha_pred: float,
    alpha_fnfp: float,
    gt_color,
    pred_color,
    fn_color,
    fp_color,
    offload_video_to_cpu: bool,
    max_frames: int | None,
    init_frame: int | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(frames_dir.glob("*.png"), key=_numeric_png_sort_key)
    if max_frames is not None:
        frame_paths = frame_paths[:max_frames]
    if not frame_paths:
        return

    inference_state = model.init_state(
        resource_path=str(frames_dir),
        offload_video_to_cpu=offload_video_to_cpu,
        async_loading_frames=False,
        video_loader_type="cv2",
    )

    num_frames = inference_state["num_frames"]
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    # Choose init frame.
    init_frame_idx = 0 if init_frame is None else int(init_frame)
    init_gt_mask = None
    if init_mode in ("gtmask", "gtbbox_text") and init_frame is None:
        found_idx, found_mask = _find_first_nonempty_gt_frame(frame_paths, gt_dir)
        if found_idx is None:
            return
        init_frame_idx = int(found_idx)
        init_gt_mask = found_mask
    elif init_mode in ("gtmask", "gtbbox_text"):
        gt_path = gt_dir / frame_paths[init_frame_idx].name
        init_gt_mask = _read_binary_mask_png(gt_path) if gt_path.exists() else None

    if init_mode == "gtmask":
        if init_gt_mask is None or not init_gt_mask.any():
            return
        # Warm up backbone/detector features for frame 0 (helps tracker init use cached features).
        if hasattr(model, "_prepare_backbone_feats"):
            model._prepare_backbone_feats(
                inference_state, frame_idx=init_frame_idx, reverse=False
            )
        tracker_states_local = inference_state["tracker_inference_states"]
        tracker_metadata = inference_state["tracker_metadata"]
        if tracker_metadata == {} and hasattr(model, "_initialize_metadata"):
            tracker_metadata.update(model._initialize_metadata())
        new_mask = torch.from_numpy(init_gt_mask.astype(np.float32)).to(model.device)
        tracker_states_local = model._tracker_add_new_objects(
            frame_idx=init_frame_idx,
            num_frames=inference_state["num_frames"],
            new_obj_ids=[0],
            new_obj_masks=new_mask[None],
            tracker_states_local=tracker_states_local,
            orig_vid_height=inference_state["orig_height"],
            orig_vid_width=inference_state["orig_width"],
            feature_cache=inference_state["feature_cache"],
        )
        inference_state["tracker_inference_states"] = tracker_states_local
        _set_single_object_tracker_metadata(
            model=model,
            inference_state=inference_state,
            obj_id=0,
            first_frame_idx=init_frame_idx,
        )
    elif init_mode == "gtbbox_text":
        if init_gt_mask is None or not init_gt_mask.any():
            return
        xyxy = _bbox_xyxy_from_mask(init_gt_mask)
        if xyxy is None:
            return
        w, h = float(inference_state["orig_width"]), float(inference_state["orig_height"])
        bbox_xywh = _xyxy_to_xywh_norm(*xyxy, w=w, h=h)
        model.add_prompt(
            inference_state=inference_state,
            frame_idx=init_frame_idx,
            text_str=prompt,
            boxes_xywh=[bbox_xywh],
            box_labels=[1],
        )
    elif init_mode == "text_only":
        model.add_prompt(
            inference_state=inference_state,
            frame_idx=0,
            text_str=prompt,
        )
    else:
        raise ValueError(f"Unknown init_mode: {init_mode}")

    for frame_idx, outputs in model.propagate_in_video(
        inference_state=inference_state,
        start_frame_idx=init_frame_idx,
        max_frame_num_to_track=num_frames,
        reverse=False,
    ):
        if frame_idx >= len(frame_paths):
            break
        frame_path = frame_paths[frame_idx]
        gt_path = gt_dir / frame_path.name
        gt_mask = _read_binary_mask_png(gt_path) if gt_path.exists() else None

        img = Image.open(frame_path).convert("RGB")
        rgb = np.array(img)

        pred_masks = outputs.get("out_binary_masks", None)
        pred_probs = outputs.get("out_probs", None)
        pred_mask = None
        if pred_masks is not None and len(pred_masks) > 0:
            if mask_select == "all":
                pred_mask = np.any(pred_masks.astype(bool), axis=0)
            else:
                # top1 by probability
                if pred_probs is None or len(pred_probs) == 0:
                    pred_mask = pred_masks[0].astype(bool)
                else:
                    pred_mask = pred_masks[int(np.argmax(pred_probs))].astype(bool)

        if draw_fnfp and gt_mask is not None and pred_mask is not None:
            inter = gt_mask & pred_mask
            fn = gt_mask & (~pred_mask)
            fp = pred_mask & (~gt_mask)
            rgb = _overlay_mask(rgb, inter, color=pred_color, alpha=alpha_pred)
            rgb = _overlay_mask(rgb, fn, color=fn_color, alpha=alpha_fnfp)
            rgb = _overlay_mask(rgb, fp, color=fp_color, alpha=alpha_fnfp)
        else:
            if gt_mask is not None:
                rgb = _overlay_mask(rgb, gt_mask, color=gt_color, alpha=alpha_gt)
            if pred_mask is not None:
                rgb = _overlay_mask(rgb, pred_mask, color=pred_color, alpha=alpha_pred)

        out_img = Image.fromarray(rgb)
        out_img.save(out_dir / frame_path.name)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", required=True, type=str, help=".../pork_dataset/pork")
    ap.add_argument("--gt-root", required=True, type=str, help=".../pork_dataset/gt")
    ap.add_argument("--sample-ids", type=str, default=None, help="Comma-separated sample ids, e.g. 1,2,3")
    ap.add_argument("--sample-txt", type=str, default=None, help="Txt file with sample ids, one per line")
    ap.add_argument("--out-dir", required=True, type=str)
    ap.add_argument("--base-sam3-pt", required=True, type=str, help="Base SAM3 weights (sam3.pt)")
    ap.add_argument("--finetune-ckpt", required=True, type=str, help="Finetuned trainer checkpoint_*.pt")
    ap.add_argument("--bpe-path", required=True, type=str)
    ap.add_argument("--prompt", type=str, default="ultrasound needle")
    ap.add_argument("--modes", type=str, default="gtmask,gtbbox_text,text_only")
    ap.add_argument("--mask-select", choices=["top1", "all"], default="top1")
    ap.add_argument("--offload-video-to-cpu", action="store_true", help="Reduce GPU memory by keeping frames on CPU")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument(
        "--init-frame",
        type=int,
        default=None,
        help="Init frame index for gtbbox_text/gtmask; default uses first non-empty GT frame.",
    )
    ap.add_argument("--draw-fnfp", action="store_true")
    ap.add_argument("--alpha-gt", type=float, default=0.25)
    ap.add_argument("--alpha-pred", type=float, default=0.35)
    ap.add_argument("--alpha-fnfp", type=float, default=0.65)
    ap.add_argument("--gt-color", type=str, default="64,128,255")
    ap.add_argument("--pred-color", type=str, default="0,255,0")
    ap.add_argument("--fn-color", type=str, default="0,255,255")
    ap.add_argument("--fp-color", type=str, default="255,0,255")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    frames_root = Path(args.frames_root)
    gt_root = Path(args.gt_root)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    selected = None
    if args.sample_ids:
        selected = {s.strip() for s in args.sample_ids.split(",") if s.strip()}
    if args.sample_txt:
        ids = [
            ln.strip()
            for ln in Path(args.sample_txt).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        selected = set(ids) if selected is None else (selected & set(ids))
    if selected is None:
        # default: all numeric folders under frames_root
        selected = {p.name for p in frames_root.iterdir() if p.is_dir()}

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = build_sam3_video_model(
        checkpoint_path=str(Path(args.base_sam3_pt)),
        load_from_HF=False,
        bpe_path=str(Path(args.bpe_path)),
        strict_state_dict_loading=False,
        device=str(device),
        apply_temporal_disambiguation=True,
    )
    model.to(device).eval()
    _apply_detector_finetune(model, Path(args.finetune_ckpt))

    gt_color = _parse_rgb(args.gt_color)
    pred_color = _parse_rgb(args.pred_color)
    fn_color = _parse_rgb(args.fn_color)
    fp_color = _parse_rgb(args.fp_color)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for sample_id in sorted(selected, key=lambda x: int(x) if x.isdigit() else x):
        frames_dir = frames_root / sample_id
        gt_dir = gt_root / sample_id
        if not frames_dir.exists() or not gt_dir.exists():
            continue
        for mode in modes:
            out_dir = out_root / mode / sample_id
            run_one_mode(
                model=model,
                frames_dir=frames_dir,
                gt_dir=gt_dir,
                out_dir=out_dir,
                prompt=args.prompt,
                init_mode=mode,
                mask_select=args.mask_select,
                draw_fnfp=args.draw_fnfp,
                alpha_gt=args.alpha_gt,
                alpha_pred=args.alpha_pred,
                alpha_fnfp=args.alpha_fnfp,
                gt_color=gt_color,
                pred_color=pred_color,
                fn_color=fn_color,
                fp_color=fp_color,
                offload_video_to_cpu=args.offload_video_to_cpu,
                max_frames=args.max_frames,
                init_frame=args.init_frame,
            )
        print(f"Done sample {sample_id}")


if __name__ == "__main__":
    os.environ.setdefault("SAM3_DISABLE_TRITON", "1")
    main()
