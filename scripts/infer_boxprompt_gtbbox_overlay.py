#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util

from sam3.eval.postprocessors import PostProcessImage
from sam3.model_builder import build_sam3_image_model
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image as Sam3ImageData,
    InferenceMetadata,
    Object,
)
from sam3.train.transforms.basic_for_api import (
    NormalizeAPI,
    PadToSizeAPI,
    RandomResizeAPI,
    ToTensorAPI,
)


def _overlay_mask(rgb: np.ndarray, mask: np.ndarray, color, alpha: float) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32)[None, None, :]
    out[mask] = alpha * color_arr + (1 - alpha) * out[mask]
    return out.clip(0, 255).astype(np.uint8)

def _parse_rgb(s: str):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError("Expected color as 'R,G,B'")
    rgb = tuple(int(p) for p in parts)
    if any(v < 0 or v > 255 for v in rgb):
        raise ValueError("RGB values must be in [0,255]")
    return rgb


def _decode_rle(segm) -> np.ndarray:
    if isinstance(segm, dict) and "counts" in segm:
        return mask_util.decode(segm).astype(bool)
    raise ValueError("Expected COCO RLE dict for segmentation.")


def _xywh_to_xyxy(bbox_xywh):
    x, y, w, h = bbox_xywh
    return [x, y, x + w, y + h]


def _load_finetune_state_dict(ckpt_path: Path) -> dict:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", required=True, type=str, help=".../pork")
    ap.add_argument("--coco-json", required=True, type=str, help=".../annotations/test.json")
    ap.add_argument("--base-sam3-pt", required=True, type=str, help="Base SAM3 weights (sam3.pt)")
    ap.add_argument("--finetune-ckpt", required=True, type=str, help="Trainer checkpoint_*.pt (370MB)")
    ap.add_argument("--bpe-path", required=True, type=str)
    ap.add_argument("--out-dir", required=True, type=str)
    ap.add_argument("--prompt", type=str, default="ultrasound needle")
    ap.add_argument("--category-id", type=int, default=1)
    ap.add_argument(
        "--sample-ids",
        type=str,
        default=None,
        help="Comma-separated sample folder ids to run (matches COCO image file_name prefix like '12/34.png').",
    )
    ap.add_argument(
        "--sample-txt",
        type=str,
        default=None,
        help="Path to a txt file containing sample ids (one per line).",
    )
    ap.add_argument("--max-images", type=int, default=200)
    ap.add_argument("--score-thresh", type=float, default=0.0)
    ap.add_argument("--alpha-gt", type=float, default=0.35)
    ap.add_argument("--alpha-pred", type=float, default=0.45)
    ap.add_argument("--gt-color", type=str, default="64,128,255", help="GT overlay color as R,G,B")
    ap.add_argument("--pred-color", type=str, default="255,64,64", help="Pred overlay color as R,G,B")
    ap.add_argument(
        "--draw-fnfp",
        action="store_true",
        help="Overlay FN (GT-only) and FP (Pred-only) regions with distinct colors.",
    )
    ap.add_argument("--fn-color", type=str, default="0,255,255", help="FN region color as R,G,B (GT only)")
    ap.add_argument("--fp-color", type=str, default="255,0,255", help="FP region color as R,G,B (Pred only)")
    ap.add_argument("--alpha-fnfp", type=float, default=0.55, help="Alpha for FN/FP overlays")
    ap.add_argument("--resolution", type=int, default=1008)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--no-box-prompt",
        action="store_true",
        help="If set, do text-only prompting (no input_bbox). Uses a dummy target box to satisfy batching.",
    )
    ap.add_argument(
        "--preserve-relpath",
        action="store_true",
        help="Save outputs under out-dir/<sample_id>/<frame>.png mirroring COCO images[].file_name.",
    )
    args = ap.parse_args()
    gt_color = _parse_rgb(args.gt_color)
    pred_color = _parse_rgb(args.pred_color)
    fn_color = _parse_rgb(args.fn_color)
    fp_color = _parse_rgb(args.fp_color)

    images_root = Path(args.images_root)
    coco_path = Path(args.coco_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_sample_ids = None
    if args.sample_ids:
        selected_sample_ids = {s.strip() for s in args.sample_ids.split(",") if s.strip()}
    if args.sample_txt:
        txt_path = Path(args.sample_txt)
        ids = [ln.strip() for ln in txt_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        selected_sample_ids = set(ids) if selected_sample_ids is None else (selected_sample_ids & set(ids))

    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    images = {int(im["id"]): im for im in coco["images"]}
    anns_by_image = {}
    for ann in coco.get("annotations", []):
        if int(ann.get("category_id", -1)) != args.category_id:
            continue
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = build_sam3_image_model(
        bpe_path=args.bpe_path,
        device="cpu",
        eval_mode=False,
        load_from_HF=False,
        checkpoint_path=str(Path(args.base_sam3_pt)),
        enable_segmentation=True,
        freeze_vision_backbone=True,
        freeze_language_backbone=True,
    )
    finetune_sd = _load_finetune_state_dict(Path(args.finetune_ckpt))
    missing, unexpected = model.load_state_dict(finetune_sd, strict=False)
    print(f"Loaded finetune ckpt: missing={len(missing)} unexpected={len(unexpected)}")

    model.to(device)
    model.eval()

    # Deterministic preprocessing consistent with training configs (square resize + normalize).
    t_resize = RandomResizeAPI(sizes=args.resolution, max_size=args.resolution, square=True, consistent_transform=False)
    t_pad = PadToSizeAPI(size=args.resolution, consistent_transform=False)
    t_tensor = ToTensorAPI()
    t_norm = NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    post = PostProcessImage(
        max_dets_per_img=1,
        iou_type="segm",
        to_cpu=True,
        use_original_ids=True,
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        always_interpolate_masks_on_gpu=False,
        use_presence=True,
        detection_threshold=args.score_thresh,
    )

    num_done = 0
    for image_id, im_info in images.items():
        if num_done >= args.max_images:
            break
        if image_id not in anns_by_image:
            continue

        rel_path = Path(im_info["file_name"])
        sample_id = rel_path.parts[0] if len(rel_path.parts) >= 2 else None
        if selected_sample_ids is not None and (sample_id is None or sample_id not in selected_sample_ids):
            continue

        img_path = images_root / im_info["file_name"]
        if not img_path.exists():
            img_path = images_root / Path(im_info["file_name"]).name
        if not img_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Merge GT masks for visualization; take the union if multiple anns exist.
        gt_mask = None
        for ann in anns_by_image[image_id]:
            if "segmentation" not in ann or ann["segmentation"] in (None, [], {}):
                continue
            m = _decode_rle(ann["segmentation"])
            gt_mask = m if gt_mask is None else (gt_mask | m)

        if gt_mask is None:
            continue

        # Use GT bbox prompt (union bbox over GT anns).
        gt_boxes_xyxy = []
        for ann in anns_by_image[image_id]:
            if "bbox" not in ann or ann["bbox"] is None:
                continue
            gt_boxes_xyxy.append(_xywh_to_xyxy(ann["bbox"]))
        if not gt_boxes_xyxy:
            continue
        gt_boxes_xyxy = torch.tensor(gt_boxes_xyxy, dtype=torch.float32)
        x1 = float(gt_boxes_xyxy[:, 0].min())
        y1 = float(gt_boxes_xyxy[:, 1].min())
        x2 = float(gt_boxes_xyxy[:, 2].max())
        y2 = float(gt_boxes_xyxy[:, 3].max())
        union_box = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

        dummy_target_box = (
            torch.tensor([0.0, 0.0, float(w), float(h)], dtype=torch.float32)
            if args.no_box_prompt
            else union_box.clone()
        )
        obj = Object(
            bbox=dummy_target_box,
            area=1.0,
            object_id=0,
            frame_index=0,
            segment=None,
            is_crowd=False,
        )
        meta = InferenceMetadata(
            coco_image_id=image_id,
            original_image_id=image_id,
            original_category_id=args.category_id,
            original_size=(h, w),
            object_id=0,
            frame_index=0,
            is_conditioning_only=False,
        )
        query = FindQueryLoaded(
            query_text=args.prompt,
            image_id=0,
            object_ids_output=[0],
            is_exhaustive=True,
            query_processing_order=0,
            input_bbox=None if args.no_box_prompt else union_box.clone(),
            input_bbox_label=None if args.no_box_prompt else torch.ones(1, dtype=torch.bool),
            inference_metadata=meta,
        )
        datapoint = Datapoint(
            find_queries=[query],
            images=[Sam3ImageData(data=img, objects=[obj], size=(h, w))],
            raw_images=[img],
        )

        datapoint = t_resize(datapoint, epoch=0)
        datapoint = t_pad(datapoint, epoch=0)
        datapoint = t_tensor(datapoint, epoch=0)
        datapoint = t_norm(datapoint, epoch=0)

        batch_dict = collate_fn_api([datapoint], dict_key="all", with_seg_masks=False)
        batch = copy_data_to_device(batch_dict["all"], device, non_blocking=True)

        find_stages = model(batch)
        results = post.process_results(find_stages=find_stages, find_metadatas=batch.find_metadatas)
        pred = results.get(image_id)
        if pred is None or "masks" not in pred or pred["masks"] is None or pred["masks"].numel() == 0:
            continue

        pred_mask = pred["masks"][0, 0].numpy().astype(bool)

        rgb = np.array(img)
        if args.draw_fnfp:
            inter = gt_mask & pred_mask
            fn = gt_mask & (~pred_mask)
            fp = pred_mask & (~gt_mask)
            rgb = _overlay_mask(rgb, inter, color=pred_color, alpha=args.alpha_pred)
            rgb = _overlay_mask(rgb, fn, color=fn_color, alpha=args.alpha_fnfp)
            rgb = _overlay_mask(rgb, fp, color=fp_color, alpha=args.alpha_fnfp)
        else:
            rgb = _overlay_mask(rgb, gt_mask, color=gt_color, alpha=args.alpha_gt)
            rgb = _overlay_mask(rgb, pred_mask, color=pred_color, alpha=args.alpha_pred)
        out_img = Image.fromarray(rgb)
        if args.preserve_relpath and sample_id is not None:
            out_path = out_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path = out_dir / f"{image_id}.png"
        out_img.save(out_path)
        num_done += 1

    print(f"Saved overlays: {num_done} -> {out_dir}")


if __name__ == "__main__":
    # Avoid accidental Triton usage in some environments.
    os.environ.setdefault("SAM3_DISABLE_TRITON", "1")
    main()
