#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_util


def _decode_rle(segm):
    if isinstance(segm, dict) and "counts" in segm:
        return mask_util.decode(segm).astype(bool)
    raise ValueError("Unsupported segmentation format; expected COCO RLE dict.")


def _overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(255, 64, 64), alpha=0.45):
    out = rgb.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)[None, None, :]
    out[mask] = alpha * color_arr + (1 - alpha) * out[mask]
    return out.clip(0, 255).astype(np.uint8)


def _load_coco_annotations(coco: dict) -> dict[int, list[dict]]:
    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)
    return anns_by_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", required=True, type=str, help="Root folder containing images, e.g. .../pork")
    ap.add_argument("--coco-json", required=True, type=str, help="COCO GT json (for image_id->file_name mapping)")
    ap.add_argument("--pred-json", required=True, type=str, help="COCO prediction json (from PredictionDumper)")
    ap.add_argument("--out-dir", required=True, type=str)
    ap.add_argument("--category-id", type=int, default=1, help="Which category_id to visualize")
    ap.add_argument("--topk", type=int, default=1, help="Top-K predictions per image to overlay")
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--draw-box", action="store_true", help="Draw predicted bbox if present")
    ap.add_argument("--draw-gt", action="store_true", help="Overlay GT segmentation in a different color")
    ap.add_argument("--gt-alpha", type=float, default=0.35)
    args = ap.parse_args()

    images_root = Path(args.images_root)
    coco_json = Path(args.coco_json)
    pred_json = Path(args.pred_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = json.loads(coco_json.read_text(encoding="utf-8"))
    images = {int(im["id"]): im for im in coco["images"]}
    anns_by_image = _load_coco_annotations(coco) if args.draw_gt else {}

    preds = json.loads(pred_json.read_text(encoding="utf-8"))
    preds_by_image = {}
    for p in preds:
        if int(p.get("category_id", -1)) != args.category_id:
            continue
        preds_by_image.setdefault(int(p["image_id"]), []).append(p)

    for image_id, im_info in images.items():
        if image_id not in preds_by_image and (not args.draw_gt or image_id not in anns_by_image):
            continue
        cur_preds = sorted(preds_by_image[image_id], key=lambda x: float(x.get("score", 0.0)), reverse=True)[
            : args.topk
        ]

        img_path = images_root / im_info["file_name"]
        if not img_path.exists():
            # Some COCO files store only basename; try that as fallback.
            img_path = images_root / Path(im_info["file_name"]).name
        if not img_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        rgb = np.array(img)

        if args.draw_gt and image_id in anns_by_image:
            for ann in anns_by_image[image_id]:
                if int(ann.get("category_id", -1)) != args.category_id:
                    continue
                if "segmentation" not in ann or ann["segmentation"] in (None, [], {}):
                    continue
                gt_mask = _decode_rle(ann["segmentation"])
                rgb = _overlay_mask(rgb, gt_mask, color=(64, 128, 255), alpha=args.gt_alpha)
            img = Image.fromarray(rgb)

        draw = ImageDraw.Draw(img)
        for pred in cur_preds:
            if "segmentation" in pred and pred["segmentation"] is not None:
                mask = _decode_rle(pred["segmentation"])
                rgb = _overlay_mask(rgb, mask, color=(255, 64, 64), alpha=args.alpha)
                img = Image.fromarray(rgb)
                draw = ImageDraw.Draw(img)

            if args.draw_box and "bbox" in pred and pred["bbox"] is not None:
                x, y, w, h = pred["bbox"]
                draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=2)

        out_path = out_dir / f"{image_id}.png"
        img.save(out_path)


if __name__ == "__main__":
    main()
