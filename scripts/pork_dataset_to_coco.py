#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_util


def _numeric_png_sort_key(p: Path):
    try:
        return int(p.stem)
    except ValueError:
        return p.stem


def _read_split_ids(path: Path) -> list[str]:
    ids = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.append(line)
    return ids


def _encode_binary_mask_rle(mask_hw: np.ndarray) -> dict:
    rle = mask_util.encode(np.asfortranarray(mask_hw.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _bbox_from_mask(mask_hw: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.nonzero(mask_hw)
    x0 = int(xs.min())
    x1 = int(xs.max())
    y0 = int(ys.min())
    y1 = int(ys.max())
    return float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)


def convert_split(
    dataset_root: Path,
    split_ids: list[str],
    images_subdir: str,
    masks_subdir: str,
    category_id: int,
    category_name: str,
) -> dict:
    images_dir = dataset_root / images_subdir
    masks_dir = dataset_root / masks_subdir

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": category_id, "name": category_name}],
    }

    image_id = 1
    ann_id = 1

    for sid in split_ids:
        sid_images_dir = images_dir / sid
        sid_masks_dir = masks_dir / sid
        if not sid_images_dir.exists():
            raise FileNotFoundError(f"Missing images dir: {sid_images_dir}")
        if not sid_masks_dir.exists():
            raise FileNotFoundError(f"Missing masks dir: {sid_masks_dir}")

        frames = sorted(sid_images_dir.glob("*.png"), key=_numeric_png_sort_key)
        for img_path in frames:
            mask_path = sid_masks_dir / img_path.name
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask: {mask_path}")

            with Image.open(img_path) as im:
                width, height = im.size

            with Image.open(mask_path) as m:
                mask = np.array(m.convert("L"), dtype=np.uint8)
            mask = (mask > 0).astype(np.uint8)

            coco["images"].append(
                {
                    "id": image_id,
                    "file_name": f"{sid}/{img_path.name}",
                    "width": int(width),
                    "height": int(height),
                }
            )

            if mask.any():
                bbox = _bbox_from_mask(mask)
                rle = _encode_binary_mask_rle(mask)
                area = int(mask.sum())
                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": list(bbox),
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": rle,
                    }
                )
                ann_id += 1

            image_id += 1

    return coco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, required=True)
    ap.add_argument("--train-txt", type=str, default="train.txt")
    ap.add_argument("--val-txt", type=str, default="val.txt")
    ap.add_argument("--test-txt", type=str, default="test.txt")
    ap.add_argument("--images-subdir", type=str, default="pork")
    ap.add_argument("--masks-subdir", type=str, default="gt")
    ap.add_argument("--out-dir", type=str, default="annotations")
    ap.add_argument("--category-id", type=int, default=1)
    ap.add_argument("--category-name", type=str, default="ultrasound needle")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    out_dir = (dataset_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [
        ("train", dataset_root / args.train_txt),
        ("val", dataset_root / args.val_txt),
        ("test", dataset_root / args.test_txt),
    ]

    for split_name, split_path in splits:
        split_ids = _read_split_ids(split_path)
        coco = convert_split(
            dataset_root=dataset_root,
            split_ids=split_ids,
            images_subdir=args.images_subdir,
            masks_subdir=args.masks_subdir,
            category_id=args.category_id,
            category_name=args.category_name,
        )
        out_path = out_dir / f"{split_name}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False)
        print(f"Wrote {split_name}: {out_path} (images={len(coco['images'])}, anns={len(coco['annotations'])})")


if __name__ == "__main__":
    main()
