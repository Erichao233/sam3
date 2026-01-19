#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_util


INSTRUMENT_CLASSES = {
    1: "Bipolar Forceps",
    2: "Prograsp Forceps",
    3: "Large Needle Driver",
    4: "Vessel Sealer",
    5: "Grasping Retractor",
    6: "Monopolar Curved Scissors",
    7: "Other",
}


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


def _seq_id_from_filename(name: str) -> int | None:
    # EndoVis2017 naming: seq_<sid>_frameXXXX.bmp
    m = re.match(r"^seq_(\d+)_frame\d+\.(bmp|png|jpg|jpeg)$", name, flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


def _load_label_bmp(path: Path) -> np.ndarray:
    with Image.open(path) as m:
        arr = np.array(m)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


def convert_folder(
    dataset_root: Path,
    split_name: str,
    seq_ids: set[int] | None,
    skip_empty_images: bool,
) -> dict:
    image_dir = dataset_root / split_name / "image"
    label_dir = dataset_root / split_name / "label"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing: {label_dir}")

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": int(k), "name": v} for k, v in INSTRUMENT_CLASSES.items()],
    }

    image_id = 1
    ann_id = 1

    img_paths = sorted(image_dir.glob("*.bmp"))
    if not img_paths:
        raise RuntimeError(f"No images found under: {image_dir}")

    for img_path in img_paths:
        sid = _seq_id_from_filename(img_path.name)
        if sid is None:
            continue
        if seq_ids is not None and sid not in seq_ids:
            continue

        label_path = label_dir / img_path.name
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label: {label_path}")

        with Image.open(img_path) as im:
            width, height = im.size

        label = _load_label_bmp(label_path)
        present_classes = [int(c) for c in np.unique(label) if int(c) in INSTRUMENT_CLASSES]

        if skip_empty_images and len(present_classes) == 0:
            continue

        coco["images"].append(
            {
                "id": image_id,
                "file_name": f"{split_name}/image/{img_path.name}",
                "width": int(width),
                "height": int(height),
            }
        )

        for class_id in present_classes:
            mask = (label == class_id).astype(np.uint8)
            if not mask.any():
                continue
            bbox = _bbox_from_mask(mask)
            rle = _encode_binary_mask_rle(mask)
            area = int(mask.sum())
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(class_id),
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
    ap.add_argument(
        "--train-seqs",
        type=int,
        nargs="*",
        default=[1, 2, 3, 4, 5, 6, 7],
        help="Sequence IDs (seq_<id>_frameXXXX.bmp) to use for training (default: 1..7).",
    )
    ap.add_argument(
        "--val-seqs",
        type=int,
        nargs="*",
        default=[8],
        help="Sequence IDs to use for validation (default: 8).",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="annotations_endovis2017",
        help="Output directory under dataset_root.",
    )
    ap.add_argument(
        "--skip-empty-images",
        action="store_true",
        help="Drop frames with no instrument pixels at all.",
    )
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    out_dir = (dataset_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_coco = convert_folder(
        dataset_root=dataset_root,
        split_name="train",
        seq_ids=set(args.train_seqs) if args.train_seqs else None,
        skip_empty_images=args.skip_empty_images,
    )
    val_coco = convert_folder(
        dataset_root=dataset_root,
        split_name="train",
        seq_ids=set(args.val_seqs) if args.val_seqs else None,
        skip_empty_images=args.skip_empty_images,
    )

    train_path = out_dir / "train.json"
    val_path = out_dir / "val.json"
    with train_path.open("w", encoding="utf-8") as f:
        json.dump(train_coco, f, ensure_ascii=False)
    with val_path.open("w", encoding="utf-8") as f:
        json.dump(val_coco, f, ensure_ascii=False)

    print(
        f"Wrote train: {train_path} (images={len(train_coco['images'])}, anns={len(train_coco['annotations'])})"
    )
    print(
        f"Wrote val:   {val_path} (images={len(val_coco['images'])}, anns={len(val_coco['annotations'])})"
    )


if __name__ == "__main__":
    main()

