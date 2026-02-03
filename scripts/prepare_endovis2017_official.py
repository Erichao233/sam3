#!/usr/bin/env python3
"""
Prepare the *official* EndoVis2017 release into the canonical layout used by this repo.

Official EndoVis2017 (common structure):
  - instrument_*_training/instrument_dataset_<sid>/{left_frames,right_frames}/frameXXX.png
  - instrument_*_training/instrument_dataset_<sid>/ground_truth/*_labels/frameXXX.png
  - instrument_*_testing/instrument_dataset_<sid>/{left_frames,right_frames}/frameYYY.png
  - instrument_2017_test/instrument_dataset_<sid>/TypeSegmentation/frameYYY.png

This script writes:
  - <out_root>/train/image/seq_<sid>_frameXXXX.png
  - <out_root>/train/label/seq_<sid>_frameXXXX.png (uint8, pixel value = class id; 0=bg)
  - <out_root>/val{sid}/image/seq_<sid>_frameXXXX.png
  - <out_root>/val{sid}/label/seq_<sid>_frameXXXX.png

Notes:
  - Train labels are *merged* from per-class binary masks under ground_truth/*_labels.
  - Val labels are copied from instrument_2017_test/.../TypeSegmentation.
  - Default camera is "left" (left_frames).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


INSTRUMENT_CLASSES = {
    1: "Bipolar Forceps",
    2: "Prograsp Forceps",
    3: "Large Needle Driver",
    4: "Vessel Sealer",
    5: "Grasping Retractor",
    6: "Monopolar Curved Scissors",
    7: "Other",
}


def _is_junk_path(p: Path) -> bool:
    # macOS resource fork files or hidden files
    return p.name.startswith("._") or p.name.startswith(".")


def _parse_instrument_dataset_id(seq_dir_name: str) -> int | None:
    # instrument_dataset_1
    parts = seq_dir_name.split("_")
    if len(parts) < 3:
        return None
    try:
        return int(parts[-1])
    except ValueError:
        return None


def _parse_frame_idx(name: str) -> int | None:
    # frame000.png / frame225.png
    if not name.lower().startswith("frame"):
        return None
    stem = Path(name).stem
    try:
        return int(stem.replace("frame", ""))
    except ValueError:
        return None


def _class_id_from_gt_folder(name: str) -> int:
    n = name.lower()
    if "bipolar" in n:
        return 1
    if "prograsp" in n:
        return 2
    if "needle" in n and "driver" in n:
        return 3
    if "vessel" in n:
        return 4
    if "retractor" in n:
        return 5
    if "monopolar" in n or "scissors" in n:
        return 6
    if "other" in n:
        return 7
    raise ValueError(f"Unrecognized GT folder (cannot map to class id): {name}")


def _load_mask_u8(path: Path) -> np.ndarray:
    with Image.open(path) as m:
        arr = np.array(m)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


def _save_mask_u8(mask: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8), mode="L").save(out_path)


def _copy_or_symlink(src: Path, dst: Path, *, symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if symlink:
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def _clear_outputs(out_root: Path) -> None:
    # Only delete the directories we create (avoid nuking arbitrary user dirs).
    for split in ["train"] + [f"val{i}" for i in range(1, 11)]:
        p = out_root / split
        if p.exists():
            shutil.rmtree(p)
    meta = out_root / "endovis2017_prepared_meta.json"
    if meta.exists():
        meta.unlink()


def _iter_training_seq_dirs(src_root: Path) -> list[Path]:
    out: list[Path] = []
    for group in sorted(src_root.glob("instrument_*_training")):
        if not group.is_dir() or _is_junk_path(group):
            continue
        for seq_dir in sorted(group.glob("instrument_dataset_*")):
            if seq_dir.is_dir() and not _is_junk_path(seq_dir):
                out.append(seq_dir)
    return out


def _iter_testing_seq_dirs(src_root: Path) -> list[Path]:
    out: list[Path] = []
    for group in sorted(src_root.glob("instrument_*_testing")):
        if not group.is_dir() or _is_junk_path(group):
            continue
        for seq_dir in sorted(group.glob("instrument_dataset_*")):
            if seq_dir.is_dir() and not _is_junk_path(seq_dir):
                out.append(seq_dir)
    return out


@dataclass(frozen=True)
class PreparedCounts:
    train_images: int = 0
    train_labels: int = 0
    val_images: int = 0
    val_labels: int = 0


def prepare_train(
    *,
    src_root: Path,
    out_root: Path,
    camera: str,
    symlink_images: bool,
) -> PreparedCounts:
    cam_dir = "left_frames" if camera == "left" else "right_frames"
    out_img_dir = out_root / "train" / "image"
    out_lbl_dir = out_root / "train" / "label"

    train_images = 0
    train_labels = 0
    frames_by_seq: dict[int, int] = {}
    labels_by_seq: dict[int, int] = {}
    gt_dirs_by_seq: dict[int, list[str]] = {}

    for seq_dir in _iter_training_seq_dirs(src_root):
        sid = _parse_instrument_dataset_id(seq_dir.name)
        if sid is None:
            continue

        img_dir = seq_dir / cam_dir
        gt_root = seq_dir / "ground_truth"
        if not img_dir.exists():
            raise FileNotFoundError(f"Missing frames dir: {img_dir}")
        if not gt_root.exists():
            raise FileNotFoundError(f"Missing ground_truth dir: {gt_root}")

        gt_dirs: list[tuple[int, Path]] = []
        skipped_gt_dirs: list[str] = []
        for d in sorted(gt_root.glob("*")):
            if not d.is_dir() or _is_junk_path(d):
                continue
            # Some EndoVis2017 releases name these folders as "*_labels", others use plain
            # names like "Left_Bipolar_Forceps". Use a content-based check (has frame*.png)
            # and then map folder name -> type class id.
            if not any(d.glob("frame*.png")):
                continue
            try:
                class_id = _class_id_from_gt_folder(d.name)
            except ValueError:
                skipped_gt_dirs.append(d.name)
                continue
            gt_dirs.append((int(class_id), d))

        if not gt_dirs:
            # Print a helpful message to debug dataset variants.
            children = [p.name for p in sorted(gt_root.glob("*")) if p.is_dir() and not _is_junk_path(p)]
            raise RuntimeError(
                "No GT label folders found under: "
                f"{gt_root}\n"
                f"Found subdirs: {children}\n"
                f"Skipped (unrecognized names): {skipped_gt_dirs}\n"
                "Expected per-instrument mask folders containing frame*.png (e.g., '*_labels' or 'Left_Bipolar_Forceps')."
            )
        gt_dirs_by_seq[int(sid)] = [p.name for _, p in gt_dirs]

        for img_path in sorted(img_dir.glob("frame*.png")):
            if not img_path.is_file() or _is_junk_path(img_path):
                continue
            frame_idx = _parse_frame_idx(img_path.name)
            if frame_idx is None:
                continue

            out_name = f"seq_{sid}_frame{frame_idx:04d}.png"
            out_img = out_img_dir / out_name
            out_lbl = out_lbl_dir / out_name

            _copy_or_symlink(img_path, out_img, symlink=symlink_images)
            train_images += 1
            frames_by_seq[int(sid)] = int(frames_by_seq.get(int(sid), 0) + 1)

            # Merge binary masks into a single uint8 type mask.
            merged: np.ndarray | None = None
            for class_id, d in gt_dirs:
                m_path = d / img_path.name
                if not m_path.exists():
                    continue
                m = _load_mask_u8(m_path)
                if merged is None:
                    merged = np.zeros((m.shape[0], m.shape[1]), dtype=np.uint8)
                merged[m > 0] = np.uint8(class_id)

            if merged is None:
                # If a frame exists but no GT mask files were present, treat as empty.
                with Image.open(img_path) as im:
                    merged = np.zeros((im.height, im.width), dtype=np.uint8)

            _save_mask_u8(merged, out_lbl)
            train_labels += 1
            labels_by_seq[int(sid)] = int(labels_by_seq.get(int(sid), 0) + 1)

    # Attach lightweight debug stats for quick sanity checks (without opening the full folder).
    (out_root / "train" / "prepare_stats.json").write_text(
        json.dumps(
            {
                "train_images": int(train_images),
                "train_labels": int(train_labels),
                "frames_by_seq": {str(k): int(v) for k, v in sorted(frames_by_seq.items())},
                "labels_by_seq": {str(k): int(v) for k, v in sorted(labels_by_seq.items())},
                "gt_dirs_by_seq": {str(k): v for k, v in sorted(gt_dirs_by_seq.items())},
                "missing_expected_train_seqs_1_to_8": [
                    int(s) for s in range(1, 9) if int(s) not in frames_by_seq
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return PreparedCounts(train_images=train_images, train_labels=train_labels)


def prepare_val(
    *,
    src_root: Path,
    out_root: Path,
    camera: str,
    symlink_images: bool,
) -> PreparedCounts:
    cam_dir = "left_frames" if camera == "left" else "right_frames"
    test_labels_root = src_root / "instrument_2017_test"
    if not test_labels_root.exists():
        raise FileNotFoundError(f"Missing test label root: {test_labels_root}")

    test_seq_dirs = { }
    for seq_dir in _iter_testing_seq_dirs(src_root):
        sid = _parse_instrument_dataset_id(seq_dir.name)
        if sid is None:
            continue
        test_seq_dirs[int(sid)] = seq_dir

    val_images = 0
    val_labels = 0
    for sid in range(1, 11):
        if sid not in test_seq_dirs:
            raise FileNotFoundError(f"Missing testing images for instrument_dataset_{sid} under instrument_*_testing/")

        seq_dir = test_seq_dirs[sid]
        img_dir = seq_dir / cam_dir
        lbl_dir = test_labels_root / f"instrument_dataset_{sid}" / "TypeSegmentation"
        if not img_dir.exists():
            raise FileNotFoundError(f"Missing frames dir: {img_dir}")
        if not lbl_dir.exists():
            raise FileNotFoundError(f"Missing TypeSegmentation labels: {lbl_dir}")

        out_img_dir = out_root / f"val{sid}" / "image"
        out_lbl_dir = out_root / f"val{sid}" / "label"

        for lbl_path in sorted(lbl_dir.glob("frame*.png")):
            if not lbl_path.is_file() or _is_junk_path(lbl_path):
                continue
            frame_idx = _parse_frame_idx(lbl_path.name)
            if frame_idx is None:
                continue

            img_path = img_dir / lbl_path.name
            if not img_path.exists():
                raise FileNotFoundError(f"Missing image for {lbl_path.name}: {img_path}")

            out_name = f"seq_{sid}_frame{frame_idx:04d}.png"
            out_img = out_img_dir / out_name
            out_lbl = out_lbl_dir / out_name

            _copy_or_symlink(img_path, out_img, symlink=symlink_images)
            val_images += 1

            m = _load_mask_u8(lbl_path)
            # Ensure values are within expected range (0..7).
            m[(m < 0) | (m > 7)] = 0
            _save_mask_u8(m, out_lbl)
            val_labels += 1

    return PreparedCounts(val_images=val_images, val_labels=val_labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True, type=str, help="Official EndoVis2017 root (contains instrument_* dirs)")
    ap.add_argument("--out-root", required=True, type=str, help="Output root (will create train/ and val*/)")
    ap.add_argument("--camera", type=str, default="left", choices=["left", "right"], help="Use left_frames or right_frames")
    ap.add_argument("--symlink-images", action="store_true", help="Symlink images instead of copying")
    ap.add_argument("--overwrite", action="store_true", help="Delete existing train/val* outputs under out-root first")
    args = ap.parse_args()

    src_root = Path(args.src_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    if not src_root.exists():
        raise FileNotFoundError(f"--src-root does not exist: {src_root}")
    if src_root == out_root:
        raise ValueError("--out-root must be different from --src-root")

    if args.overwrite:
        _clear_outputs(out_root)
    else:
        # If any expected output already exists, fail fast.
        for split in ["train"] + [f"val{i}" for i in range(1, 11)]:
            if (out_root / split).exists():
                raise FileExistsError(f"Output already exists: {out_root / split} (use --overwrite)")

    t0 = time.time()
    print(f"[prepare_endovis2017_official] src_root={src_root}")
    print(f"[prepare_endovis2017_official] out_root={out_root}")
    print(f"[prepare_endovis2017_official] camera={args.camera} symlink_images={args.symlink_images}")

    out_root.mkdir(parents=True, exist_ok=True)

    train_counts = prepare_train(
        src_root=src_root,
        out_root=out_root,
        camera=args.camera,
        symlink_images=bool(args.symlink_images),
    )
    val_counts = prepare_val(
        src_root=src_root,
        out_root=out_root,
        camera=args.camera,
        symlink_images=bool(args.symlink_images),
    )

    meta = {
        "src_root": str(src_root),
        "out_root": str(out_root),
        "camera": args.camera,
        "symlink_images": bool(args.symlink_images),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "counts": {
            "train_images": int(train_counts.train_images),
            "train_labels": int(train_counts.train_labels),
            "val_images": int(val_counts.val_images),
            "val_labels": int(val_counts.val_labels),
        },
        "classes": INSTRUMENT_CLASSES,
    }
    meta_path = out_root / "endovis2017_prepared_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    dt = time.time() - t0
    print(
        f"[done] train_images={train_counts.train_images} train_labels={train_counts.train_labels} "
        f"val_images={val_counts.val_images} val_labels={val_counts.val_labels} "
        f"secs={dt:.1f}"
    )
    print(f"[done] wrote meta: {meta_path}")


if __name__ == "__main__":
    main()
