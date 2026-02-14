#!/usr/bin/env python3
"""
Prepare an EndoVis2018-style official release into the canonical layout used by this repo.

This script targets the common MICCAI 2018 Robotic Scene Segmentation releases, e.g.:
  <src_root>/
    miccai_challenge_2018_release_1/labels.json
    miccai_challenge_2018_release_1/seq_1/{left_frames,right_frames,labels}/frame000.png
    miccai_challenge_release_2/seq_5/...
    ...

Where:
  - frames are stored as RGB PNGs under {left_frames,right_frames}/frameXXX.png
  - labels are stored as color PNGs (often RGBA) under labels/frameXXX.png
  - labels.json provides a mapping from RGB color -> class id (+ name)

It writes a *canonical* layout:
  - <out_root>/val*/image/*.png
  - <out_root>/val*/label/*.png  (uint8, pixel value = class id; 0=bg)
  - (optional) <out_root>/train/image/*.png (symlinks to val images; convenience)
  - (optional) <out_root>/train/label/*.png (symlinks to val labels; convenience)
    Use --no-train-view to disable the flat train view (useful for preparing official test splits).

Notes:
  - We intentionally keep the per-sequence directories under `val*` so that evaluation scripts can
    iterate sequences independently.
  - By default we *exclude* a top-level `test_data/` folder if present, because it commonly duplicates
    seq ids (seq_1..4) with different content. Use --include-test-data to include it.
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


def _is_junk_path(p: Path) -> bool:
    return p.name.startswith("._") or p.name.startswith(".")


def _safe_iterdir(p: Path) -> list[Path]:
    try:
        return list(p.iterdir())
    except PermissionError:
        return []


def _find_child_dir(parent: Path, *names: str) -> Path | None:
    """
    Find a direct child directory whose name matches one of `names` (case-insensitive).
    """
    want = {n.lower() for n in names if n}
    if not want:
        return None
    for c in _safe_iterdir(parent):
        if c.is_dir() and (not _is_junk_path(c)) and c.name.lower() in want:
            return c
    return None


def _resolve_seq_subdirs(seq_dir: Path) -> tuple[Path | None, Path | None, Path | None]:
    """
    Returns:
      labels_dir, left_frames_dir, right_frames_dir (each may be None)
    """
    labels_dir = seq_dir / "labels"
    if not labels_dir.is_dir():
        labels_dir = _find_child_dir(seq_dir, "labels", "label")

    left_frames = seq_dir / "left_frames"
    if not left_frames.is_dir():
        left_frames = _find_child_dir(seq_dir, "left_frames")

    right_frames = seq_dir / "right_frames"
    if not right_frames.is_dir():
        right_frames = _find_child_dir(seq_dir, "right_frames")

    return (
        labels_dir if labels_dir is not None and labels_dir.is_dir() else None,
        left_frames if left_frames is not None and left_frames.is_dir() else None,
        right_frames if right_frames is not None and right_frames.is_dir() else None,
    )


def _parse_seq_id(seq_dir_name: str) -> int | None:
    # seq_1
    n = seq_dir_name.strip().lower()
    if not n.startswith("seq_"):
        return None
    try:
        return int(n.split("_")[-1])
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


def _save_mask_u8(mask: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8), mode="L").save(out_path)


def _copy_or_symlink(src: Path, dst: Path, *, symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if symlink:
        try:
            os.symlink(src, dst)
            return
        except OSError as e:
            # Some filesystems / environments disallow symlinks. Fall back to copying.
            print(f"[warn] symlink failed ({e}); copying instead: {dst}")
    shutil.copy2(src, dst)


def _clear_outputs(out_root: Path) -> None:
    # Only delete directories we create.
    for split in ["train"] + [p.name for p in out_root.glob("val*") if p.is_dir()]:
        p = out_root / split
        if p.exists():
            shutil.rmtree(p)
    meta = out_root / "endovis2018_prepared_meta.json"
    if meta.exists():
        meta.unlink()
    stats = out_root / "prepare_stats.json"
    if stats.exists():
        stats.unlink()


def _is_miccai_seq_dir(p: Path) -> bool:
    if not p.is_dir() or _is_junk_path(p):
        return False
    if not p.name.lower().startswith("seq_"):
        return False
    labels_dir, left_frames, right_frames = _resolve_seq_subdirs(p)
    if labels_dir is None:
        return False
    if left_frames is None and right_frames is None:
        return False
    return True


def _discover_seq_dirs(
    *,
    src_root: Path,
    include_test_data: bool,
    groups: list[str] | None,
) -> list[tuple[str, Path]]:
    """
    Returns: list of (group_name, seq_dir)
    """
    out: list[tuple[str, Path]] = []

    # Case 1: seq_* directly under src_root.
    direct = [p for p in sorted(src_root.iterdir()) if _is_miccai_seq_dir(p)]
    if direct:
        out.extend([("root", p) for p in direct])
        return out

    # Case 2: seq_* under one-level groups (recommended).
    groups_allow_test = set(groups) if groups is not None else set()
    if groups is not None:
        missing = [g for g in groups if not (src_root / g).is_dir()]
        if missing:
            avail = sorted(
                [p.name for p in src_root.iterdir() if p.is_dir() and not _is_junk_path(p)]
            )
            print(
                "[warn] Some --groups entries were not found under --src-root (may be nested one level deeper). "
                f"missing={missing} available={avail}"
            )
    for group in sorted(src_root.iterdir()):
        if not group.is_dir() or _is_junk_path(group):
            continue
        if group.name == "miccai-2018-code":
            continue
        # Default safety: do not include official test_data unless explicitly asked.
        # If the user explicitly passes "--groups test_data", we honor it even if include_test_data=False.
        if group.name == "test_data" and (not include_test_data) and ("test_data" not in groups_allow_test):
            continue
        if groups is not None and group.name not in set(groups):
            continue

        for p in sorted(group.iterdir()):
            if _is_miccai_seq_dir(p):
                out.append((group.name, p))

    if out:
        return out

    # Fallback: bounded recursive search (depth<=4) with pruning (avoid descending into frame dirs).
    # This handles slightly different unzip layouts, e.g. an extra wrapper folder under each release.
    max_depth = 4
    prune_names = {"left_frames", "right_frames", "labels", "label"}
    groups_set = set(groups) if groups is not None else None

    def _is_under_named_ancestor(p: Path, name: str) -> bool:
        for parent in p.parents:
            if parent == src_root:
                break
            if parent.name == name:
                return True
        return False

    def _allowed_by_filters(seq_dir: Path) -> bool:
        # Exclude official test_data unless explicitly asked.
        if _is_under_named_ancestor(seq_dir, "test_data") and (not include_test_data) and ("test_data" not in groups_allow_test):
            return False
        if groups_set is None:
            return True
        # Allow if any ancestor matches one of the requested groups.
        for parent in seq_dir.parents:
            if parent == src_root:
                break
            if parent.name in groups_set:
                return True
        return False

    def _pick_group_name(seq_dir: Path) -> str:
        if groups_set is not None:
            for parent in seq_dir.parents:
                if parent == src_root:
                    break
                if parent.name in groups_set:
                    return parent.name
        # Default: first-level folder under src_root.
        try:
            rel = seq_dir.relative_to(src_root).parts
            if rel:
                return rel[0]
        except Exception:
            pass
        return seq_dir.parent.name

    q: list[tuple[Path, int]] = [(src_root, 0)]
    seen: set[Path] = set()
    while q:
        cur, depth = q.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        if depth > max_depth:
            continue
        for child in sorted(_safe_iterdir(cur)):
            if not child.is_dir() or _is_junk_path(child):
                continue
            name_l = child.name.lower()
            if name_l in prune_names or name_l.endswith("_frames"):
                continue
            if _is_miccai_seq_dir(child) and _allowed_by_filters(child):
                out.append((_pick_group_name(child), child))
                continue
            if depth < max_depth:
                q.append((child, depth + 1))

    # De-duplicate paths (can happen if multiple discovery strategies overlap).
    uniq: dict[str, Path] = {}
    for g, p in out:
        uniq[str(p)] = p
    out = [(_pick_group_name(p), p) for p in uniq.values()]

    return sorted(out, key=lambda x: (x[0], x[1].as_posix()))


def _diagnose_layout(src_root: Path, *, groups: list[str] | None, include_test_data: bool) -> str:
    """
    Best-effort, low-cost diagnostics for why no seq dirs were discovered.
    """
    group_names: list[str]
    if groups is not None:
        group_names = list(groups)
    else:
        group_names = [p.name for p in _safe_iterdir(src_root) if p.is_dir() and not _is_junk_path(p)]
    lines: list[str] = []
    lines.append("Layout diagnostics (first 8 entries per group):")
    for g in sorted(set(group_names)):
        gp = src_root / g
        if not gp.is_dir():
            continue
        if g == "test_data" and (not include_test_data) and (groups is None or "test_data" not in set(groups)):
            continue
        kids = [p for p in _safe_iterdir(gp) if p.is_dir() and not _is_junk_path(p)]
        lines.append(f"- group={g} subdirs={len(kids)} sample={[p.name for p in kids[:8]]}")
        cand = [p for p in kids if p.name.lower().startswith("seq")]
        if cand:
            for p in cand[:3]:
                lbl, lf, rf = _resolve_seq_subdirs(p)
                lines.append(
                    f"  - {g}/{p.name}: labels={lbl.name if lbl else None} left_frames={lf.name if lf else None} right_frames={rf.name if rf else None}"
                )
    return "\n".join(lines)


def _find_labels_json(src_root: Path, seq_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"--labels-json does not exist: {explicit}")
        return explicit

    # Prefer a labels.json near this sequence.
    for parent in [seq_dir] + list(seq_dir.parents):
        if parent == src_root.parent:
            break
        cand = parent / "labels.json"
        if cand.is_file():
            return cand

    # Fall back to common locations under src_root (1-level).
    for cand in sorted(src_root.glob("*/labels.json")):
        if cand.is_file():
            return cand
    cand = src_root / "labels.json"
    if cand.is_file():
        return cand

    raise FileNotFoundError(
        f"Could not find labels.json under {src_root} (and none near seq_dir={seq_dir}). "
        "Pass --labels-json explicitly."
    )


def _load_color_map(labels_json: Path) -> tuple[dict[tuple[int, int, int], int], dict[int, str]]:
    data = json.loads(labels_json.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"labels.json must be a list of entries, got: {type(data)}")

    color_to_id: dict[tuple[int, int, int], int] = {}
    id_to_name: dict[int, str] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        cid = int(item.get("classid", -1))
        if cid < 0:
            continue
        name = str(item.get("name", f"class_{cid}"))
        color = item.get("color", None)
        if (
            not isinstance(color, list)
            or len(color) != 3
            or not all(isinstance(x, (int, float)) for x in color)
        ):
            continue
        rgb = (int(color[0]), int(color[1]), int(color[2]))
        color_to_id[rgb] = cid
        id_to_name[cid] = name
    if not color_to_id:
        raise RuntimeError(f"Failed to parse any color entries from labels.json: {labels_json}")
    return color_to_id, id_to_name


def _load_color_map_union(
    *,
    src_root: Path,
    seq_dir: Path,
    explicit: Path | None,
) -> tuple[dict[tuple[int, int, int], int], dict[int, str], list[Path]]:
    """
    EndoVis2018 releases sometimes ship multiple `labels.json` files across releases.
    Some of them may omit rare classes (e.g., ultrasound-probe, class 11).

    This helper merges all available `labels.json` files (unless an explicit path is given),
    and sanity-checks that mappings are consistent.
    """
    if explicit is not None:
        color_to_id, id_to_name = _load_color_map(explicit)
        return color_to_id, id_to_name, [explicit]

    # Collect candidates (prefer 1-level under src_root; ignore junk).
    cands: list[Path] = []
    for p in sorted(src_root.glob("*/labels.json")):
        if p.is_file() and not _is_junk_path(p.parent):
            cands.append(p)
    p0 = src_root / "labels.json"
    if p0.is_file():
        cands.append(p0)

    # Also consider a labels.json near the first sequence (in case src_root search is empty).
    try:
        near = _find_labels_json(src_root, seq_dir, explicit=None)
        if near not in cands:
            cands.insert(0, near)
    except FileNotFoundError:
        pass

    merged_color_to_id: dict[tuple[int, int, int], int] = {}
    merged_id_to_name: dict[int, str] = {}
    used: list[Path] = []

    for p in cands:
        try:
            c2i, i2n = _load_color_map(p)
        except Exception:
            # Skip unrelated / malformed files (e.g., miccai-2018-code/labels.json).
            continue

        # Merge with consistency checks.
        for rgb, cid in c2i.items():
            if rgb in merged_color_to_id and int(merged_color_to_id[rgb]) != int(cid):
                raise RuntimeError(
                    "Conflicting labels.json mappings: "
                    f"RGB={rgb} maps to {merged_color_to_id[rgb]} and {cid}. "
                    f"Conflicting file: {p}"
                )
            merged_color_to_id[rgb] = int(cid)
        for cid, name in i2n.items():
            if cid in merged_id_to_name and merged_id_to_name[cid] != name:
                # Names differ, but class ids/colors match; keep the first name.
                continue
            merged_id_to_name[int(cid)] = str(name)
        used.append(p)

    if not merged_color_to_id:
        raise FileNotFoundError(
            f"Could not find any usable labels.json under {src_root}. "
            "Pass --labels-json explicitly."
        )
    return merged_color_to_id, merged_id_to_name, used


def _label_to_class_id_mask(
    label_path: Path, *, color_to_id: dict[tuple[int, int, int], int]
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """
    Returns:
      mask_u8: (H,W) uint8 class ids
      unknown_colors: list of RGB colors not present in color_to_id
    """
    with Image.open(label_path) as im:
        arr = np.array(im)

    if arr.ndim == 2:
        # Already a class-id mask.
        return arr.astype(np.uint8), []

    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"Unsupported label image shape at {label_path}: {arr.shape}")

    rgb = arr[..., :3].astype(np.uint8)
    h, w = rgb.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)

    # Map known colors.
    for (r, g, b), cid in color_to_id.items():
        m = (rgb[..., 0] == r) & (rgb[..., 1] == g) & (rgb[..., 2] == b)
        if m.any():
            out[m] = np.uint8(cid)

    # Detect unknown colors (for debugging).
    uniq = np.unique(rgb.reshape(-1, 3), axis=0)
    unknown: list[tuple[int, int, int]] = []
    known = set(color_to_id.keys())
    for c in uniq:
        t = (int(c[0]), int(c[1]), int(c[2]))
        if t not in known:
            unknown.append(t)
    return out, unknown


@dataclass(frozen=True)
class PreparedCounts:
    sequences: int = 0
    val_images: int = 0
    val_labels: int = 0
    train_symlinks_images: int = 0
    train_symlinks_labels: int = 0


def prepare_miccai_endovis2018(
    *,
    src_root: Path,
    out_root: Path,
    camera: str,
    symlink_images: bool,
    create_train_view: bool,
    include_test_data: bool,
    groups: list[str] | None,
    labels_json: Path | None,
    seq_ids: list[int] | None,
    max_frames_per_seq: int | None,
) -> tuple[PreparedCounts, dict[int, str]]:
    cam_dir = "left_frames" if camera == "left" else "right_frames"

    seq_entries = _discover_seq_dirs(
        src_root=src_root,
        include_test_data=bool(include_test_data),
        groups=groups,
    )
    if not seq_entries:
        avail = sorted([p.name for p in src_root.iterdir() if p.is_dir() and not _is_junk_path(p)])
        hint = ""
        if len(avail) == 1:
            hint = (
                "\nHint: your dataset may be nested one level deeper. Try setting --src-root to:\n"
                f"  {src_root / avail[0]}"
            )
        diag = _diagnose_layout(src_root, groups=groups, include_test_data=bool(include_test_data))
        raise RuntimeError(
            f"No seq_* directories found under: {src_root}\n"
            "Expected a MICCAI2018-style layout containing seq_<id>/{left_frames,right_frames,labels}."
            f"\nTop-level dirs under src_root: {avail}{hint}\n{diag}"
        )
    # Helpful debug summary when running on new servers / datasets.
    group_counts: dict[str, int] = {}
    for group_name, _ in seq_entries:
        group_counts[group_name] = int(group_counts.get(group_name, 0)) + 1
    print(f"[prepare_endovis2018_official] discovered_sequences={len(seq_entries)} by_group={group_counts}")

    # Load label color map from the first sequence's neighborhood (or explicitly provided file).
    color_to_id, id_to_name, labels_json_paths = _load_color_map_union(
        src_root=src_root,
        seq_dir=seq_entries[0][1],
        explicit=labels_json,
    )
    labels_json_paths = sorted(set(labels_json_paths))
    labels_json_primary = labels_json_paths[0] if labels_json_paths else None
    print(
        f"[prepare_endovis2018_official] labels_json_primary={labels_json_primary} "
        f"num_labels_json={len(labels_json_paths)} num_classes={len(set(color_to_id.values()))}"
    )

    out_train_img = out_root / "train" / "image" if create_train_view else None
    out_train_lbl = out_root / "train" / "label" if create_train_view else None

    sequences_written = 0
    val_images = 0
    val_labels = 0
    train_symlinks_images = 0
    train_symlinks_labels = 0

    seq_meta: list[dict] = []
    unknown_colors_global: dict[str, list[tuple[int, int, int]]] = {}

    # Track used val dir names to avoid collisions.
    used_val_dirs: set[str] = set()

    for group_name, seq_dir in seq_entries:
        sid = _parse_seq_id(seq_dir.name)
        if sid is None:
            continue
        if seq_ids is not None and int(sid) not in set(int(x) for x in seq_ids):
            continue

        # Default: use val{sid}. If collision, add group suffix.
        val_dir_name = f"val{sid}"
        if val_dir_name in used_val_dirs or (out_root / val_dir_name).exists():
            safe_group = "".join(c if c.isalnum() else "_" for c in group_name).strip("_")
            val_dir_name = f"val{sid}_{safe_group}"
        used_val_dirs.add(val_dir_name)

        labels_dir, left_frames, right_frames = _resolve_seq_subdirs(seq_dir)
        if labels_dir is None:
            raise FileNotFoundError(f"Missing labels dir under: {seq_dir}")
        img_dir = seq_dir / cam_dir
        if not img_dir.is_dir():
            # tolerate casing differences.
            img_dir = _find_child_dir(seq_dir, cam_dir) or img_dir
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Missing frames dir: {img_dir}")
        lbl_dir = labels_dir

        out_val_img = out_root / val_dir_name / "image"
        out_val_lbl = out_root / val_dir_name / "label"

        label_paths = sorted(p for p in lbl_dir.glob("frame*.png") if p.is_file() and not _is_junk_path(p))
        if max_frames_per_seq is not None and int(max_frames_per_seq) > 0:
            label_paths = label_paths[: int(max_frames_per_seq)]

        frames_written = 0
        unknown_colors_seq: set[tuple[int, int, int]] = set()

        for lbl_path in label_paths:
            frame_idx = _parse_frame_idx(lbl_path.name)
            if frame_idx is None:
                continue

            img_path = img_dir / lbl_path.name
            if not img_path.is_file():
                # Some releases have sparse labels; skip missing images.
                continue

            out_name = f"seq_{sid}_frame{frame_idx:04d}.png"
            out_img_path = out_val_img / out_name
            out_lbl_path = out_val_lbl / out_name

            _copy_or_symlink(img_path, out_img_path, symlink=bool(symlink_images))

            mask_u8, unknown = _label_to_class_id_mask(lbl_path, color_to_id=color_to_id)
            _save_mask_u8(mask_u8, out_lbl_path)

            for c in unknown:
                unknown_colors_seq.add(tuple(c))

            if create_train_view:
                assert out_train_img is not None and out_train_lbl is not None
                # Add a deduplicated train view (symlink to val outputs).
                train_img_name = f"{val_dir_name}_{out_name}"
                train_lbl_name = f"{val_dir_name}_{out_name}"
                _copy_or_symlink(out_img_path, out_train_img / train_img_name, symlink=True)
                _copy_or_symlink(out_lbl_path, out_train_lbl / train_lbl_name, symlink=True)

            val_images += 1
            val_labels += 1
            if create_train_view:
                train_symlinks_images += 1
                train_symlinks_labels += 1
            frames_written += 1

        if unknown_colors_seq:
            unknown_colors_global[val_dir_name] = sorted(list(unknown_colors_seq))

        if frames_written == 0:
            print(f"[warn] No frames written for {seq_dir} (group={group_name})")
            continue

        sequences_written += 1
        seq_meta.append(
            {
                "val_name": val_dir_name,
                "group": group_name,
                "seq_dir": str(seq_dir),
                "seq_id": int(sid),
                "camera": camera,
                "frames": int(frames_written),
            }
        )

    # Write per-run stats (root-level, so test-only outputs don't create a misleading `train/` folder).
    stats_payload = json.dumps(
        {
            "sequences": int(sequences_written),
            "val_images": int(val_images),
            "val_labels": int(val_labels),
            "train_symlinks_images": int(train_symlinks_images),
            "train_symlinks_labels": int(train_symlinks_labels),
            "unknown_colors_by_val": {
                k: [list(x) for x in v] for k, v in sorted(unknown_colors_global.items())
            },
            "labels_json_primary": str(labels_json_primary) if labels_json_primary else None,
            "labels_json_all": [str(p) for p in labels_json_paths],
            "groups_filter": groups,
            "include_test_data": bool(include_test_data),
            "seq_ids_filter": seq_ids,
            "max_frames_per_seq": max_frames_per_seq,
            "create_train_view": bool(create_train_view),
        },
        indent=2,
    )
    (out_root / "prepare_stats.json").write_text(stats_payload, encoding="utf-8")
    if create_train_view:
        (out_root / "train" / "prepare_stats.json").write_text(stats_payload, encoding="utf-8")

    # Backward compatible: keep the original `train/prepare_stats.json` structure for train roots.
    # (No additional actions needed; written above when create_train_view=True.)

    counts = PreparedCounts(
        sequences=int(sequences_written),
        val_images=int(val_images),
        val_labels=int(val_labels),
        train_symlinks_images=int(train_symlinks_images),
        train_symlinks_labels=int(train_symlinks_labels),
    )

    # Write meta at root for downstream scripts.
    meta = {
        "src_root": str(src_root),
        "out_root": str(out_root),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "camera": camera,
        "symlink_images": bool(symlink_images),
        "create_train_view": bool(create_train_view),
        "counts": {
            "sequences": int(counts.sequences),
            "val_images": int(counts.val_images),
            "val_labels": int(counts.val_labels),
            "train_symlinks_images": int(counts.train_symlinks_images),
            "train_symlinks_labels": int(counts.train_symlinks_labels),
        },
        "labels_json_primary": str(labels_json_primary) if labels_json_primary else None,
        "labels_json_all": [str(p) for p in labels_json_paths],
        "classes": {str(int(k)): str(v) for k, v in sorted(id_to_name.items())},
        "sequences": seq_meta,
    }
    meta_path = out_root / "endovis2018_prepared_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] wrote meta: {meta_path}")

    return counts, id_to_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True, type=str, help="Official EndoVis2018 root (contains miccai_challenge_* dirs)")
    ap.add_argument("--out-root", required=True, type=str, help="Output root (will create train/ and val*/)")
    ap.add_argument("--camera", type=str, default="left", choices=["left", "right"], help="Use left_frames or right_frames")
    ap.add_argument("--symlink-images", action="store_true", help="Symlink images instead of copying")
    ap.add_argument(
        "--no-train-view",
        action="store_true",
        help="Do not create a flat `train/` view (only write per-sequence `val*` outputs).",
    )
    ap.add_argument("--overwrite", action="store_true", help="Delete existing train/val* outputs under out-root first")
    ap.add_argument("--include-test-data", action="store_true", help="Also include a top-level test_data/ folder if present")
    ap.add_argument(
        "--groups",
        type=str,
        nargs="*",
        default=None,
        help="Optional: explicit group folder names under src-root to include (e.g., miccai_challenge_release_4).",
    )
    ap.add_argument("--labels-json", type=str, default=None, help="Optional: explicit path to labels.json (RGB->class id map)")
    ap.add_argument("--seq-ids", type=int, nargs="*", default=None, help="Optional: only process these seq ids (from seq_<id>)")
    ap.add_argument("--max-frames-per-seq", type=int, default=None, help="Optional: limit frames per sequence (debug)")
    args = ap.parse_args()

    src_root = Path(args.src_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    labels_json = Path(args.labels_json).expanduser().resolve() if args.labels_json else None

    if not src_root.exists():
        raise FileNotFoundError(f"--src-root does not exist: {src_root}")
    if src_root == out_root:
        raise ValueError("--out-root must be different from --src-root")

    if args.overwrite:
        _clear_outputs(out_root)
    else:
        if (out_root / "train").exists() or any(out_root.glob("val*")):
            raise FileExistsError(f"Output already exists under: {out_root} (use --overwrite)")

    t0 = time.time()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[prepare_endovis2018_official] src_root={src_root}")
    print(f"[prepare_endovis2018_official] out_root={out_root}")
    print(f"[prepare_endovis2018_official] camera={args.camera} symlink_images={args.symlink_images}")
    print(f"[prepare_endovis2018_official] include_test_data={args.include_test_data} groups={args.groups}")

    counts, _ = prepare_miccai_endovis2018(
        src_root=src_root,
        out_root=out_root,
        camera=args.camera,
        symlink_images=bool(args.symlink_images),
        create_train_view=not bool(args.no_train_view),
        include_test_data=bool(args.include_test_data),
        groups=args.groups,
        labels_json=labels_json,
        seq_ids=args.seq_ids,
        max_frames_per_seq=args.max_frames_per_seq,
    )

    dt = time.time() - t0
    print(
        f"[done] sequences={counts.sequences} val_images={counts.val_images} val_labels={counts.val_labels} "
        f"train_symlinks_images={counts.train_symlinks_images} train_symlinks_labels={counts.train_symlinks_labels} "
        f"secs={dt:.1f}"
    )


if __name__ == "__main__":
    main()
