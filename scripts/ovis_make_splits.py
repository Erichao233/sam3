#!/usr/bin/env python3
"""
Create reproducible train/val/test splits from OVIS *train* annotations.

OVIS official `annotations_valid.json` / `annotations_test.json` do not include GT
(`annotations` is null), so for local research you typically split the train set.

This script outputs three text files listing `video_id`s, plus a small JSON with
category name mapping for convenience.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def _hash01(s: str) -> float:
    # Deterministic in [0, 1). Use 64 bits for low collision / stable splits.
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:16]
    return int(h, 16) / float(16**16)


def _summarize_split(
    name: str,
    video_ids: list[int],
    vid_to_cats: dict[int, set[int]],
    cat_id_to_name: dict[int, str],
):
    cat_counts = Counter()
    for vid in video_ids:
        for cid in vid_to_cats.get(vid, set()):
            cat_counts[cid] += 1
    top = cat_counts.most_common(10)
    top_str = ", ".join(f"{cat_id_to_name[c]}:{n}" for c, n in top)
    print(f"{name}: videos={len(video_ids)}; top cats (videos containing cat) = {top_str}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ann",
        type=Path,
        required=True,
        help="Path to OVIS annotations_train.json (must include GT).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for split files.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for deterministic hash split.",
    )
    ap.add_argument("--train-frac", type=float, default=0.80)
    ap.add_argument("--val-frac", type=float, default=0.10)
    args = ap.parse_args()

    if not args.ann.exists():
        raise FileNotFoundError(args.ann)

    obj = json.loads(args.ann.read_text())
    if obj.get("annotations") is None:
        raise ValueError(
            f"{args.ann} has `annotations=null` (no GT). Use annotations_train.json."
        )

    categories = obj["categories"]
    cat_id_to_name = {int(c["id"]): str(c["name"]) for c in categories}

    videos = obj["videos"]
    all_video_ids = [int(v["id"]) for v in videos]

    # Build video->categories map for quick sanity distribution prints.
    vid_to_cats: dict[int, set[int]] = defaultdict(set)
    for ann in obj["annotations"]:
        vid_to_cats[int(ann["video_id"])].add(int(ann["category_id"]))

    train_ids: list[int] = []
    val_ids: list[int] = []
    test_ids: list[int] = []
    for vid in sorted(all_video_ids):
        r = _hash01(f"{args.seed}:{vid}")
        if r < args.train_frac:
            train_ids.append(vid)
        elif r < args.train_frac + args.val_frac:
            val_ids.append(vid)
        else:
            test_ids.append(vid)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "train.txt").write_text("\n".join(map(str, train_ids)) + "\n")
    (args.out_dir / "val.txt").write_text("\n".join(map(str, val_ids)) + "\n")
    (args.out_dir / "test.txt").write_text("\n".join(map(str, test_ids)) + "\n")
    (args.out_dir / "categories.json").write_text(
        json.dumps(cat_id_to_name, indent=2, ensure_ascii=False) + "\n"
    )

    _summarize_split("train", train_ids, vid_to_cats, cat_id_to_name)
    _summarize_split("val", val_ids, vid_to_cats, cat_id_to_name)
    _summarize_split("test", test_ids, vid_to_cats, cat_id_to_name)
    print(f"Wrote splits to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

