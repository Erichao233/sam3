#!/usr/bin/env python3

"""
Scan an ultrasound GT folder to find videos that contain GT-absent segments.

This is mainly for Phase-0 "persistence/ghost" evaluation:
- tail_absent_len: frames after last GT-present frame (needle removed)
- longest_absent_run: longest contiguous run of GT-absent frames (occlusion/out-of-view)

Expected layout (same as phase0_eval_ultrasound_signals.py):
  GT_ROOT/
    <sample_id>/
      <frame>.png

Note:
- Some datasets do not write empty GT masks; instead the GT file may be missing on absent frames.
  If so, pass `--frames-root` so we can treat missing GT as "absent" consistently with Phase-0 eval.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image


def _numeric_png_sort_key(p: Path):
    try:
        return int(p.stem)
    except ValueError:
        return p.stem


def _mask_present(path: Path, min_area: int) -> bool:
    if min_area <= 0:
        min_area = 1
    with Image.open(path) as im:
        im = im.convert("L")
        # Fast path for the common case (binary masks with large foreground).
        if min_area == 1:
            return im.getbbox() is not None
        # Robust path: treat tiny speckles as "absent" if min_area > 1.
        # This is slower, but Phase-0 scanning is usually a one-off.
        w, h = im.size
        if w == 0 or h == 0:
            return False
        # Count non-zero pixels.
        # Note: im.getdata() avoids an extra numpy dependency.
        count = 0
        for v in im.getdata():
            if v:
                count += 1
                if count >= min_area:
                    return True
        return False


def _runs_of_false(flags: list[bool]) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    start = None
    for i, v in enumerate(flags):
        if (not v) and start is None:
            start = i
        if v and start is not None:
            runs.append((start, i - 1, i - start))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1, len(flags) - start))
    return runs


@dataclass
class SampleAbsenceInfo:
    sample_id: str
    num_frames: int
    num_gt_present: int
    last_gt_present_frame: int | None
    tail_absent_len: int
    num_absent_frames: int
    num_absent_runs: int
    longest_absent_run: int


def _analyze_one_sample_from_gt_dir(sample_dir: Path, present_min_area: int) -> SampleAbsenceInfo | None:
    mask_paths = sorted(sample_dir.glob("*.png"), key=_numeric_png_sort_key)
    if not mask_paths:
        return None

    gt_present: list[bool] = []
    for p in mask_paths:
        gt_present.append(_mask_present(p, min_area=present_min_area))

    num_frames = len(gt_present)
    present_indices = [i for i, v in enumerate(gt_present) if v]
    if not present_indices:
        # No GT-positive frame -> not useful for "removed tail" persistence.
        return None

    last_present = present_indices[-1]
    tail_absent_len = max(0, num_frames - 1 - last_present)

    absent_runs = _runs_of_false(gt_present)
    num_absent_frames = sum((1 for v in gt_present if not v))
    num_absent_runs = len(absent_runs)
    longest_absent_run = max((l for _, _, l in absent_runs), default=0)

    return SampleAbsenceInfo(
        sample_id=sample_dir.name,
        num_frames=num_frames,
        num_gt_present=len(present_indices),
        last_gt_present_frame=last_present,
        tail_absent_len=tail_absent_len,
        num_absent_frames=num_absent_frames,
        num_absent_runs=num_absent_runs,
        longest_absent_run=longest_absent_run,
    )


def _analyze_one_sample_from_frames_dir(
    sample_id: str,
    frames_dir: Path,
    gt_dir: Path,
    present_min_area: int,
) -> SampleAbsenceInfo | None:
    frame_paths = sorted(frames_dir.glob("*.png"), key=_numeric_png_sort_key)
    if not frame_paths:
        return None

    gt_present: list[bool] = []
    for fp in frame_paths:
        gt_path = gt_dir / fp.name
        if not gt_path.exists():
            gt_present.append(False)
        else:
            gt_present.append(_mask_present(gt_path, min_area=present_min_area))

    num_frames = len(gt_present)
    present_indices = [i for i, v in enumerate(gt_present) if v]
    if not present_indices:
        # No GT-positive frame -> not useful for "removed tail" persistence.
        return None
    last_present = present_indices[-1]
    tail_absent_len = max(0, num_frames - 1 - last_present)

    absent_runs = _runs_of_false(gt_present)
    num_absent_frames = sum((1 for v in gt_present if not v))
    num_absent_runs = len(absent_runs)
    longest_absent_run = max((l for _, _, l in absent_runs), default=0)

    return SampleAbsenceInfo(
        sample_id=sample_id,
        num_frames=num_frames,
        num_gt_present=len(present_indices),
        last_gt_present_frame=last_present,
        tail_absent_len=tail_absent_len,
        num_absent_frames=num_absent_frames,
        num_absent_runs=num_absent_runs,
        longest_absent_run=longest_absent_run,
    )


def _canonicalize_id(raw: str) -> str:
    s = raw.strip()
    if not s:
        return ""
    # allow "id ..." lines
    s = s.split()[0]
    # allow "path/to/id" or "id.png"
    s = s.replace("\\", "/")
    s = s.split("/")[-1]
    if "." in s:
        s = s.rsplit(".", 1)[0]
    return s


def _try_parse_int(s: str) -> int | None:
    try:
        return int(s)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-root", required=True, type=str, help=".../gt (contains <sample_id>/<frame>.png)")
    ap.add_argument(
        "--frames-root",
        type=str,
        default=None,
        help="Optional: .../pork (contains <sample_id>/<frame>.png). If set, missing GT files are treated as absent.",
    )
    ap.add_argument(
        "--sample-ids",
        type=str,
        default=None,
        help="Comma-separated sample ids to consider (folder names under --gt-root).",
    )
    ap.add_argument(
        "--sample-txt",
        type=str,
        default=None,
        help="Txt with one sample id per line to consider (e.g., .../pork_dataset/test.txt).",
    )
    ap.add_argument(
        "--present-min-area",
        type=int,
        default=1,
        help="Treat GT as present only if non-zero pixels >= this value (filters tiny speckles).",
    )
    ap.add_argument("--min-tail", type=int, default=1, help="Minimum absent tail length to report.")
    ap.add_argument(
        "--min-longest-absent",
        type=int,
        default=1,
        help="Minimum longest absent run length to report.",
    )
    ap.add_argument("--top", type=int, default=50, help="Max rows to print (sorted by tail, then longest run).")
    ap.add_argument("--out-txt", type=str, default=None, help="Write selected sample ids (one per line).")
    args = ap.parse_args()

    gt_root = Path(args.gt_root)
    frames_root = Path(args.frames_root) if args.frames_root else None

    selected: set[str] | None = None
    if args.sample_ids:
        selected = {_canonicalize_id(s) for s in args.sample_ids.split(",") if _canonicalize_id(s)}
    if args.sample_txt:
        ids = [
            _canonicalize_id(ln)
            for ln in Path(args.sample_txt).read_text(encoding="utf-8").splitlines()
            if _canonicalize_id(ln)
        ]
        selected = set(ids) if selected is None else (selected & set(ids))

    selected_ints: set[int] | None = None
    if selected is not None:
        ints = [_try_parse_int(s) for s in selected]
        selected_ints = {x for x in ints if x is not None}

    infos: list[SampleAbsenceInfo] = []
    if frames_root is None:
        for sample_dir in sorted([p for p in gt_root.iterdir() if p.is_dir()], key=lambda p: p.name):
            if selected is not None:
                if sample_dir.name in selected:
                    pass
                elif selected_ints and _try_parse_int(sample_dir.name) in selected_ints:
                    pass
                else:
                    continue
            info = _analyze_one_sample_from_gt_dir(sample_dir, present_min_area=args.present_min_area)
            if info is None:
                continue
            if info.tail_absent_len < args.min_tail and info.longest_absent_run < args.min_longest_absent:
                continue
            infos.append(info)
    else:
        # Iterate sample dirs from frames_root to align with Phase-0 eval.
        for sample_frames_dir in sorted([p for p in frames_root.iterdir() if p.is_dir()], key=lambda p: p.name):
            sample_id = sample_frames_dir.name
            if selected is not None:
                if sample_id in selected:
                    pass
                elif selected_ints and _try_parse_int(sample_id) in selected_ints:
                    pass
                else:
                    continue
            gt_dir = gt_root / sample_id
            if not gt_dir.exists():
                continue
            info = _analyze_one_sample_from_frames_dir(
                sample_id=sample_id,
                frames_dir=sample_frames_dir,
                gt_dir=gt_dir,
                present_min_area=args.present_min_area,
            )
            if info is None:
                continue
            if info.tail_absent_len < args.min_tail and info.longest_absent_run < args.min_longest_absent:
                continue
            infos.append(info)

    infos.sort(key=lambda x: (x.tail_absent_len, x.longest_absent_run, x.num_absent_frames), reverse=True)
    infos = infos[: max(0, int(args.top))]

    if not infos:
        print("No samples matched the filters.")
        return

    header = [
        "sample_id",
        "num_frames",
        "num_gt_present",
        "last_gt_present_frame",
        "tail_absent_len",
        "num_absent_frames",
        "num_absent_runs",
        "longest_absent_run",
    ]
    print("\t".join(header))
    for info in infos:
        d = asdict(info)
        print("\t".join(str(d[k]) for k in header))

    if args.out_txt:
        Path(args.out_txt).write_text(
            "\n".join([i.sample_id for i in infos]) + "\n", encoding="utf-8"
        )
        print(f"Wrote {len(infos)} ids to {args.out_txt}")


if __name__ == "__main__":
    main()
