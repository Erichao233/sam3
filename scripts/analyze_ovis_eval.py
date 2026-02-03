#!/usr/bin/env python3
"""
Analyze OVIS Phase-0 eval outputs produced by `scripts/phase0_eval_ovis_signals.py`.

This script is meant to answer:
  - Did a run improve over baseline (with uncertainty)?
  - Which instances improved/regressed the most?
  - Is the improvement driven by identity swaps / drift / FP-absent reductions?

Typical usage:
  python scripts/analyze_ovis_eval.py \
    --baseline-dir test/baseline \
    --run-dir test/spme_fusion_trained \
    --top-k 20 \
    --write-signal-lists out_lists
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _load_summaries(root: Path) -> dict[tuple[int, int], dict[str, Any]]:
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for p in root.rglob("summary.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(f"Failed to read {p}: {e}") from e
        key = (int(d["video_id"]), int(d["ann_id"]))
        out[key] = d
    return out


def _fp_absent_rate_track(d: dict[str, Any]) -> float | None:
    denom = int(d.get("num_gt_absent", 0))
    if denom <= 0:
        return None
    return float(int(d.get("fp_absent_frames_track", 0)) / denom)


def _get_metric(d: dict[str, Any], name: str) -> float | None:
    if name == "fp_absent_rate_track":
        return _fp_absent_rate_track(d)
    v = d.get(name, None)
    if v is None:
        return None
    return float(v)


def _bootstrap_ci_mean(diffs: np.ndarray, *, n_boot: int, seed: int) -> tuple[float, float, float]:
    if diffs.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(int(seed))
    n = int(diffs.size)
    # (B, n) resample indices with replacement
    idx = rng.integers(0, n, size=(int(n_boot), n), endpoint=False)
    boot = diffs[idx].mean(axis=1)
    lo, med, hi = np.quantile(boot, [0.025, 0.5, 0.975])
    return float(lo), float(med), float(hi)


def _metric_lower_is_better(metric: str) -> bool:
    m = metric.strip().lower()
    if m.startswith("mean_iou") or m.startswith("mean_dice"):
        return False
    if "persistence" in m:
        return False
    # Rates / fractions / counts of failures or false positives
    if "fp" in m or "fail" in m or "swap" in m:
        return True
    return False


def _filter_keys_by_subset(
    keys: list[tuple[int, int]], base_map: dict[tuple[int, int], dict[str, Any]], subset: str
) -> list[tuple[int, int]]:
    s = subset.strip().lower()
    if s in {"all", ""}:
        return keys
    if s == "baseline_swap_gt0":
        return [k for k in keys if float(base_map[k].get("identity_swap_frac_tau", 0.0) or 0.0) > 0.0]
    if s == "baseline_fail_gt0":
        return [k for k in keys if float(base_map[k].get("fail_frac_tau", 0.0) or 0.0) > 0.0]
    if s == "baseline_gt_absent_gt0":
        return [k for k in keys if int(base_map[k].get("num_gt_absent", 0) or 0) > 0]
    if s == "baseline_fp_tail_gt0":
        return [k for k in keys if int(base_map[k].get("fp_tail_run_track", 0) or 0) > 0]
    raise ValueError(
        f"Unknown subset: {subset!r}. "
        "Use one of: all | baseline_swap_gt0 | baseline_fail_gt0 | baseline_gt_absent_gt0 | baseline_fp_tail_gt0"
    )


@dataclass(frozen=True)
class DiffRow:
    video_id: int
    ann_id: int
    category: str
    base: float
    run: float
    diff: float
    swap_base: float | None
    swap_run: float | None
    fail_base: float | None
    fail_run: float | None


def _fmt(x: float | None) -> str:
    if x is None:
        return "None"
    return f"{x:.6f}"


def _write_pairs(path: Path, rows: list[DiffRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"{r.video_id} {r.ann_id}\n")


def _summarize_one(
    *,
    baseline_dir: Path,
    run_dir: Path,
    metric: str,
    subset: str,
    top_k: int,
    n_boot: int,
    seed: int,
    write_signal_lists: Path | None,
    lower_is_better: bool | None,
) -> None:
    base_map = _load_summaries(baseline_dir)
    run_map = _load_summaries(run_dir)

    keys = sorted(set(base_map) & set(run_map))
    if not keys:
        raise RuntimeError(
            f"No overlapping instances between:\n  baseline={baseline_dir}\n  run={run_dir}\n"
        )
    keys = _filter_keys_by_subset(keys, base_map, subset=subset)
    if not keys:
        raise RuntimeError(f"No instances left after applying subset filter: {subset!r}")

    rows: list[DiffRow] = []
    diffs: list[float] = []
    for (vid, ann) in keys:
        b = base_map[(vid, ann)]
        r = run_map[(vid, ann)]
        bv = _get_metric(b, metric)
        rv = _get_metric(r, metric)
        if bv is None or rv is None:
            continue
        dif = float(rv - bv)
        rows.append(
            DiffRow(
                video_id=int(vid),
                ann_id=int(ann),
                category=str(b.get("category_name", "")),
                base=float(bv),
                run=float(rv),
                diff=dif,
                swap_base=_get_metric(b, "identity_swap_frac_tau"),
                swap_run=_get_metric(r, "identity_swap_frac_tau"),
                fail_base=_get_metric(b, "fail_frac_tau"),
                fail_run=_get_metric(r, "fail_frac_tau"),
            )
        )
        diffs.append(dif)

    diffs_np = np.asarray(diffs, dtype=np.float32)
    if diffs_np.size == 0:
        raise RuntimeError(f"Metric {metric!r} not found in summaries under {run_dir}")

    lib_default = _metric_lower_is_better(metric)
    lower = bool(lib_default if lower_is_better is None else lower_is_better)

    mean = float(diffs_np.mean())
    median = float(np.median(diffs_np))
    improved = int((diffs_np < 0).sum()) if lower else int((diffs_np > 0).sum())
    worsened = int((diffs_np > 0).sum()) if lower else int((diffs_np < 0).sum())
    equal = int((diffs_np == 0).sum())
    ci_lo, ci_med, ci_hi = _bootstrap_ci_mean(diffs_np, n_boot=n_boot, seed=seed)

    print("=" * 88)
    print(f"baseline: {baseline_dir}")
    print(f"run:      {run_dir}")
    print(f"metric:   {metric}")
    print(f"subset:   {subset}")
    print(f"direction: {'lower-better' if lower else 'higher-better'}")
    print(
        f"n={diffs_np.size}  mean={mean:+.6f}  median={median:+.6f}  "
        f"improved={improved} worsened={worsened} equal={equal}"
    )
    print(f"bootstrap mean 95% CI: [{ci_lo:+.6f}, {ci_hi:+.6f}]  (median of boot means: {ci_med:+.6f})")

    if lower:
        # More negative = better.
        rows_sorted = sorted(rows, key=lambda r: r.diff)
        top = rows_sorted[: int(top_k)]  # best improvements
        bot = sorted(rows, key=lambda r: r.diff, reverse=True)[: int(top_k)]  # worst regressions
    else:
        rows_sorted = sorted(rows, key=lambda r: r.diff, reverse=True)
        top = rows_sorted[: int(top_k)]
        bot = list(reversed(rows_sorted[-int(top_k) :]))  # most negative first

    print("\nTop improvements:")
    for r in top:
        print(
            f"  vid={r.video_id:04d} ann={r.ann_id:06d} {r.category:16s} "
            f"diff={r.diff:+.6f}  base={r.base:.6f}  run={r.run:.6f}  "
            f"swap={_fmt(r.swap_base)}→{_fmt(r.swap_run)}  fail={_fmt(r.fail_base)}→{_fmt(r.fail_run)}"
        )

    print("\nTop regressions:")
    for r in bot:
        print(
            f"  vid={r.video_id:04d} ann={r.ann_id:06d} {r.category:16s} "
            f"diff={r.diff:+.6f}  base={r.base:.6f}  run={r.run:.6f}  "
            f"swap={_fmt(r.swap_base)}→{_fmt(r.swap_run)}  fail={_fmt(r.fail_base)}→{_fmt(r.fail_run)}"
        )

    # Category-level mean diffs (only categories with >=3 samples)
    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.category:
            by_cat[r.category].append(float(r.diff))
    cat_stats = [
        (c, len(v), float(np.mean(v)), float(np.median(v)))
        for c, v in by_cat.items()
        if len(v) >= 3
    ]
    cat_stats.sort(key=lambda x: x[2], reverse=not lower)
    if cat_stats:
        print("\nCategory mean diffs (n>=3):")
        for c, n, m, med in cat_stats[:10]:
            print(f"  {c:16s} n={n:3d} mean={m:+.6f} median={med:+.6f}")

    if write_signal_lists is not None:
        out_dir = write_signal_lists
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_pairs(out_dir / "top_improved_pairs.txt", top)
        _write_pairs(out_dir / "top_regressed_pairs.txt", bot)
        print(f"\nWrote signal pair lists to: {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", required=True, type=str)
    ap.add_argument(
        "--run-dir",
        required=True,
        type=str,
        action="append",
        help="One or more run directories to compare against baseline (repeatable).",
    )
    ap.add_argument(
        "--metric",
        type=str,
        default="mean_iou_track_present",
        help="Metric key in summary.json (or 'fp_absent_rate_track' derived).",
    )
    ap.add_argument(
        "--subset",
        type=str,
        default="all",
        help="Optional subset filter: all | baseline_swap_gt0 | baseline_fail_gt0 | baseline_gt_absent_gt0 | baseline_fp_tail_gt0",
    )
    ap.add_argument(
        "--lower-is-better",
        type=int,
        default=-1,
        help="Override metric direction: 1=lower-better, 0=higher-better, -1=auto.",
    )
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--bootstrap", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--write-signal-lists",
        type=str,
        default=None,
        help="If set, writes (video_id,ann_id) lists for top improved/regressed instances.",
    )
    args = ap.parse_args()

    baseline_dir = Path(args.baseline_dir)
    if not baseline_dir.exists():
        raise SystemExit(f"baseline-dir not found: {baseline_dir}")

    write_lists = Path(args.write_signal_lists) if args.write_signal_lists else None
    lower_override = None
    if int(args.lower_is_better) in (0, 1):
        lower_override = bool(int(args.lower_is_better))

    for run in args.run_dir:
        _summarize_one(
            baseline_dir=baseline_dir,
            run_dir=Path(run),
            metric=str(args.metric),
            subset=str(args.subset),
            top_k=int(args.top_k),
            n_boot=int(args.bootstrap),
            seed=int(args.seed),
            write_signal_lists=write_lists,
            lower_is_better=lower_override,
        )


if __name__ == "__main__":
    main()
