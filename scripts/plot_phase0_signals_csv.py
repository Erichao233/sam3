#!/usr/bin/env python3

"""
Plot `signals.csv` produced by `phase0_eval_ultrasound_signals.py`.

This is useful when the server environment doesn't have matplotlib and you only
download the CSVs back to local for plotting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals-csv", required=True, type=str)
    ap.add_argument("--out-png", required=True, type=str)
    ap.add_argument("--title", type=str, default=None)
    args = ap.parse_args()

    try:
        import pandas as pd
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"pandas is required for this script: {e}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"matplotlib is required for this script: {e}")

    csv_path = Path(args.signals_csv)
    df = pd.read_csv(csv_path)
    title = args.title or csv_path.parent.as_posix()

    x = df["frame"].astype(int).to_numpy()
    det_score = df.get("det_score_max", pd.Series([np.nan] * len(df))).astype(float).to_numpy()
    presence = df.get("presence_prob", pd.Series([np.nan] * len(df))).astype(float).to_numpy()
    trk_score = df.get("tracker_score_max", pd.Series([np.nan] * len(df))).astype(float).to_numpy()
    q_cos = df.get("query_cos", pd.Series([np.nan] * len(df))).astype(float).to_numpy()
    gt_present = df.get("gt_present", pd.Series([0] * len(df))).astype(int).to_numpy()
    pred_area = df.get("pred_area", pd.Series([0] * len(df))).astype(float).to_numpy()

    fig = plt.figure(figsize=(12, 6), dpi=150)
    fig.suptitle(title)

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(x, det_score, label="det_score_max", linewidth=1.5)
    if not np.all(np.isnan(presence)):
        ax1.plot(x, presence, label="presence_prob", linewidth=1.0)
    if not np.all(np.isnan(trk_score)):
        ax1.plot(x, trk_score, label="tracker_score_max", linewidth=1.0)
    if not np.all(np.isnan(q_cos)):
        ax1.plot(x, q_cos, label="query_cos", linewidth=1.0)
    ax1.set_ylabel("score")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="upper right")

    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    ax2.plot(x, pred_area, label="pred_area", linewidth=1.0)
    ax2.fill_between(
        x,
        0,
        gt_present.astype(float) * max(float(pred_area.max()), 1.0),
        alpha=0.2,
        label="gt_present",
    )
    ax2.set_ylabel("area / present")
    ax2.set_xlabel("frame")
    ax2.legend(loc="upper right")

    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
