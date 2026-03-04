#!/usr/bin/env python3
"""
Analyze target-collision benchmark results.

Input:
  future_sl_mpc/experiments/results/target_collision_benchmark.csv

Outputs:
  - target_collision_best_per_method.csv
  - target_collision_summary.txt
  - target_collision_cost_vs_safety.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    req = {"method", "knob", "knob_value", "n", "collisions", "p_hat", "ci_hi", "solve_ms_mean", "meets_target"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return df


def best_per_method(df: pd.DataFrame) -> pd.DataFrame:
    # Among configs that meet target, choose minimal solve_ms_mean
    ok = df[df["meets_target"] == 1].copy()
    if ok.empty:
        return ok
    ok["solve_ms_mean"] = ok["solve_ms_mean"].astype(float)
    idx = ok.groupby("method")["solve_ms_mean"].idxmin()
    best = ok.loc[idx].copy()
    best = best.sort_values("solve_ms_mean", ascending=True)
    return best


def write_summary(best: pd.DataFrame, out_txt: Path) -> None:
    lines = []
    if best.empty:
        lines.append("No method met the target within the configured rollout cap.\n")
    else:
        lines.append("Best config per method (among those meeting target):\n")
        cols = ["method", "knob", "knob_value", "n", "collisions", "p_hat", "ci_hi", "solve_ms_mean", "solve_ms_p95"]
        lines.append(best[cols].to_string(index=False))
        lines.append("")
        lines.append("Interpretation:")
        lines.append("- meets_target means: 95% Wilson CI upper bound (ci_hi) <= target collision rate.")
        lines.append("- solve_ms_mean is per-step MPC solve time averaged over the rollout.")
        lines.append("")
    out_txt.write_text("\n".join(lines))


def plot_cost_vs_safety(df: pd.DataFrame, best: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    # Scatter: each config is a point.
    x = df["ci_hi"].astype(float) * 100.0
    y = df["solve_ms_mean"].astype(float)
    meets = df["meets_target"].astype(int) == 1
    ax.scatter(x[~meets], y[~meets], s=20, alpha=0.25, label="config (fails target)")
    ax.scatter(x[meets], y[meets], s=35, alpha=0.75, label="config (meets target)")

    # Highlight best points per method.
    if not best.empty:
        xb = best["ci_hi"].astype(float) * 100.0
        yb = best["solve_ms_mean"].astype(float)
        ax.scatter(xb, yb, s=140, facecolors="none", edgecolors="black", linewidths=1.5, label="best per method")
        for _, r in best.iterrows():
            ax.annotate(str(r["method"]), (float(r["ci_hi"]) * 100.0, float(r["solve_ms_mean"])),
                        xytext=(6, 4), textcoords="offset points", fontsize=8)

    ax.set_xlabel("Collision upper 95% CI (%, lower is safer)")
    ax.set_ylabel("Mean solve time per step (ms, lower is faster)")
    ax.grid(True, alpha=0.25)
    ax.set_title("Target-collision benchmark: cost vs certified safety")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="future_sl_mpc/experiments/results/target_collision_benchmark.csv")
    ap.add_argument("--out", default="future_sl_mpc/experiments/results")
    args = ap.parse_args()

    csv_path = Path(args.csv).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load(csv_path)
    best = best_per_method(df)

    best_csv = out_dir / "target_collision_best_per_method.csv"
    best.to_csv(best_csv, index=False)

    out_txt = out_dir / "target_collision_summary.txt"
    write_summary(best, out_txt)

    out_png = out_dir / "target_collision_cost_vs_safety.png"
    plot_cost_vs_safety(df, best, out_png)

    print(f"Wrote {best_csv}")
    print(f"Wrote {out_txt}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()

