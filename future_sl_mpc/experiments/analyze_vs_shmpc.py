#!/usr/bin/env python3
"""
Analyze Future SL MPC experiment results and compare each method to SHMPC for efficacy.

Expects CSV from run_future_sl_experiments (columns: method, collision, total_progress,
missed_mode_steps, total_steps, avg_solve_ms, min_clearance, etc.).

Usage:
  python3 analyze_vs_shmpc.py [path/to/future_sl_rollouts.csv]
  If no path given, uses future_sl_mpc/experiments/results/future_sl_rollouts.csv

Output:
  - Printed summary table and effect sizes vs SHMPC
  - future_sl_mpc/experiments/results/analysis_summary.txt
  - future_sl_mpc/experiments/results/efficacy_vs_shmpc.csv
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "results" / "future_sl_rollouts.csv"
OUT_DIR = SCRIPT_DIR / "results"


def load_data(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.is_absolute():
        if path.exists():
            pass  # relative to cwd
        else:
            path = SCRIPT_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    assert "method" in df.columns and "collision" in df.columns
    if "scenario" not in df.columns:
        df["scenario"] = "baseline"
    return df


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def analyze_vs_shmpc(df: pd.DataFrame, scenario_filter: str | None = None) -> dict:
    """Analyze efficacy vs SHMPC. If scenario_filter is set, restrict to that scenario."""
    if scenario_filter is not None:
        df = df[df["scenario"] == scenario_filter].copy()
    methods = df["method"].unique().tolist()
    if "SHMPC" not in methods:
        raise ValueError("SHMPC baseline not found in methods")
    baseline = "SHMPC"
    others = [m for m in methods if m != baseline]

    base_df = df[df["method"] == baseline]
    n_base = len(base_df)
    n_coll_base = int(base_df["collision"].sum())
    p_coll_base = n_coll_base / n_base if n_base else 0.0
    ci_coll_base = wilson_ci(n_coll_base, n_base)

    results = {
        "baseline": baseline,
        "n_rollouts": n_base,
        "metrics": {},
    }
    # Collision rate per method
    results["metrics"]["collision_rate"] = {}
    results["metrics"]["collision_rate"][baseline] = {
        "mean": p_coll_base,
        "ci_lo": ci_coll_base[0],
        "ci_hi": ci_coll_base[1],
        "n": n_base,
    }
    for m in others:
        sub = df[df["method"] == m]
        n = len(sub)
        k = int(sub["collision"].sum())
        p = k / n if n else 0.0
        ci = wilson_ci(k, n)
        results["metrics"]["collision_rate"][m] = {"mean": p, "ci_lo": ci[0], "ci_hi": ci[1], "n": n}

    # Effect vs SHMPC: delta = SHMPC - method (positive = method is better/safer)
    results["effect_vs_shmpc"] = {}
    for m in others:
        sub = df[df["method"] == m]
        n = len(sub)
        k = int(sub["collision"].sum())
        p_m = k / n if n else 0.0
        delta_coll = p_coll_base - p_m  # positive if method has fewer collisions
        rel_improve = (delta_coll / p_coll_base * 100) if p_coll_base > 1e-9 else 0.0
        results["effect_vs_shmpc"][m] = {
            "collision_rate_delta": delta_coll,
            "collision_rate_rel_pct": rel_improve,
            "method_collision_rate": p_m,
        }

    # Progress (efficiency): higher is better
    if "total_progress" in df.columns:
        results["metrics"]["total_progress"] = {}
        for m in methods:
            sub = df[df["method"] == m]
            results["metrics"]["total_progress"][m] = {
                "mean": float(sub["total_progress"].mean()),
                "std": float(sub["total_progress"].std()) if len(sub) > 1 else 0.0,
                "n": len(sub),
            }
        prog_base = results["metrics"]["total_progress"][baseline]["mean"]
        for m in others:
            prog_m = results["metrics"]["total_progress"][m]["mean"]
            results["effect_vs_shmpc"][m]["progress_delta"] = prog_m - prog_base
            results["effect_vs_shmpc"][m]["progress_mean"] = prog_m

    # Missed mode rate
    if "total_steps" in df.columns and "missed_mode_steps" in df.columns:
        results["metrics"]["missed_mode_rate"] = {}
        for m in methods:
            sub = df[df["method"] == m]
            total_steps = sub["total_steps"].sum()
            missed = sub["missed_mode_steps"].sum()
            rate = missed / total_steps if total_steps else 0.0
            results["metrics"]["missed_mode_rate"][m] = {"mean": rate, "n": len(sub)}
        mm_base = results["metrics"]["missed_mode_rate"][baseline]["mean"]
        for m in others:
            mm_m = results["metrics"]["missed_mode_rate"][m]["mean"]
            results["effect_vs_shmpc"][m]["missed_mode_rate_delta"] = mm_base - mm_m

    # Solve time (lower is better for efficiency)
    if "avg_solve_ms" in df.columns:
        results["metrics"]["avg_solve_ms"] = {}
        for m in methods:
            sub = df[df["method"] == m]
            results["metrics"]["avg_solve_ms"][m] = {
                "mean": float(sub["avg_solve_ms"].mean()),
                "std": float(sub["avg_solve_ms"].std()) if len(sub) > 1 else 0.0,
                "n": len(sub),
            }
        time_base = results["metrics"]["avg_solve_ms"][baseline]["mean"]
        for m in others:
            time_m = results["metrics"]["avg_solve_ms"][m]["mean"]
            results["effect_vs_shmpc"][m]["solve_time_delta_ms"] = time_m - time_base

    return results


def print_and_save(results: dict, out_dir: Path) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(s: str = ""):
        print(s)
        lines.append(s)

    log("=" * 60)
    log("Future SL MPC: Efficacy vs SHMPC Baseline")
    log("=" * 60)
    log()
    log("Collision rate (lower is better)")
    log("-" * 40)
    for m, v in results["metrics"]["collision_rate"].items():
        log(f"  {m:25s}  {v['mean']:.4f}  [{v['ci_lo']:.4f}, {v['ci_hi']:.4f}]  n={v['n']}")
    log()
    log("Effect vs SHMPC (collision rate)")
    log("  delta = SHMPC_rate - method_rate  (positive = method safer)")
    for m, v in results["effect_vs_shmpc"].items():
        d = v.get("collision_rate_delta", 0)
        r = v.get("collision_rate_rel_pct", 0)
        log(f"  {m:25s}  delta={d:+.4f}  rel_improve={r:+.1f}%")
    log()
    if "total_progress" in results["metrics"]:
        log("Total progress (higher is better)")
        for m, v in results["metrics"]["total_progress"].items():
            log(f"  {m:25s}  mean={v['mean']:.2f}  std={v['std']:.2f}")
        log("  Effect vs SHMPC (progress delta):")
        for m, v in results["effect_vs_shmpc"].items():
            log(f"  {m:25s}  delta={v.get('progress_delta', 0):+.2f}")
    log()
    if "avg_solve_ms" in results["metrics"]:
        log("Avg solve time [ms] (lower is better)")
        for m, v in results["metrics"]["avg_solve_ms"].items():
            log(f"  {m:25s}  mean={v['mean']:.2f}")
        log("  Effect vs SHMPC (solve time delta ms):")
        for m, v in results["effect_vs_shmpc"].items():
            log(f"  {m:25s}  delta={v.get('solve_time_delta_ms', 0):+.2f} ms")
    log()
    log("=" * 60)

    summary_path = out_dir / "analysis_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {summary_path}")

    # Efficacy table CSV
    rows = []
    for m in results["metrics"]["collision_rate"]:
        r = {"method": m}
        r["collision_rate"] = results["metrics"]["collision_rate"][m]["mean"]
        r["collision_ci_lo"] = results["metrics"]["collision_rate"][m]["ci_lo"]
        r["collision_ci_hi"] = results["metrics"]["collision_rate"][m]["ci_hi"]
        if m != results["baseline"] and m in results["effect_vs_shmpc"]:
            r["collision_rate_delta_vs_shmpc"] = results["effect_vs_shmpc"][m].get("collision_rate_delta")
            r["progress_delta_vs_shmpc"] = results["effect_vs_shmpc"][m].get("progress_delta")
            r["solve_time_delta_ms_vs_shmpc"] = results["effect_vs_shmpc"][m].get("solve_time_delta_ms")
        if "total_progress" in results["metrics"]:
            r["total_progress_mean"] = results["metrics"]["total_progress"][m]["mean"]
        if "avg_solve_ms" in results["metrics"]:
            r["avg_solve_ms"] = results["metrics"]["avg_solve_ms"][m]["mean"]
        rows.append(r)
    eff_df = pd.DataFrame(rows)
    eff_path = out_dir / "efficacy_vs_shmpc.csv"
    eff_df.to_csv(eff_path, index=False)
    print(f"Wrote {eff_path}")


def main():
    csv_path = sys.argv[1] if len(sys.argv) >= 2 else str(DEFAULT_CSV)
    df = load_data(csv_path)
    results = analyze_vs_shmpc(df)
    print_and_save(results, OUT_DIR)


if __name__ == "__main__":
    main()
