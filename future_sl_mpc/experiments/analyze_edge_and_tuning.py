#!/usr/bin/env python3
"""
Analyze edge-case and tuning experiment results: by-scenario efficacy, trade-offs, tuning sensitivity.

Expects CSV with columns: method, scenario, collision, total_progress, missed_mode_steps,
total_steps, avg_solve_ms, min_clearance, etc. (e.g. future_sl_rollouts_edge_tuning.csv).

Usage:
  python3 analyze_edge_and_tuning.py [path/to/future_sl_rollouts_edge_tuning.csv]
  If no path given, uses results/future_sl_rollouts_edge_tuning.csv

Output:
  - results/edge_case_summary.csv, edge_case_table.txt
  - results/tradeoff_summary.csv, tradeoff_table.txt
  - results/tuning_sensitivity.csv, tuning_summary.txt
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "results" / "future_sl_rollouts_edge_tuning.csv"
OUT_DIR = SCRIPT_DIR / "results"

EDGE_SCENARIOS = ["baseline", "high_switch", "rare_heavy", "low_S", "distribution_shift"]


def load_data(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.is_absolute() and not path.exists():
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


def edge_case_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Collision rate and progress by (scenario, method). Efficacy vs SHMPC per scenario."""
    rows = []
    for scenario in df["scenario"].unique():
        sub = df[df["scenario"] == scenario]
        if "SHMPC" not in sub["method"].values:
            continue
        shmpc = sub[sub["method"] == "SHMPC"]
        n_base = len(shmpc)
        p_coll_base = shmpc["collision"].mean()
        prog_base = shmpc["total_progress"].mean() if "total_progress" in sub.columns else 0.0
        for method in sub["method"].unique():
            msub = sub[sub["method"] == method]
            n = len(msub)
            p_coll = msub["collision"].mean()
            k = int(msub["collision"].sum())
            ci_lo, ci_hi = wilson_ci(k, n)
            prog = msub["total_progress"].mean() if "total_progress" in msub.columns else 0.0
            delta_coll = p_coll_base - p_coll  # positive = method safer
            delta_prog = prog - prog_base
            rows.append({
                "scenario": scenario,
                "method": method,
                "n": n,
                "collision_rate": p_coll,
                "collision_ci_lo": ci_lo,
                "collision_ci_hi": ci_hi,
                "total_progress_mean": prog,
                "delta_collision_vs_shmpc": delta_coll,
                "delta_progress_vs_shmpc": delta_prog,
                "shmpc_collision_rate": p_coll_base,
            })
    return pd.DataFrame(rows)


def tradeoff_analysis(df: pd.DataFrame, scenario_filter: str = "baseline") -> pd.DataFrame:
    """Safety vs progress vs solve time by method (for one scenario)."""
    sub = df[df["scenario"] == scenario_filter]
    if sub.empty:
        sub = df  # fallback to all
    rows = []
    for method in sub["method"].unique():
        msub = sub[sub["method"] == method]
        rows.append({
            "method": method,
            "scenario": scenario_filter,
            "collision_rate": msub["collision"].mean(),
            "total_progress_mean": msub["total_progress"].mean() if "total_progress" in msub.columns else 0.0,
            "avg_solve_ms": msub["avg_solve_ms"].mean() if "avg_solve_ms" in msub.columns else 0.0,
            "missed_mode_rate": msub["missed_mode_steps"].sum() / msub["total_steps"].sum() if "total_steps" in msub.columns and msub["total_steps"].sum() else 0.0,
            "n": len(msub),
        })
    return pd.DataFrame(rows)


def pareto_methods(trade_df: pd.DataFrame) -> list:
    """Methods that are not strictly dominated on (lower collision, higher progress)."""
    dominated = set()
    methods = trade_df["method"].tolist()
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i == j:
                continue
            c1 = trade_df.iloc[i]["collision_rate"]
            c2 = trade_df.iloc[j]["collision_rate"]
            p1 = trade_df.iloc[i]["total_progress_mean"]
            p2 = trade_df.iloc[j]["total_progress_mean"]
            if c2 <= c1 and p2 >= p1 and (c2 < c1 or p2 > p1):
                dominated.add(m1)
                break
    return [m for m in methods if m not in dominated]


def tuning_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """For scenarios like tune_cert_10, tune_rta_15: collision rate vs tuning parameter."""
    tune = df[df["scenario"].str.startswith("tune_", na=False)]
    if tune.empty:
        return pd.DataFrame()
    rows = []
    for scenario in tune["scenario"].unique():
        sub = tune[tune["scenario"] == scenario]
        # Parse parameter from scenario name (tune_cert_10->0.10, tune_rta_15->1.5, tune_bandit_5->0.5)
        parts = scenario.split("_")
        if len(parts) >= 3:
            try:
                raw = int(parts[-1])
                param_val = raw / 100.0 if "cert" in scenario else raw / 10.0
            except ValueError:
                param_val = np.nan
        else:
            param_val = np.nan
        for method in sub["method"].unique():
            msub = sub[sub["method"] == method]
            rows.append({
                "scenario": scenario,
                "tuning_type": "_".join(parts[1:-1]) if len(parts) >= 3 else scenario,
                "parameter_value": param_val,
                "method": method,
                "n": len(msub),
                "collision_rate": msub["collision"].mean(),
                "total_progress_mean": msub["total_progress"].mean() if "total_progress" in msub.columns else 0.0,
                "avg_solve_ms": msub["avg_solve_ms"].mean() if "avg_solve_ms" in msub.columns else 0.0,
            })
    return pd.DataFrame(rows)


def main():
    csv_path = sys.argv[1] if len(sys.argv) >= 2 else str(DEFAULT_CSV)
    df = load_data(csv_path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Edge-case summary
    edge_df = edge_case_analysis(df)
    edge_path = OUT_DIR / "edge_case_summary.csv"
    edge_df.to_csv(edge_path, index=False)
    print(f"Wrote {edge_path}")

    edge_scenarios = [s for s in EDGE_SCENARIOS if s in edge_df["scenario"].values]
    lines = ["Edge-case: collision rate by (scenario, method)", "=" * 60]
    for sc in edge_scenarios:
        lines.append(f"\n--- {sc} ---")
        sub = edge_df[edge_df["scenario"] == sc]
        for _, row in sub.iterrows():
            lines.append(f"  {row['method']:25s}  coll={row['collision_rate']:.4f}  delta_vs_SHMPC={row['delta_collision_vs_shmpc']:+.4f}  progress={row['total_progress_mean']:.2f}")
    table_path = OUT_DIR / "edge_case_table.txt"
    with open(table_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {table_path}")

    # Trade-off (baseline scenario if present, else all)
    scenario_for_trade = "baseline" if (df["scenario"] == "baseline").any() else df["scenario"].iloc[0]
    trade_df = tradeoff_analysis(df, scenario_for_trade)
    trade_path = OUT_DIR / "tradeoff_summary.csv"
    trade_df.to_csv(trade_path, index=False)
    print(f"Wrote {trade_path}")

    pareto = pareto_methods(trade_df)
    trade_lines = [
        "Trade-off: safety vs progress vs solve time (scenario=%s)" % scenario_for_trade,
        "=" * 60,
        "Pareto-efficient methods (no other method is strictly better on both collision and progress):",
        "  " + ", ".join(pareto),
        "",
        "Collision rate | Progress | Solve ms",
    ]
    for _, row in trade_df.iterrows():
        trade_lines.append(f"  {row['method']:25s}  {row['collision_rate']:.4f}  {row['total_progress_mean']:.2f}  {row['avg_solve_ms']:.2f}")
    with open(OUT_DIR / "tradeoff_table.txt", "w") as f:
        f.write("\n".join(trade_lines))
    print(f"Wrote {OUT_DIR / 'tradeoff_table.txt'}")

    # Tuning sensitivity
    tune_df = tuning_analysis(df)
    if not tune_df.empty:
        tune_path = OUT_DIR / "tuning_sensitivity.csv"
        tune_df.to_csv(tune_path, index=False)
        print(f"Wrote {tune_path}")
        tune_lines = ["Tuning sensitivity: collision rate vs parameter", "=" * 60]
        for tune_type in tune_df["tuning_type"].unique():
            tune_lines.append(f"\n--- {tune_type} ---")
            sub = tune_df[tune_df["tuning_type"] == tune_type]
            for method in sub["method"].unique():
                msub = sub[sub["method"] == method].sort_values("parameter_value")
                for _, row in msub.iterrows():
                    tune_lines.append(f"  {row['method']:25s}  param={row['parameter_value']:.2f}  coll={row['collision_rate']:.4f}  progress={row['total_progress_mean']:.2f}")
        with open(OUT_DIR / "tuning_summary.txt", "w") as f:
            f.write("\n".join(tune_lines))
        print(f"Wrote {OUT_DIR / 'tuning_summary.txt'}")
    else:
        print("No tuning scenarios found in CSV; skipping tuning summary.")


if __name__ == "__main__":
    main()
