#!/usr/bin/env python3
"""
Smoke test for edge-case and tuning analysis: run analyzers and check outputs exist and have expected structure.
Run from repo root: python3 future_sl_mpc/experiments/test_edge_tuning_analysis.py
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS = SCRIPT_DIR / "results"


def test_analyze_vs_shmpc_handles_scenario():
    """analyze_vs_shmpc works with or without scenario column."""
    sys.path.insert(0, str(SCRIPT_DIR))
    from analyze_vs_shmpc import load_data, analyze_vs_shmpc
    # Without scenario: add baseline
    import pandas as pd
    df = pd.DataFrame({
        "method": ["SHMPC", "SHMPC_RTA"] * 5,
        "collision": [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        "total_progress": [10.0, 9.0] * 5,
        "total_steps": [80] * 10,
        "missed_mode_steps": [5] * 10,
        "avg_solve_ms": [4.0] * 10,
    })
    # Simulate no scenario column
    if "scenario" in df.columns:
        df = df.drop(columns=["scenario"])
    result = analyze_vs_shmpc(df)
    assert "SHMPC" in result["metrics"]["collision_rate"]
    assert "SHMPC_RTA" in result["effect_vs_shmpc"]
    print("  analyze_vs_shmpc (no scenario column): OK")


def test_edge_tuning_analysis_modules():
    """Edge/tuning analysis modules import and run on minimal data."""
    sys.path.insert(0, str(SCRIPT_DIR))
    from analyze_edge_and_tuning import (
        load_data, edge_case_analysis, tradeoff_analysis,
        pareto_methods, tuning_analysis, EDGE_SCENARIOS,
    )
    import pandas as pd
    # Minimal CSV with scenario column
    df = pd.DataFrame({
        "method": ["SHMPC", "SHMPC_RTA", "SHMPC"] * 4,
        "scenario": ["baseline"] * 6 + ["high_switch"] * 6,
        "collision": [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        "total_progress": [15.0] * 12,
        "total_steps": [80] * 12,
        "missed_mode_steps": [5] * 12,
        "avg_solve_ms": [4.0] * 12,
    })
    edge_df = edge_case_analysis(df)
    assert "scenario" in edge_df.columns and "method" in edge_df.columns
    assert "collision_rate" in edge_df.columns
    trade_df = tradeoff_analysis(df, "baseline")
    assert len(trade_df) >= 2
    pareto = pareto_methods(trade_df)
    assert isinstance(pareto, list)
    tune_df = tuning_analysis(df)
    # No tune_ scenario in this minimal df
    assert tune_df.empty or "tuning_type" in tune_df.columns
    print("  edge_case_analysis, tradeoff_analysis, tuning_analysis: OK")


def test_analyzer_on_real_csv_if_present():
    """If edge_tuning CSV exists, run full analysis and check output files."""
    repo_root = SCRIPT_DIR.parent.parent
    csv_candidates = [
        RESULTS / "future_sl_rollouts_edge_tuning.csv",
        repo_root / "cpp_mpc" / "future_sl_mpc" / "experiments" / "results" / "future_sl_rollouts_edge_tuning.csv",
    ]
    csv_path = None
    for p in csv_candidates:
        if p.exists():
            csv_path = p
            break
    if not csv_path:
        print("  (no edge_tuning CSV found; skip full pipeline test)")
        return
    sys.path.insert(0, str(SCRIPT_DIR))
    from analyze_edge_and_tuning import load_data, edge_case_analysis, tradeoff_analysis, tuning_analysis
    df = load_data(str(csv_path))
    edge_df = edge_case_analysis(df)
    trade_df = tradeoff_analysis(df, "baseline" if (df["scenario"] == "baseline").any() else df["scenario"].iloc[0])
    tune_df = tuning_analysis(df)
    assert not edge_df.empty, "edge_case_analysis should produce rows"
    assert "high_switch" in edge_df["scenario"].values or "baseline" in edge_df["scenario"].values
    print("  full pipeline on real CSV: OK (scenarios: %s)" % list(edge_df["scenario"].unique())[:5])


if __name__ == "__main__":
    print("Testing edge/tuning analysis...")
    test_analyze_vs_shmpc_handles_scenario()
    test_edge_tuning_analysis_modules()
    test_analyzer_on_real_csv_if_present()
    print("All checks passed.")
