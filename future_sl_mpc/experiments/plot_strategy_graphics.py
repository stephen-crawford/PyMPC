#!/usr/bin/env python3
"""
Generate graphics for Future SL MPC: collision avoidance strategy types, efficacy comparison,
and conceptual flow diagrams.

Outputs (to future_sl_mpc/experiments/results/ or --out):
  - strategy_taxonomy.png        Strategy types and method grouping
  - strategy_collision_bars.png  Collision rate by method with 95% CI
  - strategy_delta_vs_shmpc.png Δ collision rate vs SHMPC (horizontal bar)
  - strategy_tradeoff_scatter.png Safety vs progress (bubble size = solve time)
  - strategy_control_flow.png    Where each strategy type intervenes in the loop
  - strategy_type_diagrams.png  Per-type conceptual diagrams (DRO, Allocation, Certificate, Compiler, RTA)

Usage:
  python3 plot_strategy_graphics.py [path/to/efficacy_vs_shmpc.csv] [--out dir]
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.collections import PatchCollection

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_CSV = RESULTS_DIR / "efficacy_vs_shmpc.csv"

# Strategy type → methods (extension + paradigm-shift)
STRATEGY_GROUPS = {
    "Baseline": ["SHMPC"],
    "DRO / Robustness": ["SHMPC_DRO", "SHMPC_AdaptiveDRO"],
    "Scenario allocation": ["SHMPC_Conformal", "SHMPC_Hazard", "SHMPC_Bandit"],
    "Certificate / tube": ["SHMPC_Certificate", "CertificateFirst"],
    "Compiler / witness": ["SHMPC_Compiler", "ScenarioCompiler"],
    "Runtime assurance": ["SHMPC_RTA"],
}
TYPE_COLORS = {
    "Baseline": "#2c3e50",
    "DRO / Robustness": "#e74c3c",
    "Scenario allocation": "#3498db",
    "Certificate / tube": "#9b59b6",
    "Compiler / witness": "#27ae60",
    "Runtime assurance": "#f39c12",
}
METHOD_DISPLAY = {
    "SHMPC": "SHMPC",
    "SHMPC_DRO": "DRO",
    "SHMPC_AdaptiveDRO": "Adaptive DRO",
    "SHMPC_RTA": "RTA",
    "SHMPC_Conformal": "Conformal",
    "SHMPC_Hazard": "Hazard",
    "SHMPC_Bandit": "Bandit",
    "SHMPC_Certificate": "Certificate",
    "SHMPC_Compiler": "Compiler",
    "CertificateFirst": "Cert.First",
    "ScenarioCompiler": "Scenario Comp.",
}


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def get_method_type(method: str) -> str:
    for typ, methods in STRATEGY_GROUPS.items():
        if method in methods:
            return typ
    return "Other"


def fig_taxonomy(out_dir: Path):
    """Strategy taxonomy: grouped boxes by type with method names."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    group_order = list(STRATEGY_GROUPS.keys())
    width = 1.85
    gap = 0.4
    x0 = 0.5
    for i, gname in enumerate(group_order):
        methods = STRATEGY_GROUPS[gname]
        n_m = len(methods)
        box_h = 0.5 + n_m * 0.38
        x = x0 + i * (width + gap)
        y = 0.5
        color = TYPE_COLORS.get(gname, "#95a5a6")
        box = FancyBboxPatch((x, y), width, box_h, boxstyle="round,pad=0.04,rounding_size=0.1",
                             facecolor=color, edgecolor="black", alpha=0.85, linewidth=1.2)
        ax.add_patch(box)
        ax.text(x + width / 2, y + box_h - 0.25, gname, ha="center", va="top", fontsize=10, fontweight="bold", wrap=True)
        for j, m in enumerate(methods):
            label = METHOD_DISPLAY.get(m, m)
            ax.text(x + width / 2, y + box_h - 0.55 - j * 0.38, label, ha="center", va="top", fontsize=8)
    ax.set_title("Future SL MPC: Collision Avoidance Strategy Types", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "strategy_taxonomy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / 'strategy_taxonomy.png'}")


def fig_collision_bars(df: pd.DataFrame, out_dir: Path):
    """Bar chart: collision rate by method with 95% CI, colored by strategy type."""
    if df is None or df.empty:
        print("  Skip collision bars (no data)")
        return
    df = df.copy()
    df["method_type"] = df["method"].map(get_method_type)
    df["display"] = df["method"].map(lambda m: METHOD_DISPLAY.get(m, m))
    # Order: baseline first, then by type order
    type_order = list(STRATEGY_GROUPS.keys())
    order = []
    for t in type_order:
        order.extend(STRATEGY_GROUPS[t])
    df["order"] = df["method"].map(lambda m: order.index(m) if m in order else 99)
    df = df.sort_values("order")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    bars = ax.bar(x, df["collision_rate"], color=[TYPE_COLORS.get(t, "#95a5a6") for t in df["method_type"]],
                  edgecolor="black", linewidth=0.8)
    ax.errorbar(x, df["collision_rate"],
                yerr=[df["collision_rate"] - df["collision_ci_lo"], df["collision_ci_hi"] - df["collision_rate"]],
                fmt="none", color="black", capsize=3, capthick=1)
    ax.axhline(df[df["method"] == "SHMPC"]["collision_rate"].values[0], color="gray", linestyle="--", linewidth=1, alpha=0.8, label="SHMPC baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(df["display"], rotation=45, ha="right")
    ax.set_ylabel("Collision rate")
    ax.set_xlabel("Method")
    ax.set_title("Collision Rate by Strategy (100 rollouts, 95% Wilson CI)")
    ax.legend(loc="upper right")
    ax.set_ylim(0, min(1.0, df["collision_ci_hi"].max() * 1.15))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "strategy_collision_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / 'strategy_collision_bars.png'}")


def fig_tradeoff_scatter(df: pd.DataFrame, out_dir: Path):
    """Scatter: safety (1 - collision_rate) vs progress; point size = solve time."""
    if df is None or df.empty:
        print("  Skip tradeoff scatter (no data)")
        return
    df = df.copy()
    df["safety"] = 1 - df["collision_rate"]
    df["method_type"] = df["method"].map(get_method_type)
    df["display"] = df["method"].map(lambda m: METHOD_DISPLAY.get(m, m))

    fig, ax = plt.subplots(figsize=(9, 7))
    for gname in STRATEGY_GROUPS:
        sub = df[df["method_type"] == gname]
        if sub.empty:
            continue
        color = TYPE_COLORS.get(gname, "#95a5a6")
        for _, row in sub.iterrows():
            sz = 80 + row["avg_solve_ms"] * 8
            ax.scatter(row["safety"], row["total_progress_mean"], s=sz, c=color, edgecolor="black",
                       linewidth=0.8, label=gname if (sub.index[0] == row.name) else "")
        for _, row in sub.iterrows():
            ax.annotate(row["display"], xy=(row["safety"], row["total_progress_mean"]),
                        xytext=(6, 6), textcoords="offset points", fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"))
    # Fix duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="lower right")
    ax.set_xlabel("Safety (1 − collision rate)")
    ax.set_ylabel("Progress (mean)")
    ax.set_title("Trade-off: Safety vs Progress (bubble size ∝ solve time)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "strategy_tradeoff_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / 'strategy_tradeoff_scatter.png'}")


def fig_delta_vs_shmpc(df: pd.DataFrame, out_dir: Path):
    """Horizontal bar: collision rate delta vs SHMPC (positive = safer)."""
    if df is None or df.empty or "collision_rate_delta_vs_shmpc" not in df.columns:
        return
    sub = df[df["method"] != "SHMPC"].copy()
    if sub.empty:
        return
    sub["display"] = sub["method"].map(lambda m: METHOD_DISPLAY.get(m, m))
    sub["method_type"] = sub["method"].map(get_method_type)
    sub = sub.sort_values("collision_rate_delta_vs_shmpc", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(sub))
    colors = [TYPE_COLORS.get(t, "#95a5a6") for t in sub["method_type"]]
    bars = ax.barh(y, sub["collision_rate_delta_vs_shmpc"], color=colors, edgecolor="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["display"], fontsize=9)
    ax.set_xlabel("SHMPC collision rate − method collision rate (bar right = safer, bar left = worse)")
    ax.set_title("Effect vs SHMPC: Reduction in Collision Rate")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "strategy_delta_vs_shmpc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / 'strategy_delta_vs_shmpc.png'}")


def fig_control_flow(out_dir: Path):
    """Single diagram: control loop with intervention points for each strategy type."""
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    # Main pipeline boxes (y=1.8, h=1.3 for more room)
    boxes = [
        (0.6, 1.8, 1.6, 1.3, "Observe\nstate", "#ecf0f1"),
        (2.5, 1.8, 1.6, 1.3, "Predict\nmodes", "#bdc3c7"),
        (4.4, 1.8, 2.0, 1.3, "Scenarios\n/ weights", "#3498db"),
        (6.7, 1.8, 2.0, 1.3, "MPC\nSolve", "#2ecc71"),
        (9.0, 1.8, 1.6, 1.3, "Apply\nu", "#1abc9c"),
    ]
    for x, y, w, h, text, col in boxes:
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", facecolor=col, edgecolor="black", linewidth=1)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    # Arrows between main boxes
    for i in range(len(boxes) - 1):
        x1, w1 = boxes[i][0], boxes[i][2]
        x2 = boxes[i + 1][0]
        ax.annotate("", xy=(x2, 2.45), xytext=(x1 + w1, 2.45),
                    arrowprops=dict(arrowstyle="->", lw=2, color="black"))

    # Intervention labels above pipeline (more vertical space, smaller font)
    interventions = [
        (2.5, 3.8, "DRO: ρ, ε"),
        (4.4, 3.8, "Allocation:\nConformal, Hazard, Bandit"),
        (6.7, 3.8, "Certificate:\ntube radii\nCompiler: witness set"),
        (9.0, 3.8, "RTA:\nmonitor + fallback"),
    ]
    for xx, yy, text in interventions:
        ax.text(xx, yy, text, ha="center", va="bottom", fontsize=7, style="italic",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="wheat", alpha=0.85))
        ax.plot([xx, xx], [yy - 0.2, 3.1], "k--", lw=0.7, alpha=0.5)

    ax.text(7, 0.6, "Strategies modify weights, constraints, or override action", ha="center", fontsize=9)
    ax.set_title("Control Loop: Where Each Strategy Type Intervenes", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "strategy_control_flow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / 'strategy_control_flow.png'}")


def fig_type_diagrams(out_dir: Path):
    """Five small conceptual diagrams: DRO, Allocation, Certificate, Compiler, RTA."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    axes[-1].axis("off")

    def draw_box(ax, x, y, w, h, text, color="#bdc3c7"):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", facecolor=color, edgecolor="black", linewidth=1)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7, wrap=True)

    diagrams = [
        ("DRO / Robustness", [
            ("Nominal\nscenarios", 0.2, 0.3, 0.35, 0.4),
            ("Wasserstein\nball ρ", 0.6, 0.35, 0.35, 0.35),
            ("Worst-case\nweights", 0.2, 0.75, 0.35, 0.2),
            ("MPC", 0.6, 0.75, 0.35, 0.2),
        ], ["#e74c3c", "#e74c3c", "#c0392b", "#27ae60"]),
        ("Scenario allocation", [
            ("Modes\nπ", 0.15, 0.25, 0.3, 0.5),
            ("Scores\n(conf/hazard/UCB)", 0.5, 0.35, 0.4, 0.35),
            ("Weights\nS_m", 0.15, 0.8, 0.35, 0.15),
            ("Sample\nscenarios", 0.55, 0.8, 0.35, 0.15),
        ], ["#3498db", "#3498db", "#2980b9", "#2ecc71"]),
        ("Certificate / tube", [
            ("Prediction\nμ", 0.2, 0.4, 0.35, 0.35),
            ("Tube\nradius r_t", 0.6, 0.4, 0.35, 0.35),
            ("Tighten\nconstraints", 0.4, 0.8, 0.35, 0.2),
        ], ["#9b59b6", "#9b59b6", "#8e44ad"]),
        ("Compiler / witness", [
            ("Initial\nS scenarios", 0.2, 0.5, 0.35, 0.3),
            ("Solve\nMPC", 0.6, 0.55, 0.3, 0.25),
            ("Adversary:\nviolating mode?", 0.2, 0.85, 0.5, 0.15),
            ("Add scenario\nre-solve", 0.55, 0.85, 0.35, 0.15),
        ], ["#27ae60", "#2ecc71", "#f39c12", "#27ae60"]),
        ("Runtime assurance", [
            ("MPC\naction u", 0.2, 0.4, 0.35, 0.4),
            ("Monitor\nM(state)", 0.6, 0.5, 0.35, 0.3),
            ("Safe?\n→ u : u_fallback", 0.4, 0.85, 0.35, 0.15),
        ], ["#1abc9c", "#f39c12", "#e67e22"]),
    ]

    for idx, (title, boxes, colors) in enumerate(diagrams):
        ax = axes[idx]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")
        for (text, x, y, w, h), c in zip(boxes, colors + ["#95a5a6"] * (len(boxes) - len(colors))):
            draw_box(ax, x, y, w, h, text, c)
        ax.set_title(title, fontsize=9, fontweight="bold")

    fig.suptitle("Conceptual Diagrams: How Each Strategy Type Works", fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "strategy_type_diagrams.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / 'strategy_type_diagrams.png'}")


def main():
    parser = argparse.ArgumentParser(description="Generate Future SL MPC strategy graphics")
    parser.add_argument("csv", nargs="?", default=str(DEFAULT_CSV), help="efficacy_vs_shmpc.csv path")
    parser.add_argument("--out", default=str(RESULTS_DIR), help="Output directory")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv)
    if not csv_path.is_absolute() and not csv_path.exists():
        csv_path = SCRIPT_DIR / args.csv
    df = load_data(csv_path)

    print("Generating strategy graphics...")
    fig_taxonomy(out_dir)
    fig_control_flow(out_dir)
    fig_type_diagrams(out_dir)
    if df is not None and not df.empty:
        fig_collision_bars(df, out_dir)
        fig_tradeoff_scatter(df, out_dir)
        fig_delta_vs_shmpc(df, out_dir)
    else:
        print("  No efficacy CSV found; skipping bar and scatter figures.")
    print("Done.")


if __name__ == "__main__":
    main()
