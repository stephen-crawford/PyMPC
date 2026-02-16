#!/usr/bin/env python3
"""
Generate publication figures from results-section CSV data.

Reads CSVs from paper_figures/ and writes PNGs to paper_figures/.

Figures:
  1. Collision probability vs switching severity  (Exp A)
  2. Mode-mismatch (missed-mode fraction) vs time (Exp A, sp=0.2)
  3. Safety-performance Pareto                    (Exp A, sp=0.2)
  4. Collision rate conditioned on rare mode       (Exp B)
  5. Solve-time distributions (box plots)          (Exp C)
  5b. Safety vs Runtime Pareto                     (Exp C)
  6. Ablation table as figure                      (Exp A)
  7. Calibration plot: predicted vs observed risk   (Exp D)
  8. Buffer size sensitivity                        (Exp E)
  9. Conservatism & control smoothness comparison   (Exp G)
 10. Forest plot: collision rate differences + bootstrap CIs (Exp H1)
 11. Missed-mode significance: bar chart w/ p-values        (Exp H2)
 12. Updated ablation table (1000 rollouts)                  (Exp H4)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

DATA_DIR = "paper_figures/"
OUT_DIR = "paper_figures/"

VARIANT_COLORS = {
    "Base": "#1f77b4",
    "DRO": "#ff7f0e",
    "OT": "#2ca02c",
    "OT+DRO": "#d62728",
}
VARIANT_MARKERS = {
    "Base": "o",
    "DRO": "s",
    "OT": "^",
    "OT+DRO": "D",
}
VARIANT_ORDER = ["Base", "DRO", "OT", "OT+DRO"]

EPS_TARGET = 0.05  # epsilon target for horizontal dashed line


def load_csv(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping.")
        return None
    return pd.read_csv(path)


# ============================================================================
# Figure 1: Collision Probability vs Switching Severity
# ============================================================================

def fig1_collision_vs_switching():
    df = load_csv("exp_a_collision_vs_switching.csv")
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    for vname in VARIANT_ORDER:
        sub = df[df["variant"] == vname].sort_values("switch_prob")
        if sub.empty:
            continue
        ax.errorbar(
            sub["switch_prob"], sub["collision_rate"],
            yerr=[sub["collision_rate"] - sub["ci_lo"],
                  sub["ci_hi"] - sub["collision_rate"]],
            label=vname, color=VARIANT_COLORS[vname],
            marker=VARIANT_MARKERS[vname], markersize=5,
            capsize=3, linewidth=1.5,
        )

    ax.axhline(EPS_TARGET, color="gray", linestyle="--", linewidth=1, label=r"$\varepsilon$ target")
    ax.set_xlabel("Mode-switch probability")
    ax.set_ylabel("Empirical collision rate")
    ax.set_title("Collision Rate vs. Switching Severity")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(bottom=-0.01)
    ax.grid(True, alpha=0.3)

    out = os.path.join(OUT_DIR, "fig1_collision_vs_switching.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 2: Mismatch vs Time
# ============================================================================

def fig2_mismatch_vs_time():
    df = load_csv("exp_a_w2_vs_time.csv")
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    for vname in VARIANT_ORDER:
        sub = df[df["variant"] == vname].sort_values("step")
        if sub.empty:
            continue
        # Smooth with rolling window
        vals = sub["missed_fraction"].values
        window = 5
        if len(vals) >= window:
            smoothed = np.convolve(vals, np.ones(window)/window, mode="valid")
            steps = sub["step"].values[:len(smoothed)]
        else:
            smoothed = vals
            steps = sub["step"].values
        ax.plot(steps * 0.1, smoothed, label=vname,
                color=VARIANT_COLORS[vname], linewidth=1.3)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Missed-mode fraction")
    ax.set_title("Distributional Mismatch vs. Time (switch prob = 0.2)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(bottom=-0.02, top=1.02)
    ax.grid(True, alpha=0.3)

    out = os.path.join(OUT_DIR, "fig2_mismatch_vs_time.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 3: Safety-Performance Pareto
# ============================================================================

def fig3_safety_performance_pareto():
    df = load_csv("exp_a_missed_mode_rate.csv")
    if df is None:
        return

    # Use switch_prob = 0.2 as representative
    df2 = df[np.isclose(df["switch_prob"], 0.2)]
    df_coll = load_csv("exp_a_collision_vs_switching.csv")
    if df_coll is None:
        return
    df_coll2 = df_coll[np.isclose(df_coll["switch_prob"], 0.2)]

    fig, ax = plt.subplots(figsize=(5.0, 3.8))

    for vname in VARIANT_ORDER:
        sub_miss = df2[df2["variant"] == vname]
        sub_coll = df_coll2[df_coll2["variant"] == vname]
        if sub_miss.empty or sub_coll.empty:
            continue
        progress = sub_miss["avg_progress"].values[0]
        coll_rate = sub_coll["collision_rate"].values[0]
        ci_lo = sub_coll["ci_lo"].values[0]
        ci_hi = sub_coll["ci_hi"].values[0]

        ax.errorbar(
            progress, coll_rate,
            yerr=[[coll_rate - ci_lo], [ci_hi - coll_rate]],
            marker=VARIANT_MARKERS[vname], color=VARIANT_COLORS[vname],
            markersize=10, capsize=4, linewidth=1.5,
            label=vname,
        )

    ax.axhline(EPS_TARGET, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Average progress [m]")
    ax.set_ylabel("Collision rate")
    ax.set_title("Safety vs. Performance (switch prob = 0.2)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    out = os.path.join(OUT_DIR, "fig3_safety_performance_pareto.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 4: Collision Rate Conditioned on Rare Mode
# ============================================================================

def fig4_rare_mode_collision():
    df = load_csv("exp_b_collision_given_rare.csv")
    if df is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    # Left: overall collision rate vs rare_prob
    ax = axes[0]
    for vname in VARIANT_ORDER:
        sub = df[df["variant"] == vname].sort_values("rare_prob")
        if sub.empty:
            continue
        ax.errorbar(
            sub["rare_prob"], sub["collision_rate"],
            yerr=[sub["collision_rate"] - sub["ci_lo"],
                  sub["ci_hi"] - sub["collision_rate"]],
            label=vname, color=VARIANT_COLORS[vname],
            marker=VARIANT_MARKERS[vname], markersize=5,
            capsize=3, linewidth=1.5,
        )
    ax.axhline(EPS_TARGET, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Rare-mode probability")
    ax.set_ylabel("Collision rate")
    ax.set_title("Overall Collision Rate")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Right: collision rate conditioned on rare mode occurring
    ax = axes[1]
    for vname in VARIANT_ORDER:
        sub = df[df["variant"] == vname].sort_values("rare_prob")
        if sub.empty:
            continue
        ax.plot(
            sub["rare_prob"], sub["collision_given_rare"],
            label=vname, color=VARIANT_COLORS[vname],
            marker=VARIANT_MARKERS[vname], markersize=5,
            linewidth=1.5,
        )
    ax.set_xlabel("Rare-mode probability")
    ax.set_ylabel("P(collision | rare mode)")
    ax.set_title("Collision Rate Given Rare Mode")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig4_rare_mode_collision.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 5: Solve-Time Distributions
# ============================================================================

def fig5_solve_time_distributions():
    df = load_csv("exp_c_solve_times.csv")
    if df is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    # Left: median solve time vs scenario count
    ax = axes[0]
    for vname in VARIANT_ORDER:
        sub = df[df["variant"] == vname].sort_values("num_scenarios")
        if sub.empty:
            continue
        ax.plot(sub["num_scenarios"], sub["median_ms"],
                label=vname, color=VARIANT_COLORS[vname],
                marker=VARIANT_MARKERS[vname], markersize=5, linewidth=1.5)
    ax.set_xlabel("Number of scenarios S")
    ax.set_ylabel("Median solve time [ms]")
    ax.set_title("Median Solve Time")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Right: p99 solve time vs scenario count
    ax = axes[1]
    for vname in VARIANT_ORDER:
        sub = df[df["variant"] == vname].sort_values("num_scenarios")
        if sub.empty:
            continue
        ax.plot(sub["num_scenarios"], sub["p99_ms"],
                label=vname, color=VARIANT_COLORS[vname],
                marker=VARIANT_MARKERS[vname], markersize=5, linewidth=1.5)
    ax.set_xlabel("Number of scenarios S")
    ax.set_ylabel("p99 solve time [ms]")
    ax.set_title("Tail Solve Time (p99)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig5_solve_time_distributions.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")

    # Also: safety vs runtime Pareto
    df_safety = load_csv("exp_c_safety_vs_runtime.csv")
    if df_safety is None:
        return

    fig2, ax2 = plt.subplots(figsize=(5.5, 3.8))
    for vname in VARIANT_ORDER:
        sub = df_safety[df_safety["variant"] == vname].sort_values("num_scenarios")
        if sub.empty:
            continue
        ax2.plot(sub["avg_solve_ms"], sub["collision_rate"],
                 label=vname, color=VARIANT_COLORS[vname],
                 marker=VARIANT_MARKERS[vname], markersize=7, linewidth=1.5)
        # Annotate with S values
        for _, row in sub.iterrows():
            ax2.annotate(f"S={int(row['num_scenarios'])}",
                        (row["avg_solve_ms"], row["collision_rate"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax2.axhline(EPS_TARGET, color="gray", linestyle="--", linewidth=1)
    ax2.set_xlabel("Average solve time [ms]")
    ax2.set_ylabel("Collision rate")
    ax2.set_title("Safety vs. Runtime Tradeoff")
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    out2 = os.path.join(OUT_DIR, "fig5b_safety_vs_runtime.png")
    fig2.savefig(out2)
    plt.close(fig2)
    print(f"  Saved {out2}")


# ============================================================================
# Figure 6: Ablation Table
# ============================================================================

def fig6_ablation_table():
    df = load_csv("exp_a_ablation_table.csv")
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(7.5, 2.2))
    ax.axis("off")

    columns = ["Variant", "OT?", "DRO?", "Collision Rate", "95% CI",
               "Missed Mode", "Avg Progress", "Avg Clearance", "Solve [ms]"]

    cell_text = []
    cell_colors = []
    for _, row in df.iterrows():
        ci = f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]"
        cells = [
            row["variant"],
            row["uses_ot"],
            row["uses_dro"],
            f"{row['collision_rate']:.3f}",
            ci,
            f"{row['missed_mode_rate']:.3f}",
            f"{row['avg_progress']:.1f}",
            f"{row['avg_clearance']:.2f}",
            f"{row['avg_solve_ms']:.1f}",
        ]
        cell_text.append(cells)
        # Color-code: green if collision_rate < eps_target
        if row["collision_rate"] < EPS_TARGET:
            cell_colors.append(["#e8f5e9"] * len(columns))
        else:
            cell_colors.append(["#ffebee"] * len(columns))

    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        cellColours=cell_colors if cell_colors else None,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)

    # Bold header
    for j in range(len(columns)):
        table[0, j].set_text_props(fontweight="bold")
        table[0, j].set_facecolor("#e0e0e0")

    ax.set_title("Ablation Study (switch prob = 0.2)", fontsize=11, pad=15)

    out = os.path.join(OUT_DIR, "fig6_ablation_table.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 7: Calibration Plot (Predicted Risk vs Observed Collision Rate)
# ============================================================================

def fig7_calibration_plot():
    df = load_csv("exp_d_calibration.csv")
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(5.0, 4.5))

    # Perfect calibration line
    ax.plot([0, 0.55], [0, 0.55], "k--", linewidth=1, alpha=0.5, label="Perfect calibration")

    for vname in VARIANT_ORDER:
        sub = df[df["variant"] == vname].sort_values("predicted_risk")
        if sub.empty:
            continue
        ax.errorbar(
            sub["predicted_risk"], sub["observed_collision_rate"],
            yerr=[sub["observed_collision_rate"] - sub["ci_lo"],
                  sub["ci_hi"] - sub["observed_collision_rate"]],
            label=vname, color=VARIANT_COLORS[vname],
            marker=VARIANT_MARKERS[vname], markersize=5,
            capsize=3, linewidth=1.5,
        )

    ax.set_xlabel(r"Predicted risk ($\varepsilon$ target)")
    ax.set_ylabel("Observed collision rate")
    ax.set_title("Calibration: Predicted vs. Observed Risk")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_xlim(-0.01, 0.55)
    ax.set_ylim(-0.01, 0.55)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    out = os.path.join(OUT_DIR, "fig7_calibration_plot.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 8: Buffer Size Sensitivity
# ============================================================================

def fig8_buffer_sensitivity():
    df = load_csv("exp_e_buffer_sensitivity.csv")
    if df is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # Left: collision rate vs buffer size
    ax = axes[0]
    ax.errorbar(df["buffer_size"], df["collision_rate"],
                yerr=[df["collision_rate"] - df["ci_lo"],
                      df["ci_hi"] - df["collision_rate"]],
                color="#d62728", marker="D", markersize=6, capsize=3, linewidth=1.5)
    ax.axhline(EPS_TARGET, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("History buffer size")
    ax.set_ylabel("Collision rate")
    ax.set_title("Safety vs. Buffer Size")
    ax.grid(True, alpha=0.3)

    # Middle: missed mode rate vs buffer size
    ax = axes[1]
    ax.plot(df["buffer_size"], df["missed_mode_rate"],
            color="#2ca02c", marker="^", markersize=6, linewidth=1.5)
    ax.set_xlabel("History buffer size")
    ax.set_ylabel("Missed-mode rate")
    ax.set_title("Mode Coverage vs. Buffer Size")
    ax.grid(True, alpha=0.3)

    # Right: clearance vs buffer size
    ax = axes[2]
    ax.plot(df["buffer_size"], df["avg_clearance"],
            color="#1f77b4", marker="o", markersize=6, linewidth=1.5)
    ax.set_xlabel("History buffer size")
    ax.set_ylabel("Average min clearance [m]")
    ax.set_title("Conservatism vs. Buffer Size")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig8_buffer_sensitivity.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 9: Conservatism & Control Smoothness
# ============================================================================

def fig9_conservatism_smoothness():
    df = load_csv("exp_g_conservatism_metrics.csv")
    if df is None:
        return

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    variants = df["variant"].tolist()
    x = np.arange(len(variants))
    width = 0.6
    colors = [VARIANT_COLORS.get(v, "#999999") for v in variants]

    # Average speed
    ax = axes[0]
    ax.bar(x, df["avg_speed"], width, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Avg speed [m/s]")
    ax.set_title("Average Speed")
    ax.grid(True, alpha=0.3, axis="y")

    # Min clearance
    ax = axes[1]
    ax.bar(x, df["min_clearance_mean"], width, color=colors,
           yerr=df["min_clearance_std"], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Min clearance [m]")
    ax.set_title("Minimum Clearance")
    ax.grid(True, alpha=0.3, axis="y")

    # Control effort
    ax = axes[2]
    ax.bar(x, df["control_effort_mean"], width, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Control effort")
    ax.set_title("Control Effort")
    ax.grid(True, alpha=0.3, axis="y")

    # Steering variation (smoothness)
    ax = axes[3]
    ax.bar(x, df["steering_variation"], width, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Steering variation")
    ax.set_title("Control Smoothness")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig9_conservatism_smoothness.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 10: Forest Plot — Collision Rate Differences with Bootstrap CIs
# ============================================================================

def fig10_forest_plot():
    df = load_csv("exp_h1_bootstrap_ci.csv")
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(6.0, 3.5))

    comparisons = df["comparison"].tolist()
    means = df["mean_diff"].values
    ci_lo = df["ci_lo"].values
    ci_hi = df["ci_hi"].values

    # Clean up labels
    labels = [c.replace("_vs_", " vs ").replace("_", " ") for c in comparisons]

    y_pos = np.arange(len(labels))
    colors = ["#d62728", "#2ca02c", "#ff7f0e"]  # match variant colors

    for i in range(len(labels)):
        ax.errorbar(
            means[i], y_pos[i],
            xerr=[[means[i] - ci_lo[i]], [ci_hi[i] - means[i]]],
            fmt="o", color=colors[i % len(colors)],
            markersize=8, capsize=5, linewidth=2, capthick=1.5,
        )

    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Collision rate difference (Base $-$ variant)")
    ax.set_title("Collision Rate Differences with 95% Bootstrap CIs")
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    # Annotate with values
    for i in range(len(labels)):
        ax.annotate(
            f"{means[i]:.3f} [{ci_lo[i]:.3f}, {ci_hi[i]:.3f}]",
            (ci_hi[i], y_pos[i]),
            textcoords="offset points", xytext=(8, 0),
            fontsize=8, va="center",
        )

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig10_forest_plot.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 11: Missed-Mode Significance — Bar Chart with p-values
# ============================================================================

def fig11_missed_mode_significance():
    df = load_csv("exp_h2_missed_mode_significance.csv")
    if df is None:
        return

    switch_probs = sorted(df["switch_prob"].unique())

    fig, axes = plt.subplots(1, len(switch_probs), figsize=(5 * len(switch_probs), 4.5))
    if len(switch_probs) == 1:
        axes = [axes]

    for ax, sp in zip(axes, switch_probs):
        sub = df[np.isclose(df["switch_prob"], sp)]

        variants = []
        rates = []
        colors = []
        for vname in VARIANT_ORDER:
            row = sub[sub["variant"] == vname]
            if row.empty:
                continue
            variants.append(vname)
            rates.append(row["missed_mode_rate"].values[0])
            colors.append(VARIANT_COLORS.get(vname, "#999999"))

        x = np.arange(len(variants))
        bars = ax.bar(x, rates, 0.6, color=colors)

        # Annotate p-values for non-Base variants
        for i, vname in enumerate(variants):
            if vname == "Base":
                continue
            row = sub[sub["variant"] == vname]
            if row.empty:
                continue
            p_z = row["p_vs_base"].values[0]
            h = row["cohens_h_vs_base"].values[0]
            perm_p = row["permutation_p_vs_base"].values[0]

            # Star annotation
            if p_z < 0.001:
                stars = "***"
            elif p_z < 0.01:
                stars = "**"
            elif p_z < 0.05:
                stars = "*"
            else:
                stars = "n.s."

            ax.annotate(
                f"p={p_z:.3f} {stars}\nh={h:.2f}",
                (x[i], rates[i]),
                textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=7,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Missed-mode rate")
        ax.set_title(f"switch prob = {sp}")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Missed-Mode Rates with Statistical Significance vs Base", fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig11_missed_mode_significance.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 12: Updated Ablation Table (1000 rollouts, tighter CIs)
# ============================================================================

def fig12_ablation_table_1000():
    df = load_csv("exp_h4_ablation_table_1000.csv")
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(7.5, 2.4))
    ax.axis("off")

    columns = ["Variant", "OT?", "DRO?", "Collision Rate", "95% CI",
               "Missed Mode", "Avg Progress", "Avg Clearance", "Solve [ms]"]

    # Handle both old ("uses_extremes") and new ("uses_dro") column names
    dro_col = "uses_dro" if "uses_dro" in df.columns else "uses_extremes"

    cell_text = []
    cell_colors = []
    for _, row in df.iterrows():
        ci = f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]"
        cells = [
            row["variant"],
            row["uses_ot"],
            row[dro_col],
            f"{row['collision_rate']:.4f}",
            ci,
            f"{row['missed_mode_rate']:.4f}",
            f"{row['avg_progress']:.1f}",
            f"{row['avg_clearance']:.2f}",
            f"{row['avg_solve_ms']:.1f}",
        ]
        cell_text.append(cells)
        if row["collision_rate"] < EPS_TARGET:
            cell_colors.append(["#e8f5e9"] * len(columns))
        else:
            cell_colors.append(["#ffebee"] * len(columns))

    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        cellColours=cell_colors if cell_colors else None,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)

    for j in range(len(columns)):
        table[0, j].set_text_props(fontweight="bold")
        table[0, j].set_facecolor("#e0e0e0")

    ax.set_title("Ablation Study — 1000 Rollouts (switch prob = 0.2)", fontsize=11, pad=15)

    out = os.path.join(OUT_DIR, "fig12_ablation_table_1000.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Main
# ============================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Generating figures from CSV data...\n")

    fig1_collision_vs_switching()
    fig2_mismatch_vs_time()
    fig3_safety_performance_pareto()
    fig4_rare_mode_collision()
    fig5_solve_time_distributions()
    fig6_ablation_table()
    fig7_calibration_plot()
    fig8_buffer_sensitivity()
    fig9_conservatism_smoothness()
    fig10_forest_plot()
    fig11_missed_mode_significance()
    fig12_ablation_table_1000()

    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
