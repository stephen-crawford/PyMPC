#!/usr/bin/env python3
"""
Generate publication-quality figures from SH-MPC + DRO experiment CSVs.

Reads from shmpc_dro_paper_figures/ and writes PNG+PDF to the same directory.
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

INPUT_DIR = "build/shmpc_dro_paper_figures"
OUT_DIR = "shmpc_dro_paper_figures"

# Publication style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'base': '#1f77b4',
    'dro': '#d62728',
    'sh': '#2ca02c',
    'multi': '#ff7f0e',
    'full': '#9467bd',
}


def save_fig(fig, name):
    """Save figure as both PNG and PDF."""
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"))
    fig.savefig(os.path.join(OUT_DIR, f"{name}.pdf"))
    print(f"  Saved {name}.png/.pdf")
    plt.close(fig)


def fig_h1_mode_coverage():
    """H1: DRO mode coverage vs Base at different scenario counts."""
    path = os.path.join(INPUT_DIR, "exp_h1_mode_coverage.csv")
    if not os.path.exists(path):
        print("  Skipping H1: CSV not found")
        return
    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(6, 4))

    for dro_val, label, color, marker in [
        ("no", "Base (no DRO)", COLORS['base'], 'o'),
        ("yes", "DRO", COLORS['dro'], 's'),
    ]:
        subset = df[df['dro_enabled'] == dro_val]
        ax.plot(subset['num_scenarios'], subset['missed_mode_fraction'],
                marker=marker, color=color, label=label, linewidth=2, markersize=7)

    ax.set_xlabel('Number of Scenarios $S$')
    ax.set_ylabel('Missed Mode Fraction')
    ax.set_title('H1: Mode Coverage — DRO vs Base')
    ax.legend()
    ax.set_ylim(bottom=0)
    save_fig(fig, 'fig_h1_mode_coverage')


def fig_h2_collision_reduction():
    """H2: Collision rate reduction with DRO."""
    path = os.path.join(INPUT_DIR, "exp_h2_collision_reduction.csv")
    if not os.path.exists(path):
        print("  Skipping H2: CSV not found")
        return
    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(6, 4))

    for variant, label, color, marker in [
        ("Base", "Base", COLORS['base'], 'o'),
        ("DRO", "DRO", COLORS['dro'], 's'),
    ]:
        subset = df[df['variant'] == variant]
        ax.errorbar(subset['num_scenarios'], subset['collision_rate'],
                     yerr=[subset['collision_rate'] - subset['ci_lo'],
                           subset['ci_hi'] - subset['collision_rate']],
                     marker=marker, color=color, label=label, linewidth=2,
                     markersize=7, capsize=4)

    ax.set_xlabel('Number of Scenarios $S$')
    ax.set_ylabel('Collision Rate')
    ax.set_title('H2: Collision Rate — DRO vs Base (paired MC)')
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    save_fig(fig, 'fig_h2_collision_reduction')


def fig_h2_significance():
    """H2: McNemar significance table."""
    path = os.path.join(INPUT_DIR, "exp_h2_missed_mode_significance.csv")
    if not os.path.exists(path):
        print("  Skipping H2 significance: CSV not found")
        return
    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.axis('off')
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)
    ax.set_title('H2: McNemar Paired Test — Base vs DRO', pad=20)
    save_fig(fig, 'fig_h2_mcnemar_table')


def fig_h3_safe_horizon():
    """H3: Safe horizon truncation — collision rate and solve time tradeoff."""
    path = os.path.join(INPUT_DIR, "exp_h3_safe_horizon.csv")
    if not os.path.exists(path):
        print("  Skipping H3: CSV not found")
        return
    df = pd.read_csv(path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for sh, label, color in [
        ("no", "Full Horizon", COLORS['base']),
        ("yes", "Safe Horizon (SH-MPC)", COLORS['sh']),
    ]:
        subset = df[df['safe_horizon_enabled'] == sh]
        ax1.errorbar(subset['num_scenarios'], subset['collision_rate'],
                      yerr=[subset['collision_rate'] - subset['ci_lo'],
                            subset['ci_hi'] - subset['collision_rate']],
                      marker='o', color=color, label=label, linewidth=2, capsize=4)
        ax2.plot(subset['num_scenarios'], subset['avg_solve_ms'],
                  marker='o', color=color, label=label, linewidth=2)

    ax1.set_xlabel('Number of Scenarios $S$')
    ax1.set_ylabel('Collision Rate')
    ax1.set_title('Collision Rate')
    ax1.legend()
    ax1.set_ylim(bottom=0)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax2.set_xlabel('Number of Scenarios $S$')
    ax2.set_ylabel('Avg Solve Time [ms]')
    ax2.set_title('Computation Cost')
    ax2.legend()

    fig.suptitle('H3: Safe Horizon Truncation — Compute/Safety Tradeoff', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig_h3_safe_horizon')


def fig_h3_theoretical():
    """H3: Theoretical safe horizon as function of S."""
    path = os.path.join(INPUT_DIR, "exp_h3_theoretical_safe_horizon.csv")
    if not os.path.exists(path):
        print("  Skipping H3 theoretical: CSV not found")
        return
    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(6, 4))

    for eps in df['epsilon'].unique():
        subset = df[df['epsilon'] == eps]
        ax.plot(subset['num_scenarios'], subset['safe_horizon_theoretical'],
                label=f'$\\varepsilon={eps}$', linewidth=2)

    ax.set_xlabel('Number of Scenarios $S$')
    ax.set_ylabel('Safe Horizon $N_{\\mathrm{safe}}$')
    ax.set_title('Theoretical Safe Horizon (Eq. 25, $\\beta=0.01$, $n_u=2$)')
    ax.legend()
    ax.set_ylim(0, 21)
    save_fig(fig, 'fig_h3_theoretical_safe_horizon')


def fig_h4_multi_disc():
    """H4: Multi-disc D=1 vs D=3."""
    path = os.path.join(INPUT_DIR, "exp_h4_multi_disc.csv")
    if not os.path.exists(path):
        print("  Skipping H4: CSV not found")
        return
    df = pd.read_csv(path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    colors_disc = {1: COLORS['base'], 3: COLORS['multi']}
    labels_disc = {1: 'D=1 (single)', 3: 'D=3 (multi-disc)'}

    for _, row in df.iterrows():
        D = int(row['num_discs'])
        ax1.bar(labels_disc[D], row['collision_rate'],
                yerr=[[row['collision_rate'] - row['ci_lo']],
                      [row['ci_hi'] - row['collision_rate']]],
                color=colors_disc[D], capsize=5, width=0.5)
        ax2.bar(labels_disc[D], row['avg_clearance'],
                color=colors_disc[D], width=0.5)

    ax1.set_ylabel('Collision Rate')
    ax1.set_title('Collision Rate')
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax2.set_ylabel('Avg Min Clearance [m]')
    ax2.set_title('Minimum Clearance')

    fig.suptitle('H4: Multi-Disc Collision Model (D=1 vs D=3)', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig_h4_multi_disc')


def fig_h5_ablation():
    """H5: Full ablation table with error bars."""
    path = os.path.join(INPUT_DIR, "exp_h5_ablation_table.csv")
    if not os.path.exists(path):
        print("  Skipping H5: CSV not found")
        return
    df = pd.read_csv(path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    x = np.arange(len(df))
    variant_colors = [COLORS['base'], COLORS['base'], COLORS['dro'],
                      COLORS['multi'], COLORS['full']][:len(df)]

    yerr_lo = np.maximum(df['collision_rate'].values - df['ci_lo'].values, 0)
    yerr_hi = np.maximum(df['ci_hi'].values - df['collision_rate'].values, 0)
    ax1.bar(x, df['collision_rate'],
            yerr=[yerr_lo, yerr_hi],
            color=variant_colors, capsize=4, width=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['variant'], rotation=30, ha='right')
    ax1.set_ylabel('Collision Rate')
    ax1.set_title('Collision Rate by Variant')
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax2.bar(x, df['missed_mode_rate'],
            color=variant_colors, width=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['variant'], rotation=30, ha='right')
    ax2.set_ylabel('Missed Mode Rate')
    ax2.set_title('Missed Mode Rate by Variant')

    fig.suptitle('H5: Full Ablation — Base vs DRO vs Multi-Disc vs SH-MPC', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig_h5_ablation')


def fig_h6_transport_cost():
    """H6: OT ground cost heatmap."""
    path = os.path.join(INPUT_DIR, "exp_h6_ground_cost_matrix.csv")
    if not os.path.exists(path):
        print("  Skipping H6: CSV not found")
        return
    df = pd.read_csv(path)

    modes = df['mode_i'].unique()
    M = len(modes)
    matrix = np.zeros((M, M))
    for _, row in df.iterrows():
        i = list(modes).index(row['mode_i'])
        j = list(modes).index(row['mode_j'])
        matrix[i][j] = row['w2_distance']

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(M))
    ax.set_yticks(range(M))
    short_names = [m[:12] for m in modes]
    ax.set_xticklabels(short_names, rotation=45, ha='right')
    ax.set_yticklabels(short_names)

    for i in range(M):
        for j in range(M):
            ax.text(j, i, f'{matrix[i][j]:.2f}', ha='center', va='center',
                    color='white' if matrix[i][j] > matrix.max() * 0.6 else 'black',
                    fontsize=9)

    plt.colorbar(im, label='W2 Bures Distance')
    ax.set_title('H6: OT Ground Cost Matrix $D[i][j]$')
    fig.tight_layout()
    save_fig(fig, 'fig_h6_transport_cost')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Generating publication figures from {INPUT_DIR}/")
    print(f"Output to {OUT_DIR}/\n")

    fig_h1_mode_coverage()
    fig_h2_collision_reduction()
    fig_h2_significance()
    fig_h3_safe_horizon()
    fig_h3_theoretical()
    fig_h4_multi_disc()
    fig_h5_ablation()
    fig_h6_transport_cost()

    print("\nDone! All figures saved.")


if __name__ == "__main__":
    main()
