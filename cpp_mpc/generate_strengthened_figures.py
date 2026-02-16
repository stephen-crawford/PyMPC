#!/usr/bin/env python3
"""
Generate publication figures from strengthened experiment results.

Reads CSVs from strengthened_results/ and produces ~15 figures.
Usage: python3 generate_strengthened_figures.py [output_dir]
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

INPUT_DIR = "strengthened_results"
OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "strengthened_results/figures"

# Publication style
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox_inches': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'no_injection': '#1f77b4',
    'dro_full': '#d62728',
    'dro_no_cov': '#ff7f0e',
    'dro_distance_only': '#2ca02c',
    'random_injection': '#9467bd',
    'always_inject': '#8c564b',
    'w2_bures': '#d62728',
    'zero_one': '#ff7f0e',
    'euclidean_mean': '#2ca02c',
}

LABELS = {
    'no_injection': 'Base (no DRO)',
    'dro_full': 'DRO (full)',
    'dro_no_cov': 'DRO (no cov)',
    'dro_distance_only': 'DRO (dist only)',
    'random_injection': 'Random inject',
    'always_inject': 'Always inject',
    'w2_bures': '$W_2$ Bures',
    'zero_one': '0/1 cost',
    'euclidean_mean': 'Euclidean mean',
}


def safe_read(name):
    path = os.path.join(INPUT_DIR, name)
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return None
    return pd.read_csv(path)


def savefig(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  -> {name}.png/.pdf")


# ============================================================================
# Figure functions
# ============================================================================

def fig_ablation_matrix():
    """Grouped bar chart: 6 variants x 3 S values."""
    df = safe_read("ablation_matrix.csv")
    if df is None:
        return

    groups = df.groupby(['S', 'method'])['collision'].mean().reset_index()
    S_vals = sorted(groups['S'].unique())
    methods = ['no_injection', 'dro_full', 'dro_no_cov',
               'dro_distance_only', 'random_injection', 'always_inject']

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(S_vals))
    width = 0.12
    offsets = np.arange(len(methods)) - len(methods) / 2 + 0.5

    for i, method in enumerate(methods):
        rates = []
        for S in S_vals:
            sub = groups[(groups['S'] == S) & (groups['method'] == method)]
            rates.append(sub['collision'].values[0] if len(sub) > 0 else 0)
        ax.bar(x + offsets[i] * width, rates, width,
               label=LABELS.get(method, method),
               color=COLORS.get(method, f'C{i}'))

    ax.set_xlabel('Number of scenarios (S)')
    ax.set_ylabel('Collision rate')
    ax.set_title('Ablation Matrix: Collision Rate by Variant and S')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in S_vals])
    ax.legend(loc='upper right', ncol=2)
    ax.set_ylim(bottom=0)
    savefig(fig, 'fig_ablation_matrix')


def fig_distribution_shift():
    """Collision rate vs rho, faceted by boost."""
    df = safe_read("shift_sweep.csv")
    if df is None:
        return

    boosts = sorted(df['shift_boost'].unique())
    fig, axes = plt.subplots(1, len(boosts), figsize=(4 * len(boosts), 4), sharey=True)
    if len(boosts) == 1:
        axes = [axes]

    for ax, boost in zip(axes, boosts):
        sub = df[df['shift_boost'] == boost]
        grp = sub.groupby('shift_rho')['collision'].mean().reset_index()
        ax.plot(grp['shift_rho'], grp['collision'], 'o-', color=COLORS['dro_full'])
        ax.set_xlabel(r'Shift $\rho$')
        ax.set_title(f'boost={boost:.2f}')
        ax.set_ylim(bottom=0)

    axes[0].set_ylabel('Collision rate')
    fig.suptitle('Distribution Shift Robustness', y=1.02)
    fig.tight_layout()
    savefig(fig, 'fig_distribution_shift')


def fig_hyperparameter_eps():
    """Collision rate vs eps_wass."""
    df = safe_read("hyperparam_sweep.csv")
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for sigma in sorted(df['sigma'].unique()):
        sub = df[df['sigma'] == sigma]
        grp = sub.groupby('eps_wass')['collision'].mean().reset_index()
        ax.plot(grp['eps_wass'], grp['collision'], 'o-', label=f'$\\sigma$={sigma}')

    ax.set_xlabel(r'Wasserstein ball radius $\varepsilon$')
    ax.set_ylabel('Collision rate')
    ax.set_title('Hyperparameter Sensitivity: $\\varepsilon$')
    ax.legend()
    ax.set_ylim(bottom=0)
    savefig(fig, 'fig_hyperparameter_eps')


def fig_hyperparameter_sigma():
    """Collision rate vs sigma."""
    df = safe_read("hyperparam_sweep.csv")
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for eps in sorted(df['eps_wass'].unique()):
        sub = df[df['eps_wass'] == eps]
        grp = sub.groupby('sigma')['collision'].mean().reset_index()
        ax.plot(grp['sigma'], grp['collision'], 'o-', label=f'$\\varepsilon$={eps}')

    ax.set_xlabel(r'Risk sigma scale $\sigma$')
    ax.set_ylabel('Collision rate')
    ax.set_title('Hyperparameter Sensitivity: $\\sigma$')
    ax.legend()
    ax.set_ylim(bottom=0)
    savefig(fig, 'fig_hyperparameter_sigma')


def fig_multi_seed_stability():
    """Scatter of collision rate per seed + mean line."""
    df = safe_read("multi_seed.csv")
    if df is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    S_vals = sorted(df['S'].unique())

    for ax, S in zip(axes, S_vals):
        for method in ['no_injection', 'dro_full']:
            sub = df[(df['S'] == S) & (df['method'] == method)]
            # Group by seed to get per-seed collision rate
            by_seed = sub.groupby('seed')['collision'].mean().values
            ax.scatter(range(len(by_seed)), by_seed, alpha=0.5, s=20,
                       color=COLORS.get(method), label=LABELS.get(method))
            ax.axhline(by_seed.mean(), color=COLORS.get(method), ls='--', alpha=0.7)

        ax.set_xlabel('Seed index')
        ax.set_title(f'S={S}')
        ax.set_ylim(-0.05, 1.05)

    axes[0].set_ylabel('Collision rate')
    axes[0].legend()
    fig.suptitle('Multi-Seed Stability', y=1.02)
    fig.tight_layout()
    savefig(fig, 'fig_multi_seed_stability')


def fig_solve_time_cdf():
    """CDF overlay of solve times: Base vs DRO."""
    df = safe_read("solve_time_cdf.csv")
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for method in ['no_injection', 'dro_full']:
        sub = df[df['method'] == method]
        times = sub['avg_solve_ms'].dropna().sort_values()
        cdf = np.arange(1, len(times) + 1) / len(times)
        ax.plot(times, cdf, label=LABELS.get(method), color=COLORS.get(method))

    ax.set_xlabel('Avg solve time per rollout (ms)')
    ax.set_ylabel('CDF')
    ax.set_title('Solve Time CDF: Base vs DRO')
    ax.legend()
    savefig(fig, 'fig_solve_time_cdf')


def fig_solve_time_boxplot():
    """Box plot of solve times per variant."""
    df = safe_read("ablation_matrix.csv")
    if df is None:
        return

    methods = ['no_injection', 'dro_full', 'dro_no_cov',
               'dro_distance_only', 'random_injection', 'always_inject']
    fig, ax = plt.subplots(figsize=(8, 4))

    data = []
    labels = []
    for method in methods:
        sub = df[df['method'] == method]
        if len(sub) > 0:
            data.append(sub['avg_solve_ms'].dropna().values)
            labels.append(LABELS.get(method, method))

    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, method in zip(bp['boxes'], methods):
            patch.set_facecolor(COLORS.get(method, 'lightblue'))
            patch.set_alpha(0.6)

    ax.set_ylabel('Avg solve time (ms)')
    ax.set_title('Solve Time Distribution by Variant')
    plt.xticks(rotation=30, ha='right')
    fig.tight_layout()
    savefig(fig, 'fig_solve_time_boxplot')


def fig_ground_cost_comparison():
    """Bar chart: 3 cost types collision rates."""
    df = safe_read("ground_cost.csv")
    if df is None:
        return

    grp = df.groupby('ground_cost')['collision'].mean().reset_index()
    cost_types = ['w2_bures', 'zero_one', 'euclidean_mean']

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(cost_types))
    rates = [grp[grp['ground_cost'] == ct]['collision'].values[0]
             if len(grp[grp['ground_cost'] == ct]) > 0 else 0
             for ct in cost_types]
    colors = [COLORS.get(ct) for ct in cost_types]
    bars = ax.bar(x, rates, color=colors, width=0.5)

    ax.set_xlabel('Ground cost type')
    ax.set_ylabel('Collision rate')
    ax.set_title('Ground Cost Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(ct, ct) for ct in cost_types])
    ax.set_ylim(bottom=0)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{rate:.3f}', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    savefig(fig, 'fig_ground_cost_comparison')


def fig_calibration():
    """Empirical violation vs target epsilon, with 45-degree line."""
    df = safe_read("calibration.csv")
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 0.55], [0, 0.55], 'k--', alpha=0.4, label='Perfect calibration')
    ax.errorbar(df['target_eps'], df['empirical_violation'],
                yerr=[df['empirical_violation'] - df['ci_low'],
                      df['ci_high'] - df['empirical_violation']],
                fmt='o-', color=COLORS['dro_full'], capsize=3, label='DRO empirical')

    ax.set_xlabel(r'Target $\varepsilon$')
    ax.set_ylabel('Empirical violation rate')
    ax.set_title('Calibration: Target vs Empirical Violation')
    ax.legend()
    ax.set_xlim(-0.02, 0.55)
    ax.set_ylim(-0.02, 0.55)
    ax.set_aspect('equal')
    savefig(fig, 'fig_calibration')


def fig_forest_plot():
    """Forest plot of collision rate differences + CIs."""
    df = safe_read("all_rollouts.csv")
    boot_df = safe_read("bootstrap_ci.csv")
    if df is None:
        return

    base = df[df['method'] == 'no_injection']
    dro = df[df['method'] == 'dro_full']

    if len(base) == 0 or len(dro) == 0:
        return

    p_base = base['collision'].mean()
    p_dro = dro['collision'].mean()
    delta = p_base - p_dro

    fig, ax = plt.subplots(figsize=(6, 3))

    # Use bootstrap CI if available
    if boot_df is not None and len(boot_df) > 0:
        ci_lo = boot_df['ci_low'].values[0]
        ci_hi = boot_df['ci_high'].values[0]
    else:
        # Approximate CI
        n = len(base)
        se = np.sqrt(p_base * (1 - p_base) / n + p_dro * (1 - p_dro) / n)
        ci_lo = delta - 1.96 * se
        ci_hi = delta + 1.96 * se

    ax.errorbar([delta], [0], xerr=[[delta - ci_lo], [ci_hi - delta]],
                fmt='D', color=COLORS['dro_full'], capsize=5, markersize=8)
    ax.axvline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Collision rate reduction (Base - DRO)')
    ax.set_yticks([0])
    ax.set_yticklabels(['S=20'])
    ax.set_title('Forest Plot: DRO Effect on Collision Rate')
    fig.tight_layout()
    savefig(fig, 'fig_forest_plot')


def fig_qualitative_trajectories():
    """Side-by-side Base vs DRO for hard cases."""
    for i in range(1, 4):
        df = safe_read(f"qualitative_case_{i}.csv")
        if df is None:
            continue

        fig, ax = plt.subplots(figsize=(6, 3))
        for _, row in df.iterrows():
            method = row['method']
            label = LABELS.get(method, method)
            color = COLORS.get(method, 'gray')
            marker = 'x' if row['collision'] else 'o'
            ax.scatter(row['total_progress'], row['min_clearance'],
                       color=color, marker=marker, s=100, label=label)

        ax.set_xlabel('Progress (m)')
        ax.set_ylabel('Min clearance (m)')
        ax.set_title(f'Qualitative Case {i}')
        ax.legend()
        fig.tight_layout()
        savefig(fig, f'fig_qualitative_case_{i}')


def fig_safety_efficiency_pareto():
    """Progress vs clearance scatter for Base vs DRO."""
    df = safe_read("all_rollouts.csv")
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    for method in ['no_injection', 'dro_full']:
        sub = df[df['method'] == method]
        ax.scatter(sub['total_progress'], sub['min_clearance'],
                   alpha=0.3, s=15, color=COLORS.get(method),
                   label=LABELS.get(method))

    ax.set_xlabel('Total progress (m)')
    ax.set_ylabel('Min clearance (m)')
    ax.set_title('Safety-Efficiency Pareto')
    ax.legend()
    fig.tight_layout()
    savefig(fig, 'fig_safety_efficiency_pareto')


def fig_clearance_histogram():
    """Histogram of min clearance per variant."""
    df = safe_read("all_rollouts.csv")
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for method in ['no_injection', 'dro_full']:
        sub = df[df['method'] == method]
        ax.hist(sub['min_clearance'], bins=30, alpha=0.5,
                color=COLORS.get(method), label=LABELS.get(method))

    ax.set_xlabel('Min clearance (m)')
    ax.set_ylabel('Count')
    ax.set_title('Clearance Distribution')
    ax.legend()
    fig.tight_layout()
    savefig(fig, 'fig_clearance_histogram')


def fig_effect_size_summary():
    """Bar chart of effect sizes from stat_summary."""
    df = safe_read("stat_summary.csv")
    if df is None:
        return

    metrics = {}
    for _, row in df.iterrows():
        try:
            metrics[row['metric']] = float(row['value'])
        except (ValueError, TypeError):
            metrics[row['metric']] = row['value']

    names = ['abs_delta', 'cohens_h']
    values = [metrics.get(n, 0) for n in names]
    labels = ['Absolute $\\Delta$', "Cohen's $h$"]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(range(len(names)), values, color=[COLORS['dro_full']] * len(names))
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Effect size')
    ax.set_title('Effect Size Summary')
    ax.axvline(0, color='gray', ls='--', alpha=0.5)
    fig.tight_layout()
    savefig(fig, 'fig_effect_size_summary')


def table_ablation_latex():
    """Generate LaTeX table for ablation results."""
    df = safe_read("ablation_matrix.csv")
    if df is None:
        return

    grp = df.groupby(['S', 'method']).agg(
        collision_rate=('collision', 'mean'),
        n=('collision', 'count'),
        mean_progress=('total_progress', 'mean'),
        mean_clearance=('min_clearance', 'mean'),
    ).reset_index()

    methods = ['no_injection', 'dro_full', 'dro_no_cov',
               'dro_distance_only', 'random_injection', 'always_inject']
    S_vals = sorted(grp['S'].unique())

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation matrix: collision rate (\%) by variant and $S$.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{l" + "r" * len(S_vals) + "}",
        r"\toprule",
        "Variant & " + " & ".join([f"$S={s}$" for s in S_vals]) + r" \\",
        r"\midrule",
    ]

    for method in methods:
        row = LABELS.get(method, method)
        for S in S_vals:
            sub = grp[(grp['S'] == S) & (grp['method'] == method)]
            if len(sub) > 0:
                rate = sub['collision_rate'].values[0]
                row += f" & {rate*100:.1f}"
            else:
                row += " & --"
        row += r" \\"
        lines.append(row)

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    outpath = os.path.join(OUTPUT_DIR, "table_ablation.tex")
    with open(outpath, 'w') as f:
        f.write("\n".join(lines))
    print(f"  -> table_ablation.tex")


# ============================================================================
# Main
# ============================================================================

def main():
    print(f"Reading from: {INPUT_DIR}/")
    print(f"Writing to:   {OUTPUT_DIR}/")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig_ablation_matrix()
    fig_distribution_shift()
    fig_hyperparameter_eps()
    fig_hyperparameter_sigma()
    fig_multi_seed_stability()
    fig_solve_time_cdf()
    fig_solve_time_boxplot()
    fig_ground_cost_comparison()
    fig_calibration()
    fig_forest_plot()
    fig_qualitative_trajectories()
    fig_safety_efficiency_pareto()
    fig_clearance_histogram()
    fig_effect_size_summary()
    table_ablation_latex()

    print("\nDone.")


if __name__ == "__main__":
    main()
