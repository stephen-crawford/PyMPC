#!/usr/bin/env python3
"""
Visualize uncertainty limits experiment results.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import csv
import os

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
})


def load_csv(filename):
    if not os.path.exists(filename):
        return None
    rows = []
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    if '.' in v or 'e' in v.lower():
                        parsed[k] = float(v)
                    else:
                        parsed[k] = int(v)
                except ValueError:
                    parsed[k] = v
            rows.append(parsed)
    return rows


def generate_baseline_figure():
    rows = load_csv('uncertainty_baseline.csv')
    if not rows:
        print("  Skipping baseline: data not found"); return

    offsets = []
    clearances = []
    for r in rows:
        offsets.append(r['obs_y_offset'])
        clearances.append(r['min_clearance'])

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    colors = ['red' if c < 0 else 'green' for c in clearances]
    ax.bar(range(len(offsets)), clearances, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='red', linewidth=2, linestyle='--', label='Collision boundary')
    ax.set_xticks(range(len(offsets)))
    ax.set_xticklabels([f'{o:.1f}' for o in offsets])
    ax.set_xlabel('Obstacle Lateral Offset [m]')
    ax.set_ylabel('Min Clearance [m]')
    ax.set_title('Experiment 1: Baseline MPC Capability (deterministic obstacle, noise=0)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    plt.tight_layout()
    plt.savefig('uncertainty_baseline.png')
    print("  Saved uncertainty_baseline.png")
    plt.close()


def generate_matched_noise_figure():
    rows = load_csv('uncertainty_matched_noise.csv')
    if not rows:
        print("  Skipping matched noise: data not found"); return

    by_pair = defaultdict(list)
    for r in rows:
        by_pair[(r['obs_y_offset'], r['noise_scale'])].append(r)

    offsets = sorted(set(r['obs_y_offset'] for r in rows))
    noises = sorted(set(r['noise_scale'] for r in rows))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, len(offsets)-1)) for i in range(len(offsets))]

    # Panel 1: Collision rate vs noise
    ax = axes[0]
    for i, off in enumerate(offsets):
        rates = []
        for ns in noises:
            trials = by_pair.get((off, ns), [])
            n = len(trials) if trials else 1
            rate = sum(1 for t in trials if t['collision_steps'] > 0) / n * 100
            rates.append(rate)
        ax.plot(noises, rates, '-o', color=colors[i], linewidth=2,
                markersize=4, label=f'offset={off}m')
    ax.set_xlabel('Noise Scale')
    ax.set_ylabel('Collision Rate (%)')
    ax.set_title('Collision Rate vs Noise (matched model)')
    ax.set_ylim(-5, 105)
    ax.set_xscale('symlog', linthresh=1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 2: Min clearance vs noise
    ax = axes[1]
    for i, off in enumerate(offsets):
        clears = []
        clears_std = []
        for ns in noises:
            trials = by_pair.get((off, ns), [])
            if trials:
                vals = [t['min_clearance'] for t in trials]
                clears.append(np.mean(vals))
                clears_std.append(np.std(vals))
            else:
                clears.append(0)
                clears_std.append(0)
        ax.errorbar(noises, clears, yerr=clears_std,
                    fmt='-o', color=colors[i], linewidth=2,
                    markersize=4, capsize=3, label=f'offset={off}m')
    ax.axhline(y=0, color='red', linewidth=2, linestyle='--')
    ax.set_xlabel('Noise Scale')
    ax.set_ylabel('Min Clearance [m]')
    ax.set_title('Safety Margin vs Noise')
    ax.set_xscale('symlog', linthresh=1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 3: Max obstacle deviation vs noise
    ax = axes[2]
    for i, off in enumerate(offsets):
        devs = []
        for ns in noises:
            trials = by_pair.get((off, ns), [])
            if trials:
                devs.append(np.mean([t['max_obs_deviation'] for t in trials]))
            else:
                devs.append(0)
        ax.plot(noises, devs, '-o', color=colors[i], linewidth=2,
                markersize=4, label=f'offset={off}m')
    ax.set_xlabel('Noise Scale')
    ax.set_ylabel('Max Obstacle Deviation [m]')
    ax.set_title('Obstacle Trajectory Deviation')
    ax.set_xscale('symlog', linthresh=1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle('Experiment 2: Matched Model Noise (actual_noise = prediction_noise)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('uncertainty_matched_noise.png')
    print("  Saved uncertainty_matched_noise.png")
    plt.close()


def generate_phase_diagram():
    rows = load_csv('uncertainty_phase_diagram.csv')
    if not rows:
        print("  Skipping phase diagram: data not found"); return

    by_pair = defaultdict(list)
    for r in rows:
        by_pair[(r['obs_y_offset'], r['noise_scale'])].append(r)

    offsets = sorted(set(r['obs_y_offset'] for r in rows))
    noises = sorted(set(r['noise_scale'] for r in rows))

    # Build matrices
    coll_matrix = np.zeros((len(offsets), len(noises)))
    clear_matrix = np.zeros((len(offsets), len(noises)))

    for i, off in enumerate(offsets):
        for j, ns in enumerate(noises):
            trials = by_pair.get((off, ns), [])
            if trials:
                n = len(trials)
                coll_matrix[i, j] = sum(1 for t in trials if t['collision_steps'] > 0) / n * 100
                clear_matrix[i, j] = np.mean([t['min_clearance'] for t in trials])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Collision rate heatmap
    ax = axes[0]
    im = ax.imshow(coll_matrix, aspect='auto', cmap='RdYlGn_r',
                    vmin=0, vmax=100, origin='lower')
    ax.set_xticks(range(len(noises)))
    ax.set_xticklabels([f'{n:.0f}' if n >= 1 else f'{n}' for n in noises], rotation=45)
    ax.set_yticks(range(len(offsets)))
    ax.set_yticklabels([f'{o:.1f}' for o in offsets])
    ax.set_xlabel('Noise Scale (G multiplier)')
    ax.set_ylabel('Obstacle Lateral Offset [m]')
    ax.set_title('Collision Rate (%)')
    plt.colorbar(im, ax=ax, label='Collision Rate (%)')

    for i in range(len(offsets)):
        for j in range(len(noises)):
            val = coll_matrix[i, j]
            color = 'white' if val > 50 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=8, color=color)

    # Right: Clearance heatmap
    ax = axes[1]
    im2 = ax.imshow(clear_matrix, aspect='auto', cmap='RdYlGn',
                     origin='lower')
    ax.set_xticks(range(len(noises)))
    ax.set_xticklabels([f'{n:.0f}' if n >= 1 else f'{n}' for n in noises], rotation=45)
    ax.set_yticks(range(len(offsets)))
    ax.set_yticklabels([f'{o:.1f}' for o in offsets])
    ax.set_xlabel('Noise Scale (G multiplier)')
    ax.set_ylabel('Obstacle Lateral Offset [m]')
    ax.set_title('Mean Min Clearance [m]')
    plt.colorbar(im2, ax=ax, label='Min Clearance [m]')

    for i in range(len(offsets)):
        for j in range(len(noises)):
            val = clear_matrix[i, j]
            color = 'white' if val < 0 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)

    fig.suptitle('Experiment 3: Safety Phase Diagram (stochastic obstacle, matched model)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('uncertainty_phase_diagram.png')
    print("  Saved uncertainty_phase_diagram.png")
    plt.close()


def generate_mismatch_figure():
    rows = load_csv('uncertainty_mismatch.csv')
    if not rows:
        print("  Skipping model mismatch: data not found"); return

    by_pair = defaultdict(list)
    for r in rows:
        by_pair[(r['actual_noise'], r['pred_ratio'])].append(r)

    actual_noises = sorted(set(r['actual_noise'] for r in rows))
    pred_ratios = sorted(set(r['pred_ratio'] for r in rows))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.cm.plasma
    colors = [cmap(i / max(1, len(actual_noises)-1)) for i in range(len(actual_noises))]

    # Panel 1: Collision rate vs prediction ratio
    ax = axes[0]
    for i, an in enumerate(actual_noises):
        rates = []
        for ratio in pred_ratios:
            trials = by_pair.get((an, ratio), [])
            n = len(trials) if trials else 1
            rate = sum(1 for t in trials if t['collision_steps'] > 0) / n * 100
            rates.append(rate)
        ax.plot(pred_ratios, rates, '-o', color=colors[i], linewidth=2,
                markersize=5, label=f'actual={an:.0f}')
    ax.axvline(x=1.0, color='gray', linewidth=2, linestyle='--', label='Matched (ratio=1)')
    ax.set_xlabel('Prediction/Actual Noise Ratio')
    ax.set_ylabel('Collision Rate (%)')
    ax.set_title('Collision Rate vs Model Mismatch')
    ax.set_ylim(-5, 105)
    ax.set_xscale('symlog', linthresh=0.1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 2: Mean clearance vs prediction ratio
    ax = axes[1]
    for i, an in enumerate(actual_noises):
        clears = []
        for ratio in pred_ratios:
            trials = by_pair.get((an, ratio), [])
            if trials:
                clears.append(np.mean([t['min_clearance'] for t in trials]))
            else:
                clears.append(0)
        ax.plot(pred_ratios, clears, '-o', color=colors[i], linewidth=2,
                markersize=5, label=f'actual={an:.0f}')
    ax.axvline(x=1.0, color='gray', linewidth=2, linestyle='--', label='Matched')
    ax.axhline(y=0, color='red', linewidth=2, linestyle='--')
    ax.set_xlabel('Prediction/Actual Noise Ratio')
    ax.set_ylabel('Min Clearance [m]')
    ax.set_title('Safety Margin vs Model Mismatch')
    ax.set_xscale('symlog', linthresh=0.1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle('Experiment 4: Model Mismatch (offset=2.0m)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('uncertainty_mismatch.png')
    print("  Saved uncertainty_mismatch.png")
    plt.close()


def generate_scenario_recovery():
    rows = load_csv('uncertainty_scenario_recovery.csv')
    if not rows:
        print("  Skipping scenario recovery: data not found"); return

    by_pair = defaultdict(list)
    for r in rows:
        by_pair[(r['noise_scale'], int(r['num_scenarios']))].append(r)

    noise_levels = sorted(set(r['noise_scale'] for r in rows))
    scenario_counts = sorted(set(int(r['num_scenarios']) for r in rows))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.cm.plasma
    colors = [cmap(i / max(1, len(noise_levels)-1)) for i in range(len(noise_levels))]

    ax = axes[0]
    for i, ns in enumerate(noise_levels):
        rates = []
        for S in scenario_counts:
            trials = by_pair.get((ns, S), [])
            n = len(trials) if trials else 1
            rate = sum(1 for t in trials if t['collision_steps'] > 0) / n * 100
            rates.append(rate)
        ax.plot(scenario_counts, rates, '-o', color=colors[i], linewidth=2,
                markersize=4, label=f'noise={ns:.0f}')
    ax.set_xlabel('Number of Scenarios')
    ax.set_ylabel('Collision Rate (%)')
    ax.set_title('Can More Scenarios Help?')
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for i, ns in enumerate(noise_levels):
        clears = []
        for S in scenario_counts:
            trials = by_pair.get((ns, S), [])
            clears.append(np.mean([t['min_clearance'] for t in trials]) if trials else 0)
        ax.plot(scenario_counts, clears, '-o', color=colors[i], linewidth=2,
                markersize=4, label=f'noise={ns:.0f}')
    ax.axhline(y=0, color='red', linewidth=2, linestyle='--')
    ax.set_xlabel('Number of Scenarios')
    ax.set_ylabel('Min Clearance [m]')
    ax.set_title('Safety Margin vs Scenario Count')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle('Experiment 5: Scenario Count Recovery (offset=2.0m, matched model)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('uncertainty_scenario_recovery.png')
    print("  Saved uncertainty_scenario_recovery.png")
    plt.close()


def main():
    print("Generating uncertainty limits analysis figures...")
    generate_baseline_figure()
    generate_matched_noise_figure()
    generate_phase_diagram()
    generate_mismatch_figure()
    generate_scenario_recovery()
    print("\nDone!")


if __name__ == '__main__':
    main()
