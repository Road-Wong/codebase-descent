"""
Generate Paper-Quality Figures for Codebase Descent

Figure 1: Optimization Trajectory PCA ("Spaghetti vs Spiral")
Figure 2: Oscillation Index Comparison Bar Chart
"""

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# Publication-quality defaults
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
rcParams['axes.linewidth'] = 1.2
rcParams['figure.dpi'] = 150


def load_results(json_path):
    """Load experiment results from JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def codes_to_features(codes_list, max_features=200):
    """
    Convert code strings to TF-IDF feature vectors.
    Each code snapshot becomes a point in feature space.
    """
    # Flatten all codes across all trials for fitting
    all_codes = []
    for codes in codes_list:
        all_codes.extend(codes)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        analyzer='char_wb',
        ngram_range=(2, 4),
        sublinear_tf=True,
    )
    vectorizer.fit(all_codes)

    # Transform each trial's trajectory
    trial_features = []
    for codes in codes_list:
        features = vectorizer.transform(codes).toarray()
        trial_features.append(features)

    return trial_features, vectorizer


def plot_trajectory_pca(baseline_trials, smo_trials, save_path):
    """
    Figure 1: PCA trajectory visualization.

    Red lines  = Baseline trajectories (spaghetti — chaotic)
    Blue lines = SMO trajectories (spiral — converging)
    """
    # Collect all codes
    baseline_codes = [t["codes"] for t in baseline_trials if "codes" in t]
    smo_codes = [t["codes"] for t in smo_trials if "codes" in t]

    if not baseline_codes and not smo_codes:
        print("WARNING: No code trajectories found. Skipping PCA plot.")
        return

    # Get TF-IDF features
    all_codes = baseline_codes + smo_codes
    trial_features, vectorizer = codes_to_features(all_codes)

    # Fit PCA on all points
    all_points = np.vstack(trial_features)
    pca = PCA(n_components=2)
    pca.fit(all_points)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot Baseline trajectories (red, spaghetti)
    for i, features in enumerate(trial_features[:len(baseline_codes)]):
        projected = pca.transform(features)
        ax.plot(projected[:, 0], projected[:, 1],
                color='#E74C3C', alpha=0.6, linewidth=1.5,
                label='Baseline' if i == 0 else None)
        # Mark start and end
        ax.scatter(projected[0, 0], projected[0, 1], c='#E74C3C', s=40, zorder=5, marker='o')
        ax.scatter(projected[-1, 0], projected[-1, 1], c='#E74C3C', s=80, zorder=5, marker='*')

    # Plot SMO trajectories (blue, spiral)
    for i, features in enumerate(trial_features[len(baseline_codes):]):
        projected = pca.transform(features)
        ax.plot(projected[:, 0], projected[:, 1],
                color='#2980B9', alpha=0.6, linewidth=1.5,
                label='SMO' if i == 0 else None)
        ax.scatter(projected[0, 0], projected[0, 1], c='#2980B9', s=40, zorder=5, marker='o')
        ax.scatter(projected[-1, 0], projected[-1, 1], c='#2980B9', s=80, zorder=5, marker='*')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', fontsize=13)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', fontsize=13)
    ax.set_title('Optimization Trajectories in Code Embedding Space', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Figure 1 saved to: {save_path}")


def plot_oscillation_bar(baseline_trials, smo_trials, save_path):
    """
    Figure 2: Oscillation Index comparison bar chart.
    """
    baseline_oi = [t["oscillation_index"] for t in baseline_trials]
    smo_oi = [t["oscillation_index"] for t in smo_trials]

    means = [np.mean(baseline_oi), np.mean(smo_oi)]
    stds = [np.std(baseline_oi), np.std(smo_oi)]

    fig, ax = plt.subplots(figsize=(6, 5))

    bars = ax.bar(
        ['Baseline', 'SMO (β=0.7)'],
        means,
        yerr=stds,
        capsize=8,
        color=['#E74C3C', '#2980B9'],
        edgecolor='black',
        linewidth=1.2,
        width=0.5,
        alpha=0.85,
    )

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=13, fontweight='bold')

    ax.set_ylabel('Oscillation Index', fontsize=13)
    ax.set_title('Oscillation Index: Baseline vs SMO', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(means) * 1.3 + 0.5)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Figure 2 saved to: {save_path}")


def plot_loss_curves(baseline_trials, smo_trials, save_path):
    """
    Bonus Figure: Loss curves over steps.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot individual trial curves
    for i, trial in enumerate(baseline_trials):
        steps = [t["step"] for t in trial["trajectory"]]
        losses = [t["loss"] for t in trial["trajectory"]]
        ax.plot(steps, losses, color='#E74C3C', alpha=0.3, linewidth=1)
        if i == 0:
            ax.plot([], [], color='#E74C3C', alpha=0.6, label='Baseline')

    for i, trial in enumerate(smo_trials):
        steps = [t["step"] for t in trial["trajectory"]]
        losses = [t["loss"] for t in trial["trajectory"]]
        ax.plot(steps, losses, color='#2980B9', alpha=0.3, linewidth=1)
        if i == 0:
            ax.plot([], [], color='#2980B9', alpha=0.6, label='SMO (β=0.7)')

    # Plot mean curves
    max_steps_b = max(len(t["trajectory"]) for t in baseline_trials) if baseline_trials else 0
    max_steps_s = max(len(t["trajectory"]) for t in smo_trials) if smo_trials else 0
    max_steps = max(max_steps_b, max_steps_s)

    for agent_type, trials, color in [("Baseline", baseline_trials, '#E74C3C'), ("SMO", smo_trials, '#2980B9')]:
        if not trials:
            continue
        mean_losses = []
        for s in range(max_steps):
            losses_at_s = [t["trajectory"][s]["loss"] for t in trials if s < len(t["trajectory"])]
            mean_losses.append(np.mean(losses_at_s) if losses_at_s else np.nan)
        ax.plot(range(max_steps), mean_losses, color=color, linewidth=2.5, label=f'{agent_type} (mean)')

    ax.set_xlabel('Optimization Step', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_title('Loss Trajectories: Baseline vs SMO', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Figure 3 saved to: {save_path}")


def generate_all_figures(json_path, output_dir=None):
    """Generate all paper figures from experiment data."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(json_path), "figures")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nLoading results from: {json_path}")
    data = load_results(json_path)

    baseline = data.get("baseline", [])
    smo = data.get("smo", [])

    print(f"  Baseline trials: {len(baseline)}")
    print(f"  SMO trials: {len(smo)}")
    print(f"\nGenerating figures...")

    # Figure 1: PCA Trajectories
    if any("codes" in t for t in baseline + smo):
        plot_trajectory_pca(
            baseline, smo,
            os.path.join(output_dir, "fig1_trajectory_pca.png")
        )
    else:
        print("  WARNING: No code data in results. Skipping PCA plot.")
        print("  (Re-run experiment with --save-codes flag)")

    # Figure 2: Oscillation Index
    plot_oscillation_bar(
        baseline, smo,
        os.path.join(output_dir, "fig2_oscillation_index.png")
    )

    # Figure 3: Loss Curves
    if any("trajectory" in t for t in baseline + smo):
        plot_loss_curves(
            baseline, smo,
            os.path.join(output_dir, "fig3_loss_curves.png")
        )

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help="Path to comparison results JSON")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    generate_all_figures(args.json_path, args.output_dir)
