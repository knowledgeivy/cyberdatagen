#!/usr/bin/env python3
"""
Generate combined performance curves comparing GPT-4.1-mini, Claude-3.5-Haiku, and SMOTE baseline.

This script creates comprehensive performance comparison visualizations and tables that combine
all three methods (two LLMs + SMOTE baseline) for direct comparison in the paper.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats

# Configuration
BASE_DIR = Path("/Users/tianyu/Notebooks/cyberdata/output/full_experiments")
OUTPUT_DIR = Path("/Users/tianyu/Notebooks/cyberdata/paper/pic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Experiment configurations
EXPERIMENTS = {
    'GPT-4.1-mini': {
        'dir': BASE_DIR / 'ceas08_gpt41mini',
        'model_key': 'gpt41mini',
        'prompts': ['original', 'strong', 'weak']
    },
    'Claude-3.5-Haiku': {
        'dir': BASE_DIR / 'ceas08_claude35haiku',
        'model_key': 'claude35haiku',
        'prompts': ['original', 'strong', 'weak']
    },
    'SMOTE': {
        'dir': BASE_DIR / 'ceas08_smote',
        'model_key': 'smote',
        'prompts': ['smote']  # SMOTE has no prompt variations
    }
}

STRATEGIES = ['within_group', 'cross_group']
CLASSIFIERS = ['svm', 'random_forest']
SYNTHETIC_RATIOS = list(range(0, 101, 10))

def load_statistical_analysis(exp_dir, model_key, prompt, strategy):
    """Load statistical analysis JSON file."""
    analysis_file = exp_dir / "reports" / f"full_ceas08_{model_key}_v1_{prompt}_{strategy}_statistical_analysis.json"

    if not analysis_file.exists():
        print(f"Warning: File not found: {analysis_file}")
        return None

    with open(analysis_file, 'r') as f:
        data = json.load(f)

    return data

def extract_performance_data(analysis_results, classifier='svm', metric='f1_score'):
    """Extract mean and std for performance metric across synthetic ratios from statistical analysis results."""
    means = []
    stds = []
    ratios = []

    descriptive_stats = analysis_results.get('descriptive_statistics', {})

    for ratio in SYNTHETIC_RATIOS:
        ratio_key = str(ratio)
        if ratio_key in descriptive_stats and classifier in descriptive_stats[ratio_key]:
            metric_data = descriptive_stats[ratio_key][classifier].get(metric, {})
            if isinstance(metric_data, dict):
                means.append(metric_data.get('mean', np.nan))
                stds.append(metric_data.get('std', np.nan))
                ratios.append(ratio)
            elif not np.isnan(metric_data):
                means.append(metric_data)
                stds.append(0.0)
                ratios.append(ratio)

    return np.array(ratios), np.array(means), np.array(stds)

def plot_combined_performance_curves(strategy='within_group', metric='f1_score',
                                    prompt_filter='original', figsize=(14, 10)):
    """
    Generate combined performance curves for all models and classifiers.

    Args:
        strategy: 'within_group' or 'cross_group'
        metric: performance metric to plot
        prompt_filter: which prompt to use for LLMs ('original', 'strong', 'weak')
        figsize: figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Combined Performance Comparison: {strategy.replace("_", "-").title()} Strategy',
                 fontsize=16, fontweight='bold', y=0.995)

    # Color scheme
    colors = {
        'GPT-4.1-mini': '#1f77b4',      # Blue
        'Claude-3.5-Haiku': '#ff7f0e',  # Orange
        'SMOTE': '#2ca02c'               # Green
    }

    linestyles = {
        'svm': '-',
        'random_forest': '--'
    }

    # Plot configurations: (classifier, col)
    plot_configs = [
        ('svm', 0),
        ('random_forest', 1)
    ]

    for idx, (classifier, col) in enumerate(plot_configs):
        # F1-Score subplot
        ax_f1 = axes[0, col]
        ax_acc = axes[1, col]

        # Plot each model
        for model_name, config in EXPERIMENTS.items():
            # Determine which prompt to use
            if model_name == 'SMOTE':
                prompt = 'smote'
            else:
                prompt = prompt_filter

            # Load data
            analysis_results = load_statistical_analysis(
                config['dir'],
                config['model_key'],
                prompt,
                strategy
            )

            if analysis_results is None:
                continue

            # Extract F1-score data
            ratios, means_f1, stds_f1 = extract_performance_data(analysis_results, classifier, 'f1_score')
            ratios, means_acc, stds_acc = extract_performance_data(analysis_results, classifier, 'accuracy')

            if len(ratios) == 0:
                continue

            # Plot F1-score
            ax_f1.plot(ratios, means_f1, label=model_name,
                      color=colors[model_name], linestyle=linestyles[classifier],
                      linewidth=2.5, marker='o', markersize=5, alpha=0.9)
            ax_f1.fill_between(ratios, means_f1 - stds_f1, means_f1 + stds_f1,
                              alpha=0.15, color=colors[model_name])

            # Plot Accuracy
            ax_acc.plot(ratios, means_acc, label=model_name,
                       color=colors[model_name], linestyle=linestyles[classifier],
                       linewidth=2.5, marker='s', markersize=5, alpha=0.9)
            ax_acc.fill_between(ratios, means_acc - stds_acc, means_acc + stds_acc,
                               alpha=0.15, color=colors[model_name])

        # Format F1-score subplot
        classifier_display = classifier.upper().replace('_', ' ')
        ax_f1.set_title(f'{classifier_display} - F1 Score', fontsize=12, fontweight='bold', pad=10)
        ax_f1.set_xlabel('Synthetic Ratio (%)', fontsize=11)
        ax_f1.set_ylabel('F1 Score', fontsize=11)
        ax_f1.grid(True, alpha=0.3, linestyle='--')
        ax_f1.set_xlim(-5, 105)
        ax_f1.set_ylim(0.4, 1.0)
        ax_f1.legend(loc='best', framealpha=0.9, fontsize=10)

        # Format Accuracy subplot
        ax_acc.set_title(f'{classifier_display} - Accuracy', fontsize=12, fontweight='bold', pad=10)
        ax_acc.set_xlabel('Synthetic Ratio (%)', fontsize=11)
        ax_acc.set_ylabel('Accuracy', fontsize=11)
        ax_acc.grid(True, alpha=0.3, linestyle='--')
        ax_acc.set_xlim(-5, 105)
        ax_acc.set_ylim(0.85, 1.0)
        ax_acc.legend(loc='best', framealpha=0.9, fontsize=10)

    plt.tight_layout()

    # Save figure
    output_filename = f'combined_performance_curves_{strategy}_{prompt_filter}.png'
    output_path = OUTPUT_DIR / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    plt.close()

def generate_performance_comparison_table(strategy='within_group', prompt='original'):
    """Generate comparison table data for all models."""

    table_data = []

    for model_name, config in EXPERIMENTS.items():
        # Determine prompt
        if model_name == 'SMOTE':
            prompt_use = 'smote'
        else:
            prompt_use = prompt

        # Load data
        analysis_results = load_statistical_analysis(
            config['dir'],
            config['model_key'],
            prompt_use,
            strategy
        )

        if analysis_results is None:
            continue

        for classifier in CLASSIFIERS:
            # Get baseline (0%) and 100% synthetic
            ratios, means, stds = extract_performance_data(analysis_results, classifier, 'f1_score')

            if len(ratios) < 2:
                continue

            baseline_mean = means[ratios == 0][0] if 0 in ratios else np.nan
            baseline_std = stds[ratios == 0][0] if 0 in ratios else np.nan

            synthetic100_mean = means[ratios == 100][0] if 100 in ratios else np.nan
            synthetic100_std = stds[ratios == 100][0] if 100 in ratios else np.nan

            degradation = ((synthetic100_mean - baseline_mean) / baseline_mean * 100) if not np.isnan(baseline_mean) else np.nan

            table_data.append({
                'Model': model_name,
                'Classifier': classifier.replace('_', ' ').title(),
                'Baseline (0%)': f'{baseline_mean:.3f} ± {baseline_std:.3f}',
                '100% Synthetic': f'{synthetic100_mean:.3f} ± {synthetic100_std:.3f}',
                'Degradation (%)': f'{degradation:.1f}%'
            })

    df = pd.DataFrame(table_data)

    # Save to CSV
    output_csv = OUTPUT_DIR.parent / 'tables' / f'performance_comparison_{strategy}_{prompt}.csv'
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✓ Saved table: {output_csv}")

    return df

def main():
    """Generate all combined visualizations and tables."""

    print("=" * 80)
    print("Generating Combined Performance Visualizations")
    print("=" * 80)

    # Generate performance curves for all combinations
    for strategy in STRATEGIES:
        for prompt in ['original', 'strong', 'weak']:
            print(f"\nGenerating: {strategy} - {prompt} prompt")
            plot_combined_performance_curves(
                strategy=strategy,
                prompt_filter=prompt,
                figsize=(14, 10)
            )

    # Generate comparison tables
    print("\n" + "=" * 80)
    print("Generating Comparison Tables")
    print("=" * 80)

    for strategy in STRATEGIES:
        for prompt in ['original', 'strong', 'weak']:
            print(f"\nGenerating table: {strategy} - {prompt}")
            df = generate_performance_comparison_table(strategy, prompt)
            print(df.to_string(index=False))

    print("\n" + "=" * 80)
    print("✓ All visualizations and tables generated successfully!")
    print("=" * 80)

if __name__ == '__main__':
    main()
