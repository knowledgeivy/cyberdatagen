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

def plot_combined_performance_curves_v2(strategy='within_group', figsize=(12, 5)):
    """
    Generate compact combined performance curves (Scheme A).

    Layout: 1 row × 2 columns (SVM left, RF right)
    Shows: F1-Score only
    Lines: GPT (3 prompts) + Claude (3 prompts) + SMOTE (1 baseline)

    Args:
        strategy: 'within_group' or 'cross_group'
        figsize: figure size (width, height)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Professional color scheme (colorblind-friendly)
    colors = {
        'GPT-4.1-mini': '#1f77b4',      # Blue
        'Claude-3.5-Haiku': '#ff7f0e',  # Orange
        'SMOTE': '#2ca02c'               # Green
    }

    # Linestyles for prompts
    linestyles = {
        'original': '-',
        'strong': '--',
        'weak': ':'
    }

    # Markers for prompts
    markers = {
        'original': 'o',
        'strong': 's',
        'weak': '^'
    }

    classifiers = ['svm', 'random_forest']
    prompts = ['original', 'strong', 'weak']

    for idx, classifier in enumerate(classifiers):
        ax = axes[idx]

        # Plot LLM methods (GPT and Claude) with all prompts
        for model_name in ['GPT-4.1-mini', 'Claude-3.5-Haiku']:
            config = EXPERIMENTS[model_name]

            for prompt in prompts:
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
                ratios, means, stds = extract_performance_data(analysis_results, classifier, 'f1_score')

                if len(ratios) == 0:
                    continue

                # Plot line
                ax.plot(ratios, means,
                       color=colors[model_name],
                       linestyle=linestyles[prompt],
                       marker=markers[prompt],
                       linewidth=2.0 if prompt == 'original' else 1.5,
                       markersize=4,
                       alpha=0.85,
                       label=f'{model_name} ({prompt.capitalize()})')

                # Add confidence interval (lighter)
                ax.fill_between(ratios, means - stds, means + stds,
                               alpha=0.10, color=colors[model_name])

        # Plot SMOTE baseline (thick prominent line)
        smote_config = EXPERIMENTS['SMOTE']
        analysis_results = load_statistical_analysis(
            smote_config['dir'],
            smote_config['model_key'],
            'smote',
            strategy
        )

        if analysis_results is not None:
            ratios, means, stds = extract_performance_data(analysis_results, classifier, 'f1_score')

            if len(ratios) > 0:
                # SMOTE as thick baseline
                ax.plot(ratios, means,
                       color=colors['SMOTE'],
                       linestyle='-',
                       linewidth=3.0,
                       marker='D',
                       markersize=5,
                       alpha=0.95,
                       label='SMOTE (Baseline)',
                       zorder=10)  # Bring to front

                ax.fill_between(ratios, means - stds, means + stds,
                               alpha=0.15, color=colors['SMOTE'])

        # Formatting
        classifier_display = classifier.upper().replace('_', ' ')
        ax.set_title(f'{classifier_display}', fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel('Synthetic Ratio (%)', fontsize=11)
        ax.set_ylabel('F1 Score', fontsize=11)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
        ax.set_xlim(-5, 105)

        # Unified y-axis range: 0 to 1.05 (showing 0-1.0 on ticks)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        # Legend - put on SVM (left) subplot to avoid blocking RF curves
        if idx == 0:  # Show legend on SVM (left subplot)
            # Create custom legend with hierarchical grouping
            handles, labels = ax.get_legend_handles_labels()

            # Reorder: GPT methods, then Claude methods, then SMOTE
            gpt_handles = [h for h, l in zip(handles, labels) if 'GPT' in l]
            gpt_labels = [l for l in labels if 'GPT' in l]

            claude_handles = [h for h, l in zip(handles, labels) if 'Claude' in l]
            claude_labels = [l for l in labels if 'Claude' in l]

            smote_handles = [h for h, l in zip(handles, labels) if 'SMOTE' in l]
            smote_labels = [l for l in labels if 'SMOTE' in l]

            # Combine in order
            ordered_handles = gpt_handles + claude_handles + smote_handles
            ordered_labels = gpt_labels + claude_labels + smote_labels

            ax.legend(ordered_handles, ordered_labels,
                     loc='lower left',
                     framealpha=0.95,
                     fontsize=8.5,
                     ncol=1,
                     columnspacing=0.5,
                     handlelength=2.5)

    # Overall title
    strategy_title = strategy.replace('_', '-').title()
    fig.suptitle(f'F1-Score Comparison: {strategy_title} Strategy',
                fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    output_filename = f'combined_f1_comparison_{strategy}.png'
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
    print("Generating Combined Performance Visualizations (Scheme A)")
    print("=" * 80)

    # Generate v2 compact performance curves (F1-Score only, all prompts in one figure)
    for strategy in STRATEGIES:
        print(f"\nGenerating: {strategy} strategy (compact view)")
        plot_combined_performance_curves_v2(
            strategy=strategy,
            figsize=(12, 5)
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
    print(f"\n📊 Generated figures:")
    print(f"  - {OUTPUT_DIR}/combined_f1_comparison_within_group.png")
    print(f"  - {OUTPUT_DIR}/combined_f1_comparison_cross_group.png")

if __name__ == '__main__':
    main()
