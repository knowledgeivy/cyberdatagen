#!/usr/bin/env python3
"""
Generate Track 1 paper plot: GPT and Claude side-by-side
Showing Precision, Recall, and F1-Score
Optimized for publication with larger fonts
"""

import argparse
import sys
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))


def setup_matplotlib_for_paper():
    """Setup matplotlib style for paper publication"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['lines.markersize'] = 8


def load_analysis_results(file_path: str) -> dict:
    """Load analysis results from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def create_track1_combined_plot(gpt_base_dir: str, claude_base_dir: str,
                                 output_path: str, strategy: str = 'cross_group'):
    """
    Create combined Track 1 plot with GPT and Claude side by side
    Showing Precision, Recall, and F1-Score metrics

    Args:
        gpt_base_dir: GPT experiment base directory
        claude_base_dir: Claude experiment base directory
        output_path: Output file path
        strategy: within_group or cross_group
    """
    logger.info("Creating Track 1 combined paper plot...")

    prompts = ['original', 'strong', 'weak']
    prompt_labels = {
        'original': 'Original',
        'strong': 'Strong',
        'weak': 'Weak'
    }

    # Metrics to plot
    metrics = ['precision', 'recall', 'f1_score']
    metric_labels = {
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1-Score'
    }

    # Classifier styles
    clf_styles = {
        'svm': {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o', 'label': 'SVM'},
        'random_forest': {'color': '#ff7f0e', 'linestyle': '--', 'marker': 's', 'label': 'Random Forest'}
    }

    # Create figure: 3 rows (prompts) × 6 cols (3 metrics × 2 models)
    fig, axes = plt.subplots(3, 6, figsize=(24, 12))

    for model_idx, (base_dir, model_name, exp_name) in enumerate([
        (gpt_base_dir, 'GPT-4.1-mini', 'full_ceas08_gpt41mini_v1'),
        (claude_base_dir, 'Claude-3.5-Haiku', 'full_ceas08_claude35haiku_v1')
    ]):

        for prompt_idx, prompt in enumerate(prompts):
            # Load data
            reports_dir = os.path.join(base_dir, 'reports')
            analysis_file = os.path.join(reports_dir,
                                        f'{exp_name}_{prompt}_{strategy}_statistical_analysis.json')

            if not os.path.exists(analysis_file):
                logger.warning(f"File not found: {analysis_file}")
                continue

            data = load_analysis_results(analysis_file)
            descriptive_stats = data.get('descriptive_statistics', {})

            for metric_idx, metric in enumerate(metrics):
                col_idx = model_idx * 3 + metric_idx
                ax = axes[prompt_idx, col_idx]

                # Plot each classifier
                for clf_name, style in clf_styles.items():
                    ratios = []
                    means = []
                    stds = []

                    for ratio_str in sorted(descriptive_stats.keys(), key=int):
                        ratio = int(ratio_str)
                        clf_data = descriptive_stats[ratio_str].get(clf_name, {})
                        metric_data = clf_data.get(metric, {})

                        if metric_data:
                            ratios.append(ratio)
                            means.append(metric_data['mean'])
                            stds.append(metric_data['std'])

                    if ratios:
                        # Plot line
                        ax.plot(ratios, means,
                               color=style['color'],
                               linestyle=style['linestyle'],
                               marker=style['marker'],
                               linewidth=2.5,
                               markersize=8,
                               label=style['label'],
                               alpha=0.9)

                        # Confidence band
                        means_array = np.array(means)
                        stds_array = np.array(stds)
                        ax.fill_between(ratios,
                                       means_array - stds_array,
                                       means_array + stds_array,
                                       color=style['color'],
                                       alpha=0.2)

                # Formatting
                ax.set_ylim([0, 1.1])
                ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
                ax.grid(True, alpha=0.3, linewidth=1.0)
                ax.set_axisbelow(True)

                # Column titles (only first row)
                if prompt_idx == 0:
                    if metric_idx == 0:
                        ax.set_title(f'({chr(97+model_idx)}) {model_name}\n{metric_labels[metric]}',
                                   fontsize=18, fontweight='bold', pad=15)
                    else:
                        ax.set_title(metric_labels[metric],
                                   fontsize=18, fontweight='bold', pad=15)

                # Row labels (only first column of each model)
                if metric_idx == 0:
                    ax.set_ylabel(f'{prompt_labels[prompt]}\n{metric_labels[metric]}',
                                fontsize=16, fontweight='bold')
                else:
                    ax.set_ylabel(metric_labels[metric], fontsize=14)

                # X-axis labels (only bottom row)
                if prompt_idx == 2:
                    ax.set_xlabel('Synthetic Ratio (%)', fontsize=14)

                # Legend (top-right subplot of each model, avoid overlapping)
                if prompt_idx == 0 and metric_idx == 2:
                    ax.legend(loc='lower left', fontsize=13, framealpha=0.95,
                             edgecolor='gray', fancybox=True, shadow=True)

    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate Track 1 paper plot')
    parser.add_argument('--gpt-dir', required=True,
                       help='GPT experiment base directory')
    parser.add_argument('--claude-dir', required=True,
                       help='Claude experiment base directory')
    parser.add_argument('--output', required=True,
                       help='Output file path')
    parser.add_argument('--strategy', default='cross_group',
                       choices=['cross_group', 'within_group'],
                       help='Mixing strategy')

    args = parser.parse_args()

    setup_matplotlib_for_paper()
    create_track1_combined_plot(args.gpt_dir, args.claude_dir,
                               args.output, args.strategy)


if __name__ == '__main__':
    main()
