#!/usr/bin/env python3
"""
Generate Track 2a paper plot: GPT and Claude side-by-side
Real-to-Synthetic Detection (Baseline)
Showing Precision, Recall, and F1-Score
Optimized for publication with larger fonts
"""

import argparse
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


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


def load_plotting_data(data_file: str) -> dict:
    """Load plotting data from JSON file"""
    with open(data_file, 'r') as f:
        return json.load(f)


def create_track2a_combined_plot(data_file: str, output_path: str,
                                  strategy: str = 'cross_group'):
    """
    Create combined Track 2a plot with GPT and Claude side by side
    Showing Precision, Recall, and F1-Score

    Args:
        data_file: Path to plotting data JSON
        output_path: Output file path
        strategy: within_group or cross_group
    """
    logger.info("Creating Track 2a combined paper plot...")

    plotting_data = load_plotting_data(data_file)

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

    for model_idx, (method, model_name) in enumerate([
        ('gpt41mini', 'GPT-4.1-mini'),
        ('claude35haiku', 'Claude-3.5-Haiku')
    ]):

        if method not in plotting_data:
            logger.warning(f"Method {method} not found in plotting data")
            continue

        for prompt_idx, prompt in enumerate(prompts):
            if prompt not in plotting_data[method]:
                logger.warning(f"Prompt {prompt} not found for {method}")
                continue

            prompt_data = plotting_data[method][prompt]
            if strategy not in prompt_data:
                logger.warning(f"Strategy {strategy} not found")
                continue

            strategy_data = prompt_data[strategy]

            for metric_idx, metric in enumerate(metrics):
                col_idx = model_idx * 3 + metric_idx
                ax = axes[prompt_idx, col_idx]

                # Plot each classifier
                for clf_name, style in clf_styles.items():
                    if clf_name not in strategy_data:
                        continue

                    clf_data = strategy_data[clf_name]
                    train_sizes = sorted([int(r) for r in clf_data.keys()])

                    means = []
                    stds = []

                    for size in train_sizes:
                        size_stats = clf_data[str(size)]
                        if metric in size_stats:
                            means.append(size_stats[metric]['mean'])
                            stds.append(size_stats[metric]['std'])
                        else:
                            means.append(np.nan)
                            stds.append(np.nan)

                    if means:
                        # Plot line
                        ax.plot(train_sizes, means,
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
                        ax.fill_between(train_sizes,
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
                        ax.set_title(f'({chr(97+model_idx)}) Detecting {model_name}\n{metric_labels[metric]}',
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
                    ax.set_xlabel('Training Real Spam Count', fontsize=14)

                # Legend (top-right subplot of each model)
                if prompt_idx == 0 and metric_idx == 2:
                    ax.legend(loc='lower right', fontsize=13, framealpha=0.95,
                             edgecolor='gray', fancybox=True, shadow=True)

    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate Track 2a paper plot')
    parser.add_argument('--data-file', required=True,
                       help='Path to plotting data JSON file')
    parser.add_argument('--output', required=True,
                       help='Output file path')
    parser.add_argument('--strategy', default='cross_group',
                       choices=['cross_group', 'within_group'],
                       help='Mixing strategy')

    args = parser.parse_args()

    setup_matplotlib_for_paper()
    create_track2a_combined_plot(args.data_file, args.output, args.strategy)


if __name__ == '__main__':
    main()
