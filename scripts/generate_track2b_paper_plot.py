#!/usr/bin/env python3
"""
Generate Track 2b paper plot: GPT and Claude side-by-side
Mixed-to-Synthetic Detection (Cross-Model Augmentation)
Showing Precision, Recall, and F1-Score
Reading directly from results files to avoid plotting_data.json issues
"""

import argparse
import os
import json
import glob
import statistics
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


def load_results_data(results_dir: str, method: str, prompt: str, strategy: str):
    """Load results directly from result files

    Args:
        results_dir: Directory containing result files
        method: gpt41mini or claude35haiku
        prompt: original, strong, or weak
        strategy: cross_group or within_group

    Returns:
        Dict with metrics data organized by classifier and ratio
    """
    data = {
        'svm': {},
        'random_forest': {}
    }

    # Find all result files for this configuration
    pattern = f"{method}_{prompt}_{strategy}_ratio*_reverse_results.json"
    files = glob.glob(os.path.join(results_dir, pattern))

    logger.info(f"Found {len(files)} files for {method}/{prompt}/{strategy}")

    for filepath in files:
        # Extract ratio from filename
        filename = os.path.basename(filepath)
        ratio_str = filename.split('_ratio')[1].split('_')[0]
        ratio = int(ratio_str)

        # Load file
        with open(filepath, 'r') as f:
            file_data = json.load(f)

        # Extract metrics for each classifier
        for clf_name in ['svm', 'random_forest']:
            metrics_list = {
                'precision': [],
                'recall': [],
                'f1_score': []
            }

            # Collect metrics from all groups
            for result in file_data['results']:
                if 'classifiers' in result and clf_name in result['classifiers']:
                    clf_metrics = result['classifiers'][clf_name]['metrics']
                    metrics_list['precision'].append(clf_metrics['precision'])
                    metrics_list['recall'].append(clf_metrics['recall'])
                    metrics_list['f1_score'].append(clf_metrics['f1_score'])

            # Calculate statistics
            if ratio not in data[clf_name]:
                data[clf_name][ratio] = {}

            for metric_name, values in metrics_list.items():
                if values:
                    data[clf_name][ratio][metric_name] = {
                        'mean': statistics.mean(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0
                    }

    return data


def create_track2b_combined_plot(results_dir: str, output_path: str,
                                  strategy: str = 'cross_group'):
    """
    Create combined Track 2b plot with GPT and Claude side by side
    Showing Precision, Recall, and F1-Score

    Args:
        results_dir: Directory containing result files
        output_path: Output file path
        strategy: within_group or cross_group
    """
    logger.info("Creating Track 2b combined paper plot...")

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

        for prompt_idx, prompt in enumerate(prompts):
            # Load data for this configuration
            data = load_results_data(results_dir, method, prompt, strategy)

            for metric_idx, metric in enumerate(metrics):
                col_idx = model_idx * 3 + metric_idx
                ax = axes[prompt_idx, col_idx]

                # Plot each classifier
                for clf_name, style in clf_styles.items():
                    if clf_name not in data:
                        continue

                    clf_data = data[clf_name]
                    ratios = sorted(clf_data.keys())

                    means = []
                    stds = []

                    for ratio in ratios:
                        if metric in clf_data[ratio]:
                            means.append(clf_data[ratio][metric]['mean'])
                            stds.append(clf_data[ratio][metric]['std'])
                        else:
                            means.append(np.nan)
                            stds.append(np.nan)

                    if means:
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
                    ax.set_xlabel('Training Synthetic Ratio (%)', fontsize=14)

                # Legend (F1-Score column of each model)
                if prompt_idx == 0 and metric_idx == 2:
                    # GPT legend: lower right
                    # Claude legend: center right to avoid overlapping
                    if model_idx == 0:
                        ax.legend(loc='lower right', fontsize=13, framealpha=0.95,
                                 edgecolor='gray', fancybox=True, shadow=True)
                    else:
                        ax.legend(loc='center right', fontsize=13, framealpha=0.95,
                                 edgecolor='gray', fancybox=True, shadow=True)

    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate Track 2b paper plot')
    parser.add_argument('--results-dir', default='output/reverse_detection/results',
                       help='Directory containing result files')
    parser.add_argument('--output', required=True,
                       help='Output file path')
    parser.add_argument('--strategy', default='cross_group',
                       choices=['cross_group', 'within_group'],
                       help='Mixing strategy')

    args = parser.parse_args()

    setup_matplotlib_for_paper()
    create_track2b_combined_plot(args.results_dir, args.output, args.strategy)


if __name__ == '__main__':
    main()
