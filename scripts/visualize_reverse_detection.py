#!/usr/bin/env python3
"""
Reverse Detection Visualization Script
Generates plots for cross-model reverse detection experiment results
"""

import argparse
import sys
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger


def setup_matplotlib_style():
    """Setup matplotlib style"""
    plt.style.use('seaborn-v0_8')
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.figsize'] = [12, 8]
    sns.set_palette('husl')


def load_plotting_data(data_file: str) -> dict:
    """Load plotting data from JSON file

    Args:
        data_file: Path to plotting data JSON file

    Returns:
        Dictionary with plotting data
    """
    logger.info(f"Loading plotting data from: {data_file}")

    with open(data_file, 'r') as f:
        data = json.load(f)

    return data


def create_performance_curves_by_method(plotting_data: dict, output_dir: str):
    """Create performance curves for each method separately

    Args:
        plotting_data: Plotting data organized by method
        output_dir: Output directory for plots
    """
    logger.info("Creating performance curves by method...")

    # Metric order for ACM academic paper: accuracy, precision, recall, f1, auc-roc
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    metric_labels = {
        'f1_score': 'F1-Score',
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'auc_roc': 'AUC-ROC'
    }

    for method, method_data in plotting_data.items():
        method_display = 'GPT-4.1-mini' if method == 'gpt41mini' else 'Claude-3.5-Haiku'
        opposite_method = 'Claude-3.5-Haiku' if method == 'gpt41mini' else 'GPT-4.1-mini'

        for prompt, prompt_data in method_data.items():
            for strategy, strategy_data in prompt_data.items():
                logger.info(f"  {method} - {prompt} - {strategy}")

                # Create figure with 2 rows (classifiers) x 5 cols (metrics)
                fig, axes = plt.subplots(2, 5, figsize=(20, 8))

                classifiers = sorted(strategy_data.keys())
                if len(classifiers) != 2:
                    logger.warning(f"Expected 2 classifiers, found {len(classifiers)}")
                    continue

                for clf_idx, clf_name in enumerate(classifiers):
                    clf_display = clf_name.replace('_', ' ').upper()
                    clf_data = strategy_data[clf_name]

                    # Get sorted ratios
                    ratios = sorted([int(r) for r in clf_data.keys()])

                    for metric_idx, metric in enumerate(metrics):
                        ax = axes[clf_idx, metric_idx]

                        # Collect data for this metric
                        means = []
                        stds = []

                        for ratio in ratios:
                            ratio_stats = clf_data[str(ratio)]
                            if metric in ratio_stats:
                                means.append(ratio_stats[metric]['mean'])
                                stds.append(ratio_stats[metric]['std'])
                            else:
                                means.append(np.nan)
                                stds.append(np.nan)

                        # Plot
                        ax.errorbar(ratios, means, yerr=stds, marker='o',
                                   linewidth=2, markersize=6, capsize=5, capthick=2)

                        # Labels and title
                        if clf_idx == 0:
                            ax.set_title(metric_labels[metric], fontsize=11, fontweight='bold')

                        if metric_idx == 0:
                            ax.set_ylabel(f'{clf_display}\n{metric_labels[metric]}',
                                        fontsize=10, fontweight='bold')
                        else:
                            ax.set_ylabel(metric_labels[metric], fontsize=9)

                        if clf_idx == 1:
                            ax.set_xlabel('Training Ratio (%)', fontsize=9)

                        # Set y-axis to 0-1.1 range, but only show ticks up to 1.0
                        ax.set_ylim([0, 1.1])
                        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

                        # Set x-axis
                        ax.set_xticks(ratios)
                        ax.grid(True, alpha=0.3)

                # Overall title
                strategy_display = strategy.replace('_', '-').title()
                fig.suptitle(
                    f'Detecting AI-Generated Spam (Enhanced): {method_display} Testing '
                    f'(Trained on {opposite_method})\n'
                    f'Prompt: {prompt.capitalize()}, Strategy: {strategy_display}',
                    fontsize=14, fontweight='bold'
                )

                plt.tight_layout(rect=[0, 0, 1, 0.96])

                # Save
                filename = f'{method}_{prompt}_{strategy}_reverse_performance_curves.png'
                filepath = os.path.join(output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()

                logger.info(f"    Saved: {filename}")

    logger.info("Performance curves created successfully")


def create_prompt_comparison_plots(plotting_data: dict, output_dir: str):
    """Create combined prompt comparison plots (similar to forward detection)

    Args:
        plotting_data: Plotting data organized by method
        output_dir: Output directory for plots
    """
    logger.info("Creating prompt comparison plots...")

    prompts = ['original', 'strong', 'weak']
    # Metric order for ACM academic paper: accuracy, precision, recall, f1, auc-roc
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    metric_labels = {
        'f1_score': 'F1-Score',
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'auc_roc': 'AUC-ROC'
    }

    # Define classifier styles
    clf_styles = {
        'svm': {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o', 'label': 'SVM'},
        'random_forest': {'color': '#ff7f0e', 'linestyle': '--', 'marker': 's', 'label': 'Random Forest'}
    }

    for method, method_data in plotting_data.items():
        method_display = 'GPT-4.1-mini' if method == 'gpt41mini' else 'Claude-3.5-Haiku'
        opposite_method = 'Claude-3.5-Haiku' if method == 'gpt41mini' else 'GPT-4.1-mini'

        for strategy in ['within_group', 'cross_group']:
            logger.info(f"  {method} - {strategy}")

            # Create figure: 3 rows (prompts) x 5 cols (metrics)
            fig, axes = plt.subplots(3, 5, figsize=(20, 10))

            for prompt_idx, prompt in enumerate(prompts):
                if prompt not in method_data:
                    logger.warning(f"Prompt {prompt} not found in {method} data")
                    continue

                prompt_data = method_data[prompt]
                if strategy not in prompt_data:
                    logger.warning(f"Strategy {strategy} not found in {method}-{prompt} data")
                    continue

                strategy_data = prompt_data[strategy]

                for metric_idx, metric in enumerate(metrics):
                    ax = axes[prompt_idx, metric_idx]

                    # Plot each classifier
                    for clf_name, style in clf_styles.items():
                        if clf_name not in strategy_data:
                            continue

                        clf_data = strategy_data[clf_name]
                        ratios = sorted([int(r) for r in clf_data.keys()])

                        means = []
                        stds = []

                        for ratio in ratios:
                            ratio_stats = clf_data[str(ratio)]
                            if metric in ratio_stats:
                                means.append(ratio_stats[metric]['mean'])
                                stds.append(ratio_stats[metric]['std'])
                            else:
                                means.append(np.nan)
                                stds.append(np.nan)

                        # Plot line with error band
                        ax.plot(ratios, means,
                               color=style['color'],
                               linestyle=style['linestyle'],
                               marker=style['marker'],
                               linewidth=2,
                               markersize=5,
                               label=style['label'],
                               alpha=0.9)

                        # Add confidence band
                        means_array = np.array(means)
                        stds_array = np.array(stds)
                        ax.fill_between(ratios,
                                       means_array - stds_array,
                                       means_array + stds_array,
                                       color=style['color'],
                                       alpha=0.15)

                    # Formatting
                    if prompt_idx == 0:
                        ax.set_title(metric_labels[metric], fontsize=11, fontweight='bold')

                    if metric_idx == 0:
                        ax.set_ylabel(f'{prompt.capitalize()}\n{metric_labels[metric]}',
                                    fontsize=10, fontweight='bold')
                    else:
                        ax.set_ylabel(metric_labels[metric], fontsize=9)

                    if prompt_idx == 2:
                        ax.set_xlabel('Training Ratio (%)', fontsize=9)
                    else:
                        ax.set_xlabel('')

                    # Add legend only to top-right subplot
                    if prompt_idx == 0 and metric_idx == 4:
                        ax.legend(loc='lower right', fontsize=9, framealpha=0.9)

                    # Set y-axis range
                    ax.set_ylim([0, 1.1])
                    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

                    # Grid
                    ax.grid(True, alpha=0.3)
                    ax.set_axisbelow(True)

            # Overall title
            strategy_display = strategy.replace('_', '-').title()
            fig.suptitle(
                f'Detecting AI-Generated Spam (Enhanced): {method_display} Testing (Trained on {opposite_method})\n'
                f'Prompt Comparison - {strategy_display} Strategy',
                fontsize=14, fontweight='bold', y=0.995
            )

            plt.tight_layout(rect=[0, 0, 1, 0.99])

            # Save to combined folder
            combined_dir = os.path.join(output_dir, 'combined')
            os.makedirs(combined_dir, exist_ok=True)

            filename = f'{method}_prompts_comparison_{strategy}_reverse_performance_curves.png'
            filepath = os.path.join(combined_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"    Saved: {filename}")

    logger.info("Prompt comparison plots created successfully")


def create_cross_method_comparison(plotting_data: dict, output_dir: str):
    """Create comparison plots across both methods

    Args:
        plotting_data: Plotting data organized by method
        output_dir: Output directory for plots
    """
    logger.info("Creating cross-method comparison plots...")

    prompts = ['original', 'strong', 'weak']
    strategies = ['within_group', 'cross_group']
    metrics = ['f1_score', 'accuracy']

    metric_labels = {
        'f1_score': 'F1-Score',
        'accuracy': 'Accuracy'
    }

    # Method styles
    method_styles = {
        'gpt41mini': {'color': '#1f77b4', 'label': 'GPT-4.1-mini'},
        'claude35haiku': {'color': '#ff7f0e', 'label': 'Claude-3.5-Haiku'}
    }

    clf_styles = {
        'svm': {'linestyle': '-', 'marker': 'o'},
        'random_forest': {'linestyle': '--', 'marker': 's'}
    }

    for strategy in strategies:
        for prompt in prompts:
            logger.info(f"  {strategy} - {prompt}")

            # Create figure: 2 rows (classifiers) x 2 cols (metrics)
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            clf_names = ['svm', 'random_forest']

            for clf_idx, clf_name in enumerate(clf_names):
                clf_display = clf_name.replace('_', ' ').upper()

                for metric_idx, metric in enumerate(metrics):
                    ax = axes[clf_idx, metric_idx]

                    # Plot each method
                    for method, method_style in method_styles.items():
                        if method not in plotting_data:
                            continue
                        if prompt not in plotting_data[method]:
                            continue
                        if strategy not in plotting_data[method][prompt]:
                            continue
                        if clf_name not in plotting_data[method][prompt][strategy]:
                            continue

                        clf_data = plotting_data[method][prompt][strategy][clf_name]
                        ratios = sorted([int(r) for r in clf_data.keys()])

                        means = []
                        stds = []

                        for ratio in ratios:
                            ratio_stats = clf_data[str(ratio)]
                            if metric in ratio_stats:
                                means.append(ratio_stats[metric]['mean'])
                                stds.append(ratio_stats[metric]['std'])
                            else:
                                means.append(np.nan)
                                stds.append(np.nan)

                        # Plot
                        style = clf_styles[clf_name]
                        ax.plot(ratios, means,
                               color=method_style['color'],
                               linestyle=style['linestyle'],
                               marker=style['marker'],
                               linewidth=2,
                               markersize=6,
                               label=method_style['label'],
                               alpha=0.9)

                        # Confidence band
                        means_array = np.array(means)
                        stds_array = np.array(stds)
                        ax.fill_between(ratios,
                                       means_array - stds_array,
                                       means_array + stds_array,
                                       color=method_style['color'],
                                       alpha=0.15)

                    # Formatting
                    if clf_idx == 0:
                        ax.set_title(metric_labels[metric], fontsize=12, fontweight='bold')

                    ax.set_ylabel(f'{clf_display} - {metric_labels[metric]}',
                                fontsize=10, fontweight='bold')

                    if clf_idx == 1:
                        ax.set_xlabel('Training Ratio (%)', fontsize=10)

                    # Legend in top-right subplot
                    if clf_idx == 0 and metric_idx == 1:
                        ax.legend(loc='lower right', fontsize=9, framealpha=0.9)

                    # Y-axis range
                    ax.set_ylim([0, 1.1])
                    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

                    ax.grid(True, alpha=0.3)

            # Overall title
            strategy_display = strategy.replace('_', '-').title()
            fig.suptitle(
                f'Detecting AI-Generated Spam (Enhanced): Method Comparison\n'
                f'Prompt: {prompt.capitalize()}, Strategy: {strategy_display}',
                fontsize=14, fontweight='bold'
            )

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            # Save
            comparison_dir = os.path.join(output_dir, 'comparison')
            os.makedirs(comparison_dir, exist_ok=True)

            filename = f'method_comparison_{prompt}_{strategy}_reverse.png'
            filepath = os.path.join(comparison_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"    Saved: {filename}")

    logger.info("Cross-method comparison plots created successfully")


def main():
    parser = argparse.ArgumentParser(description='Visualize reverse detection results')
    parser.add_argument(
        '--data_file',
        type=str,
        default='output/reverse_detection/analysis/plotting_data.json',
        help='Path to plotting data JSON file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output/reverse_detection/plots',
        help='Output directory for plots'
    )
    parser.add_argument(
        '--plot_types',
        nargs='+',
        default=['individual', 'combined', 'comparison'],
        choices=['individual', 'combined', 'comparison'],
        help='Types of plots to generate'
    )

    args = parser.parse_args()

    try:
        # Setup matplotlib
        setup_matplotlib_style()

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Reverse Detection Visualization")
        logger.info("=" * 60)

        # Load plotting data
        plotting_data = load_plotting_data(args.data_file)

        # Generate plots
        if 'individual' in args.plot_types:
            logger.info("\nGenerating individual performance curves...")
            create_performance_curves_by_method(plotting_data, args.output_dir)

        if 'combined' in args.plot_types:
            logger.info("\nGenerating prompt comparison plots...")
            create_prompt_comparison_plots(plotting_data, args.output_dir)

        if 'comparison' in args.plot_types:
            logger.info("\nGenerating cross-method comparison plots...")
            create_cross_method_comparison(plotting_data, args.output_dir)

        logger.success("\nVisualization completed successfully!")
        logger.info(f"Plots saved to: {args.output_dir}")

        # Count generated files
        total_files = 0
        for root, dirs, files in os.walk(args.output_dir):
            total_files += len([f for f in files if f.endswith('.png')])

        logger.info(f"Total plots generated: {total_files}")

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        raise


if __name__ == "__main__":
    main()
