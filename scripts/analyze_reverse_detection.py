#!/usr/bin/env python3
"""
Reverse Detection Analysis Script
Analyzes cross-model reverse detection experiment results
"""

import argparse
import sys
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger


def load_reverse_detection_results(results_dir: str) -> List[Dict[str, Any]]:
    """Load all reverse detection result files

    Args:
        results_dir: Directory containing result JSON files

    Returns:
        List of all results from all files
    """
    all_results = []

    result_files = [f for f in os.listdir(results_dir) if f.endswith('_reverse_results.json')]
    logger.info(f"Found {len(result_files)} result files")

    for result_file in sorted(result_files):
        file_path = os.path.join(results_dir, result_file)
        logger.info(f"Loading: {result_file}")

        with open(file_path, 'r') as f:
            data = json.load(f)

        # Add file metadata to each result
        for result in data['results']:
            result['testing_method'] = data['testing_method']
            result['testing_prompt'] = data['testing_prompt']
            result['testing_strategy'] = data['testing_strategy']
            result['training_source'] = data['training_synthetic_source']
            result['training_ratio_pct'] = data['training_synthetic_ratio']

        all_results.extend(data['results'])

    logger.info(f"Loaded {len(all_results)} total experiments")
    return all_results


def organize_results_by_config(results: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
    """Organize results by experimental configuration

    Args:
        results: List of all experimental results

    Returns:
        Dictionary organized by configuration
    """
    organized = defaultdict(lambda: defaultdict(list))

    for result in results:
        # Create configuration key
        config_key = (
            result['testing_method'],
            result['testing_prompt'],
            result['testing_strategy'],
            result['training_ratio_pct']
        )

        # Store by classifier
        for clf_name, clf_data in result['classifiers'].items():
            organized[config_key][clf_name].append(clf_data['metrics'])

    return organized


def compute_statistics(organized_results: Dict) -> Dict[str, Any]:
    """Compute descriptive statistics for each configuration

    Args:
        organized_results: Results organized by configuration

    Returns:
        Statistics for each configuration
    """
    statistics = {}

    for config_key, classifiers in organized_results.items():
        testing_method, testing_prompt, testing_strategy, training_ratio = config_key

        config_id = f"{testing_method}_{testing_prompt}_{testing_strategy}_ratio{training_ratio}"
        statistics[config_id] = {
            'testing_method': testing_method,
            'testing_prompt': testing_prompt,
            'testing_strategy': testing_strategy,
            'training_ratio': training_ratio,
            'classifiers': {}
        }

        for clf_name, metrics_list in classifiers.items():
            clf_stats = {}

            # Get all metric names from first experiment
            if metrics_list:
                metric_names = metrics_list[0].keys()

                for metric_name in metric_names:
                    values = [m[metric_name] for m in metrics_list]

                    clf_stats[metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values, ddof=1),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values),
                        'n_groups': len(values)
                    }

            statistics[config_id]['classifiers'][clf_name] = clf_stats

    return statistics


def create_comparison_table(statistics: Dict[str, Any]) -> pd.DataFrame:
    """Create comparison table for all configurations

    Args:
        statistics: Statistics for all configurations

    Returns:
        DataFrame with comparison data
    """
    rows = []

    for config_id, config_data in statistics.items():
        for clf_name, clf_stats in config_data['classifiers'].items():
            row = {
                'Testing_Method': config_data['testing_method'],
                'Testing_Prompt': config_data['testing_prompt'],
                'Testing_Strategy': config_data['testing_strategy'],
                'Training_Ratio': config_data['training_ratio'],
                'Classifier': clf_name,
            }

            # Add metric means
            for metric_name, metric_stats in clf_stats.items():
                row[f'{metric_name}_mean'] = metric_stats['mean']
                row[f'{metric_name}_std'] = metric_stats['std']

            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def generate_summary_report(statistics: Dict[str, Any], output_dir: str):
    """Generate summary report

    Args:
        statistics: Statistics for all configurations
        output_dir: Output directory for report
    """
    report = {
        'experiment_type': 'cross_model_reverse_detection',
        'total_configurations': len(statistics),
        'configurations': statistics,
        'summary': {
            'testing_methods': set(),
            'testing_prompts': set(),
            'testing_strategies': set(),
            'training_ratios': set()
        }
    }

    # Collect unique values
    for config_data in statistics.values():
        report['summary']['testing_methods'].add(config_data['testing_method'])
        report['summary']['testing_prompts'].add(config_data['testing_prompt'])
        report['summary']['testing_strategies'].add(config_data['testing_strategy'])
        report['summary']['training_ratios'].add(config_data['training_ratio'])

    # Convert sets to sorted lists
    for key in report['summary']:
        report['summary'][key] = sorted(list(report['summary'][key]))

    # Save report
    report_file = os.path.join(output_dir, 'reverse_detection_summary.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Summary report saved to: {report_file}")
    return report


def create_method_comparison_by_ratio(statistics: Dict[str, Any]) -> Dict[str, Any]:
    """Create comparison data organized by method and ratio for plotting

    Args:
        statistics: Statistics for all configurations

    Returns:
        Dictionary organized by method, prompt, strategy for plotting
    """
    comparison_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for config_id, config_data in statistics.items():
        method = config_data['testing_method']
        prompt = config_data['testing_prompt']
        strategy = config_data['testing_strategy']
        ratio = config_data['training_ratio']

        for clf_name, clf_stats in config_data['classifiers'].items():
            if clf_name not in comparison_data[method][prompt][strategy]:
                comparison_data[method][prompt][strategy][clf_name] = {}

            comparison_data[method][prompt][strategy][clf_name][ratio] = clf_stats

    return comparison_data


def main():
    parser = argparse.ArgumentParser(description='Analyze reverse detection experiment results')
    parser.add_argument(
        '--results_dir',
        type=str,
        default='output/reverse_detection/results',
        help='Directory containing reverse detection results'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output/reverse_detection/analysis',
        help='Output directory for analysis results'
    )

    args = parser.parse_args()

    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Reverse Detection Analysis")
        logger.info("=" * 60)

        # Load results
        logger.info("Loading results...")
        results = load_reverse_detection_results(args.results_dir)

        # Organize results
        logger.info("Organizing results...")
        organized_results = organize_results_by_config(results)
        logger.info(f"Organized into {len(organized_results)} configurations")

        # Compute statistics
        logger.info("Computing statistics...")
        statistics = compute_statistics(organized_results)

        # Create comparison table
        logger.info("Creating comparison table...")
        comparison_df = create_comparison_table(statistics)
        comparison_csv = os.path.join(args.output_dir, 'reverse_detection_comparison.csv')
        comparison_df.to_csv(comparison_csv, index=False)
        logger.info(f"Comparison table saved to: {comparison_csv}")

        # Generate summary report
        logger.info("Generating summary report...")
        summary_report = generate_summary_report(statistics, args.output_dir)

        # Create method comparison data for plotting
        logger.info("Creating plotting data...")
        comparison_data = create_method_comparison_by_ratio(statistics)
        plotting_data_file = os.path.join(args.output_dir, 'plotting_data.json')
        with open(plotting_data_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        logger.info(f"Plotting data saved to: {plotting_data_file}")

        # Print summary
        logger.info("=" * 60)
        logger.info("Analysis Summary")
        logger.info("=" * 60)
        logger.info(f"Total configurations analyzed: {len(statistics)}")
        logger.info(f"Testing methods: {summary_report['summary']['testing_methods']}")
        logger.info(f"Testing prompts: {summary_report['summary']['testing_prompts']}")
        logger.info(f"Testing strategies: {summary_report['summary']['testing_strategies']}")
        logger.info(f"Training ratios: {summary_report['summary']['training_ratios']}")

        # Display sample results
        logger.info("\nSample Results (GPT-4.1-mini, Original, Within-Group, Ratio=50%):")
        sample_key = "gpt41mini_original_within_group_ratio50"
        if sample_key in statistics:
            sample_data = statistics[sample_key]
            for clf_name, clf_stats in sample_data['classifiers'].items():
                logger.info(f"\n  {clf_name.upper()}:")
                for metric, stats in clf_stats.items():
                    logger.info(f"    {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")

        logger.success("\nReverse detection analysis completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
