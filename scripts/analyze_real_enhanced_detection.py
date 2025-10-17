#!/usr/bin/env python3
"""
Real-Enhanced Detection Analysis Script
Analyzes real-enhanced detection experiment results where training uses increased real spam
to detect synthetic spam
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


def load_real_enhanced_results(results_dir: str) -> List[Dict[str, Any]]:
    """
    Load all real-enhanced detection result files

    Args:
        results_dir: Directory containing result JSON files

    Returns:
        List of all results from all files
    """
    all_results = []

    result_files = [f for f in os.listdir(results_dir) if f.endswith('_real_enhanced_results.json')]
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
            result['real_spam_count'] = data['real_spam_count']

        all_results.extend(data['results'])

    logger.info(f"Loaded {len(all_results)} total experiments")
    return all_results


def organize_results_by_config(results: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
    """
    Organize results by experimental configuration

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
            result['real_spam_count']
        )

        # Store by classifier
        for clf_name, clf_data in result['classifiers'].items():
            organized[config_key][clf_name].append(clf_data['metrics'])

    return organized


def compute_statistics(organized_results: Dict) -> Dict[str, Any]:
    """
    Compute descriptive statistics for each configuration

    Args:
        organized_results: Results organized by configuration

    Returns:
        Statistics for each configuration
    """
    statistics = {}

    for config_key, classifiers in organized_results.items():
        testing_method, testing_prompt, testing_strategy, real_spam_count = config_key

        config_id = f"{testing_method}_{testing_prompt}_{testing_strategy}_count{real_spam_count}"
        statistics[config_id] = {
            'testing_method': testing_method,
            'testing_prompt': testing_prompt,
            'testing_strategy': testing_strategy,
            'real_spam_count': real_spam_count,
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
    """
    Create comparison table for all configurations

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
                'Real_Spam_Count': config_data['real_spam_count'],
                'Classifier': clf_name,
            }

            # Add metric means and stds
            for metric_name, metric_stats in clf_stats.items():
                row[f'{metric_name}_mean'] = metric_stats['mean']
                row[f'{metric_name}_std'] = metric_stats['std']

            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def generate_summary_report(statistics: Dict[str, Any], output_dir: str):
    """
    Generate summary report

    Args:
        statistics: Statistics for all configurations
        output_dir: Output directory for report
    """
    report = {
        'experiment_type': 'real_enhanced_detection',
        'total_configurations': len(statistics),
        'configurations': statistics,
        'summary': {
            'testing_methods': set(),
            'testing_prompts': set(),
            'testing_strategies': set(),
            'real_spam_counts': set()
        }
    }

    # Collect unique values
    for config_data in statistics.values():
        report['summary']['testing_methods'].add(config_data['testing_method'])
        report['summary']['testing_prompts'].add(config_data['testing_prompt'])
        report['summary']['testing_strategies'].add(config_data['testing_strategy'])
        report['summary']['real_spam_counts'].add(config_data['real_spam_count'])

    # Convert sets to sorted lists
    for key in report['summary']:
        report['summary'][key] = sorted(list(report['summary'][key]))

    # Save report
    report_file = os.path.join(output_dir, 'real_enhanced_summary.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Summary report saved to: {report_file}")
    return report


def create_method_comparison_by_count(statistics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create comparison data organized by method and real spam count for plotting

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
        count = config_data['real_spam_count']

        for clf_name, clf_stats in config_data['classifiers'].items():
            if clf_name not in comparison_data[method][prompt][strategy]:
                comparison_data[method][prompt][strategy][clf_name] = {}

            comparison_data[method][prompt][strategy][clf_name][count] = clf_stats

    return comparison_data


def compute_improvement_analysis(statistics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute improvement analysis: how much does performance improve from 100 -> 200 -> 300?

    Args:
        statistics: Statistics for all configurations

    Returns:
        Improvement analysis data
    """
    improvement_data = defaultdict(lambda: defaultdict(dict))

    # Group by method, prompt, strategy, classifier
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for config_id, config_data in statistics.items():
        method = config_data['testing_method']
        prompt = config_data['testing_prompt']
        strategy = config_data['testing_strategy']
        count = config_data['real_spam_count']

        key = (method, prompt, strategy)

        for clf_name, clf_stats in config_data['classifiers'].items():
            grouped[key][clf_name][count] = clf_stats

    # Compute improvements
    for key, classifiers in grouped.items():
        method, prompt, strategy = key

        for clf_name, counts_data in classifiers.items():
            if 100 in counts_data and 200 in counts_data and 300 in counts_data:
                baseline_f1 = counts_data[100]['f1_score']['mean']
                count200_f1 = counts_data[200]['f1_score']['mean']
                count300_f1 = counts_data[300]['f1_score']['mean']

                improvement_200 = count200_f1 - baseline_f1
                improvement_300 = count300_f1 - baseline_f1

                improvement_key = f"{method}_{prompt}_{strategy}_{clf_name}"
                improvement_data[improvement_key] = {
                    'method': method,
                    'prompt': prompt,
                    'strategy': strategy,
                    'classifier': clf_name,
                    'baseline_f1': baseline_f1,
                    'count200_f1': count200_f1,
                    'count300_f1': count300_f1,
                    'improvement_200': improvement_200,
                    'improvement_300': improvement_300,
                    'improvement_200_pct': (improvement_200 / baseline_f1) * 100 if baseline_f1 > 0 else 0,
                    'improvement_300_pct': (improvement_300 / baseline_f1) * 100 if baseline_f1 > 0 else 0
                }

    return improvement_data


def create_improvement_table(improvement_data: Dict[str, Any], output_dir: str):
    """
    Create improvement analysis table

    Args:
        improvement_data: Improvement analysis data
        output_dir: Output directory
    """
    rows = []

    for key, data in improvement_data.items():
        rows.append({
            'Method': data['method'],
            'Prompt': data['prompt'],
            'Strategy': data['strategy'],
            'Classifier': data['classifier'],
            'Baseline_F1': data['baseline_f1'],
            'Count200_F1': data['count200_f1'],
            'Count300_F1': data['count300_f1'],
            'Improvement_200': data['improvement_200'],
            'Improvement_300': data['improvement_300'],
            'Improvement_200_Pct': data['improvement_200_pct'],
            'Improvement_300_Pct': data['improvement_300_pct']
        })

    df = pd.DataFrame(rows)

    # Sort by improvement_300_pct descending
    df = df.sort_values('Improvement_300_Pct', ascending=False)

    improvement_csv = os.path.join(output_dir, 'real_enhanced_improvement_analysis.csv')
    df.to_csv(improvement_csv, index=False)

    logger.info(f"Improvement analysis saved to: {improvement_csv}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Analyze real-enhanced detection experiment results')
    parser.add_argument(
        '--results_dir',
        type=str,
        default='output/real_enhanced_detection/results',
        help='Directory containing real-enhanced detection results'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output/real_enhanced_detection/analysis',
        help='Output directory for analysis results'
    )

    args = parser.parse_args()

    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Real-Enhanced Detection Analysis")
        logger.info("=" * 60)

        # Load results
        logger.info("Loading results...")
        results = load_real_enhanced_results(args.results_dir)

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
        comparison_csv = os.path.join(args.output_dir, 'real_enhanced_comparison.csv')
        comparison_df.to_csv(comparison_csv, index=False)
        logger.info(f"Comparison table saved to: {comparison_csv}")

        # Generate summary report
        logger.info("Generating summary report...")
        summary_report = generate_summary_report(statistics, args.output_dir)

        # Create method comparison data for plotting
        logger.info("Creating plotting data...")
        comparison_data = create_method_comparison_by_count(statistics)
        plotting_data_file = os.path.join(args.output_dir, 'plotting_data.json')
        with open(plotting_data_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        logger.info(f"Plotting data saved to: {plotting_data_file}")

        # Compute improvement analysis
        logger.info("Computing improvement analysis...")
        improvement_data = compute_improvement_analysis(statistics)
        improvement_df = create_improvement_table(improvement_data, args.output_dir)

        # Print summary
        logger.info("=" * 60)
        logger.info("Analysis Summary")
        logger.info("=" * 60)
        logger.info(f"Total configurations analyzed: {len(statistics)}")
        logger.info(f"Testing methods: {summary_report['summary']['testing_methods']}")
        logger.info(f"Testing prompts: {summary_report['summary']['testing_prompts']}")
        logger.info(f"Testing strategies: {summary_report['summary']['testing_strategies']}")
        logger.info(f"Real spam counts: {summary_report['summary']['real_spam_counts']}")

        # Display sample results
        logger.info("\nSample Results (GPT-4.1-mini, Original, Within-Group, Count=100):")
        sample_key = "gpt41mini_original_within_group_count100"
        if sample_key in statistics:
            sample_data = statistics[sample_key]
            for clf_name, clf_stats in sample_data['classifiers'].items():
                logger.info(f"\n  {clf_name.upper()}:")
                for metric, stats in clf_stats.items():
                    logger.info(f"    {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")

        # Display top improvements
        logger.info("\n" + "=" * 60)
        logger.info("Top 10 Improvements (100 -> 300 spam count)")
        logger.info("=" * 60)
        for idx, row in improvement_df.head(10).iterrows():
            logger.info(f"{row['Method']:15s} {row['Prompt']:8s} {row['Strategy']:12s} {row['Classifier']:15s}: "
                       f"F1 {row['Baseline_F1']:.4f} -> {row['Count300_F1']:.4f} "
                       f"(+{row['Improvement_300_Pct']:.2f}%)")

        logger.success("\nReal-enhanced detection analysis completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
