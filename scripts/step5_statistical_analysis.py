#!/usr/bin/env python3
"""
Step 5: Statistical Analysis Script
Responsible for performing statistical analysis on classification experiment results
"""

import argparse
import sys
import os
from pathlib import Path

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
from src.config.config_manager import load_config
from src.analysis.statistical_analysis import StatisticalAnalyzer


def setup_logging(config):
    """Setup logging configuration"""
    log_config = config.logging
    log_file = os.path.join(
        config.output.get('logs_path', './logs/'),
        f"{config.name}_statistical_analysis.log"
    )

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger.add(
        log_file,
        level=log_config.get('level', 'INFO'),
        format=log_config.get('format', "{time} | {level} | {message}"),
        rotation=log_config.get('rotation', "100 MB"),
        retention=log_config.get('retention', "30 days")
    )


def main():
    parser = argparse.ArgumentParser(description='Statistical analysis script')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='配置文件路径'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        help='Experiment name (for finding result files)'
    )
    parser.add_argument(
        '--results_file',
        type=str,
        help='Classification results file path (overrides default path)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Output directory'
    )
    parser.add_argument(
        '--analysis_types',
        type=str,
        nargs='+',
        default=['descriptive', 'hypothesis', 'degradation'],
        choices=['descriptive', 'hypothesis', 'degradation', 'prompt_comparison'],
        help='Analysis types to execute'
    )

    args = parser.parse_args()

    try:
        # Load configuration
        logger.info("Loading configuration file")
        config = load_config(args.config)

        # Setup logging
        setup_logging(config)

        # Determine results file path
        if args.results_file:
            results_file = args.results_file
        else:
            experiment_name = args.experiment_name or config.name
            results_file = os.path.join(
                config.output.get('results_path', './output/results/'),
                f"{experiment_name}_classification_results.json"
            )

        output_dir = args.output_dir or config.output.get('results_path', './output/results/')

        logger.info(f"Starting statistical analysis")
        logger.info(f"Results file: {results_file}")
        logger.info(f"Analysis types: {args.analysis_types}")

        # Check if results file exists
        if not os.path.exists(results_file):
            logger.error(f"Results file does not exist: {results_file}")
            return

        # Create statistical analyzer
        analyzer = StatisticalAnalyzer(config)

        # Load results
        results = analyzer.load_classification_results(results_file)

        # Organize results
        organized_results = analyzer.organize_results_by_ratio(results)

        logger.info(f"Analyzing data: {len(organized_results)} synthetic ratios")

        # Execute various analyses
        analysis_results = {}

        if 'descriptive' in args.analysis_types:
            logger.info("Executing descriptive statistical analysis")
            analysis_results['descriptive_statistics'] = analyzer.compute_descriptive_statistics(organized_results)

        if 'hypothesis' in args.analysis_types:
            logger.info("Executing hypothesis testing")
            analysis_results['hypothesis_tests'] = analyzer.perform_hypothesis_tests(organized_results)

        if 'degradation' in args.analysis_types:
            logger.info("Executing performance degradation analysis")
            analysis_results['performance_degradation'] = analyzer.analyze_performance_degradation(organized_results)

        if 'prompt_comparison' in args.analysis_types:
            logger.info("Executing prompt strategy comparison")
            # TODO: Need to load results from multiple prompts for comparison
            logger.warning("Prompt comparison analysis requires results from multiple prompts, currently skipping")

        # Generate comprehensive report
        logger.info("Generating comprehensive statistical analysis report")
        comprehensive_report = analyzer.generate_comprehensive_report(results_file, output_dir)

        # Output key findings
        logger.info("=" * 50)
        logger.info("Statistical analysis completed")
        logger.info("=" * 50)

        findings = comprehensive_report['summary_findings']

        # Display baseline performance
        if findings['baseline_performance']:
            logger.info("Baseline performance (synthetic ratio = 0%):")
            for classifier, metrics in findings['baseline_performance'].items():
                f1_score = metrics.get('f1_score', 'N/A')
                logger.info(f"  {classifier}: F1-score = {f1_score:.4f}" if isinstance(f1_score, float) else f"  {classifier}: F1-score = {f1_score}")

        # Display significant performance degradations
        if findings['significant_degradations']:
            logger.info(f"Found {len(findings['significant_degradations'])} significant performance degradations:")
            for degradation in findings['significant_degradations'][:5]:  # Display top 5
                logger.info(
                    f"  Ratio {degradation['ratio']}%, {degradation['classifier']}, "
                    f"{degradation['metric']}: degradation {degradation['degradation']:.4f}, "
                    f"p-value = {degradation['p_value']:.4f}"
                )

        # Display critical ratios
        if findings['critical_ratios']:
            logger.info("Performance degradation critical ratios:")
            for classifier, metrics in findings['critical_ratios'].items():
                for metric, critical_ratio in metrics.items():
                    logger.info(f"  {classifier} ({metric}): {critical_ratio}%")

        # Display best synthetic ratios
        if findings['best_synthetic_ratios']:
            logger.info("Best synthetic ratio (minimum performance degradation):")
            for classifier, metrics in findings['best_synthetic_ratios'].items():
                for metric, best_info in metrics.items():
                    logger.info(
                        f"  {classifier} ({metric}): {best_info['ratio']}% "
                        f"(degradation: {best_info['degradation']:.4f})"
                    )

        # Display file location
        logger.info(f"Detailed analysis report: {comprehensive_report['report_file']}")

        # Display statistical test configuration
        eval_config = config.evaluation.get('statistical_tests', {})
        logger.info(f"Statistical significance level: α = {eval_config.get('alpha', 0.05)}")
        logger.info(f"Multiple comparison correction method: {eval_config.get('multiple_comparison_method', 'fdr_bh')}")

        logger.success("Statistical analysis completed successfully")

    except Exception as e:
        logger.error(f"Statistical analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()