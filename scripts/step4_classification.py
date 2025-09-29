#!/usr/bin/env python3
"""
Step 4: Classification Experiment Script
Responsible for running machine learning classification experiments
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
from src.config.config_manager import load_config
from src.classification.classifiers import ClassificationExperiment


def setup_logging(config):
    """Setup logging configuration"""
    log_config = config.logging
    log_file = os.path.join(
        config.output.get('logs_path', './logs/'),
        f"{config.name}_classification.log"
    )

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger.add(
        log_file,
        level=log_config.get('level', 'INFO'),
        format=log_config.get('format', "{time} | {level} | {message}"),
        rotation=log_config.get('rotation', "100 MB"),
        retention=log_config.get('retention', "30 days")
    )


def create_experiment_configs(
    config,
    strategy: str,
    synthetic_ratios: Optional[List[int]] = None,
    specific_ratio: Optional[int] = None,
    specific_group: Optional[int] = None,
    specific_trial: Optional[int] = None
) -> List[dict]:
    """Create experiment configuration list"""

    if synthetic_ratios is None:
        synthetic_ratios = config.synthetic_ratios

    if specific_ratio is not None:
        synthetic_ratios = [specific_ratio]

    experiment_configs = []

    for ratio in synthetic_ratios:
        groups = [specific_group] if specific_group is not None else range(config.n_groups)

        for group_id in groups:
            trials = [specific_trial] if specific_trial is not None else range(config.trials_per_config)

            for trial in trials:
                experiment_configs.append({
                    'strategy': strategy,
                    'synthetic_ratio': ratio,
                    'group_id': group_id,
                    'trial': trial
                })

    return experiment_configs


def main():
    parser = argparse.ArgumentParser(description='Classification experiment script')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Configuration file path'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='within_group',
        choices=['within_group', 'cross_group'],
        help='Mixing strategy'
    )
    parser.add_argument(
        '--synthetic_ratio',
        type=int,
        help='Specify single synthetic ratio for experiment'
    )
    parser.add_argument(
        '--group_id',
        type=int,
        help='Specify single group for experiment'
    )
    parser.add_argument(
        '--trial',
        type=int,
        help='Specify single trial for experiment'
    )
    parser.add_argument(
        '--classifiers',
        type=str,
        nargs='+',
        help='Specify classifiers to run'
    )
    parser.add_argument(
        '--datasets_dir',
        type=str,
        help='Datasets directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Output directory'
    )
    parser.add_argument(
        '--parallel_jobs',
        type=int,
        default=1,
        help='Number of parallel jobs'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume experiment from existing results'
    )

    args = parser.parse_args()

    try:
        # Load configuration
        logger.info("Loading configuration file")
        config = load_config(args.config)

        # Setup logging
        setup_logging(config)

        # Set parameters
        datasets_dir = args.datasets_dir or config.datasets_path
        output_dir = args.output_dir or config.output.get('results_path', './output/results/')

        # Filter classifiers
        if args.classifiers:
            enabled_classifiers = set(args.classifiers)
            for classifier_name in config.classifiers:
                if classifier_name not in enabled_classifiers:
                    config.classifiers[classifier_name]['enabled'] = False
                else:
                    config.classifiers[classifier_name]['enabled'] = True

        logger.info(f"Starting classification experiment")
        logger.info(f"Dataset: {config.dataset}")
        logger.info(f"Strategy: {args.strategy}")
        logger.info(f"Enabled classifiers: {[name for name, cfg in config.classifiers.items() if cfg.get('enabled', True)]}")

        # Create experiment configurations
        experiment_configs = create_experiment_configs(
            config,
            args.strategy,
            specific_ratio=args.synthetic_ratio,
            specific_group=args.group_id,
            specific_trial=args.trial
        )

        logger.info(f"Total number of experiments: {len(experiment_configs)}")

        # Check existing results
        results_file = os.path.join(output_dir, f"{config.name}_classification_results.json")
        existing_results = []

        if os.path.exists(results_file) and args.resume:
            logger.info(f"Loading existing results: {results_file}")
            import json
            with open(results_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                existing_results = existing_data.get('results', [])

            # Create set of completed experiments
            completed_experiments = set()
            for result in existing_results:
                exp_key = (
                    result['strategy'],
                    result['synthetic_ratio'],
                    result['group_id'],
                    result['trial']
                )
                completed_experiments.add(exp_key)

            # Filter out completed experiments
            original_count = len(experiment_configs)
            experiment_configs = [
                config for config in experiment_configs
                if (config['strategy'], config['synthetic_ratio'], config['group_id'], config['trial'])
                not in completed_experiments
            ]

            logger.info(f"Skipped completed experiments: {original_count - len(experiment_configs)}")
            logger.info(f"Remaining experiments: {len(experiment_configs)}")

        if not experiment_configs:
            logger.info("No experiments to run")
            return

        # Create classification experiment manager
        experiment = ClassificationExperiment(config)

        # Run experiments
        if args.parallel_jobs > 1:
            logger.info(f"Running experiments in parallel: {args.parallel_jobs} jobs")
            # TODO: Implement parallel execution
            batch_results = experiment.run_batch_experiments(experiment_configs, datasets_dir, output_dir)
        else:
            logger.info("Running experiments sequentially")
            batch_results = experiment.run_batch_experiments(experiment_configs, datasets_dir, output_dir)

        # If resuming, merge results
        if existing_results:
            logger.info("Merging existing results")
            batch_results['results'] = existing_results + batch_results['results']
            batch_results['summary']['total_experiments'] += len(existing_results)

            # Re-save merged results
            import json
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(batch_results, f, indent=2, ensure_ascii=False, default=str)

        # Output experiment results statistics
        logger.info("=" * 50)
        logger.info("Classification experiment completed")
        logger.info("=" * 50)

        summary = batch_results['summary']
        logger.info(f"Total experiments: {summary['total_experiments']}")
        logger.info(f"Successful experiments: {summary['successful_experiments']}")
        logger.info(f"Failed experiments: {summary['failed_experiments']}")
        logger.info(f"Success rate: {summary['successful_experiments'] / summary['total_experiments'] * 100:.1f}%")

        # Statistics of each classifier's performance
        if batch_results['results']:
            classifier_stats = {}
            for result in batch_results['results']:
                for classifier_name, classifier_result in result['classifiers'].items():
                    if classifier_result['success']:
                        if classifier_name not in classifier_stats:
                            classifier_stats[classifier_name] = []
                        classifier_stats[classifier_name].append(classifier_result['metrics']['f1_score'])

            logger.info("Average F1 scores by classifier:")
            for classifier_name, f1_scores in classifier_stats.items():
                avg_f1 = sum(f1_scores) / len(f1_scores)
                logger.info(f"  {classifier_name}: {avg_f1:.4f} (based on {len(f1_scores)} experiments)")

        logger.info(f"Results saved to: {summary['results_file']}")
        logger.success("Classification experiment completed successfully")

    except Exception as e:
        logger.error(f"Classification experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()