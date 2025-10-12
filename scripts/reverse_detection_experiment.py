#!/usr/bin/env python3
"""
Reverse Detection Experiment
Tests whether classifiers trained on real data can detect LLM-generated synthetic spam

Training: 0% synthetic (pure real data)
Testing: 100% synthetic spam + real ham
"""

import argparse
import sys
import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
from src.classification.classifiers import (
    SVMClassifier,
    RandomForestClassifier_Custom,
    FeatureExtractor
)
from src.config.config_manager import ExperimentConfig, load_config


class ReverseDetectionExperiment:
    """Reverse Detection Experiment Manager"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.feature_extractor = FeatureExtractor(config)

    def load_baseline_training_data(self, group_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load pure real training data (0% synthetic)

        Args:
            group_id: Group ID (0-19)

        Returns:
            Tuple of (train_data, train_labels)
        """
        baseline_path = Path('output/prepared_data/groups') / f'group_{group_id}' / 'train.pkl'

        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline training data not found: {baseline_path}")

        logger.info(f"Loading baseline training data from: {baseline_path}")
        with open(baseline_path, 'rb') as f:
            train_data = pickle.load(f)

        logger.info(f"Loaded {len(train_data)} training samples")
        logger.info(f"Spam ratio: {(train_data['label'] == 1).mean():.2%}")

        return train_data

    def load_synthetic_spam(
        self,
        method: str,
        prompt: Optional[str],
        strategy: str,
        group_id: int
    ) -> pd.DataFrame:
        """
        Load 100% synthetic spam samples

        Args:
            method: 'gpt41mini' | 'claude35haiku' | 'smote'
            prompt: 'original' | 'strong' | 'weak' (None for SMOTE)
            strategy: 'within_group' | 'cross_group'
            group_id: Group ID (0-19)

        Returns:
            DataFrame with synthetic spam samples
        """
        # Construct path based on method
        if method == 'smote':
            dataset_dir = Path('output/full_experiments/ceas08_smote/datasets')
            subdir = f'smote_{strategy}_100'
        else:
            dataset_dir = Path(f'output/full_experiments/ceas08_{method}/datasets')
            subdir = f'{prompt}_{strategy}_100'

        dataset_path = dataset_dir / subdir / f'group_{group_id}_train.pkl'

        if not dataset_path.exists():
            raise FileNotFoundError(f"Synthetic dataset not found: {dataset_path}")

        logger.info(f"Loading synthetic spam from: {dataset_path}")
        with open(dataset_path, 'rb') as f:
            synthetic_data = pickle.load(f)

        # Extract only spam samples (label == 1)
        synthetic_spam = synthetic_data[synthetic_data['label'] == 1].copy()

        logger.info(f"Extracted {len(synthetic_spam)} synthetic spam samples")

        return synthetic_spam

    def load_real_ham(self, group_id: int) -> pd.DataFrame:
        """
        Load real ham samples from test set

        Args:
            group_id: Group ID (0-19)

        Returns:
            DataFrame with real ham samples
        """
        test_path = Path('output/prepared_data/groups') / f'group_{group_id}' / 'test.pkl'

        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")

        logger.info(f"Loading real ham from: {test_path}")
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)

        # Extract only ham samples (label == 0)
        real_ham = test_data[test_data['label'] == 0].copy()

        logger.info(f"Extracted {len(real_ham)} real ham samples")

        return real_ham

    def construct_reverse_test_set(
        self,
        synthetic_spam: pd.DataFrame,
        real_ham: pd.DataFrame,
        target_spam_ratio: float = 0.10
    ) -> pd.DataFrame:
        """
        Construct test set: synthetic spam + real ham

        Args:
            synthetic_spam: Synthetic spam samples
            real_ham: Real ham samples
            target_spam_ratio: Target spam ratio (default 0.10)

        Returns:
            Combined test set DataFrame
        """
        n_spam = len(synthetic_spam)
        n_ham_needed = int(n_spam / target_spam_ratio) - n_spam

        # Sample ham to match target ratio
        if n_ham_needed > len(real_ham):
            logger.warning(
                f"Not enough ham samples. Needed: {n_ham_needed}, Available: {len(real_ham)}"
            )
            n_ham_needed = len(real_ham)

        sampled_ham = real_ham.sample(n=n_ham_needed, random_state=42)

        # Combine
        test_set = pd.concat([synthetic_spam, sampled_ham], ignore_index=True)

        # Shuffle
        test_set = test_set.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(f"Constructed reverse test set:")
        logger.info(f"  Total samples: {len(test_set)}")
        logger.info(f"  Synthetic spam: {n_spam} ({n_spam/len(test_set):.2%})")
        logger.info(f"  Real ham: {len(sampled_ham)} ({len(sampled_ham)/len(test_set):.2%})")

        return test_set

    def train_and_evaluate(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        classifier_type: str
    ) -> Dict[str, Any]:
        """
        Train classifier on real data and evaluate on reverse test set

        Args:
            train_data: Training data (pure real)
            test_data: Test data (synthetic spam + real ham)
            classifier_type: 'svm' | 'random_forest'

        Returns:
            Dict with evaluation metrics
        """
        logger.info(f"Training {classifier_type} classifier")

        # Initialize classifier
        if classifier_type == 'svm':
            classifier = SVMClassifier(self.config)
        elif classifier_type == 'random_forest':
            classifier = RandomForestClassifier_Custom(self.config)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        # Prepare training data
        X_train = train_data[['subject', 'body']].copy()
        y_train = train_data['label']

        # Train
        classifier.fit(X_train, y_train)

        # Prepare test data
        X_test = test_data[['subject', 'body']].copy()
        y_test = test_data['label']

        # Evaluate
        metrics = classifier.evaluate(X_test, y_test)

        logger.info(f"{classifier_type} F1-score: {metrics['f1_score']:.4f}")

        return {
            'classifier': classifier_type,
            'metrics': metrics,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'test_spam_count': int((y_test == 1).sum()),
            'test_ham_count': int((y_test == 0).sum()),
            'test_spam_ratio': float((y_test == 1).mean())
        }

    def run_experiment(
        self,
        method: str,
        prompt: Optional[str],
        strategy: str,
        group_id: int,
        classifiers: List[str]
    ) -> Dict[str, Any]:
        """
        Run complete reverse detection experiment for one configuration

        Args:
            method: Synthetic generation method
            prompt: Prompt strategy (None for SMOTE)
            strategy: Mixing strategy
            group_id: Group ID
            classifiers: List of classifier types to run

        Returns:
            Dict with experiment results
        """
        logger.info("=" * 60)
        logger.info(f"Running Reverse Detection Experiment")
        logger.info(f"Method: {method}, Prompt: {prompt}, Strategy: {strategy}, Group: {group_id}")
        logger.info("=" * 60)

        try:
            # Step 1: Load baseline training data (pure real)
            train_data = self.load_baseline_training_data(group_id)

            # Step 2: Load synthetic spam (100%)
            synthetic_spam = self.load_synthetic_spam(method, prompt, strategy, group_id)

            # Step 3: Load real ham
            real_ham = self.load_real_ham(group_id)

            # Step 4: Construct reverse test set
            test_data = self.construct_reverse_test_set(synthetic_spam, real_ham)

            # Step 5: Train and evaluate classifiers
            results = {
                'method': method,
                'prompt': prompt if prompt else 'N/A',
                'strategy': strategy,
                'group_id': group_id,
                'classifiers': {},
                'success': True,
                'error': None
            }

            for classifier_type in classifiers:
                try:
                    classifier_result = self.train_and_evaluate(
                        train_data, test_data, classifier_type
                    )
                    results['classifiers'][classifier_type] = classifier_result
                except Exception as e:
                    logger.error(f"Classifier {classifier_type} failed: {e}")
                    results['classifiers'][classifier_type] = {
                        'classifier': classifier_type,
                        'metrics': None,
                        'error': str(e)
                    }

            logger.success(f"Experiment completed for group {group_id}")
            return results

        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            return {
                'method': method,
                'prompt': prompt if prompt else 'N/A',
                'strategy': strategy,
                'group_id': group_id,
                'classifiers': {},
                'success': False,
                'error': str(e)
            }


def main():
    parser = argparse.ArgumentParser(
        description='Reverse Detection Experiment: Train on real, test on synthetic'
    )
    parser.add_argument(
        '--method',
        type=str,
        required=True,
        choices=['gpt41mini', 'claude35haiku', 'smote'],
        help='Synthetic generation method'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        choices=['original', 'strong', 'weak'],
        help='Prompt strategy (not applicable for SMOTE)'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        required=True,
        choices=['within_group', 'cross_group'],
        help='Mixing strategy'
    )
    parser.add_argument(
        '--group_id',
        type=int,
        help='Specific group ID (0-19). If not specified, run all groups'
    )
    parser.add_argument(
        '--classifiers',
        type=str,
        nargs='+',
        default=['svm', 'random_forest'],
        choices=['svm', 'random_forest'],
        help='Classifiers to run (default: both)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/ceas08_gpt41mini_config.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output/reverse_detection',
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.method != 'smote' and not args.prompt:
        parser.error(f"--prompt is required for method '{args.method}'")

    if args.method == 'smote' and args.prompt:
        logger.warning("--prompt is ignored for SMOTE method")
        args.prompt = None

    # Setup logging
    log_dir = Path(args.output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'reverse_detection.log'

    logger.add(
        log_file,
        level='INFO',
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        rotation="100 MB"
    )

    logger.info("Starting Reverse Detection Experiment")
    logger.info(f"Method: {args.method}")
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Classifiers: {args.classifiers}")

    try:
        # Load config
        config = load_config(args.config)

        # Initialize experiment
        experiment = ReverseDetectionExperiment(config)

        # Determine groups to run
        if args.group_id is not None:
            groups = [args.group_id]
        else:
            groups = range(config.n_groups)  # Default: 20 groups

        # Run experiments
        all_results = []

        for group_id in groups:
            result = experiment.run_experiment(
                method=args.method,
                prompt=args.prompt,
                strategy=args.strategy,
                group_id=group_id,
                classifiers=args.classifiers
            )
            all_results.append(result)

        # Save results
        results_dir = Path(args.output_dir) / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)

        prompt_str = f"{args.prompt}_" if args.prompt else ""
        results_file = results_dir / f"{args.method}_{prompt_str}{args.strategy}_reverse_results.json"

        results_data = {
            'experiment': 'reverse_detection',
            'method': args.method,
            'prompt': args.prompt,
            'strategy': args.strategy,
            'n_groups': len(groups),
            'classifiers': args.classifiers,
            'results': all_results
        }

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False, default=str)

        logger.success(f"Results saved to: {results_file}")

        # Print summary
        successful = sum(1 for r in all_results if r['success'])
        failed = len(all_results) - successful

        logger.info("=" * 60)
        logger.info("Experiment Summary")
        logger.info("=" * 60)
        logger.info(f"Total experiments: {len(all_results)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")

        # Compute average F1 scores
        for classifier_type in args.classifiers:
            f1_scores = []
            for result in all_results:
                if result['success'] and classifier_type in result['classifiers']:
                    classifier_result = result['classifiers'][classifier_type]
                    if classifier_result.get('metrics'):
                        f1_scores.append(classifier_result['metrics']['f1_score'])

            if f1_scores:
                mean_f1 = np.mean(f1_scores)
                std_f1 = np.std(f1_scores)
                logger.info(f"{classifier_type}: F1 = {mean_f1:.4f} ± {std_f1:.4f}")

        logger.success("Reverse Detection Experiment completed successfully!")

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
