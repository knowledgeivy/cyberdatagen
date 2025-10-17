#!/usr/bin/env python3
"""
Real-Enhanced Detection Experiment
Evaluates whether increasing real spam training data improves synthetic spam detection

Training: Real ham (900) + Real spam (100, 150, or 200 samples)
Testing: Real ham (900) + Synthetic spam (100% from target LLM)
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
from src.classification.classifiers import SVMClassifier, RandomForestClassifier_Custom
from src.config.config_manager import ExperimentConfig, load_config


class RealEnhancedDetectionExperiment:
    """Real-Enhanced Detection Experiment Manager"""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def load_baseline_training(self, strategy: str, group_id: int, trial: int = 0) -> pd.DataFrame:
        """
        Load baseline (r0 = 100% real) training data

        Args:
            strategy: 'within_group' or 'cross_group'
            group_id: Group ID (0-19)
            trial: Trial number (default 0)

        Returns:
            DataFrame with baseline training data
        """
        # Use r0 = 100% real data (any method works since r0 is the same)
        dataset_path = f"data/full_experiments/ceas08_gpt41mini/datasets/{strategy}/train_original_r0_g{group_id}_t{trial}.csv.gz"

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Baseline training data not found: {dataset_path}")

        logger.info(f"Loading baseline training from: {dataset_path}")
        data = pd.read_csv(dataset_path, compression='gzip')

        logger.info(f"Loaded {len(data)} samples - Spam: {(data['label'] == 1).sum()}, Ham: {(data['label'] == 0).sum()}")

        return data

    def load_test_set(self, strategy: str) -> pd.DataFrame:
        """
        Load shared test set

        Args:
            strategy: 'within_group' or 'cross_group'

        Returns:
            DataFrame with test set
        """
        # Test set is shared across all methods
        test_path = f"data/full_experiments/ceas08_gpt41mini/datasets/{strategy}/test_set.csv.gz"

        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test set not found: {test_path}")

        logger.info(f"Loading test set from: {test_path}")
        data = pd.read_csv(test_path, compression='gzip')

        logger.info(f"Test set - Spam: {(data['label'] == 1).sum()}, Ham: {(data['label'] == 0).sum()}")

        return data

    def load_additional_real_spam(
        self,
        strategy: str,
        group_id: int,
        count: int,
        trial: int = 0
    ) -> pd.DataFrame:
        """
        Load additional real spam from broader corpus
        Uses real spam from other groups to augment training

        Args:
            strategy: 'within_group' or 'cross_group'
            group_id: Current group ID (to exclude)
            count: Total number of real spam samples needed (150 or 200)
            trial: Trial number

        Returns:
            DataFrame with additional real spam samples
        """
        logger.info(f"Loading {count} real spam samples (excluding group {group_id})")

        # Collect real spam from other groups
        all_real_spam = []

        for other_group_id in range(20):  # Assuming 20 groups total
            if other_group_id == group_id:
                continue  # Skip current group to maintain independence

            try:
                dataset_path = f"data/full_experiments/ceas08_gpt41mini/datasets/{strategy}/train_original_r0_g{other_group_id}_t{trial}.csv.gz"

                if os.path.exists(dataset_path):
                    data = pd.read_csv(dataset_path, compression='gzip')
                    real_spam = data[data['label'] == 1].copy()
                    all_real_spam.append(real_spam)
            except Exception as e:
                logger.warning(f"Failed to load group {other_group_id}: {e}")
                continue

        if not all_real_spam:
            raise ValueError("No additional real spam samples found")

        # Combine all real spam
        combined_spam = pd.concat(all_real_spam, ignore_index=True)

        logger.info(f"Collected {len(combined_spam)} real spam samples from {len(all_real_spam)} groups")

        # Sample the required count
        if len(combined_spam) < count:
            logger.warning(f"Not enough real spam samples. Need {count}, have {len(combined_spam)}. Using all available.")
            sampled_spam = combined_spam
        else:
            sampled_spam = combined_spam.sample(n=count, random_state=42)

        logger.info(f"Sampled {len(sampled_spam)} real spam samples for training")

        return sampled_spam

    def construct_training_set(
        self,
        baseline_train: pd.DataFrame,
        real_spam_count: int,
        strategy: str,
        group_id: int,
        trial: int = 0
    ) -> pd.DataFrame:
        """
        Construct training set with increased real spam

        Args:
            baseline_train: Baseline real training data
            real_spam_count: Number of real spam samples (100, 150, or 200)
            strategy: Mixing strategy
            group_id: Group ID
            trial: Trial number

        Returns:
            Training DataFrame
        """
        real_ham = baseline_train[baseline_train['label'] == 0].copy()
        baseline_real_spam = baseline_train[baseline_train['label'] == 1].copy()

        if real_spam_count == 100:
            # Use baseline 100 real spam
            real_spam = baseline_real_spam
            logger.info(f"Training set (100 spam): {len(real_spam)} real spam (baseline)")

        elif real_spam_count == 150 or real_spam_count == 200:
            # Load additional real spam from broader corpus
            additional_spam = self.load_additional_real_spam(
                strategy, group_id, real_spam_count, trial
            )
            real_spam = additional_spam
            logger.info(f"Training set ({real_spam_count} spam): {len(real_spam)} real spam (from broader corpus)")

        else:
            raise ValueError(f"Invalid real_spam_count: {real_spam_count}. Must be 100, 150, or 200")

        # Combine spam and ham
        train_set = pd.concat([real_spam, real_ham], ignore_index=True)
        train_set = train_set.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(f"Training set: {len(train_set)} samples ({len(real_spam)} spam + {len(real_ham)} ham, ratio={(len(real_spam)/len(train_set)):.2%})")

        return train_set

    def load_synthetic_spam(
        self,
        method: str,
        prompt: str,
        strategy: str,
        group_id: int,
        trial: int = 0
    ) -> pd.DataFrame:
        """
        Load synthetic spam from r100 (100% synthetic) datasets for testing

        Args:
            method: 'gpt41mini' or 'claude35haiku'
            prompt: 'original', 'strong', or 'weak'
            strategy: 'within_group' or 'cross_group'
            group_id: Group ID (0-19)
            trial: Trial number (default 0)

        Returns:
            DataFrame with synthetic spam samples only
        """
        dataset_path = f"data/full_experiments/ceas08_{method}/datasets/{strategy}/train_{prompt}_r100_g{group_id}_t{trial}.csv.gz"

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Synthetic dataset not found: {dataset_path}")

        logger.info(f"Loading synthetic spam from: {dataset_path}")
        data = pd.read_csv(dataset_path, compression='gzip')

        # Extract only spam samples (label == 1)
        synthetic_spam = data[data['label'] == 1].copy()

        logger.info(f"Extracted {len(synthetic_spam)} synthetic spam samples from {method}")

        return synthetic_spam

    def construct_testing_set(
        self,
        testing_synthetic_spam: pd.DataFrame,
        test_set: pd.DataFrame,
        target_spam_ratio: float = 0.10
    ) -> pd.DataFrame:
        """
        Construct testing set: 100% synthetic spam + real ham from test set

        Args:
            testing_synthetic_spam: Synthetic spam for testing (from target model, r100)
            test_set: Shared test set
            target_spam_ratio: Target spam ratio (default 0.10)

        Returns:
            Testing DataFrame
        """
        n_spam = len(testing_synthetic_spam)
        n_ham_needed = int(n_spam / target_spam_ratio) - n_spam

        # Get real ham from test set
        real_ham = test_set[test_set['label'] == 0].copy()

        if n_ham_needed > len(real_ham):
            logger.warning(f"Not enough ham samples. Needed: {n_ham_needed}, Available: {len(real_ham)}")
            n_ham_needed = len(real_ham)

        sampled_ham = real_ham.sample(n=n_ham_needed, random_state=42)

        # Combine and shuffle
        test_data = pd.concat([testing_synthetic_spam, sampled_ham], ignore_index=True)
        test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(f"Testing set: {len(test_data)} samples ({n_spam} synthetic spam + {len(sampled_ham)} ham, ratio={(n_spam/len(test_data)):.2%})")

        return test_data

    def train_and_evaluate(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        classifier_type: str
    ) -> Dict[str, Any]:
        """
        Train classifier and evaluate

        Args:
            train_data: Training data
            test_data: Testing data
            classifier_type: 'svm' or 'random_forest'

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

        # Prepare data
        X_train = train_data[['subject', 'body']].copy()
        y_train = train_data['label']
        X_test = test_data[['subject', 'body']].copy()
        y_test = test_data['label']

        # Train
        classifier.fit(X_train, y_train)

        # Evaluate
        metrics = classifier.evaluate(X_test, y_test)

        logger.info(f"{classifier_type} F1-score: {metrics['f1_score']:.4f}")

        return {
            'classifier': classifier_type,
            'metrics': metrics,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'train_spam_count': int((y_train == 1).sum()),
            'train_ham_count': int((y_train == 0).sum()),
            'test_synthetic_spam_count': int((y_test == 1).sum()),
            'test_ham_count': int((y_test == 0).sum()),
            'test_spam_ratio': float((y_test == 1).mean())
        }

    def run_experiment(
        self,
        testing_method: str,
        testing_prompt: str,
        testing_strategy: str,
        real_spam_count: int,
        group_id: int,
        classifiers: List[str],
        trial: int = 0
    ) -> Dict[str, Any]:
        """
        Run complete real-enhanced detection experiment

        Args:
            testing_method: Target model for testing ('gpt41mini' or 'claude35haiku')
            testing_prompt: Prompt strategy for testing
            testing_strategy: Mixing strategy for testing
            real_spam_count: Number of real spam samples in training (100, 150, or 200)
            group_id: Group ID
            classifiers: List of classifier types
            trial: Trial number (default 0)

        Returns:
            Dict with experiment results
        """
        logger.info("=" * 60)
        logger.info(f"Real-Enhanced Detection Experiment")
        logger.info(f"Testing: {testing_method}-{testing_prompt}-{testing_strategy}")
        logger.info(f"Real Spam Count: {real_spam_count}")
        logger.info(f"Group: {group_id}, Trial: {trial}")
        logger.info("=" * 60)

        try:
            # Step 1: Load baseline training data (r0 = 100% real)
            baseline_train = self.load_baseline_training(testing_strategy, group_id, trial)

            # Step 2: Construct training set with increased real spam
            train_data = self.construct_training_set(
                baseline_train, real_spam_count, testing_strategy, group_id, trial
            )

            # Step 3: Load testing synthetic spam (from target model, r100)
            testing_synthetic_spam = self.load_synthetic_spam(
                testing_method, testing_prompt, testing_strategy, group_id, trial
            )

            # Step 4: Load shared test set
            test_set = self.load_test_set(testing_strategy)

            # Step 5: Construct testing set (100% synthetic spam + real ham)
            test_data = self.construct_testing_set(testing_synthetic_spam, test_set)

            # Step 6: Train and evaluate classifiers
            results = {
                'testing_method': testing_method,
                'testing_prompt': testing_prompt,
                'testing_strategy': testing_strategy,
                'real_spam_count': real_spam_count,
                'group_id': group_id,
                'trial': trial,
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
            import traceback
            traceback.print_exc()
            return {
                'testing_method': testing_method,
                'testing_prompt': testing_prompt,
                'testing_strategy': testing_strategy,
                'real_spam_count': real_spam_count,
                'group_id': group_id,
                'trial': trial,
                'classifiers': {},
                'success': False,
                'error': str(e)
            }


def main():
    parser = argparse.ArgumentParser(
        description='Real-Enhanced Detection: Train on increased real spam, Test on synthetic spam'
    )
    parser.add_argument(
        '--testing_method',
        type=str,
        required=True,
        choices=['gpt41mini', 'claude35haiku'],
        help='Testing target model'
    )
    parser.add_argument(
        '--testing_prompt',
        type=str,
        required=True,
        choices=['original', 'strong', 'weak'],
        help='Prompt strategy'
    )
    parser.add_argument(
        '--testing_strategy',
        type=str,
        required=True,
        choices=['within_group', 'cross_group'],
        help='Mixing strategy'
    )
    parser.add_argument(
        '--real_spam_count',
        type=int,
        required=True,
        choices=[100, 150, 200],
        help='Number of real spam samples in training (100, 150, or 200)'
    )
    parser.add_argument(
        '--group_id',
        type=int,
        help='Specific group ID (0-19). If not specified, run all groups'
    )
    parser.add_argument(
        '--trial',
        type=int,
        default=0,
        help='Trial number (default 0)'
    )
    parser.add_argument(
        '--classifiers',
        type=str,
        nargs='+',
        default=['svm'],
        choices=['svm', 'random_forest'],
        help='Classifiers to run (default: svm only)'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Configuration file path'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output/real_enhanced_detection',
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Setup logging
    log_dir = Path(args.output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'real_enhanced_detection.log'

    logger.add(
        log_file,
        level='INFO',
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        rotation="100 MB"
    )

    logger.info("Starting Real-Enhanced Detection Experiment")
    logger.info(f"Testing Method: {args.testing_method}")
    logger.info(f"Testing Prompt: {args.testing_prompt}")
    logger.info(f"Testing Strategy: {args.testing_strategy}")
    logger.info(f"Real Spam Count: {args.real_spam_count}")
    logger.info(f"Classifiers: {args.classifiers}")

    try:
        # Load config
        config = load_config(args.config)

        # Initialize experiment
        experiment = RealEnhancedDetectionExperiment(config)

        # Determine groups to run
        if args.group_id is not None:
            groups = [args.group_id]
        else:
            groups = range(20)  # Default: 20 groups

        # Run experiments
        all_results = []

        for group_id in groups:
            result = experiment.run_experiment(
                testing_method=args.testing_method,
                testing_prompt=args.testing_prompt,
                testing_strategy=args.testing_strategy,
                real_spam_count=args.real_spam_count,
                group_id=group_id,
                classifiers=args.classifiers,
                trial=args.trial
            )
            all_results.append(result)

        # Save results
        results_dir = Path(args.output_dir) / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)

        results_file = results_dir / f"{args.testing_method}_{args.testing_prompt}_{args.testing_strategy}_count{args.real_spam_count}_real_enhanced_results.json"

        results_data = {
            'experiment': 'real_enhanced_detection',
            'testing_method': args.testing_method,
            'testing_prompt': args.testing_prompt,
            'testing_strategy': args.testing_strategy,
            'real_spam_count': args.real_spam_count,
            'training_type': 'pure_real_spam',
            'testing_synthetic_ratio': 100,
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

        logger.success("Real-Enhanced Detection Experiment completed successfully!")

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
