#!/usr/bin/env python3
"""
SMOTE Baseline Experiments Runner
Generates SMOTE-based synthetic features and runs classification experiments
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import pickle
from typing import Dict, Any
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
from src.config.config_manager import load_config
from src.traditional_methods import SMOTEGenerator
from src.classification.classification_pipeline import ClassificationPipeline
import numpy as np


def setup_logging(config):
    """Setup logging"""
    log_config = config.logging
    log_file = os.path.join(
        config.output.get('logs_path', './logs/'),
        f"{config.name}_smote_experiments.log"
    )

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger.add(
        log_file,
        level=log_config.get('level', 'INFO'),
        format=log_config.get('format', "{time} | {level} | {message}"),
        rotation=log_config.get('rotation', "100 MB"),
        retention=log_config.get('retention', "30 days")
    )


def load_processed_data(config, group_id: int):
    """Load processed data for a specific group"""
    processed_path = Path(config.data.get('processed_data_path'))

    # Find the processed file
    processed_files = list(processed_path.glob(f"*_processed.csv.gz"))
    if not processed_files:
        raise FileNotFoundError(f"No processed data found in {processed_path}")

    processed_file = processed_files[0]
    logger.info(f"Loading processed data from {processed_file}")

    df = pd.read_csv(processed_file, compression='gzip')

    # Filter by group
    group_df = df[df['group_id'] == group_id].copy()

    # Split into train and test
    test_df = group_df[group_df['split'] == 'test'].copy()
    train_df = group_df[group_df['split'] == 'train'].copy()

    logger.info(f"Group {group_id}: {len(train_df)} train, {len(test_df)} test samples")

    return train_df, test_df


def generate_smote_datasets(
    config,
    smote_variant: str,
    strategy: str,
    group_id: int
):
    """Generate SMOTE-based datasets for all synthetic ratios"""
    logger.info(f"Generating SMOTE datasets: variant={smote_variant}, strategy={strategy}, group={group_id}")

    # Load data
    train_df, test_df = load_processed_data(config, group_id)

    # Initialize SMOTE generator
    smote_params = config.smote.get(f'{smote_variant}_params', {})
    vectorizer_params = config.smote.get('vectorizer_params', {})

    generator = SMOTEGenerator(
        method=smote_variant,
        vectorizer_params=vectorizer_params,
        smote_params=smote_params,
        random_state=config.experiment.get('random_seed', 42)
    )

    # Prepare output directory
    datasets_path = Path(config.data.get('datasets_path'))
    variant_dir = datasets_path / f"{smote_variant}_{strategy}"
    variant_dir.mkdir(parents=True, exist_ok=True)

    # Generate datasets for each synthetic ratio
    synthetic_ratios = config.data.get('synthetic_ratios', [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    datasets = {}

    for ratio in synthetic_ratios:
        logger.info(f"Generating dataset for ratio {ratio}%")

        # Select source data based on strategy
        if strategy == 'within_group':
            source_df = train_df.copy()
        elif strategy == 'cross_group':
            # For cross_group, use different group (simulate by shuffling group labels)
            all_train = []
            for gid in range(config.data.get('n_groups', 20)):
                if gid != group_id:
                    temp_train, _ = load_processed_data(config, gid)
                    all_train.append(temp_train)
            if all_train:
                source_df = pd.concat(all_train, ignore_index=True)
                # Sample same amount as train_df
                if len(source_df) > len(train_df):
                    source_df = source_df.sample(n=len(train_df), random_state=42)
            else:
                source_df = train_df.copy()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Build training dataset with SMOTE
        X_train, y_train, metadata = generator.build_training_dataset(
            real_texts=source_df['text'],
            real_labels=source_df['label'],
            synthetic_ratio=ratio,
            strategy=strategy
        )

        # Get vectorizer for test set
        vectorizer = generator.get_vectorizer()
        X_test = vectorizer.transform(test_df['text'])
        y_test = test_df['label'].values

        # Save dataset
        dataset = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'metadata': metadata,
            'test_ids': test_df['id'].tolist() if 'id' in test_df.columns else None
        }

        dataset_file = variant_dir / f"group_{group_id}_ratio_{ratio}.pkl"
        with open(dataset_file, 'wb') as f:
            pickle.dump(dataset, f)

        logger.info(f"Saved dataset: {dataset_file}")
        datasets[ratio] = dataset

    logger.success(f"Generated {len(datasets)} datasets for group {group_id}")
    return datasets


def run_classification_experiments(
    config,
    smote_variant: str,
    strategy: str,
    group_id: int,
    resume: bool = True
):
    """Run classification experiments on SMOTE datasets"""
    logger.info(f"Running classification: variant={smote_variant}, strategy={strategy}, group={group_id}")

    # Load datasets
    datasets_path = Path(config.data.get('datasets_path'))
    variant_dir = datasets_path / f"{smote_variant}_{strategy}"

    if not variant_dir.exists():
        logger.error(f"Datasets directory not found: {variant_dir}")
        return None

    # Prepare results
    results = {
        'experiment_name': config.name,
        'smote_variant': smote_variant,
        'strategy': strategy,
        'group_id': group_id,
        'results': []
    }

    # Get classifiers
    classifiers_config = config.classifiers
    enabled_classifiers = {
        name: cfg for name, cfg in classifiers_config.items()
        if cfg.get('enabled', True)
    }

    # Run experiments for each ratio
    synthetic_ratios = config.data.get('synthetic_ratios', [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    for ratio in synthetic_ratios:
        dataset_file = variant_dir / f"group_{group_id}_ratio_{ratio}.pkl"

        if not dataset_file.exists():
            logger.warning(f"Dataset file not found: {dataset_file}")
            continue

        # Load dataset
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)

        X_train = dataset['X_train']
        y_train = dataset['y_train']
        X_test = dataset['X_test']
        y_test = dataset['y_test']

        logger.info(f"Ratio {ratio}%: Training shape {X_train.shape}, Test shape {X_test.shape}")

        # Train each classifier
        for clf_name, clf_config in enabled_classifiers.items():
            logger.info(f"Training {clf_name} classifier")

            try:
                # Initialize pipeline
                pipeline = ClassificationPipeline(
                    classifier_name=clf_name,
                    classifier_params=clf_config.get('params', {}),
                    random_state=config.experiment.get('random_seed', 42)
                )

                # Train (features already extracted by SMOTE)
                pipeline.classifier.fit(X_train, y_train)

                # Evaluate
                metrics = pipeline.evaluate(X_test, y_test)

                # Record results
                result = {
                    'group_id': group_id,
                    'synthetic_ratio': ratio,
                    'method': smote_variant,
                    'strategy': strategy,
                    'classifier': clf_name,
                    'metrics': metrics,
                    'metadata': dataset.get('metadata', {}),
                    'success': True
                }

                results['results'].append(result)
                logger.success(f"✓ {clf_name} @ ratio {ratio}%: F1={metrics['f1_score']:.4f}")

            except Exception as e:
                logger.error(f"✗ {clf_name} @ ratio {ratio}% failed: {e}")
                result = {
                    'group_id': group_id,
                    'synthetic_ratio': ratio,
                    'method': smote_variant,
                    'strategy': strategy,
                    'classifier': clf_name,
                    'success': False,
                    'error': str(e)
                }
                results['results'].append(result)

    # Save results
    results_path = Path(config.output.get('results_path'))
    results_path.mkdir(parents=True, exist_ok=True)

    results_file = results_path / f"{config.name}_{smote_variant}_{strategy}_group{group_id}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.success(f"Results saved: {results_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description='SMOTE Baseline Experiments')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Configuration file path'
    )
    parser.add_argument(
        '--variant',
        type=str,
        default='smote',
        choices=['smote', 'adasyn'],
        help='SMOTE variant to use'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        required=True,
        choices=['within_group', 'cross_group'],
        help='Data mixing strategy'
    )
    parser.add_argument(
        '--group_id',
        type=int,
        required=True,
        help='Group ID to process'
    )
    parser.add_argument(
        '--skip_generation',
        action='store_true',
        help='Skip dataset generation (use existing)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        default=True,
        help='Resume from existing results'
    )

    args = parser.parse_args()

    try:
        # Load configuration
        logger.info("Loading configuration")
        config = load_config(args.config)

        # Setup logging
        setup_logging(config)

        logger.info("=" * 60)
        logger.info(f"SMOTE Baseline Experiment: {config.name}")
        logger.info(f"Variant: {args.variant}")
        logger.info(f"Strategy: {args.strategy}")
        logger.info(f"Group: {args.group_id}")
        logger.info("=" * 60)

        # Step 1: Generate SMOTE datasets
        if not args.skip_generation:
            logger.info("[Step 1] Generating SMOTE datasets...")
            generate_smote_datasets(
                config=config,
                smote_variant=args.variant,
                strategy=args.strategy,
                group_id=args.group_id
            )
        else:
            logger.info("[Step 1] Skipping dataset generation (using existing)")

        # Step 2: Run classification experiments
        logger.info("[Step 2] Running classification experiments...")
        results = run_classification_experiments(
            config=config,
            smote_variant=args.variant,
            strategy=args.strategy,
            group_id=args.group_id,
            resume=args.resume
        )

        if results:
            successful = sum(1 for r in results['results'] if r.get('success', False))
            total = len(results['results'])
            logger.success(f"Completed: {successful}/{total} experiments successful")

        logger.success("=" * 60)
        logger.success("SMOTE baseline experiment completed!")
        logger.success("=" * 60)

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()
