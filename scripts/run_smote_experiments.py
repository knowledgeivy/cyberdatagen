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
import yaml
from types import SimpleNamespace
from src.traditional_methods import SMOTEGenerator
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score
)
import numpy as np


def dict_to_namespace(d):
    """Convert dict to namespace recursively"""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d


def load_smote_config(config_path):
    """Load SMOTE config directly from YAML (bypass LLM validation)"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return dict_to_namespace(config_dict)


def setup_logging(config):
    """Setup logging"""
    log_config = config.logging
    log_file = os.path.join(
        config.output.logs_path,
        f"{config.experiment.name}_smote_experiments.log"
    )

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger.add(
        log_file,
        level=log_config.level,
        format=log_config.format,
        rotation=log_config.rotation,
        retention=log_config.retention
    )


def load_processed_data(config, group_id: int):
    """Load processed data for a specific group"""
    processed_path = Path(config.data.processed_data_path)

    # Find the processed file
    processed_files = list(processed_path.glob(f"*_processed.csv.gz"))
    if not processed_files:
        raise FileNotFoundError(f"No processed data found in {processed_path}")

    processed_file = processed_files[0]
    logger.info(f"Loading processed data from {processed_file}")

    df = pd.read_csv(processed_file, compression='gzip')

    # Filter by group
    group_df = df[df['group_id'] == group_id].copy()

    # Create 'text' column by combining subject and body
    if 'text' not in group_df.columns:
        group_df['text'] = group_df['subject'].fillna('') + ' ' + group_df['body'].fillna('')

    # Handle split: if no split column or all NaN, split manually
    if 'split' not in group_df.columns or group_df['split'].isna().all():
        # Manual split: 80% train, 20% test
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(
            group_df.index,
            test_size=config.data.test_split,
            random_state=config.experiment.random_seed,
            stratify=group_df['label']
        )
        train_df = group_df.loc[train_idx].copy()
        test_df = group_df.loc[test_idx].copy()
    else:
        # Use existing split
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
    smote_params = getattr(config.smote, f'{smote_variant}_params', {})
    if isinstance(smote_params, SimpleNamespace):
        smote_params = vars(smote_params)

    vectorizer_params = config.smote.vectorizer_params
    if isinstance(vectorizer_params, SimpleNamespace):
        vectorizer_params = vars(vectorizer_params)

    # Convert list to tuple for ngram_range (sklearn requirement)
    if 'ngram_range' in vectorizer_params and isinstance(vectorizer_params['ngram_range'], list):
        vectorizer_params['ngram_range'] = tuple(vectorizer_params['ngram_range'])

    generator = SMOTEGenerator(
        method=smote_variant,
        vectorizer_params=vectorizer_params,
        smote_params=smote_params,
        random_state=config.experiment.random_seed
    )

    # Prepare output directory
    datasets_path = Path(config.data.datasets_path)
    variant_dir = datasets_path / f"{smote_variant}_{strategy}"
    variant_dir.mkdir(parents=True, exist_ok=True)

    # Generate datasets for each synthetic ratio
    synthetic_ratios = config.data.synthetic_ratios

    datasets = {}

    for ratio in synthetic_ratios:
        logger.info(f"Generating dataset for ratio {ratio}%")

        # Select source data based on strategy
        if strategy == 'within_group':
            source_df = train_df.copy()
        elif strategy == 'cross_group':
            # For cross_group, use different group (simulate by shuffling group labels)
            all_train = []
            for gid in range(config.data.n_groups):
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
    datasets_path = Path(config.data.datasets_path)
    variant_dir = datasets_path / f"{smote_variant}_{strategy}"

    if not variant_dir.exists():
        logger.error(f"Datasets directory not found: {variant_dir}")
        return None

    # Prepare results
    results = {
        'experiment_name': config.experiment.name,
        'smote_variant': smote_variant,
        'strategy': strategy,
        'group_id': group_id,
        'results': []
    }

    # Get classifiers
    classifiers_config = vars(config.classifiers)
    enabled_classifiers = {}
    for name, cfg in classifiers_config.items():
        cfg_dict = vars(cfg) if isinstance(cfg, SimpleNamespace) else cfg
        if cfg_dict.get('enabled', True):
            enabled_classifiers[name] = cfg_dict

    # Run experiments for each ratio
    synthetic_ratios = config.data.synthetic_ratios

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
                # Initialize classifier
                params = clf_config.get('params', {})
                if isinstance(params, SimpleNamespace):
                    params = vars(params)

                if clf_name == 'svm':
                    clf = SVC(**params, probability=True)
                elif clf_name == 'random_forest':
                    clf = RandomForestClassifier(**params)
                else:
                    logger.warning(f"Unknown classifier: {clf_name}, skipping")
                    continue

                # Train (features already extracted by SMOTE)
                clf.fit(X_train, y_train)

                # Predict
                y_pred = clf.predict(X_test)
                y_pred_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else y_pred

                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1_score': f1_score(y_test, y_pred, zero_division=0),
                    'auc_roc': roc_auc_score(y_test, y_pred_proba),
                    'auc_pr': average_precision_score(y_test, y_pred_proba),
                    'balanced_accuracy': balanced_accuracy_score(y_test, y_pred)
                }

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
    results_path = Path(config.output.results_path)
    results_path.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj

    results = convert_numpy_types(results)

    results_file = results_path / f"{config.experiment.name}_{smote_variant}_{strategy}_group{group_id}_results.json"
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
        # Load configuration (directly from YAML, bypass LLM validation)
        logger.info("Loading configuration")
        config = load_smote_config(args.config)

        # Setup logging
        setup_logging(config)

        logger.info("=" * 60)
        logger.info(f"SMOTE Baseline Experiment: {config.experiment.name}")
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
