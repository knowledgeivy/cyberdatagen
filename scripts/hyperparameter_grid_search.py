#!/usr/bin/env python3
"""
Hyperparameter Grid Search Script
Systematically search for optimal hyperparameters for SVM and Random Forest classifiers
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, make_scorer

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
from src.config.config_manager import load_config


def load_training_data(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load training data for grid search

    Args:
        data_path: Path to training data CSV file

    Returns:
        Tuple of (X, y)
    """
    logger.info(f"Loading training data from: {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")

    data = pd.read_csv(data_path, compression='gzip')

    # Combine subject and body
    X = data[['subject', 'body']].copy()
    y = data['label']

    logger.info(f"Loaded {len(data)} samples - Spam: {(y == 1).sum()}, Ham: {(y == 0).sum()}")

    return X, y


def create_text_features(X: pd.DataFrame) -> pd.Series:
    """
    Create combined text from subject and body

    Args:
        X: DataFrame with subject and body columns

    Returns:
        Series of combined text
    """
    return X['subject'].fillna('') + ' ' + X['body'].fillna('')


def grid_search_svm(X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, Any]:
    """
    Perform grid search for SVM hyperparameters

    Args:
        X: Training features
        y: Training labels
        cv: Number of cross-validation folds

    Returns:
        Dictionary with grid search results
    """
    logger.info("=" * 60)
    logger.info("SVM Hyperparameter Grid Search")
    logger.info("=" * 60)

    # Create text features
    texts = create_text_features(X)

    # Define parameter grid
    param_grid = {
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__kernel': ['linear', 'rbf'],
        'classifier__gamma': ['scale', 'auto'],  # Only for RBF
    }

    # Create pipeline
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            min_df=2,
            max_df=0.95
        )),
        ('scaler', StandardScaler(with_mean=False)),  # Sparse matrix compatible
        ('classifier', SVC(
            class_weight='balanced',
            random_state=42,
            probability=True
        ))
    ])

    # Custom scorer for F1
    f1_scorer = make_scorer(f1_score, average='binary')

    # Create GridSearchCV
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv_strategy,
        scoring={
            'f1': f1_scorer,
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, average='binary'),
            'recall': make_scorer(recall_score, average='binary'),
            'auc_roc': 'roc_auc'
        },
        refit='f1',  # Optimize for F1-score
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )

    logger.info("Starting SVM grid search...")
    logger.info(f"Parameter grid: {param_grid}")
    logger.info(f"Total combinations: {len(param_grid['classifier__C']) * len(param_grid['classifier__kernel']) * len(param_grid['classifier__gamma'])}")

    # Fit grid search
    grid_search.fit(texts, y)

    logger.info("SVM grid search completed!")
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best F1-score: {grid_search.best_score_:.4f}")

    # Extract results
    results = {
        'classifier': 'svm',
        'best_params': grid_search.best_params_,
        'best_score': float(grid_search.best_score_),
        'best_index': int(grid_search.best_index_),
        'cv_results': {}
    }

    # Store CV results
    cv_results = grid_search.cv_results_
    for key in cv_results.keys():
        if isinstance(cv_results[key], np.ndarray):
            results['cv_results'][key] = cv_results[key].tolist()
        else:
            results['cv_results'][key] = cv_results[key]

    # Create results DataFrame for analysis
    results_df = pd.DataFrame({
        'params': cv_results['params'],
        'mean_test_f1': cv_results['mean_test_f1'],
        'std_test_f1': cv_results['std_test_f1'],
        'mean_test_accuracy': cv_results['mean_test_accuracy'],
        'mean_test_precision': cv_results['mean_test_precision'],
        'mean_test_recall': cv_results['mean_test_recall'],
        'mean_test_auc_roc': cv_results['mean_test_auc_roc'],
        'rank_test_f1': cv_results['rank_test_f1']
    })

    # Sort by F1 score
    results_df = results_df.sort_values('mean_test_f1', ascending=False)

    logger.info("\nTop 5 SVM configurations:")
    for idx, row in results_df.head(5).iterrows():
        logger.info(f"  {row['params']}: F1={row['mean_test_f1']:.4f} ± {row['std_test_f1']:.4f}")

    results['results_df'] = results_df.to_dict('records')

    return results, grid_search


def grid_search_random_forest(X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, Any]:
    """
    Perform grid search for Random Forest hyperparameters

    Args:
        X: Training features
        y: Training labels
        cv: Number of cross-validation folds

    Returns:
        Dictionary with grid search results
    """
    logger.info("=" * 60)
    logger.info("Random Forest Hyperparameter Grid Search")
    logger.info("=" * 60)

    # Create text features
    texts = create_text_features(X)

    # Define parameter grid
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [5, 10, 20, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }

    # Create pipeline
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            min_df=2,
            max_df=0.95
        )),
        ('classifier', RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Custom scorer for F1
    f1_scorer = make_scorer(f1_score, average='binary')

    # Create GridSearchCV
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv_strategy,
        scoring={
            'f1': f1_scorer,
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, average='binary'),
            'recall': make_scorer(recall_score, average='binary'),
            'auc_roc': 'roc_auc'
        },
        refit='f1',  # Optimize for F1-score
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )

    logger.info("Starting Random Forest grid search...")
    logger.info(f"Parameter grid: {param_grid}")
    total_combinations = (len(param_grid['classifier__n_estimators']) *
                         len(param_grid['classifier__max_depth']) *
                         len(param_grid['classifier__min_samples_split']) *
                         len(param_grid['classifier__min_samples_leaf']))
    logger.info(f"Total combinations: {total_combinations}")

    # Fit grid search
    grid_search.fit(texts, y)

    logger.info("Random Forest grid search completed!")
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best F1-score: {grid_search.best_score_:.4f}")

    # Extract results
    results = {
        'classifier': 'random_forest',
        'best_params': grid_search.best_params_,
        'best_score': float(grid_search.best_score_),
        'best_index': int(grid_search.best_index_),
        'cv_results': {}
    }

    # Store CV results
    cv_results = grid_search.cv_results_
    for key in cv_results.keys():
        if isinstance(cv_results[key], np.ndarray):
            results['cv_results'][key] = cv_results[key].tolist()
        else:
            results['cv_results'][key] = cv_results[key]

    # Create results DataFrame for analysis
    results_df = pd.DataFrame({
        'params': cv_results['params'],
        'mean_test_f1': cv_results['mean_test_f1'],
        'std_test_f1': cv_results['std_test_f1'],
        'mean_test_accuracy': cv_results['mean_test_accuracy'],
        'mean_test_precision': cv_results['mean_test_precision'],
        'mean_test_recall': cv_results['mean_test_recall'],
        'mean_test_auc_roc': cv_results['mean_test_auc_roc'],
        'rank_test_f1': cv_results['rank_test_f1']
    })

    # Sort by F1 score
    results_df = results_df.sort_values('mean_test_f1', ascending=False)

    logger.info("\nTop 5 Random Forest configurations:")
    for idx, row in results_df.head(5).iterrows():
        logger.info(f"  {row['params']}: F1={row['mean_test_f1']:.4f} ± {row['std_test_f1']:.4f}")

    results['results_df'] = results_df.to_dict('records')

    return results, grid_search


def create_svm_heatmap(results: Dict[str, Any], output_path: str):
    """
    Create heatmap visualization for SVM grid search results

    Args:
        results: Grid search results
        output_path: Path to save the heatmap
    """
    logger.info("Creating SVM heatmap...")

    results_df = pd.DataFrame(results['results_df'])

    # Extract C and kernel from params
    results_df['C'] = results_df['params'].apply(lambda x: x['classifier__C'])
    results_df['kernel'] = results_df['params'].apply(lambda x: x['classifier__kernel'])
    results_df['gamma'] = results_df['params'].apply(lambda x: x.get('classifier__gamma', 'N/A'))

    # Create separate heatmaps for linear and rbf kernels
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, kernel in enumerate(['linear', 'rbf']):
        kernel_df = results_df[results_df['kernel'] == kernel].copy()

        if kernel == 'linear':
            # Linear kernel: only C parameter
            pivot_data = kernel_df.groupby('C')['mean_test_f1'].mean().to_frame()
            pivot_data.columns = ['F1-Score']

            ax = axes[idx]
            sns.heatmap(
                pivot_data.T,
                annot=True,
                fmt='.4f',
                cmap='YlGnBu',
                vmin=0.5,
                vmax=1.0,
                ax=ax,
                cbar_kws={'label': 'F1-Score'}
            )
            ax.set_title(f'SVM - {kernel.capitalize()} Kernel', fontsize=12, fontweight='bold')
            ax.set_xlabel('C', fontsize=10)
            ax.set_ylabel('', fontsize=10)

        else:  # rbf kernel
            # RBF kernel: C and gamma
            pivot_data = kernel_df.pivot_table(
                values='mean_test_f1',
                index='gamma',
                columns='C',
                aggfunc='mean'
            )

            ax = axes[idx]
            sns.heatmap(
                pivot_data,
                annot=True,
                fmt='.4f',
                cmap='YlGnBu',
                vmin=0.5,
                vmax=1.0,
                ax=ax,
                cbar_kws={'label': 'F1-Score'}
            )
            ax.set_title(f'SVM - {kernel.upper()} Kernel', fontsize=12, fontweight='bold')
            ax.set_xlabel('C', fontsize=10)
            ax.set_ylabel('Gamma', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"SVM heatmap saved to: {output_path}")


def create_rf_heatmap(results: Dict[str, Any], output_path: str):
    """
    Create heatmap visualization for Random Forest grid search results

    Args:
        results: Grid search results
        output_path: Path to save the heatmap
    """
    logger.info("Creating Random Forest heatmap...")

    results_df = pd.DataFrame(results['results_df'])

    # Extract parameters from params dict
    results_df['n_estimators'] = results_df['params'].apply(lambda x: x['classifier__n_estimators'])
    results_df['max_depth'] = results_df['params'].apply(lambda x: str(x['classifier__max_depth']))
    results_df['min_samples_split'] = results_df['params'].apply(lambda x: x['classifier__min_samples_split'])
    results_df['min_samples_leaf'] = results_df['params'].apply(lambda x: x['classifier__min_samples_leaf'])

    # Create 2x2 subplot for different parameter combinations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: n_estimators vs max_depth (averaged over other params)
    pivot1 = results_df.groupby(['n_estimators', 'max_depth'])['mean_test_f1'].mean().reset_index()
    pivot1 = pivot1.pivot(index='max_depth', columns='n_estimators', values='mean_test_f1')

    sns.heatmap(
        pivot1,
        annot=True,
        fmt='.4f',
        cmap='YlGnBu',
        vmin=0.7,
        vmax=1.0,
        ax=axes[0, 0],
        cbar_kws={'label': 'F1-Score'}
    )
    axes[0, 0].set_title('n_estimators vs max_depth', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('n_estimators', fontsize=10)
    axes[0, 0].set_ylabel('max_depth', fontsize=10)

    # Plot 2: n_estimators vs min_samples_split
    pivot2 = results_df.groupby(['n_estimators', 'min_samples_split'])['mean_test_f1'].mean().reset_index()
    pivot2 = pivot2.pivot(index='min_samples_split', columns='n_estimators', values='mean_test_f1')

    sns.heatmap(
        pivot2,
        annot=True,
        fmt='.4f',
        cmap='YlGnBu',
        vmin=0.7,
        vmax=1.0,
        ax=axes[0, 1],
        cbar_kws={'label': 'F1-Score'}
    )
    axes[0, 1].set_title('n_estimators vs min_samples_split', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('n_estimators', fontsize=10)
    axes[0, 1].set_ylabel('min_samples_split', fontsize=10)

    # Plot 3: max_depth vs min_samples_split
    pivot3 = results_df.groupby(['max_depth', 'min_samples_split'])['mean_test_f1'].mean().reset_index()
    pivot3 = pivot3.pivot(index='min_samples_split', columns='max_depth', values='mean_test_f1')

    sns.heatmap(
        pivot3,
        annot=True,
        fmt='.4f',
        cmap='YlGnBu',
        vmin=0.7,
        vmax=1.0,
        ax=axes[1, 0],
        cbar_kws={'label': 'F1-Score'}
    )
    axes[1, 0].set_title('max_depth vs min_samples_split', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('max_depth', fontsize=10)
    axes[1, 0].set_ylabel('min_samples_split', fontsize=10)

    # Plot 4: min_samples_split vs min_samples_leaf
    pivot4 = results_df.groupby(['min_samples_split', 'min_samples_leaf'])['mean_test_f1'].mean().reset_index()
    pivot4 = pivot4.pivot(index='min_samples_leaf', columns='min_samples_split', values='mean_test_f1')

    sns.heatmap(
        pivot4,
        annot=True,
        fmt='.4f',
        cmap='YlGnBu',
        vmin=0.7,
        vmax=1.0,
        ax=axes[1, 1],
        cbar_kws={'label': 'F1-Score'}
    )
    axes[1, 1].set_title('min_samples_split vs min_samples_leaf', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('min_samples_split', fontsize=10)
    axes[1, 1].set_ylabel('min_samples_leaf', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Random Forest heatmap saved to: {output_path}")


def create_comparison_table(svm_results: Dict[str, Any], rf_results: Dict[str, Any], output_path: str):
    """
    Create comparison table for all hyperparameter combinations

    Args:
        svm_results: SVM grid search results
        rf_results: Random Forest grid search results
        output_path: Path to save the table
    """
    logger.info("Creating comparison table...")

    # SVM results
    svm_df = pd.DataFrame(svm_results['results_df'])
    svm_df['classifier'] = 'SVM'
    svm_df['params_str'] = svm_df['params'].apply(lambda x: str(x))

    # RF results
    rf_df = pd.DataFrame(rf_results['results_df'])
    rf_df['classifier'] = 'Random Forest'
    rf_df['params_str'] = rf_df['params'].apply(lambda x: str(x))

    # Combine
    combined_df = pd.concat([
        svm_df[['classifier', 'params_str', 'mean_test_f1', 'std_test_f1',
                'mean_test_accuracy', 'mean_test_precision', 'mean_test_recall',
                'mean_test_auc_roc', 'rank_test_f1']],
        rf_df[['classifier', 'params_str', 'mean_test_f1', 'std_test_f1',
               'mean_test_accuracy', 'mean_test_precision', 'mean_test_recall',
               'mean_test_auc_roc', 'rank_test_f1']]
    ], ignore_index=True)

    # Sort by F1 score
    combined_df = combined_df.sort_values('mean_test_f1', ascending=False)

    # Save to CSV
    combined_df.to_csv(output_path, index=False)

    logger.info(f"Comparison table saved to: {output_path}")

    return combined_df


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Grid Search for SVM and Random Forest')
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/full_experiments/ceas08_gpt41mini/datasets/within_group/train_original_r0_g0_t0.csv.gz',
        help='Path to training data CSV file (default: group 0, r0 baseline)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output/hyperparameter_search',
        help='Output directory for results'
    )
    parser.add_argument(
        '--cv',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    parser.add_argument(
        '--classifiers',
        nargs='+',
        default=['svm', 'random_forest'],
        choices=['svm', 'random_forest'],
        help='Classifiers to run grid search for (default: both)'
    )

    args = parser.parse_args()

    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Setup logging
        log_file = os.path.join(args.output_dir, 'grid_search.log')
        logger.add(
            log_file,
            level='INFO',
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            rotation="100 MB"
        )

        logger.info("=" * 60)
        logger.info("Hyperparameter Grid Search")
        logger.info("=" * 60)
        logger.info(f"Training data: {args.data_path}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Cross-validation folds: {args.cv}")
        logger.info(f"Classifiers: {args.classifiers}")

        # Load training data
        X, y = load_training_data(args.data_path)

        # Store all results
        all_results = {
            'data_path': args.data_path,
            'cv_folds': args.cv,
            'n_samples': len(X),
            'n_spam': int((y == 1).sum()),
            'n_ham': int((y == 0).sum()),
            'classifiers': {}
        }

        # Grid search for SVM
        if 'svm' in args.classifiers:
            svm_results, svm_grid = grid_search_svm(X, y, cv=args.cv)
            all_results['classifiers']['svm'] = svm_results

            # Create SVM heatmap
            svm_heatmap_path = os.path.join(args.output_dir, 'svm_grid_search_heatmap.png')
            create_svm_heatmap(svm_results, svm_heatmap_path)

        # Grid search for Random Forest
        if 'random_forest' in args.classifiers:
            rf_results, rf_grid = grid_search_random_forest(X, y, cv=args.cv)
            all_results['classifiers']['random_forest'] = rf_results

            # Create RF heatmap
            rf_heatmap_path = os.path.join(args.output_dir, 'rf_grid_search_heatmap.png')
            create_rf_heatmap(rf_results, rf_heatmap_path)

        # Create comparison table
        if 'svm' in args.classifiers and 'random_forest' in args.classifiers:
            comparison_path = os.path.join(args.output_dir, 'hyperparameter_comparison.csv')
            comparison_df = create_comparison_table(svm_results, rf_results, comparison_path)

            # Print top 10 overall
            logger.info("\n" + "=" * 60)
            logger.info("Top 10 Configurations Overall:")
            logger.info("=" * 60)
            for idx, row in comparison_df.head(10).iterrows():
                logger.info(f"{row['classifier']:15s} F1={row['mean_test_f1']:.4f} ± {row['std_test_f1']:.4f}")
                logger.info(f"  {row['params_str']}")

        # Save complete results
        results_file = os.path.join(args.output_dir, 'grid_search_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"\nComplete results saved to: {results_file}")

        # Save best hyperparameters in a separate file for easy access
        best_params = {}
        for clf_name, clf_results in all_results['classifiers'].items():
            best_params[clf_name] = clf_results['best_params']

        best_params_file = os.path.join(args.output_dir, 'best_hyperparameters.json')
        with open(best_params_file, 'w', encoding='utf-8') as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False)

        logger.info(f"Best hyperparameters saved to: {best_params_file}")

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("Grid Search Summary")
        logger.info("=" * 60)

        for clf_name, clf_results in all_results['classifiers'].items():
            logger.info(f"\n{clf_name.upper()}:")
            logger.info(f"  Best F1-score: {clf_results['best_score']:.4f}")
            logger.info(f"  Best parameters:")
            for param, value in clf_results['best_params'].items():
                logger.info(f"    {param}: {value}")

        logger.success("\nHyperparameter grid search completed successfully!")

    except Exception as e:
        logger.error(f"Grid search failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
