#!/usr/bin/env python3
"""
Data Verification Script for Reverse Detection Experiment
Checks availability and validity of all required data before running experiments
"""

import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


def check_baseline_data(n_groups: int = 20) -> Dict[str, any]:
    """Check baseline training data availability"""
    print("Checking baseline training data...")

    results = {
        'available': [],
        'missing': [],
        'stats': []
    }

    base_path = Path('output/prepared_data/groups')

    for group_id in range(n_groups):
        train_path = base_path / f'group_{group_id}' / 'train.pkl'
        test_path = base_path / f'group_{group_id}' / 'test.pkl'

        if train_path.exists() and test_path.exists():
            # Load and check
            try:
                with open(train_path, 'rb') as f:
                    train_data = pickle.load(f)
                with open(test_path, 'rb') as f:
                    test_data = pickle.load(f)

                results['available'].append(group_id)
                results['stats'].append({
                    'group_id': group_id,
                    'train_size': len(train_data),
                    'train_spam_ratio': (train_data['label'] == 1).mean(),
                    'test_size': len(test_data),
                    'test_spam_count': (test_data['label'] == 1).sum(),
                    'test_ham_count': (test_data['label'] == 0).sum()
                })
            except Exception as e:
                print(f"  ✗ Group {group_id}: Error loading - {e}")
                results['missing'].append(group_id)
        else:
            results['missing'].append(group_id)

    print(f"  ✓ Available: {len(results['available'])}/{n_groups} groups")
    if results['missing']:
        print(f"  ✗ Missing: {results['missing']}")

    return results


def check_synthetic_data(
    method: str,
    prompts: List[str],
    strategies: List[str],
    n_groups: int = 20
) -> Dict[str, any]:
    """Check synthetic spam data availability"""
    print(f"\nChecking {method} synthetic data...")

    results = {
        'available': {},
        'missing': {},
        'stats': {}
    }

    if method == 'smote':
        base_path = Path('output/full_experiments/ceas08_smote/datasets')
        prompts = ['smote']  # SMOTE has no prompts
    else:
        base_path = Path(f'output/full_experiments/ceas08_{method}/datasets')

    for prompt in prompts:
        for strategy in strategies:
            config_key = f"{prompt}_{strategy}"

            if method == 'smote':
                subdir = f"smote_{strategy}_100"
            else:
                subdir = f"{prompt}_{strategy}_100"

            dataset_dir = base_path / subdir

            available_groups = []
            missing_groups = []
            config_stats = []

            for group_id in range(n_groups):
                dataset_path = dataset_dir / f'group_{group_id}_train.pkl'

                if dataset_path.exists():
                    try:
                        with open(dataset_path, 'rb') as f:
                            data = pickle.load(f)

                        spam_count = (data['label'] == 1).sum()
                        ham_count = (data['label'] == 0).sum()

                        available_groups.append(group_id)
                        config_stats.append({
                            'group_id': group_id,
                            'total_size': len(data),
                            'spam_count': int(spam_count),
                            'ham_count': int(ham_count)
                        })
                    except Exception as e:
                        print(f"  ✗ {config_key} Group {group_id}: Error - {e}")
                        missing_groups.append(group_id)
                else:
                    missing_groups.append(group_id)

            results['available'][config_key] = available_groups
            results['missing'][config_key] = missing_groups
            results['stats'][config_key] = config_stats

            if len(available_groups) == n_groups:
                print(f"  ✓ {config_key}: All {n_groups} groups available")
            else:
                print(f"  ✗ {config_key}: {len(available_groups)}/{n_groups} available (missing: {missing_groups})")

    return results


def main():
    print("=" * 60)
    print("Reverse Detection Data Verification")
    print("=" * 60)

    n_groups = 20

    # Check baseline data
    baseline_results = check_baseline_data(n_groups)

    if len(baseline_results['available']) < n_groups:
        print(f"\n⚠️  WARNING: Missing baseline data for {len(baseline_results['missing'])} groups")
        print("Cannot proceed with reverse detection experiments!")
        return False

    # Check GPT synthetic data
    gpt_results = check_synthetic_data(
        'gpt41mini',
        ['original', 'strong', 'weak'],
        ['within_group', 'cross_group'],
        n_groups
    )

    # Check Claude synthetic data
    claude_results = check_synthetic_data(
        'claude35haiku',
        ['original', 'strong', 'weak'],
        ['within_group', 'cross_group'],
        n_groups
    )

    # Check SMOTE data
    smote_results = check_synthetic_data(
        'smote',
        [],  # No prompts for SMOTE
        ['within_group', 'cross_group'],
        n_groups
    )

    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)

    all_checks_passed = True

    # Baseline
    if len(baseline_results['available']) == n_groups:
        print("✓ Baseline data: PASS")
    else:
        print("✗ Baseline data: FAIL")
        all_checks_passed = False

    # GPT
    gpt_configs = len(gpt_results['available'])
    gpt_complete = sum(1 for v in gpt_results['available'].values() if len(v) == n_groups)
    if gpt_complete == gpt_configs:
        print(f"✓ GPT-4.1-mini data: PASS ({gpt_configs} configurations)")
    else:
        print(f"✗ GPT-4.1-mini data: FAIL ({gpt_complete}/{gpt_configs} configurations complete)")
        all_checks_passed = False

    # Claude
    claude_configs = len(claude_results['available'])
    claude_complete = sum(1 for v in claude_results['available'].values() if len(v) == n_groups)
    if claude_complete == claude_configs:
        print(f"✓ Claude-3.5-Haiku data: PASS ({claude_configs} configurations)")
    else:
        print(f"✗ Claude-3.5-Haiku data: FAIL ({claude_complete}/{claude_configs} configurations complete)")
        all_checks_passed = False

    # SMOTE
    smote_configs = len(smote_results['available'])
    smote_complete = sum(1 for v in smote_results['available'].values() if len(v) == n_groups)
    if smote_complete == smote_configs:
        print(f"✓ SMOTE data: PASS ({smote_configs} configurations)")
    else:
        print(f"✗ SMOTE data: FAIL ({smote_complete}/{smote_configs} configurations complete)")
        all_checks_passed = False

    print("\n" + "=" * 60)

    if all_checks_passed:
        print("✓ All data checks PASSED!")
        print("\nReady to run reverse detection experiments:")
        print("  bash scripts/run_all_reverse_detection.sh")
        return True
    else:
        print("✗ Some data checks FAILED!")
        print("\nPlease resolve missing data before running experiments.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
