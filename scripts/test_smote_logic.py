#!/usr/bin/env python3
"""
Test script to verify SMOTE logic matches GPT/Claude experiments
验证SMOTE逻辑是否与GPT/Claude实验一致
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.config.config_manager import load_config
from src.traditional_methods.smote_generator import SMOTEGenerator

def test_smote_logic():
    """Test SMOTE dataset construction logic"""

    print("=" * 80)
    print("SMOTE Logic Verification Test")
    print("=" * 80)
    print()

    # Load Group 0 data
    group_file = Path("data/full_experiments/ceas08_gpt41mini/processed/groups/ceas08_group_0.csv.gz")
    if not group_file.exists():
        print(f"✗ Error: Group file not found: {group_file}")
        return False

    group_df = pd.read_csv(group_file, compression='gzip')

    # Create text column
    group_df['text'] = group_df['subject'].fillna('') + ' ' + group_df['body'].fillna('')

    print(f"Loaded Group 0: {len(group_df)} samples")
    print(f"  Spam: {(group_df['label'] == 1).sum()}")
    print(f"  Ham: {(group_df['label'] == 0).sum()}")
    print()

    # Initialize SMOTE generator with same config as GPT/Claude
    vectorizer_params = {
        'max_features': 10000,
        'ngram_range': (1, 2),
        'stop_words': 'english',
        'lowercase': True
    }

    smote_params = {
        'sampling_strategy': 'auto',
        'k_neighbors': 5,
        'random_state': 42
    }

    generator = SMOTEGenerator(
        method='smote',
        vectorizer_params=vectorizer_params,
        smote_params=smote_params,
        random_state=42
    )

    # Test different synthetic ratios
    test_ratios = [0, 10, 20, 50, 100]

    print("Testing SMOTE dataset construction:")
    print("-" * 80)
    print(f"{'Ratio':<10} {'Train Size':<12} {'Spam Count':<12} {'Ham Count':<12} {'Spam Ratio':<12}")
    print("-" * 80)

    results = []

    for ratio in test_ratios:
        X_train, y_train, metadata = generator.build_training_dataset(
            real_texts=group_df['text'],
            real_labels=group_df['label'],
            synthetic_ratio=ratio,
            strategy='within_group'
        )

        n_train = X_train.shape[0]
        n_spam = (y_train == 1).sum()
        n_ham = (y_train == 0).sum()
        spam_ratio = n_spam / n_train

        results.append({
            'ratio': ratio,
            'train_size': n_train,
            'spam_count': n_spam,
            'ham_count': n_ham,
            'spam_ratio': spam_ratio,
            'metadata': metadata
        })

        print(f"{ratio}%{'':<8} {n_train:<12} {n_spam:<12} {n_ham:<12} {spam_ratio:.4f}")

    print("-" * 80)
    print()

    # Verification checks
    print("Verification Checks:")
    print("=" * 80)

    # Check 1: Training set size should be constant (or very close)
    train_sizes = [r['train_size'] for r in results]
    size_variation = max(train_sizes) - min(train_sizes)

    print(f"✓ Check 1: Training set size consistency")
    print(f"  Min size: {min(train_sizes)}, Max size: {max(train_sizes)}, Variation: {size_variation}")
    if size_variation == 0:
        print(f"  ✓ PASS: Training set size is constant ({train_sizes[0]} samples)")
    else:
        print(f"  ⚠ WARNING: Training set size varies by {size_variation} samples")
    print()

    # Check 2: Spam count should be constant (or very close)
    spam_counts = [r['spam_count'] for r in results]
    spam_variation = max(spam_counts) - min(spam_counts)

    print(f"✓ Check 2: Spam count consistency")
    print(f"  Min spam: {min(spam_counts)}, Max spam: {max(spam_counts)}, Variation: {spam_variation}")
    if spam_variation == 0:
        print(f"  ✓ PASS: Spam count is constant ({spam_counts[0]} samples)")
    else:
        print(f"  ⚠ WARNING: Spam count varies by {spam_variation} samples")
    print()

    # Check 3: Spam ratio should be constant
    spam_ratios = [r['spam_ratio'] for r in results]
    ratio_variation = max(spam_ratios) - min(spam_ratios)

    print(f"✓ Check 3: Spam ratio consistency")
    print(f"  Min ratio: {min(spam_ratios):.4f}, Max ratio: {max(spam_ratios):.4f}, Variation: {ratio_variation:.6f}")
    if ratio_variation < 0.001:  # Allow tiny floating point errors
        print(f"  ✓ PASS: Spam ratio is constant (~{spam_ratios[0]:.4f})")
    else:
        print(f"  ✗ FAIL: Spam ratio varies by {ratio_variation:.6f}")
    print()

    # Check 4: Synthetic ratio progression
    print(f"✓ Check 4: Synthetic ratio progression")
    for r in results:
        meta = r['metadata']
        expected_ratio = r['ratio']
        actual_ratio = meta.get('actual_synthetic_ratio', 0)
        n_real = meta.get('n_real_spam', 0)
        n_synthetic = meta.get('n_synthetic_spam', 0)

        print(f"  {expected_ratio}% synthetic: {n_real} real + {n_synthetic} synthetic = {n_real + n_synthetic} total spam")
    print()

    # Final summary
    print("=" * 80)
    print("Summary:")
    print("=" * 80)

    all_pass = (size_variation == 0) and (spam_variation == 0) and (ratio_variation < 0.001)

    if all_pass:
        print("✓ ALL CHECKS PASSED!")
        print("  SMOTE logic now matches GPT/Claude experiments:")
        print("  - Training set size: CONSTANT")
        print("  - Spam count: CONSTANT")
        print("  - Spam:ham ratio: CONSTANT")
        print("  - Only synthetic:real ratio changes")
    else:
        print("✗ SOME CHECKS FAILED")
        print("  Please review the code for potential issues")

    print("=" * 80)
    print()

    return all_pass

if __name__ == "__main__":
    success = test_smote_logic()
    sys.exit(0 if success else 1)
