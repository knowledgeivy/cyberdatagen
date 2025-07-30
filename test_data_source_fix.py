#!/usr/bin/env python3
"""
Quick test script to verify the data source categorization fix
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def test_data_source_categorization():
    """Test the corrected data source categorization"""
    
    print("=== Testing Data Source Categorization Fix ===")
    
    # Paths
    batch4_dir = Path("data/batch4_fresh")
    
    # 1. Load train malicious data
    train_mal = pd.read_csv(batch4_dir / "train_malicious.csv")
    print(f"Total train malicious samples: {len(train_mal):,}")
    
    # 2. Extract seed IDs from all synthetic data
    all_seed_ids = set()
    for layer in ['core', 'inner', 'outer', 'edge']:
        synthetic_df = pd.read_csv(batch4_dir / f"synthetic/{layer}_synthetic.csv")
        if 'original_id' in synthetic_df.columns:
            seed_ids = synthetic_df['original_id'].unique()
            all_seed_ids.update(seed_ids)
            print(f"{layer} layer: {len(seed_ids)} unique seed IDs")
    
    print(f"Total unique seed IDs across all layers: {len(all_seed_ids)}")
    
    # 3. Extract actual seed samples from stratified layer data
    all_layer_samples = []
    for layer in ['core', 'inner', 'outer', 'edge']:
        layer_samples = pd.read_csv(batch4_dir / f"stratified_layers/{layer}_samples.csv")
        all_layer_samples.append(layer_samples)
    
    all_stratified_samples = pd.concat(all_layer_samples, ignore_index=True)
    seed_mask = all_stratified_samples['unique_id'].isin(all_seed_ids)
    seed_samples = all_stratified_samples[seed_mask]
    print(f"Actual seed samples found: {len(seed_samples):,}")
    
    # 4. Create background sample (reduced from 50k to 15k)
    non_seed_mask = ~seed_mask
    background_samples = train_mal[non_seed_mask]
    if len(background_samples) > 15000:
        background_sample = background_samples.sample(n=15000, random_state=2025)
    else:
        background_sample = background_samples
    
    print(f"Background samples: {len(background_sample):,}")
    
    # 5. Verify synthetic data counts
    synthetic_counts = {}
    for layer in ['core', 'inner', 'outer', 'edge']:
        synthetic_df = pd.read_csv(batch4_dir / f"synthetic/{layer}_synthetic.csv")
        for variant in ['rewrite', 'rewrite_strong', 'rewrite_weak']:
            if 'prompt_variant' in synthetic_df.columns:
                variant_count = len(synthetic_df[synthetic_df['prompt_variant'] == variant])
            else:
                # Parse from unique_id
                variant_count = len(synthetic_df[synthetic_df['unique_id'].str.endswith(variant)])
            
            if variant not in synthetic_counts:
                synthetic_counts[variant] = 0
            synthetic_counts[variant] += variant_count
    
    print(f"Synthetic data counts: {synthetic_counts}")
    
    # 6. Summary
    print("\n=== New Data Source Categories ===")
    print(f"real_malicious_background: {len(background_sample):,}")
    print(f"real_malicious_seeds: {len(seed_samples):,}")
    for variant, count in synthetic_counts.items():
        print(f"synthetic_{variant}: {count:,}")
    
    # Verify 1:1 correspondence
    total_synthetic = sum(synthetic_counts.values())
    expected_seeds = total_synthetic // 3  # 3 variants per seed
    print(f"\nExpected seeds for 1:1 correspondence: {expected_seeds}")
    print(f"Actual seeds found: {len(seed_samples)}")
    print(f"Correspondence check: {'✓ PASS' if len(seed_samples) == expected_seeds else '✗ FAIL'}")

if __name__ == "__main__":
    test_data_source_categorization()