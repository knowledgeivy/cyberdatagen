# src/cyberdata/scripts/batch1_prepare_base_data.py

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path

# Determine PROJECT_ROOT
current_file = Path(__file__).resolve()
if 'src/cyberdata/scripts' in str(current_file):
    PROJECT_ROOT = current_file.parents[3]
else:
    PROJECT_ROOT = Path.cwd()

print(f"Current file: {current_file}")
print(f"PROJECT_ROOT: {PROJECT_ROOT}")

# Paths
RAW_DATA_FILE = PROJECT_ROOT / 'raw/SevenPhishingEmailDataset/CEAS-08.csv'
OUTPUT_DIR = PROJECT_ROOT / 'data/batch1'
BASE_SAMPLES_DIR = OUTPUT_DIR / 'base_samples'
SEED_SAMPLES_DIR = OUTPUT_DIR / 'seed_samples'

# Create directories
BASE_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
SEED_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

def load_and_sample_data():
    """Load CEAS-08 data and create base samples"""
    print("Loading CEAS-08 dataset...")
    df = pd.read_csv(RAW_DATA_FILE)
    
    print(f"Total data: {len(df)}")
    print(f"Label distribution:")
    print(df['label'].value_counts())
    
    # Separate by label
    malicious_data = df[df['label'] == 1].copy()
    benign_data = df[df['label'] == 0].copy()
    
    print(f"Malicious: {len(malicious_data)}, Benign: {len(benign_data)}")
    
    # Sample 5K from each
    np.random.seed(42)
    malicious_sample = malicious_data.sample(n=min(5000, len(malicious_data)), random_state=42)
    benign_sample = benign_data.sample(n=min(5000, len(benign_data)), random_state=42)
    
    # Add ID annotations
    malicious_sample = malicious_sample.reset_index(drop=True)
    benign_sample = benign_sample.reset_index(drop=True)
    
    malicious_sample['data_id'] = [f"CEAS08_REAL_M_{i:05d}" for i in range(len(malicious_sample))]
    benign_sample['data_id'] = [f"CEAS08_REAL_B_{i:05d}" for i in range(len(benign_sample))]
    
    # Save base samples
    malicious_sample.to_csv(BASE_SAMPLES_DIR / 'malicious_5k.csv', index=False)
    benign_sample.to_csv(BASE_SAMPLES_DIR / 'benign_5k.csv', index=False)
    
    print(f"Saved base samples to {BASE_SAMPLES_DIR}")
    return malicious_sample, benign_sample

def create_seed_samples(malicious_sample, benign_sample):
    """Create 1000 groups and select seeds"""
    print("Creating seed samples...")
    
    # Group sampling for malicious (1000 groups)
    np.random.seed(42)
    malicious_shuffled = malicious_sample.sample(frac=1, random_state=42).reset_index(drop=True)
    
    n_groups = 1000
    group_size = len(malicious_shuffled) // n_groups
    
    seed_malicious = []
    for i in range(n_groups):
        start_idx = i * group_size
        end_idx = min(start_idx + group_size, len(malicious_shuffled))
        group = malicious_shuffled.iloc[start_idx:end_idx]
        # Take first sample from each group as seed
        seed_malicious.append(group.iloc[0])
    
    seed_malicious_df = pd.DataFrame(seed_malicious)
    
    # For benign, just take random 1000 samples
    seed_benign_df = benign_sample.sample(n=1000, random_state=42)
    
    # Save seed samples
    seed_malicious_df.to_csv(SEED_SAMPLES_DIR / 'malicious_seeds_1k.csv', index=False)
    seed_benign_df.to_csv(SEED_SAMPLES_DIR / 'benign_seeds_1k.csv', index=False)
    
    print(f"Saved seed samples to {SEED_SAMPLES_DIR}")
    
    # Save metadata
    metadata = {
        'experiment': 'batch1',
        'base_malicious_count': len(malicious_sample),
        'base_benign_count': len(benign_sample),
        'seed_malicious_count': len(seed_malicious_df),
        'seed_benign_count': len(seed_benign_df),
        'group_strategy': 'sequential_groups',
        'n_groups': n_groups,
        'random_seed': 42
    }
    
    with open(OUTPUT_DIR / 'batch1_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return seed_malicious_df, seed_benign_df

if __name__ == "__main__":
    print("=== BATCH 1: PREPARE BASE DATA ===")
    
    # Load and sample data
    malicious_sample, benign_sample = load_and_sample_data()
    
    # Create seed samples
    seed_malicious_df, seed_benign_df = create_seed_samples(malicious_sample, benign_sample)
    
    print(f"\nCompleted:")
    print(f"- Base samples: {len(malicious_sample)} malicious, {len(benign_sample)} benign")
    print(f"- Seed samples: {len(seed_malicious_df)} malicious, {len(seed_benign_df)} benign")
    print(f"- Output directory: {OUTPUT_DIR}")