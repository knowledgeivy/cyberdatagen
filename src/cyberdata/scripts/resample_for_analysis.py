# src/cyberdata/scripts/resample_for_analysis.py

import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path

# Determine PROJECT_ROOT
current_file = Path(__file__).resolve()
if 'src/cyberdata/scripts' in str(current_file):
    PROJECT_ROOT = current_file.parents[3]
else:
    PROJECT_ROOT = Path.cwd()

print(f"PROJECT_ROOT: {PROJECT_ROOT}")

def resample_batch_data(batch_name, malicious_per_type=200, benign_count=1000, random_seed=42):
    """Resample data for a specific batch"""
    print(f"\n=== RESAMPLING {batch_name.upper()} DATA ===")
    print(f"Target: {malicious_per_type} samples per malicious type, {benign_count} benign samples")
    
    # Set paths based on batch
    if batch_name == 'batch1':
        batch_dir = PROJECT_ROOT / 'data/batch1'
        seed_file = batch_dir / 'seed_samples/malicious_seeds_1k.csv'
        synthetic_files = {
            'rewrite': batch_dir / 'synthetic/malicious_rewrite_1k.csv',
            'rewrite_strong': batch_dir / 'synthetic/malicious_rewrite_strong_1k.csv',
            'rewrite_weak': batch_dir / 'synthetic/malicious_rewrite_weak_1k.csv'
        }
        benign_file = batch_dir / 'seed_samples/benign_seeds_1k.csv'
    else:  # batch2
        batch_dir = PROJECT_ROOT / 'data/batch2'
        seed_file = batch_dir / 'enhanced_seeds/malicious_enhanced_seeds.csv'
        synthetic_files = {
            'rewrite': batch_dir / 'synthetic/malicious_rewrite_enhanced.csv',
            'rewrite_strong': batch_dir / 'synthetic/malicious_rewrite_strong_enhanced.csv',
            'rewrite_weak': batch_dir / 'synthetic/malicious_rewrite_weak_enhanced.csv'
        }
        benign_file = batch_dir / 'enhanced_seeds/benign_seeds_1k.csv'
    
    # Create resampled directory
    resampled_dir = batch_dir / 'resampled'
    resampled_dir.mkdir(exist_ok=True)
    
    np.random.seed(random_seed)
    
    resampled_data = {}
    
    # Resample real malicious data
    if seed_file.exists():
        real_data = pd.read_csv(seed_file)
        if len(real_data) >= malicious_per_type:
            real_resampled = real_data.sample(n=malicious_per_type, random_state=random_seed)
        else:
            print(f"Warning: Only {len(real_data)} real samples available, using all")
            real_resampled = real_data.copy()
        
        real_resampled['data_source'] = f'{batch_name}_real'
        resampled_data['real'] = real_resampled
        
        output_file = resampled_dir / f'real_malicious_{malicious_per_type}.csv'
        real_resampled.to_csv(output_file, index=False)
        print(f"Real malicious: {len(real_resampled)} samples → {output_file}")
    else:
        print(f"Warning: Real seed file not found: {seed_file}")
    
    # Resample synthetic data
    for variant, file_path in synthetic_files.items():
        if file_path.exists():
            synthetic_data = pd.read_csv(file_path)
            if len(synthetic_data) >= malicious_per_type:
                synthetic_resampled = synthetic_data.sample(n=malicious_per_type, random_state=random_seed)
            else:
                print(f"Warning: Only {len(synthetic_data)} {variant} samples available, using all")
                synthetic_resampled = synthetic_data.copy()
            
            synthetic_resampled['data_source'] = f'{batch_name}_{variant}'
            resampled_data[variant] = synthetic_resampled
            
            output_file = resampled_dir / f'{variant}_malicious_{malicious_per_type}.csv'
            synthetic_resampled.to_csv(output_file, index=False)
            print(f"{variant}: {len(synthetic_resampled)} samples → {output_file}")
        else:
            print(f"Warning: Synthetic file not found: {file_path}")
    
    # Resample benign data
    if benign_file.exists():
        benign_data = pd.read_csv(benign_file)
        if len(benign_data) >= benign_count:
            benign_resampled = benign_data.sample(n=benign_count, random_state=random_seed)
        else:
            print(f"Warning: Only {len(benign_data)} benign samples available, using all")
            benign_resampled = benign_data.copy()
        
        benign_resampled['data_source'] = f'{batch_name}_benign'
        resampled_data['benign'] = benign_resampled
        
        output_file = resampled_dir / f'benign_{benign_count}.csv'
        benign_resampled.to_csv(output_file, index=False)
        print(f"Benign: {len(benign_resampled)} samples → {output_file}")
    else:
        print(f"Warning: Benign file not found: {benign_file}")
    
    # Save resampling metadata
    metadata = {
        'batch_name': batch_name,
        'resampling_timestamp': pd.Timestamp.now().isoformat(),
        'target_malicious_per_type': malicious_per_type,
        'target_benign_count': benign_count,
        'random_seed': random_seed,
        'actual_counts': {
            data_type: len(data) for data_type, data in resampled_data.items()
        },
        'resampled_files': {
            'real': f'real_malicious_{malicious_per_type}.csv',
            'rewrite': f'rewrite_malicious_{malicious_per_type}.csv',
            'rewrite_strong': f'rewrite_strong_malicious_{malicious_per_type}.csv',
            'rewrite_weak': f'rewrite_weak_malicious_{malicious_per_type}.csv',
            'benign': f'benign_{benign_count}.csv'
        }
    }
    
    with open(resampled_dir / 'resampling_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Resampling metadata saved to: {resampled_dir / 'resampling_metadata.json'}")
    return resampled_data, resampled_dir

def create_combined_datasets_for_experiments(batch_name, resampled_dir, malicious_per_type, benign_count):
    """Create combined datasets for each experiment type"""
    print(f"\nCreating combined datasets for {batch_name} experiments...")
    
    # Load benign data (same for all experiments)
    benign_file = resampled_dir / f'benign_{benign_count}.csv'
    benign_data = pd.read_csv(benign_file)
    
    experiments = {
        'real_only': ['real'],
        'rewrite': ['real', 'rewrite'],
        'rewrite_strong': ['real', 'rewrite_strong'],
        'rewrite_weak': ['real', 'rewrite_weak']
    }
    
    for exp_name, malicious_types in experiments.items():
        print(f"  Creating {exp_name} dataset...")
        
        # Combine malicious data for this experiment
        malicious_datasets = []
        for mal_type in malicious_types:
            file_path = resampled_dir / f'{mal_type}_malicious_{malicious_per_type}.csv'
            if file_path.exists():
                df = pd.read_csv(file_path)
                malicious_datasets.append(df)
            else:
                print(f"    Warning: {file_path} not found")
        
        if malicious_datasets:
            combined_malicious = pd.concat(malicious_datasets, ignore_index=True)
            
            # Combine with benign data
            combined_data = pd.concat([combined_malicious, benign_data], ignore_index=True)
            
            # Save combined dataset
            output_file = resampled_dir / f'{exp_name}_combined_{len(combined_data)}.csv'
            combined_data.to_csv(output_file, index=False)
            
            print(f"    {exp_name}: {len(combined_data)} samples ({len(combined_malicious)} malicious + {len(benign_data)} benign)")
            print(f"    Saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Resample data for analysis experiments')
    parser.add_argument('--batch', choices=['batch1', 'batch2', 'both'], default='both',
                       help='Which batch to resample (default: both)')
    parser.add_argument('--malicious-per-type', type=int, default=200,
                       help='Number of samples per malicious type (default: 200)')
    parser.add_argument('--benign-count', type=int, default=1000,
                       help='Number of benign samples (default: 1000)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    print("=== DATA RESAMPLING FOR ANALYSIS ===")
    print(f"Configuration:")
    print(f"  Malicious per type: {args.malicious_per_type}")
    print(f"  Benign count: {args.benign_count}")
    print(f"  Random seed: {args.random_seed}")
    print(f"  Target batches: {args.batch}")
    
    batches_to_process = []
    if args.batch == 'both':
        batches_to_process = ['batch1', 'batch2']
    else:
        batches_to_process = [args.batch]
    
    for batch_name in batches_to_process:
        # Resample data
        resampled_data, resampled_dir = resample_batch_data(
            batch_name, 
            args.malicious_per_type, 
            args.benign_count, 
            args.random_seed
        )
        
        # Create combined datasets for experiments
        create_combined_datasets_for_experiments(
            batch_name, 
            resampled_dir, 
            args.malicious_per_type, 
            args.benign_count
        )
    
    print(f"\n=== RESAMPLING COMPLETE ===")
    print(f"Next steps:")
    print(f"1. Run ML analysis on resampled data using modified scripts")
    print(f"2. Run embedding analysis on resampled data")
    print(f"3. Compare results between batches")
    
    print(f"\nResampled data available in:")
    for batch_name in batches_to_process:
        batch_dir = PROJECT_ROOT / f'data/{batch_name}/resampled'
        print(f"  {batch_dir}")

if __name__ == "__main__":
    main()