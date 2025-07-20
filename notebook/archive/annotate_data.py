#!/usr/bin/env python3
"""
Data Annotation Script for CyberDataGen Project
Adds unique IDs to all data types and saves to annotated directory

Usage:
    python notebook/annotate_data.py
"""

import pandas as pd
import numpy as np
import gzip
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# Add parent directory to path for imports
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

# Configuration
DATASET_NAME = "email_phishing_CEAS-08"  # Change this to your dataset name
OUTPUT_DIR = PROJECT_ROOT / "data" / "annotated"

# Input file paths based on your project structure
INPUT_FILES = {
    # Real data (original training data)
    "real": PROJECT_ROOT / "raw" / f"{DATASET_NAME}_train.csv.gz",
    
    # Synthetic data files (rewritten versions)
    "syn_rewrite": PROJECT_ROOT / "data" / "rewrite" / f"{DATASET_NAME}_malicious_original_rewritten.csv.gz",
    "syn_rewrite_strong": PROJECT_ROOT / "data" / "rewrite" / f"{DATASET_NAME}_malicious_strong_rewritten.csv.gz", 
    "syn_rewrite_weak": PROJECT_ROOT / "data" / "rewrite" / f"{DATASET_NAME}_malicious_weak_rewritten.csv.gz"
}

def create_output_directory():
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created output directory: {OUTPUT_DIR}")

def load_data(file_path: Path) -> pd.DataFrame:
    """Load data from compressed or uncompressed CSV file."""
    if not file_path.exists():
        print(f"⚠ Warning: File not found: {file_path}")
        return pd.DataFrame()
    
    try:
        if file_path.suffix == '.gz':
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                df = pd.read_csv(f)
        else:
            df = pd.read_csv(file_path)
        
        print(f"✓ Loaded {len(df)} rows from {file_path.name}")
        return df
    
    except Exception as e:
        print(f"✗ Error loading {file_path}: {str(e)}")
        return pd.DataFrame()

def generate_unique_ids(data_type: str, df: pd.DataFrame) -> List[str]:
    """Generate unique IDs based on data type and label."""
    unique_ids = []
    
    # Separate malicious and benign data
    if 'label' in df.columns:
        malicious_mask = df['label'] == 1
        benign_mask = df['label'] == 0
        
        malicious_count = malicious_mask.sum()
        benign_count = benign_mask.sum()
        
        print(f"  - {data_type}: {malicious_count} malicious, {benign_count} benign")
        
        # Generate IDs for malicious data
        mal_counter = 1
        ben_counter = 1
        
        for idx, row in df.iterrows():
            if row['label'] == 1:  # Malicious
                unique_id = f"{data_type}_mal_{mal_counter:03d}"
                mal_counter += 1
            else:  # Benign
                unique_id = f"{data_type}_ben_{ben_counter:03d}"
                ben_counter += 1
            
            unique_ids.append(unique_id)
    else:
        # If no label column, assume all are malicious (for synthetic rewrite data)
        print(f"  - {data_type}: {len(df)} samples (assuming malicious)")
        for i in range(len(df)):
            unique_id = f"{data_type}_mal_{i+1:03d}"
            unique_ids.append(unique_id)
    
    return unique_ids

def annotate_single_dataset(data_type: str, file_path: Path) -> pd.DataFrame:
    """Annotate a single dataset with unique IDs."""
    print(f"\n📝 Processing {data_type} data...")
    
    # Load data
    df = load_data(file_path)
    if df.empty:
        return df
    
    # Generate unique IDs
    unique_ids = generate_unique_ids(data_type, df)
    
    # Add unique ID column
    df_annotated = df.copy()
    df_annotated.insert(0, 'unique_id', unique_ids)
    
    # Add data type column for easy filtering
    df_annotated['data_type'] = data_type
    
    # Add source file information
    df_annotated['source_file'] = file_path.name
    
    print(f"✓ Generated {len(unique_ids)} unique IDs for {data_type}")
    
    return df_annotated

def save_annotated_data(df: pd.DataFrame, data_type: str):
    """Save annotated data to output directory."""
    if df.empty:
        print(f"⚠ Skipping save for {data_type} (no data)")
        return
    
    output_file = OUTPUT_DIR / f"{DATASET_NAME}_{data_type}_annotated.csv.gz"
    
    try:
        with gzip.open(output_file, 'wt', encoding='utf-8') as f:
            df.to_csv(f, index=False)
        
        print(f"✓ Saved {len(df)} rows to {output_file.name}")
        
        # Print sample IDs for verification
        sample_ids = df['unique_id'].head(3).tolist()
        print(f"  Sample IDs: {sample_ids}")
        
    except Exception as e:
        print(f"✗ Error saving {data_type}: {str(e)}")

def create_combined_dataset(annotated_datasets: Dict[str, pd.DataFrame]):
    """Create a combined dataset with all annotated data."""
    print(f"\n🔄 Creating combined dataset...")
    
    non_empty_datasets = {k: v for k, v in annotated_datasets.items() if not v.empty}
    
    if not non_empty_datasets:
        print("⚠ No data to combine")
        return
    
    # Combine all datasets
    combined_df = pd.concat(non_empty_datasets.values(), ignore_index=True)
    
    # Sort by unique_id for better organization
    combined_df = combined_df.sort_values('unique_id').reset_index(drop=True)
    
    # Save combined dataset
    combined_file = OUTPUT_DIR / f"{DATASET_NAME}_all_annotated.csv.gz"
    
    try:
        with gzip.open(combined_file, 'wt', encoding='utf-8') as f:
            combined_df.to_csv(f, index=False)
        
        print(f"✓ Saved combined dataset: {len(combined_df)} total rows")
        print(f"✓ File: {combined_file.name}")
        
        # Print summary statistics
        print(f"\n📊 Combined Dataset Summary:")
        print(f"  Total samples: {len(combined_df)}")
        
        if 'label' in combined_df.columns:
            label_counts = combined_df['label'].value_counts()
            print(f"  Malicious: {label_counts.get(1, 0)}")
            print(f"  Benign: {label_counts.get(0, 0)}")
        
        data_type_counts = combined_df['data_type'].value_counts()
        print(f"  Data type distribution:")
        for dtype, count in data_type_counts.items():
            print(f"    {dtype}: {count}")
            
    except Exception as e:
        print(f"✗ Error saving combined dataset: {str(e)}")

def print_summary_report(annotated_datasets: Dict[str, pd.DataFrame]):
    """Print a summary report of the annotation process."""
    print(f"\n" + "="*60)
    print(f"📋 ANNOTATION SUMMARY REPORT")
    print(f"="*60)
    
    total_samples = 0
    total_malicious = 0
    total_benign = 0
    
    for data_type, df in annotated_datasets.items():
        if df.empty:
            print(f"{data_type:20}: No data")
            continue
            
        samples = len(df)
        total_samples += samples
        
        if 'label' in df.columns:
            malicious = (df['label'] == 1).sum()
            benign = (df['label'] == 0).sum()
            total_malicious += malicious
            total_benign += benign
            print(f"{data_type:20}: {samples:4d} total ({malicious:3d} mal, {benign:3d} ben)")
        else:
            # Assume all synthetic rewrite data is malicious
            total_malicious += samples
            print(f"{data_type:20}: {samples:4d} total (assumed malicious)")
    
    print(f"{"="*40}")
    print(f"{"Total":20}: {total_samples:4d} total ({total_malicious:3d} mal, {total_benign:3d} ben)")
    
    print(f"\n📁 Output files saved in: {OUTPUT_DIR}")
    print(f"✓ Individual annotated files for each data type")
    print(f"✓ Combined file: {DATASET_NAME}_all_annotated.csv.gz")

def main():
    """Main annotation process."""
    parser = argparse.ArgumentParser(description="Annotate datasets with unique IDs")
    parser.add_argument("--dataset-name", type=str, default=DATASET_NAME, 
                       help=f"Dataset name (default: {DATASET_NAME})")
    parser.add_argument("--skip-missing", action="store_true",
                       help="Skip missing files instead of showing warnings")
    
    args = parser.parse_args()
    
    # Update global dataset name if provided
    global DATASET_NAME, INPUT_FILES
    DATASET_NAME = args.dataset_name
    
    # Update input file paths with new dataset name
    INPUT_FILES = {
        "real": PROJECT_ROOT / "raw" / f"{DATASET_NAME}_train.csv.gz",
        "syn_rewrite": PROJECT_ROOT / "data" / "rewrite" / f"{DATASET_NAME}_malicious_original_rewritten.csv.gz",
        "syn_rewrite_strong": PROJECT_ROOT / "data" / "rewrite" / f"{DATASET_NAME}_malicious_strong_rewritten.csv.gz", 
        "syn_rewrite_weak": PROJECT_ROOT / "data" / "rewrite" / f"{DATASET_NAME}_malicious_weak_rewritten.csv.gz"
    }
    
    print("🏷️  CyberDataGen - Data Annotation Script")
    print("="*60)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Create output directory
    create_output_directory()
    
    # Process each dataset
    annotated_datasets = {}
    
    for data_type, file_path in INPUT_FILES.items():
        if not file_path.exists() and args.skip_missing:
            continue
            
        annotated_df = annotate_single_dataset(data_type, file_path)
        annotated_datasets[data_type] = annotated_df
        
        # Save individual annotated dataset
        save_annotated_data(annotated_df, data_type)
    
    # Create combined dataset
    create_combined_dataset(annotated_datasets)
    
    # Print summary report
    print_summary_report(annotated_datasets)
    
    print(f"\n🎉 Annotation process completed successfully!")

if __name__ == "__main__":
    main()