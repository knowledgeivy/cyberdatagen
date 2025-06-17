# cyberdata/process/data_mixer.py
#
# python data_mixer.py five_email_phishing.csv.gz

import re
import argparse
import gzip
import json
import os
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Add parent directory to path for imports
CURRENT_DIR = Path(__file__).parent
sys.path.append(str(CURRENT_DIR.parent))

from cyberdata.utils.config_manager import get_config_manager
from cyberdata.utils.logger_config import setup_logger

# Set up logger
logger = setup_logger("cyberdata.scripts.data_mixer")

# Load environment variables
load_dotenv()

# Get config manager instance
config_manager = get_config_manager()

logger.info(f"Project root: {config_manager.project_root}")
logger.info(f"Raw data directory: {config_manager.project_root / 'raw'}")
logger.info(f"Large samples directory: {config_manager.large_samples_dir}")


def load_raw_data(file_path: Path, label_column: str = "label") -> pd.DataFrame:
    """
    Load raw data from CSV file (supports .gz compression).
    
    Args:
        file_path (Path): Path to the CSV file
        label_column (str): Name of the label column
        
    Returns:
        pd.DataFrame: Loaded data
    """
    logger.info(f"Loading raw data from: {file_path}")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {file_path}")
    
    try:
        if file_path.suffix == '.gz':
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                df = pd.read_csv(f)
        else:
            df = pd.read_csv(file_path)
        
        logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
        
        # Validate label column exists
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in data. Available columns: {list(df.columns)}")
        
        # Log label distribution
        label_counts = df[label_column].value_counts()
        logger.info(f"Raw data label distribution: {dict(label_counts)}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading raw data: {str(e)}")
        raise


def load_synthetic_data(file_path: Path) -> pd.DataFrame:
    """
    Load synthetic data from JSON file.
    
    Args:
        file_path (Path): Path to the JSON file
        
    Returns:
        pd.DataFrame: Loaded synthetic data as DataFrame
    """
    logger.info(f"Loading synthetic data from: {file_path}")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Synthetic data file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract samples from the JSON structure
        samples = data.get('samples', [])
        if not samples:
            raise ValueError("No samples found in synthetic data file")
        
        # Convert to DataFrame
        df = pd.DataFrame(samples)
        
        logger.info(f"Loaded {len(df)} synthetic samples with columns: {list(df.columns)}")
        
        # Log sample type distribution if available
        if 'sample_type' in df.columns:
            type_counts = df['sample_type'].value_counts()
            logger.info(f"Synthetic data sample type distribution: {dict(type_counts)}")
        
        # Check for label columns
        label_candidates = ['label', 'Label', 'is_attack', 'sample_type']
        found_labels = [col for col in label_candidates if col in df.columns]
        if found_labels:
            logger.info(f"Found potential label columns: {found_labels}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading synthetic data: {str(e)}")
        raise


def stratified_sample_raw(df: pd.DataFrame, 
                         label_column: str, 
                         n_samples: int = 100,
                         random_state: int = 42) -> pd.DataFrame:
    """
    Perform stratified sampling on raw data to get balanced samples.
    
    Args:
        df (pd.DataFrame): Input dataframe
        label_column (str): Name of the label column
        n_samples (int): Total number of samples to extract
        random_state (int): Random state for reproducibility
        
    Returns:
        pd.DataFrame: Stratified sample
    """
    logger.info(f"Performing stratified sampling on raw data: {n_samples} total samples")
    
    # Get unique labels and their counts
    label_counts = df[label_column].value_counts()
    unique_labels = label_counts.index.tolist()
    
    logger.info(f"Available labels: {dict(label_counts)}")
    
    # Calculate samples per label (proportional to original distribution)
    samples_per_label = {}
    total_available = len(df)
    
    for label in unique_labels:
        label_count = label_counts[label]
        proportion = label_count / total_available
        target_samples = max(1, int(n_samples * proportion))  # At least 1 sample per label
        
        # Don't exceed available samples for this label
        target_samples = min(target_samples, label_count)
        samples_per_label[label] = target_samples
    
    # Adjust for rounding differences to hit exact target
    total_targeted = sum(samples_per_label.values())
    if total_targeted > n_samples:
        # Reduce from the largest group
        largest_label = max(samples_per_label.keys(), key=lambda x: samples_per_label[x])
        samples_per_label[largest_label] -= (total_targeted - n_samples)
    elif total_targeted < n_samples:
        # Add missing samples to the largest group (up to available samples)
        missing_samples = n_samples - total_targeted
        largest_label = max(samples_per_label.keys(), key=lambda x: samples_per_label[x])
        
        # Don't exceed available samples for this label
        available_for_largest = label_counts[largest_label] - samples_per_label[largest_label]
        can_add = min(missing_samples, available_for_largest)
        samples_per_label[largest_label] += can_add
        
        # If we still need more samples, distribute to other labels
        remaining_needed = missing_samples - can_add
        if remaining_needed > 0:
            for label in sorted(samples_per_label.keys(), key=lambda x: samples_per_label[x], reverse=True):
                if remaining_needed <= 0:
                    break
                if label != largest_label:  # Skip the one we already maxed out
                    available = label_counts[label] - samples_per_label[label]
                    can_add_here = min(remaining_needed, available)
                    samples_per_label[label] += can_add_here
                    remaining_needed -= can_add_here
    
    logger.info(f"Target samples per label: {samples_per_label}")
    final_total = sum(samples_per_label.values())
    logger.info(f"Final distribution plan: {samples_per_label} (total: {final_total}/{n_samples})")
    
    if final_total != n_samples:
        logger.warning(f"Could not achieve exact target of {n_samples} samples due to data constraints. Getting {final_total} samples instead.")
    
    # Sample from each label
    sampled_dfs = []
    for label, target_count in samples_per_label.items():
        if target_count > 0:
            label_df = df[df[label_column] == label]
            if len(label_df) >= target_count:
                sample_df = label_df.sample(n=target_count, random_state=random_state)
            else:
                sample_df = label_df  # Take all available
            sampled_dfs.append(sample_df)
    
    # Combine all samples
    result = pd.concat(sampled_dfs, ignore_index=True)
    
    # Shuffle the result
    result = result.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    logger.info(f"Stratified sampling completed: {len(result)} samples extracted")
    final_distribution = result[label_column].value_counts()
    logger.info(f"Final raw sample distribution: {dict(final_distribution)}")
    
    return result


def sample_synthetic_data(df: pd.DataFrame, 
                         n_samples: int = 100,
                         random_state: int = 42) -> pd.DataFrame:
    """
    Sample synthetic data, maintaining proportions if sample_type column exists.
    
    Args:
        df (pd.DataFrame): Synthetic data DataFrame
        n_samples (int): Number of samples to extract
        random_state (int): Random state for reproducibility
        
    Returns:
        pd.DataFrame: Sampled synthetic data
    """
    logger.info(f"Sampling synthetic data: {n_samples} samples")
    
    if len(df) <= n_samples:
        logger.info(f"Synthetic data has {len(df)} samples, using all")
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # If we have sample_type, try to maintain proportions
    if 'sample_type' in df.columns:
        type_counts = df['sample_type'].value_counts()
        logger.info(f"Original synthetic type distribution: {dict(type_counts)}")
        
        # Calculate samples per type
        samples_per_type = {}
        total_available = len(df)
        
        for sample_type in type_counts.index:
            type_count = type_counts[sample_type]
            proportion = type_count / total_available
            target_samples = max(1, int(n_samples * proportion))
            target_samples = min(target_samples, type_count)
            samples_per_type[sample_type] = target_samples
        
        # Adjust if over target
        total_targeted = sum(samples_per_type.values())
        if total_targeted > n_samples:
            largest_type = max(samples_per_type.keys(), key=lambda x: samples_per_type[x])
            samples_per_type[largest_type] -= (total_targeted - n_samples)
        
        logger.info(f"Target synthetic samples per type: {samples_per_type}")
        
        # Sample from each type
        sampled_dfs = []
        for sample_type, target_count in samples_per_type.items():
            if target_count > 0:
                type_df = df[df['sample_type'] == sample_type]
                if len(type_df) >= target_count:
                    sample_df = type_df.sample(n=target_count, random_state=random_state)
                else:
                    sample_df = type_df
                sampled_dfs.append(sample_df)
        
        result = pd.concat(sampled_dfs, ignore_index=True)
    else:
        # Random sampling if no sample_type column
        result = df.sample(n=n_samples, random_state=random_state)
    
    # Shuffle the result
    result = result.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    logger.info(f"Synthetic sampling completed: {len(result)} samples")
    if 'sample_type' in result.columns:
        final_distribution = result['sample_type'].value_counts()
        logger.info(f"Final synthetic sample distribution: {dict(final_distribution)}")
    
    return result


def align_schemas(raw_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align schemas between raw and synthetic data for concatenation.
    
    Args:
        raw_df (pd.DataFrame): Raw data DataFrame
        synthetic_df (pd.DataFrame): Synthetic data DataFrame
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Aligned DataFrames
    """
    logger.info("Aligning schemas between raw and synthetic data")
    
    raw_cols = set(raw_df.columns)
    synthetic_cols = set(synthetic_df.columns)
    
    logger.info(f"Raw data columns: {sorted(raw_cols)}")
    logger.info(f"Synthetic data columns: {sorted(synthetic_cols)}")
    
    # Find common columns
    common_cols = raw_cols.intersection(synthetic_cols)
    raw_only_cols = raw_cols - synthetic_cols
    synthetic_only_cols = synthetic_cols - raw_cols
    
    logger.info(f"Common columns: {sorted(common_cols)}")
    if raw_only_cols:
        logger.warning(f"Columns only in raw data: {sorted(raw_only_cols)}")
    if synthetic_only_cols:
        logger.warning(f"Columns only in synthetic data: {sorted(synthetic_only_cols)}")
    
    # Create aligned DataFrames
    aligned_raw = raw_df.copy()
    aligned_synthetic = synthetic_df.copy()
    
    # Add missing columns with default values
    for col in synthetic_only_cols:
        if col not in ['sample_type', 'is_attack', 'generation_task_id', 'generation_timestamp']:
            # Only add if it's not a metadata column
            aligned_raw[col] = None
            logger.info(f"Added column '{col}' to raw data with None values")
    
    for col in raw_only_cols:
        aligned_synthetic[col] = None
        logger.info(f"Added column '{col}' to synthetic data with None values")
    
    # Ensure same column order
    all_cols = sorted(set(aligned_raw.columns).union(set(aligned_synthetic.columns)))
    aligned_raw = aligned_raw.reindex(columns=all_cols)
    aligned_synthetic = aligned_synthetic.reindex(columns=all_cols)
    
    logger.info(f"Schema alignment completed. Final columns: {all_cols}")
    
    return aligned_raw, aligned_synthetic


def create_mixed_dataset(raw_df: pd.DataFrame, 
                        synthetic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a mixed dataset with synthetic column marker.
    
    Args:
        raw_df (pd.DataFrame): Raw data samples
        synthetic_df (pd.DataFrame): Synthetic data samples
        
    Returns:
        pd.DataFrame: Mixed dataset with synthetic column
    """
    logger.info("Creating mixed dataset with synthetic marker")
    
    # Align schemas
    aligned_raw, aligned_synthetic = align_schemas(raw_df, synthetic_df)
    
    # Add synthetic marker column
    aligned_raw['synthetic'] = 0  # Real data
    aligned_synthetic['synthetic'] = 1  # Synthetic data
    
    # Concatenate
    mixed_df = pd.concat([aligned_raw, aligned_synthetic], ignore_index=True)
    
    # Shuffle the combined dataset
    mixed_df = mixed_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Mixed dataset created: {len(mixed_df)} total samples")
    logger.info(f"Real samples: {len(aligned_raw)}, Synthetic samples: {len(aligned_synthetic)}")
    
    # Log synthetic distribution
    synthetic_distribution = mixed_df['synthetic'].value_counts()
    logger.info(f"Synthetic marker distribution: {dict(synthetic_distribution)}")
    
    return mixed_df


def save_mixed_dataset(mixed_df: pd.DataFrame, 
                      output_path: Path,
                      raw_file_name: str,
                      synthetic_file_name: str) -> None:
    """
    Save the mixed dataset with metadata.
    
    Args:
        mixed_df (pd.DataFrame): Mixed dataset
        output_path (Path): Output file path
        raw_file_name (str): Original raw file name
        synthetic_file_name (str): Original synthetic file name
    """
    logger.info(f"Saving mixed dataset to: {output_path}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create metadata
    metadata = {
        'total_samples': len(mixed_df),
        'real_samples': int((mixed_df['synthetic'] == 0).sum()),
        'synthetic_samples': int((mixed_df['synthetic'] == 1).sum()),
        'columns': list(mixed_df.columns),
        'source_files': {
            'raw_data': raw_file_name,
            'synthetic_data': synthetic_file_name
        },
        'mixing_timestamp': pd.Timestamp.now().isoformat(),
        'description': 'Mixed dataset combining real-world and synthetic cybersecurity data'
    }
    
    # Save as CSV (more common for mixed datasets)
    if output_path.suffix.lower() == '.csv':
        mixed_df.to_csv(output_path, index=False)
        logger.info(f"Saved mixed dataset as CSV: {output_path}")
        
        # Save metadata separately
        metadata_path = output_path.with_suffix('.metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata: {metadata_path}")
    
    else:
        # Save as JSON with metadata included
        output_data = {
            'data': mixed_df.to_dict('records'),
            'metadata': metadata
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Saved mixed dataset as JSON: {output_path}")
    
    # Log summary statistics
    logger.info("="*50)
    logger.info("MIXED DATASET SUMMARY")
    logger.info("="*50)
    logger.info(f"Total samples: {metadata['total_samples']}")
    logger.info(f"Real samples: {metadata['real_samples']} ({metadata['real_samples']/metadata['total_samples']*100:.1f}%)")
    logger.info(f"Synthetic samples: {metadata['synthetic_samples']} ({metadata['synthetic_samples']/metadata['total_samples']*100:.1f}%)")
    logger.info(f"Columns: {len(metadata['columns'])}")
    logger.info(f"Output file: {output_path}")
    logger.info("="*50)


def check_schema_compatibility(raw_df: pd.DataFrame, synthetic_file: Path) -> Tuple[bool, int, set, set]:
    """
    Check if a synthetic file has compatible schema with raw data.
    
    Args:
        raw_df (pd.DataFrame): Raw data DataFrame
        synthetic_file (Path): Path to synthetic JSON file
        
    Returns:
        Tuple[bool, int, set, set]: (is_compatible, sample_count, common_cols, synthetic_only_cols)
    """
    try:
        with open(synthetic_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = data.get('samples', [])
        if not samples:
            return False, 0, set(), set()
        
        # Get synthetic columns from first sample
        synthetic_cols = set(samples[0].keys())
        raw_cols = set(raw_df.columns)
        
        # Remove synthetic-specific metadata columns
        metadata_cols = {'sample_type', 'is_attack', 'generation_task_id', 'generation_timestamp', 'synthetic'}
        synthetic_cols_clean = synthetic_cols - metadata_cols
        
        # Calculate compatibility metrics
        common_cols = raw_cols.intersection(synthetic_cols_clean)
        synthetic_only_cols = synthetic_cols_clean - raw_cols
        raw_only_cols = raw_cols - synthetic_cols_clean
        
        # Consider compatible if:
        # 1. At least 70% of raw columns are present in synthetic
        # 2. Synthetic doesn't have too many extra non-metadata columns
        raw_coverage = len(common_cols) / len(raw_cols) if raw_cols else 0
        extra_cols_ratio = len(synthetic_only_cols) / len(synthetic_cols_clean) if synthetic_cols_clean else 1
        
        is_compatible = raw_coverage >= 0.7 and extra_cols_ratio <= 0.3
        
        logger.debug(f"Schema compatibility for {synthetic_file.name}:")
        logger.debug(f"  Raw columns: {len(raw_cols)}, Synthetic columns: {len(synthetic_cols_clean)}")
        logger.debug(f"  Common: {len(common_cols)}, Raw coverage: {raw_coverage:.2f}")
        logger.debug(f"  Extra synthetic cols: {len(synthetic_only_cols)}, Ratio: {extra_cols_ratio:.2f}")
        logger.debug(f"  Compatible: {is_compatible}")
        
        return is_compatible, len(samples), common_cols, synthetic_only_cols
        
    except Exception as e:
        logger.debug(f"Error checking schema compatibility for {synthetic_file}: {e}")
        return False, 0, set(), set()


def find_best_synthetic_file(raw_df: pd.DataFrame, 
                           problem_area: str = None, 
                           problem_nature: str = None) -> Optional[Path]:
    """
    Find the best synthetic data file that matches the raw data schema.
    
    Args:
        raw_df (pd.DataFrame): Raw data DataFrame to match schema against
        problem_area (str): Problem area (optional)
        problem_nature (str): Problem nature (optional)
        
    Returns:
        Optional[Path]: Path to best matching synthetic file
    """
    logger.info("Searching for schema-compatible synthetic data files...")
    
    # Get all synthetic files to check
    search_dirs = []
    
    if problem_area:
        # Search in specific area first
        area_variations = [
            problem_area.replace(' ', '_'),
            problem_area.replace(' ', '_').replace('(', '').replace(')', ''),
            problem_area.replace(' ', '').replace('-', '_')
        ]
        
        for area_var in area_variations:
            area_dir = config_manager.large_samples_dir / area_var
            if area_dir.exists():
                search_dirs.append(area_dir)
                break
    
    # Always include full search as fallback
    search_dirs.append(config_manager.large_samples_dir)
    
    compatible_files = []
    
    # Search for compatible files
    for search_dir in search_dirs:
        synthetic_files = list(search_dir.rglob('*_large.json'))
        
        for synthetic_file in synthetic_files:
            is_compatible, sample_count, common_cols, extra_cols = check_schema_compatibility(raw_df, synthetic_file)
            
            if is_compatible:
                # Calculate compatibility score
                raw_cols = set(raw_df.columns)
                coverage_score = len(common_cols) / len(raw_cols) if raw_cols else 0
                
                # Bonus for exact name match
                name_bonus = 0
                if problem_nature:
                    nature_clean = problem_nature.lower().replace(' ', '_').replace('-', '_')
                    file_name_clean = synthetic_file.stem.lower().replace(' ', '_').replace('-', '_')
                    if nature_clean in file_name_clean:
                        name_bonus = 0.2
                
                # Penalty for extra columns
                extra_penalty = len(extra_cols) * 0.02
                
                compatibility_score = coverage_score + name_bonus - extra_penalty
                
                compatible_files.append({
                    'path': synthetic_file,
                    'score': compatibility_score,
                    'sample_count': sample_count,
                    'common_cols': len(common_cols),
                    'extra_cols': len(extra_cols),
                    'coverage': coverage_score
                })
                
                logger.info(f"Compatible file found: {synthetic_file.relative_to(config_manager.large_samples_dir)}")
                logger.info(f"  Score: {compatibility_score:.3f}, Coverage: {coverage_score:.2f}, Samples: {sample_count}")
    
    if not compatible_files:
        logger.warning("No schema-compatible synthetic files found!")
        logger.warning("Available synthetic files:")
        all_files = list(config_manager.large_samples_dir.rglob('*_large.json'))
        for file_path in all_files[:10]:  # Show first 10
            logger.warning(f"  - {file_path.relative_to(config_manager.large_samples_dir)}")
        return None
    
    # Sort by compatibility score (highest first)
    compatible_files.sort(key=lambda x: x['score'], reverse=True)
    
    best_file = compatible_files[0]
    logger.info(f"Selected best match: {best_file['path'].relative_to(config_manager.large_samples_dir)}")
    logger.info(f"  Final score: {best_file['score']:.3f}")
    logger.info(f"  Schema coverage: {best_file['coverage']:.1%}")
    logger.info(f"  Sample count: {best_file['sample_count']}")
    logger.info(f"  Common columns: {best_file['common_cols']}")
    logger.info(f"  Extra columns: {best_file['extra_cols']}")
    
    # Show alternatives if available
    if len(compatible_files) > 1:
        logger.info("Other compatible options:")
        for alt_file in compatible_files[1:3]:  # Show top 2 alternatives
            logger.info(f"  - {alt_file['path'].name} (score: {alt_file['score']:.3f})")
    
    return best_file['path']


def infer_problem_from_raw_file(raw_file: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Infer problem area and nature from raw file name.
    
    Args:
        raw_file (str): Raw file name
        
    Returns:
        Tuple[Optional[str], Optional[str]]: (problem_area, problem_nature)
    """
    file_name = Path(raw_file).stem.lower()
    
    # Remove common extensions and numbers
    file_name = file_name.replace('.csv', '').replace('.gz', '')
    file_name = re.sub(r'[0-9]+', '', file_name)  # Remove numbers
    
    # Common mappings
    area_mappings = {
        'phishing': 'Social Engineering',
        'email': 'Social Engineering', 
        'spam': 'Social Engineering',
        'social': 'Social Engineering',
        'network': 'Enterprise',
        'intrusion': 'Enterprise',
        'malware': 'Enterprise',
        'ddos': 'Enterprise',
        'cloud': 'Cloud',
        'web': 'Enterprise'
    }
    
    nature_mappings = {
        'phishing': 'phishing',
        'email': 'phishing',
        'spam': 'phishing', 
        'intrusion': 'network_intrusion',
        'network': 'network_intrusion',
        'malware': 'malware',
        'ddos': 'ddos',
        'web': 'web_attack'
    }
    
    # Find matches
    inferred_area = None
    inferred_nature = None
    
    for keyword, area in area_mappings.items():
        if keyword in file_name:
            inferred_area = area
            break
    
    for keyword, nature in nature_mappings.items():
        if keyword in file_name:
            inferred_nature = nature
            break
    
    logger.info(f"Inferred from '{raw_file}': area='{inferred_area}', nature='{inferred_nature}'")
    return inferred_area, inferred_nature


def main(raw_file: str = "five_email_phishing.csv.gz",
         synthetic_file: str = None,
         problem_area: str = None,
         problem_nature: str = None,
         label_column: str = "label",
         n_raw_samples: int = 100,
         n_synthetic_samples: int = 100,
         output_file: str = None,
         output_format: str = "csv"):
    """
    Main function to mix raw and synthetic data.
    
    Args:
        raw_file (str): Path to raw CSV file
        synthetic_file (str): Path to synthetic JSON file (optional)
        problem_area (str): Problem area for finding synthetic file (optional)
        problem_nature (str): Problem nature for finding synthetic file (optional)
        label_column (str): Name of the label column in raw data
        n_raw_samples (int): Number of raw samples to extract
        n_synthetic_samples (int): Number of synthetic samples to extract
        output_file (str): Output file path (optional)
        output_format (str): Output format ('csv' or 'json')
    """
    logger.info("Starting intelligent data mixing process")
    logger.info(f"Parameters: raw_samples={n_raw_samples}, synthetic_samples={n_synthetic_samples}")
    
    try:
        # Define raw data path
        raw_dir = config_manager.project_root / "raw"
        raw_path = raw_dir / raw_file
        
        # Load raw data first to understand schema
        raw_df = load_raw_data(raw_path, label_column)
        logger.info(f"Raw data schema: {list(raw_df.columns)}")
        
        # Infer problem area/nature from filename if not provided
        if not problem_area or not problem_nature:
            inferred_area, inferred_nature = infer_problem_from_raw_file(raw_file)
            problem_area = problem_area or inferred_area
            problem_nature = problem_nature or inferred_nature
            if problem_area or problem_nature:
                logger.info(f"Using inferred parameters: area='{problem_area}', nature='{problem_nature}'")
        
        # Find schema-compatible synthetic data file
        if synthetic_file:
            # Use specified synthetic file, but verify compatibility
            if not synthetic_file.startswith('/'):
                # Relative path, search in large_samples_dir
                synthetic_path = config_manager.large_samples_dir / synthetic_file
                if not synthetic_path.exists():
                    # Try adding .json extension
                    synthetic_path = config_manager.large_samples_dir / f"{synthetic_file}.json"
                if not synthetic_path.exists():
                    # Search recursively
                    found_files = list(config_manager.large_samples_dir.rglob(f"*{synthetic_file}*"))
                    if found_files:
                        synthetic_path = found_files[0]
                    else:
                        raise FileNotFoundError(f"Synthetic file not found: {synthetic_file}")
            else:
                synthetic_path = Path(synthetic_file)
            
            # Verify schema compatibility
            is_compatible, sample_count, common_cols, extra_cols = check_schema_compatibility(raw_df, synthetic_path)
            if not is_compatible:
                logger.warning(f"Specified synthetic file has poor schema compatibility!")
                logger.warning(f"Consider using automatic selection for better results.")
                # Continue anyway but warn user
        else:
            # Automatically find best schema-compatible file
            synthetic_path = find_best_synthetic_file(raw_df, problem_area, problem_nature)
            if not synthetic_path:
                # Fallback: try any available file
                logger.warning("No schema-compatible files found, trying any available synthetic file...")
                all_files = list(config_manager.large_samples_dir.rglob('*_large.json'))
                if all_files:
                    synthetic_path = all_files[0]
                    logger.warning(f"Using fallback file: {synthetic_path.relative_to(config_manager.large_samples_dir)}")
                else:
                    raise FileNotFoundError("No synthetic data files found in large_samples directory")
        
        logger.info(f"Selected synthetic file: {synthetic_path.relative_to(config_manager.large_samples_dir)}")
        
        # Load synthetic data
        synthetic_df = load_synthetic_data(synthetic_path)
        logger.info(f"Synthetic data schema: {list(synthetic_df.columns)}")
        
        # Final schema compatibility check
        raw_cols = set(raw_df.columns)
        synthetic_cols = set(synthetic_df.columns)
        common_cols = raw_cols.intersection(synthetic_cols)
        coverage = len(common_cols) / len(raw_cols) if raw_cols else 0
        
        logger.info(f"Schema compatibility: {coverage:.1%} coverage ({len(common_cols)}/{len(raw_cols)} columns)")
        if coverage < 0.5:
            logger.warning("Low schema compatibility! Mixed dataset may have many missing values.")
        
        # Sample raw data
        sampled_raw = stratified_sample_raw(raw_df, label_column, n_raw_samples)
        
        # Sample synthetic data
        sampled_synthetic = sample_synthetic_data(synthetic_df, n_synthetic_samples)
        
        # Create mixed dataset
        mixed_df = create_mixed_dataset(sampled_raw, sampled_synthetic)
        
        # Determine output path
        if output_file:
            output_path = Path(output_file)
        else:
            # Generate descriptive output filename
            raw_name = Path(raw_file).stem.replace('.csv', '').replace('.gz', '')
            synthetic_name = synthetic_path.stem.replace('_large', '')
            # Clean up names for filename
            raw_name = re.sub(r'[^a-zA-Z0-9_]', '_', raw_name)
            synthetic_name = re.sub(r'[^a-zA-Z0-9_]', '_', synthetic_name)
            output_name = f"mixed_{raw_name}_{synthetic_name}_{n_raw_samples}+{n_synthetic_samples}.{output_format}"
            output_path = config_manager.data_dir / "mixed" / output_name
        
        # Save mixed dataset
        save_mixed_dataset(
            mixed_df, 
            output_path,
            raw_file,
            str(synthetic_path.relative_to(config_manager.project_root))
        )
        
        logger.info("Data mixing completed successfully!")
        logger.info(f"Final dataset: {len(mixed_df)} samples with {len(mixed_df.columns)} columns")
        logger.info(f"Schema coverage: {coverage:.1%}")
        
    except Exception as e:
        logger.error(f"Error in data mixing process: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Intelligently mix raw data with schema-compatible synthetic data")
    
    # Required arguments
    parser.add_argument('raw_file', nargs='?', default='five_email_phishing.csv.gz', 
                       help='Raw CSV file name in raw/ directory (default: five_email_phishing.csv.gz)')
    
    # Synthetic data selection (all optional - will auto-infer if not provided)
    parser.add_argument('--synthetic-file', help='Specific synthetic JSON file path (optional)')
    parser.add_argument('--problem-area', help='Problem area to find synthetic data (optional - will auto-infer)')
    parser.add_argument('--problem-nature', help='Problem nature to find synthetic data (optional - will auto-infer)')
    
    # Data parameters
    parser.add_argument('--label-column', default='label', help='Label column name in raw data (default: label)')
    parser.add_argument('--n-raw-samples', type=int, default=500, help='Number of raw samples to extract (default: 100)')
    parser.add_argument('--n-synthetic-samples', type=int, default=500, help='Number of synthetic samples to extract (default: 100)')
    
    # Output parameters
    parser.add_argument('--output-file', help='Output file path (optional - will auto-generate)')
    parser.add_argument('--output-format', choices=['csv', 'json'], default='csv', help='Output format (default: csv)')
    
    args = parser.parse_args()
    
    # Show usage examples if only filename provided
    if len(sys.argv) == 2:  # Only script name and raw_file provided
        print(f"\n✨ Auto-detection mode enabled for: {args.raw_file}")
        print("The script will automatically:")
        print("  1. Infer problem area/nature from filename")
        print("  2. Find schema-compatible synthetic data")
        print("  3. Generate descriptive output filename")
        print("\nFor manual control, you can specify:")
        print("  --problem-area 'Social Engineering' --problem-nature 'phishing'")
        print("  --synthetic-file 'specific_file.json'")
        print("")
    
    main(
        raw_file=args.raw_file,
        synthetic_file=args.synthetic_file,
        problem_area=args.problem_area,
        problem_nature=args.problem_nature,
        label_column=args.label_column,
        n_raw_samples=args.n_raw_samples,
        n_synthetic_samples=args.n_synthetic_samples,
        output_file=args.output_file,
        output_format=args.output_format
    )