# cyberdata/process/real_data_processor_augment.py

import gzip
import json
import pandas as pd
import random
import sys
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

# Add parent directory to path for imports
CURRENT_DIR = Path(__file__).parent
sys.path.append(str(CURRENT_DIR.parent))

from cyberdata.utils.config_manager import get_config_manager
from cyberdata.utils.llm_invoke import process_llm_request
from cyberdata.utils.logger_config import setup_logger
from cyberdata.utils.prompt_loader import load_prompt

# Set up logger
logger = setup_logger("cyberdata.scripts.real_data_processor_augment")

# Load environment variables
load_dotenv()

# Configuration
MODEL_NAME = "gpt-4.1-mini"
DATA_PATH = "email_phishing_CEAS-08_train.csv.gz"
N_GROUPS = 100
N_AUGMENT = 10

# Get config manager instance
config_manager = get_config_manager()

logger.info(f"Using model: {MODEL_NAME}")
logger.info(f"Configuration: N_GROUPS={N_GROUPS}, N_AUGMENT={N_AUGMENT}")


def load_raw_data(file_path: Path) -> pd.DataFrame:
    """Load raw data from CSV file."""
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
        
        # Log label distribution
        if 'label' in df.columns:
            label_counts = df['label'].value_counts()
            logger.info(f"Label distribution: {dict(label_counts)}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading raw data: {str(e)}")
        raise


def sample_malicious_data(df: pd.DataFrame) -> pd.DataFrame:
    """Sample only malicious data (label = 1)."""
    logger.info("Sampling malicious data...")
    
    malicious_df = df[df['label'] == 1].copy()
    logger.info(f"Found {len(malicious_df)} malicious samples")
    
    if len(malicious_df) == 0:
        raise ValueError("No malicious samples found in dataset")
    
    return malicious_df


def create_groups(malicious_df: pd.DataFrame, n_groups: int) -> List[pd.DataFrame]:
    """Split malicious data into N_GROUPS subgroups."""
    logger.info(f"Creating {n_groups} groups from {len(malicious_df)} malicious samples")
    
    # Shuffle the data
    shuffled_df = malicious_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate group size
    group_size = len(shuffled_df) // n_groups
    if group_size == 0:
        raise ValueError(f"Not enough samples to create {n_groups} groups")
    
    groups = []
    for i in range(n_groups):
        start_idx = i * group_size
        if i == n_groups - 1:  # Last group gets remaining samples
            end_idx = len(shuffled_df)
        else:
            end_idx = (i + 1) * group_size
        
        group = shuffled_df.iloc[start_idx:end_idx].copy()
        groups.append(group)
        logger.debug(f"Group {i+1}: {len(group)} samples")
    
    logger.info(f"Created {len(groups)} groups with average size: {group_size}")
    return groups


def sample_example_from_group(group: pd.DataFrame) -> Dict:
    """Random sample one example from a group."""
    sampled_row = group.sample(n=1).iloc[0]
    
    # Convert to dictionary and clean up
    example = {
        'subject': str(sampled_row.get('subject', '')),
        'body': str(sampled_row.get('body', '')),
        'label': int(sampled_row.get('label', 1)),
        'source': str(sampled_row.get('source', ''))
    }
    
    return example


def generate_augmented_samples(example: Dict, n_augment: int, group_id: int) -> List[Dict]:
    """Use LLM to generate N_AUGMENT samples based on the example."""
    logger.info(f"Generating {n_augment} augmented samples for group {group_id}")
    
    try:
        # Prepare context for the prompt
        context = {
            'example': example,
            'n_augment': n_augment,
            'group_id': group_id
        }
        
        context_json = json.dumps(context, indent=2)
        
        # Load augmentation prompts
        system_prompt = load_prompt(
            "augment_generation",
            "prompts.system.template"
        )
        
        user_prompt = load_prompt(
            "augment_generation", 
            "prompts.user.template",
            context_json=context_json,
            n_augment=n_augment
        )
        
        # Call LLM for augmentation
        response_content = process_llm_request(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=MODEL_NAME,
            temperature=0.8  # Higher temperature for diversity
        )
        
        # Parse response
        augmented_samples = parse_augmentation_response(response_content, group_id)
        
        logger.info(f"Successfully generated {len(augmented_samples)} samples for group {group_id}")
        return augmented_samples
        
    except Exception as e:
        logger.error(f"Error generating augmented samples for group {group_id}: {str(e)}")
        return []


def parse_augmentation_response(response_content: str, group_id: int) -> List[Dict]:
    """Parse LLM augmentation response."""
    try:
        # Clean markdown code blocks
        if response_content.startswith('```'):
            first_backticks_end = response_content.find('\n', 3)
            if first_backticks_end != -1:
                last_backticks_start = response_content.rfind('```')
                if last_backticks_start > first_backticks_end:
                    response_content = response_content[first_backticks_end + 1:last_backticks_start].strip()
        
        parsed = json.loads(response_content)
        samples = parsed.get('augmented_samples', [])
        
        if not isinstance(samples, list):
            logger.warning(f"Invalid samples format in response for group {group_id}")
            return []
        
        # Validate and clean samples
        valid_samples = []
        for i, sample in enumerate(samples):
            if isinstance(sample, dict) and 'subject' in sample and 'body' in sample:
                # Ensure required fields
                clean_sample = {
                    'subject': str(sample.get('subject', '')),
                    'body': str(sample.get('body', '')),
                    'label': 1,  # All augmented samples are malicious
                    'source': f'augmented_group_{group_id}',
                    'augmented': True,
                    'group_id': group_id,
                    'sample_id': i
                }
                valid_samples.append(clean_sample)
        
        return valid_samples
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse augmentation response for group {group_id}: {str(e)}")
        return []


def save_augmented_dataset(augmented_samples: List[Dict], original_df: pd.DataFrame, 
                          n_groups: int, n_augment: int) -> None:
    """Save the augmented dataset in both CSV and JSON formats."""
    logger.info(f"Saving augmented dataset with {len(augmented_samples)} samples")
    
    # Create output directory
    output_dir = config_manager.data_dir / "seeds-augment"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create augmented DataFrame
    augmented_df = pd.DataFrame(augmented_samples)
    
    # Combine with original benign data (label = 0)
    benign_df = original_df[original_df['label'] == 0].copy()
    logger.info(f"Adding {len(benign_df)} benign samples to augmented dataset")
    
    # Add augmented flag to benign samples
    benign_df['augmented'] = False
    benign_df['group_id'] = -1
    benign_df['sample_id'] = -1
    
    # Combine datasets
    final_df = pd.concat([augmented_df, benign_df], ignore_index=True)
    
    # Shuffle the final dataset
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Final dataset: {len(final_df)} total samples")
    
    # Generate output filenames
    base_name = f"email_phishing_CEAS-08_train_augment_malicious_{n_groups}_{n_augment}"
    csv_path = output_dir / f"{base_name}.csv.gz"
    json_path = output_dir / f"{base_name}.json"
    
    # Save as CSV
    final_df.to_csv(csv_path, compression='gzip', index=False)
    logger.info(f"Saved CSV dataset: {csv_path}")
    
    # Prepare metadata for JSON format
    metadata = {
        'total_samples': len(final_df),
        'augmented_samples': len(augmented_samples),
        'original_benign_samples': len(benign_df),
        'n_groups': n_groups,
        'n_augment': n_augment,
        'generation_method': 'llm_augmentation_by_groups',
        'model_used': MODEL_NAME,
        'augmentation_timestamp': pd.Timestamp.now().isoformat(),
        'label_distribution': {
            'malicious': int((final_df['label'] == 1).sum()),
            'benign': int((final_df['label'] == 0).sum())
        }
    }
    
    # Save as JSON with metadata
    json_data = {
        'samples': final_df.to_dict('records'),
        'metadata': metadata
    }
    
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, default=str)
    
    logger.info(f"Saved JSON dataset: {json_path}")
    
    # Log summary
    logger.info("="*50)
    logger.info("AUGMENTATION SUMMARY")
    logger.info("="*50)
    logger.info(f"Original malicious samples: {len(original_df[original_df['label'] == 1])}")
    logger.info(f"Groups created: {n_groups}")
    logger.info(f"Augmentation per group: {n_augment}")
    logger.info(f"Total augmented samples: {len(augmented_samples)}")
    logger.info(f"Original benign samples: {len(benign_df)}")
    logger.info(f"Final dataset size: {len(final_df)}")
    logger.info(f"Augmentation ratio: {len(augmented_samples)}/{len(original_df[original_df['label'] == 1]):.1f}x")
    logger.info("="*50)


def main():
    """Main function for malicious data augmentation."""
    logger.info("="*80)
    logger.info("MALICIOUS DATA AUGMENTATION WITH LLM")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  - Data file: {DATA_PATH}")
    logger.info(f"  - Number of groups: {N_GROUPS}")
    logger.info(f"  - Augmentations per group: {N_AUGMENT}")
    logger.info(f"  - Expected augmented samples: {N_GROUPS * N_AUGMENT}")
    
    try:
        # Load raw data
        raw_dir = config_manager.project_root / "raw"
        file_path = raw_dir / DATA_PATH
        df = load_raw_data(file_path)
        
        # Sample malicious data
        malicious_df = sample_malicious_data(df)
        
        # Create groups
        groups = create_groups(malicious_df, N_GROUPS)
        
        # Generate augmented samples for each group
        all_augmented_samples = []
        
        for i, group in enumerate(groups):
            logger.info(f"Processing group {i+1}/{len(groups)}")
            
            # Sample one example from the group
            example = sample_example_from_group(group)
            logger.debug(f"Sampled example from group {i+1}: {example['subject'][:50]}...")
            
            # Generate augmented samples
            augmented_samples = generate_augmented_samples(example, N_AUGMENT, i+1)
            
            if augmented_samples:
                all_augmented_samples.extend(augmented_samples)
                logger.info(f"Group {i+1}: Generated {len(augmented_samples)} samples")
            else:
                logger.warning(f"Group {i+1}: Failed to generate samples")
        
        logger.info(f"Total augmented samples generated: {len(all_augmented_samples)}")
        
        # Save the augmented dataset
        save_augmented_dataset(all_augmented_samples, df, N_GROUPS, N_AUGMENT)
        
        logger.info("Augmentation process completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in augmentation process: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()