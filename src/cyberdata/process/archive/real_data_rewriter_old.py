# real_data_rewrite.py

import concurrent.futures
import gzip
import json
import pandas as pd
import sys
import time
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
logger = setup_logger("cyberdata.scripts.real_data_rewrite")

# Load environment variables
load_dotenv()

# Configuration
MODEL_NAME = "gpt-4.1-mini"
N_SAMPLE = 1000
MAX_WORKERS = 20
BATCH_SIZE = 1

# Get config manager instance
config_manager = get_config_manager()

# File paths
TRAIN_FILE = config_manager.project_root / "raw" / "email_phishing_CEAS-08_train.csv.gz"
SAMPLED_DATASET_FILE = config_manager.project_root / "data" / "seeds-augment" / "email_phishing_CEAS-08_sampled_1K+1K.csv.gz"
SYNTHETIC_FILE = config_manager.project_root / "data" / "seeds-augment" / "email_phishing_CEAS-08_train_malicious_rewrite_1K.csv.gz"
SYNTHETIC_STRONG_FILE = config_manager.project_root / "data" / "seeds-augment" / "email_phishing_CEAS-08_train_malicious_rewrite_strong_1K.csv.gz"

logger.info(f"Using model: {MODEL_NAME}")
logger.info(f"Sample size: {N_SAMPLE} malicious + {N_SAMPLE} benign")
logger.info(f"Max workers: {MAX_WORKERS}")


def create_consistent_sample():
    """Create consistent 1K malicious + 1K benign sample and save it."""
    logger.info("="*60)
    logger.info("CREATING CONSISTENT SAMPLE DATASET")
    logger.info("="*60)
    
    logger.info(f"Loading data from: {TRAIN_FILE}")
    
    try:
        with gzip.open(TRAIN_FILE, 'rt', encoding='utf-8') as f:
            df = pd.read_csv(f)
        
        logger.info(f"Loaded {len(df)} total rows")
        
        # Filter malicious and benign data
        malicious_df = df[df['label'] == 1].copy()
        benign_df = df[df['label'] == 0].copy()
        
        logger.info(f"Found {len(malicious_df)} malicious rows")
        logger.info(f"Found {len(benign_df)} benign rows")
        
        # Sample N_SAMPLE from each class
        sampled_malicious = malicious_df.sample(n=min(N_SAMPLE, len(malicious_df)), random_state=42)
        sampled_benign = benign_df.sample(n=min(N_SAMPLE, len(benign_df)), random_state=42)
        
        # Combine samples
        combined_sample = pd.concat([sampled_malicious, sampled_benign], ignore_index=True)
        combined_sample = combined_sample.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
        
        # Create output directory
        SAMPLED_DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Save sampled dataset
        with gzip.open(SAMPLED_DATASET_FILE, 'wt', encoding='utf-8') as f:
            combined_sample.to_csv(f, index=False)
        
        logger.info(f"Saved consistent sample: {len(combined_sample)} rows ({len(sampled_malicious)} malicious, {len(sampled_benign)} benign)")
        logger.info(f"File: {SAMPLED_DATASET_FILE}")
        
        return sampled_malicious, sampled_benign
        
    except Exception as e:
        logger.error(f"Error creating sample: {str(e)}")
        raise


def load_consistent_sample():
    """Load the consistent sample dataset."""
    logger.info(f"Loading consistent sample from: {SAMPLED_DATASET_FILE}")
    
    try:
        with gzip.open(SAMPLED_DATASET_FILE, 'rt', encoding='utf-8') as f:
            df = pd.read_csv(f)
        
        malicious_df = df[df['label'] == 1].copy()
        benign_df = df[df['label'] == 0].copy()
        
        logger.info(f"Loaded consistent sample: {len(df)} rows ({len(malicious_df)} malicious, {len(benign_df)} benign)")
        
        return malicious_df, benign_df
        
    except FileNotFoundError:
        logger.info("Consistent sample not found, creating new one...")
        return create_consistent_sample()
    except Exception as e:
        logger.error(f"Error loading sample: {str(e)}")
        raise


def rewrite_single_record(record: Dict, index: int, prompt_file: str) -> Dict:
    """Rewrite a single email record using specified prompt."""
    try:
        # Load prompts from specified file
        system_prompt = load_prompt(prompt_file, "prompts.rewrite_phishing.system.template")
        user_prompt = load_prompt(prompt_file, "prompts.rewrite_phishing.user.template",
                                 original_subject=record['subject'],
                                 original_body=record['body'])
        
        # Call LLM
        response = process_llm_request(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=MODEL_NAME,
            temperature=0.8
        )
        
        # Parse response
        try:
            response_data = json.loads(response)
            rewritten_record = {
                'subject': response_data.get('rewritten_subject', record['subject']),
                'body': response_data.get('rewritten_body', record['body']),
                'label': record['label'],  # Keep original label
                'source': record.get('source', 'unknown')
            }
            
            logger.debug(f"Successfully rewrote record {index} with {prompt_file}")
            return rewritten_record
            
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON response for record {index}, using original")
            return record
            
    except Exception as e:
        logger.error(f"Error rewriting record {index}: {str(e)}")
        return record


def rewrite_data_parallel(df: pd.DataFrame, prompt_file: str, output_file: Path) -> pd.DataFrame:
    """Rewrite data using parallel processing with specified prompt."""
    logger.info(f"Starting parallel rewriting with {prompt_file}")
    logger.info(f"Using {MAX_WORKERS} workers, output: {output_file}")
    
    # Convert to records
    records = df.to_dict('records')
    rewritten_records = []
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(rewrite_single_record, record, i, prompt_file): i 
            for i, record in enumerate(records)
        }
        
        # Collect results
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                rewritten_records.append(result)
                
                # Log progress
                if (len(rewritten_records)) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = len(rewritten_records) / elapsed * 60
                    logger.info(f"Progress: {len(rewritten_records)}/{len(records)} ({rate:.1f} records/min)")
                    
            except Exception as e:
                logger.error(f"Task {index} failed: {e}")
                # Use original record as fallback
                rewritten_records.append(records[index])
    
    # Convert back to DataFrame
    result_df = pd.DataFrame(rewritten_records)
    
    total_time = time.time() - start_time
    logger.info(f"Parallel rewriting completed in {total_time:.2f} seconds")
    logger.info(f"Average rate: {len(records) / total_time * 60:.1f} records/minute")
    
    # Save synthetic data
    logger.info(f"Saving to: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with gzip.open(output_file, 'wt', encoding='utf-8') as f:
        result_df.to_csv(f, index=False)
    
    logger.info(f"Saved {len(result_df)} rewritten records")
    
    return result_df


def main():
    """Main function for consistent data rewriting with two prompts."""
    logger.info("="*80)
    logger.info("CONSISTENT MALICIOUS EMAIL DATA REWRITING WITH TWO PROMPTS")
    logger.info("="*80)
    
    try:
        # Step 1: Load or create consistent sample
        malicious_df, benign_df = load_consistent_sample()
        
        # Step 2: Rewrite with regular prompt
        logger.info("\n" + "="*60)
        logger.info("REWRITING WITH REGULAR PROMPT")
        logger.info("="*60)
        
        rewritten_regular = rewrite_data_parallel(
            malicious_df, 
            "rewrite_generation", 
            SYNTHETIC_FILE
        )
        
        # Step 3: Rewrite with strong prompt
        logger.info("\n" + "="*60)
        logger.info("REWRITING WITH STRONG PROMPT")
        logger.info("="*60)
        
        rewritten_strong = rewrite_data_parallel(
            malicious_df, 
            "rewrite_generation_strong", 
            SYNTHETIC_STRONG_FILE
        )
        
        logger.info("="*80)
        logger.info("CONSISTENT DATA REWRITING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Source malicious records: {len(malicious_df)}")
        logger.info(f"Source benign records: {len(benign_df)}")
        logger.info(f"Regular synthetic records: {len(rewritten_regular)}")
        logger.info(f"Strong synthetic records: {len(rewritten_strong)}")
        logger.info("")
        logger.info("Files created:")
        logger.info(f"  - Consistent sample: {SAMPLED_DATASET_FILE}")
        logger.info(f"  - Regular synthetic: {SYNTHETIC_FILE}")
        logger.info(f"  - Strong synthetic: {SYNTHETIC_STRONG_FILE}")
        logger.info("")
        logger.info("All datasets are now comparable using the same source data!")
        
    except Exception as e:
        logger.error(f"Error in data rewriting: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()