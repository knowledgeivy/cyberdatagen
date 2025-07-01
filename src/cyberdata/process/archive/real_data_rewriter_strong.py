# real_data_rewrite_strong.py

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
SYNTHETIC_FILE = config_manager.project_root / "data" / "seeds-augment" / "email_phishing_CEAS-08_train_malicious_rewrite_strong_1K.csv.gz"

logger.info(f"Using model: {MODEL_NAME}")
logger.info(f"Sample size: {N_SAMPLE}")
logger.info(f"Max workers: {MAX_WORKERS}")


def load_malicious_data() -> pd.DataFrame:
    """Load and sample malicious data from training file."""
    logger.info(f"Loading data from: {TRAIN_FILE}")
    
    try:
        with gzip.open(TRAIN_FILE, 'rt', encoding='utf-8') as f:
            df = pd.read_csv(f)
        
        logger.info(f"Loaded {len(df)} total rows")
        
        # Filter malicious data (label = 1)
        malicious_df = df[df['label'] == 1].copy()
        logger.info(f"Found {len(malicious_df)} malicious rows")
        
        # Sample N_SAMPLE rows
        if len(malicious_df) > N_SAMPLE:
            sampled_df = malicious_df.sample(n=N_SAMPLE, random_state=42)
            logger.info(f"Sampled {N_SAMPLE} malicious rows")
        else:
            sampled_df = malicious_df
            logger.info(f"Using all {len(sampled_df)} malicious rows")
        
        return sampled_df.reset_index(drop=True)
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def rewrite_single_record(record: Dict, index: int) -> Dict:
    """Rewrite a single email record using LLM."""
    try:
        # Load prompts
        system_prompt = load_prompt("rewrite_generation_strong", "prompts.rewrite_phishing.system.template")
        user_prompt = load_prompt("rewrite_generation_strong", "prompts.rewrite_phishing.user.template",
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
            
            logger.debug(f"Successfully rewrote record {index}")
            return rewritten_record
            
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON response for record {index}, using original")
            return record
            
    except Exception as e:
        logger.error(f"Error rewriting record {index}: {str(e)}")
        return record


def rewrite_data_parallel(df: pd.DataFrame) -> pd.DataFrame:
    """Rewrite data using parallel processing."""
    logger.info(f"Starting parallel rewriting with {MAX_WORKERS} workers")
    
    # Convert to records
    records = df.to_dict('records')
    rewritten_records = []
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(rewrite_single_record, record, i): i 
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
    
    return result_df


def save_synthetic_data(df: pd.DataFrame) -> None:
    """Save synthetic data to file."""
    logger.info(f"Saving synthetic data to: {SYNTHETIC_FILE}")
    
    # Create output directory
    SYNTHETIC_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as compressed CSV
    with gzip.open(SYNTHETIC_FILE, 'wt', encoding='utf-8') as f:
        df.to_csv(f, index=False)
    
    logger.info(f"Saved {len(df)} rewritten records")


def main():
    """Main function for data rewriting."""
    logger.info("="*60)
    logger.info("MALICIOUS EMAIL DATA REWRITING")
    logger.info("="*60)
    
    try:
        # Load malicious data
        malicious_df = load_malicious_data()
        
        # Rewrite data
        rewritten_df = rewrite_data_parallel(malicious_df)
        
        # Save synthetic data
        save_synthetic_data(rewritten_df)
        
        logger.info("="*60)
        logger.info("DATA REWRITING COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Input: {len(malicious_df)} malicious records")
        logger.info(f"Output: {len(rewritten_df)} rewritten records")
        logger.info(f"File: {SYNTHETIC_FILE}")
        
    except Exception as e:
        logger.error(f"Error in data rewriting: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()