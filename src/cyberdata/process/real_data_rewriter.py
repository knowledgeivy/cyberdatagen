# real_data_rewriter.py
# Test individual prompts
# python real_data_rewriter.py --prompt original
# python real_data_rewriter.py --prompt strong  
# python real_data_rewriter.py --prompt weak

# Run ALL prompts at once (convenient for comparison)
# python real_data_rewriter.py --prompt all

import argparse
import concurrent.futures
import gzip
import json
import pandas as pd
import sys
import time
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

# Add parent directory to path for imports
CURRENT_DIR = Path(__file__).parent
sys.path.append(str(CURRENT_DIR.parent))

from cyberdata.utils.config_manager import get_config_manager
from cyberdata.utils.llm_invoke import process_llm_request
from cyberdata.utils.logger_config import setup_logger
from cyberdata.utils.prompt_loader import load_prompt

# Set up logger
logger = setup_logger("cyberdata.scripts.real_data_rewriter")

# Load environment variables
load_dotenv()

# Configuration
MODEL_NAME = "gpt-4.1-mini"
DEFAULT_MAX_WORKERS = 20
BATCH_SIZE = 1

# Get config manager instance
config_manager = get_config_manager()

# File paths
RAW_REWRITE_DIR = config_manager.project_root / "raw" / "rewrite"
DATA_REWRITE_DIR = config_manager.project_root / "data" / "rewrite"
MALICIOUS_SAMPLE_FILE = RAW_REWRITE_DIR / "malicious_sample.csv.gz"

# Available prompts mapping
AVAILABLE_PROMPTS = {
    "original": "rewrite_generation",
    "strong": "rewrite_generation_strong", 
    "weak": "rewrite_generation_weak",
    # Add more prompts as needed
}


def create_output_directories():
    """Create output directories if they don't exist."""
    DATA_REWRITE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {DATA_REWRITE_DIR}")


def load_malicious_data() -> pd.DataFrame:
    """Load malicious data from raw/rewrite/ directory."""
    logger.info(f"Loading malicious data from: {MALICIOUS_SAMPLE_FILE}")
    
    try:
        with gzip.open(MALICIOUS_SAMPLE_FILE, 'rt', encoding='utf-8') as f:
            df = pd.read_csv(f)
        
        logger.info(f"Loaded {len(df)} malicious rows")
        return df
        
    except Exception as e:
        logger.error(f"Error loading malicious data: {str(e)}")
        logger.error("Make sure to run real_data_sampler.py first to create the sample data")
        raise


def rewrite_single_record(record: Dict, index: int, prompt_file: str) -> Dict:
    """Rewrite a single malicious email record using LLM."""
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
                'label': record['label'],  # Keep original label (should be 1 for malicious)
                'source': record.get('source', 'unknown'),
                'prompt_used': prompt_file,
                'original_index': index
            }
            
            logger.debug(f"Successfully rewrote malicious record {index} with {prompt_file}")
            return rewritten_record
            
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON response for record {index}, using original")
            record['prompt_used'] = prompt_file
            record['original_index'] = index
            return record
            
    except Exception as e:
        logger.error(f"Error rewriting record {index}: {str(e)}")
        record['prompt_used'] = prompt_file
        record['original_index'] = index
        return record


def rewrite_malicious_data_parallel(df: pd.DataFrame, prompt_file: str, max_workers: int) -> pd.DataFrame:
    """Rewrite malicious data using parallel processing with specified prompt."""
    logger.info(f"Starting parallel rewriting of malicious data with {prompt_file} prompt...")
    logger.info(f"Processing {len(df)} malicious records with {max_workers} workers")
    
    start_time = time.time()
    rewritten_records = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create futures for all records
        futures = []
        for index, record in df.iterrows():
            future = executor.submit(rewrite_single_record, record.to_dict(), index, prompt_file)
            futures.append(future)
        
        # Collect results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                result = future.result()
                rewritten_records.append(result)
                
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"Processed {i + 1}/{len(futures)} records (Elapsed: {elapsed:.1f}s)")
                    
            except Exception as e:
                logger.error(f"Failed to process record: {str(e)}")
    
    # Convert to DataFrame
    rewritten_df = pd.DataFrame(rewritten_records)
    
    elapsed = time.time() - start_time
    logger.info(f"Parallel rewriting completed in {elapsed:.1f} seconds")
    logger.info(f"Successfully rewritten {len(rewritten_df)} malicious records")
    
    return rewritten_df


def save_rewritten_data(df: pd.DataFrame, prompt_name: str) -> Path:
    """Save rewritten malicious data to data/rewrite/ directory."""
    output_file = DATA_REWRITE_DIR / f"malicious_{prompt_name}_rewritten.csv.gz"
    
    logger.info(f"Saving rewritten data to: {output_file}")
    
    try:
        with gzip.open(output_file, 'wt', encoding='utf-8') as f:
            df.to_csv(f, index=False)
        
        logger.info(f"Saved {len(df)} rewritten malicious records to: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Error saving rewritten data: {str(e)}")
        raise


def main():
    """Main function to run malicious data rewriting."""
    parser = argparse.ArgumentParser(description="Rewrite malicious data using LLM with specified prompt")
    parser.add_argument("--prompt", "-p", 
                       choices=list(AVAILABLE_PROMPTS.keys()) + ["all"],
                       required=True,
                       help="Prompt to use for rewriting malicious data, or 'all' to run all prompts")
    parser.add_argument("--max-workers", "-w",
                       type=int,
                       default=DEFAULT_MAX_WORKERS,
                       help="Maximum number of worker threads")
    
    args = parser.parse_args()
    
    max_workers = args.max_workers
    
    logger.info("="*80)
    if args.prompt == "all":
        logger.info("MALICIOUS DATA REWRITING WITH ALL PROMPTS")
        prompts_to_run = list(AVAILABLE_PROMPTS.keys())
    else:
        logger.info(f"MALICIOUS DATA REWRITING WITH {args.prompt.upper()} PROMPT")
        prompts_to_run = [args.prompt]
    
    logger.info("="*80)
    logger.info(f"Prompts to run: {', '.join(prompts_to_run)}")
    logger.info(f"Max workers: {max_workers}")
    logger.info(f"Model: {MODEL_NAME}")
    
    try:
        # Create output directories
        create_output_directories()
        
        # Load malicious data once
        malicious_df = load_malicious_data()
        
        results = []
        
        # Process each prompt
        for i, prompt_name in enumerate(prompts_to_run, 1):
            prompt_file = AVAILABLE_PROMPTS[prompt_name]
            
            logger.info(f"\n{'='*60}")
            logger.info(f"PROCESSING PROMPT {i}/{len(prompts_to_run)}: {prompt_name.upper()}")
            logger.info(f"{'='*60}")
            
            # Rewrite malicious data
            rewritten_malicious = rewrite_malicious_data_parallel(malicious_df, prompt_file, max_workers)
            
            # Save rewritten data
            output_file = save_rewritten_data(rewritten_malicious, prompt_name)
            
            results.append({
                'prompt': prompt_name,
                'prompt_file': prompt_file,
                'input_count': len(malicious_df),
                'output_count': len(rewritten_malicious),
                'output_file': output_file
            })
        
        # Final summary
        logger.info("="*80)
        logger.info("MALICIOUS DATA REWRITING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
        for result in results:
            logger.info(f"Prompt: {result['prompt']} ({result['prompt_file']})")
            logger.info(f"  Input: {result['input_count']} original malicious records")
            logger.info(f"  Output: {result['output_count']} rewritten malicious records")
            logger.info(f"  Saved to: {result['output_file']}")
            logger.info("")
        
        if len(results) > 1:
            logger.info("All prompts completed! You can now compare the different rewriting approaches.")
        
    except Exception as e:
        logger.error(f"Error in malicious data rewriting: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()