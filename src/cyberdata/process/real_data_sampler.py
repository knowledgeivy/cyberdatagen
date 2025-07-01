# real_data_sampler.py

import gzip
import pandas as pd
import sys
from pathlib import Path
from typing import Tuple

# Add parent directory to path for imports
CURRENT_DIR = Path(__file__).parent
sys.path.append(str(CURRENT_DIR.parent))

from cyberdata.utils.config_manager import get_config_manager
from cyberdata.utils.logger_config import setup_logger

# Set up logger
logger = setup_logger("cyberdata.scripts.real_data_sampler")

# Configuration
N_SAMPLE = 1000

# Get config manager instance
config_manager = get_config_manager()

# File paths
TRAIN_FILE = config_manager.project_root / "raw" / "email_phishing_CEAS-08_train.csv.gz"
RAW_REWRITE_DIR = config_manager.project_root / "raw" / "rewrite"
MALICIOUS_SAMPLE_FILE = RAW_REWRITE_DIR / "malicious_sample.csv.gz"
BENIGN_SAMPLE_FILE = RAW_REWRITE_DIR / "benign_sample.csv.gz"


def create_output_directories():
    """Create output directories if they don't exist."""
    RAW_REWRITE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {RAW_REWRITE_DIR}")


def load_and_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and sample malicious and benign data from training file."""
    logger.info(f"Loading data from: {TRAIN_FILE}")
    
    try:
        with gzip.open(TRAIN_FILE, 'rt', encoding='utf-8') as f:
            df = pd.read_csv(f)
        
        logger.info(f"Loaded {len(df)} total rows")
        
        # Filter malicious data (label = 1)
        malicious_df = df[df['label'] == 1].copy()
        logger.info(f"Found {len(malicious_df)} malicious rows")
        
        # Filter benign data (label = 0)
        benign_df = df[df['label'] == 0].copy()
        logger.info(f"Found {len(benign_df)} benign rows")
        
        # Sample N_SAMPLE rows for each category
        if len(malicious_df) > N_SAMPLE:
            malicious_sampled = malicious_df.sample(n=N_SAMPLE, random_state=42)
            logger.info(f"Sampled {N_SAMPLE} malicious rows")
        else:
            malicious_sampled = malicious_df
            logger.info(f"Using all {len(malicious_sampled)} malicious rows")
        
        if len(benign_df) > N_SAMPLE:
            benign_sampled = benign_df.sample(n=N_SAMPLE, random_state=42)
            logger.info(f"Sampled {N_SAMPLE} benign rows")
        else:
            benign_sampled = benign_df
            logger.info(f"Using all {len(benign_sampled)} benign rows")
        
        return malicious_sampled.reset_index(drop=True), benign_sampled.reset_index(drop=True)
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def save_sampled_data(malicious_df: pd.DataFrame, benign_df: pd.DataFrame):
    """Save sampled data to raw/rewrite/ directory."""
    logger.info("Saving sampled data...")
    
    try:
        # Save malicious data
        with gzip.open(MALICIOUS_SAMPLE_FILE, 'wt', encoding='utf-8') as f:
            malicious_df.to_csv(f, index=False)
        logger.info(f"Saved {len(malicious_df)} malicious samples to: {MALICIOUS_SAMPLE_FILE}")
        
        # Save benign data
        with gzip.open(BENIGN_SAMPLE_FILE, 'wt', encoding='utf-8') as f:
            benign_df.to_csv(f, index=False)
        logger.info(f"Saved {len(benign_df)} benign samples to: {BENIGN_SAMPLE_FILE}")
        
    except Exception as e:
        logger.error(f"Error saving sampled data: {str(e)}")
        raise


def main():
    """Main function to sample and save data."""
    logger.info("="*80)
    logger.info("REAL WORLD DATA SAMPLING")
    logger.info("="*80)
    
    try:
        # Create output directories
        create_output_directories()
        
        # Load and sample data
        malicious_df, benign_df = load_and_sample_data()
        
        # Save sampled data
        save_sampled_data(malicious_df, benign_df)
        
        logger.info("="*80)
        logger.info("DATA SAMPLING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Malicious samples: {len(malicious_df)} saved to {MALICIOUS_SAMPLE_FILE}")
        logger.info(f"Benign samples: {len(benign_df)} saved to {BENIGN_SAMPLE_FILE}")
        logger.info("")
        logger.info("Next step: Run real_data_rewriter.py with your chosen prompt")
        
    except Exception as e:
        logger.error(f"Error in data sampling: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()