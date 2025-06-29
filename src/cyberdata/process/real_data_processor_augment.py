# cyberdata/process/real_data_processor_augment.py

import gzip
import json
import pandas as pd
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

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

# Configuration constants
MODEL_NAME = "gpt-4.1-mini"
N_SAMPLE_TIME = 20
SAMPLES_PER_ITERATION = 5
DATA_PATH = "email_phishing_CEAS-08_train.csv.gz"
OUTPUT_DIR_NAME = "seeds-augment"
OUTPUT_FILE_NAME = "email_phishing_CEAS-08_train_augment.csv.gz"

# Get config manager instance
config_manager = get_config_manager()

logger.info(f"Using model: {MODEL_NAME}")
logger.info(f"Project root: {config_manager.project_root}")


class DataAugmenter:
    """Enhanced data augmenter with quality control and metadata tracking."""
    
    def __init__(self, config_override: Dict[str, Any] = None):
        self.config_manager = config_manager
        
        # Configuration
        self.n_sample_time = config_override.get('n_sample_time', N_SAMPLE_TIME) if config_override else N_SAMPLE_TIME
        self.samples_per_iteration = config_override.get('samples_per_iteration', SAMPLES_PER_ITERATION) if config_override else SAMPLES_PER_ITERATION
        self.data_path = config_override.get('data_path', DATA_PATH) if config_override else DATA_PATH
        self.output_dir_name = config_override.get('output_dir_name', OUTPUT_DIR_NAME) if config_override else OUTPUT_DIR_NAME
        self.output_file_name = config_override.get('output_file_name', OUTPUT_FILE_NAME) if config_override else OUTPUT_FILE_NAME
        
        # Statistics tracking
        self.stats = {
            'total_iterations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'total_augmented_samples': 0,
            'duplicate_samples_filtered': 0,
            'schema_violations_filtered': 0,
            'source_samples_used': 0,
            'processing_start_time': None,
            'processing_end_time': None
        }
        
        # Quality tracking
        self.generated_samples = []
        self.seen_content = set()  # For duplicate detection
        self.source_samples_used = []
        
        logger.info(f"DataAugmenter initialized:")
        logger.info(f"  - Sample iterations: {self.n_sample_time}")
        logger.info(f"  - Samples per iteration: {self.samples_per_iteration}")
        logger.info(f"  - Expected total samples: {self.n_sample_time * self.samples_per_iteration}")
    
    def load_source_data(self, file_path: Path, label_column: str = "label") -> pd.DataFrame:
        """Load source data from CSV file with validation."""
        logger.info(f"Loading source data from: {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Source data file not found: {file_path}")
        
        try:
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    df = pd.read_csv(f)
            else:
                df = pd.read_csv(file_path)
            
            logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
            
            # Validate required columns for email phishing data
            required_cols = ['subject', 'body', 'label', 'source']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Handle missing values
            df['subject'] = df['subject'].fillna('')
            df['body'] = df['body'].fillna('')
            
            # Log label distribution
            label_counts = df[label_column].value_counts()
            logger.info(f"Source data label distribution: {dict(label_counts)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading source data: {str(e)}")
            raise
    
    def stratified_sample_source(self, df: pd.DataFrame, label_column: str = "label") -> List[Dict[str, Any]]:
        """Perform stratified sampling to ensure diverse source examples."""
        logger.info("Performing stratified sampling for source examples")
        
        # Get samples by label to ensure representation
        malicious_samples = df[df[label_column] == 1].to_dict('records')
        benign_samples = df[df[label_column] == 0].to_dict('records')
        
        logger.info(f"Available samples - Malicious: {len(malicious_samples)}, Benign: {len(benign_samples)}")
        
        # Create balanced source pool
        source_pool = []
        
        # Add samples from each class
        if malicious_samples:
            # Use more malicious samples if they're rarer
            malicious_needed = max(self.n_sample_time // 2, min(self.n_sample_time, len(malicious_samples)))
            selected_malicious = random.sample(malicious_samples, min(malicious_needed, len(malicious_samples)))
            source_pool.extend(selected_malicious)
            logger.info(f"Selected {len(selected_malicious)} malicious source samples")
        
        if benign_samples:
            benign_needed = self.n_sample_time - len([s for s in source_pool if s.get('label') == 1])
            benign_needed = max(benign_needed, self.n_sample_time // 2)
            selected_benign = random.sample(benign_samples, min(benign_needed, len(benign_samples)))
            source_pool.extend(selected_benign)
            logger.info(f"Selected {len(selected_benign)} benign source samples")
        
        # Shuffle and ensure we have enough samples
        random.shuffle(source_pool)
        
        # If we need more samples, add random ones
        while len(source_pool) < self.n_sample_time:
            additional_sample = random.choice(malicious_samples + benign_samples)
            source_pool.append(additional_sample)
        
        # Truncate to exact needed amount
        source_pool = source_pool[:self.n_sample_time]
        
        logger.info(f"Final source pool: {len(source_pool)} samples")
        final_distribution = pd.Series([s.get('label') for s in source_pool]).value_counts()
        logger.info(f"Source pool label distribution: {dict(final_distribution)}")
        
        return source_pool
    
    def create_content_hash(self, sample: Dict[str, Any]) -> str:
        """Create a hash of sample content for duplicate detection."""
        # Combine key content fields
        content_parts = [
            str(sample.get('subject', '')).lower().strip(),
            str(sample.get('body', ''))[:200].lower().strip()  # First 200 chars for efficiency
        ]
        content_str = '|'.join(content_parts)
        return str(hash(content_str))
    
    def is_duplicate(self, sample: Dict[str, Any]) -> bool:
        """Check if sample is too similar to existing samples."""
        content_hash = self.create_content_hash(sample)
        if content_hash in self.seen_content:
            return True
        self.seen_content.add(content_hash)
        return False
    
    def validate_schema(self, sample: Dict[str, Any], expected_columns: List[str]) -> bool:
        """Validate that sample maintains required schema."""
        # Check required columns exist
        for col in expected_columns:
            if col not in sample:
                logger.warning(f"Schema violation: missing column '{col}'")
                return False
        
        # Check data types and basic validation
        try:
            # Subject and body should be strings
            if not isinstance(sample.get('subject', ''), str):
                return False
            if not isinstance(sample.get('body', ''), str):
                return False
            
            # Label should be 0 or 1
            label = sample.get('label')
            if label not in [0, 1]:
                return False
            
            # Source should be string
            if not isinstance(sample.get('source', ''), str):
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Schema validation error: {e}")
            return False
    
    def augment_single_sample(self, source_sample: Dict[str, Any], iteration: int) -> List[Dict[str, Any]]:
        """Augment a single source sample into multiple variations."""
        logger.info(f"Augmenting sample from iteration {iteration + 1}/{self.n_sample_time}")
        
        try:
            # Determine sample type and variation strategies
            label = source_sample.get('label', 0)
            label_type = "malicious_phishing" if label == 1 else "benign_legitimate"
            label_description = "MALICIOUS/PHISHING" if label == 1 else "BENIGN/LEGITIMATE"
            
            # Load appropriate variation strategies
            try:
                variation_strategies = load_prompt(
                    "augment_generation",
                    f"prompts.variation_strategies.{label_type}.template"
                )
            except Exception:
                # Fallback variation strategies
                if label == 1:
                    variation_strategies = "Focus on varying phishing techniques while preserving malicious intent and attack patterns."
                else:
                    variation_strategies = "Focus on varying legitimate business communication while preserving professional characteristics."
            
            # Prepare context for prompt
            augmentation_context = {
                'source_sample': source_sample,
                'target_variations': self.samples_per_iteration,
                'iteration_number': iteration + 1,
                'total_iterations': self.n_sample_time,
                'schema_requirements': {
                    'columns': ['subject', 'body', 'label', 'source'],
                    'label_encoding': {'malicious': 1, 'benign': 0}
                }
            }
            
            context_json = json.dumps(augmentation_context, indent=2, default=str)
            
            # Load augmentation prompts
            try:
                system_prompt = load_prompt(
                    "augment_generation",
                    "prompts.email_augmentation.system.template"
                )
                
                user_prompt = load_prompt(
                    "augment_generation",
                    "prompts.email_augmentation.user.template",
                    augmentation_context_json=context_json,
                    source_sample=source_sample,
                    target_variations=self.samples_per_iteration,
                    label_type=label_description,
                    variation_strategies=variation_strategies,
                    # Individual source sample fields for template access
                    source_subject=source_sample.get('subject', ''),
                    source_body=source_sample.get('body', ''),
                    source_label=source_sample.get('label', 0),
                    source_source=source_sample.get('source', '')
                )
            except Exception as e:
                logger.error(f"Error loading prompts: {e}")
                # Fallback to basic prompts
                system_prompt = self._get_fallback_system_prompt()
                user_prompt = self._get_fallback_user_prompt(source_sample)
            
            # Call LLM for augmentation
            response_content = process_llm_request(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name=MODEL_NAME,
                temperature=0.8  # Higher temperature for diversity
            )
            
            # Parse response
            augmented_samples = self._parse_augmentation_response(response_content, source_sample)
            
            # Validate and filter samples
            valid_samples = []
            expected_columns = ['subject', 'body', 'label', 'source']
            
            for sample in augmented_samples:
                # Add metadata
                sample['_augmentation_metadata'] = {
                    'source_iteration': iteration + 1,
                    'augmentation_timestamp': time.time(),
                    'source_sample_hash': self.create_content_hash(source_sample)
                }
                
                # Validate schema
                if not self.validate_schema(sample, expected_columns):
                    self.stats['schema_violations_filtered'] += 1
                    logger.warning(f"Filtered sample due to schema violation")
                    continue
                
                # Check for duplicates
                if self.is_duplicate(sample):
                    self.stats['duplicate_samples_filtered'] += 1
                    logger.debug(f"Filtered duplicate sample")
                    continue
                
                valid_samples.append(sample)
            
            logger.info(f"Iteration {iteration + 1}: Generated {len(valid_samples)}/{self.samples_per_iteration} valid samples")
            return valid_samples
            
        except Exception as e:
            logger.error(f"Error augmenting sample in iteration {iteration + 1}: {e}")
            self.stats['failed_generations'] += 1
            return []
    
    def _get_fallback_system_prompt(self) -> str:
        """Fallback system prompt if loading from YAML fails."""
        return """You are an expert cybersecurity data augmentation specialist. Your task is to generate realistic variations of email phishing dataset samples while preserving their essential characteristics and labels.

Generate diverse but realistic variations that:
1. Maintain the same classification label (malicious or benign)
2. Preserve the core semantic meaning and attack patterns
3. Use different wording, phrasing, and structural approaches
4. Remain realistic for cybersecurity training datasets
5. Follow the exact same schema structure

Focus on creating high-quality, diverse examples that would be valuable for training robust cybersecurity models."""
    
    def _get_fallback_user_prompt(self, source_sample: Dict[str, Any]) -> str:
        """Fallback user prompt if loading from YAML fails."""
        return f"""Generate {self.samples_per_iteration} realistic variations of this email sample:

Source Sample:
- Subject: {source_sample.get('subject', '')}
- Body: {source_sample.get('body', '')}
- Label: {source_sample.get('label', 0)}
- Source: {source_sample.get('source', '')}

Return ONLY valid JSON in this exact format:
{{
  "augmented_samples": [
    {{
      "subject": "variation 1 subject",
      "body": "variation 1 body",
      "label": {source_sample.get('label', 0)},
      "source": "augmented_{source_sample.get('source', '')}"
    }}
    // ... {self.samples_per_iteration} total variations
  ]
}}

Requirements:
- All variations must have the same label as the source
- Create diverse but realistic variations
- Maintain email format and cybersecurity domain relevance
- Each variation should be unique and distinct"""
    
    def _parse_augmentation_response(self, response_content: str, source_sample: Dict[str, Any]) -> List[Dict[str, Any]]:
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
            augmented_samples = parsed.get('augmented_samples', [])
            
            if not isinstance(augmented_samples, list):
                logger.warning("Invalid augmented_samples format in response")
                return []
            
            return augmented_samples
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse augmentation response: {e}")
            # Try to extract samples manually as fallback
            return self._extract_samples_fallback(response_content, source_sample)
    
    def _extract_samples_fallback(self, response_content: str, source_sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fallback method to extract samples if JSON parsing fails."""
        logger.warning("Using fallback sample extraction")
        # This is a very basic fallback - in practice you might want more sophisticated parsing
        return []
    
    def process_augmentation(self, source_pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process the complete augmentation pipeline."""
        logger.info("Starting augmentation process")
        logger.info("="*60)
        
        self.stats['processing_start_time'] = time.time()
        all_augmented_samples = []
        
        for iteration in range(self.n_sample_time):
            logger.info(f"\n{'='*40}")
            logger.info(f"ITERATION {iteration + 1}/{self.n_sample_time}")
            logger.info(f"{'='*40}")
            
            # Get source sample for this iteration
            source_sample = source_pool[iteration]
            self.source_samples_used.append(source_sample)
            
            logger.info(f"Source sample: Label={source_sample.get('label')}, "
                       f"Subject='{source_sample.get('subject', '')[:50]}...'")
            
            # Augment the source sample
            iteration_samples = self.augment_single_sample(source_sample, iteration)
            
            if iteration_samples:
                all_augmented_samples.extend(iteration_samples)
                self.stats['successful_generations'] += 1
                self.stats['total_augmented_samples'] += len(iteration_samples)
                logger.info(f"✓ Added {len(iteration_samples)} samples to dataset")
            else:
                self.stats['failed_generations'] += 1
                logger.warning(f"✗ No valid samples generated in iteration {iteration + 1}")
            
            # Progress update
            current_total = len(all_augmented_samples)
            expected_total = self.n_sample_time * self.samples_per_iteration
            progress = (iteration + 1) / self.n_sample_time * 100
            
            logger.info(f"Progress: {progress:.1f}% | Total samples: {current_total} | "
                       f"Expected: {expected_total}")
            
            # Small delay to avoid overwhelming API
            time.sleep(0.5)
        
        self.stats['processing_end_time'] = time.time()
        self.stats['total_iterations'] = self.n_sample_time
        self.stats['source_samples_used'] = len(self.source_samples_used)
        
        logger.info("\n" + "="*60)
        logger.info("AUGMENTATION COMPLETED")
        logger.info("="*60)
        
        return all_augmented_samples
    
    def save_augmented_dataset(self, augmented_samples: List[Dict[str, Any]]) -> Path:
        """Save the augmented dataset with comprehensive metadata."""
        logger.info("Saving augmented dataset...")
        
        # Create output directory
        output_dir = self.config_manager.data_dir / self.output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame
        df = pd.DataFrame(augmented_samples)
        
        # Count by label for reporting
        if len(df) > 0:
            label_counts = df['label'].value_counts()
            malicious_count = label_counts.get(1, 0)
            benign_count = label_counts.get(0, 0)
        else:
            malicious_count = benign_count = 0
        
        # Create comprehensive metadata
        metadata = {
            'generation_method': 'llm_augmentation',
            'model_used': MODEL_NAME,
            'configuration': {
                'n_sample_time': self.n_sample_time,
                'samples_per_iteration': self.samples_per_iteration,
                'target_total_samples': self.n_sample_time * self.samples_per_iteration,
                'actual_total_samples': len(augmented_samples)
            },
            'source_data': {
                'source_file': self.data_path,
                'source_samples_used': self.stats['source_samples_used']
            },
            'generation_stats': self.stats.copy(),
            'quality_metrics': {
                'total_samples_generated': len(augmented_samples),
                'malicious_samples': int(malicious_count),
                'benign_samples': int(benign_count),
                'malicious_ratio': malicious_count / len(augmented_samples) if augmented_samples else 0,
                'duplicate_filter_rate': self.stats['duplicate_samples_filtered'] / max(1, self.stats['total_augmented_samples']),
                'schema_violation_rate': self.stats['schema_violations_filtered'] / max(1, self.stats['total_augmented_samples']),
                'success_rate': self.stats['successful_generations'] / max(1, self.stats['total_iterations'])
            },
            'processing_time': {
                'total_time_seconds': self.stats['processing_end_time'] - self.stats['processing_start_time'],
                'average_time_per_iteration': (self.stats['processing_end_time'] - self.stats['processing_start_time']) / self.stats['total_iterations']
            },
            'dataset_timestamp': time.time(),
            'description': 'LLM-augmented email phishing dataset with quality filtering and diversity controls'
        }
        
        # Save as compressed CSV with metadata
        output_path = output_dir / self.output_file_name
        
        if output_path.suffix == '.gz':
            with gzip.open(output_path, 'wt', encoding='utf-8') as f:
                df.to_csv(f, index=False)
        else:
            df.to_csv(output_path, index=False)
        
        # Save metadata separately
        metadata_path = output_dir / f"{Path(self.output_file_name).stem}_metadata.json"
        with metadata_path.open('w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Augmented dataset saved to: {output_path}")
        logger.info(f"Metadata saved to: {metadata_path}")
        
        return output_path
    
    def generate_summary_report(self, augmented_samples: List[Dict[str, Any]]) -> None:
        """Generate and log a comprehensive summary report."""
        total_time = self.stats['processing_end_time'] - self.stats['processing_start_time']
        
        logger.info("\n" + "="*80)
        logger.info("AUGMENTATION SUMMARY REPORT")
        logger.info("="*80)
        
        logger.info(f"📊 Generation Configuration:")
        logger.info(f"   Source file: {self.data_path}")
        logger.info(f"   Iterations: {self.n_sample_time}")
        logger.info(f"   Samples per iteration: {self.samples_per_iteration}")
        logger.info(f"   Target total: {self.n_sample_time * self.samples_per_iteration}")
        logger.info(f"   Model used: {MODEL_NAME}")
        
        logger.info(f"\n📈 Generation Results:")
        logger.info(f"   Actual samples generated: {len(augmented_samples)}")
        logger.info(f"   Successful iterations: {self.stats['successful_generations']}/{self.stats['total_iterations']}")
        logger.info(f"   Success rate: {self.stats['successful_generations']/self.stats['total_iterations']*100:.1f}%")
        
        if len(augmented_samples) > 0:
            df = pd.DataFrame(augmented_samples)
            label_counts = df['label'].value_counts()
            malicious_count = label_counts.get(1, 0)
            benign_count = label_counts.get(0, 0)
            
            logger.info(f"\n🏷️  Label Distribution:")
            logger.info(f"   Malicious samples: {malicious_count} ({malicious_count/len(augmented_samples)*100:.1f}%)")
            logger.info(f"   Benign samples: {benign_count} ({benign_count/len(augmented_samples)*100:.1f}%)")
        
        logger.info(f"\n🔍 Quality Control:")
        logger.info(f"   Duplicates filtered: {self.stats['duplicate_samples_filtered']}")
        logger.info(f"   Schema violations filtered: {self.stats['schema_violations_filtered']}")
        logger.info(f"   Failed generations: {self.stats['failed_generations']}")
        
        logger.info(f"\n⏱️  Performance:")
        logger.info(f"   Total processing time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        logger.info(f"   Average per iteration: {total_time/self.stats['total_iterations']:.2f} seconds")
        logger.info(f"   Samples per minute: {len(augmented_samples)/(total_time/60):.1f}")
        
        logger.info(f"\n📁 Output:")
        logger.info(f"   Output directory: {self.config_manager.data_dir / self.output_dir_name}")
        logger.info(f"   Output file: {self.output_file_name}")
        
        logger.info("="*80)


def main(n_sample_time: int = None,
         samples_per_iteration: int = None,
         data_path: str = None,
         output_file: str = None):
    """
    Main function for LLM-based data augmentation.
    
    Args:
        n_sample_time (int): Number of iterations (default: 20)
        samples_per_iteration (int): Samples to generate per iteration (default: 5)
        data_path (str): Path to source data file (default: email_phishing_CEAS-08_train.csv.gz)
        output_file (str): Output file name (default: email_phishing_CEAS-08_train_augment.csv.gz)
    """
    logger.info("="*80)
    logger.info("LLM-BASED DATA AUGMENTATION")
    logger.info("="*80)
    
    # Build configuration overrides
    config_override = {}
    if n_sample_time is not None:
        config_override['n_sample_time'] = n_sample_time
    if samples_per_iteration is not None:
        config_override['samples_per_iteration'] = samples_per_iteration
    if data_path is not None:
        config_override['data_path'] = data_path
    if output_file is not None:
        config_override['output_file_name'] = output_file
    
    try:
        # Initialize augmenter
        augmenter = DataAugmenter(config_override if config_override else None)
        
        # Load source data
        raw_dir = config_manager.project_root / "raw"
        source_file_path = raw_dir / augmenter.data_path
        source_df = augmenter.load_source_data(source_file_path)
        
        # Create stratified source pool
        source_pool = augmenter.stratified_sample_source(source_df)
        
        # Process augmentation
        augmented_samples = augmenter.process_augmentation(source_pool)
        
        # Save results
        if augmented_samples:
            output_path = augmenter.save_augmented_dataset(augmented_samples)
            
            # Generate summary report
            augmenter.generate_summary_report(augmented_samples)
            
            logger.info(f"\n🎉 Augmentation completed successfully!")
            logger.info(f"Generated {len(augmented_samples)} high-quality augmented samples")
            logger.info(f"Output saved to: {output_path}")
        else:
            logger.error("No augmented samples were generated. Check the logs for issues.")
            
    except Exception as e:
        logger.error(f"Error in augmentation process: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM-based data augmentation for cybersecurity datasets")
    parser.add_argument('--n-sample-time', type=int, default=N_SAMPLE_TIME,
                       help=f'Number of augmentation iterations (default: {N_SAMPLE_TIME})')
    parser.add_argument('--samples-per-iteration', type=int, default=SAMPLES_PER_ITERATION,
                       help=f'Number of samples to generate per iteration (default: {SAMPLES_PER_ITERATION})')
    parser.add_argument('--data-path', default=DATA_PATH,
                       help=f'Source data file path in raw/ directory (default: {DATA_PATH})')
    parser.add_argument('--output-file', default=OUTPUT_FILE_NAME,
                       help=f'Output file name (default: {OUTPUT_FILE_NAME})')
    
    args = parser.parse_args()
    
    main(
        n_sample_time=args.n_sample_time,
        samples_per_iteration=args.samples_per_iteration,
        data_path=args.data_path,
        output_file=args.output_file
    )