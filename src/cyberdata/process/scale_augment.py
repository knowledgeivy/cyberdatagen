# cyberdata/process/scale_augment.py

import concurrent.futures
import gzip
import json
import pandas as pd
import random
import sys
import time
import threading
from collections import defaultdict
from dataclasses import dataclass
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
logger = setup_logger("cyberdata.scripts.scale_augment")

# Load environment variables
load_dotenv()

# Configuration constants
MODEL_NAME = "gpt-4.1-mini"
DEFAULT_SCALE_COUNT = 1000
DEFAULT_MALICIOUS_RATIO = 0.5
DEFAULT_MAX_WORKERS = 8
DEFAULT_BATCH_SIZE = 10

# Get config manager instance
config_manager = get_config_manager()

logger.info(f"Using model: {MODEL_NAME}")


@dataclass
class ScaleAugmentTask:
    """Represents a single scale augmentation task."""
    task_id: str
    sample_type: str  # 'malicious' or 'benign'
    batch_size: int
    seed_samples: List[Dict[str, Any]]
    priority: int = 0


@dataclass
class ScaleAugmentResult:
    """Represents the result of a scale augmentation task."""
    task_id: str
    samples: List[Dict]
    sample_type: str
    success: bool
    generation_time: float = 0.0
    error: Optional[str] = None


class EnhancedDuplicateDetector:
    """Advanced duplicate detection for scale augmentation."""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.seen_samples: List[Dict] = []
        self.lock = threading.Lock()
    
    def _calculate_similarity(self, sample1: Dict, sample2: Dict) -> float:
        """Calculate semantic similarity between two samples."""
        try:
            # Extract comparable fields
            exclude_fields = {'generation_timestamp', 'task_id', '_metadata', 'sample_id', '_augmentation_metadata'}
            
            text1_parts = []
            text2_parts = []
            
            for key in set(sample1.keys()).union(set(sample2.keys())):
                if key not in exclude_fields:
                    val1 = str(sample1.get(key, '')).lower().strip()
                    val2 = str(sample2.get(key, '')).lower().strip()
                    text1_parts.append(val1)
                    text2_parts.append(val2)
            
            text1 = ' '.join(text1_parts)
            text2 = ' '.join(text2_parts)
            
            # Simple similarity calculation
            words1 = set(text1.split())
            words2 = set(text2.split())
            
            if not words1 and not words2:
                return 1.0
            elif not words1 or not words2:
                return 0.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating similarity: {e}")
            return 0.0
    
    def is_duplicate(self, sample: Dict) -> bool:
        """Check if sample is too similar to existing samples."""
        with self.lock:
            for existing_sample in self.seen_samples:
                similarity = self._calculate_similarity(sample, existing_sample)
                if similarity >= self.similarity_threshold:
                    return True
            
            # Add to seen samples (limit memory usage)
            self.seen_samples.append(sample)
            if len(self.seen_samples) > 2000:  # Keep only recent samples
                self.seen_samples = self.seen_samples[-1000:]
            
            return False


class ScaleAugmenter:
    """Scale augmentation system for generating large datasets from seed examples."""
    
    def __init__(self, config_override: Dict[str, Any] = None):
        self.config_manager = config_manager
        
        # Configuration
        config = config_override or {}
        self.scale_count = config.get('scale_count', DEFAULT_SCALE_COUNT)
        self.malicious_ratio = config.get('malicious_ratio', DEFAULT_MALICIOUS_RATIO)
        self.max_workers = config.get('max_workers', DEFAULT_MAX_WORKERS)
        self.batch_size = config.get('batch_size', DEFAULT_BATCH_SIZE)
        
        # Initialize duplicate detector
        self.duplicate_detector = EnhancedDuplicateDetector()
        
        # Statistics tracking
        self.stats = {
            'total_generated': 0,
            'total_unique': 0,
            'duplicates_filtered': 0,
            'failed_generations': 0,
            'successful_batches': 0,
            'total_api_calls': 0,
            'generation_start_time': None,
            'generation_end_time': None,
            'malicious_generated': 0,
            'benign_generated': 0
        }
        
        # Thread lock for stats
        self.stats_lock = threading.Lock()
        
        logger.info(f"ScaleAugmenter initialized:")
        logger.info(f"  - Scale count: {self.scale_count}")
        logger.info(f"  - Malicious ratio: {self.malicious_ratio:.1%}")
        logger.info(f"  - Max workers: {self.max_workers}")
        logger.info(f"  - Batch size: {self.batch_size}")
    
    def load_augmented_seeds(self, file_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load augmented seed data from compressed CSV file."""
        logger.info(f"Loading augmented seeds from: {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Augmented seeds file not found: {file_path}")
        
        try:
            # Load the compressed CSV
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    df = pd.read_csv(f)
            else:
                df = pd.read_csv(file_path)
            
            logger.info(f"Loaded {len(df)} augmented seed samples")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Load metadata if available
            metadata_file = file_path.parent / f"{file_path.stem}_metadata.json"
            if metadata_file.exists():
                with metadata_file.open('r', encoding='utf-8') as f:
                    metadata = json.load(f)
                logger.info("Loaded augmented seeds metadata")
            else:
                metadata = {}
                logger.warning("No metadata file found for augmented seeds")
            
            # Log label distribution
            if 'label' in df.columns:
                label_counts = df['label'].value_counts()
                logger.info(f"Seed label distribution: {dict(label_counts)}")
            
            return df, metadata
            
        except Exception as e:
            logger.error(f"Error loading augmented seeds: {str(e)}")
            raise
    
    def stratify_seed_samples(self, df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """Stratify seed samples by label for balanced generation."""
        logger.info("Stratifying seed samples by label")
        
        # Convert to records and separate by label
        malicious_seeds = df[df['label'] == 1].to_dict('records')
        benign_seeds = df[df['label'] == 0].to_dict('records')
        
        logger.info(f"Seed distribution - Malicious: {len(malicious_seeds)}, Benign: {len(benign_seeds)}")
        
        if not malicious_seeds and not benign_seeds:
            raise ValueError("No valid seed samples found")
        
        # Ensure we have at least some seeds of each type if they exist
        if not malicious_seeds:
            logger.warning("No malicious seed samples found")
        if not benign_seeds:
            logger.warning("No benign seed samples found")
        
        return malicious_seeds, benign_seeds
    
    def create_scale_tasks(self, malicious_seeds: List[Dict], benign_seeds: List[Dict]) -> List[ScaleAugmentTask]:
        """Create scale augmentation tasks based on target distribution."""
        logger.info("Creating scale augmentation tasks")
        
        # Calculate target counts
        malicious_target = int(self.scale_count * self.malicious_ratio)
        benign_target = self.scale_count - malicious_target
        
        logger.info(f"Target distribution - Malicious: {malicious_target}, Benign: {benign_target}")
        
        tasks = []
        task_id = 0
        
        # Create malicious generation tasks
        if malicious_seeds and malicious_target > 0:
            remaining_malicious = malicious_target
            while remaining_malicious > 0:
                current_batch = min(self.batch_size, remaining_malicious)
                task = ScaleAugmentTask(
                    task_id=f"mal_scale_{task_id}",
                    sample_type='malicious',
                    batch_size=current_batch,
                    seed_samples=malicious_seeds,  # All malicious seeds available for inspiration
                    priority=2  # Higher priority for malicious
                )
                tasks.append(task)
                remaining_malicious -= current_batch
                task_id += 1
        
        # Create benign generation tasks
        if benign_seeds and benign_target > 0:
            remaining_benign = benign_target
            while remaining_benign > 0:
                current_batch = min(self.batch_size, remaining_benign)
                task = ScaleAugmentTask(
                    task_id=f"ben_scale_{task_id}",
                    sample_type='benign',
                    batch_size=current_batch,
                    seed_samples=benign_seeds,  # All benign seeds available for inspiration
                    priority=1
                )
                tasks.append(task)
                remaining_benign -= current_batch
                task_id += 1
        
        # Sort by priority
        tasks.sort(key=lambda x: x.priority, reverse=True)
        
        logger.info(f"Created {len(tasks)} scale augmentation tasks")
        return tasks
    
    def execute_scale_task(self, task: ScaleAugmentTask) -> ScaleAugmentResult:
        """Execute a single scale augmentation task."""
        start_time = time.time()
        
        logger.info(f"Executing task {task.task_id}: {task.batch_size} {task.sample_type} samples")
        
        try:
            # Select random seed samples for this batch
            selected_seeds = random.sample(task.seed_samples, min(3, len(task.seed_samples)))
            
            # Generate samples
            samples = self._generate_scale_batch(task, selected_seeds)
            
            # Filter duplicates
            unique_samples = []
            duplicates_count = 0
            
            for sample in samples:
                if not self.duplicate_detector.is_duplicate(sample):
                    # Add minimal metadata
                    sample['sample_id'] = f"{task.task_id}_{len(unique_samples)}"
                    sample['generation_timestamp'] = time.time()
                    sample['_scale_metadata'] = {
                        'generation_method': 'scale_augmentation',
                        'task_id': task.task_id,
                        'sample_type': task.sample_type
                    }
                    unique_samples.append(sample)
                else:
                    duplicates_count += 1
            
            # Update stats
            with self.stats_lock:
                self.stats['total_generated'] += len(samples)
                self.stats['total_unique'] += len(unique_samples)
                self.stats['duplicates_filtered'] += duplicates_count
                self.stats['total_api_calls'] += 1
                if unique_samples:
                    self.stats['successful_batches'] += 1
                if task.sample_type == 'malicious':
                    self.stats['malicious_generated'] += len(unique_samples)
                else:
                    self.stats['benign_generated'] += len(unique_samples)
            
            generation_time = time.time() - start_time
            
            logger.info(f"Task {task.task_id} completed: {len(unique_samples)} unique samples "
                       f"({duplicates_count} duplicates) in {generation_time:.2f}s")
            
            return ScaleAugmentResult(
                task_id=task.task_id,
                samples=unique_samples,
                sample_type=task.sample_type,
                success=True,
                generation_time=generation_time
            )
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            
            with self.stats_lock:
                self.stats['failed_generations'] += 1
            
            return ScaleAugmentResult(
                task_id=task.task_id,
                samples=[],
                sample_type=task.sample_type,
                success=False,
                generation_time=time.time() - start_time,
                error=str(e)
            )
    
    def _generate_scale_batch(self, task: ScaleAugmentTask, seed_samples: List[Dict]) -> List[Dict]:
        """Generate a batch of samples using LLM."""
        try:
            # Format seed examples for prompt
            seed_examples_formatted = self._format_seed_examples(seed_samples, detailed=True)
            seed_examples_simple = self._format_seed_examples(seed_samples, detailed=False)
            
            # Prepare generation context
            generation_context = {
                'sample_type': task.sample_type,
                'batch_size': task.batch_size,
                'seed_samples': seed_samples,
                'task_id': task.task_id,
                'schema_requirements': {
                    'columns': ['subject', 'body', 'label', 'source'],
                    'label_encoding': {'malicious': 1, 'benign': 0}
                }
            }
            
            context_json = json.dumps(generation_context, indent=2, default=str)
            
            # Determine label and description
            label_value = 1 if task.sample_type == 'malicious' else 0
            label_description = "MALICIOUS/PHISHING" if task.sample_type == 'malicious' else "BENIGN/LEGITIMATE"
            
            # Load scale augmentation prompts
            try:
                if task.sample_type == 'malicious':
                    system_prompt = load_prompt(
                        "scale_augment",
                        "prompts.malicious_scale_generation.system.template"
                    )
                    user_prompt = load_prompt(
                        "scale_augment",
                        "prompts.malicious_scale_generation.user.template",
                        generation_context_json=context_json,
                        target_samples=task.batch_size,
                        sample_type=task.sample_type,
                        label_value=label_value,
                        label_description=label_description,
                        seed_examples_formatted=seed_examples_formatted
                    )
                else:
                    system_prompt = load_prompt(
                        "scale_augment",
                        "prompts.benign_scale_generation.system.template"
                    )
                    user_prompt = load_prompt(
                        "scale_augment",
                        "prompts.benign_scale_generation.user.template",
                        generation_context_json=context_json,
                        target_samples=task.batch_size,
                        sample_type=task.sample_type,
                        label_value=label_value,
                        label_description=label_description,
                        seed_examples_formatted=seed_examples_formatted
                    )
            except Exception as e:
                logger.warning(f"Error loading specific prompts: {e}, using fallback")
                system_prompt = self._get_fallback_system_prompt(task.sample_type)
                user_prompt = self._get_fallback_user_prompt(task, seed_samples)
            
            # Call LLM for generation
            response_content = process_llm_request(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name=MODEL_NAME,
                temperature=0.8  # Higher temperature for diversity
            )
            
            # Parse response
            return self._parse_scale_response(response_content)
            
        except Exception as e:
            logger.error(f"Error generating scale batch: {e}")
            return []
    
    def _format_seed_examples(self, seed_samples: List[Dict], detailed: bool = True) -> str:
        """Format seed examples for prompt inclusion."""
        if not seed_samples:
            return "No seed examples available."
        
        formatted_examples = []
        
        for i, seed in enumerate(seed_samples[:3], 1):  # Limit to 3 examples to avoid token issues
            subject = seed.get('subject', '')
            body = seed.get('body', '')
            label = seed.get('label', 0)
            source = seed.get('source', '')
            
            if detailed:
                # Truncate body for readability
                body_preview = body[:200] + "..." if len(body) > 200 else body
                label_desc = "MALICIOUS" if label == 1 else "BENIGN"
                
                example = f"""**Example {i}:**
- Subject: {subject}
- Body: {body_preview}
- Label: {label} ({label_desc})
- Source: {source}"""
            else:
                # Simple format for fallback
                body_preview = body[:100] + "..." if len(body) > 100 else body
                example = f'{i}. Subject: "{subject}" | Body: "{body_preview}" | Label: {label}'
            
            formatted_examples.append(example)
        
        return "\n\n".join(formatted_examples)
    
    def _get_fallback_system_prompt(self, sample_type: str) -> str:
        """Fallback system prompt if loading from YAML fails."""
        if sample_type == 'malicious':
            return """You are an expert cybersecurity dataset generator specializing in creating realistic phishing email samples for training machine learning models.

Generate diverse, realistic malicious email samples that represent authentic phishing attacks while maintaining proper dataset structure and labeling."""
        else:
            return """You are an expert cybersecurity dataset generator specializing in creating realistic legitimate email samples for training machine learning models.

Generate diverse, realistic benign email samples that represent authentic business communications while maintaining proper dataset structure and labeling."""
    
    def _get_fallback_user_prompt(self, task: ScaleAugmentTask, seed_samples: List[Dict]) -> str:
        """Fallback user prompt if loading from YAML fails."""
        label_value = 1 if task.sample_type == 'malicious' else 0
        sample_type_desc = "malicious phishing" if task.sample_type == 'malicious' else "benign legitimate"
        
        return f"""Generate {task.batch_size} realistic {sample_type_desc} email samples based on these seed examples:

Seed Examples:
{json.dumps(seed_samples[:2], indent=2)}

Requirements:
- Generate {task.batch_size} unique samples
- All samples must have label = {label_value}
- Follow exact schema: subject, body, label, source
- Create diverse but realistic variations
- Maintain {sample_type_desc} characteristics

Return ONLY valid JSON:
{{
  "samples": [
    {{
      "subject": "sample subject",
      "body": "sample body",
      "label": {label_value},
      "source": "scale_generated"
    }}
    // ... {task.batch_size} total samples
  ]
}}"""
    
    def _parse_scale_response(self, response_content: str) -> List[Dict]:
        """Parse LLM scale generation response."""
        try:
            # Clean markdown code blocks
            if response_content.startswith('```'):
                first_backticks_end = response_content.find('\n', 3)
                if first_backticks_end != -1:
                    last_backticks_start = response_content.rfind('```')
                    if last_backticks_start > first_backticks_end:
                        response_content = response_content[first_backticks_end + 1:last_backticks_start].strip()
            
            parsed = json.loads(response_content)
            samples = parsed.get('samples', [])
            
            if not isinstance(samples, list):
                logger.warning("Invalid samples format in response")
                return []
            
            return samples
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse scale generation response: {e}")
            return []
    
    def process_scale_augmentation(self, seeds_file: Path) -> List[Dict]:
        """Process the complete scale augmentation pipeline."""
        logger.info("Starting scale augmentation process")
        logger.info("="*80)
        
        self.stats['generation_start_time'] = time.time()
        
        # Load augmented seeds
        seeds_df, seeds_metadata = self.load_augmented_seeds(seeds_file)
        
        # Stratify seeds by label
        malicious_seeds, benign_seeds = self.stratify_seed_samples(seeds_df)
        
        # Create scale tasks
        scale_tasks = self.create_scale_tasks(malicious_seeds, benign_seeds)
        
        # Execute tasks in parallel
        all_samples = []
        
        logger.info(f"🚀 Starting parallel scale generation with {self.max_workers} workers")
        logger.info("="*80)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self.execute_scale_task, task): task 
                for task in scale_tasks
            }
            
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    if result.success:
                        all_samples.extend(result.samples)
                        logger.info(f"✓ Task {result.task_id} contributed {len(result.samples)} samples")
                    else:
                        logger.error(f"✗ Task {result.task_id} failed: {result.error}")
                except Exception as e:
                    logger.error(f"Task {task.task_id} raised exception: {e}")
        
        # Shuffle samples
        random.shuffle(all_samples)
        
        self.stats['generation_end_time'] = time.time()
        
        logger.info("\n" + "="*80)
        logger.info("SCALE AUGMENTATION COMPLETED")
        logger.info("="*80)
        
        return all_samples
    
    def save_scaled_dataset(self, samples: List[Dict], seeds_file: Path, seeds_metadata: Dict) -> Path:
        """Save the scaled dataset with comprehensive metadata."""
        logger.info("Saving scaled dataset...")
        
        # Create output directory
        output_dir = self.config_manager.data_dir / "scaled-augment"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename based on input
        base_name = seeds_file.stem.replace('_metadata', '')
        output_name = f"{base_name}_scaled_{self.scale_count}.csv.gz"
        output_path = output_dir / output_name
        
        # Convert to DataFrame and save
        df = pd.DataFrame(samples)
        
        if len(df) > 0:
            with gzip.open(output_path, 'wt', encoding='utf-8') as f:
                df.to_csv(f, index=False)
        else:
            logger.warning("No samples to save")
            return None
        
        # Calculate statistics
        total_time = self.stats['generation_end_time'] - self.stats['generation_start_time']
        
        # Create comprehensive metadata
        metadata = {
            'generation_method': 'scale_augmentation_from_seeds',
            'model_used': MODEL_NAME,
            'source_data': {
                'seeds_file': str(seeds_file),
                'seeds_metadata': seeds_metadata
            },
            'configuration': {
                'scale_count': self.scale_count,
                'malicious_ratio': self.malicious_ratio,
                'max_workers': self.max_workers,
                'batch_size': self.batch_size
            },
            'generation_stats': self.stats.copy(),
            'quality_metrics': {
                'total_samples_generated': len(samples),
                'malicious_samples': self.stats['malicious_generated'],
                'benign_samples': self.stats['benign_generated'],
                'actual_malicious_ratio': self.stats['malicious_generated'] / len(samples) if samples else 0,
                'duplicate_filter_rate': self.stats['duplicates_filtered'] / max(1, self.stats['total_generated']),
                'success_rate': self.stats['successful_batches'] / max(1, len(samples) // self.batch_size)
            },
            'performance_metrics': {
                'total_time_seconds': total_time,
                'samples_per_minute': len(samples) / (total_time / 60) if total_time > 0 else 0,
                'average_batch_time': total_time / max(1, self.stats['total_api_calls'])
            },
            'dataset_timestamp': time.time(),
            'description': f'Scaled cybersecurity dataset generated from augmented seeds using LLM'
        }
        
        # Save metadata
        metadata_path = output_dir / f"{base_name}_scaled_{self.scale_count}_metadata.json"
        with metadata_path.open('w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Scaled dataset saved to: {output_path}")
        logger.info(f"Metadata saved to: {metadata_path}")
        
        return output_path
    
    def generate_summary_report(self, samples: List[Dict], seeds_file: Path) -> None:
        """Generate and log comprehensive summary report."""
        total_time = self.stats['generation_end_time'] - self.stats['generation_start_time']
        
        logger.info("\n" + "="*80)
        logger.info("SCALE AUGMENTATION SUMMARY REPORT")
        logger.info("="*80)
        
        logger.info(f"📊 Scale Configuration:")
        logger.info(f"   Source seeds: {seeds_file}")
        logger.info(f"   Target scale count: {self.scale_count}")
        logger.info(f"   Target malicious ratio: {self.malicious_ratio:.1%}")
        logger.info(f"   Workers used: {self.max_workers}")
        logger.info(f"   Batch size: {self.batch_size}")
        logger.info(f"   Model used: {MODEL_NAME}")
        
        logger.info(f"\n📈 Generation Results:")
        logger.info(f"   Actual samples generated: {len(samples)}")
        logger.info(f"   Target vs Actual: {len(samples)}/{self.scale_count} ({len(samples)/self.scale_count*100:.1f}%)")
        logger.info(f"   Successful batches: {self.stats['successful_batches']}")
        logger.info(f"   Failed generations: {self.stats['failed_generations']}")
        
        if len(samples) > 0:
            df = pd.DataFrame(samples)
            label_counts = df['label'].value_counts()
            malicious_count = label_counts.get(1, 0)
            benign_count = label_counts.get(0, 0)
            actual_ratio = malicious_count / len(samples) if len(samples) > 0 else 0
            
            logger.info(f"\n🏷️  Label Distribution:")
            logger.info(f"   Malicious samples: {malicious_count} ({actual_ratio:.1%})")
            logger.info(f"   Benign samples: {benign_count} ({(1-actual_ratio):.1%})")
            logger.info(f"   Target vs Actual ratio: {self.malicious_ratio:.1%} vs {actual_ratio:.1%}")
        
        logger.info(f"\n🔍 Quality Control:")
        logger.info(f"   Total generated: {self.stats['total_generated']}")
        logger.info(f"   Unique samples: {self.stats['total_unique']}")
        logger.info(f"   Duplicates filtered: {self.stats['duplicates_filtered']}")
        logger.info(f"   Duplicate rate: {self.stats['duplicates_filtered']/max(1, self.stats['total_generated'])*100:.1f}%")
        
        logger.info(f"\n⏱️  Performance:")
        logger.info(f"   Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        logger.info(f"   API calls made: {self.stats['total_api_calls']}")
        logger.info(f"   Samples per minute: {len(samples)/(total_time/60):.1f}")
        logger.info(f"   Average batch time: {total_time/max(1, self.stats['total_api_calls']):.2f}s")
        
        logger.info("="*80)


def find_augmented_seeds_files() -> List[Path]:
    """Find all augmented seeds files in the seeds-augment directory."""
    seeds_augment_dir = config_manager.data_dir / "seeds-augment"
    
    if not seeds_augment_dir.exists():
        logger.warning(f"Seeds-augment directory not found: {seeds_augment_dir}")
        return []
    
    # Find .csv.gz files (excluding metadata files)
    seed_files = []
    for file_path in seeds_augment_dir.glob("*.csv.gz"):
        if "_metadata" not in file_path.name:
            seed_files.append(file_path)
    
    logger.info(f"Found {len(seed_files)} augmented seeds files")
    for file_path in seed_files:
        logger.info(f"  - {file_path.name}")
    
    return seed_files


def main(scale_count: int = None,
         malicious_ratio: float = None,
         max_workers: int = None,
         batch_size: int = None,
         seeds_file: str = None):
    """
    Main function for scale augmentation.
    
    Args:
        scale_count (int): Number of samples to generate
        malicious_ratio (float): Ratio of malicious samples
        max_workers (int): Maximum number of worker threads
        batch_size (int): Batch size for generation
        seeds_file (str): Specific seeds file to use (optional)
    """
    logger.info("="*80)
    logger.info("SCALE AUGMENTATION FROM SEEDS")
    logger.info("="*80)
    
    # Build configuration overrides
    config_override = {}
    if scale_count is not None:
        config_override['scale_count'] = scale_count
    if malicious_ratio is not None:
        config_override['malicious_ratio'] = malicious_ratio
    if max_workers is not None:
        config_override['max_workers'] = max_workers
    if batch_size is not None:
        config_override['batch_size'] = batch_size
    
    try:
        # Initialize scale augmenter
        augmenter = ScaleAugmenter(config_override if config_override else None)
        
        # Find or load specific seeds file
        if seeds_file:
            seeds_file_path = config_manager.data_dir / "seeds-augment" / seeds_file
            if not seeds_file_path.exists():
                raise FileNotFoundError(f"Specified seeds file not found: {seeds_file_path}")
            seeds_files = [seeds_file_path]
        else:
            seeds_files = find_augmented_seeds_files()
            if not seeds_files:
                raise FileNotFoundError("No augmented seeds files found. Run real_data_processor_augment.py first.")
        
        # Process each seeds file
        for seeds_file_path in seeds_files:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing seeds file: {seeds_file_path.name}")
            logger.info(f"{'='*60}")
            
            # Process scale augmentation
            scaled_samples = augmenter.process_scale_augmentation(seeds_file_path)
            
            if scaled_samples:
                # Load seeds metadata for context
                seeds_metadata = {}
                metadata_file = seeds_file_path.parent / f"{seeds_file_path.stem}_metadata.json"
                if metadata_file.exists():
                    with metadata_file.open('r', encoding='utf-8') as f:
                        seeds_metadata = json.load(f)
                
                # Save scaled dataset
                output_path = augmenter.save_scaled_dataset(scaled_samples, seeds_file_path, seeds_metadata)
                
                # Generate summary report
                augmenter.generate_summary_report(scaled_samples, seeds_file_path)
                
                logger.info(f"\n🎉 Scale augmentation completed successfully!")
                logger.info(f"Generated {len(scaled_samples)} samples from {seeds_file_path.name}")
                logger.info(f"Output saved to: {output_path}")
            else:
                logger.error(f"No scaled samples generated from {seeds_file_path.name}")
        
    except Exception as e:
        logger.error(f"Error in scale augmentation process: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Scale augmentation from augmented seeds using LLM")
    parser.add_argument('--scale-count', type=int, default=DEFAULT_SCALE_COUNT,
                       help=f'Number of samples to generate (default: {DEFAULT_SCALE_COUNT})')
    parser.add_argument('--malicious-ratio', type=float, default=DEFAULT_MALICIOUS_RATIO,
                       help=f'Ratio of malicious samples (default: {DEFAULT_MALICIOUS_RATIO})')
    parser.add_argument('--max-workers', type=int, default=DEFAULT_MAX_WORKERS,
                       help=f'Maximum number of worker threads (default: {DEFAULT_MAX_WORKERS})')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                       help=f'Batch size for generation (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--seeds-file',
                       help='Specific seeds file to use (optional, will process all if not specified)')
    
    args = parser.parse_args()
    
    main(
        scale_count=args.scale_count,
        malicious_ratio=args.malicious_ratio,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        seeds_file=args.seeds_file
    )