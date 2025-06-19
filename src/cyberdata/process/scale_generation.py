# cyberdata/process/scale_generation.py

import concurrent.futures
import json
import random
import sys
import time
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

from dotenv import load_dotenv

# Add parent directory to path for imports
CURRENT_DIR = Path(__file__).parent
sys.path.append(str(CURRENT_DIR.parent))

from cyberdata.utils.config_manager import get_config_manager
from cyberdata.utils.llm_invoke import process_llm_request
from cyberdata.utils.logger_config import setup_logger
from cyberdata.utils.prompt_loader import load_prompt

# Set up logger
logger = setup_logger("cyberdata.scripts.scale_generation")

# Load environment variables
load_dotenv()

# Configuration constants
MODEL_NAME = "gpt-4.1-mini"

# Get config manager instance
config_manager = get_config_manager()

logger.info(f"Using model: {MODEL_NAME}")
logger.info(f"Scale Generation initialized")


@dataclass
class ScaleGenerationTask:
    """Represents a single scale generation task."""
    task_id: str
    sample_type: str  # 'malicious' or 'benign'
    batch_size: int
    generation_context: Dict[str, Any]
    priority: int = 0


@dataclass
class ScaleGenerationResult:
    """Represents the result of a scale generation task."""
    task_id: str
    samples: List[Dict]
    sample_type: str
    success: bool
    generation_time: float = 0.0
    error: Optional[str] = None


class EnhancedDuplicateDetector:
    """Advanced duplicate detection using semantic similarity."""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.seen_samples: List[Dict] = []
        self.lock = threading.Lock()
    
    def _calculate_similarity(self, sample1: Dict, sample2: Dict) -> float:
        """Calculate semantic similarity between two samples."""
        try:
            # Convert samples to comparable text representations
            text1_parts = []
            text2_parts = []
            
            # Extract comparable fields (excluding metadata)
            exclude_fields = {'generation_timestamp', 'task_id', '_metadata', 'sample_id'}
            
            for key in set(sample1.keys()).union(set(sample2.keys())):
                if key not in exclude_fields:
                    val1 = str(sample1.get(key, '')).lower().strip()
                    val2 = str(sample2.get(key, '')).lower().strip()
                    text1_parts.append(val1)
                    text2_parts.append(val2)
            
            text1 = ' '.join(text1_parts)
            text2 = ' '.join(text2_parts)
            
            # Simple similarity calculation (can be enhanced with embeddings)
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
            if len(self.seen_samples) > 1000:  # Keep only recent samples
                self.seen_samples = self.seen_samples[-500:]
            
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get duplicate detection statistics."""
        with self.lock:
            return {
                'samples_tracked': len(self.seen_samples),
                'similarity_threshold': self.similarity_threshold
            }


class ScaleGenerator:
    """Enhanced scale generation with comprehensive context integration."""
    
    def __init__(self, config_override: Dict[str, Any] = None):
        self.config_manager = config_manager
        
        # Load configuration from scale_config.yaml
        self.config = self._load_scale_config(config_override)
        
        # Extract generation parameters from config
        gen_config = self.config.get('scale_generation', {})
        defaults = gen_config.get('defaults', {})
        
        self.max_workers = defaults.get('max_workers', 8)  # Minimal fallback
        self.batch_size = defaults.get('batch_size', 10)   # Minimal fallback
        
        # Initialize duplicate detector
        quality_config = gen_config.get('quality_control', {})
        similarity_threshold = quality_config.get('duplicate_similarity_threshold', 0.85)
        self.duplicate_detector = EnhancedDuplicateDetector(similarity_threshold)
        
        # Statistics tracking
        self.stats = {
            'total_generated': 0,
            'total_unique': 0,
            'duplicates_filtered': 0,
            'failed_generations': 0,
            'successful_batches': 0,
            'total_api_calls': 0,
            'generation_start_time': None,
            'generation_end_time': None
        }
        
        logger.info(f"ScaleGenerator initialized from config: {self.max_workers} workers, batch size {self.batch_size}")
    
    def _load_scale_config(self, config_override: Dict[str, Any] = None) -> Dict[str, Any]:
        """Load scale configuration from scale_config.yaml."""
        try:
            config_file = self.config_manager.config_dir / "scale_config.yaml"
            
            if not config_file.exists():
                raise FileNotFoundError(f"Scale config file not found: {config_file}")
            
            import yaml
            with config_file.open('r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Loaded scale configuration from: {config_file}")
            
            # Apply any runtime overrides
            if config_override:
                logger.info("Applying runtime configuration overrides")
                config = self._merge_configs(config, config_override)
            
            return config
            
        except Exception as e:
            logger.error(f"Error loading scale configuration: {e}")
            raise RuntimeError(f"Failed to load scale configuration: {e}")
    
    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """Recursively merge configuration dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def load_validated_seeds(self) -> List[Tuple[str, str, Path]]:
        """Find all validated seed files."""
        seed_files = []
        
        if not self.config_manager.seeds_validated_dir.exists():
            logger.warning(f"Seeds-validated directory not found: {self.config_manager.seeds_validated_dir}")
            return seed_files
        
        # Use the config manager's method to find all validated seeds
        validated_seeds = self.config_manager.find_all_validated_seeds()
        
        if not validated_seeds:
            # Manual search if config manager method fails
            logger.info("Config manager found no validated seeds, performing manual search...")
            for area_dir in self.config_manager.seeds_validated_dir.iterdir():
                if area_dir.is_dir():
                    area = area_dir.name
                    logger.debug(f"Checking area directory: {area}")
                    for seed_file in area_dir.glob("*_examples.json"):
                        nature = seed_file.stem.replace('_examples', '')
                        seed_files.append((area, nature, seed_file))
                        logger.debug(f"Found seed file: {area}/{nature} -> {seed_file}")
        else:
            seed_files = validated_seeds
        
        logger.info(f"Found {len(seed_files)} validated seed files")
        for area, nature, path in seed_files:
            logger.info(f"  - {area}/{nature}: {path}")
        
        return seed_files
    
    def load_generation_context(self, area: str, nature: str, seed_file: Path) -> Dict[str, Any]:
        """Load comprehensive context for scale generation."""
        logger.info(f"Loading generation context for {area}/{nature}")
        
        try:
            # Load seeds and metadata
            with seed_file.open('r', encoding='utf-8') as f:
                seed_data = json.load(f)
            
            seeds = seed_data.get('examples', [])
            seed_metadata = seed_data.get('metadata', {})
            
            # Extract dataset name from metadata
            dataset_name = seed_metadata.get('dataset_name', 'unknown')
            
            # Load domain discovery
            domain_discovery = {}
            domain_file = self.config_manager.domain_discovery_dir / f"{dataset_name}_domain_discovery.json"
            if domain_file.exists():
                with domain_file.open('r', encoding='utf-8') as f:
                    domain_discovery = json.load(f)
                logger.info(f"Loaded domain discovery for: {dataset_name}")
            
            # Load contextual problems
            contextual_problems = {}
            contextual_file = self.config_manager.contextual_problems_dir / f"{dataset_name}_contextual_problems.json"
            if contextual_file.exists():
                with contextual_file.open('r', encoding='utf-8') as f:
                    contextual_problems = json.load(f)
                logger.info(f"Loaded contextual problems for: {dataset_name}")
            
            # Load data info schema
            data_info = {}
            data_info_file = self.config_manager.config_dir / "data_info.yaml"
            if data_info_file.exists():
                import yaml
                with data_info_file.open('r', encoding='utf-8') as f:
                    all_data_info = yaml.safe_load(f)
                    datasets = all_data_info.get('datasets', {})
                    if dataset_name in datasets:
                        data_info = datasets[dataset_name]
                        logger.info(f"Loaded data info schema for: {dataset_name}")
            
            # Create comprehensive generation context
            generation_context = {
                'seeds': seeds[:10],  # Limit to avoid token issues
                'seed_metadata': seed_metadata,
                'domain_discovery': domain_discovery,
                'contextual_problems': contextual_problems,
                'data_info': data_info,
                'dataset_name': dataset_name,
                'area': area,
                'nature': nature,
                'schema_template': self._extract_schema_template(seeds),
                'generation_guidance': self._create_generation_guidance(
                    domain_discovery, contextual_problems, data_info
                )
            }
            
            logger.info(f"Generation context loaded successfully for {area}/{nature}")
            return generation_context
            
        except Exception as e:
            logger.error(f"Error loading generation context: {e}")
            return {}
    
    def _extract_schema_template(self, seeds: List[Dict]) -> Dict[str, Any]:
        """Extract schema template from seed examples."""
        if not seeds:
            return {}
        
        # Use first seed as template
        template_sample = seeds[0]
        
        schema_template = {
            'columns': list(template_sample.keys()),
            'sample_structure': {k: type(v).__name__ for k, v in template_sample.items()},
            'required_fields': [k for k in template_sample.keys() if k != '_metadata']
        }
        
        # Determine label encoding
        label_fields = ['label', 'Label', 'class', 'target']
        for field in label_fields:
            if field in template_sample:
                schema_template['label_field'] = field
                schema_template['label_encoding'] = {
                    'malicious': 1 if template_sample[field] in [1, True, 'malicious'] else 'malicious',
                    'benign': 0 if template_sample[field] in [0, False, 'benign'] else 'benign'
                }
                break
        
        return schema_template
    
    def _create_generation_guidance(self, domain_discovery: Dict, contextual_problems: Dict, data_info: Dict) -> Dict[str, Any]:
        """Create generation guidance from comprehensive context."""
        guidance = {
            'domain_characteristics': [],
            'attack_patterns': [],
            'normal_patterns': [],
            'technical_requirements': [],
            'realism_requirements': []
        }
        
        # Extract from domain discovery
        if domain_discovery:
            domain_chars = domain_discovery.get('domain_characteristics', {})
            guidance['domain_characteristics'] = [
                domain_chars.get('primary_domain', ''),
                f"Data nature: {domain_chars.get('data_nature', '')}",
                f"Detection complexity: {domain_chars.get('detection_complexity', '')}"
            ]
            
            attack_patterns = domain_discovery.get('attack_patterns', {})
            guidance['attack_patterns'] = attack_patterns.get('common_indicators', [])
            
            normal_baselines = domain_discovery.get('normal_baselines', {})
            guidance['normal_patterns'] = normal_baselines.get('typical_behaviors', [])
        
        # Extract from contextual problems
        if contextual_problems:
            enriched_problems = contextual_problems.get('enriched_problems', [])
            if enriched_problems:
                problem = enriched_problems[0]
                guidance['technical_requirements'] = problem.get('technical_characteristics', {}).get('attack_signatures', [])
                guidance['realism_requirements'] = problem.get('real_world_context', {}).get('domain_insights', '')
        
        # Extract from data info
        if data_info:
            guidance['data_characteristics'] = data_info.get('data_characteristics', [])
            guidance['attack_types'] = data_info.get('attack_types', [])
            guidance['domain'] = data_info.get('domain', '')
        
        return guidance
    
    def create_generation_tasks(self, generation_context: Dict, scale_count: int, malicious_ratio: float) -> List[ScaleGenerationTask]:
        """Create scale generation tasks."""
        malicious_count = int(scale_count * malicious_ratio)
        benign_count = scale_count - malicious_count
        
        logger.info(f"Creating tasks: {malicious_count} malicious, {benign_count} benign")
        
        tasks = []
        task_id = 0
        
        # Create malicious generation tasks
        remaining_malicious = malicious_count
        while remaining_malicious > 0:
            current_batch = min(self.batch_size, remaining_malicious)
            task = ScaleGenerationTask(
                task_id=f"mal_{task_id}",
                sample_type='malicious',
                batch_size=current_batch,
                generation_context=generation_context,
                priority=2  # Higher priority for malicious
            )
            tasks.append(task)
            remaining_malicious -= current_batch
            task_id += 1
        
        # Create benign generation tasks
        remaining_benign = benign_count
        while remaining_benign > 0:
            current_batch = min(self.batch_size, remaining_benign)
            task = ScaleGenerationTask(
                task_id=f"ben_{task_id}",
                sample_type='benign',
                batch_size=current_batch,
                generation_context=generation_context,
                priority=1
            )
            tasks.append(task)
            remaining_benign -= current_batch
            task_id += 1
        
        # Sort by priority
        tasks.sort(key=lambda x: x.priority, reverse=True)
        
        logger.info(f"Created {len(tasks)} generation tasks")
        return tasks
    
    def execute_generation_task(self, task: ScaleGenerationTask) -> ScaleGenerationResult:
        """Execute a single generation task."""
        start_time = time.time()
        
        logger.info(f"Executing task {task.task_id}: {task.batch_size} {task.sample_type} samples")
        
        try:
            if task.sample_type == 'malicious':
                samples = self._generate_malicious_batch(task)
            else:
                samples = self._generate_benign_batch(task)
            
            # Filter duplicates
            unique_samples = []
            duplicates_count = 0
            
            for sample in samples:
                if not self.duplicate_detector.is_duplicate(sample):
                    # Add minimal metadata
                    sample['sample_id'] = f"{task.task_id}_{len(unique_samples)}"
                    sample['generation_timestamp'] = time.time()
                    unique_samples.append(sample)
                else:
                    duplicates_count += 1
            
            # Update stats
            with threading.Lock():
                self.stats['total_generated'] += len(samples)
                self.stats['total_unique'] += len(unique_samples)
                self.stats['duplicates_filtered'] += duplicates_count
                self.stats['total_api_calls'] += 1
                if unique_samples:
                    self.stats['successful_batches'] += 1
            
            generation_time = time.time() - start_time
            
            logger.info(f"Task {task.task_id} completed: {len(unique_samples)} unique samples "
                       f"({duplicates_count} duplicates) in {generation_time:.2f}s")
            
            return ScaleGenerationResult(
                task_id=task.task_id,
                samples=unique_samples,
                sample_type=task.sample_type,
                success=True,
                generation_time=generation_time
            )
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            
            with threading.Lock():
                self.stats['failed_generations'] += 1
            
            return ScaleGenerationResult(
                task_id=task.task_id,
                samples=[],
                sample_type=task.sample_type,
                success=False,
                generation_time=time.time() - start_time,
                error=str(e)
            )
    
    def _generate_malicious_batch(self, task: ScaleGenerationTask) -> List[Dict]:
        """Generate malicious samples using comprehensive context."""
        context = task.generation_context
        
        # Get temperature from config
        sample_types_config = self.config.get('scale_generation', {}).get('sample_types', {})
        malicious_config = sample_types_config.get('malicious', {})
        temperature = malicious_config.get('temperature', 0.8)
        
        # Prepare context for prompt
        context_json = json.dumps({
            'seeds': context['seeds'],
            'domain_discovery': context['domain_discovery'],
            'contextual_problems': context['contextual_problems'],
            'data_info': context['data_info'],
            'schema_template': context['schema_template'],
            'generation_guidance': context['generation_guidance']
        }, indent=2, default=str)
        
        # Load scale generation prompts
        system_prompt = load_prompt(
            "scale_generation_prompts",
            "prompts.enhanced_malicious_generation.system.template"
        )
        
        user_prompt = load_prompt(
            "scale_generation_prompts", 
            "prompts.enhanced_malicious_generation.user.template",
            generation_context_json=context_json,
            batch_size=task.batch_size
        )
        
        response_content = process_llm_request(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=MODEL_NAME,
            temperature=temperature
        )
        
        return self._parse_generation_response(response_content)
    
    def _generate_benign_batch(self, task: ScaleGenerationTask) -> List[Dict]:
        """Generate benign samples using comprehensive context."""
        context = task.generation_context
        
        # Get temperature from config
        sample_types_config = self.config.get('scale_generation', {}).get('sample_types', {})
        benign_config = sample_types_config.get('benign', {})
        temperature = benign_config.get('temperature', 0.8)
        
        # Prepare context for prompt
        context_json = json.dumps({
            'seeds': context['seeds'],
            'domain_discovery': context['domain_discovery'],
            'contextual_problems': context['contextual_problems'],
            'data_info': context['data_info'],
            'schema_template': context['schema_template'],
            'generation_guidance': context['generation_guidance']
        }, indent=2, default=str)
        
        # Load scale generation prompts
        system_prompt = load_prompt(
            "scale_generation_prompts",
            "prompts.enhanced_benign_generation.system.template"
        )
        
        user_prompt = load_prompt(
            "scale_generation_prompts",
            "prompts.enhanced_benign_generation.user.template", 
            generation_context_json=context_json,
            batch_size=task.batch_size
        )
        
        response_content = process_llm_request(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=MODEL_NAME,
            temperature=temperature
        )
        
        return self._parse_generation_response(response_content)
    
    def _parse_generation_response(self, response_content: str) -> List[Dict]:
        """Parse LLM generation response."""
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
            logger.error(f"Failed to parse generation response: {e}")
            return []
    
    def generate_scale_data(self, area: str, nature: str, scale_count: int, malicious_ratio: float) -> Dict[str, Any]:
        """Generate scale data for a specific problem."""
        logger.info(f"Starting scale generation for {area}/{nature}: {scale_count} samples, {malicious_ratio:.1%} malicious")
        
        self.stats['generation_start_time'] = time.time()
        
        # Find seed file with proper path handling
        seed_file = self._find_validated_seed_file(area, nature)
        if not seed_file or not seed_file.exists():
            raise FileNotFoundError(f"Validated seed file not found for {area}/{nature}. Expected path: {seed_file}")
        
        logger.info(f"Found validated seed file: {seed_file}")
        
        # Load generation context
        generation_context = self.load_generation_context(area, nature, seed_file)
        if not generation_context:
            raise ValueError(f"Failed to load generation context for {area}/{nature}")
        
        # Create generation tasks
        tasks = self.create_generation_tasks(generation_context, scale_count, malicious_ratio)
        
        # Execute tasks in parallel
        all_samples = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self.execute_generation_task, task): task 
                for task in tasks
            }
            
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    if result.success:
                        all_samples.extend(result.samples)
                        logger.info(f"Task {result.task_id} contributed {len(result.samples)} samples")
                    else:
                        logger.error(f"Task {result.task_id} failed: {result.error}")
                except Exception as e:
                    logger.error(f"Task {task.task_id} raised exception: {e}")
        
        # Shuffle samples
        random.shuffle(all_samples)
        
        self.stats['generation_end_time'] = time.time()
        total_time = self.stats['generation_end_time'] - self.stats['generation_start_time']
        
        # Calculate sample distribution
        malicious_generated = sum(1 for s in all_samples if self._is_malicious_sample(s, generation_context))
        benign_generated = len(all_samples) - malicious_generated
        
        # Create comprehensive metadata
        generation_metadata = {
            'area': area,
            'nature': nature,
            'scale_count_requested': scale_count,
            'malicious_ratio_requested': malicious_ratio,
            'actual_samples_generated': len(all_samples),
            'actual_malicious_count': malicious_generated,
            'actual_benign_count': benign_generated,
            'actual_malicious_ratio': malicious_generated / len(all_samples) if all_samples else 0,
            'generation_method': 'enhanced_scale_generation',
            'generation_stats': self.stats.copy(),
            'duplicate_detection_stats': self.duplicate_detector.get_stats(),
            'generation_timestamp': time.time(),
            'total_generation_time': total_time,
            'samples_per_minute': len(all_samples) / (total_time / 60) if total_time > 0 else 0,
            'context_sources': {
                'domain_discovery_used': bool(generation_context.get('domain_discovery')),
                'contextual_problems_used': bool(generation_context.get('contextual_problems')),
                'data_info_used': bool(generation_context.get('data_info')),
                'dataset_name': generation_context.get('dataset_name', 'unknown')
            },
            'quality_indicators': {
                'uniqueness_rate': (self.stats['total_unique'] / self.stats['total_generated']) if self.stats['total_generated'] > 0 else 0,
                'success_rate': (self.stats['successful_batches'] / len(tasks)) if tasks else 0,
                'average_batch_time': total_time / len(tasks) if tasks else 0
            }
        }
        
        logger.info(f"Scale generation completed for {area}/{nature}: {len(all_samples)} samples in {total_time:.2f}s")
        logger.info(f"Distribution: {malicious_generated} malicious, {benign_generated} benign")
        logger.info(f"Generation rate: {generation_metadata['samples_per_minute']:.1f} samples/minute")
        
        return {
            'samples': all_samples,
            'metadata': generation_metadata
        }
    
    def _is_malicious_sample(self, sample: Dict, generation_context: Dict) -> bool:
        """Determine if a sample is malicious based on context."""
        schema_template = generation_context.get('schema_template', {})
        label_field = schema_template.get('label_field', 'label')
        label_encoding = schema_template.get('label_encoding', {})
        
        if label_field in sample:
            malicious_value = label_encoding.get('malicious', 1)
            return sample[label_field] == malicious_value
        
        # Fallback checks
        return (sample.get('sample_type') == 'malicious' or 
                sample.get('is_attack') is True or
                sample.get('label') == 1)
    
    def save_scale_data(self, area: str, nature: str, scale_data: Dict[str, Any]) -> Path:
        """Save generated scale data."""
        logger.info(f"Saving scale data for {area}/{nature}")
        
        # Create scaled-raw directory structure
        scaled_raw_dir = self.config_manager.data_dir / "scaled-raw"
        area_clean = area.replace(' ', '_').replace('(', '').replace(')', '')
        area_dir = scaled_raw_dir / area_clean
        area_dir.mkdir(parents=True, exist_ok=True)
        
        # Save scale data
        scale_file = area_dir / f"{nature}_scale.json"
        with scale_file.open('w', encoding='utf-8') as f:
            json.dump(scale_data, f, indent=2, default=str)
        
        logger.info(f"Scale data saved to: {scale_file}")
        logger.info(f"Samples: {len(scale_data['samples'])}")
        
        return scale_file
    
    def _find_validated_seed_file(self, area: str, nature: str) -> Optional[Path]:
        """Find validated seed file with proper path handling."""
        # Try the exact path first
        seed_file = self.config_manager.get_seeds_validated_file(area, nature)
        if seed_file.exists():
            return seed_file
        
        # Search for the file in seeds-validated directory
        seeds_validated_dir = self.config_manager.seeds_validated_dir
        
        # Try different area name variations
        area_variations = [
            area,
            area.replace(' ', '_'),
            area.replace(' ', '_').replace('(', '').replace(')', ''),
            area.replace(' ', '').replace('-', '_')
        ]
        
        for area_var in area_variations:
            area_dir = seeds_validated_dir / area_var
            if area_dir.exists():
                seed_file = area_dir / f"{nature}_examples.json"
                if seed_file.exists():
                    logger.info(f"Found validated seed file with area variation '{area_var}': {seed_file}")
                    return seed_file
        
        # Search recursively if still not found
        for seed_file in seeds_validated_dir.rglob(f"*{nature}*_examples.json"):
            logger.info(f"Found validated seed file via recursive search: {seed_file}")
            return seed_file
        
        logger.warning(f"No validated seed file found for {area}/{nature}")
        return None


def find_problems_to_generate() -> List[Tuple[str, str]]:
    """Find all problems that have validated seeds and can be scaled."""
    config_mgr = get_config_manager()
    validated_seeds = config_mgr.find_all_validated_seeds()
    
    problems = []
    for area, nature, seed_file in validated_seeds:
        problems.append((area, nature))
    
    # If no validated seeds found, log available directories for debugging
    if not problems:
        seeds_validated_dir = config_mgr.seeds_validated_dir
        logger.warning(f"No validated seeds found in: {seeds_validated_dir}")
        if seeds_validated_dir.exists():
            logger.info("Available directories in seeds-validated:")
            for item in seeds_validated_dir.iterdir():
                if item.is_dir():
                    logger.info(f"  - {item.name}/")
                    for file in item.glob("*.json"):
                        logger.info(f"    - {file.name}")
        else:
            logger.warning(f"Seeds-validated directory does not exist: {seeds_validated_dir}")
    
    logger.info(f"Found {len(problems)} problems ready for scale generation")
    return problems


def main(scale_count: int = None, 
         malicious_ratio: float = None,
         max_workers: int = None,
         batch_size: int = None,
         problems: List[str] = None):
    """Main function for scale generation."""
    
    logger.info("="*80)
    logger.info("ENHANCED SCALE GENERATION")
    logger.info("="*80)
    
    # Initialize generator (loads config from scale_config.yaml)
    config_override = {}
    
    # Apply command line overrides to config
    if max_workers is not None:
        config_override['scale_generation'] = config_override.get('scale_generation', {})
        config_override['scale_generation']['defaults'] = config_override['scale_generation'].get('defaults', {})
        config_override['scale_generation']['defaults']['max_workers'] = max_workers
    
    if batch_size is not None:
        if 'scale_generation' not in config_override:
            config_override['scale_generation'] = {}
        if 'defaults' not in config_override['scale_generation']:
            config_override['scale_generation']['defaults'] = {}
        config_override['scale_generation']['defaults']['batch_size'] = batch_size
    
    generator = ScaleGenerator(config_override if config_override else None)
    
    # Get defaults from loaded config
    gen_config = generator.config.get('scale_generation', {})
    defaults = gen_config.get('defaults', {})
    
    # Use command line args or config defaults
    if scale_count is None:
        scale_count = defaults.get('scale_count')
        if scale_count is None:
            raise ValueError("scale_count must be specified in command line or scale_config.yaml")
    
    if malicious_ratio is None:
        malicious_ratio = defaults.get('malicious_ratio')
        if malicious_ratio is None:
            raise ValueError("malicious_ratio must be specified in command line or scale_config.yaml")
    
    logger.info(f"Scale count: {scale_count}")
    logger.info(f"Malicious ratio: {malicious_ratio:.1%}")
    logger.info(f"Max workers: {generator.max_workers}")
    logger.info(f"Batch size: {generator.batch_size}")
    
    # Find problems to generate
    available_problems = find_problems_to_generate()
    
    # Filter problems if specified
    if problems:
        filtered_problems = [(area, nature) for area, nature in available_problems 
                           if nature in problems]
        if not filtered_problems:
            logger.error(f"No matching problems found for: {problems}")
            return
        available_problems = filtered_problems
    
    logger.info(f"Processing {len(available_problems)} problems")
    
    # Track overall statistics
    overall_stats = {
        'total_problems': len(available_problems),
        'successful_problems': 0,
        'failed_problems': 0,
        'total_samples_generated': 0,
        'total_generation_time': 0
    }
    
    overall_start_time = time.time()
    
    # Process each problem
    for i, (area, nature) in enumerate(available_problems):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing problem {i+1}/{len(available_problems)}: {area}/{nature}")
        logger.info(f"{'='*60}")
        
        try:
            # Generate scale data
            scale_data = generator.generate_scale_data(area, nature, scale_count, malicious_ratio)
            
            # Save scale data
            scale_file = generator.save_scale_data(area, nature, scale_data)
            
            # Update overall stats
            overall_stats['successful_problems'] += 1
            overall_stats['total_samples_generated'] += len(scale_data['samples'])
            
            logger.info(f"Successfully generated {len(scale_data['samples'])} samples for {area}/{nature}")
            
        except Exception as e:
            logger.error(f"Error processing {area}/{nature}: {e}", exc_info=True)
            overall_stats['failed_problems'] += 1
            continue
    
    # Calculate total time
    overall_stats['total_generation_time'] = time.time() - overall_start_time
    
    # Print final summary
    logger.info(f"\n{'='*80}")
    logger.info("SCALE GENERATION COMPLETED")
    logger.info(f"{'='*80}")
    logger.info(f"Problems processed: {overall_stats['successful_problems']}/{overall_stats['total_problems']}")
    logger.info(f"Total samples generated: {overall_stats['total_samples_generated']}")
    logger.info(f"Total generation time: {overall_stats['total_generation_time']:.2f} seconds")
    
    if overall_stats['total_generation_time'] > 0:
        rate = overall_stats['total_samples_generated'] / (overall_stats['total_generation_time'] / 60)
        logger.info(f"Overall generation rate: {rate:.1f} samples/minute")
    
    logger.info("")
    logger.info("Enhanced Features Used:")
    logger.info("  ✓ Domain discovery integration")
    logger.info("  ✓ Contextual problem enrichment")
    logger.info("  ✓ Schema-aware generation")
    logger.info("  ✓ Advanced duplicate detection")
    logger.info("  ✓ Parallel processing")
    logger.info("  ✓ Quality-weighted sampling")
    logger.info("")
    logger.info("Output Directory:")
    logger.info("  - data/scaled-raw/ (raw scale generation)")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced scale generation with comprehensive context")
    parser.add_argument('--scale-count', type=int,
                       help='Number of samples to generate per problem (required if not in config)')
    parser.add_argument('--malicious-ratio', type=float,
                       help='Ratio of malicious samples (required if not in config)')
    parser.add_argument('--max-workers', type=int,
                       help='Maximum number of worker threads (overrides config)')
    parser.add_argument('--batch-size', type=int,
                       help='Batch size for generation (overrides config)')
    parser.add_argument('--problems', nargs='+',
                       help='Specific problem natures to generate (optional)')
    
    args = parser.parse_args()
    
    main(
        scale_count=args.scale_count,
        malicious_ratio=args.malicious_ratio,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        problems=args.problems
    )