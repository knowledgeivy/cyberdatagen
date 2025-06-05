# cyberdata/scripts/parallel_generator_v2.py

import json
import os
import sys
import re
import random
import hashlib
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import time
import threading
import logging

from dotenv import load_dotenv

# Add parent directory to path for imports
CURRENT_DIR = Path(__file__).parent
sys.path.append(str(CURRENT_DIR.parent))

# Import utilities
from cyberdata.utils.llm_invoke import process_llm_request
from cyberdata.utils.logger_config import setup_logger
from cyberdata.utils.prompt_loader import load_prompt
from cyberdata.utils.config_manager import get_config_manager

# Set up logger
logger = setup_logger("cyberdata.scripts.parallel_generator_v2")

# Load environment variables
load_dotenv()

@dataclass
class GenerationTask:
    """Represents a single generation task"""
    problem: Dict
    sample_type: str  # 'malicious' or 'benign'
    batch_size: int
    task_id: str
    examples: Optional[List] = None
    priority: int = 0

@dataclass
class GenerationResult:
    """Represents the result of a generation task"""
    task_id: str
    samples: List[Dict]
    sample_type: str
    problem_nature: str
    success: bool
    error: Optional[str] = None
    generation_time: float = 0.0

class ConfigurableParallelGenerator:
    """Parallel generator with YAML configuration support"""
    
    def __init__(self, config_override: Optional[Dict] = None, environment: str = "production"):
        """Initialize with configuration from YAML"""
        self.config_manager = get_config_manager()
        self.environment = environment
        
        # Load configuration from YAML
        self.config = self._load_configuration(config_override)
        
        # Initialize components
        self.duplicate_detector = ConfigurableDuplicateDetector(self.config)
        self.stats = {
            "total_generated": 0,
            "duplicates_filtered": 0,
            "failed_generations": 0,
            "successful_batches": 0,
            "total_api_calls": 0,
            "generation_start_time": None,
            "generation_end_time": None
        }
        
        # Get generation settings
        gen_config = self.config.get("parallel_generation", {})
        method = gen_config.get("defaults", {}).get("method", "thread_based")
        
        if method == "thread_based":
            thread_config = gen_config.get("thread_based", {})
            self.max_workers = thread_config.get("max_workers", 4)
            self.default_batch_size = thread_config.get("batch_size", 3)
            self.retry_attempts = thread_config.get("retry_attempts", 3)
        else:
            async_config = gen_config.get("async_based", {})
            self.max_workers = async_config.get("max_concurrent", 15)
            self.default_batch_size = async_config.get("batch_size", 3)
            self.retry_attempts = async_config.get("max_retry_attempts", 3)
        
        logger.info(f"Initialized parallel generator: {self.max_workers} workers, "
                   f"batch size {self.default_batch_size}, environment: {environment}")
    
    def _load_configuration(self, config_override: Optional[Dict] = None) -> Dict:
        """Load configuration from YAML with environment overrides"""
        try:
            # Load the main parallel generation config
            config_file = self.config_manager.prompts_dir / "parallel_generation_config.yaml"
            
            if not config_file.exists():
                logger.warning(f"Parallel generation config not found: {config_file}")
                logger.info("Creating default configuration...")
                self._create_default_config(config_file)
            
            from cyberdata.utils.prompt_loader import get_prompt_loader
            loader = get_prompt_loader()
            config = loader.load_prompt_file("parallel_generation_config")
            
            # Apply environment-specific overrides
            env_config = config.get("environments", {}).get(self.environment, {})
            if env_config:
                logger.info(f"Applying {self.environment} environment overrides")
                config = self._merge_configs(config, {"parallel_generation": {"defaults": env_config}})
            
            # Apply any runtime overrides
            if config_override:
                logger.info("Applying runtime configuration overrides")
                config = self._merge_configs(config, config_override)
            
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.warning("Using fallback default configuration")
            return self._get_fallback_config()
    
    def _create_default_config(self, config_file: Path):
        """Create a default configuration file"""
        default_config = """version: "1.0"
description: "Configuration for parallel synthetic data generation"

parallel_generation:
  thread_based:
    max_workers: 4
    batch_size: 3
    retry_attempts: 3
  
  defaults:
    method: "thread_based"
    temperature:
      malicious_generation: 0.7
      benign_generation: 0.7

duplicate_detection:
  enabled: true
  check_fields: 
    - "scenario"
    - "technical_data"

quality_control:
  validation:
    enabled: true
    sample_rate: 0.1

logging:
  level: "INFO"
  detailed_progress: true
"""
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(default_config)
        logger.info(f"Created default config at {config_file}")
    
    def _get_fallback_config(self) -> Dict:
        """Fallback configuration if YAML loading fails"""
        return {
            "parallel_generation": {
                "thread_based": {
                    "max_workers": 4,
                    "batch_size": 3,
                    "retry_attempts": 3
                },
                "defaults": {
                    "method": "thread_based",
                    "temperature": {
                        "malicious_generation": 0.7,
                        "benign_generation": 0.7
                    }
                }
            },
            "duplicate_detection": {
                "enabled": True,
                "check_fields": ["scenario", "technical_data"]
            },
            "quality_control": {
                "validation": {"enabled": True, "sample_rate": 0.1}
            }
        }
    
    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """Recursively merge configuration dictionaries"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def get_problem_config(self, problem: Dict) -> Dict:
        """Get configuration specific to a problem"""
        nature = problem.get('nature', '')
        problem_configs = self.config.get('problem_specific', {})
        
        # Check for problem-specific config
        if nature in problem_configs:
            logger.debug(f"Using problem-specific config for {nature}")
            return problem_configs[nature]
        
        # Use defaults
        defaults = self.config.get('parallel_generation', {}).get('defaults', {})
        return {
            'batch_size': self.default_batch_size,
            'max_workers': self.max_workers,
            'malicious_ratio': 0.5,
            'priority': 'medium',
            'temperature': defaults.get('temperature', {}).get('malicious_generation', 0.7)
        }
    
    def load_problems(self) -> List[Dict]:
        """Load problem definitions"""
        return self.config_manager.load_problems()
    
    def load_existing_samples(self, problem: Dict) -> List[Dict]:
        """Load existing samples and add them to duplicate detector"""
        try:
            samples_file = self.find_samples_file(problem)
            if samples_file and samples_file.exists():
                data = json.loads(samples_file.read_text(encoding='utf-8'))
                existing_samples = data.get('samples', [])
                logger.info(f"Loaded {len(existing_samples)} existing samples for {problem['nature']}")
                
                # Add to duplicate detector
                self.duplicate_detector.add_existing_samples(existing_samples)
                return existing_samples
        except Exception as e:
            logger.warning(f"Could not load existing samples: {e}")
        
        return []
    
    def find_samples_file(self, problem: Dict) -> Optional[Path]:
        """Find samples file for a problem"""
        area = problem['area']
        nature = problem['nature']
        
        expected_file = self.config_manager.get_large_samples_file(area, nature)
        if expected_file.exists():
            return expected_file
        
        found_file = self.config_manager.find_existing_file(
            self.config_manager.large_samples_dir, 
            nature, 
            "_large.json"
        )
        return found_file
    
    def load_examples(self, problem: Dict) -> List[Dict]:
        """Load seed examples for malicious generation"""
        try:
            area = problem['area']
            nature = problem['nature']
            
            expected_file = self.config_manager.get_seeds_file(area, nature)
            if expected_file.exists():
                data = json.loads(expected_file.read_text(encoding='utf-8'))
                examples = data.get('examples', [])
                logger.debug(f"Loaded {len(examples)} examples for {nature}")
                return examples[:2]  # Limit to avoid token issues
        except Exception as e:
            logger.warning(f"Could not load examples for {problem['nature']}: {e}")
        
        return []
    
    def create_generation_tasks(self, problem: Dict, malicious_count: int, benign_count: int) -> List[GenerationTask]:
        """Create generation tasks based on problem configuration"""
        problem_config = self.get_problem_config(problem)
        batch_size = problem_config.get('batch_size', self.default_batch_size)
        
        tasks = []
        task_counter = 0
        
        # Create malicious generation tasks
        if malicious_count > 0:
            examples = self.load_examples(problem)
            remaining_malicious = malicious_count
            
            while remaining_malicious > 0:
                current_batch = min(batch_size, remaining_malicious)
                task = GenerationTask(
                    problem=problem,
                    sample_type='malicious',
                    batch_size=current_batch,
                    task_id=f"{problem['nature']}_mal_{task_counter}",
                    examples=examples,
                    priority=2  # Higher priority for malicious
                )
                tasks.append(task)
                remaining_malicious -= current_batch
                task_counter += 1
        
        # Create benign generation tasks
        if benign_count > 0:
            remaining_benign = benign_count
            
            while remaining_benign > 0:
                current_batch = min(batch_size, remaining_benign)
                task = GenerationTask(
                    problem=problem,
                    sample_type='benign',
                    batch_size=current_batch,
                    task_id=f"{problem['nature']}_ben_{task_counter}",
                    priority=1
                )
                tasks.append(task)
                remaining_benign -= current_batch
                task_counter += 1
        
        # Sort tasks by priority
        tasks.sort(key=lambda x: x.priority, reverse=True)
        
        logger.info(f"Created {len(tasks)} generation tasks for {problem['nature']} "
                   f"(batch size: {batch_size})")
        return tasks
    
    def execute_generation_task(self, task: GenerationTask) -> GenerationResult:
        """Execute a single generation task with retry logic"""
        start_time = time.time()
        
        for attempt in range(self.retry_attempts):
            try:
                logger.info(f"Executing task {task.task_id} (attempt {attempt + 1}): "
                           f"{task.batch_size} {task.sample_type} samples")
                
                if task.sample_type == 'malicious':
                    samples = self._generate_malicious_batch(task)
                else:
                    samples = self._generate_benign_batch(task)
                
                # Filter duplicates
                unique_samples = []
                duplicates_count = 0
                
                for sample in samples:
                    if not self.duplicate_detector.is_duplicate(sample):
                        # Add sample metadata
                        sample['sample_type'] = task.sample_type
                        sample['is_attack'] = task.sample_type == 'malicious'
                        sample['generation_task_id'] = task.task_id
                        sample['generation_timestamp'] = time.time()
                        unique_samples.append(sample)
                    else:
                        duplicates_count += 1
                
                # Update stats
                self.stats["total_generated"] += len(samples)
                self.stats["duplicates_filtered"] += duplicates_count
                self.stats["total_api_calls"] += 1
                if unique_samples:
                    self.stats["successful_batches"] += 1
                
                generation_time = time.time() - start_time
                
                logger.info(f"Task {task.task_id} completed: {len(unique_samples)} unique samples "
                           f"({duplicates_count} duplicates filtered) in {generation_time:.2f}s")
                
                return GenerationResult(
                    task_id=task.task_id,
                    samples=unique_samples,
                    sample_type=task.sample_type,
                    problem_nature=task.problem['nature'],
                    success=True,
                    generation_time=generation_time
                )
                
            except Exception as e:
                logger.warning(f"Task {task.task_id} attempt {attempt + 1} failed: {e}")
                if attempt == self.retry_attempts - 1:
                    # Final attempt failed
                    self.stats["failed_generations"] += 1
                    generation_time = time.time() - start_time
                    return GenerationResult(
                        task_id=task.task_id,
                        samples=[],
                        sample_type=task.sample_type,
                        problem_nature=task.problem['nature'],
                        success=False,
                        error=str(e),
                        generation_time=generation_time
                    )
                else:
                    # Wait before retry
                    retry_delay = self.config.get('parallel_generation', {}).get('thread_based', {}).get('retry_delay', 2)
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
    
    def _generate_malicious_batch(self, task: GenerationTask) -> List[Dict]:
        """Generate a batch of malicious samples"""
        problem_config = self.get_problem_config(task.problem)
        temperature = problem_config.get('temperature', 0.7)
        
        examples_json = json.dumps(task.examples, indent=2) if task.examples else "[]"
        
        system_content = load_prompt(
            "large_generation_prompts",
            "prompts.generation.system.template"
        )
        
        user_content = load_prompt(
            "large_generation_prompts",
            "prompts.generation.user.template",
            nature=task.problem['nature'],
            area=task.problem['area'],
            description=task.problem.get('description', ''),
            examples_json=examples_json,
            count=task.batch_size
        )
        
        response_content = process_llm_request(
            system_prompt=system_content,
            user_prompt=user_content,
            model_name="gpt-4.1-mini",
            temperature=temperature
        )
        
        parsed = self._extract_json_from_response(response_content)
        return parsed.get('samples', [])
    
    def _generate_benign_batch(self, task: GenerationTask) -> List[Dict]:
        """Generate a batch of benign samples"""
        problem_config = self.get_problem_config(task.problem)
        temperature = problem_config.get('temperature', 0.7)
        
        system_content = load_prompt(
            "large_generation_prompts",
            "prompts.benign_generation.system.template"
        )
        
        user_content = load_prompt(
            "large_generation_prompts",
            "prompts.benign_generation.user.template",
            area=task.problem['area'],
            count=task.batch_size,
            context=task.problem.get('description', '')
        )
        
        response_content = process_llm_request(
            system_prompt=system_content,
            user_prompt=user_content,
            model_name="gpt-4.1-mini",
            temperature=temperature
        )
        
        parsed = self._extract_json_from_response(response_content)
        return parsed.get('samples', [])
    
    def _extract_json_from_response(self, response_content: str) -> Dict:
        """Extract JSON from LLM response"""
        if '```' in response_content:
            pattern = r'```(?:json)?\s*([\s\S]*?)```'
            matches = re.findall(pattern, response_content)
            if matches:
                response_content = matches[0]
        
        try:
            return json.loads(response_content)
        except json.JSONDecodeError:
            try:
                # Fix common JSON issues
                fixed_content = response_content.replace("'", '"')
                fixed_content = re.sub(r'}\s*{', '},{', fixed_content)
                fixed_content = re.sub(r',\s*}', '}', fixed_content)
                fixed_content = re.sub(r',\s*]', ']', fixed_content)
                return json.loads(fixed_content)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response")
                return {"samples": []}
    
    def generate_parallel(self, problem: Dict, malicious_count: int, benign_count: int) -> List[Dict]:
        """Generate samples for a problem using parallel execution"""
        logger.info(f"Starting parallel generation for {problem['nature']}: "
                   f"{malicious_count} malicious, {benign_count} benign")
        
        self.stats["generation_start_time"] = time.time()
        
        # Load existing samples for duplicate detection
        existing_samples = self.load_existing_samples(problem)
        
        # Create generation tasks
        tasks = self.create_generation_tasks(problem, malicious_count, benign_count)
        
        if not tasks:
            logger.warning(f"No tasks created for {problem['nature']}")
            return []
        
        # Get worker count for this problem
        problem_config = self.get_problem_config(problem)
        max_workers = problem_config.get('max_workers', self.max_workers)
        
        # Execute tasks in parallel using ThreadPoolExecutor
        all_samples = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.execute_generation_task, task): task 
                for task in tasks
            }
            
            # Process completed tasks
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
        
        # Shuffle to mix malicious and benign samples
        random.shuffle(all_samples)
        
        self.stats["generation_end_time"] = time.time()
        total_time = self.stats["generation_end_time"] - self.stats["generation_start_time"]
        
        logger.info(f"Parallel generation completed for {problem['nature']}: "
                   f"{len(all_samples)} total samples generated in {total_time:.2f} seconds")
        
        return all_samples
    
    def save_samples(self, problem: Dict, new_samples: List[Dict], existing_samples: List[Dict]):
        """Save generated samples to file with enhanced metadata"""
        if not new_samples:
            logger.warning(f"No samples to save for {problem['nature']}")
            return
        
        area = problem['area']
        nature = problem['nature']
        
        out_file = self.config_manager.get_large_samples_file(area, nature)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Combine existing and new samples
        all_samples = existing_samples + new_samples
        
        # Calculate distributions
        malicious_new = sum(1 for s in new_samples if s.get('sample_type') == 'malicious')
        benign_new = sum(1 for s in new_samples if s.get('sample_type') == 'benign')
        
        # Enhanced metadata
        generation_time = (self.stats.get("generation_end_time", 0) - 
                          self.stats.get("generation_start_time", 0))
        
        metadata = {
            'total_samples': len(all_samples),
            'new_samples_added': len(new_samples),
            'new_malicious_samples': malicious_new,
            'new_benign_samples': benign_new,
            'generation_method': 'configurable_parallel',
            'configuration_used': self.get_problem_config(problem),
            'generation_stats': self.stats.copy(),
            'duplicate_detection_stats': self.duplicate_detector.get_stats(),
            'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_generation_time': generation_time,
            'samples_per_minute': len(new_samples) / (generation_time / 60) if generation_time > 0 else 0,
            'environment': self.environment
        }
        
        output_data = {
            'samples': all_samples,
            'metadata': metadata
        }
        
        with out_file.open('w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Saved {len(new_samples)} new samples to {out_file}")
        logger.info(f"Total samples in file: {len(all_samples)}")
        logger.info(f"Generation rate: {metadata['samples_per_minute']:.1f} samples/minute")

class ConfigurableDuplicateDetector:
    """Duplicate detector with YAML configuration"""
    
    def __init__(self, config: Dict):
        dup_config = config.get('duplicate_detection', {})
        self.enabled = dup_config.get('enabled', True)
        self.check_fields = dup_config.get('check_fields', ['scenario', 'technical_data'])
        
        # Structural similarity settings
        struct_config = dup_config.get('structural_similarity', {})
        self.structural_enabled = struct_config.get('enabled', False)
        self.similarity_threshold = struct_config.get('similarity_threshold', 0.85)
        
        # Normalization settings
        norm_config = dup_config.get('normalization', {})
        self.normalize_lowercase = norm_config.get('lowercase', True)
        self.normalize_whitespace = norm_config.get('remove_extra_whitespace', True)
        self.normalize_values = norm_config.get('remove_specific_values', True)
        
        self.seen_hashes: Set[str] = set()
        self.lock = threading.Lock()
        
        logger.debug(f"Duplicate detector initialized: enabled={self.enabled}, "
                    f"fields={self.check_fields}")
    
    def _normalize_content(self, content: str) -> str:
        """Normalize content based on configuration"""
        if self.normalize_lowercase:
            content = content.lower()
        
        if self.normalize_whitespace:
            content = re.sub(r'\s+', ' ', content.strip())
        
        if self.normalize_values:
            # Replace specific values with placeholders
            content = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', 'IP_ADDR', content)
            content = re.sub(r'\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', 'DOMAIN', content)
            content = re.sub(r'\b[0-9a-fA-F]{8,64}\b', 'HASH', content)
            content = re.sub(r'\b\d{4}-\d{2}-\d{2}', 'DATE', content)
            content = re.sub(r'\b\d{2}:\d{2}:\d{2}', 'TIME', content)
        
        return content
    
    def _generate_content_hash(self, sample: Dict) -> str:
        """Generate a hash based on configured fields"""
        if not self.enabled:
            return str(random.random())  # Always unique if disabled
        
        content_parts = []
        for field in self.check_fields:
            if field in sample:
                content = str(sample[field])
                normalized_content = self._normalize_content(content)
                content_parts.append(normalized_content)
        
        combined_content = "|".join(content_parts)
        return hashlib.sha256(combined_content.encode()).hexdigest()
    
    def is_duplicate(self, sample: Dict) -> bool:
        """Check if sample is a duplicate (thread-safe)"""
        content_hash = self._generate_content_hash(sample)
        
        with self.lock:
            if content_hash in self.seen_hashes:
                return True
            self.seen_hashes.add(content_hash)
            return False
    
    def add_existing_samples(self, samples: List[Dict]):
        """Add existing samples to the duplicate detector"""
        if not self.enabled:
            return
        
        with self.lock:
            for sample in samples:
                content_hash = self._generate_content_hash(sample)
                self.seen_hashes.add(content_hash)
        
        logger.debug(f"Added {len(samples)} existing samples to duplicate detector")
    
    def get_stats(self) -> Dict:
        """Get statistics about duplicate detection"""
        with self.lock:
            return {
                "enabled": self.enabled,
                "total_hashes": len(self.seen_hashes),
                "check_fields": self.check_fields,
                "structural_similarity_enabled": self.structural_enabled
            }

def main(count: int = 100, malicious_ratio: float = 0.5, problem_filter: List[str] = None, 
         config_override: Dict = None, environment: str = "production"):
    """Main function for configurable parallel generation"""
    logger.info(f"Starting configurable parallel generation: count={count}, "
               f"ratio={malicious_ratio}, environment={environment}")
    
    # Validate inputs
    if not 0.0 <= malicious_ratio <= 1.0:
        logger.error(f"Invalid malicious ratio: {malicious_ratio}")
        return
    
    # Calculate sample counts
    malicious_count = int(count * malicious_ratio)
    benign_count = count - malicious_count
    
    logger.info(f"Target distribution: {malicious_count} malicious, {benign_count} benign")
    
    # Initialize configurable parallel generator
    generator = ConfigurableParallelGenerator(
        config_override=config_override,
        environment=environment
    )
    
    # Load problems
    problems = generator.load_problems()
    
    # Filter problems if specified
    if problem_filter:
        problems = [p for p in problems if p['nature'] in problem_filter]
        if not problems:
            logger.error(f"No matching problems found for {problem_filter}")
            return
    
    logger.info(f"Processing {len(problems)} problems")
    
    # Track overall statistics
    overall_stats = {
        "total_problems": len(problems),
        "successful_problems": 0,
        "failed_problems": 0,
        "total_samples_generated": 0,
        "total_duplicates_filtered": 0,
        "total_generation_time": 0
    }
    
    overall_start_time = time.time()
    
    # Process each problem
    for i, problem in enumerate(problems):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing problem {i+1}/{len(problems)}: {problem['area']}/{problem['nature']}")
        logger.info(f"{'='*60}")
        
        try:
            # Reset generator stats for this problem
            generator.stats = {
                "total_generated": 0,
                "duplicates_filtered": 0,
                "failed_generations": 0,
                "successful_batches": 0,
                "total_api_calls": 0,
                "generation_start_time": None,
                "generation_end_time": None
            }
            
            # Generate samples
            new_samples = generator.generate_parallel(problem, malicious_count, benign_count)
            
            if new_samples:
                # Load existing samples for saving
                existing_samples = generator.load_existing_samples(problem)
                
                # Save samples
                generator.save_samples(problem, new_samples, existing_samples)
                
                # Update overall stats
                overall_stats["successful_problems"] += 1
                overall_stats["total_samples_generated"] += len(new_samples)
                overall_stats["total_duplicates_filtered"] += generator.stats["duplicates_filtered"]
                
                logger.info(f"Successfully generated {len(new_samples)} samples for {problem['nature']}")
            else:
                logger.warning(f"No samples generated for {problem['nature']}")
                overall_stats["failed_problems"] += 1
            
        except Exception as e:
            logger.error(f"Error processing {problem['nature']}: {e}", exc_info=True)
            overall_stats["failed_problems"] += 1
            continue
    
    # Calculate total time
    overall_stats["total_generation_time"] = time.time() - overall_start_time
    
    # Print final statistics
    logger.info(f"\n{'='*60}")
    logger.info("CONFIGURABLE PARALLEL GENERATION COMPLETED")
    logger.info(f"{'='*60}")
    logger.info(f"Problems processed: {overall_stats['successful_problems']}/{overall_stats['total_problems']}")
    logger.info(f"Total samples generated: {overall_stats['total_samples_generated']}")
    logger.info(f"Total duplicates filtered: {overall_stats['total_duplicates_filtered']}")
    logger.info(f"Total generation time: {overall_stats['total_generation_time']:.2f} seconds")
    
    if overall_stats['total_generation_time'] > 0:
        rate = overall_stats['total_samples_generated'] / (overall_stats['total_generation_time'] / 60)
        logger.info(f"Overall generation rate: {rate:.1f} samples/minute")
    
    logger.info(f"Environment used: {environment}")
    logger.info(f"{'='*60}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Configurable parallel synthetic data generation")
    parser.add_argument('--count', type=int, default=100, help='Total samples per problem (default: 100)')
    parser.add_argument('--malicious-ratio', type=float, default=0.5, help='Ratio of malicious samples (default: 0.5)')
    parser.add_argument('--problems', nargs='+', help='Specific problem natures to generate samples for')
    parser.add_argument('--environment', choices=['development', 'production', 'testing'], 
                       default='production', help='Environment configuration to use')
    parser.add_argument('--config-file', type=str, help='Path to custom configuration file')
    parser.add_argument('--max-workers', type=int, help='Override max workers')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--disable-duplicates', action='store_true', help='Disable duplicate detection')
    
    args = parser.parse_args()
    
    # Build configuration overrides
    config_override = {}
    
    if args.max_workers:
        config_override['parallel_generation'] = {
            'thread_based': {'max_workers': args.max_workers},
            'async_based': {'max_concurrent': args.max_workers}
        }
    
    if args.batch_size:
        if 'parallel_generation' not in config_override:
            config_override['parallel_generation'] = {}
        config_override['parallel_generation']['thread_based'] = config_override['parallel_generation'].get('thread_based', {})
        config_override['parallel_generation']['async_based'] = config_override['parallel_generation'].get('async_based', {})
        config_override['parallel_generation']['thread_based']['batch_size'] = args.batch_size
        config_override['parallel_generation']['async_based']['batch_size'] = args.batch_size
    
    if args.disable_duplicates:
        config_override['duplicate_detection'] = {'enabled': False}
    
    if args.config_file:
        try:
            import yaml
            with open(args.config_file, 'r') as f:
                file_config = yaml.safe_load(f)
            config_override.update(file_config)
            logger.info(f"Loaded custom configuration from {args.config_file}")
        except Exception as e:
            logger.error(f"Error loading config file {args.config_file}: {e}")
    
    # Run main function
    main(
        count=args.count,
        malicious_ratio=args.malicious_ratio,
        problem_filter=args.problems,
        config_override=config_override if config_override else None,
        environment=args.environment
    )