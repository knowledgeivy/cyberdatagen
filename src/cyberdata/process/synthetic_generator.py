# cyberdata/process/synthetic_generator.py

import concurrent.futures
import hashlib
import json
import logging
import os
import random
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv

# Add parent directory to path for imports
CURRENT_DIR = Path(__file__).parent
sys.path.append(str(CURRENT_DIR.parent))

from cyberdata.utils.config_manager import get_config_manager
from cyberdata.utils.llm_invoke import process_llm_request
from cyberdata.utils.logger_config import setup_logger
from cyberdata.utils.prompt_loader import load_prompt

# Set up logger
logger = setup_logger("cyberdata.scripts.generator")

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
    schema_info: Optional[Dict] = None
    is_realworld_based: bool = False
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

class DuplicateDetector:
    """Thread-safe duplicate detection using content hashing"""
    
    def __init__(self, config: Dict):
        dup_config = config.get('duplicate_detection', {})
        self.enabled = dup_config.get('enabled', True)
        self.check_fields = dup_config.get('check_fields', ['scenario', 'technical_data'])
        
        # Normalization settings
        norm_config = dup_config.get('normalization', {})
        self.normalize_lowercase = norm_config.get('lowercase', True)
        self.normalize_whitespace = norm_config.get('remove_extra_whitespace', True)
        self.normalize_values = norm_config.get('remove_specific_values', True)
        
        self.seen_hashes: Set[str] = set()
        self.lock = threading.Lock()
        
        logger.debug(f"Duplicate detector initialized: enabled={self.enabled}")
    
    def _normalize_content(self, content: str) -> str:
        """Normalize content based on configuration"""
        try:
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
            
            return content
        except Exception as e:
            logger.warning(f"Error normalizing content: {e}")
            return str(content)
    
    def _generate_content_hash(self, sample: Dict) -> str:
        """Generate a hash based on available fields (flexible for different schemas)"""
        if not self.enabled:
            return str(random.random())  # Always unique if disabled
        
        try:
            content_parts = []
            
            # Use configured fields if they exist, otherwise use all string fields
            fields_to_check = []
            for field in self.check_fields:
                if field in sample:
                    fields_to_check.append(field)
            
            # If configured fields don't exist, use any string fields
            if not fields_to_check:
                fields_to_check = [k for k, v in sample.items() if isinstance(v, str)]
            
            for field in fields_to_check:
                if field in sample:
                    content = str(sample[field])
                    normalized_content = self._normalize_content(content)
                    content_parts.append(normalized_content)
            
            combined_content = "|".join(content_parts)
            return hashlib.sha256(combined_content.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Error generating content hash: {e}")
            return str(random.random())
    
    def is_duplicate(self, sample: Dict) -> bool:
        """Check if sample is a duplicate (thread-safe)"""
        try:
            content_hash = self._generate_content_hash(sample)
            
            with self.lock:
                if content_hash in self.seen_hashes:
                    return True
                self.seen_hashes.add(content_hash)
                return False
        except Exception as e:
            logger.warning(f"Error checking duplicate: {e}")
            return False
    
    def add_existing_samples(self, samples: List[Dict]):
        """Add existing samples to the duplicate detector"""
        if not self.enabled:
            return
        
        try:
            with self.lock:
                for sample in samples:
                    content_hash = self._generate_content_hash(sample)
                    self.seen_hashes.add(content_hash)
            
            logger.debug(f"Added {len(samples)} existing samples to duplicate detector")
        except Exception as e:
            logger.warning(f"Error adding existing samples: {e}")
    
    def get_stats(self) -> Dict:
        """Get statistics about duplicate detection"""
        with self.lock:
            return {
                "enabled": self.enabled,
                "total_hashes": len(self.seen_hashes),
                "check_fields": self.check_fields
            }

class ParallelGenerator:
    """Unified parallel generator with schema preservation"""
    
    def __init__(self, config_override: Optional[Dict] = None):
        """Initialize with configuration from unified YAML"""
        self.config_manager = get_config_manager()
        
        # Load unified configuration
        self.config = self._load_configuration(config_override)
        
        # Initialize components
        self.duplicate_detector = DuplicateDetector(self.config)
        self.stats = {
            "total_generated": 0,
            "duplicates_filtered": 0,
            "failed_generations": 0,
            "successful_batches": 0,
            "total_api_calls": 0,
            "generation_start_time": None,
            "generation_end_time": None
        }
        
        # Get generation settings from unified config
        gen_config = self.config.get("generation", {})
        self.max_workers = gen_config.get("max_workers", 4)
        self.default_batch_size = gen_config.get("batch_size", 2)
        self.retry_attempts = gen_config.get("retry_attempts", 3)
        self.retry_delay = gen_config.get("retry_delay", 2)
        
        # Get default values
        self.default_count = gen_config.get("default_count", 20)
        self.default_malicious_ratio = gen_config.get("default_malicious_ratio", 0.5)
        
        logger.info(f"Generator initialized: {self.max_workers} workers, "
                   f"batch size {self.default_batch_size}, defaults: {self.default_count} samples, "
                   f"{self.default_malicious_ratio:.1f} malicious ratio")
    
    def _load_configuration(self, config_override: Optional[Dict] = None) -> Dict:
        """Load configuration from unified YAML"""
        try:
            # Load the unified generation config
            config_file = self.config_manager.prompts_dir / "generation_config.yaml"
            
            if not config_file.exists():
                logger.warning(f"Generation config not found: {config_file}, using defaults")
                return self._get_default_config()
            
            from cyberdata.utils.prompt_loader import get_prompt_loader
            loader = get_prompt_loader()
            config = loader.load_prompt_file("generation_config")
            
            # Apply any runtime overrides
            if config_override:
                logger.info("Applying runtime configuration overrides")
                config = self._merge_configs(config, config_override)
            
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Return default configuration"""
        return {
            'generation': {
                'default_count': 20,
                'default_malicious_ratio': 0.5,
                'max_workers': 4,
                'batch_size': 2,
                'retry_attempts': 3,
                'retry_delay': 2,
                'temperature': {
                    'malicious_generation': 0.7,
                    'benign_generation': 0.7,
                    'validation': 0.0
                }
            },
            'duplicate_detection': {
                'enabled': True,
                'check_fields': ['scenario', 'technical_data']
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
    
    def load_problems(self) -> List[Dict]:
        """Load problem definitions"""
        try:
            return self.config_manager.load_problems()
        except Exception as e:
            logger.error(f"Error loading problems: {e}")
            return []
    
    def load_existing_samples(self, problem: Dict) -> List[Dict]:
        """Load existing samples and add them to duplicate detector"""
        try:
            samples_file = self.find_samples_file(problem)
            if samples_file and samples_file.exists():
                with open(samples_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
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
        try:
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
        except Exception as e:
            logger.warning(f"Error finding samples file: {e}")
            return None
    
    def load_examples(self, problem: Dict) -> Tuple[List[Dict], Dict]:
        """Load seed examples and determine if they're from real-world data"""
        try:
            area = problem['area']
            nature = problem['nature']
            
            expected_file = self.config_manager.get_seeds_file(area, nature)
            if expected_file.exists():
                with open(expected_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                examples = data.get('examples', [])
                metadata = data.get('metadata', {})
                
                # Check if this is real-world derived data
                is_realworld = metadata.get('source') == 'real_world_data'
                schema_info = metadata.get('generation_metadata', {}) if is_realworld else {}
                
                logger.info(f"Loaded {len(examples)} examples for {nature} (real-world: {is_realworld})")
                
                # For real-world data, preserve schema exactly
                if is_realworld and examples:
                    # Clean examples to remove any metadata fields added during processing
                    cleaned_examples = []
                    for example in examples[:3]:  # Limit to avoid token issues
                        if '_metadata' in example:
                            # Remove metadata but keep original data structure
                            cleaned_example = {k: v for k, v in example.items() if k != '_metadata'}
                            cleaned_examples.append(cleaned_example)
                        else:
                            cleaned_examples.append(example)
                    
                    schema_info['original_schema'] = {
                        'columns': list(cleaned_examples[0].keys()) if cleaned_examples else [],
                        'sample_record': cleaned_examples[0] if cleaned_examples else {}
                    }
                    
                    return cleaned_examples, schema_info
                else:
                    # Traditional seed examples (limit to avoid token issues)
                    return examples[:2], {}
                    
        except Exception as e:
            logger.warning(f"Could not load examples for {problem['nature']}: {e}")
        
        return [], {}
    
    def create_generation_tasks(self, problem: Dict, malicious_count: int, benign_count: int) -> List[GenerationTask]:
        """Create generation tasks with schema preservation awareness"""
        tasks = []
        task_counter = 0
        
        try:
            # Load examples and schema info
            examples, schema_info = self.load_examples(problem)
            is_realworld_based = bool(schema_info)
            
            # Create malicious generation tasks
            if malicious_count > 0:
                remaining_malicious = malicious_count
                
                while remaining_malicious > 0:
                    current_batch = min(self.default_batch_size, remaining_malicious)
                    task = GenerationTask(
                        problem=problem,
                        sample_type='malicious',
                        batch_size=current_batch,
                        task_id=f"{problem['nature']}_mal_{task_counter}",
                        examples=examples,
                        schema_info=schema_info,
                        is_realworld_based=is_realworld_based,
                        priority=2  # Higher priority for malicious
                    )
                    tasks.append(task)
                    remaining_malicious -= current_batch
                    task_counter += 1
            
            # Create benign generation tasks
            if benign_count > 0:
                remaining_benign = benign_count
                
                while remaining_benign > 0:
                    current_batch = min(self.default_batch_size, remaining_benign)
                    task = GenerationTask(
                        problem=problem,
                        sample_type='benign',
                        batch_size=current_batch,
                        task_id=f"{problem['nature']}_ben_{task_counter}",
                        examples=examples,  # Use same examples for schema reference
                        schema_info=schema_info,
                        is_realworld_based=is_realworld_based,
                        priority=1
                    )
                    tasks.append(task)
                    remaining_benign -= current_batch
                    task_counter += 1
            
            # Sort tasks by priority
            tasks.sort(key=lambda x: x.priority, reverse=True)
            
            logger.info(f"Created {len(tasks)} generation tasks for {problem['nature']} (real-world: {is_realworld_based})")
            return tasks
            
        except Exception as e:
            logger.error(f"Error creating generation tasks: {e}")
            return []
    
    def execute_generation_task(self, task: GenerationTask) -> GenerationResult:
        """Execute a single generation task with retry logic"""
        start_time = time.time()
        
        for attempt in range(self.retry_attempts):
            try:
                logger.info(f"Executing task {task.task_id} (attempt {attempt + 1}): "
                           f"{task.batch_size} {task.sample_type} samples (schema-aware: {task.is_realworld_based})")
                
                if task.sample_type == 'malicious':
                    samples = self._generate_malicious_batch(task)
                else:
                    samples = self._generate_benign_batch(task)
                
                # Validate samples
                if not isinstance(samples, list):
                    logger.warning(f"Invalid samples format from task {task.task_id}")
                    samples = []
                
                # Filter duplicates and add minimal metadata
                unique_samples = []
                duplicates_count = 0
                
                for sample in samples:
                    if not isinstance(sample, dict):
                        logger.warning(f"Invalid sample format: {type(sample)}")
                        continue
                        
                    if not self.duplicate_detector.is_duplicate(sample):
                        # Only add minimal metadata if not present in original schema
                        if not task.is_realworld_based:
                            # Traditional approach - add metadata
                            sample['sample_type'] = task.sample_type
                            sample['is_attack'] = task.sample_type == 'malicious'
                            sample['generation_task_id'] = task.task_id
                            sample['generation_timestamp'] = time.time()
                        # For real-world based data, preserve original schema exactly
                        
                        unique_samples.append(sample)
                    else:
                        duplicates_count += 1
                
                # Update stats (thread-safe)
                with threading.Lock():
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
                    with threading.Lock():
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
                    time.sleep(self.retry_delay * (attempt + 1))
        
        # Should never reach here, but just in case
        return GenerationResult(
            task_id=task.task_id,
            samples=[],
            sample_type=task.sample_type,
            problem_nature=task.problem['nature'],
            success=False,
            error="Max retries exceeded",
            generation_time=time.time() - start_time
        )
    
    def _generate_malicious_batch(self, task: GenerationTask) -> List[Dict]:
        """Generate a batch of malicious samples with schema preservation"""
        try:
            gen_config = self.config.get('generation', {})
            temperature_config = gen_config.get('temperature', {})
            temperature = temperature_config.get('malicious_generation', 0.7)
            
            if task.is_realworld_based and task.schema_info:
                # Use schema-preserving generation for real-world data
                return self._generate_schema_preserving_batch(task, temperature)
            else:
                # Use traditional generation for manual problem definitions
                return self._generate_traditional_malicious_batch(task, temperature)
                
        except Exception as e:
            logger.error(f"Error generating malicious batch: {e}")
            return []
    
    def _generate_schema_preserving_batch(self, task: GenerationTask, temperature: float) -> List[Dict]:
        """Generate samples that preserve the original data schema exactly"""
        try:
            examples_json = json.dumps(task.examples, indent=2) if task.examples else "[]"
            
            # Extract schema information
            original_schema = task.schema_info.get('original_schema', {})
            column_names = original_schema.get('columns', [])
            
            # Determine label mapping from examples
            label_mapping = {"malicious": 1, "benign": 0}  # Default
            if task.examples:
                # Try to infer label mapping from examples
                for example in task.examples:
                    for key, value in example.items():
                        if 'label' in key.lower() and isinstance(value, (int, bool)):
                            # Assume 1/True = malicious, 0/False = benign
                            break
            
            system_content = load_prompt(
                "generation_config",
                "prompts.schema_preserving_generation.system.template"
            )
            
            user_content = load_prompt(
                "generation_config",
                "prompts.schema_preserving_generation.user.template",
                schema_info=json.dumps(task.schema_info, indent=2),
                examples_json=examples_json,
                count=task.batch_size,
                column_names=json.dumps(column_names),
                label_mapping=json.dumps(label_mapping)
            )
            
            response_content = process_llm_request(
                system_prompt=system_content,
                user_prompt=user_content,
                model_name="gpt-4.1-mini",
                temperature=temperature
            )
            
            parsed = self._extract_json_from_response(response_content)
            samples = parsed.get('samples', [])
            
            # Validate that samples maintain the original schema
            validated_samples = []
            for sample in samples:
                if self._validate_schema_compliance(sample, column_names):
                    validated_samples.append(sample)
                else:
                    logger.warning(f"Sample failed schema validation: {list(sample.keys())}")
            
            return validated_samples
            
        except Exception as e:
            logger.error(f"Error in schema-preserving generation: {e}")
            return []
    
    def _generate_traditional_malicious_batch(self, task: GenerationTask, temperature: float) -> List[Dict]:
        """Generate malicious samples using traditional approach for manual problem definitions"""
        try:
            examples_json = json.dumps(task.examples, indent=2) if task.examples else "[]"
            
            if task.examples:
                # Use generation prompt with seed examples
                system_content = load_prompt(
                    "generation_config",
                    "prompts.generation.system.template"
                )
                
                user_content = load_prompt(
                    "generation_config",
                    "prompts.generation.user.template",
                    examples_json=examples_json,
                    count=task.batch_size
                )
            else:
                # Use backup generation prompt
                system_content = load_prompt(
                    "generation_config",
                    "prompts.backup_generation.system.template",
                    nature=task.problem['nature'],
                    area=task.problem['area']
                )
                
                user_content = load_prompt(
                    "generation_config",
                    "prompts.backup_generation.user.template",
                    count=task.batch_size,
                    nature=task.problem['nature'],
                    description=task.problem.get('description', '')
                )
            
            response_content = process_llm_request(
                system_prompt=system_content,
                user_prompt=user_content,
                model_name="gpt-4.1-mini",
                temperature=temperature
            )
            
            parsed = self._extract_json_from_response(response_content)
            return parsed.get('samples', [])
            
        except Exception as e:
            logger.error(f"Error in traditional malicious generation: {e}")
            return []
    
    def _generate_benign_batch(self, task: GenerationTask) -> List[Dict]:
        """Generate a batch of benign samples with schema preservation"""
        try:
            gen_config = self.config.get('generation', {})
            temperature_config = gen_config.get('temperature', {})
            temperature = temperature_config.get('benign_generation', 0.7)
            
            if task.is_realworld_based and task.examples:
                # Use schema-preserving benign generation for real-world data
                return self._generate_schema_preserving_benign_batch(task, temperature)
            else:
                # Use traditional benign generation
                return self._generate_traditional_benign_batch(task, temperature)
                
        except Exception as e:
            logger.error(f"Error generating benign batch: {e}")
            return []
    
    def _generate_schema_preserving_benign_batch(self, task: GenerationTask, temperature: float) -> List[Dict]:
        """Generate benign samples that preserve the original data schema"""
        try:
            # Use the same schema as the examples
            schema_reference = json.dumps(task.examples[0], indent=2) if task.examples else "{}"
            
            system_content = load_prompt(
                "generation_config",
                "prompts.benign_generation.system.template"
            )
            
            user_content = load_prompt(
                "generation_config",
                "prompts.benign_generation.user.template",
                count=task.batch_size,
                schema_reference=schema_reference
            )
            
            response_content = process_llm_request(
                system_prompt=system_content,
                user_prompt=user_content,
                model_name="gpt-4.1-mini",
                temperature=temperature
            )
            
            parsed = self._extract_json_from_response(response_content)
            samples = parsed.get('samples', [])
            
            # Validate schema compliance
            if task.examples:
                expected_columns = list(task.examples[0].keys())
                validated_samples = []
                for sample in samples:
                    if self._validate_schema_compliance(sample, expected_columns):
                        validated_samples.append(sample)
                    else:
                        logger.warning(f"Benign sample failed schema validation")
                return validated_samples
            
            return samples
            
        except Exception as e:
            logger.error(f"Error in schema-preserving benign generation: {e}")
            return []
    
    def _generate_traditional_benign_batch(self, task: GenerationTask, temperature: float) -> List[Dict]:
        """Generate benign samples using traditional approach"""
        try:
            system_content = load_prompt(
                "generation_config",
                "prompts.benign_generation.system.template"
            )
            
            user_content = load_prompt(
                "generation_config",
                "prompts.benign_generation.user.template",
                area=task.problem['area'],
                count=task.batch_size,
                schema_reference=json.dumps(task.examples[0] if task.examples else {}, indent=2)
            )
            
            response_content = process_llm_request(
                system_prompt=system_content,
                user_prompt=user_content,
                model_name="gpt-4.1-mini",
                temperature=temperature
            )
            
            parsed = self._extract_json_from_response(response_content)
            return parsed.get('samples', [])
            
        except Exception as e:
            logger.error(f"Error in traditional benign generation: {e}")
            return []
    
    def _validate_schema_compliance(self, sample: Dict, expected_columns: List[str]) -> bool:
        """Validate that a sample complies with the expected schema"""
        try:
            if not expected_columns:
                return True  # No schema to validate against
            
            sample_columns = set(sample.keys())
            expected_columns_set = set(expected_columns)
            
            # Check if sample has all expected columns
            missing_columns = expected_columns_set - sample_columns
            extra_columns = sample_columns - expected_columns_set
            
            if missing_columns:
                logger.debug(f"Sample missing columns: {missing_columns}")
                return False
            
            if extra_columns:
                logger.debug(f"Sample has extra columns: {extra_columns}")
                # Extra columns might be okay, but log for awareness
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating schema compliance: {e}")
            return False
    
    def _extract_json_from_response(self, response_content: str) -> Dict:
        """Extract JSON from LLM response with robust error handling"""
        if not response_content:
            logger.warning("Empty response content")
            return {"samples": []}
        
        try:
            # Clean markdown code blocks
            if '```' in response_content:
                pattern = r'```(?:json)?\s*([\s\S]*?)```'
                matches = re.findall(pattern, response_content)
                if matches:
                    response_content = matches[0].strip()
            
            # Try direct parsing
            try:
                result = json.loads(response_content)
                if isinstance(result, dict):
                    return result
                elif isinstance(result, list):
                    return {"samples": result}
                else:
                    logger.warning(f"Unexpected JSON type: {type(result)}")
                    return {"samples": []}
            except json.JSONDecodeError:
                # Try fixing common JSON issues
                fixed_content = response_content
                
                # Fix single quotes
                fixed_content = fixed_content.replace("'", '"')
                
                # Fix missing commas between objects
                fixed_content = re.sub(r'}\s*{', '},{', fixed_content)
                
                # Fix trailing commas
                fixed_content = re.sub(r',\s*}', '}', fixed_content)
                fixed_content = re.sub(r',\s*]', ']', fixed_content)
                
                # Fix unescaped quotes in strings
                fixed_content = re.sub(r'(?<!\\)"(?=\w)', '\\"', fixed_content)
                
                result = json.loads(fixed_content)
                if isinstance(result, dict):
                    return result
                elif isinstance(result, list):
                    return {"samples": result}
                else:
                    return {"samples": []}
                    
        except Exception as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response_content[:500]}...")
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
        
        # Execute tasks in parallel using ThreadPoolExecutor
        all_samples = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.execute_generation_task, task): task 
                for task in tasks
            }
            
            # Process completed tasks
            for future in concurrent.futures.as_completed(future_to_task, timeout=300):  # 5 minute timeout
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
        
        try:
            area = problem['area']
            nature = problem['nature']
            
            out_file = self.config_manager.get_large_samples_file(area, nature)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Combine existing and new samples
            all_samples = existing_samples + new_samples
            
            # Calculate distributions - check for different labeling schemes
            malicious_new = self._count_malicious_samples(new_samples)
            benign_new = len(new_samples) - malicious_new
            
            # Enhanced metadata
            generation_time = (self.stats.get("generation_end_time", 0) - 
                              self.stats.get("generation_start_time", 0))
            
            # Check if this is schema-preserving generation
            schema_preserved = self._check_schema_preservation(new_samples, existing_samples)
            
            metadata = {
                'total_samples': len(all_samples),
                'new_samples_added': len(new_samples),
                'new_malicious_samples': malicious_new,
                'new_benign_samples': benign_new,
                'generation_method': 'parallel_schema_preserving' if schema_preserved else 'parallel',
                'schema_preserved': schema_preserved,
                'generation_stats': self.stats.copy(),
                'duplicate_detection_stats': self.duplicate_detector.get_stats(),
                'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_generation_time': generation_time,
                'samples_per_minute': len(new_samples) / (generation_time / 60) if generation_time > 0 else 0,
                'max_workers': self.max_workers,
                'batch_size': self.default_batch_size
            }
            
            output_data = {
                'samples': all_samples,
                'metadata': metadata
            }
            
            with out_file.open('w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"Saved {len(new_samples)} new samples to {out_file}")
            logger.info(f"Total samples in file: {len(all_samples)} (schema preserved: {schema_preserved})")
            logger.info(f"Generation rate: {metadata['samples_per_minute']:.1f} samples/minute")
            
        except Exception as e:
            logger.error(f"Error saving samples: {e}")
    
    def _count_malicious_samples(self, samples: List[Dict]) -> int:
        """Count malicious samples using various labeling schemes"""
        malicious_count = 0
        
        for sample in samples:
            # Check sample_type field (traditional approach)
            if sample.get('sample_type') == 'malicious':
                malicious_count += 1
                continue
            
            # Check is_attack field
            if sample.get('is_attack') is True:
                malicious_count += 1
                continue
            
            # Check label fields (common in real-world data)
            for key in ['label', 'Label', 'class', 'Class', 'target']:
                if key in sample:
                    value = sample[key]
                    # Assume 1 = malicious, 0 = benign for numeric labels
                    if isinstance(value, (int, bool)) and value == 1:
                        malicious_count += 1
                        break
                    # Check for string labels
                    elif isinstance(value, str) and value.lower() in ['malicious', 'attack', 'bad', 'positive']:
                        malicious_count += 1
                        break
        
        return malicious_count
    
    def _check_schema_preservation(self, new_samples: List[Dict], existing_samples: List[Dict]) -> bool:
        """Check if new samples preserve the schema of existing samples"""
        if not new_samples or not existing_samples:
            return False
        
        try:
            # Get schema from existing samples
            existing_schema = set(existing_samples[0].keys()) if existing_samples else set()
            
            # Check if new samples follow the same schema
            for sample in new_samples[:5]:  # Check first 5 samples
                sample_schema = set(sample.keys())
                
                # Allow some flexibility - new samples shouldn't have traditional metadata
                # if they're preserving real-world schema
                traditional_metadata = {'sample_type', 'is_attack', 'generation_task_id', 'generation_timestamp'}
                sample_schema_clean = sample_schema - traditional_metadata
                existing_schema_clean = existing_schema - traditional_metadata
                
                if sample_schema_clean == existing_schema_clean:
                    return True
                
                # If schemas don't match exactly, check if it's close
                if len(sample_schema_clean.symmetric_difference(existing_schema_clean)) <= 2:
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking schema preservation: {e}")
            return False

def main(count: int = None, malicious_ratio: float = None, problem_filter: List[str] = None, 
         config_override: Dict = None):
    """Main function for unified parallel generation"""
    
    try:
        # Initialize generator first to get default values
        generator = ParallelGenerator(config_override=config_override)
        
        # Use defaults from config if not specified
        if count is None:
            count = generator.default_count
        if malicious_ratio is None:
            malicious_ratio = generator.default_malicious_ratio
        
        logger.info(f"Starting generation: count={count}, malicious_ratio={malicious_ratio}")
        
        # Validate inputs
        if not 0.0 <= malicious_ratio <= 1.0:
            logger.error(f"Invalid malicious ratio: {malicious_ratio}")
            return
        
        # Calculate sample counts
        malicious_count = int(count * malicious_ratio)
        benign_count = count - malicious_count
        
        logger.info(f"Target distribution: {malicious_count} malicious, {benign_count} benign")
        
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
            "total_generation_time": 0,
            "schema_preserved_problems": 0
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
                    
                    # Check if schema was preserved
                    if generator._check_schema_preservation(new_samples, existing_samples):
                        overall_stats["schema_preserved_problems"] += 1
                    
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
        logger.info("GENERATION COMPLETED")
        logger.info(f"{'='*60}")
        logger.info(f"Problems processed: {overall_stats['successful_problems']}/{overall_stats['total_problems']}")
        logger.info(f"Schema preserved: {overall_stats['schema_preserved_problems']}/{overall_stats['successful_problems']}")
        logger.info(f"Total samples generated: {overall_stats['total_samples_generated']}")
        logger.info(f"Total duplicates filtered: {overall_stats['total_duplicates_filtered']}")
        logger.info(f"Total generation time: {overall_stats['total_generation_time']:.2f} seconds")
        
        if overall_stats['total_generation_time'] > 0:
            rate = overall_stats['total_samples_generated'] / (overall_stats['total_generation_time'] / 60)
            logger.info(f"Overall generation rate: {rate:.1f} samples/minute")
        
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified synthetic data generation with schema preservation")
    parser.add_argument('--count', type=int, help='Total samples per problem (default from config: 20)')
    parser.add_argument('--malicious-ratio', type=float, help='Ratio of malicious samples (default from config: 0.5)')
    parser.add_argument('--problems', nargs='+', help='Specific problem natures to generate samples for')
    parser.add_argument('--max-workers', type=int, help='Override max workers (default from config: 4)')
    parser.add_argument('--batch-size', type=int, help='Override batch size (default from config: 2)')
    parser.add_argument('--disable-duplicates', action='store_true', help='Disable duplicate detection')
    
    args = parser.parse_args()
    
    # Build configuration overrides
    config_override = {}
    
    if args.max_workers:
        config_override['generation'] = config_override.get('generation', {})
        config_override['generation']['max_workers'] = args.max_workers
    
    if args.batch_size:
        config_override['generation'] = config_override.get('generation', {})
        config_override['generation']['batch_size'] = args.batch_size
    
    if args.disable_duplicates:
        config_override['duplicate_detection'] = {'enabled': False}
    
    # Run main function
    main(
        count=args.count,
        malicious_ratio=args.malicious_ratio,
        problem_filter=args.problems,
        config_override=config_override if config_override else None
    )