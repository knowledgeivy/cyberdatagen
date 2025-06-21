# cyberdata/process/seed_validator.py

import concurrent.futures
import json
import numpy as np
import os
import sys
import time
import threading
import yaml
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from dotenv import load_dotenv

# Add parent directory to path for imports
CURRENT_DIR = Path(__file__).parent
sys.path.append(str(CURRENT_DIR.parent))

from cyberdata.utils.config_manager import get_config_manager
from cyberdata.utils.llm_invoke import process_llm_request
from cyberdata.utils.logger_config import setup_logger
from cyberdata.utils.prompt_loader import load_prompt

# Set up logger
logger = setup_logger("cyberdata.scripts.enhanced_parallel_seed_validator")

# Load environment variables
load_dotenv()

# Constants
MODEL_NAME = "gpt-4.1-mini"

# Get config manager instance
config_manager = get_config_manager()

logger.info(f"Using model: {MODEL_NAME}")


def load_validation_config() -> Dict[str, Any]:
    """Load seed validation configuration from scale_config.yaml."""
    try:
        config_file = config_manager.config_dir / "scale_config.yaml"
        
        if not config_file.exists():
            logger.warning(f"Scale config file not found: {config_file}. Using defaults.")
            return get_default_seed_config()
        
        with config_file.open('r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded seed validation configuration from: {config_file}")
        return config
        
    except Exception as e:
        logger.error(f"Error loading scale configuration: {e}. Using defaults.")
        return get_default_seed_config()


def get_default_seed_config() -> Dict[str, Any]:
    """Get default configuration if scale_config.yaml is not available."""
    return {
        'seed_validation': {
            'parallel_processing': {
                'max_workers': 6,
                'batch_delay': 0.15,
                'enable_parallel': True
            },
            'quality_thresholds': {
                'technical_accuracy': 0.7,
                'schema_consistency': 0.9,
                'realism_assessment': 0.7,
                'semantic_uniqueness': 0.6,
                'domain_alignment': 0.7,
                'overall_minimum': 0.75
            },
            'performance_optimization': {
                'enable_progress_logging': True,
                'log_every_n_samples': 25,
                'enable_performance_metrics': True
            }
        }
    }


@dataclass
class QualityScore:
    """Multi-dimensional quality score for seed validation."""
    technical_accuracy: float
    schema_consistency: float
    realism_assessment: float
    semantic_uniqueness: float
    domain_alignment: float
    composite_score: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'technical_accuracy': self.technical_accuracy,
            'schema_consistency': self.schema_consistency,
            'realism_assessment': self.realism_assessment,
            'semantic_uniqueness': self.semantic_uniqueness,
            'domain_alignment': self.domain_alignment,
            'composite_score': self.composite_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'QualityScore':
        return cls(
            technical_accuracy=data.get('technical_accuracy', 0.0),
            schema_consistency=data.get('schema_consistency', 0.0),
            realism_assessment=data.get('realism_assessment', 0.0),
            semantic_uniqueness=data.get('semantic_uniqueness', 0.0),
            domain_alignment=data.get('domain_alignment', 0.0),
            composite_score=data.get('composite_score', 0.0)
        )
    
    def is_high_quality(self, thresholds: Dict[str, float]) -> bool:
        """Check if this score meets high quality thresholds."""
        return (
            self.technical_accuracy >= thresholds['technical_accuracy'] and
            self.schema_consistency >= thresholds['schema_consistency'] and
            self.realism_assessment >= thresholds['realism_assessment'] and
            self.semantic_uniqueness >= thresholds['semantic_uniqueness'] and
            self.domain_alignment >= thresholds['domain_alignment'] and
            self.composite_score >= thresholds['overall_minimum']
        )


@dataclass
class ValidationResult:
    """Complete validation result for a seed example."""
    example_index: int
    example: Dict[str, Any]
    quality_score: QualityScore
    is_high_quality: bool
    validation_details: Dict[str, Any]
    sample_type: str
    issues: List[str]
    strengths: List[str]
    recommendations: List[str]


@dataclass
class ValidationTask:
    """Represents a single validation task for parallel processing."""
    task_id: str
    example_index: int
    example: Dict[str, Any]
    validation_context: Dict[str, Any]
    uniqueness_score: float
    priority: int = 0


@dataclass
class ValidationTaskResult:
    """Result of a validation task."""
    task_id: str
    validation_result: ValidationResult
    success: bool
    processing_time: float = 0.0
    error: Optional[str] = None


class EnhancedParallelSeedValidator:
    """Enhanced seed validator with parallel multi-dimensional quality assessment."""
    
    def __init__(self, config_override: Dict[str, Any] = None):
        self.config_manager = config_manager
        
        # Load configuration
        self.config = load_validation_config()
        if config_override:
            self.config = self._merge_configs(self.config, config_override)
        
        # Extract validation configuration (look for seed_validation or fall back to scale_validation)
        validation_config = self.config.get('seed_validation', self.config.get('scale_validation', {}))
        
        # Parallel processing settings
        parallel_config = validation_config.get('parallel_processing', {})
        self.max_workers = parallel_config.get('max_workers', 6)
        self.batch_delay = parallel_config.get('batch_delay', 0.15)
        self.enable_parallel = parallel_config.get('enable_parallel', True)
        
        # Quality thresholds
        self.quality_thresholds = validation_config.get('quality_thresholds', {
            'technical_accuracy': 0.7,
            'schema_consistency': 0.9,
            'realism_assessment': 0.7,
            'semantic_uniqueness': 0.6,
            'domain_alignment': 0.7,
            'overall_minimum': 0.75
        })
        
        # Performance settings
        perf_config = validation_config.get('performance_optimization', {})
        self.enable_progress_logging = perf_config.get('enable_progress_logging', True)
        self.log_interval = perf_config.get('log_every_n_samples', 25)
        
        # Statistics tracking
        self.stats = {
            'total_validated': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'high_quality_count': 0,
            'validation_start_time': None,
            'validation_end_time': None,
            'total_api_calls': 0
        }
        
        # Thread lock for stats
        self.stats_lock = threading.Lock()
        
        logger.info(f"EnhancedParallelSeedValidator initialized from config:")
        logger.info(f"  - Parallel workers: {self.max_workers}")
        logger.info(f"  - Batch delay: {self.batch_delay}s")
        logger.info(f"  - Parallel enabled: {self.enable_parallel}")
        logger.info(f"  - Quality thresholds: {self.quality_thresholds}")
    
    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """Recursively merge configuration dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def load_domain_discovery(self, dataset_name: str) -> Dict[str, Any]:
        """Load domain discovery results for validation context."""
        try:
            domain_file = self.config_manager.config_dir / "domain_discovery" / f"{dataset_name}_domain_discovery.json"
            
            if not domain_file.exists():
                logger.warning(f"Domain discovery file not found: {domain_file}")
                return {}
            
            with domain_file.open('r', encoding='utf-8') as f:
                domain_discovery = json.load(f)
            
            logger.info(f"Loaded domain discovery for validation: {dataset_name}")
            return domain_discovery
            
        except Exception as e:
            logger.warning(f"Error loading domain discovery: {str(e)}")
            return {}
    
    def load_contextual_problems(self, dataset_name: str) -> Dict[str, Any]:
        """Load contextual problems for validation context."""
        try:
            contextual_file = self.config_manager.config_dir / "contextual_problems" / f"{dataset_name}_contextual_problems.json"
            
            if not contextual_file.exists():
                logger.warning(f"Contextual problems file not found: {contextual_file}")
                return {}
            
            with contextual_file.open('r', encoding='utf-8') as f:
                contextual_problems = json.load(f)
            
            logger.info(f"Loaded contextual problems for validation: {dataset_name}")
            return contextual_problems
            
        except Exception as e:
            logger.warning(f"Error loading contextual problems: {str(e)}")
            return {}
    
    def load_data_info(self, dataset_name: str) -> Dict[str, Any]:
        """Load data info schema for validation."""
        try:
            data_info_file = self.config_manager.config_dir / "data_info.yaml"
            
            if not data_info_file.exists():
                logger.warning(f"Data info file not found: {data_info_file}")
                return {}
            
            import yaml
            with data_info_file.open('r', encoding='utf-8') as f:
                data_info = yaml.safe_load(f)
            
            datasets = data_info.get('datasets', {})
            if dataset_name in datasets:
                logger.info(f"Loaded data info schema for validation: {dataset_name}")
                return datasets[dataset_name]
            else:
                logger.warning(f"Dataset {dataset_name} not found in data_info.yaml")
                return {}
                
        except Exception as e:
            logger.warning(f"Error loading data info: {str(e)}")
            return {}
    
    def load_raw_seeds(self, area: str, nature: str) -> Tuple[List[Dict], Dict[str, Any]]:
        """Load raw seeds for validation."""
        try:
            # Look in seeds-raw directory
            seeds_raw_dir = self.config_manager.data_dir / "seeds-raw"
            area_clean = area.replace(' ', '_').replace('(', '').replace(')', '')
            seeds_file = seeds_raw_dir / area_clean / f"{nature}_examples.json"
            
            if not seeds_file.exists():
                # Fallback to regular seeds directory
                seeds_file = self.config_manager.get_seeds_file(area, nature)
            
            if not seeds_file.exists():
                raise FileNotFoundError(f"Seeds file not found: {seeds_file}")
            
            with seeds_file.open('r', encoding='utf-8') as f:
                data = json.load(f)
            
            examples = data.get('examples', [])
            metadata = data.get('metadata', {})
            
            logger.info(f"Loaded {len(examples)} raw seeds from: {seeds_file}")
            return examples, metadata
            
        except Exception as e:
            logger.error(f"Error loading raw seeds: {str(e)}")
            raise
    
    def calculate_semantic_uniqueness_batch(self, examples: List[Dict[str, Any]]) -> Dict[int, float]:
        """Calculate semantic uniqueness scores for examples."""
        logger.info("Calculating semantic uniqueness scores...")
        
        uniqueness_scores = {}
        
        # Create text representations
        text_representations = {}
        for i, example in enumerate(examples):
            text_parts = []
            for key, value in example.items():
                if key not in ['_metadata', 'sample_id', 'generation_timestamp']:
                    text_parts.append(str(value).lower())
            text_representations[i] = ' '.join(text_parts)
        
        # Calculate uniqueness scores
        for i, text_repr in text_representations.items():
            words_current = set(text_repr.split())
            similar_count = 0
            
            for other_idx, other_repr in text_representations.items():
                if i != other_idx:
                    words_other = set(other_repr.split())
                    
                    if len(words_current) > 0 and len(words_other) > 0:
                        intersection = len(words_current.intersection(words_other))
                        union = len(words_current.union(words_other))
                        jaccard_similarity = intersection / union if union > 0 else 0
                        
                        if jaccard_similarity > 0.7:  # High similarity threshold
                            similar_count += 1
            
            # Calculate uniqueness score
            total_comparisons = len(text_representations) - 1
            uniqueness_score = 1.0 - (similar_count / total_comparisons) if total_comparisons > 0 else 1.0
            uniqueness_scores[i] = max(0.0, min(1.0, uniqueness_score))
        
        logger.info(f"Calculated uniqueness scores for {len(examples)} examples")
        return uniqueness_scores
    
    def create_validation_tasks(self, examples: List[Dict[str, Any]], 
                               validation_context: Dict[str, Any], 
                               uniqueness_scores: Dict[int, float]) -> List[ValidationTask]:
        """Create validation tasks for parallel processing."""
        tasks = []
        
        for i, example in enumerate(examples):
            task = ValidationTask(
                task_id=f"seed_val_{i}",
                example_index=i,
                example=example,
                validation_context=validation_context.copy(),  # Each task gets its own copy
                uniqueness_score=uniqueness_scores.get(i, 0.5),
                priority=1  # All seed validation tasks have same priority
            )
            tasks.append(task)
        
        logger.info(f"Created {len(tasks)} validation tasks")
        return tasks
    
    def execute_validation_task(self, task: ValidationTask) -> ValidationTaskResult:
        """Execute a single validation task."""
        start_time = time.time()
        
        logger.debug(f"Executing validation task {task.task_id} (example {task.example_index})")
        
        try:
            # Validate the example
            result = self.validate_individual_example(
                task.example, 
                task.example_index, 
                task.validation_context, 
                task.uniqueness_score
            )
            
            # Update stats
            with self.stats_lock:
                self.stats['successful_validations'] += 1
                self.stats['total_api_calls'] += 1
                if result.is_high_quality:
                    self.stats['high_quality_count'] += 1
            
            processing_time = time.time() - start_time
            
            logger.debug(f"Task {task.task_id} completed successfully in {processing_time:.2f}s")
            
            return ValidationTaskResult(
                task_id=task.task_id,
                validation_result=result,
                success=True,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            
            with self.stats_lock:
                self.stats['failed_validations'] += 1
            
            # Create a failed validation result
            failed_result = ValidationResult(
                example_index=task.example_index,
                example=task.example,
                quality_score=QualityScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                is_high_quality=False,
                validation_details={'error': str(e)},
                sample_type='unknown',
                issues=[f'Validation error: {str(e)}'],
                strengths=[],
                recommendations=['Review example format and validation process']
            )
            
            return ValidationTaskResult(
                task_id=task.task_id,
                validation_result=failed_result,
                success=False,
                processing_time=time.time() - start_time,
                error=str(e)
            )
    
    def validate_individual_example(self, 
                                   example: Dict[str, Any],
                                   example_index: int,
                                   validation_context: Dict[str, Any],
                                   uniqueness_score: float) -> ValidationResult:
        """Validate a single example with multi-dimensional quality assessment."""
        logger.debug(f"Validating example {example_index}")
        
        try:
            # Prepare validation context with uniqueness score
            full_context = validation_context.copy()
            full_context['example'] = example
            full_context['example_index'] = example_index
            full_context['semantic_uniqueness_score'] = uniqueness_score
            
            context_json = json.dumps(full_context, indent=2, default=str)
            
            # Load validation prompts
            system_prompt = load_prompt(
                "enhanced_validation_prompts",
                "prompts.multi_dimensional_validation.system.template"
            )
            
            user_prompt = load_prompt(
                "enhanced_validation_prompts",
                "prompts.multi_dimensional_validation.user.template",
                validation_context_json=context_json
            )
            
            # Call LLM for validation
            response_content = process_llm_request(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name=MODEL_NAME,
                temperature=0.0  # Consistent validation
            )
            
            # Parse response
            validation_data = self._parse_validation_response(response_content)
            
            # Extract quality scores
            quality_score = QualityScore.from_dict(validation_data.get('quality_scores', {}))
            
            # Determine sample type
            sample_type = self._determine_sample_type(example)
            
            # Check if high quality
            is_high_quality = quality_score.is_high_quality(self.quality_thresholds)
            
            # Create validation result
            result = ValidationResult(
                example_index=example_index,
                example=example,
                quality_score=quality_score,
                is_high_quality=is_high_quality,
                validation_details=validation_data,
                sample_type=sample_type,
                issues=validation_data.get('issues', []),
                strengths=validation_data.get('strengths', []),
                recommendations=validation_data.get('recommendations', [])
            )
            
            logger.debug(f"Example {example_index} validation completed - Quality: {is_high_quality}")
            return result
            
        except Exception as e:
            logger.error(f"Error validating example {example_index}: {str(e)}")
            # Return failed validation
            return ValidationResult(
                example_index=example_index,
                example=example,
                quality_score=QualityScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                is_high_quality=False,
                validation_details={'error': str(e)},
                sample_type='unknown',
                issues=[f'Validation error: {str(e)}'],
                strengths=[],
                recommendations=['Review example format and content']
            )
    
    def _parse_validation_response(self, response_content: str) -> Dict[str, Any]:
        """Parse LLM validation response."""
        try:
            # Clean up content
            if response_content.startswith('```'):
                first_backticks_end = response_content.find('\n', 3)
                if first_backticks_end != -1:
                    last_backticks_start = response_content.rfind('```')
                    if last_backticks_start > first_backticks_end:
                        response_content = response_content[first_backticks_end + 1:last_backticks_start].strip()
            
            return json.loads(response_content)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse validation response: {str(e)}")
            return {
                'quality_scores': {
                    'technical_accuracy': 0.0,
                    'schema_consistency': 0.0,
                    'realism_assessment': 0.0,
                    'semantic_uniqueness': 0.0,
                    'domain_alignment': 0.0,
                    'composite_score': 0.0
                },
                'issues': ['Failed to parse validation response'],
                'strengths': [],
                'recommendations': ['Review validation prompt and response']
            }
    
    def _determine_sample_type(self, example: Dict[str, Any]) -> str:
        """Determine if example is malicious or benign."""
        # Check various labeling schemes
        if example.get('label') == 1 or example.get('Label') == 1:
            return 'malicious'
        elif example.get('label') == 0 or example.get('Label') == 0:
            return 'benign'
        elif example.get('_metadata', {}).get('sample_type') == 'malicious':
            return 'malicious'
        elif example.get('_metadata', {}).get('sample_type') == 'benign':
            return 'benign'
        else:
            return 'unknown'
    
    def validate_seed_batch_parallel(self, 
                                   examples: List[Dict[str, Any]], 
                                   metadata: Dict[str, Any],
                                   max_examples: int = None) -> Tuple[List[ValidationResult], Dict[str, Any]]:
        """Validate a batch of seed examples with parallel multi-dimensional quality assessment."""
        logger.info(f"🎯 Starting enhanced parallel validation of {len(examples)} seed examples")
        logger.info("="*80)
        
        self.stats['validation_start_time'] = time.time()
        
        # Limit examples if specified
        if max_examples and len(examples) > max_examples:
            logger.info(f"Limiting validation to {max_examples} examples")
            examples = examples[:max_examples]
        
        # Extract dataset information from metadata
        dataset_name = metadata.get('dataset_name', metadata.get('source', 'unknown'))
        
        # Load validation context
        logger.info("📋 Loading validation context...")
        domain_discovery = self.load_domain_discovery(dataset_name)
        contextual_problems = self.load_contextual_problems(dataset_name)
        data_info = self.load_data_info(dataset_name)
        
        # Calculate semantic uniqueness scores
        logger.info("🧮 Calculating semantic uniqueness scores...")
        uniqueness_scores = self.calculate_semantic_uniqueness_batch(examples)
        
        # Prepare validation context
        validation_context = {
            'domain_discovery': domain_discovery,
            'contextual_problems': contextual_problems,
            'data_info': data_info,
            'metadata': metadata,
            'quality_thresholds': self.quality_thresholds,
            'validation_timestamp': time.time()
        }
        
        # Create validation tasks
        logger.info("📝 Creating validation tasks...")
        validation_tasks = self.create_validation_tasks(examples, validation_context, uniqueness_scores)
        
        # Determine processing mode
        processing_mode = "Parallel" if self.enable_parallel and len(validation_tasks) > 1 else "Sequential"
        logger.info(f"🔄 Processing Mode: {processing_mode}")
        logger.info(f"👥 Workers: {self.max_workers if self.enable_parallel else 1}")
        logger.info("="*80)
        
        # Execute validation tasks
        validation_results = []
        
        if self.enable_parallel and len(validation_tasks) > 1:
            # Parallel execution
            validation_results = self._execute_parallel_seed_validation(validation_tasks)
        else:
            # Sequential execution
            validation_results = self._execute_sequential_seed_validation(validation_tasks)
        
        # Update final stats
        with self.stats_lock:
            self.stats['total_validated'] = len(validation_results)
            self.stats['validation_end_time'] = time.time()
        
        # Calculate batch statistics
        logger.info("📈 Calculating validation statistics...")
        batch_stats = self._calculate_batch_statistics(validation_results)
        
        # Final validation summary
        total_time = self.stats['validation_end_time'] - self.stats['validation_start_time']
        self._log_seed_validation_completion_summary(batch_stats, total_time)
        
        return validation_results, batch_stats
    
    def _execute_parallel_seed_validation(self, validation_tasks: List[ValidationTask]) -> List[ValidationResult]:
        """Execute seed validation tasks in parallel with enhanced progress tracking."""
        validation_results = []
        total_tasks = len(validation_tasks)
        
        # Progress tracking variables
        completed_count = 0
        high_quality_count = 0
        failed_count = 0
        start_time = time.time()
        last_progress_time = start_time
        
        logger.info(f"🚀 Starting parallel seed validation: {total_tasks} examples with {self.max_workers} workers")
        logger.info("="*80)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.execute_validation_task, task): task 
                for task in validation_tasks
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    task_result = future.result()
                    validation_results.append(task_result.validation_result)
                    completed_count += 1
                    
                    # Track quality
                    if task_result.validation_result.is_high_quality:
                        high_quality_count += 1
                    
                    # Enhanced progress logging
                    current_time = time.time()
                    should_log = (
                        completed_count % self.log_interval == 0 or 
                        completed_count == total_tasks or
                        completed_count == 1 or  # Log first completion
                        (current_time - last_progress_time) >= 10  # Log every 10 seconds minimum
                    )
                    
                    if should_log and self.enable_progress_logging:
                        self._log_seed_progress(
                            completed_count, total_tasks, high_quality_count, 
                            failed_count, start_time, current_time
                        )
                        last_progress_time = current_time
                    
                    # Add small delay to avoid overwhelming the API
                    if completed_count < total_tasks:
                        time.sleep(self.batch_delay)
                        
                except Exception as e:
                    logger.error(f"Task {task.task_id} raised exception: {e}")
                    failed_count += 1
                    completed_count += 1
                    
                    # Add a failed result
                    failed_result = ValidationResult(
                        example_index=task.example_index,
                        example=task.example,
                        quality_score=QualityScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                        is_high_quality=False,
                        validation_details={'error': str(e)},
                        sample_type='unknown',
                        issues=[f'Task execution error: {str(e)}'],
                        strengths=[],
                        recommendations=['Review task execution and validation process']
                    )
                    validation_results.append(failed_result)
        
        # Final progress summary
        total_time = time.time() - start_time
        self._log_final_seed_progress_summary(total_tasks, high_quality_count, failed_count, total_time)
        
        return validation_results
    
    def _execute_sequential_seed_validation(self, validation_tasks: List[ValidationTask]) -> List[ValidationResult]:
        """Execute seed validation tasks sequentially with enhanced progress tracking."""
        validation_results = []
        total_tasks = len(validation_tasks)
        
        # Progress tracking variables
        high_quality_count = 0
        failed_count = 0
        start_time = time.time()
        
        logger.info(f"🔄 Starting sequential seed validation: {total_tasks} examples")
        logger.info("="*80)
        
        for i, task in enumerate(validation_tasks):
            current_count = i + 1
            
            try:
                task_result = self.execute_validation_task(task)
                validation_results.append(task_result.validation_result)
                
                # Track quality
                if task_result.validation_result.is_high_quality:
                    high_quality_count += 1
                
                # Progress logging for sequential
                if current_count % (self.log_interval // 2) == 0 or current_count == total_tasks or current_count == 1:
                    self._log_sequential_seed_progress(
                        current_count, total_tasks, high_quality_count, 
                        failed_count, start_time, time.time()
                    )
                
                # Add delay between tasks
                if i < total_tasks - 1:
                    time.sleep(self.batch_delay)
                    
            except Exception as e:
                logger.error(f"Sequential task {task.task_id} failed: {e}")
                failed_count += 1
                
                # Add a failed result
                failed_result = ValidationResult(
                    example_index=task.example_index,
                    example=task.example,
                    quality_score=QualityScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    is_high_quality=False,
                    validation_details={'error': str(e)},
                    sample_type='unknown',
                    issues=[f'Sequential execution error: {str(e)}'],
                    strengths=[],
                    recommendations=['Review sequential execution and validation process']
                )
                validation_results.append(failed_result)
        
        # Final summary
        total_time = time.time() - start_time
        self._log_final_seed_progress_summary(total_tasks, high_quality_count, failed_count, total_time)
        
        return validation_results
    
    def _log_seed_progress(self, completed: int, total: int, high_quality: int, 
                          failed: int, start_time: float, current_time: float):
        """Log detailed seed validation progress with visual indicators."""
        # Calculate metrics
        progress_pct = (completed / total) * 100
        elapsed_time = current_time - start_time
        rate_per_min = (completed / elapsed_time) * 60 if elapsed_time > 0 else 0
        
        # Estimate remaining time
        if completed > 0 and elapsed_time > 0:
            estimated_total_time = (total / completed) * elapsed_time
            eta_seconds = estimated_total_time - elapsed_time
            eta_minutes = eta_seconds / 60
        else:
            eta_minutes = 0
        
        # Quality metrics
        quality_rate = (high_quality / completed) * 100 if completed > 0 else 0
        success_rate = ((completed - failed) / completed) * 100 if completed > 0 else 0
        
        # Create progress bar
        bar_width = 40
        filled_width = int((progress_pct / 100) * bar_width)
        progress_bar = "█" * filled_width + "░" * (bar_width - filled_width)
        
        # Log progress with visual indicators
        logger.info(f"🌱 Seed Progress: [{progress_bar}] {progress_pct:.1f}% ({completed:,}/{total:,})")
        logger.info(f"   ⚡ Rate: {rate_per_min:.1f} seeds/min | ⏱️  ETA: {eta_minutes:.1f} min | ⏰ Elapsed: {elapsed_time:.1f}s")
        logger.info(f"   ✅ High Quality: {high_quality:,} ({quality_rate:.1f}%) | ❌ Failed: {failed:,} | 📈 Success: {success_rate:.1f}%")
        logger.info("   " + "-" * 70)
    
    def _log_sequential_seed_progress(self, completed: int, total: int, high_quality: int, 
                                     failed: int, start_time: float, current_time: float):
        """Log sequential seed validation progress with visual indicators."""
        # Calculate metrics
        progress_pct = (completed / total) * 100
        elapsed_time = current_time - start_time
        rate_per_min = (completed / elapsed_time) * 60 if elapsed_time > 0 else 0
        
        # Estimate remaining time
        if completed > 0 and elapsed_time > 0:
            estimated_total_time = (total / completed) * elapsed_time
            eta_seconds = estimated_total_time - elapsed_time
            eta_minutes = eta_seconds / 60
        else:
            eta_minutes = 0
        
        # Quality metrics
        quality_rate = (high_quality / completed) * 100 if completed > 0 else 0
        
        # Create simple progress bar
        bar_width = 30
        filled_width = int((progress_pct / 100) * bar_width)
        progress_bar = "█" * filled_width + "░" * (bar_width - filled_width)
        
        # Log sequential progress
        logger.info(f"🔄 Sequential: [{progress_bar}] {progress_pct:.1f}% ({completed}/{total})")
        logger.info(f"   ⚡ Rate: {rate_per_min:.1f}/min | ⏱️  ETA: {eta_minutes:.1f}min | ✅ Quality: {quality_rate:.1f}%")
    
    def _log_final_seed_progress_summary(self, total: int, high_quality: int, failed: int, total_time: float):
        """Log final seed validation progress summary with comprehensive metrics."""
        successful = total - failed
        quality_rate = (high_quality / total) * 100 if total > 0 else 0
        success_rate = (successful / total) * 100 if total > 0 else 0
        avg_rate = (total / total_time) * 60 if total_time > 0 else 0
        
        logger.info("="*80)
        logger.info("🎉 PARALLEL SEED VALIDATION COMPLETED!")
        logger.info("="*80)
        logger.info(f"🌱 Total Seeds: {total:,}")
        logger.info(f"✅ High Quality: {high_quality:,} ({quality_rate:.1f}%)")
        logger.info(f"❌ Failed: {failed:,} ({(failed/total)*100:.1f}%)")
        logger.info(f"📈 Success Rate: {success_rate:.1f}%")
        logger.info(f"⏱️  Total Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        logger.info(f"⚡ Average Rate: {avg_rate:.1f} seeds/minute")
        logger.info(f"🔄 Workers Used: {self.max_workers}")
        logger.info("="*80)
    
    def _log_seed_validation_completion_summary(self, batch_stats: Dict[str, Any], total_time: float):
        """Log comprehensive seed validation completion summary."""
        logger.info("\n" + "="*80)
        logger.info(f"🎉 SEED VALIDATION COMPLETED")
        logger.info("="*80)
        
        # Basic metrics
        logger.info(f"🌱 Seed Metrics:")
        logger.info(f"   🎯 Total Examples: {batch_stats['total_count']:,}")
        logger.info(f"   🏆 High Quality: {batch_stats['high_quality_count']:,}")
        logger.info(f"   📈 Quality Rate: {batch_stats['high_quality_ratio']*100:.1f}%")
        
        # Performance metrics
        logger.info(f"\n⚡ Performance Metrics:")
        logger.info(f"   ⏱️  Total Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        logger.info(f"   🚀 Validation Rate: {batch_stats['total_count']/(total_time/60):.1f} seeds/minute")
        logger.info(f"   🔄 Workers Used: {self.max_workers if self.enable_parallel else 1}")
        logger.info(f"   📞 API Calls: {self.stats['total_api_calls']:,}")
        logger.info(f"   ✅ Success Rate: {(self.stats['successful_validations']/(self.stats['successful_validations']+self.stats['failed_validations'])*100):.1f}%" if self.stats['successful_validations']+self.stats['failed_validations'] > 0 else "N/A")
        
        # Quality breakdown
        logger.info(f"\n🎯 Quality Analysis:")
        avg_scores = batch_stats['average_scores']
        logger.info(f"   🔧 Technical Accuracy: {avg_scores['technical_accuracy']:.3f}")
        logger.info(f"   📋 Schema Consistency: {avg_scores['schema_consistency']:.3f}")
        logger.info(f"   🌍 Realism Assessment: {avg_scores['realism_assessment']:.3f}")
        logger.info(f"   🎨 Semantic Uniqueness: {avg_scores['semantic_uniqueness']:.3f}")
        logger.info(f"   🎯 Domain Alignment: {avg_scores['domain_alignment']:.3f}")
        logger.info(f"   🏆 Composite Score: {avg_scores['composite_score']:.3f}")
        
        # Sample type breakdown
        type_counts = batch_stats['type_counts']
        type_quality_counts = batch_stats['type_quality_counts']
        if type_counts:
            logger.info(f"\n📊 Sample Type Analysis:")
            for sample_type, count in type_counts.items():
                quality_count = type_quality_counts.get(sample_type, 0)
                quality_rate = (quality_count / count * 100) if count > 0 else 0
                if sample_type == 'malicious':
                    logger.info(f"   🔴 Malicious: {count:,} samples, {quality_count:,} high quality ({quality_rate:.1f}%)")
                elif sample_type == 'benign':
                    logger.info(f"   🟢 Benign: {count:,} samples, {quality_count:,} high quality ({quality_rate:.1f}%)")
                else:
                    logger.info(f"   ⚪ {sample_type.title()}: {count:,} samples, {quality_count:,} high quality ({quality_rate:.1f}%)")
        
        logger.info("="*80)
    
    def _calculate_batch_statistics(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Calculate comprehensive statistics for validation batch."""
        total_count = len(results)
        if total_count == 0:
            return {'total_count': 0, 'high_quality_count': 0, 'high_quality_ratio': 0.0}
        
        high_quality_count = sum(1 for r in results if r.is_high_quality)
        
        # Calculate average scores by dimension
        avg_scores = {
            'technical_accuracy': np.mean([r.quality_score.technical_accuracy for r in results]),
            'schema_consistency': np.mean([r.quality_score.schema_consistency for r in results]),
            'realism_assessment': np.mean([r.quality_score.realism_assessment for r in results]),
            'semantic_uniqueness': np.mean([r.quality_score.semantic_uniqueness for r in results]),
            'domain_alignment': np.mean([r.quality_score.domain_alignment for r in results]),
            'composite_score': np.mean([r.quality_score.composite_score for r in results])
        }
        
        # Count by sample type
        type_counts = defaultdict(int)
        type_quality_counts = defaultdict(int)
        
        for result in results:
            type_counts[result.sample_type] += 1
            if result.is_high_quality:
                type_quality_counts[result.sample_type] += 1
        
        # Calculate type-specific quality rates
        type_quality_rates = {}
        for sample_type, count in type_counts.items():
            quality_count = type_quality_counts[sample_type]
            type_quality_rates[sample_type] = quality_count / count if count > 0 else 0.0
        
        # Most common issues
        all_issues = []
        for result in results:
            all_issues.extend(result.issues)
        
        issue_counts = defaultdict(int)
        for issue in all_issues:
            issue_counts[issue] += 1
        
        common_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_count': total_count,
            'high_quality_count': high_quality_count,
            'high_quality_ratio': high_quality_count / total_count,
            'average_scores': avg_scores,
            'type_counts': dict(type_counts),
            'type_quality_counts': dict(type_quality_counts),
            'type_quality_rates': type_quality_rates,
            'common_issues': common_issues,
            'score_distribution': {
                'excellent': sum(1 for r in results if r.quality_score.composite_score >= 0.9),
                'good': sum(1 for r in results if 0.7 <= r.quality_score.composite_score < 0.9),
                'fair': sum(1 for r in results if 0.5 <= r.quality_score.composite_score < 0.7),
                'poor': sum(1 for r in results if r.quality_score.composite_score < 0.5)
            }
        }
    
    def adaptive_seed_selection(self, 
                               validation_results: List[ValidationResult],
                               target_count: int = None) -> Tuple[List[Dict], List[Dict]]:
        """Perform adaptive seed selection based on quality scores and diversity."""
        logger.info("Performing adaptive seed selection...")
        
        # Separate high and low quality examples
        high_quality_results = [r for r in validation_results if r.is_high_quality]
        low_quality_results = [r for r in validation_results if not r.is_high_quality]
        
        logger.info(f"Quality filtering: {len(high_quality_results)} high quality, {len(low_quality_results)} low quality")
        
        # If target count specified, perform intelligent selection
        if target_count and len(high_quality_results) > target_count:
            logger.info(f"Selecting top {target_count} examples based on composite scores and diversity")
            
            # Sort by composite score
            high_quality_results.sort(key=lambda x: x.quality_score.composite_score, reverse=True)
            
            # Select diverse high-quality examples
            selected_results = self._select_diverse_examples(high_quality_results, target_count)
            high_quality_results = selected_results
        
        # Extract examples
        high_quality_examples = [r.example for r in high_quality_results]
        low_quality_examples = [r.example for r in low_quality_results]
        
        logger.info(f"Adaptive selection completed: {len(high_quality_examples)} high quality selected")
        
        return high_quality_examples, low_quality_examples
    
    def _select_diverse_examples(self, 
                                results: List[ValidationResult], 
                                target_count: int) -> List[ValidationResult]:
        """Select diverse examples from high-quality results."""
        if len(results) <= target_count:
            return results
        
        selected = []
        remaining = results.copy()
        
        # Always select the highest quality example first
        best_result = max(remaining, key=lambda x: x.quality_score.composite_score)
        selected.append(best_result)
        remaining.remove(best_result)
        
        # Select remaining examples to maximize diversity
        while len(selected) < target_count and remaining:
            # Calculate diversity scores for remaining examples
            diversity_scores = []
            
            for candidate in remaining:
                # Calculate minimum similarity to selected examples
                min_similarity = float('inf')
                
                for selected_result in selected:
                    similarity = self._calculate_example_similarity(candidate.example, selected_result.example)
                    min_similarity = min(min_similarity, similarity)
                
                # Combine quality and diversity (higher diversity = lower similarity)
                diversity_score = candidate.quality_score.composite_score + (1.0 - min_similarity) * 0.3
                diversity_scores.append((candidate, diversity_score))
            
            # Select example with highest diversity score
            best_candidate, _ = max(diversity_scores, key=lambda x: x[1])
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        
        logger.info(f"Diversity-based selection: {len(selected)} examples selected")
        return selected
    
    def _calculate_example_similarity(self, example1: Dict, example2: Dict) -> float:
        """Calculate similarity between two examples (simple implementation)."""
        # Simple similarity based on string representation
        text1_parts = []
        text2_parts = []
        
        for key in set(example1.keys()).union(set(example2.keys())):
            if key != '_metadata':
                val1 = str(example1.get(key, '')).lower()
                val2 = str(example2.get(key, '')).lower()
                text1_parts.append(val1)
                text2_parts.append(val2)
        
        text1 = ' '.join(text1_parts)
        text2 = ' '.join(text2_parts)
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if len(words1) == 0 and len(words2) == 0:
            return 1.0
        elif len(words1) == 0 or len(words2) == 0:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def save_validation_results(self, 
                               validation_results: List[ValidationResult],
                               batch_stats: Dict[str, Any],
                               area: str,
                               nature: str):
        """Save comprehensive validation results."""
        logger.info("Saving seed validation results...")
        
        # Create validation directory
        seed_validation_dir = self.config_manager.data_dir / "seed_validation"
        area_clean = area.replace(' ', '_').replace('(', '').replace(')', '')
        area_dir = seed_validation_dir / area_clean
        area_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare validation report
        validation_report = {
            'validation_metadata': {
                'validation_timestamp': time.time(),
                'validator_version': '3.0_enhanced_parallel_config_driven',
                'total_examples_validated': len(validation_results),
                'validation_approach': 'parallel_multi_dimensional_quality_assessment',
                'quality_thresholds': self.quality_thresholds,
                'parallel_workers': self.max_workers,
                'parallel_enabled': self.enable_parallel,
                'configuration_source': 'scale_config.yaml'
            },
            'batch_statistics': batch_stats,
            'performance_metrics': {
                'total_validation_time': self.stats['validation_end_time'] - self.stats['validation_start_time'],
                'validation_rate_per_minute': len(validation_results) / ((self.stats['validation_end_time'] - self.stats['validation_start_time']) / 60),
                'successful_validations': self.stats['successful_validations'],
                'failed_validations': self.stats['failed_validations'],
                'total_api_calls': self.stats['total_api_calls'],
                'parallel_workers_used': self.max_workers if self.enable_parallel else 1,
                'parallel_mode': self.enable_parallel
            },
            'individual_results': []
        }
        
        # Add individual results (with truncated examples for space)
        for result in validation_results:
            result_data = {
                'example_index': result.example_index,
                'sample_type': result.sample_type,
                'is_high_quality': result.is_high_quality,
                'quality_scores': result.quality_score.to_dict(),
                'issues': result.issues,
                'strengths': result.strengths,
                'recommendations': result.recommendations,
                'validation_details': result.validation_details
            }
            validation_report['individual_results'].append(result_data)
        
        # Save validation report
        validation_file = area_dir / f"{nature}_seeds_validation_report.json"
        with validation_file.open('w', encoding='utf-8') as f:
            json.dump(validation_report, f, indent=2, default=str)
        
        logger.info(f"Seed validation report saved to: {validation_file}")
        
        # Save quality summary
        quality_summary = {
            'validation_mode': 'Parallel Multi-Dimensional Assessment',
            'total_examples': len(validation_results),
            'high_quality_count': batch_stats['high_quality_count'],
            'high_quality_ratio': batch_stats['high_quality_ratio'],
            'average_scores': batch_stats['average_scores'],
            'quality_distribution': batch_stats['score_distribution'],
            'type_quality_rates': batch_stats['type_quality_rates'],
            'top_issues': batch_stats['common_issues'][:3],
            'performance_metrics': validation_report['performance_metrics'],
            'configuration_used': {
                'max_workers': self.max_workers,
                'parallel_enabled': self.enable_parallel,
                'quality_thresholds': self.quality_thresholds
            },
            'validation_timestamp': time.time()
        }
        
        summary_file = area_dir / f"{nature}_quality_summary.json"
        with summary_file.open('w', encoding='utf-8') as f:
            json.dump(quality_summary, f, indent=2, default=str)
        
        logger.info(f"Seed quality summary saved to: {summary_file}")
        
        return validation_file, summary_file
    
    def save_filtered_seeds(self, 
                           high_quality_examples: List[Dict],
                           low_quality_examples: List[Dict],
                           original_metadata: Dict[str, Any],
                           area: str,
                           nature: str):
        """Save filtered high-quality seeds and low-quality seeds separately."""
        logger.info("Saving filtered seeds...")
        
        # Create seeds-validated directory
        seeds_validated_dir = self.config_manager.data_dir / "seeds-validated"
        area_clean = area.replace(' ', '_').replace('(', '').replace(')', '')
        area_dir = seeds_validated_dir / area_clean
        area_dir.mkdir(parents=True, exist_ok=True)
        
        # Create seeds-filtered directory for low quality examples
        seeds_filtered_dir = self.config_manager.data_dir / "seeds-filtered"
        filtered_area_dir = seeds_filtered_dir / area_clean
        filtered_area_dir.mkdir(parents=True, exist_ok=True)
        
        # Count by type for high quality examples
        high_malicious = sum(1 for ex in high_quality_examples 
                           if ex.get('label') == 1 or ex.get('_metadata', {}).get('sample_type') == 'malicious')
        high_benign = len(high_quality_examples) - high_malicious
        
        # Count by type for low quality examples
        low_malicious = sum(1 for ex in low_quality_examples 
                          if ex.get('label') == 1 or ex.get('_metadata', {}).get('sample_type') == 'malicious')
        low_benign = len(low_quality_examples) - low_malicious
        
        # Enhanced metadata for high quality seeds
        high_quality_metadata = original_metadata.copy()
        high_quality_metadata.update({
            'quality_status': 'validated_high_quality_parallel_config',
            'validation_method': 'parallel_multi_dimensional_enhanced_config_driven',
            'validation_timestamp': time.time(),
            'quality_filtering_applied': True,
            'parallel_validation': self.enable_parallel,
            'parallel_workers': self.max_workers,
            'configuration_driven': True,
            'original_count': original_metadata.get('total_examples', 0),
            'high_quality_count': len(high_quality_examples),
            'high_quality_malicious': high_malicious,
            'high_quality_benign': high_benign,
            'quality_thresholds_used': self.quality_thresholds,
            'adaptive_selection_applied': True,
            'configuration_source': 'scale_config.yaml'
        })
        
        # Save high quality seeds
        high_quality_file = area_dir / f"{nature}_examples.json"
        high_quality_data = {
            'examples': high_quality_examples,
            'metadata': high_quality_metadata
        }
        
        with high_quality_file.open('w', encoding='utf-8') as f:
            json.dump(high_quality_data, f, indent=2, default=str)
        
        logger.info(f"High quality seeds saved to: {high_quality_file}")
        logger.info(f"High quality results: {high_malicious} malicious, {high_benign} benign")
        
        # Enhanced metadata for filtered out examples
        filtered_metadata = original_metadata.copy()
        filtered_metadata.update({
            'quality_status': 'filtered_low_quality_parallel_config',
            'validation_method': 'parallel_multi_dimensional_enhanced_config_driven',
            'validation_timestamp': time.time(),
            'quality_filtering_applied': True,
            'parallel_validation': self.enable_parallel,
            'parallel_workers': self.max_workers,
            'configuration_driven': True,
            'filtered_count': len(low_quality_examples),
            'filtered_malicious': low_malicious,
            'filtered_benign': low_benign,
            'quality_thresholds_used': self.quality_thresholds,
            'reason_for_filtering': 'Failed to meet quality thresholds in parallel config-driven validation',
            'configuration_source': 'scale_config.yaml'
        })
        
        # Save filtered out examples for analysis
        filtered_file = None
        if low_quality_examples:
            filtered_file = filtered_area_dir / f"{nature}_examples.json"
            filtered_data = {
                'examples': low_quality_examples,
                'metadata': filtered_metadata
            }
            
            with filtered_file.open('w', encoding='utf-8') as f:
                json.dump(filtered_data, f, indent=2, default=str)
            
            logger.info(f"Filtered examples saved to: {filtered_file}")
            logger.info(f"Filtered results: {low_malicious} malicious, {low_benign} benign")
        
        return high_quality_file, filtered_file


def find_raw_seeds_files() -> List[Tuple[str, str, Path]]:
    """Find all raw seed files to validate."""
    seeds_raw_dir = config_manager.data_dir / "seeds-raw"
    
    if not seeds_raw_dir.exists():
        logger.warning(f"Seeds-raw directory not found: {seeds_raw_dir}")
        return []
    
    seed_files = []
    
    for area_dir in seeds_raw_dir.iterdir():
        if area_dir.is_dir():
            area = area_dir.name
            
            for seed_file in area_dir.glob("*_examples.json"):
                nature = seed_file.stem.replace('_examples', '')
                seed_files.append((area, nature, seed_file))
    
    logger.info(f"Found {len(seed_files)} raw seed files to validate")
    return seed_files


def main(config_override: Dict[str, Any] = None):
    """Main function to validate all raw seeds with enhanced parallel multi-dimensional assessment."""
    logger.info("="*80)
    logger.info("ENHANCED PARALLEL MULTI-DIMENSIONAL SEED VALIDATION")
    logger.info("="*80)
    
    # Initialize validator with configuration
    validator = EnhancedParallelSeedValidator(config_override=config_override)
    
    logger.info(f"Validation mode: Parallel Multi-Dimensional Assessment")
    logger.info(f"Parallel processing: {'Enabled' if validator.enable_parallel else 'Disabled'}")
    logger.info(f"Parallel workers: {validator.max_workers}")
    logger.info(f"Batch delay: {validator.batch_delay}s")
    logger.info(f"Quality thresholds: {validator.quality_thresholds}")
    
    # Find all raw seed files
    seed_files = find_raw_seeds_files()
    
    if not seed_files:
        logger.warning("No raw seed files found to validate")
        return
    
    # Process each seed file
    total_processed = 0
    total_high_quality = 0
    overall_start_time = time.time()
    
    # Track overall statistics
    overall_stats = {
        'total_datasets': len(seed_files),
        'successful_validations': 0,
        'failed_validations': 0,
        'total_examples_processed': 0,
        'total_high_quality_examples': 0,
        'total_validation_time': 0
    }
    
    for i, (area, nature, seed_file) in enumerate(seed_files):
        logger.info(f"\n{'='*60}")
        logger.info(f"🌱 VALIDATING DATASET {i+1}/{len(seed_files)}: {area}/{nature}")
        logger.info(f"{'='*60}")
        
        try:
            # Load raw seeds
            examples, metadata = validator.load_raw_seeds(area, nature)
            
            if not examples:
                logger.warning(f"No examples found in {seed_file}")
                continue
            
            # Validate seed batch with parallel processing
            validation_results, batch_stats = validator.validate_seed_batch_parallel(examples, metadata)
            
            # Adaptive seed selection
            high_quality_examples, low_quality_examples = validator.adaptive_seed_selection(validation_results)
            
            # Save validation results
            validation_file, summary_file = validator.save_validation_results(
                validation_results, batch_stats, area, nature
            )
            
            # Save filtered seeds
            high_quality_file, filtered_file = validator.save_filtered_seeds(
                high_quality_examples, low_quality_examples, metadata, area, nature
            )
            
            # Update totals
            total_processed += len(examples)
            total_high_quality += len(high_quality_examples)
            overall_stats['successful_validations'] += 1
            overall_stats['total_examples_processed'] += len(examples)
            overall_stats['total_high_quality_examples'] += len(high_quality_examples)
            
            # Log results
            logger.info(f"✅ Validation completed for {area}/{nature}:")
            logger.info(f"  - Processing mode: {'Parallel' if validator.enable_parallel else 'Sequential'} ({validator.max_workers} workers)")
            logger.info(f"  - Original examples: {len(examples):,}")
            logger.info(f"  - High quality: {len(high_quality_examples):,} ({batch_stats['high_quality_ratio']:.1%})")
            logger.info(f"  - Low quality: {len(low_quality_examples):,}")
            logger.info(f"  - Average composite score: {batch_stats['average_scores']['composite_score']:.3f}")
            logger.info(f"  - Validation time: {validator.stats['validation_end_time'] - validator.stats['validation_start_time']:.2f}s")
            logger.info(f"  - Validation rate: {len(examples)/((validator.stats['validation_end_time'] - validator.stats['validation_start_time'])/60):.1f} examples/minute")
            logger.info(f"  - Files saved:")
            logger.info(f"    → {validation_file}")
            logger.info(f"    → {summary_file}")
            logger.info(f"    → {high_quality_file}")
            if filtered_file:
                logger.info(f"    → {filtered_file}")
            
        except Exception as e:
            logger.error(f"Error validating {area}/{nature}: {str(e)}", exc_info=True)
            overall_stats['failed_validations'] += 1
            continue
    
    # Calculate total time
    overall_stats['total_validation_time'] = time.time() - overall_start_time
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("ENHANCED PARALLEL SEED VALIDATION COMPLETED")
    logger.info(f"{'='*80}")
    logger.info(f"Configuration source: scale_config.yaml")
    logger.info(f"Datasets processed: {overall_stats['successful_validations']}/{overall_stats['total_datasets']}")
    logger.info(f"Total examples processed: {overall_stats['total_examples_processed']:,}")
    logger.info(f"Total high quality examples: {overall_stats['total_high_quality_examples']:,}")
    
    if overall_stats['total_examples_processed'] > 0:
        overall_quality_rate = overall_stats['total_high_quality_examples'] / overall_stats['total_examples_processed']
        logger.info(f"Overall quality rate: {overall_quality_rate:.1%}")
    
    logger.info(f"Total validation time: {overall_stats['total_validation_time']:.2f} seconds")
    
    if overall_stats['total_validation_time'] > 0:
        overall_rate = overall_stats['total_examples_processed'] / (overall_stats['total_validation_time'] / 60)
        logger.info(f"Overall validation rate: {overall_rate:.1f} examples/minute")
    
    logger.info("")
    logger.info("Enhanced Parallel Seed Validation Features:")
    logger.info(f"  ✓ Configuration-driven from scale_config.yaml")
    logger.info(f"  ✓ {'Parallel processing' if validator.enable_parallel else 'Sequential processing'} with {validator.max_workers} workers")
    logger.info(f"  ✓ Multi-dimensional quality assessment")
    logger.info(f"  ✓ Real-time progress tracking with visual indicators")
    logger.info(f"  ✓ Adaptive seed selection and diversity optimization")
    logger.info(f"  ✓ Quality-weighted filtering")
    logger.info(f"  ✓ Comprehensive performance metrics")
    logger.info(f"  ✓ Intelligent rate limiting")
    logger.info(f"  ✓ Domain discovery integration")
    logger.info(f"  ✓ Contextual problem alignment")
    logger.info("")
    logger.info("Output Directories:")
    logger.info("  - data/seed_validation/ (validation reports)")
    logger.info("  - data/seeds-validated/ (high quality seeds)")
    logger.info("  - data/seeds-filtered/ (filtered analysis data)")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced parallel seed validation with configuration file support")
    parser.add_argument('--max-workers', type=int,
                       help='Override maximum number of parallel worker threads')
    parser.add_argument('--batch-delay', type=float,
                       help='Override delay between batches in seconds')
    parser.add_argument('--sequential', action='store_true',
                       help='Force sequential processing (disable parallel)')
    parser.add_argument('--disable-progress', action='store_true',
                       help='Disable detailed progress logging')
    
    args = parser.parse_args()
    
    # Build configuration overrides
    config_override = {}
    
    if args.max_workers is not None or args.batch_delay is not None or args.sequential or args.disable_progress:
        config_override['seed_validation'] = {}
        
        if args.max_workers is not None or args.batch_delay is not None or args.sequential:
            config_override['seed_validation']['parallel_processing'] = {}
            if args.max_workers is not None:
                config_override['seed_validation']['parallel_processing']['max_workers'] = args.max_workers
            if args.batch_delay is not None:
                config_override['seed_validation']['parallel_processing']['batch_delay'] = args.batch_delay
            if args.sequential:
                config_override['seed_validation']['parallel_processing']['enable_parallel'] = False
        
        if args.disable_progress:
            config_override['seed_validation']['performance_optimization'] = {
                'enable_progress_logging': False
            }
    
    main(config_override=config_override if config_override else None)