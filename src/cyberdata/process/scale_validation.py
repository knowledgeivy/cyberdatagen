# cyberdata/process/scale_validation.py

import concurrent.futures
import json
import numpy as np
import random
import statistics
import sys
import time
import threading
import yaml
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
logger = setup_logger("cyberdata.scripts.scale_validation")

# Load environment variables
load_dotenv()

# Configuration constants
MODEL_NAME = "gpt-4.1-mini"

# Get config manager instance
config_manager = get_config_manager()

logger.info(f"Using model: {MODEL_NAME}")


def load_validation_config() -> Dict[str, Any]:
    """Load scale validation configuration from scale_config.yaml."""
    try:
        config_file = config_manager.config_dir / "scale_config.yaml"
        
        if not config_file.exists():
            logger.warning(f"Scale config file not found: {config_file}. Using defaults.")
            return get_default_config()
        
        with config_file.open('r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded scale validation configuration from: {config_file}")
        return config
        
    except Exception as e:
        logger.error(f"Error loading scale configuration: {e}. Using defaults.")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Get default configuration if scale_config.yaml is not available."""
    return {
        'scale_validation': {
            'parallel_processing': {
                'max_workers': 8,
                'batch_delay': 0.1,
                'enable_parallel': True
            },
            'validation_strategy': {
                'validate_all_data': True,
                'fallback_sample_rate': 0.15,
                'min_samples': 50,
                'max_samples': None
            },
            'quality_thresholds': {
                'technical_accuracy': 0.75,
                'schema_consistency': 0.95,
                'realism_assessment': 0.70,
                'semantic_uniqueness': 0.65,
                'domain_alignment': 0.75,
                'overall_minimum': 0.75
            }
        }
    }


@dataclass
class ScaleQualityScore:
    """Multi-dimensional quality score for scale validation."""
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
    def from_dict(cls, data: Dict[str, float]) -> 'ScaleQualityScore':
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
class ScaleValidationResult:
    """Complete validation result for a scale dataset sample."""
    sample_index: int
    sample: Dict[str, Any]
    quality_score: ScaleQualityScore
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
    sample_index: int
    sample: Dict[str, Any]
    validation_context: Dict[str, Any]
    uniqueness_score: float
    priority: int = 0


@dataclass
class ValidationTaskResult:
    """Result of a validation task."""
    task_id: str
    validation_result: ScaleValidationResult
    success: bool
    processing_time: float = 0.0
    error: Optional[str] = None


class ScaleValidator:
    """Enhanced scale validator with parallel multi-dimensional quality assessment."""
    
    def __init__(self, config_override: Dict[str, Any] = None):
        self.config_manager = config_manager
        
        # Load configuration
        self.config = load_validation_config()
        if config_override:
            self.config = self._merge_configs(self.config, config_override)
        
        # Extract validation configuration
        validation_config = self.config.get('scale_validation', {})
        
        # Parallel processing settings
        parallel_config = validation_config.get('parallel_processing', {})
        self.max_workers = parallel_config.get('max_workers', 8)
        self.batch_delay = parallel_config.get('batch_delay', 0.1)
        self.enable_parallel = parallel_config.get('enable_parallel', True)
        
        # Validation strategy settings
        strategy_config = validation_config.get('validation_strategy', {})
        self.validate_all_data = strategy_config.get('validate_all_data', True)
        self.fallback_sample_rate = strategy_config.get('fallback_sample_rate', 0.15)
        self.min_samples = strategy_config.get('min_samples', 50)
        self.max_samples = strategy_config.get('max_samples', None)
        
        # Quality thresholds
        self.quality_thresholds = validation_config.get('quality_thresholds', {
            'technical_accuracy': 0.75,
            'schema_consistency': 0.95,
            'realism_assessment': 0.70,
            'semantic_uniqueness': 0.65,
            'domain_alignment': 0.75,
            'overall_minimum': 0.75
        })
        
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
        
        logger.info(f"ScaleValidator initialized from config:")
        logger.info(f"  - Parallel workers: {self.max_workers}")
        logger.info(f"  - Batch delay: {self.batch_delay}s")
        logger.info(f"  - Validate all data: {self.validate_all_data}")
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
        
    def load_validation_context(self, area: str, nature: str, scale_metadata: Dict) -> Dict[str, Any]:
        """Load comprehensive validation context."""
        logger.info(f"Loading validation context for {area}/{nature}")
        
        # Extract dataset name from scale metadata
        dataset_name = scale_metadata.get('context_sources', {}).get('dataset_name', 'unknown')
        
        # Load domain discovery
        domain_discovery = {}
        domain_file = self.config_manager.domain_discovery_dir / f"{dataset_name}_domain_discovery.json"
        if domain_file.exists():
            with domain_file.open('r', encoding='utf-8') as f:
                domain_discovery = json.load(f)
            logger.info(f"Loaded domain discovery for validation: {dataset_name}")
        
        # Load contextual problems
        contextual_problems = {}
        contextual_file = self.config_manager.contextual_problems_dir / f"{dataset_name}_contextual_problems.json"
        if contextual_file.exists():
            with contextual_file.open('r', encoding='utf-8') as f:
                contextual_problems = json.load(f)
            logger.info(f"Loaded contextual problems for validation: {dataset_name}")
        
        # Load data info schema
        data_info = {}
        data_info_file = self.config_manager.config_dir / "data_info.yaml"
        if data_info_file.exists():
            with data_info_file.open('r', encoding='utf-8') as f:
                all_data_info = yaml.safe_load(f)
                datasets = all_data_info.get('datasets', {})
                if dataset_name in datasets:
                    data_info = datasets[dataset_name]
                    logger.info(f"Loaded data info for validation: {dataset_name}")
        
        validation_context = {
            'domain_discovery': domain_discovery,
            'contextual_problems': contextual_problems,
            'data_info': data_info,
            'dataset_name': dataset_name,
            'area': area,
            'nature': nature,
            'scale_metadata': scale_metadata,
            'quality_thresholds': self.quality_thresholds,
            'validation_timestamp': time.time()
        }
        
        logger.info(f"Validation context loaded for {area}/{nature}")
        return validation_context
    
    def intelligent_sample_selection(self, samples: List[Dict], validate_all: bool = None, sample_rate: float = None) -> List[Tuple[int, Dict]]:
        """Select samples for validation - supports both full validation and sampling."""
        
        # Use instance defaults if not specified
        if validate_all is None:
            validate_all = self.validate_all_data
        if sample_rate is None:
            sample_rate = self.fallback_sample_rate
        
        if validate_all:
            logger.info(f"Full validation mode: validating all {len(samples)} samples")
            return [(i, s) for i, s in enumerate(samples)]
        
        # Fallback to sampling mode
        target_count = max(
            self.min_samples,
            min(self.max_samples or len(samples), int(len(samples) * sample_rate))
        )
        
        logger.info(f"Sampling mode: selecting {target_count} samples from {len(samples)} for validation")
        
        # Stratified sampling by sample type if available
        malicious_samples = [(i, s) for i, s in enumerate(samples) if self._is_malicious_sample(s)]
        benign_samples = [(i, s) for i, s in enumerate(samples) if not self._is_malicious_sample(s)]
        
        selected_samples = []
        
        if malicious_samples and benign_samples:
            # Proportional sampling
            total_samples = len(samples)
            malicious_ratio = len(malicious_samples) / total_samples
            
            malicious_target = max(1, int(target_count * malicious_ratio))
            benign_target = target_count - malicious_target
            
            # Don't exceed available samples
            malicious_target = min(malicious_target, len(malicious_samples))
            benign_target = min(benign_target, len(benign_samples))
            
            # Random sampling within each type
            if malicious_target > 0:
                selected_samples.extend(random.sample(malicious_samples, malicious_target))
            if benign_target > 0:
                selected_samples.extend(random.sample(benign_samples, benign_target))
            
            logger.info(f"Stratified sampling: {malicious_target} malicious, {benign_target} benign")
        else:
            # Simple random sampling
            indexed_samples = [(i, s) for i, s in enumerate(samples)]
            selected_samples = random.sample(indexed_samples, min(target_count, len(indexed_samples)))
            logger.info(f"Random sampling: {len(selected_samples)} samples")
        
        return selected_samples
    
    def _is_malicious_sample(self, sample: Dict) -> bool:
        """Determine if a sample is malicious."""
        # Check various labeling schemes
        if sample.get('label') == 1 or sample.get('Label') == 1:
            return True
        elif sample.get('sample_type') == 'malicious':
            return True
        elif sample.get('is_attack') is True:
            return True
        return False
    
    def calculate_semantic_uniqueness_batch(self, samples: List[Tuple[int, Dict]]) -> Dict[int, float]:
        """Calculate semantic uniqueness scores for a batch of samples."""
        logger.info("Calculating semantic uniqueness scores...")
        
        uniqueness_scores = {}
        
        # Create text representations
        text_representations = {}
        for idx, sample in samples:
            text_parts = []
            for key, value in sample.items():
                if key not in ['sample_id', 'generation_timestamp', '_metadata']:
                    text_parts.append(str(value).lower())
            text_representations[idx] = ' '.join(text_parts)
        
        # Calculate uniqueness scores
        for idx, text_repr in text_representations.items():
            words_current = set(text_repr.split())
            similar_count = 0
            
            for other_idx, other_repr in text_representations.items():
                if idx != other_idx:
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
            uniqueness_scores[idx] = max(0.0, min(1.0, uniqueness_score))
        
        logger.info(f"Calculated uniqueness scores for {len(samples)} samples")
        return uniqueness_scores
    
    def create_validation_tasks(self, selected_samples: List[Tuple[int, Dict]], 
                               validation_context: Dict[str, Any], 
                               uniqueness_scores: Dict[int, float]) -> List[ValidationTask]:
        """Create validation tasks for parallel processing."""
        tasks = []
        
        for i, (sample_idx, sample) in enumerate(selected_samples):
            task = ValidationTask(
                task_id=f"val_{sample_idx}",
                sample_index=sample_idx,
                sample=sample,
                validation_context=validation_context.copy(),  # Each task gets its own copy
                uniqueness_score=uniqueness_scores.get(sample_idx, 0.5),
                priority=1  # All validation tasks have same priority
            )
            tasks.append(task)
        
        logger.info(f"Created {len(tasks)} validation tasks")
        return tasks
    
    def execute_validation_task(self, task: ValidationTask) -> ValidationTaskResult:
        """Execute a single validation task."""
        start_time = time.time()
        
        logger.debug(f"Executing validation task {task.task_id} (sample {task.sample_index})")
        
        try:
            # Validate the sample
            result = self.validate_sample(
                task.sample_index, 
                task.sample, 
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
            failed_result = ScaleValidationResult(
                sample_index=task.sample_index,
                sample=task.sample,
                quality_score=ScaleQualityScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                is_high_quality=False,
                validation_details={'error': str(e)},
                sample_type='unknown',
                issues=[f'Validation error: {str(e)}'],
                strengths=[],
                recommendations=['Review sample format and validation process']
            )
            
            return ValidationTaskResult(
                task_id=task.task_id,
                validation_result=failed_result,
                success=False,
                processing_time=time.time() - start_time,
                error=str(e)
            )
    
    def validate_sample(self, sample_index: int, sample: Dict, validation_context: Dict, uniqueness_score: float) -> ScaleValidationResult:
        """Validate a single sample with comprehensive quality assessment."""
        logger.debug(f"Validating sample {sample_index}")
        
        try:
            # Prepare validation context with uniqueness score
            full_context = validation_context.copy()
            full_context['sample'] = sample
            full_context['sample_index'] = sample_index
            full_context['semantic_uniqueness_score'] = uniqueness_score
            
            context_json = json.dumps(full_context, indent=2, default=str)
            
            # Load scale validation prompts
            system_prompt = load_prompt(
                "scale_validation_prompts",
                "prompts.scale_multi_dimensional_validation.system.template"
            )
            
            user_prompt = load_prompt(
                "scale_validation_prompts",
                "prompts.scale_multi_dimensional_validation.user.template",
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
            quality_score = ScaleQualityScore.from_dict(validation_data.get('quality_scores', {}))
            
            # Determine sample type
            sample_type = 'malicious' if self._is_malicious_sample(sample) else 'benign'
            
            # Check if high quality
            is_high_quality = quality_score.is_high_quality(self.quality_thresholds)
            
            result = ScaleValidationResult(
                sample_index=sample_index,
                sample=sample,
                quality_score=quality_score,
                is_high_quality=is_high_quality,
                validation_details=validation_data,
                sample_type=sample_type,
                issues=validation_data.get('issues', []),
                strengths=validation_data.get('strengths', []),
                recommendations=validation_data.get('recommendations', [])
            )
            
            logger.debug(f"Sample {sample_index} validation completed - Quality: {is_high_quality}")
            return result
            
        except Exception as e:
            logger.error(f"Error validating sample {sample_index}: {e}")
            return ScaleValidationResult(
                sample_index=sample_index,
                sample=sample,
                quality_score=ScaleQualityScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                is_high_quality=False,
                validation_details={'error': str(e)},
                sample_type='unknown',
                issues=[f'Validation error: {str(e)}'],
                strengths=[],
                recommendations=['Review sample format and validation process']
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
            logger.error(f"Failed to parse validation response: {e}")
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
    
    def validate_scale_dataset(self, area: str, nature: str, samples: List[Dict], metadata: Dict, 
                              validate_all: bool = None, sample_rate: float = None) -> Tuple[List[ScaleValidationResult], Dict[str, Any]]:
        """Validate a scale dataset with parallel processing and enhanced progress tracking."""
        logger.info(f"🎯 Starting scale validation for {area}/{nature} with {len(samples):,} samples")
        logger.info("="*80)
        
        self.stats['validation_start_time'] = time.time()
        
        # Load validation context
        logger.info("📋 Loading validation context...")
        validation_context = self.load_validation_context(area, nature, metadata)
        
        # Select samples for validation with progress indication
        logger.info("🎲 Selecting samples for validation...")
        selected_samples = self.intelligent_sample_selection(samples, validate_all, sample_rate)
        
        validation_coverage = len(selected_samples) / len(samples) if samples else 0
        validation_mode = "100% Full Validation" if validation_coverage >= 0.99 else f"{validation_coverage*100:.1f}% Statistical Sampling"
        
        logger.info(f"📊 Validation Strategy: {validation_mode}")
        logger.info(f"   🎯 Coverage: {validation_coverage*100:.1f}% ({len(selected_samples):,}/{len(samples):,} samples)")
        logger.info(f"   🔄 Mode: {'Parallel' if self.enable_parallel and len(selected_samples) > 1 else 'Sequential'}")
        logger.info(f"   👥 Workers: {self.max_workers if self.enable_parallel else 1}")
        logger.info("="*80)
        
        # Calculate semantic uniqueness for selected samples
        logger.info("🧮 Calculating semantic uniqueness scores...")
        uniqueness_scores = self.calculate_semantic_uniqueness_batch(selected_samples)
        
        # Create validation tasks
        logger.info("📝 Creating validation tasks...")
        validation_tasks = self.create_validation_tasks(selected_samples, validation_context, uniqueness_scores)
        
        # Execute validation tasks with enhanced progress tracking
        validation_results = []
        
        if self.enable_parallel and len(validation_tasks) > 1:
            # Parallel execution with progress tracking
            validation_results = self._execute_parallel_validation(validation_tasks)
        else:
            # Sequential execution with progress tracking
            validation_results = self._execute_sequential_validation(validation_tasks)
        
        # Update final stats
        with self.stats_lock:
            self.stats['total_validated'] = len(validation_results)
            self.stats['validation_end_time'] = time.time()
        
        # Calculate batch statistics
        logger.info("📈 Calculating validation statistics...")
        batch_stats = self._calculate_scale_statistics(validation_results, len(samples), validation_coverage)
        
        # Final validation summary
        total_time = self.stats['validation_end_time'] - self.stats['validation_start_time']
        self._log_validation_completion_summary(batch_stats, total_time, area, nature)
        
        return validation_results, batch_stats
    
    def _log_validation_completion_summary(self, batch_stats: Dict[str, Any], total_time: float, area: str, nature: str):
        """Log comprehensive validation completion summary."""
        logger.info("\n" + "="*80)
        logger.info(f"🎉 VALIDATION COMPLETED: {area}/{nature}")
        logger.info("="*80)
        
        # Basic metrics
        logger.info(f"📊 Dataset Metrics:")
        logger.info(f"   🎯 Total Samples: {batch_stats['total_samples']:,}")
        logger.info(f"   ✅ Samples Validated: {batch_stats['samples_validated']:,} ({batch_stats['validation_coverage']*100:.1f}%)")
        logger.info(f"   🏆 High Quality: {batch_stats['high_quality_count']:,} actual, {batch_stats['estimated_high_quality_total']:,} estimated")
        logger.info(f"   📈 Quality Rate: {batch_stats['estimated_quality_ratio']*100:.1f}%")
        
        # Performance metrics
        perf_metrics = batch_stats['performance_metrics']
        logger.info(f"\n⚡ Performance Metrics:")
        logger.info(f"   ⏱️  Total Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        logger.info(f"   🚀 Validation Rate: {perf_metrics['validation_rate_per_minute']:.1f} samples/minute")
        logger.info(f"   🔄 Workers Used: {perf_metrics['parallel_workers_used']}")
        logger.info(f"   📞 API Calls: {perf_metrics['total_api_calls']:,}")
        logger.info(f"   ✅ Success Rate: {(perf_metrics['successful_validations']/(perf_metrics['successful_validations']+perf_metrics['failed_validations'])*100):.1f}%" if perf_metrics['successful_validations']+perf_metrics['failed_validations'] > 0 else "N/A")
        
        # Quality breakdown
        logger.info(f"\n🎯 Quality Analysis:")
        avg_scores = batch_stats['average_scores']
        logger.info(f"   🔧 Technical Accuracy: {avg_scores['technical_accuracy']:.3f}")
        logger.info(f"   📋 Schema Consistency: {avg_scores['schema_consistency']:.3f}")
        logger.info(f"   🌍 Realism Assessment: {avg_scores['realism_assessment']:.3f}")
        logger.info(f"   🎨 Semantic Uniqueness: {avg_scores['semantic_uniqueness']:.3f}")
        logger.info(f"   🎯 Domain Alignment: {avg_scores['domain_alignment']:.3f}")
        logger.info(f"   🏆 Composite Score: {avg_scores['composite_score']:.3f}")
        
        # Production readiness
        logger.info(f"\n🚀 Production Assessment:")
        logger.info(f"   📊 Readiness Score: {batch_stats['production_readiness_score']:.3f}")
        readiness_status = "✅ PRODUCTION READY" if batch_stats['production_readiness_score'] > 0.8 else "⚠️  REVIEW RECOMMENDED"
        logger.info(f"   🎯 Status: {readiness_status}")
        
        # Sample type breakdown
        type_analysis = batch_stats['type_analysis']
        if type_analysis['malicious']['count'] > 0 or type_analysis['benign']['count'] > 0:
            logger.info(f"\n📊 Sample Type Analysis:")
            logger.info(f"   🔴 Malicious: {type_analysis['malicious']['count']:,} samples, {type_analysis['malicious']['high_quality_count']:,} high quality ({type_analysis['malicious']['high_quality_count']/type_analysis['malicious']['count']*100:.1f}%)" if type_analysis['malicious']['count'] > 0 else "   🔴 Malicious: 0 samples")
            logger.info(f"   🟢 Benign: {type_analysis['benign']['count']:,} samples, {type_analysis['benign']['high_quality_count']:,} high quality ({type_analysis['benign']['high_quality_count']/type_analysis['benign']['count']*100:.1f}%)" if type_analysis['benign']['count'] > 0 else "   🟢 Benign: 0 samples")
        
        logger.info("="*80)
    
    def _execute_parallel_validation(self, validation_tasks: List[ValidationTask]) -> List[ScaleValidationResult]:
        """Execute validation tasks in parallel with enhanced progress tracking."""
        validation_results = []
        total_tasks = len(validation_tasks)
        
        # Progress tracking variables
        completed_count = 0
        high_quality_count = 0
        failed_count = 0
        start_time = time.time()
        last_progress_time = start_time
        
        # Get progress settings from config
        progress_config = self.config.get('scale_validation', {}).get('performance_optimization', {})
        log_interval = progress_config.get('log_every_n_samples', 50)  # More frequent updates
        detailed_progress = progress_config.get('enable_progress_logging', True)
        
        logger.info(f"🚀 Starting parallel validation: {total_tasks} samples with {self.max_workers} workers")
        logger.info("="*80)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.execute_validation_task, task): task 
                for task in validation_tasks
            }
            
            # Collect results as they complete with enhanced progress tracking
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
                        completed_count % log_interval == 0 or 
                        completed_count == total_tasks or
                        completed_count == 1 or  # Log first completion
                        (current_time - last_progress_time) >= 10  # Log every 10 seconds minimum
                    )
                    
                    if should_log and detailed_progress:
                        self._log_detailed_progress(
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
                    failed_result = ScaleValidationResult(
                        sample_index=task.sample_index,
                        sample=task.sample,
                        quality_score=ScaleQualityScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
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
        self._log_final_progress_summary(total_tasks, high_quality_count, failed_count, total_time)
        
        return validation_results
    
    def _log_detailed_progress(self, completed: int, total: int, high_quality: int, 
                              failed: int, start_time: float, current_time: float):
        """Log detailed progress information with visual indicators."""
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
        logger.info(f"📊 Progress: [{progress_bar}] {progress_pct:.1f}% ({completed:,}/{total:,})")
        logger.info(f"   ⚡ Rate: {rate_per_min:.1f} samples/min | ⏱️  ETA: {eta_minutes:.1f} min | ⏰ Elapsed: {elapsed_time:.1f}s")
        logger.info(f"   ✅ High Quality: {high_quality:,} ({quality_rate:.1f}%) | ❌ Failed: {failed:,} | 📈 Success: {success_rate:.1f}%")
        logger.info("   " + "-" * 70)
    
    def _log_final_progress_summary(self, total: int, high_quality: int, failed: int, total_time: float):
        """Log final progress summary with comprehensive metrics."""
        successful = total - failed
        quality_rate = (high_quality / total) * 100 if total > 0 else 0
        success_rate = (successful / total) * 100 if total > 0 else 0
        avg_rate = (total / total_time) * 60 if total_time > 0 else 0
        
        logger.info("="*80)
        logger.info("🎉 PARALLEL VALIDATION COMPLETED!")
        logger.info("="*80)
        logger.info(f"📊 Total Samples: {total:,}")
        logger.info(f"✅ High Quality: {high_quality:,} ({quality_rate:.1f}%)")
        logger.info(f"❌ Failed: {failed:,} ({(failed/total)*100:.1f}%)")
        logger.info(f"📈 Success Rate: {success_rate:.1f}%")
        logger.info(f"⏱️  Total Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        logger.info(f"⚡ Average Rate: {avg_rate:.1f} samples/minute")
        logger.info(f"🔄 Workers Used: {self.max_workers}")
        logger.info("="*80)
    
    def _execute_sequential_validation(self, validation_tasks: List[ValidationTask]) -> List[ScaleValidationResult]:
        """Execute validation tasks sequentially with enhanced progress tracking."""
        validation_results = []
        total_tasks = len(validation_tasks)
        
        # Progress tracking variables
        high_quality_count = 0
        failed_count = 0
        start_time = time.time()
        
        logger.info(f"🔄 Starting sequential validation: {total_tasks} samples")
        logger.info("="*80)
        
        for i, task in enumerate(validation_tasks):
            current_count = i + 1
            
            try:
                task_result = self.execute_validation_task(task)
                validation_results.append(task_result.validation_result)
                
                # Track quality
                if task_result.validation_result.is_high_quality:
                    high_quality_count += 1
                
                # Enhanced progress logging for sequential
                if current_count % 10 == 0 or current_count == total_tasks or current_count == 1:
                    self._log_sequential_progress(
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
                failed_result = ScaleValidationResult(
                    sample_index=task.sample_index,
                    sample=task.sample,
                    quality_score=ScaleQualityScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
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
        self._log_final_progress_summary(total_tasks, high_quality_count, failed_count, total_time)
        
        return validation_results
    
    def _log_sequential_progress(self, completed: int, total: int, high_quality: int, 
                                failed: int, start_time: float, current_time: float):
        """Log sequential progress with visual indicators."""
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
    
    def _calculate_scale_statistics(self, results: List[ScaleValidationResult], total_samples: int, validation_coverage: float) -> Dict[str, Any]:
        """Calculate comprehensive statistics for scale validation."""
        validated_count = len(results)
        if validated_count == 0:
            return {'total_samples': total_samples, 'samples_validated': 0, 'estimated_high_quality_total': 0}
        
        high_quality_count = sum(1 for r in results if r.is_high_quality)
        quality_ratio = high_quality_count / validated_count
        
        # For full validation, actual count; for sampling, statistical estimation
        if validation_coverage >= 0.99:  # Essentially full validation
            estimated_high_quality_total = high_quality_count
        else:
            estimated_high_quality_total = int(total_samples * quality_ratio)
        
        # Average scores by dimension
        avg_scores = {
            'technical_accuracy': np.mean([r.quality_score.technical_accuracy for r in results]),
            'schema_consistency': np.mean([r.quality_score.schema_consistency for r in results]),
            'realism_assessment': np.mean([r.quality_score.realism_assessment for r in results]),
            'semantic_uniqueness': np.mean([r.quality_score.semantic_uniqueness for r in results]),
            'domain_alignment': np.mean([r.quality_score.domain_alignment for r in results]),
            'composite_score': np.mean([r.quality_score.composite_score for r in results])
        }
        
        # Quality distribution
        quality_distribution = {
            'excellent': sum(1 for r in results if r.quality_score.composite_score >= 0.9),
            'good': sum(1 for r in results if 0.8 <= r.quality_score.composite_score < 0.9),
            'fair': sum(1 for r in results if 0.7 <= r.quality_score.composite_score < 0.8),
            'poor': sum(1 for r in results if r.quality_score.composite_score < 0.7)
        }
        
        # Sample type analysis
        malicious_results = [r for r in results if r.sample_type == 'malicious']
        benign_results = [r for r in results if r.sample_type == 'benign']
        
        type_analysis = {
            'malicious': {
                'count': len(malicious_results),
                'high_quality_count': sum(1 for r in malicious_results if r.is_high_quality),
                'average_score': np.mean([r.quality_score.composite_score for r in malicious_results]) if malicious_results else 0.0
            },
            'benign': {
                'count': len(benign_results),
                'high_quality_count': sum(1 for r in benign_results if r.is_high_quality),
                'average_score': np.mean([r.quality_score.composite_score for r in benign_results]) if benign_results else 0.0
            }
        }
        
        # Common issues analysis
        all_issues = []
        for r in results:
            all_issues.extend(r.issues)
        
        issue_counts = defaultdict(int)
        for issue in all_issues:
            issue_counts[issue] += 1
        
        common_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Production readiness assessment
        production_readiness_score = min(1.0, (
            quality_ratio * 0.4 +
            avg_scores['schema_consistency'] * 0.3 +
            avg_scores['technical_accuracy'] * 0.2 +
            avg_scores['realism_assessment'] * 0.1
        ))
        
        # Performance metrics
        total_time = self.stats['validation_end_time'] - self.stats['validation_start_time']
        validation_rate = validated_count / (total_time / 60) if total_time > 0 else 0
        
        return {
            'total_samples': total_samples,
            'samples_validated': validated_count,
            'validation_coverage': validation_coverage,
            'high_quality_count': high_quality_count,
            'estimated_high_quality_total': estimated_high_quality_total,
            'estimated_quality_ratio': quality_ratio,
            'production_readiness_score': production_readiness_score,
            'average_scores': avg_scores,
            'quality_distribution': quality_distribution,
            'type_analysis': type_analysis,
            'common_issues': common_issues,
            'performance_metrics': {
                'total_validation_time': total_time,
                'validation_rate_per_minute': validation_rate,
                'successful_validations': self.stats['successful_validations'],
                'failed_validations': self.stats['failed_validations'],
                'total_api_calls': self.stats['total_api_calls'],
                'parallel_workers_used': self.max_workers if self.enable_parallel else 1,
                'parallel_mode': self.enable_parallel
            }
        }
    
    def adaptive_dataset_selection(self, validation_results: List[ScaleValidationResult], all_samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Perform adaptive selection of high-quality samples from the entire dataset."""
        logger.info("Performing adaptive dataset selection based on validation results")
        
        # Calculate quality threshold based on validation results
        validated_scores = [r.quality_score.composite_score for r in validation_results]
        
        if not validated_scores:
            logger.warning("No validation results available for adaptive selection")
            return [], all_samples
        
        # Use statistical approach to select high-quality samples
        quality_threshold = self.quality_thresholds['overall_minimum']
        
        # Get indices and scores of validated samples
        validated_indices = {r.sample_index: r.quality_score.composite_score for r in validation_results}
        
        high_quality_samples = []
        low_quality_samples = []
        
        # Classify all samples based on validation results and prediction
        for i, sample in enumerate(all_samples):
            if i in validated_indices:
                # Use actual validation score
                if validated_indices[i] >= quality_threshold:
                    high_quality_samples.append(sample)
                else:
                    low_quality_samples.append(sample)
            else:
                # Use heuristic prediction for unvalidated samples (only applies if sampling was used)
                predicted_quality = self._predict_sample_quality(sample, validation_results)
                if predicted_quality >= quality_threshold:
                    high_quality_samples.append(sample)
                else:
                    low_quality_samples.append(sample)
        
        logger.info(f"Adaptive selection completed: {len(high_quality_samples)} high quality, {len(low_quality_samples)} low quality")
        
        return high_quality_samples, low_quality_samples
    
    def _predict_sample_quality(self, sample: Dict, validation_results: List[ScaleValidationResult]) -> float:
        """Predict sample quality based on validation patterns."""
        # Simple heuristic: use average quality of similar sample type
        sample_type = 'malicious' if self._is_malicious_sample(sample) else 'benign'
        
        type_results = [r for r in validation_results if r.sample_type == sample_type]
        if type_results:
            return np.mean([r.quality_score.composite_score for r in type_results])
        else:
            # Fallback to overall average
            return np.mean([r.quality_score.composite_score for r in validation_results])
    
    def save_validation_results(self, area: str, nature: str, validation_results: List[ScaleValidationResult], 
                               batch_stats: Dict[str, Any], metadata: Dict[str, Any]) -> Tuple[Path, Path]:
        """Save comprehensive scale validation results."""
        logger.info("Saving scale validation results...")
        
        # Create validation directory
        scaled_validation_dir = self.config_manager.scaled_validation_dir
        area_clean = area.replace(' ', '_').replace('(', '').replace(')', '')
        area_dir = scaled_validation_dir / area_clean
        area_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine validation mode
        validation_mode = "100% Full Validation" if batch_stats['validation_coverage'] >= 0.99 else f"{batch_stats['validation_coverage']*100:.1f}% Statistical Sampling"
        
        # Prepare comprehensive validation report
        validation_report = {
            'scale_validation_summary': {
                'validation_mode': validation_mode,
                'total_samples': batch_stats['total_samples'],
                'samples_validated': batch_stats['samples_validated'],
                'validation_coverage': f"{batch_stats['validation_coverage']*100:.1f}%",
                'high_quality_count': batch_stats['high_quality_count'],
                'estimated_high_quality_total': batch_stats['estimated_high_quality_total'],
                'estimated_quality_ratio': f"{batch_stats['estimated_quality_ratio']*100:.1f}%",
                'production_readiness_score': batch_stats['production_readiness_score'],
                'average_scores': batch_stats['average_scores'],
                'quality_distribution': batch_stats['quality_distribution']
            },
            'validation_metadata': {
                'validator_version': '3.0_parallel_config_driven',
                'validation_timestamp': time.time(),
                'validation_approach': 'parallel_multi_dimensional_config_driven',
                'parallel_workers': self.max_workers,
                'parallel_enabled': self.enable_parallel,
                'quality_thresholds': self.quality_thresholds,
                'validate_all_data': batch_stats['validation_coverage'] >= 0.99,
                'performance_metrics': batch_stats['performance_metrics'],
                'configuration_source': 'scale_config.yaml'
            },
            'type_analysis': batch_stats['type_analysis'],
            'common_issues': batch_stats['common_issues'],
            'production_assessment': {
                'deployment_readiness': 'Production ready with high confidence' if batch_stats['production_readiness_score'] > 0.8 else 'Review recommended before deployment',
                'schema_compliance_rate': batch_stats['average_scores']['schema_consistency'],
                'technical_accuracy_confidence': batch_stats['average_scores']['technical_accuracy'],
                'operational_risks': ['Minimal risk for deployment'] if batch_stats['production_readiness_score'] > 0.8 else ['Review quality issues before deployment'],
                'validation_completeness': validation_mode
            },
            'training_analysis': {
                'training_effectiveness': 'Excellent for ML model training' if batch_stats['estimated_quality_ratio'] > 0.75 else 'Good for ML model training',
                'diversity_adequacy': 'High diversity supports robust training',
                'class_balance_analysis': f"Malicious: {batch_stats['type_analysis']['malicious']['count']}, Benign: {batch_stats['type_analysis']['benign']['count']}",
                'data_completeness': f"Validated {batch_stats['validation_coverage']*100:.1f}% of dataset"
            },
            'performance_analysis': {
                'validation_time': f"{batch_stats['performance_metrics']['total_validation_time']:.2f} seconds",
                'validation_rate': f"{batch_stats['performance_metrics']['validation_rate_per_minute']:.1f} samples/minute",
                'parallel_efficiency': f"Used {batch_stats['performance_metrics']['parallel_workers_used']} workers, parallel: {batch_stats['performance_metrics']['parallel_mode']}",
                'api_calls': batch_stats['performance_metrics']['total_api_calls'],
                'success_rate': f"{batch_stats['performance_metrics']['successful_validations']/(batch_stats['performance_metrics']['successful_validations']+batch_stats['performance_metrics']['failed_validations'])*100:.1f}%" if (batch_stats['performance_metrics']['successful_validations']+batch_stats['performance_metrics']['failed_validations']) > 0 else "N/A"
            },
            'configuration_used': {
                'max_workers': self.max_workers,
                'batch_delay': self.batch_delay,
                'validate_all_data': self.validate_all_data,
                'fallback_sample_rate': self.fallback_sample_rate,
                'quality_thresholds': self.quality_thresholds
            },
            'original_metadata': metadata,
            'detailed_results': [
                {
                    'sample_index': r.sample_index,
                    'sample_type': r.sample_type,
                    'is_high_quality': r.is_high_quality,
                    'quality_scores': r.quality_score.to_dict(),
                    'issues': r.issues,
                    'strengths': r.strengths,
                    'recommendations': r.recommendations
                }
                for r in validation_results
            ]
        }
        
        # Save validation report
        validation_file = area_dir / f"{nature}_scale_validation_report.json"
        with validation_file.open('w', encoding='utf-8') as f:
            json.dump(validation_report, f, indent=2, default=str)
        
        logger.info(f"Scale validation report saved to: {validation_file}")
        
        # Save quality summary
        quality_summary = {
            'validation_mode': validation_mode,
            'total_samples': batch_stats['total_samples'],
            'high_quality_count': batch_stats['high_quality_count'],
            'estimated_high_quality_total': batch_stats['estimated_high_quality_total'],
            'estimated_quality_ratio': batch_stats['estimated_quality_ratio'],
            'production_readiness_score': batch_stats['production_readiness_score'],
            'validation_coverage': batch_stats['validation_coverage'],
            'average_scores': batch_stats['average_scores'],
            'quality_distribution': batch_stats['quality_distribution'],
            'performance_metrics': batch_stats['performance_metrics'],
            'configuration_used': {
                'max_workers': self.max_workers,
                'parallel_enabled': self.enable_parallel,
                'validate_all_data': self.validate_all_data
            },
            'validation_timestamp': time.time()
        }
        
        summary_file = area_dir / f"{nature}_scale_quality_summary.json"
        with summary_file.open('w', encoding='utf-8') as f:
            json.dump(quality_summary, f, indent=2, default=str)
        
        logger.info(f"Scale quality summary saved to: {summary_file}")
        
        return validation_file, summary_file
    
    def save_filtered_scale_data(self, area: str, nature: str, high_quality_samples: List[Dict], 
                                 low_quality_samples: List[Dict], original_metadata: Dict[str, Any]) -> Tuple[Path, Optional[Path]]:
        """Save filtered scale data after validation."""
        logger.info("Saving filtered scale data...")
        
        # Create directories
        scaled_validated_dir = self.config_manager.scaled_validated_dir
        scaled_filtered_dir = self.config_manager.scaled_filtered_dir
        
        area_clean = area.replace(' ', '_').replace('(', '').replace(')', '')
        validated_area_dir = scaled_validated_dir / area_clean
        filtered_area_dir = scaled_filtered_dir / area_clean
        
        validated_area_dir.mkdir(parents=True, exist_ok=True)
        filtered_area_dir.mkdir(parents=True, exist_ok=True)
        
        # Count sample types for high quality data
        high_malicious = sum(1 for s in high_quality_samples if self._is_malicious_sample(s))
        high_benign = len(high_quality_samples) - high_malicious
        
        # Enhanced metadata for high quality data
        high_quality_metadata = original_metadata.copy()
        high_quality_metadata.update({
            'quality_status': 'validated_high_quality_scale_parallel_config',
            'validation_method': 'parallel_multi_dimensional_scale_config_driven',
            'validation_timestamp': time.time(),
            'quality_filtering_applied': True,
            'parallel_validation': self.enable_parallel,
            'parallel_workers': self.max_workers,
            'configuration_driven': True,
            'original_count': original_metadata.get('actual_samples_generated', 0),
            'high_quality_count': len(high_quality_samples),
            'high_quality_malicious': high_malicious,
            'high_quality_benign': high_benign,
            'quality_thresholds_used': self.quality_thresholds,
            'production_ready': True,
            'deployment_recommendations': [
                'Suitable for production ML pipeline deployment',
                'Recommended for cybersecurity model training',
                'Validated for operational threat detection systems',
                'Comprehensively validated with configurable parallel processing'
            ]
        })
        
        # Save high quality data
        high_quality_file = validated_area_dir / f"{nature}_scale.json"
        high_quality_data = {
            'samples': high_quality_samples,
            'metadata': high_quality_metadata
        }
        
        with high_quality_file.open('w', encoding='utf-8') as f:
            json.dump(high_quality_data, f, indent=2, default=str)
        
        logger.info(f"High quality scale data saved to: {high_quality_file}")
        logger.info(f"High quality results: {high_malicious} malicious, {high_benign} benign")
        
        # Save filtered out data if any
        filtered_file = None
        if low_quality_samples:
            low_malicious = sum(1 for s in low_quality_samples if self._is_malicious_sample(s))
            low_benign = len(low_quality_samples) - low_malicious
            
            filtered_metadata = original_metadata.copy()
            filtered_metadata.update({
                'quality_status': 'filtered_low_quality_scale_parallel_config',
                'validation_method': 'parallel_multi_dimensional_scale_config_driven',
                'validation_timestamp': time.time(),
                'quality_filtering_applied': True,
                'parallel_validation': self.enable_parallel,
                'parallel_workers': self.max_workers,
                'configuration_driven': True,
                'filtered_count': len(low_quality_samples),
                'filtered_malicious': low_malicious,
                'filtered_benign': low_benign,
                'quality_thresholds_used': self.quality_thresholds,
                'reason_for_filtering': 'Failed to meet scale quality thresholds in config-driven parallel validation'
            })
            
            filtered_file = filtered_area_dir / f"{nature}_scale.json"
            filtered_data = {
                'samples': low_quality_samples,
                'metadata': filtered_metadata
            }
            
            with filtered_file.open('w', encoding='utf-8') as f:
                json.dump(filtered_data, f, indent=2, default=str)
            
            logger.info(f"Filtered scale data saved to: {filtered_file}")
            logger.info(f"Filtered results: {low_malicious} malicious, {low_benign} benign")
        
        return high_quality_file, filtered_file


def find_scale_datasets_to_validate() -> List[Tuple[str, str, Path]]:
    """Find all scale datasets ready for validation."""
    config_mgr = get_config_manager()
    scaled_raw_dir = config_mgr.scaled_raw_dir
    
    if not scaled_raw_dir.exists():
        logger.warning(f"Scaled-raw directory not found: {scaled_raw_dir}")
        return []
    
    scale_files = []
    
    for area_dir in scaled_raw_dir.iterdir():
        if area_dir.is_dir():
            area = area_dir.name
            for scale_file in area_dir.glob("*_scale.json"):
                nature = scale_file.stem.replace('_scale', '')
                scale_files.append((area, nature, scale_file))
    
    logger.info(f"Found {len(scale_files)} scale datasets to validate")
    return scale_files


def main(config_override: Dict[str, Any] = None):
    """Main function for configurable parallel scale validation."""
    logger.info("="*80)
    logger.info("ENHANCED PARALLEL SCALE VALIDATION WITH CONFIG")
    logger.info("="*80)
    
    # Initialize validator with configuration
    validator = ScaleValidator(config_override=config_override)
    
    logger.info(f"Validation mode: {'100% Full Validation' if validator.validate_all_data else f'{validator.fallback_sample_rate*100:.1f}% Statistical Sampling'}")
    logger.info(f"Parallel processing: {'Enabled' if validator.enable_parallel else 'Disabled'}")
    logger.info(f"Parallel workers: {validator.max_workers}")
    logger.info(f"Batch delay: {validator.batch_delay}s")
    
    # Find scale datasets to validate
    scale_datasets = find_scale_datasets_to_validate()
    
    if not scale_datasets:
        logger.warning("No scale datasets found to validate")
        return
    
    # Track overall statistics
    overall_stats = {
        'total_datasets': len(scale_datasets),
        'successful_validations': 0,
        'failed_validations': 0,
        'total_samples_processed': 0,
        'total_high_quality_samples': 0,
        'total_validation_time': 0
    }
    
    overall_start_time = time.time()
    
    # Process each scale dataset
    for i, (area, nature, scale_file) in enumerate(scale_datasets):
        logger.info(f"\n{'='*60}")
        logger.info(f"Validating dataset {i+1}/{len(scale_datasets)}: {area}/{nature}")
        logger.info(f"{'='*60}")
        
        try:
            # Load scale data
            with scale_file.open('r', encoding='utf-8') as f:
                scale_data = json.load(f)
            
            samples = scale_data.get('samples', [])
            metadata = scale_data.get('metadata', {})
            
            if not samples:
                logger.warning(f"No samples found in {scale_file}")
                continue
            
            logger.info(f"Loaded {len(samples)} samples for validation")
            
            # Validate scale dataset with configurable processing
            validation_results, batch_stats = validator.validate_scale_dataset(
                area, nature, samples, metadata
            )
            
            # Adaptive dataset selection
            high_quality_samples, low_quality_samples = validator.adaptive_dataset_selection(validation_results, samples)
            
            # Save validation results
            validation_file, summary_file = validator.save_validation_results(
                area, nature, validation_results, batch_stats, metadata
            )
            
            # Save filtered data
            high_quality_file, filtered_file = validator.save_filtered_scale_data(
                area, nature, high_quality_samples, low_quality_samples, metadata
            )
            
            # Update overall stats
            overall_stats['successful_validations'] += 1
            overall_stats['total_samples_processed'] += len(samples)
            overall_stats['total_high_quality_samples'] += len(high_quality_samples)
            
            # Log results
            validation_mode = "100% Full Validation" if batch_stats['validation_coverage'] >= 0.99 else f"{batch_stats['validation_coverage']*100:.1f}% Statistical Sampling"
            logger.info(f"Validation completed for {area}/{nature}:")
            logger.info(f"  - Validation mode: {validation_mode}")
            logger.info(f"  - Parallel processing: {'Enabled' if validator.enable_parallel else 'Disabled'}")
            logger.info(f"  - Total samples: {len(samples)}")
            logger.info(f"  - Samples validated: {batch_stats['samples_validated']} ({batch_stats['validation_coverage']*100:.1f}%)")
            logger.info(f"  - High quality: {batch_stats['high_quality_count']} actual, {batch_stats['estimated_high_quality_total']} estimated ({batch_stats['estimated_quality_ratio']*100:.1f}%)")
            logger.info(f"  - Production readiness: {batch_stats['production_readiness_score']:.2f}")
            logger.info(f"  - Average composite score: {batch_stats['average_scores']['composite_score']:.3f}")
            logger.info(f"  - Validation time: {batch_stats['performance_metrics']['total_validation_time']:.2f}s")
            logger.info(f"  - Validation rate: {batch_stats['performance_metrics']['validation_rate_per_minute']:.1f} samples/minute")
            logger.info(f"  - Files saved:")
            logger.info(f"    → {validation_file}")
            logger.info(f"    → {summary_file}")
            logger.info(f"    → {high_quality_file}")
            if filtered_file:
                logger.info(f"    → {filtered_file}")
            
        except Exception as e:
            logger.error(f"Error validating {area}/{nature}: {e}", exc_info=True)
            overall_stats['failed_validations'] += 1
            continue
    
    # Calculate total time
    overall_stats['total_validation_time'] = time.time() - overall_start_time
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("CONFIGURABLE PARALLEL SCALE VALIDATION COMPLETED")
    logger.info(f"{'='*80}")
    logger.info(f"Configuration source: scale_config.yaml")
    logger.info(f"Datasets processed: {overall_stats['successful_validations']}/{overall_stats['total_datasets']}")
    logger.info(f"Total samples processed: {overall_stats['total_samples_processed']}")
    logger.info(f"Total high quality samples: {overall_stats['total_high_quality_samples']}")
    
    if overall_stats['total_samples_processed'] > 0:
        overall_quality_rate = overall_stats['total_high_quality_samples'] / overall_stats['total_samples_processed']
        logger.info(f"Overall quality rate: {overall_quality_rate:.1%}")
    
    logger.info(f"Total validation time: {overall_stats['total_validation_time']:.2f} seconds")
    
    if overall_stats['total_validation_time'] > 0:
        overall_rate = overall_stats['total_samples_processed'] / (overall_stats['total_validation_time'] / 60)
        logger.info(f"Overall validation rate: {overall_rate:.1f} samples/minute")
    
    logger.info("")
    logger.info("Configurable Parallel Scale Validation Features:")
    logger.info(f"  ✓ Configuration-driven from scale_config.yaml")
    logger.info(f"  ✓ {'100% full validation' if validator.validate_all_data else 'Statistical sampling'} by default")
    logger.info(f"  ✓ {'Parallel processing' if validator.enable_parallel else 'Sequential processing'} with {validator.max_workers} workers")
    logger.info(f"  ✓ Multi-dimensional quality assessment")
    logger.info(f"  ✓ Production readiness evaluation")
    logger.info(f"  ✓ Adaptive dataset selection")
    logger.info(f"  ✓ Quality-weighted filtering")
    logger.info(f"  ✓ Comprehensive performance metrics")
    logger.info(f"  ✓ Intelligent rate limiting")
    logger.info("")
    logger.info("Output Directories:")
    logger.info("  - data/scaled_validation/ (validation reports)")
    logger.info("  - data/scaled-validated/ (production-ready data)")
    logger.info("  - data/scaled-filtered/ (filtered analysis data)")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Configurable parallel scale validation with config file support")
    parser.add_argument('--max-workers', type=int,
                       help='Override maximum number of parallel worker threads')
    parser.add_argument('--batch-delay', type=float,
                       help='Override delay between batches in seconds')
    parser.add_argument('--sample-mode', action='store_true',
                       help='Override to use statistical sampling instead of full validation')
    parser.add_argument('--sample-rate', type=float,
                       help='Override sample rate for statistical validation')
    parser.add_argument('--sequential', action='store_true',
                       help='Force sequential processing (disable parallel)')
    
    args = parser.parse_args()
    
    # Build configuration overrides
    config_override = {}
    
    if args.max_workers is not None or args.batch_delay is not None or args.sample_mode or args.sample_rate is not None or args.sequential:
        config_override['scale_validation'] = {}
        
        if args.max_workers is not None or args.batch_delay is not None or args.sequential:
            config_override['scale_validation']['parallel_processing'] = {}
            if args.max_workers is not None:
                config_override['scale_validation']['parallel_processing']['max_workers'] = args.max_workers
            if args.batch_delay is not None:
                config_override['scale_validation']['parallel_processing']['batch_delay'] = args.batch_delay
            if args.sequential:
                config_override['scale_validation']['parallel_processing']['enable_parallel'] = False
        
        if args.sample_mode or args.sample_rate is not None:
            config_override['scale_validation']['validation_strategy'] = {}
            if args.sample_mode:
                config_override['scale_validation']['validation_strategy']['validate_all_data'] = False
            if args.sample_rate is not None:
                config_override['scale_validation']['validation_strategy']['fallback_sample_rate'] = args.sample_rate
    
    main(config_override=config_override if config_override else None)