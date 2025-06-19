# cyberdata/process/seed_validator.py

import json
import numpy as np
import os
import sys
import time
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
logger = setup_logger("cyberdata.scripts.enhanced_seed_validator")

# Load environment variables
load_dotenv()

# Constants
MODEL_NAME = "gpt-4.1-mini"

# Quality thresholds
QUALITY_THRESHOLDS = {
    'technical_accuracy': 0.7,
    'schema_consistency': 0.9,
    'realism_assessment': 0.7,
    'semantic_uniqueness': 0.6,
    'domain_alignment': 0.7,
    'overall_minimum': 0.75
}

# Get config manager instance
config_manager = get_config_manager()

logger.info(f"Using model: {MODEL_NAME}")
logger.info(f"Quality thresholds: {QUALITY_THRESHOLDS}")


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
    
    def is_high_quality(self, thresholds: Dict[str, float] = None) -> bool:
        """Check if this score meets high quality thresholds."""
        if thresholds is None:
            thresholds = QUALITY_THRESHOLDS
        
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


class EnhancedSeedValidator:
    """Enhanced seed validator with multi-dimensional quality assessment."""
    
    def __init__(self):
        self.config_manager = config_manager
        self.quality_thresholds = QUALITY_THRESHOLDS.copy()
        self.validation_cache = {}
        
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
    
    def validate_individual_example(self, 
                                   example: Dict[str, Any],
                                   example_index: int,
                                   validation_context: Dict[str, Any]) -> ValidationResult:
        """Validate a single example with multi-dimensional quality assessment."""
        logger.debug(f"Validating example {example_index}")
        
        try:
            # Prepare validation context
            context_json = json.dumps({
                'example': example,
                'example_index': example_index,
                'validation_context': validation_context
            }, indent=2, default=str)
            
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
    
    def calculate_semantic_uniqueness(self, examples: List[Dict[str, Any]]) -> Dict[int, float]:
        """Calculate semantic uniqueness scores for examples."""
        logger.info("Calculating semantic uniqueness scores...")
        
        uniqueness_scores = {}
        
        # Simple text-based uniqueness calculation
        # In a real implementation, you might use embeddings
        text_representations = []
        
        for i, example in enumerate(examples):
            # Create text representation of example
            text_parts = []
            for key, value in example.items():
                if key != '_metadata' and isinstance(value, (str, int, float)):
                    text_parts.append(str(value).lower())
            
            text_repr = ' '.join(text_parts)
            text_representations.append(text_repr)
        
        # Calculate uniqueness based on text similarity
        for i, text_repr in enumerate(text_representations):
            similar_count = 0
            for j, other_repr in enumerate(text_representations):
                if i != j:
                    # Simple similarity check (in real implementation, use more sophisticated methods)
                    words_i = set(text_repr.split())
                    words_j = set(other_repr.split())
                    
                    if len(words_i) > 0 and len(words_j) > 0:
                        jaccard_similarity = len(words_i.intersection(words_j)) / len(words_i.union(words_j))
                        if jaccard_similarity > 0.7:  # High similarity threshold
                            similar_count += 1
            
            # Higher uniqueness score = fewer similar examples
            total_comparisons = len(text_representations) - 1
            uniqueness_score = 1.0 - (similar_count / total_comparisons) if total_comparisons > 0 else 1.0
            uniqueness_scores[i] = max(0.0, min(1.0, uniqueness_score))
        
        logger.info(f"Calculated uniqueness scores for {len(examples)} examples")
        return uniqueness_scores
    
    def validate_seed_batch(self, 
                           examples: List[Dict[str, Any]], 
                           metadata: Dict[str, Any],
                           max_examples: int = None) -> Tuple[List[ValidationResult], Dict[str, Any]]:
        """Validate a batch of seed examples with comprehensive quality assessment."""
        logger.info(f"Starting enhanced validation of {len(examples)} seed examples")
        
        # Limit examples if specified
        if max_examples and len(examples) > max_examples:
            logger.info(f"Limiting validation to {max_examples} examples")
            examples = examples[:max_examples]
        
        # Extract dataset information from metadata
        dataset_name = metadata.get('dataset_name', metadata.get('source', 'unknown'))
        
        # Load validation context
        domain_discovery = self.load_domain_discovery(dataset_name)
        contextual_problems = self.load_contextual_problems(dataset_name)
        data_info = self.load_data_info(dataset_name)
        
        # Calculate semantic uniqueness scores
        uniqueness_scores = self.calculate_semantic_uniqueness(examples)
        
        # Prepare validation context
        validation_context = {
            'domain_discovery': domain_discovery,
            'contextual_problems': contextual_problems,
            'data_info': data_info,
            'metadata': metadata,
            'quality_thresholds': self.quality_thresholds,
            'validation_timestamp': time.time()
        }
        
        # Validate individual examples
        validation_results = []
        
        for i, example in enumerate(examples):
            logger.info(f"Validating example {i+1}/{len(examples)}")
            
            # Add uniqueness score to validation context
            validation_context['semantic_uniqueness_score'] = uniqueness_scores.get(i, 0.5)
            
            result = self.validate_individual_example(example, i, validation_context)
            validation_results.append(result)
            
            # Small delay to avoid rate limiting
            if i < len(examples) - 1:
                time.sleep(0.2)
        
        # Calculate batch statistics
        batch_stats = self._calculate_batch_statistics(validation_results)
        
        logger.info(f"Batch validation completed: {batch_stats['high_quality_count']}/{len(validation_results)} high quality")
        
        return validation_results, batch_stats
    
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
        logger.info("Saving validation results...")
        
        # Create validation directory
        seed_validation_dir = self.config_manager.data_dir / "seed_validation"
        area_clean = area.replace(' ', '_').replace('(', '').replace(')', '')
        area_dir = seed_validation_dir / area_clean
        area_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare validation report
        validation_report = {
            'validation_metadata': {
                'validation_timestamp': time.time(),
                'validator_version': '2.0_enhanced',
                'total_examples_validated': len(validation_results),
                'validation_approach': 'multi_dimensional_quality_assessment',
                'quality_thresholds': self.quality_thresholds
            },
            'batch_statistics': batch_stats,
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
        validation_file = area_dir / f"{nature}_seed_validation_report.json"
        with validation_file.open('w', encoding='utf-8') as f:
            json.dump(validation_report, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to: {validation_file}")
        
        # Save quality summary
        quality_summary = {
            'total_examples': len(validation_results),
            'high_quality_count': batch_stats['high_quality_count'],
            'high_quality_ratio': batch_stats['high_quality_ratio'],
            'average_scores': batch_stats['average_scores'],
            'quality_distribution': batch_stats['score_distribution'],
            'type_quality_rates': batch_stats['type_quality_rates'],
            'top_issues': batch_stats['common_issues'][:3],
            'validation_timestamp': time.time()
        }
        
        summary_file = area_dir / f"{nature}_quality_summary.json"
        with summary_file.open('w', encoding='utf-8') as f:
            json.dump(quality_summary, f, indent=2, default=str)
        
        logger.info(f"Quality summary saved to: {summary_file}")
        
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
            'quality_status': 'validated_high_quality',
            'validation_method': 'multi_dimensional_enhanced',
            'validation_timestamp': time.time(),
            'quality_filtering_applied': True,
            'original_count': original_metadata.get('total_examples', 0),
            'high_quality_count': len(high_quality_examples),
            'high_quality_malicious': high_malicious,
            'high_quality_benign': high_benign,
            'quality_thresholds_used': self.quality_thresholds,
            'adaptive_selection_applied': True
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
            'quality_status': 'filtered_low_quality',
            'validation_method': 'multi_dimensional_enhanced',
            'validation_timestamp': time.time(),
            'quality_filtering_applied': True,
            'filtered_count': len(low_quality_examples),
            'filtered_malicious': low_malicious,
            'filtered_benign': low_benign,
            'quality_thresholds_used': self.quality_thresholds,
            'reason_for_filtering': 'Failed to meet quality thresholds'
        })
        
        # Save filtered out examples for analysis
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
        
        return high_quality_file, filtered_file if low_quality_examples else None


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


def main():
    """Main function to validate all raw seeds with enhanced multi-dimensional assessment."""
    logger.info("="*80)
    logger.info("ENHANCED MULTI-DIMENSIONAL SEED VALIDATION")
    logger.info("="*80)
    
    # Initialize validator
    validator = EnhancedSeedValidator()
    
    # Find all raw seed files
    seed_files = find_raw_seeds_files()
    
    if not seed_files:
        logger.warning("No raw seed files found to validate")
        return
    
    # Process each seed file
    total_processed = 0
    total_high_quality = 0
    
    for area, nature, seed_file in seed_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"VALIDATING: {area}/{nature}")
        logger.info(f"{'='*60}")
        
        try:
            # Load raw seeds
            examples, metadata = validator.load_raw_seeds(area, nature)
            
            if not examples:
                logger.warning(f"No examples found in {seed_file}")
                continue
            
            # Validate seed batch
            validation_results, batch_stats = validator.validate_seed_batch(examples, metadata)
            
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
            
            # Log results
            logger.info(f"Validation completed for {area}/{nature}:")
            logger.info(f"  - Original examples: {len(examples)}")
            logger.info(f"  - High quality: {len(high_quality_examples)} ({batch_stats['high_quality_ratio']:.1%})")
            logger.info(f"  - Low quality: {len(low_quality_examples)}")
            logger.info(f"  - Average composite score: {batch_stats['average_scores']['composite_score']:.3f}")
            logger.info(f"  - Files saved:")
            logger.info(f"    → {validation_file}")
            logger.info(f"    → {summary_file}")
            logger.info(f"    → {high_quality_file}")
            if filtered_file:
                logger.info(f"    → {filtered_file}")
            
        except Exception as e:
            logger.error(f"Error validating {area}/{nature}: {str(e)}", exc_info=True)
            continue
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("ENHANCED VALIDATION COMPLETED")
    logger.info(f"{'='*80}")
    logger.info(f"Total examples processed: {total_processed}")
    logger.info(f"Total high quality: {total_high_quality}")
    logger.info(f"Overall quality rate: {total_high_quality/total_processed:.1%}" if total_processed > 0 else "N/A")
    logger.info("")
    logger.info("Quality Enhancement Features:")
    logger.info("  ✓ Multi-dimensional quality scoring")
    logger.info("  ✓ Domain discovery integration")
    logger.info("  ✓ Contextual problem alignment")
    logger.info("  ✓ Schema consistency validation")
    logger.info("  ✓ Semantic uniqueness assessment")
    logger.info("  ✓ Adaptive seed selection")
    logger.info("  ✓ Quality-weighted filtering")
    logger.info("  ✓ Diversity-driven selection")
    logger.info("")
    logger.info("Output Directories:")
    logger.info("  - data/seed_validation/ (validation reports)")
    logger.info("  - data/seeds-validated/ (high quality seeds)")
    logger.info("  - data/seeds-filtered/ (filtered examples for analysis)")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    main()