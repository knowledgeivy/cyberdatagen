# cyberdata/process/validate_large.py

import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

# Add parent directory to path for imports
CURRENT_DIR = Path(__file__).parent
sys.path.append(str(CURRENT_DIR.parent))  # Add cyberdata package to path

from cyberdata.utils.config_manager import get_config_manager
# Import utilities
from cyberdata.utils.llm_invoke import process_llm_request
from cyberdata.utils.logger_config import setup_logger
from cyberdata.utils.prompt_loader import load_prompt

# Set up logger
logger = setup_logger("cyberdata.scripts.validate_large")

# Load environment variables
load_dotenv()

# Constants
MODEL_NAME = "gpt-4.1-mini"

# Get config manager instance
config_manager = get_config_manager()

# Validation configuration
SAMPLE_VALIDATION_BATCH_SIZE = 10  # Number of samples to validate individually
VALIDATION_SAMPLE_PERCENTAGE = 0.2  # Validate 20% of samples (or at least SAMPLE_VALIDATION_BATCH_SIZE)

logger.info(f"Using model: {MODEL_NAME}")
logger.info(f"Project root: {config_manager.project_root}")
logger.info(f"Large samples directory: {config_manager.large_samples_dir}")
logger.info(f"Reports directory: {config_manager.quality_reports_dir}")


def load_problems() -> list:
    """Load problem definitions using config manager"""
    return config_manager.load_problems()


def find_samples_file(problem):
    """
    Find the large samples file for a specific problem using config manager.
    """
    area = problem['area']
    nature = problem['nature']
    
    # First try the standard path
    expected_file = config_manager.get_large_samples_file(area, nature)
    if expected_file.exists():
        logger.debug(f"Found samples file: {expected_file}")
        return expected_file
    
    # If not found, use the config manager's search function
    found_file = config_manager.find_existing_file(
        config_manager.large_samples_dir, 
        nature, 
        "_large.json"
    )
    
    if found_file:
        logger.debug(f"Found samples file in alternative location: {found_file}")
        return found_file
    
    logger.warning(f"No samples file found for {area}/{nature}")
    return None


def load_samples(problem):
    """
    Load large samples for a given problem.
    
    Args:
        problem (dict): The problem definition
        
    Returns:
        tuple: (samples, metadata) - The samples and any metadata
    """
    samples_file = find_samples_file(problem)
    
    if not samples_file:
        logger.error(f"Large samples not found for {problem['area']}/{problem['nature']}")
        raise FileNotFoundError(f"Large samples not found for {problem['area']}/{problem['nature']}")
    
    data = json.loads(samples_file.read_text(encoding='utf-8'))
    samples = data.get('samples', [])
    metadata = data.get('metadata', {})
    logger.info(f"Loaded {len(samples)} samples from {samples_file}")
    
    # Log sample type distribution if available
    malicious_count = sum(1 for s in samples if s.get('sample_type') == 'malicious')
    benign_count = sum(1 for s in samples if s.get('sample_type') == 'benign')
    other_count = len(samples) - malicious_count - benign_count
    
    if malicious_count > 0 or benign_count > 0:
        logger.info(f"Sample distribution - Malicious: {malicious_count}, Benign: {benign_count}, Other: {other_count}")
    
    return samples, metadata


def automated_metrics(samples):
    """
    Calculate automated metrics for the samples, including sample type distribution.
    
    Args:
        samples (list): The samples to analyze
        
    Returns:
        dict: Metrics about the samples
    """
    logger.info(f"Calculating automated metrics for {len(samples)} samples")
    
    total = len(samples)
    
    # Unique count by serialized sample
    unique_samples = {json.dumps(s, sort_keys=True) for s in samples}
    unique_count = len(unique_samples)
    
    logger.debug(f"Found {unique_count} unique samples out of {total} total samples")
    
    # Required keys from first sample
    req_keys = set(samples[0].keys()) if samples else set()
    missing_counts = sum(1 for s in samples if not req_keys.issubset(s.keys()))
    
    logger.debug(f"Found {missing_counts} samples with missing keys")
    
    # Sample type distribution
    malicious_count = sum(1 for s in samples if s.get('sample_type') == 'malicious')
    benign_count = sum(1 for s in samples if s.get('sample_type') == 'benign')
    other_count = total - malicious_count - benign_count
    
    # Attack indicator distribution
    attack_true_count = sum(1 for s in samples if s.get('is_attack') is True)
    attack_false_count = sum(1 for s in samples if s.get('is_attack') is False)
    attack_missing_count = total - attack_true_count - attack_false_count
    
    metrics = {
        'total_samples': total,
        'unique_samples': unique_count,
        'uniqueness_ratio': unique_count / total if total else 0,
        'missing_keys_count': missing_counts,
        'completeness_ratio': (total - missing_counts) / total if total else 0,
        'sample_type_distribution': {
            'malicious': malicious_count,
            'benign': benign_count,
            'other': other_count,
            'malicious_ratio': malicious_count / total if total else 0,
            'benign_ratio': benign_count / total if total else 0
        },
        'attack_indicator_distribution': {
            'attack_true': attack_true_count,
            'attack_false': attack_false_count,
            'attack_missing': attack_missing_count,
            'attack_true_ratio': attack_true_count / total if total else 0
        }
    }
    
    logger.info(f"Metrics calculated: uniqueness ratio = {metrics['uniqueness_ratio']:.2f}")
    logger.info(f"Sample types - Malicious: {malicious_count}, Benign: {benign_count}")
    return metrics


def validate_individual_sample(problem: dict, sample: dict, sample_index: int) -> dict:
    """
    Validate an individual sample using LLM, considering both malicious and benign types.
    
    Args:
        problem (dict): The problem definition
        sample (dict): The sample to validate
        sample_index (int): Index of the sample in the dataset
        
    Returns:
        dict: Validation results for the sample
    """
    sample_type = sample.get('sample_type', 'unknown')
    logger.debug(f"Validating individual {sample_type} sample {sample_index} for problem: {problem['nature']}")
    
    # Serialize sample to JSON
    sample_json = json.dumps(sample, indent=2)
    
    # Choose appropriate validation prompt based on sample type
    if sample_type == 'benign':
        # Load benign validation prompts
        system_prompt = load_prompt(
            "validation_prompts",
            "prompts.benign_sample_validation.system.template",
            area=problem['area'],
            nature=problem['nature'],
            description=problem.get('description', '')
        )
        
        user_prompt = load_prompt(
            "validation_prompts",
            "prompts.benign_sample_validation.user.template",
            sample_json=sample_json
        )
    else:
        # Use existing malicious validation prompts
        system_prompt = load_prompt(
            "validation_prompts",
            "prompts.individual_sample_validation.system.template",
            area=problem['area'],
            nature=problem['nature'],
            description=problem.get('description', '')
        )
        
        user_prompt = load_prompt(
            "validation_prompts",
            "prompts.individual_sample_validation.user.template",
            sample_json=sample_json
        )
    
    # Use process_llm_request for LLM call
    response_content = process_llm_request(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_name=MODEL_NAME,
        temperature=0.0  # Use 0 for consistent evaluation
    )
    
    try:
        # Clean up content by removing markdown code blocks if present
        if response_content.startswith('```'):
            first_backticks_end = response_content.find('\n', 3)
            if first_backticks_end != -1:
                last_backticks_start = response_content.rfind('```')
                if last_backticks_start > first_backticks_end:
                    response_content = response_content[first_backticks_end + 1:last_backticks_start].strip()
        
        result = json.loads(response_content)
        result['sample_index'] = sample_index
        result['sample_type'] = sample_type
        logger.debug(f"Sample {sample_index} ({sample_type}) validation: valid={result.get('is_valid', False)}, score={result.get('overall_score', 0):.2f}")
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response for sample {sample_index}: {str(e)}")
        # Return a default failed validation
        return {
            'sample_index': sample_index,
            'sample_type': sample_type,
            'is_valid': False,
            'scores': {
                'relevance': 0.0,
                'consistency': 0.0,
                'correctness': 0.0,
                'completeness': 0.0,
                'realism': 0.0
            },
            'overall_score': 0.0,
            'issues': ['Failed to parse LLM validation response'],
            'strengths': []
        }


def validate_sample_batch(problem: dict, samples: List[dict], max_samples: int = None) -> List[dict]:
    """
    Validate a batch of samples individually, considering sample types.
    
    Args:
        problem (dict): The problem definition
        samples (list): All samples to potentially validate
        max_samples (int): Maximum number of samples to validate
        
    Returns:
        list: Validation results for each validated sample
    """
    # Determine how many samples to validate
    total_samples = len(samples)
    samples_to_validate = max(
        SAMPLE_VALIDATION_BATCH_SIZE,
        int(total_samples * VALIDATION_SAMPLE_PERCENTAGE)
    )
    
    if max_samples:
        samples_to_validate = min(samples_to_validate, max_samples)
    
    samples_to_validate = min(samples_to_validate, total_samples)
    
    logger.info(f"Validating {samples_to_validate} out of {total_samples} samples individually")
    
    # Try to select samples that represent both malicious and benign types
    malicious_indices = [i for i, s in enumerate(samples) if s.get('sample_type') == 'malicious']
    benign_indices = [i for i, s in enumerate(samples) if s.get('sample_type') == 'benign']
    other_indices = [i for i, s in enumerate(samples) if s.get('sample_type') not in ['malicious', 'benign']]
    
    selected_indices = []
    
    # If we have both types, try to select proportionally
    if malicious_indices and benign_indices:
        total_typed = len(malicious_indices) + len(benign_indices)
        malicious_to_select = int(samples_to_validate * len(malicious_indices) / total_typed)
        benign_to_select = int(samples_to_validate * len(benign_indices) / total_typed)
        
        # Ensure we don't exceed available samples
        malicious_to_select = min(malicious_to_select, len(malicious_indices))
        benign_to_select = min(benign_to_select, len(benign_indices))
        
        # Select random samples from each type
        selected_indices.extend(random.sample(malicious_indices, malicious_to_select))
        selected_indices.extend(random.sample(benign_indices, benign_to_select))
        
        # Fill remaining slots with other samples if available
        remaining_slots = samples_to_validate - len(selected_indices)
        if remaining_slots > 0 and other_indices:
            other_to_select = min(remaining_slots, len(other_indices))
            selected_indices.extend(random.sample(other_indices, other_to_select))
        
        logger.info(f"Selected {malicious_to_select} malicious, {benign_to_select} benign, {len(selected_indices) - malicious_to_select - benign_to_select} other samples for validation")
    else:
        # If we only have one type or no type info, select randomly
        selected_indices = random.sample(range(total_samples), samples_to_validate)
    
    validation_results = []
    
    for i, idx in enumerate(selected_indices):
        logger.info(f"Validating sample {i+1}/{len(selected_indices)} (index: {idx}, type: {samples[idx].get('sample_type', 'unknown')})...")
        
        try:
            result = validate_individual_sample(problem, samples[idx], idx)
            validation_results.append(result)
            
            # Add small delay to avoid rate limiting
            if i < len(selected_indices) - 1:
                time.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Error validating sample {idx}: {str(e)}")
            validation_results.append({
                'sample_index': idx,
                'sample_type': samples[idx].get('sample_type', 'unknown'),
                'is_valid': False,
                'overall_score': 0.0,
                'error': str(e)
            })
    
    return validation_results


def aggregate_validation_results(validation_results: List[dict]) -> dict:
    """
    Aggregate individual validation results into summary statistics, considering sample types.
    
    Args:
        validation_results (list): List of individual validation results
        
    Returns:
        dict: Aggregated statistics
    """
    if not validation_results:
        return {
            'samples_validated': 0,
            'valid_samples': 0,
            'invalid_samples': 0,
            'validity_rate': 0.0,
            'average_scores': {},
            'score_distribution': {},
            'type_specific_results': {}
        }
    
    valid_count = sum(1 for r in validation_results if r.get('is_valid', False))
    invalid_count = len(validation_results) - valid_count
    
    # Calculate overall score statistics
    overall_scores = [r.get('overall_score', 0.0) for r in validation_results]
    
    # Calculate average scores
    score_types = ['relevance', 'consistency', 'correctness', 'completeness', 'realism']
    average_scores = {}
    score_lists = {score_type: [] for score_type in score_types}
    
    for result in validation_results:
        scores = result.get('scores', {})
        for score_type in score_types:
            if score_type in scores:
                score_lists[score_type].append(scores[score_type])
    
    for score_type, scores in score_lists.items():
        if scores:
            average_scores[score_type] = statistics.mean(scores)
            average_scores[f'{score_type}_std'] = statistics.stdev(scores) if len(scores) > 1 else 0.0
    
    # Score distribution (bins)
    score_bins = {
        'excellent': sum(1 for s in overall_scores if s >= 0.9),
        'good': sum(1 for s in overall_scores if 0.7 <= s < 0.9),
        'fair': sum(1 for s in overall_scores if 0.5 <= s < 0.7),
        'poor': sum(1 for s in overall_scores if s < 0.5)
    }
    
    # Type-specific analysis
    type_specific_results = {}
    for sample_type in ['malicious', 'benign', 'unknown']:
        type_results = [r for r in validation_results if r.get('sample_type') == sample_type]
        if type_results:
            type_valid = sum(1 for r in type_results if r.get('is_valid', False))
            type_scores = [r.get('overall_score', 0.0) for r in type_results]
            
            type_specific_results[sample_type] = {
                'count': len(type_results),
                'valid_count': type_valid,
                'validity_rate': type_valid / len(type_results) if type_results else 0.0,
                'average_score': statistics.mean(type_scores) if type_scores else 0.0,
                'score_std': statistics.stdev(type_scores) if len(type_scores) > 1 else 0.0
            }
    
    # Common issues
    all_issues = []
    for r in validation_results:
        all_issues.extend(r.get('issues', []))
    
    # Count issue frequencies
    issue_counts = {}
    for issue in all_issues:
        issue_counts[issue] = issue_counts.get(issue, 0) + 1
    
    # Sort issues by frequency
    common_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        'samples_validated': len(validation_results),
        'valid_samples': valid_count,
        'invalid_samples': invalid_count,
        'validity_rate': valid_count / len(validation_results) if validation_results else 0.0,
        'average_scores': average_scores,
        'overall_score_mean': statistics.mean(overall_scores) if overall_scores else 0.0,
        'overall_score_std': statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0.0,
        'score_distribution': score_bins,
        'type_specific_results': type_specific_results,
        'common_issues': common_issues
    }


def evaluate_with_llm(problem, samples, snippet_size=5):
    """
    Use the LLM to evaluate overall sample quality, considering mixed sample types.
    
    Args:
        problem (dict): The problem definition
        samples (list): The samples to evaluate
        snippet_size (int): Number of samples to include in the evaluation
        
    Returns:
        dict: The evaluation results
    """
    # Try to get a representative sample including both types if available
    malicious_samples = [s for s in samples if s.get('sample_type') == 'malicious']
    benign_samples = [s for s in samples if s.get('sample_type') == 'benign']
    other_samples = [s for s in samples if s.get('sample_type') not in ['malicious', 'benign']]
    
    snippet = []
    
    # If we have both types, include samples from each
    if malicious_samples and benign_samples:
        mal_count = min(snippet_size // 2, len(malicious_samples))
        ben_count = min(snippet_size - mal_count, len(benign_samples))
        
        snippet.extend(random.sample(malicious_samples, mal_count))
        snippet.extend(random.sample(benign_samples, ben_count))
        
        # Fill remaining slots if needed
        remaining = snippet_size - len(snippet)
        if remaining > 0 and other_samples:
            snippet.extend(random.sample(other_samples, min(remaining, len(other_samples))))
    else:
        # Random sample if we don't have mixed types
        snippet = random.sample(samples, min(snippet_size, len(samples)))
    
    logger.info(f"Evaluating {len(snippet)} samples with LLM for overall quality assessment")
    
    # Count sample types in snippet
    snippet_malicious = sum(1 for s in snippet if s.get('sample_type') == 'malicious')
    snippet_benign = sum(1 for s in snippet if s.get('sample_type') == 'benign')
    logger.info(f"Snippet composition: {snippet_malicious} malicious, {snippet_benign} benign")
    
    # Serialize the snippet to JSON
    snippet_json = json.dumps(snippet, indent=2)
    
    # Load prompts from YAML - use mixed evaluation if we have both types
    if snippet_malicious > 0 and snippet_benign > 0:
        system_prompt = load_prompt(
            "validation_prompts",
            "prompts.mixed_quality_assessment.system.template"
        )
        
        user_prompt = load_prompt(
            "validation_prompts",
            "prompts.mixed_quality_assessment.user.template",
            nature=problem['nature'],
            area=problem['area'],
            description=problem.get('description', ''),
            snippet_json=snippet_json,
            malicious_count=snippet_malicious,
            benign_count=snippet_benign
        )
    else:
        # Use existing single-type evaluation
        system_prompt = load_prompt(
            "validation_prompts",
            "prompts.quality_assessment.system.template"
        )
        
        user_prompt = load_prompt(
            "validation_prompts",
            "prompts.quality_assessment.user.template",
            nature=problem['nature'],
            area=problem['area'],
            description=problem.get('description', ''),
            snippet_json=snippet_json
        )
    
    # Use process_llm_request for LLM call
    logger.info("Calling LLM for overall quality evaluation")
    response_content = process_llm_request(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_name=MODEL_NAME,
        temperature=0.0  # Use 0 for consistent evaluation
    )
    
    logger.debug(f"Response received, length: {len(response_content)} characters")
    
    try:
        # Clean up content by removing markdown code blocks if present
        if response_content.startswith('```'):
            logger.debug("Response starts with code block, cleaning up")
            first_backticks_end = response_content.find('\n', 3)
            if first_backticks_end != -1:
                last_backticks_start = response_content.rfind('```')
                if last_backticks_start > first_backticks_end:
                    response_content = response_content[first_backticks_end + 1:last_backticks_start].strip()
        
        result = json.loads(response_content)
        logger.info("Successfully parsed JSON evaluation response")
        return result
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {str(e)}")
        logger.debug(f"Raw response: {response_content[:100]}...")
        return {'realism': None, 'consistency': None, 'suggestions': [response_content]}


def main():
    """Main function to evaluate quality of samples for all problems."""
    logger.info("Starting quality evaluation of large samples (including mixed malicious/benign datasets)")
    problems = load_problems()
    
    for problem in problems:
        area = problem['area']
        nature = problem['nature']
        logger.info(f"Evaluating quality for {area}/{nature}...")
        
        try:
            samples, metadata = load_samples(problem)
            
            # Skip if no samples
            if not samples:
                logger.warning(f"No samples found for {area}/{nature}, skipping.")
                continue
            
            # Calculate automated metrics (now includes sample type distribution)
            logger.info("Calculating automated metrics...")
            metrics = automated_metrics(samples)
            
            # Validate individual samples
            logger.info("Validating individual samples...")
            individual_validations = validate_sample_batch(problem, samples)
            
            # Aggregate validation results (now includes type-specific analysis)
            logger.info("Aggregating validation results...")
            validation_summary = aggregate_validation_results(individual_validations)
            
            # Get overall quality assessment
            logger.info("Getting overall quality assessment...")
            overall_assessment = evaluate_with_llm(problem, samples)
            
            # Compile full report
            report = {
                'problem': {
                    'area': area,
                    'nature': nature,
                    'description': problem.get('description', '')
                },
                'automated_metrics': metrics,
                'individual_validations': {
                    'summary': validation_summary,
                    'detailed_results': individual_validations  # Include detailed results
                },
                'overall_assessment': overall_assessment,
                'dataset_metadata': metadata,
                'report_metadata': {
                    'total_samples': len(samples),
                    'samples_validated_individually': len(individual_validations),
                    'validation_percentage': len(individual_validations) / len(samples) * 100 if samples else 0,
                    'has_mixed_types': metrics['sample_type_distribution']['malicious'] > 0 and metrics['sample_type_distribution']['benign'] > 0
                }
            }
            
            # Get the report file path using config manager
            report_path = config_manager.get_quality_report_file(area, nature)
            
            # Ensure the directory exists
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the report
            with report_path.open('w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Saved quality report to {report_path}")
            
            # Log summary statistics
            logger.info(f"Summary for {area}/{nature}:")
            logger.info(f"  - Total samples: {len(samples)}")
            logger.info(f"  - Sample types: {metrics['sample_type_distribution']['malicious']} malicious, {metrics['sample_type_distribution']['benign']} benign")
            logger.info(f"  - Unique samples: {metrics['unique_samples']} ({metrics['uniqueness_ratio']:.1%})")
            logger.info(f"  - Valid samples: {validation_summary['valid_samples']}/{validation_summary['samples_validated']} ({validation_summary['validity_rate']:.1%})")
            logger.info(f"  - Average overall score: {validation_summary['overall_score_mean']:.2f}")
            
            # Log type-specific results
            for sample_type, type_results in validation_summary['type_specific_results'].items():
                if type_results['count'] > 0:
                    logger.info(f"  - {sample_type.capitalize()} samples: {type_results['count']} total, {type_results['valid_count']} valid ({type_results['validity_rate']:.1%}), avg score: {type_results['average_score']:.2f}")
            
            if validation_summary['average_scores']:
                logger.info("  - Average dimension scores:")
                for score_type in ['relevance', 'consistency', 'correctness', 'completeness', 'realism']:
                    if score_type in validation_summary['average_scores']:
                        logger.info(f"    - {score_type.capitalize()}: {validation_summary['average_scores'][score_type]:.2f}")
            
        except FileNotFoundError as e:
            logger.error(f"Error: {e}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error evaluating {area}/{nature}: {e}", exc_info=True)
            continue

    logger.info("Completed quality evaluation of large samples")


if __name__ == '__main__':
    main()