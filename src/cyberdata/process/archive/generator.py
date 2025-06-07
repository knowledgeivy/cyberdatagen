# cyberdata/scripts/generator.py

import json
import os
import sys
import re
import random
from pathlib import Path

from dotenv import load_dotenv

# Add parent directory to path for imports
CURRENT_DIR = Path(__file__).parent
sys.path.append(str(CURRENT_DIR.parent))  # Add cyberdata package to path

# Import utilities
from cyberdata.utils.llm_invoke import process_llm_request
from cyberdata.utils.logger_config import setup_logger
from cyberdata.utils.prompt_loader import load_prompt
from cyberdata.utils.config_manager import get_config_manager

# Set up logger
logger = setup_logger("cyberdata.scripts.generator")

# Load environment variables
load_dotenv()

# Constants
MODEL_NAME = "gpt-4.1-mini"

# Get config manager instance
config_manager = get_config_manager()

logger.info(f"Using model: {MODEL_NAME}")
logger.info(f"Project root: {config_manager.project_root}")
logger.info(f"Seeds directory: {config_manager.seeds_dir}")
logger.info(f"Output directory: {config_manager.large_samples_dir}")


def load_problems() -> list:
    """Load problem definitions using config manager"""
    return config_manager.load_problems()


def calculate_sample_counts(total_count: int, malicious_ratio: float) -> tuple:
    """
    Calculate the number of malicious and benign samples based on ratio.
    
    Args:
        total_count (int): Total number of samples to generate
        malicious_ratio (float): Ratio of malicious samples (0.0 to 1.0)
        
    Returns:
        tuple: (malicious_count, benign_count)
    """
    if not 0.0 <= malicious_ratio <= 1.0:
        raise ValueError("Malicious ratio must be between 0.0 and 1.0")
    
    malicious_count = int(total_count * malicious_ratio)
    benign_count = total_count - malicious_count
    
    logger.info(f"Sample distribution: {malicious_count} malicious, {benign_count} benign (ratio: {malicious_ratio:.2f})")
    return malicious_count, benign_count


def find_examples_file(problem):
    """
    Find the examples file for a specific problem using config manager.
    """
    area = problem['area']
    nature = problem['nature']
    
    # First try the standard path
    expected_file = config_manager.get_seeds_file(area, nature)
    if expected_file.exists():
        logger.debug(f"Found examples file: {expected_file}")
        return expected_file
    
    # If not found, use the config manager's search function
    found_file = config_manager.find_existing_file(
        config_manager.seeds_dir, 
        nature, 
        "_examples.json"
    )
    
    if found_file:
        logger.debug(f"Found examples file in alternative location: {found_file}")
        return found_file
            
    logger.warning(f"No examples file found for {area}/{nature}")
    return None


def load_examples(problem: dict) -> list:
    """
    Load few-shot examples for a given problem.
    
    Args:
        problem (dict): The problem dictionary containing 'area' and 'nature'
    
    Returns:
        list: The examples for the given problem
    """
    example_file = find_examples_file(problem)
    
    if not example_file or not example_file.exists():
        logger.error(f"Examples file not found for {problem['area']}/{problem['nature']}")
        raise FileNotFoundError(f"Examples file not found for {problem['area']}/{problem['nature']}")
    
    data = json.loads(example_file.read_text(encoding='utf-8'))
    examples = data.get('examples', [])
    logger.info(f"Loaded {len(examples)} examples from {example_file}")
    return examples


def extract_json_from_response(response_content: str) -> dict:
    """
    Extract valid JSON from LLM response that might contain markdown or other content.
    """
    logger.debug("Extracting JSON from LLM response")
    
    # Clean up content by removing markdown code blocks if present
    if '```' in response_content:
        logger.debug("Response contains code blocks, cleaning up")
        # Find the first and last backtick groups
        pattern = r'```(?:json)?\s*([\s\S]*?)```'
        matches = re.findall(pattern, response_content)
        if matches:
            response_content = matches[0]
            logger.debug("Extracted content from code block")
    
    # Try direct JSON parsing
    try:
        result = json.loads(response_content)
        logger.debug("Successfully parsed JSON directly")
        return result
    except json.JSONDecodeError as e:
        logger.warning(f"Direct JSON parsing failed: {str(e)}")
        # Try fixing common JSON issues
        try:
            logger.debug("Attempting to fix common JSON issues")
            # Replace single quotes with double quotes
            fixed_content = response_content.replace("'", '"')
            # Fix missing commas after closing braces in arrays
            fixed_content = re.sub(r'}\s*{', '},{', fixed_content)
            # Fix trailing commas in arrays/objects
            fixed_content = re.sub(r',\s*}', '}', fixed_content)
            fixed_content = re.sub(r',\s*]', ']', fixed_content)
            
            result = json.loads(fixed_content)
            logger.debug("Successfully parsed JSON after fixing common issues")
            return result
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing after fixes failed: {str(e)}")
            # Try to find any JSON object in the response
            try:
                logger.debug("Attempting to extract JSON object from response")
                start_idx = response_content.find('{')
                end_idx = response_content.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    potential_json = response_content[start_idx:end_idx+1]
                    result = json.loads(potential_json)
                    logger.debug("Successfully extracted JSON object from response")
                    return result
            except Exception as e:
                logger.warning(f"JSON extraction from response failed: {str(e)}")
                pass
            
            logger.error(f"Failed to extract JSON from response")
            return {"samples": []}


def generate_malicious_samples(problem: dict, examples: list, count: int, batch_size: int = 5) -> list:
    """
    Generate malicious samples using existing prompts.
    """
    if count <= 0:
        return []
    
    logger.info(f"Generating {count} malicious samples for {problem['nature']}")
    all_samples = []
    remaining = count
    
    # Only use the first two examples to avoid token limit issues
    examples_to_show = examples[:2]
    examples_json = json.dumps(examples_to_show, indent=2)
    
    while remaining > 0:
        current_batch_size = min(batch_size, remaining)
        logger.info(f"Generating malicious batch of {current_batch_size} samples for {problem['nature']} ({len(all_samples)}/{count} so far)...")
        
        # Load prompts from YAML
        system_content = load_prompt(
            "large_generation_prompts",
            "prompts.generation.system.template"
        )
        
        user_content = load_prompt(
            "large_generation_prompts",
            "prompts.generation.user.template",
            nature=problem['nature'],
            area=problem['area'],
            description=problem.get('description', ''),
            examples_json=examples_json,
            count=current_batch_size
        )
        
        # Use process_llm_request to generate samples
        logger.info(f"Calling LLM for malicious batch generation")
        response_content = process_llm_request(
            system_prompt=system_content,
            user_prompt=user_content,
            model_name=MODEL_NAME,
            temperature=0.7
        )
        
        logger.debug(f"Malicious response received, length: {len(response_content)} characters")
        
        # Extract samples from response
        parsed = extract_json_from_response(response_content)
        samples = parsed.get('samples', [])
        
        if samples:
            # Mark samples as malicious
            for sample in samples:
                sample['sample_type'] = 'malicious'
                sample['is_attack'] = True
            
            all_samples.extend(samples)
            remaining -= len(samples)
            logger.info(f"Generated {len(samples)} malicious samples in this batch, total now: {len(all_samples)}")
        else:
            logger.warning(f"Failed to generate malicious samples in this batch, continuing...")
            remaining -= current_batch_size
        
        if len(all_samples) >= count:
            logger.info(f"Reached target malicious sample count of {count}")
            break
    
    return all_samples[:count]


def generate_benign_samples(problem: dict, count: int, batch_size: int = 5) -> list:
    """
    Generate benign samples using new benign prompts.
    """
    if count <= 0:
        return []
    
    logger.info(f"Generating {count} benign samples for {problem['area']} area")
    all_samples = []
    remaining = count
    
    while remaining > 0:
        current_batch_size = min(batch_size, remaining)
        logger.info(f"Generating benign batch of {current_batch_size} samples for {problem['area']} ({len(all_samples)}/{count} so far)...")
        
        # Load benign prompts from YAML
        system_content = load_prompt(
            "large_generation_prompts",
            "prompts.benign_generation.system.template"
        )
        
        user_content = load_prompt(
            "large_generation_prompts",
            "prompts.benign_generation.user.template",
            area=problem['area'],
            count=current_batch_size,
            context=problem.get('description', '')
        )
        
        # Use process_llm_request to generate samples
        logger.info(f"Calling LLM for benign batch generation")
        response_content = process_llm_request(
            system_prompt=system_content,
            user_prompt=user_content,
            model_name=MODEL_NAME,
            temperature=0.7
        )
        
        logger.debug(f"Benign response received, length: {len(response_content)} characters")
        
        # Extract samples from response
        parsed = extract_json_from_response(response_content)
        samples = parsed.get('samples', [])
        
        if samples:
            # Mark samples as benign
            for sample in samples:
                sample['sample_type'] = 'benign'
                sample['is_attack'] = False
            
            all_samples.extend(samples)
            remaining -= len(samples)
            logger.info(f"Generated {len(samples)} benign samples in this batch, total now: {len(all_samples)}")
        else:
            logger.warning(f"Failed to generate benign samples in this batch, continuing...")
            remaining -= current_batch_size
        
        if len(all_samples) >= count:
            logger.info(f"Reached target benign sample count of {count}")
            break
    
    return all_samples[:count]


def generate_for_problem(problem: dict, malicious_count: int, benign_count: int) -> list:
    """
    Generate both malicious and benign samples for the given problem.
    """
    nature = problem['nature']
    area = problem['area']
    total_count = malicious_count + benign_count
    logger.info(f"Generating {total_count} samples for problem: {area}/{nature} (malicious: {malicious_count}, benign: {benign_count})")
    
    all_samples = []
    
    try:
        # Generate malicious samples if needed
        if malicious_count > 0:
            # Load examples from appropriate area subdirectory
            examples = load_examples(problem)
            
            if not examples:
                logger.warning(f"No seed examples found for {area}/{nature}, skipping malicious generation")
            else:
                malicious_samples = generate_malicious_samples(problem, examples, malicious_count)
                all_samples.extend(malicious_samples)
        
        # Generate benign samples if needed
        if benign_count > 0:
            benign_samples = generate_benign_samples(problem, benign_count)
            all_samples.extend(benign_samples)
        
        # Shuffle the combined samples to avoid clustering by type
        random.shuffle(all_samples)
        
        logger.info(f"Generated total of {len(all_samples)} samples for {area}/{nature}")
        return all_samples
            
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error generating samples for {nature}: {e}", exc_info=True)
        return []


def save_samples(problem: dict, samples: list):
    """
    Save generated samples to a JSON file using config manager.
    """
    if not samples:
        logger.warning(f"No samples to save for {problem['nature']}")
        return
    
    area = problem['area']
    nature = problem['nature']
    
    # Get the output file path using config manager
    out_file = config_manager.get_large_samples_file(area, nature)
    
    # Ensure the directory exists
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if there are existing samples
    existing_samples = []
    if out_file.exists():
        try:
            existing_data = json.loads(out_file.read_text(encoding='utf-8'))
            existing_samples = existing_data.get('samples', [])
            logger.info(f"Found {len(existing_samples)} existing samples in {out_file}")
        except Exception as e:
            logger.error(f"Error reading existing file: {e}", exc_info=True)
    
    # Combine existing and new samples
    all_samples = existing_samples + samples
    
    # Calculate sample type distribution
    malicious_count = sum(1 for s in samples if s.get('sample_type') == 'malicious')
    benign_count = sum(1 for s in samples if s.get('sample_type') == 'benign')
    
    # Add metadata about the generation
    metadata = {
        'total_samples': len(all_samples),
        'new_samples_added': len(samples),
        'new_malicious_samples': malicious_count,
        'new_benign_samples': benign_count,
        'generation_timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else None
    }
    
    # Save samples with metadata
    output_data = {
        'samples': all_samples,
        'metadata': metadata
    }
    
    with out_file.open('w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Saved {len(samples)} new samples (malicious: {malicious_count}, benign: {benign_count}) to {out_file}")
    logger.info(f"Total samples in file: {len(all_samples)}")


def main(count: int = 10, malicious_ratio: float = 0.5, problem_filter: list = None):
    """
    Main function to generate samples for all problems with specified ratio.
    """
    logger.info(f"Starting sample generation with count={count}, malicious_ratio={malicious_ratio}")
    
    # Validate ratio
    if not 0.0 <= malicious_ratio <= 1.0:
        logger.error(f"Invalid malicious ratio: {malicious_ratio}. Must be between 0.0 and 1.0")
        return
    
    # Calculate sample counts
    total_malicious, total_benign = calculate_sample_counts(count, malicious_ratio)
    
    # Load problem definitions using config manager
    problems = load_problems()
    
    # Filter problems if specified
    if problem_filter:
        logger.info(f"Filtering problems to: {problem_filter}")
        problems = [p for p in problems if p['nature'] in problem_filter]
        if not problems:
            logger.error(f"No matching problems found for {problem_filter}")
            return
    
    # Group problems by area for better organization in output
    areas = set(problem['area'] for problem in problems)
    logger.info(f"Found {len(problems)} problems across {len(areas)} areas: {', '.join(areas)}")
    
    # Calculate per-problem sample counts
    problems_count = len(problems)
    malicious_per_problem = total_malicious // problems_count if problems_count > 0 else 0
    benign_per_problem = total_benign // problems_count if problems_count > 0 else 0
    
    # Handle remainder samples
    malicious_remainder = total_malicious % problems_count if problems_count > 0 else 0
    benign_remainder = total_benign % problems_count if problems_count > 0 else 0
    
    logger.info(f"Per-problem distribution: {malicious_per_problem} malicious, {benign_per_problem} benign")
    
    for i, problem in enumerate(problems):
        area = problem['area']
        nature = problem['nature']
        
        # Distribute remainder samples to first few problems
        current_malicious = malicious_per_problem + (1 if i < malicious_remainder else 0)
        current_benign = benign_per_problem + (1 if i < benign_remainder else 0)
        current_total = current_malicious + current_benign
        
        logger.info(f"\nGenerating {current_total} samples for {area}/{nature} (malicious: {current_malicious}, benign: {current_benign})...")
        
        try:
            # Generate samples
            samples = generate_for_problem(problem, current_malicious, current_benign)
            
            if not samples:
                logger.warning(f"No samples generated for {nature}, skipping to next problem")
                continue
            
            # Save samples
            save_samples(problem, samples)
            
        except Exception as e:
            logger.error(f"Error processing {nature}: {e}", exc_info=True)
            continue  # Continue with the next problem
    
    logger.info("Sample generation completed!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic cybersecurity data samples with configurable malicious/benign ratio")
    parser.add_argument('--count', type=int, default=10, help='Total number of samples to generate (default: 10)')
    parser.add_argument('--malicious-ratio', type=float, default=0.5, help='Ratio of malicious samples (0.0 to 1.0, default: 0.5)')
    parser.add_argument('--malicious-count', type=int, help='Specific number of malicious samples (overrides ratio)')
    parser.add_argument('--benign-count', type=int, help='Specific number of benign samples (overrides ratio)')
    parser.add_argument('--problems', nargs='+', help='Specific problem natures to generate samples for (optional)')
    
    args = parser.parse_args()
    
    # Handle specific count arguments
    if args.malicious_count is not None and args.benign_count is not None:
        total_count = args.malicious_count + args.benign_count
        malicious_ratio = args.malicious_count / total_count if total_count > 0 else 0.5
        logger.info(f"Using specific counts: malicious={args.malicious_count}, benign={args.benign_count}")
        logger.info(f"Calculated ratio: {malicious_ratio:.2f}")
        main(total_count, malicious_ratio, args.problems)
    else:
        logger.info(f"Starting generator.py with args: count={args.count}, malicious_ratio={args.malicious_ratio}, problems={args.problems}")
        main(args.count, args.malicious_ratio, args.problems)