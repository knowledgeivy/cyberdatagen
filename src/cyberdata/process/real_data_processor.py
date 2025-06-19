# cyberdata/process/real_data_processor.py

import gzip
import json
import os
import pandas as pd
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from dotenv import load_dotenv

# Add parent directory to path for imports
CURRENT_DIR = Path(__file__).parent
sys.path.append(str(CURRENT_DIR.parent))

from cyberdata.utils.config_manager import get_config_manager
from cyberdata.utils.llm_invoke import process_llm_request
from cyberdata.utils.logger_config import setup_logger
from cyberdata.utils.prompt_loader import load_prompt

# Set up logger
logger = setup_logger("cyberdata.scripts.real_data_processor")

# Load environment variables
load_dotenv()

# Configuration variables
SAMPLES_PER_CLASS = 100  # Number of real-world samples per class, default: 20
TARGET_SEEDS_PER_CLASS = 50  # Target number of seeds per class, default: 25

# Dataset configuration
CSV_FILE = "nsl_kdd_rare_train.csv.gz"  # Raw CSV file in raw/ directory
DATA_INFO_NAME = "nsl_kdd_rare"   # Dataset name in data_info.yaml
MODEL_NAME = "gpt-4.1-mini"

# Get config manager instance
config_manager = get_config_manager()

logger.info(f"Using model: {MODEL_NAME}")
logger.info(f"Project root: {config_manager.project_root}")
logger.info(f"Raw data directory: {config_manager.project_root / 'raw'}")
logger.info(f"Configured CSV file: {CSV_FILE}")
logger.info(f"Configured data info: {DATA_INFO_NAME}")
logger.info(f"Samples per class: {SAMPLES_PER_CLASS}")
logger.info(f"Target seeds per class: {TARGET_SEEDS_PER_CLASS}")


def load_data_info(data_name: str = None) -> Dict:
    """
    Load detailed data information from data_info.yaml.
    
    Args:
        data_name (str): Name of the dataset (uses DATA_INFO_NAME if None)
        
    Returns:
        Dict: Data information including schema, description, and characteristics
    """
    if data_name is None:
        data_name = DATA_INFO_NAME
        
    logger.info(f"Loading data information for: {data_name}")
    
    try:
        data_info_file = config_manager.config_dir / "data_info.yaml"
        
        if not data_info_file.exists():
            logger.error(f"Data info file not found: {data_info_file}")
            raise FileNotFoundError(f"Data info file not found: {data_info_file}")
        
        with data_info_file.open('r', encoding='utf-8') as f:
            data_info = yaml.safe_load(f)
        
        datasets = data_info.get('datasets', {})
        
        if data_name not in datasets:
            logger.error(f"Dataset '{data_name}' not found in data_info.yaml")
            logger.info(f"Available datasets: {list(datasets.keys())}")
            raise KeyError(f"Dataset '{data_name}' not found in data_info.yaml")
        
        dataset_info = datasets[data_name]
        logger.info(f"Successfully loaded data information for: {data_name}")
        logger.info(f"Domain: {dataset_info.get('domain', 'unknown')}")
        logger.info(f"Attack types: {dataset_info.get('attack_types', [])}")
        logger.info(f"Data characteristics: {dataset_info.get('data_characteristics', [])}")
        
        return dataset_info
        
    except Exception as e:
        logger.error(f"Error loading data info: {str(e)}")
        raise


def load_raw_data(file_path: Path, label_column: str = "label") -> pd.DataFrame:
    """
    Load raw data from CSV file (supports .gz compression).
    
    Args:
        file_path (Path): Path to the CSV file
        label_column (str): Name of the label column
        
    Returns:
        pd.DataFrame: Loaded data
    """
    logger.info(f"Loading raw data from: {file_path}")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {file_path}")
    
    try:
        if file_path.suffix == '.gz':
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                df = pd.read_csv(f)
        else:
            df = pd.read_csv(file_path)
        
        logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
        
        # Validate label column exists
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in data. Available columns: {list(df.columns)}")
        
        # Log label distribution
        label_counts = df[label_column].value_counts()
        logger.info(f"Label distribution: {dict(label_counts)}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading raw data: {str(e)}")
        raise


def stratified_sample(df: pd.DataFrame, 
                     label_column: str = "label", 
                     samples_per_class: int = SAMPLES_PER_CLASS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform stratified sampling to get equal samples from each class.
    
    Args:
        df (pd.DataFrame): Input dataframe
        label_column (str): Name of the label column
        samples_per_class (int): Number of samples per class
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (malicious_samples, benign_samples)
    """
    logger.info(f"Performing stratified sampling: {samples_per_class} samples per class")
    
    # Get samples for each class
    malicious_df = df[df[label_column] == 1]
    benign_df = df[df[label_column] == 0]
    
    logger.info(f"Available samples - Malicious: {len(malicious_df)}, Benign: {len(benign_df)}")
    
    # Sample from each class
    if len(malicious_df) < samples_per_class:
        logger.warning(f"Not enough malicious samples. Using all {len(malicious_df)} available.")
        malicious_samples = malicious_df
    else:
        malicious_samples = malicious_df.sample(n=samples_per_class, random_state=42)
    
    if len(benign_df) < samples_per_class:
        logger.warning(f"Not enough benign samples. Using all {len(benign_df)} available.")
        benign_samples = benign_df
    else:
        benign_samples = benign_df.sample(n=samples_per_class, random_state=42)
    
    logger.info(f"Sampled - Malicious: {len(malicious_samples)}, Benign: {len(benign_samples)}")
    
    return malicious_samples, benign_samples


def create_enhanced_data_context(data_info: Dict, 
                               malicious_samples: pd.DataFrame,
                               benign_samples: pd.DataFrame) -> Dict:
    """
    Create enhanced data context combining data_info.yaml with actual samples.
    
    Args:
        data_info (Dict): Data information from data_info.yaml
        malicious_samples (pd.DataFrame): Sample malicious data
        benign_samples (pd.DataFrame): Sample benign data
        
    Returns:
        Dict: Enhanced context for LLM prompts
    """
    logger.info("Creating enhanced data context")
    
    # Use more examples for context
    max_examples_per_class = min(10, max(len(malicious_samples), len(benign_samples)) // 2)
    malicious_examples = malicious_samples.head(max_examples_per_class).to_dict('records') if len(malicious_samples) > 0 else []
    benign_examples = benign_samples.head(max_examples_per_class).to_dict('records') if len(benign_samples) > 0 else []
    
    logger.info(f"Using {len(malicious_examples)} malicious and {len(benign_examples)} benign examples for context")
    
    # Create comprehensive context
    enhanced_context = {
        # From data_info.yaml
        'data_info': {
            'data_name': data_info.get('data_name', ''),
            'data_description': data_info.get('data_description', ''),
            'data_schema': data_info.get('data_schema', ''),
            'label_column': data_info.get('label_column', 'label'),
            'label_encoding': data_info.get('label_encoding', {}),
            'domain': data_info.get('domain', ''),
            'attack_types': data_info.get('attack_types', []),
            'features': data_info.get('features', []),
            'data_characteristics': data_info.get('data_characteristics', [])
        },
        
        # From actual data samples
        'sample_data': {
            'malicious_examples': malicious_examples,
            'benign_examples': benign_examples,
            'malicious_count': len(malicious_samples),
            'benign_count': len(benign_samples),
            'total_samples': len(malicious_samples) + len(benign_samples)
        },
        
        # Schema information
        'schema_info': {
            'columns': list(malicious_samples.columns) if len(malicious_samples) > 0 else list(benign_samples.columns),
            'column_count': len(malicious_samples.columns) if len(malicious_samples) > 0 else len(benign_samples.columns),
            'data_types': {col: str(dtype) for col, dtype in malicious_samples.dtypes.items()} if len(malicious_samples) > 0 else {}
        }
    }
    
    logger.info(f"Enhanced context created with {len(enhanced_context['sample_data']['malicious_examples'])} malicious and {len(enhanced_context['sample_data']['benign_examples'])} benign examples")
    
    return enhanced_context


def infer_problem_definition_from_data_info(enhanced_context: Dict) -> Dict:
    """
    Use LLM to infer cybersecurity problem definition from enhanced data context.
    
    Args:
        enhanced_context (Dict): Enhanced context with data_info and samples
        
    Returns:
        Dict: Inferred problem definition
    """
    logger.info("Inferring problem definition from enhanced data context")
    
    # Convert to JSON for prompt
    context_json = json.dumps(enhanced_context, indent=2, default=str)
    
    # Load prompts from YAML
    system_prompt = load_prompt(
        "real_data_analysis_prompts",
        "prompts.problem_inference_enhanced.system.template"
    )
    
    user_prompt = load_prompt(
        "real_data_analysis_prompts",
        "prompts.problem_inference_enhanced.user.template",
        enhanced_context_json=context_json
    )
    
    # Call LLM for problem inference
    logger.info("Calling LLM for enhanced problem definition inference")
    response_content = process_llm_request(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_name=MODEL_NAME,
        temperature=0.3
    )
    
    try:
        # Clean up and parse JSON response
        if response_content.startswith('```'):
            first_backticks_end = response_content.find('\n', 3)
            if first_backticks_end != -1:
                last_backticks_start = response_content.rfind('```')
                if last_backticks_start > first_backticks_end:
                    response_content = response_content[first_backticks_end + 1:last_backticks_start].strip()
        
        problem_definition = json.loads(response_content)
        logger.info(f"Inferred problem: {problem_definition.get('area', 'Unknown')}/{problem_definition.get('nature', 'Unknown')}")
        return problem_definition
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse problem definition: {str(e)}")
        # Return fallback based on data_info
        data_info = enhanced_context.get('data_info', {})
        return {
            "area": "Real-world Data",
            "nature": data_info.get('data_name', 'realworld_data'),
            "description": data_info.get('data_description', 'Problem inferred from real-world cybersecurity data'),
            "domain": data_info.get('domain', 'cybersecurity'),
            "attack_types": data_info.get('attack_types', []),
            "risk_reduction": ["Implement data validation", "Monitor for anomalies", "Deploy detection systems"]
        }


def generate_seed_examples_from_data_info(enhanced_context: Dict,
                                        problem_definition: Dict) -> Dict:
    """
    Generate seed examples using enhanced data context from data_info.yaml.
    
    Args:
        enhanced_context (Dict): Enhanced context with data_info and samples
        problem_definition (Dict): Inferred problem definition
        
    Returns:
        Dict: Generated seed examples with metadata
    """
    logger.info(f"Generating seed examples from enhanced data context (target: {TARGET_SEEDS_PER_CLASS} per class)")
    
    # Convert context to JSON for prompt
    context_json = json.dumps(enhanced_context, indent=2, default=str)
    problem_json = json.dumps(problem_definition, indent=2, default=str)
    
    # Generate seeds in multiple batches to reach target count
    all_seeds = []
    seeds_per_batch = 8
    batches_needed = (TARGET_SEEDS_PER_CLASS * 2) // seeds_per_batch + 1
    
    logger.info(f"Generating seeds in {batches_needed} batches of {seeds_per_batch} each")
    
    for batch_num in range(batches_needed):
        logger.info(f"Generating seed batch {batch_num + 1}/{batches_needed}")
        
        # Load prompts from YAML
        system_prompt = load_prompt(
            "real_data_analysis_prompts",
            "prompts.seed_generation_enhanced.system.template"
        )
        
        user_prompt = load_prompt(
            "real_data_analysis_prompts",
            "prompts.seed_generation_enhanced.user.template",
            enhanced_context_json=context_json,
            problem_definition_json=problem_json
        )
        
        # Add instruction for more seeds
        user_prompt += f"\n\nGenerate exactly {seeds_per_batch} seed examples in this batch (batch {batch_num + 1}/{batches_needed}). Include both malicious and benign examples with high diversity and technical accuracy."
        
        # Call LLM for seed generation
        response_content = process_llm_request(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=MODEL_NAME,
            temperature=0.8
        )
        
        try:
            # Clean up and parse JSON response
            if response_content.startswith('```'):
                first_backticks_end = response_content.find('\n', 3)
                if first_backticks_end != -1:
                    last_backticks_start = response_content.rfind('```')
                    if last_backticks_start > first_backticks_end:
                        response_content = response_content[first_backticks_end + 1:last_backticks_start].strip()
            
            seed_data = json.loads(response_content)
            
            # Ensure we have the expected structure
            if 'examples' not in seed_data:
                seed_data = {'examples': seed_data}
            
            batch_seeds = seed_data.get('examples', [])
            all_seeds.extend(batch_seeds)
            logger.info(f"Batch {batch_num + 1} generated {len(batch_seeds)} seeds. Total so far: {len(all_seeds)}")
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse seed batch {batch_num + 1}: {str(e)}")
            continue
    
    # Add generation metadata
    generation_metadata = {
        'data_source': 'enhanced_real_world_analysis',
        'data_info_used': True,
        'schema_preserved': True,
        'original_data_info': enhanced_context['data_info'],
        'generation_method': 'llm_with_data_info_multi_batch',
        'schema_template': enhanced_context['schema_info'],
        'target_seeds_per_class': TARGET_SEEDS_PER_CLASS,
        'total_seeds_generated': len(all_seeds),
        'batches_used': batches_needed,
        'seeds_per_batch': seeds_per_batch
    }
    
    logger.info(f"Generated {len(all_seeds)} total seeds (target was {TARGET_SEEDS_PER_CLASS * 2})")
    
    return {
        'examples': all_seeds,
        'generation_metadata': generation_metadata
    }


def create_fallback_seed_examples(enhanced_context: Dict, problem_definition: Dict) -> Dict:
    """
    Create fallback seed examples from actual data samples.
    
    Args:
        enhanced_context (Dict): Enhanced context
        problem_definition (Dict): Problem definition
        
    Returns:
        Dict: Fallback seed examples
    """
    logger.warning(f"Creating fallback seed examples from actual samples (target: {TARGET_SEEDS_PER_CLASS} per class)")
    
    sample_data = enhanced_context.get('sample_data', {})
    malicious_examples = sample_data.get('malicious_examples', [])
    benign_examples = sample_data.get('benign_examples', [])
    
    all_examples = []
    
    # Replicate examples to reach target count
    target_malicious = TARGET_SEEDS_PER_CLASS
    target_benign = TARGET_SEEDS_PER_CLASS
    
    # Add malicious examples with metadata (replicate if needed)
    for i in range(target_malicious):
        if malicious_examples:
            source_idx = i % len(malicious_examples)
            enhanced_example = malicious_examples[source_idx].copy()
            enhanced_example['_metadata'] = {
                'sample_type': 'malicious',
                'source': 'real_world_fallback',
                'enhanced_context_available': True,
                'replication_id': i,
                'source_sample_idx': source_idx
            }
            all_examples.append(enhanced_example)
    
    # Add benign examples with metadata (replicate if needed)
    for i in range(target_benign):
        if benign_examples:
            source_idx = i % len(benign_examples)
            enhanced_example = benign_examples[source_idx].copy()
            enhanced_example['_metadata'] = {
                'sample_type': 'benign',
                'source': 'real_world_fallback',
                'enhanced_context_available': True,
                'replication_id': i,
                'source_sample_idx': source_idx
            }
            all_examples.append(enhanced_example)
    
    generation_metadata = {
        'data_source': 'fallback_from_samples',
        'data_info_used': True,
        'schema_preserved': True,
        'original_data_info': enhanced_context.get('data_info', {}),
        'generation_method': 'fallback_direct_samples_with_replication',
        'schema_template': enhanced_context.get('schema_info', {}),
        'target_seeds_per_class': TARGET_SEEDS_PER_CLASS,
        'fallback_replication_applied': True
    }
    
    logger.info(f"Generated {len(all_examples)} fallback seeds from {len(malicious_examples)} malicious and {len(benign_examples)} benign source samples")
    
    return {
        'examples': all_examples,
        'generation_metadata': generation_metadata
    }


def save_realworld_problem(problem_definition: Dict, enhanced_context: Dict) -> str:
    """
    Save the inferred problem definition to problems.json (replacing any existing content).
    
    Args:
        problem_definition (Dict): Problem definition
        enhanced_context (Dict): Enhanced context with data info
        
    Returns:
        str: Problem nature identifier
    """
    logger.info("Saving real-world problem definition (replacing existing problems)")
    
    # Create new problems list with only the real-world problem
    enhanced_problem = problem_definition.copy()
    enhanced_problem['source'] = 'real_world_data_with_info'
    enhanced_problem['data_info_used'] = True
    enhanced_problem['original_data_info'] = enhanced_context.get('data_info', {})
    enhanced_problem['schema_info'] = enhanced_context.get('schema_info', {})
    enhanced_problem['samples_per_class_used'] = SAMPLES_PER_CLASS
    enhanced_problem['target_seeds_per_class'] = TARGET_SEEDS_PER_CLASS
    
    # Save as the only problem (replace existing)
    new_problems = [enhanced_problem]
    config_manager.save_problems(new_problems, "problems")
    
    nature = problem_definition.get('nature', 'realworld_data')
    logger.info(f"Replaced problems.json with real-world problem: {nature}")
    
    return nature


def save_seed_examples(problem_definition: Dict, seed_data: Dict):
    """
    Save seed examples using the existing config manager structure with enhanced metadata.
    
    Args:
        problem_definition (Dict): Problem definition
        seed_data (Dict): Generated seed data with examples and metadata
    """
    area = problem_definition.get('area', 'Real-world Data')
    nature = problem_definition.get('nature', 'realworld_data')
    
    # Get file path using config manager
    file_path = config_manager.get_seeds_file(area, nature)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Count malicious and benign examples
    malicious_count = sum(1 for ex in seed_data['examples'] 
                         if ex.get('_metadata', {}).get('sample_type') == 'malicious' or
                            any(str(val).lower() in ['malicious', 'attack', '1'] for val in ex.values()))
    benign_count = sum(1 for ex in seed_data['examples'] 
                      if ex.get('_metadata', {}).get('sample_type') == 'benign' or
                         any(str(val).lower() in ['benign', 'normal', '0'] for val in ex.values()))
    
    # Prepare enhanced metadata
    metadata = {
        'source': 'real_world_data_with_enhanced_info',
        'data_info_used': True,
        'total_examples': len(seed_data['examples']),
        'malicious_examples': malicious_count,
        'benign_examples': benign_count,
        'problem_definition': problem_definition,
        'schema_preserved': True,
        'generation_metadata': seed_data['generation_metadata'],
        'enhancement_method': 'data_info_yaml_integration',
        'configuration': {
            'samples_per_class': SAMPLES_PER_CLASS,
            'target_seeds_per_class': TARGET_SEEDS_PER_CLASS,
            'actual_seeds_generated': len(seed_data['examples'])
        }
    }
    
    # Save seed examples with enhanced metadata
    complete_seed_data = {
        'examples': seed_data['examples'],
        'metadata': metadata
    }
    
    with file_path.open('w', encoding='utf-8') as f:
        json.dump(complete_seed_data, f, indent=2, default=str)
    
    logger.info(f"Saved {len(seed_data['examples'])} enhanced seed examples to {file_path}")
    logger.info(f"Results: {malicious_count} malicious, {benign_count} benign seeds")
    logger.info(f"Enhanced with data_info.yaml context and schema preservation")


def main(csv_file: str = None, 
         data_info_name: str = None,
         label_column: str = "label",
         samples_per_class: int = None):
    """
    Main function to process real-world data using enhanced data_info.yaml context.
    
    Args:
        csv_file (str): Name of the CSV file in raw/ directory (uses CSV_FILE if None)
        data_info_name (str): Dataset name in data_info.yaml (uses DATA_INFO_NAME if None)
        label_column (str): Name of the label column
        samples_per_class (int): Number of samples per class (uses SAMPLES_PER_CLASS if None)
    """
    # Use configured values if not provided
    if csv_file is None:
        csv_file = CSV_FILE
    if data_info_name is None:
        data_info_name = DATA_INFO_NAME
    if samples_per_class is None:
        samples_per_class = SAMPLES_PER_CLASS
        
    logger.info(f"Starting enhanced real-world data processing")
    logger.info(f"CSV file: {csv_file}")
    logger.info(f"Data info name: {data_info_name}")
    logger.info(f"Samples per class: {samples_per_class}")
    logger.info(f"Target seeds per class: {TARGET_SEEDS_PER_CLASS}")
    
    try:
        # Define file path
        raw_dir = config_manager.project_root / "raw"
        file_path = raw_dir / csv_file
        
        # Step 1: Load data info
        data_info = load_data_info(data_info_name)
        
        # Use label column from data_info if available
        if 'label_column' in data_info and data_info['label_column']:
            label_column = data_info['label_column']
            logger.info(f"Using label column from data_info.yaml: {label_column}")
        
        # Step 2: Load raw data
        df = load_raw_data(file_path, label_column)
        
        # Step 3: Stratified sampling
        malicious_samples, benign_samples = stratified_sample(
            df, label_column, samples_per_class
        )
        
        # Step 4: Create enhanced data context
        enhanced_context = create_enhanced_data_context(
            data_info, malicious_samples, benign_samples
        )
        
        # Step 5: Infer problem definition using enhanced context
        problem_definition = infer_problem_definition_from_data_info(enhanced_context)
        
        # Step 6: Generate seed examples using enhanced context
        seed_data = generate_seed_examples_from_data_info(
            enhanced_context, problem_definition
        )
        
        # Step 7: Save problem definition with enhanced context (replace existing)
        nature = save_realworld_problem(problem_definition, enhanced_context)
        
        # Step 8: Save enhanced seed examples
        save_seed_examples(problem_definition, seed_data)
        
        # Summary
        logger.info("="*60)
        logger.info("ENHANCED REAL-WORLD DATA PROCESSING COMPLETED")
        logger.info("="*60)
        logger.info(f"Dataset: {csv_file}")
        logger.info(f"Data info used: {data_info.get('data_name', 'unknown')}")
        logger.info(f"Domain: {data_info.get('domain', 'unknown')}")
        logger.info(f"Attack types: {', '.join(data_info.get('attack_types', []))}")
        logger.info(f"Problem: {problem_definition.get('area', 'Unknown')}/{nature}")
        logger.info(f"Configuration:")
        logger.info(f"  - Samples per class used: {samples_per_class}")
        logger.info(f"  - Target seeds per class: {TARGET_SEEDS_PER_CLASS}")
        logger.info(f"  - Actual seeds generated: {len(seed_data['examples'])}")
        logger.info(f"Enhancement method: data_info.yaml integration")
        logger.info(f"Files updated:")
        logger.info(f"  - problems.json (replaced with real-world problem)")
        logger.info(f"  - seeds/{problem_definition.get('area', 'Unknown')}/{nature}_examples.json")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Error in enhanced processing: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Process real-world data using enhanced data_info.yaml context")
    parser.add_argument('--csv-file', help=f'CSV file name in raw/ directory (default: {CSV_FILE})')
    parser.add_argument('--data-info-name', help=f'Dataset name in data_info.yaml (default: {DATA_INFO_NAME})')
    parser.add_argument('--label-column', default='label', 
                       help='Name of the label column (can be overridden by data_info.yaml)')
    parser.add_argument('--samples-per-class', type=int, default=SAMPLES_PER_CLASS,
                       help=f'Number of samples per class to extract (default: {SAMPLES_PER_CLASS})')
    
    args = parser.parse_args()
    
    main(
        csv_file=args.csv_file,
        data_info_name=args.data_info_name,
        label_column=args.label_column,
        samples_per_class=args.samples_per_class
    )