# cyberdata/process/realworld_processor.py

import gzip
import json
import os
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

# Add parent directory to path for imports
CURRENT_DIR = Path(__file__).parent
sys.path.append(str(CURRENT_DIR.parent))

from cyberdata.utils.config_manager import get_config_manager
from cyberdata.utils.llm_invoke import process_llm_request
from cyberdata.utils.logger_config import setup_logger
from cyberdata.utils.prompt_loader import load_prompt

# Set up logger
logger = setup_logger("cyberdata.scripts.realworld_sampler")

# Load environment variables
load_dotenv()

# Model configuration
MODEL_NAME = "gpt-4.1-mini"

# Get config manager instance
config_manager = get_config_manager()

logger.info(f"Using model: {MODEL_NAME}")
logger.info(f"Project root: {config_manager.project_root}")
logger.info(f"Raw data directory: {config_manager.project_root / 'raw'}")


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
                     samples_per_class: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def analyze_data_schema(df: pd.DataFrame, label_column: str = "label") -> Dict:
    """
    Analyze the data schema and patterns using LLM with focus on preserving original structure.
    
    Args:
        df (pd.DataFrame): Input dataframe
        label_column (str): Name of the label column
        
    Returns:
        Dict: Schema analysis results
    """
    logger.info("Analyzing data schema with LLM for schema preservation")
    
    # Prepare comprehensive data summary for LLM
    schema_info = {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "shape": df.shape,
        "label_column": label_column,
        "label_distribution": df[label_column].value_counts().to_dict() if label_column in df.columns else {},
        "sample_records": {
            "malicious_examples": df[df[label_column] == 1].head(2).to_dict('records') if label_column in df.columns else [],
            "benign_examples": df[df[label_column] == 0].head(2).to_dict('records') if label_column in df.columns else []
        },
        "column_statistics": {}
    }
    
    # Add column-level statistics
    for col in df.columns:
        col_data = df[col]
        stats = {
            "dtype": str(col_data.dtype),
            "null_count": int(col_data.isnull().sum()),
            "unique_count": int(col_data.nunique()),
            "sample_values": col_data.dropna().head(3).tolist()
        }
        
        # Add numeric statistics if applicable
        if col_data.dtype in ['int64', 'float64', 'int32', 'float32']:
            stats.update({
                "min": float(col_data.min()) if not col_data.empty else None,
                "max": float(col_data.max()) if not col_data.empty else None,
                "mean": float(col_data.mean()) if not col_data.empty else None
            })
        
        schema_info["column_statistics"][col] = stats
    
    # Convert to JSON for prompt
    schema_json = json.dumps(schema_info, indent=2, default=str)
    
    # Load prompts from YAML
    system_prompt = load_prompt(
        "realworld_analysis_prompts",
        "prompts.schema_preservation.system.template"
    )
    
    user_prompt = load_prompt(
        "realworld_analysis_prompts",
        "prompts.schema_preservation.user.template",
        schema_json=schema_json
    )
    
    # Call LLM for schema analysis
    logger.info("Calling LLM for schema preservation analysis")
    response_content = process_llm_request(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_name=MODEL_NAME,
        temperature=0.1  # Low temperature for consistent schema analysis
    )
    
    try:
        # Clean up and parse JSON response
        if response_content.startswith('```'):
            first_backticks_end = response_content.find('\n', 3)
            if first_backticks_end != -1:
                last_backticks_start = response_content.rfind('```')
                if last_backticks_start > first_backticks_end:
                    response_content = response_content[first_backticks_end + 1:last_backticks_start].strip()
        
        schema_analysis = json.loads(response_content)
        schema_analysis['original_columns'] = list(df.columns)  # Preserve original column order
        schema_analysis['label_column'] = label_column
        logger.info("Successfully parsed schema preservation analysis")
        return schema_analysis
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse schema analysis: {str(e)}")
        # Return basic schema preserving the original structure
        return {
            "domain": "cybersecurity",
            "data_type": "tabular_data",
            "original_columns": list(df.columns),
            "label_column": label_column,
            "preserve_schema": True,
            "key_fields": list(df.columns),
            "generation_instructions": {
                "maintain_column_names": True,
                "maintain_data_types": True,
                "maintain_label_encoding": True
            }
        }


def infer_problem_definition(malicious_samples: pd.DataFrame, 
                           benign_samples: pd.DataFrame,
                           schema_analysis: Dict) -> Dict:
    """
    Use LLM to infer cybersecurity problem definition from real data.
    
    Args:
        malicious_samples (pd.DataFrame): Malicious data samples
        benign_samples (pd.DataFrame): Benign data samples
        schema_analysis (Dict): Schema analysis results
        
    Returns:
        Dict: Inferred problem definition
    """
    logger.info("Inferring problem definition from real data")
    
    # Prepare sample data for LLM analysis
    analysis_data = {
        "schema_analysis": schema_analysis,
        "malicious_examples": malicious_samples.head(3).to_dict('records'),
        "benign_examples": benign_samples.head(3).to_dict('records'),
        "malicious_count": len(malicious_samples),
        "benign_count": len(benign_samples)
    }
    
    analysis_json = json.dumps(analysis_data, indent=2, default=str)
    
    # Load prompts from XML
    system_prompt = load_prompt(
        "realworld_analysis_prompts",
        "prompts.problem_inference.system.template"
    )
    
    user_prompt = load_prompt(
        "realworld_analysis_prompts",
        "prompts.problem_inference.user.template",
        analysis_json=analysis_json
    )
    
    # Call LLM for problem inference
    logger.info("Calling LLM for problem definition inference")
    response_content = process_llm_request(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_name=MODEL_NAME,
        temperature=0.5
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
        # Return fallback problem definition
        return {
            "area": "Real-world Data",
            "nature": "data_analysis",
            "description": "Problem inferred from real-world cybersecurity data",
            "risk_reduction": ["Implement data validation", "Monitor for anomalies"]
        }


def generate_seed_examples(malicious_samples: pd.DataFrame,
                         benign_samples: pd.DataFrame,
                         problem_definition: Dict,
                         schema_analysis: Dict) -> List[Dict]:
    """
    Generate seed examples that preserve the original data schema.
    
    Args:
        malicious_samples (pd.DataFrame): Malicious data samples
        benign_samples (pd.DataFrame): Benign data samples
        problem_definition (Dict): Inferred problem definition
        schema_analysis (Dict): Schema analysis results
        
    Returns:
        List[Dict]: Generated seed examples maintaining original schema
    """
    logger.info("Generating schema-preserving seed examples")
    
    # Create schema template for synthetic generation
    all_data = pd.concat([malicious_samples, benign_samples], ignore_index=True)
    schema_template = extract_schema_template(all_data, schema_analysis)
    
    # Create seed examples that include both original samples and generation instructions
    seed_examples = create_schema_preserving_seeds(
        malicious_samples, benign_samples, schema_analysis, problem_definition
    )
    
    # Add generation instructions for synthetic data that will maintain schema
    generation_metadata = {
        'schema_template': schema_template,
        'generation_instructions': {
            'preserve_column_names': True,
            'preserve_data_types': True,
            'preserve_label_encoding': True,
            'maintain_statistical_properties': True,
            'column_relationships': schema_analysis.get('column_relationships', {}),
            'value_constraints': schema_analysis.get('value_constraints', {})
        }
    }
    
    return {
        'examples': seed_examples,
        'generation_metadata': generation_metadata
    }


def create_schema_preserving_seeds(malicious_samples: pd.DataFrame,
                                  benign_samples: pd.DataFrame,
                                  schema_analysis: Dict,
                                  problem_definition: Dict) -> List[Dict]:
    """
    Create seed examples that preserve the original data schema exactly.
    
    Args:
        malicious_samples (pd.DataFrame): Malicious data samples
        benign_samples (pd.DataFrame): Benign data samples
        schema_analysis (Dict): Schema analysis results
        problem_definition (Dict): Inferred problem definition
        
    Returns:
        List[Dict]: Seed examples maintaining original schema
    """
    logger.info("Creating schema-preserving seed examples")
    
    all_samples = []
    
    # Process malicious samples
    for idx, row in malicious_samples.iterrows():
        sample = row.to_dict()
        # Add metadata while preserving original schema
        sample['_metadata'] = {
            'sample_type': 'malicious',
            'source': 'real_world_data',
            'original_index': idx
        }
        all_samples.append(sample)
    
    # Process benign samples
    for idx, row in benign_samples.iterrows():
        sample = row.to_dict()
        # Add metadata while preserving original schema
        sample['_metadata'] = {
            'sample_type': 'benign',
            'source': 'real_world_data',
            'original_index': idx
        }
        all_samples.append(sample)
    
    logger.info(f"Created {len(all_samples)} schema-preserving seed examples")
    return all_samples


def extract_schema_template(df: pd.DataFrame, schema_analysis: Dict) -> Dict:
    """
    Extract a schema template for generating new data with the same structure.
    
    Args:
        df (pd.DataFrame): Original dataframe
        schema_analysis (Dict): Schema analysis from LLM
        
    Returns:
        Dict: Schema template for generation
    """
    logger.info("Extracting schema template for synthetic generation")
    
    # Get column information
    columns_info = {}
    for col in df.columns:
        col_data = df[col]
        columns_info[col] = {
            'dtype': str(col_data.dtype),
            'sample_values': col_data.dropna().head(5).tolist(),
            'null_count': col_data.isnull().sum(),
            'unique_count': col_data.nunique(),
            'is_categorical': col_data.dtype == 'object' or col_data.nunique() < 20
        }
        
        # Add statistical info for numeric columns
        if col_data.dtype in ['int64', 'float64']:
            columns_info[col].update({
                'min': float(col_data.min()) if not col_data.empty else None,
                'max': float(col_data.max()) if not col_data.empty else None,
                'mean': float(col_data.mean()) if not col_data.empty else None,
                'std': float(col_data.std()) if not col_data.empty else None
            })
    
    schema_template = {
        'columns': list(df.columns),
        'column_details': columns_info,
        'total_records': len(df),
        'schema_analysis': schema_analysis,
        'label_column': schema_analysis.get('label_column', 'label'),
        'label_mapping': {
            'malicious': 1,
            'benign': 0
        }
    }
    
    return schema_template


def save_realworld_problem(problem_definition: Dict, schema_analysis: Dict) -> str:
    """
    Save the inferred problem definition to problems.json.
    
    Args:
        problem_definition (Dict): Problem definition
        schema_analysis (Dict): Schema analysis
        
    Returns:
        str: Problem nature identifier
    """
    logger.info("Saving real-world problem definition")
    
    # Load existing problems
    try:
        existing_problems = config_manager.load_problems(prefer_updated=False)
    except FileNotFoundError:
        existing_problems = []
    
    # Add metadata to problem definition
    enhanced_problem = problem_definition.copy()
    enhanced_problem['source'] = 'real_world_data'
    enhanced_problem['schema_analysis'] = schema_analysis
    
    # Add to existing problems
    existing_problems.append(enhanced_problem)
    
    # Save updated problems
    config_manager.save_problems(existing_problems, "problems")
    
    nature = problem_definition.get('nature', 'realworld_data')
    logger.info(f"Added real-world problem: {nature}")
    
    return nature


def save_seed_examples(problem_definition: Dict, seed_data: Dict):
    """
    Save seed examples using the existing config manager structure with schema preservation.
    
    Args:
        problem_definition (Dict): Problem definition
        seed_data (Dict): Generated seed data with examples and metadata
    """
    area = problem_definition.get('area', 'Real-world Data')
    nature = problem_definition.get('nature', 'realworld_data')
    
    # Get file path using config manager
    file_path = config_manager.get_seeds_file(area, nature)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare enhanced metadata
    metadata = {
        'source': 'real_world_data',
        'total_examples': len(seed_data['examples']),
        'malicious_examples': sum(1 for ex in seed_data['examples'] if ex.get('_metadata', {}).get('sample_type') == 'malicious'),
        'benign_examples': sum(1 for ex in seed_data['examples'] if ex.get('_metadata', {}).get('sample_type') == 'benign'),
        'problem_definition': problem_definition,
        'schema_preserved': True,
        'generation_metadata': seed_data['generation_metadata']
    }
    
    # Save seed examples with schema preservation
    complete_seed_data = {
        'examples': seed_data['examples'],
        'metadata': metadata
    }
    
    with file_path.open('w', encoding='utf-8') as f:
        json.dump(complete_seed_data, f, indent=2, default=str)
    
    logger.info(f"Saved {len(seed_data['examples'])} schema-preserving seed examples to {file_path}")
    logger.info(f"Schema template and generation instructions included for synthetic data generation")


def main(csv_file: str = "five_email_phishing.csv.gz", 
         label_column: str = "label",
         samples_per_class: int = 10):
    """
    Main function to process real-world data and generate seed examples.
    
    Args:
        csv_file (str): Name of the CSV file in raw/ directory
        label_column (str): Name of the label column
        samples_per_class (int): Number of samples per class to extract
    """
    logger.info(f"Starting real-world data processing for: {csv_file}")
    
    try:
        # Define file path
        raw_dir = config_manager.project_root / "raw"
        file_path = raw_dir / csv_file
        
        # Step 1: Load raw data
        df = load_raw_data(file_path, label_column)
        
        # Step 2: Stratified sampling
        malicious_samples, benign_samples = stratified_sample(
            df, label_column, samples_per_class
        )
        
        # Step 3: Analyze data schema with preservation focus
        schema_analysis = analyze_data_schema(df, label_column)
        
        # Step 4: Infer problem definition
        problem_definition = infer_problem_definition(
            malicious_samples, benign_samples, schema_analysis
        )
        
        # Step 5: Generate schema-preserving seed examples
        seed_data = generate_seed_examples(
            malicious_samples, benign_samples, problem_definition, schema_analysis
        )
        
        # Step 6: Save problem definition
        nature = save_realworld_problem(problem_definition, schema_analysis)
        
        # Step 7: Save schema-preserving seed examples
        save_seed_examples(problem_definition, seed_data)
        
        # Summary
        logger.info("="*60)
        logger.info("REAL-WORLD DATA PROCESSING COMPLETED")
        logger.info("="*60)
        logger.info(f"Dataset: {csv_file}")
        logger.info(f"Problem: {problem_definition.get('area', 'Unknown')}/{nature}")
        logger.info(f"Seed examples generated: {len(seed_data['examples'])}")
        logger.info(f"Files updated:")
        logger.info(f"  - problems.json (added real-world problem)")
        logger.info(f"  - seeds/{problem_definition.get('area', 'Unknown')}/{nature}_examples.json")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Process real-world data for seed generation")
    parser.add_argument('--csv-file', default='five_email_phishing.csv.gz', 
                       help='CSV file name in raw/ directory')
    parser.add_argument('--label-column', default='label', 
                       help='Name of the label column')
    parser.add_argument('--samples-per-class', type=int, default=5,
                       help='Number of samples per class to extract')
    
    args = parser.parse_args()
    
    main(
        csv_file=args.csv_file,
        label_column=args.label_column,
        samples_per_class=args.samples_per_class
    )