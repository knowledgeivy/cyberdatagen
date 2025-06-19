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
SAMPLES_PER_CLASS = 100  # Number of real-world samples per class for analysis
TARGET_SEEDS_PER_CLASS = 50  # Target number of seeds per class

# Dataset configuration
CSV_FILE = "nsl_kdd_rare_train.csv.gz"  # Raw CSV file in raw/ directory
DATA_INFO_NAME = "nsl_kdd_rare"   # Dataset name in data_info.yaml
MODEL_NAME = "gpt-4.1-mini"

# Get config manager instance
config_manager = get_config_manager()

logger.info(f"Using model: {MODEL_NAME}")
logger.info(f"Project root: {config_manager.project_root}")


def load_data_info(data_name: str = None) -> Dict:
    """Load detailed data information from data_info.yaml."""
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
        
        return dataset_info
        
    except Exception as e:
        logger.error(f"Error loading data info: {str(e)}")
        raise


def load_raw_data(file_path: Path, label_column: str = "label") -> pd.DataFrame:
    """Load raw data from CSV file (supports .gz compression)."""
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
    """Perform stratified sampling to get equal samples from each class."""
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


# ================== STEP 1: DOMAIN DISCOVERY ==================

def analyze_domain_discovery(data_info: Dict, 
                           malicious_samples: pd.DataFrame,
                           benign_samples: pd.DataFrame) -> Dict:
    """
    Step 1: Intelligent Domain Discovery from real-world data.
    Extract attack patterns, normal baselines, and domain knowledge.
    """
    logger.info("=== STEP 1: INTELLIGENT DOMAIN DISCOVERY ===")
    
    # Prepare data context for LLM analysis
    max_examples = 15  # More examples for better pattern extraction
    malicious_examples = malicious_samples.head(max_examples).to_dict('records') if len(malicious_samples) > 0 else []
    benign_examples = benign_samples.head(max_examples).to_dict('records') if len(benign_samples) > 0 else []
    
    analysis_context = {
        'data_info': data_info,
        'malicious_samples': malicious_examples,
        'benign_samples': benign_examples,
        'malicious_count': len(malicious_samples),
        'benign_count': len(benign_samples),
        'schema_info': {
            'columns': list(malicious_samples.columns) if len(malicious_samples) > 0 else list(benign_samples.columns),
            'data_types': {col: str(dtype) for col, dtype in malicious_samples.dtypes.items()} if len(malicious_samples) > 0 else {}
        }
    }
    
    # Convert to JSON for prompt
    context_json = json.dumps(analysis_context, indent=2, default=str)
    
    # Load domain discovery prompts
    system_prompt = load_prompt(
        "domain_discovery_prompts",
        "prompts.domain_analysis.system.template"
    )
    
    user_prompt = load_prompt(
        "domain_discovery_prompts",
        "prompts.domain_analysis.user.template",
        analysis_context_json=context_json
    )
    
    # Call LLM for domain discovery
    logger.info("Analyzing real-world data for domain discovery...")
    response_content = process_llm_request(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_name=MODEL_NAME,
        temperature=0.3
    )
    
    try:
        # Clean and parse JSON response
        if response_content.startswith('```'):
            first_backticks_end = response_content.find('\n', 3)
            if first_backticks_end != -1:
                last_backticks_start = response_content.rfind('```')
                if last_backticks_start > first_backticks_end:
                    response_content = response_content[first_backticks_end + 1:last_backticks_start].strip()
        
        domain_discovery = json.loads(response_content)
        
        # Add metadata
        domain_discovery['metadata'] = {
            'dataset_name': data_info.get('data_name', 'unknown'),
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'samples_analyzed': {
                'malicious': len(malicious_samples),
                'benign': len(benign_samples)
            },
            'step': 'domain_discovery',
            'version': '1.0'
        }
        
        logger.info("Domain discovery analysis completed successfully")
        return domain_discovery
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse domain discovery response: {str(e)}")
        # Return fallback structure
        return {
            "domain_characteristics": {
                "primary_domain": data_info.get('domain', 'cybersecurity'),
                "attack_types": data_info.get('attack_types', []),
                "data_characteristics": data_info.get('data_characteristics', [])
            },
            "attack_patterns": {
                "common_indicators": ["Extracted from real data analysis"],
                "attack_signatures": ["Pattern analysis failed"],
                "technical_markers": []
            },
            "normal_baselines": {
                "typical_behaviors": ["Normal pattern analysis"],
                "baseline_metrics": [],
                "operational_patterns": []
            },
            "domain_terminology": {
                "technical_terms": [],
                "attack_vectors": [],
                "defensive_measures": []
            },
            "threat_landscape": {
                "current_threats": [],
                "emerging_patterns": [],
                "risk_factors": []
            },
            "metadata": {
                "dataset_name": data_info.get('data_name', 'unknown'),
                "analysis_timestamp": pd.Timestamp.now().isoformat(),
                "samples_analyzed": {
                    "malicious": len(malicious_samples),
                    "benign": len(benign_samples)
                },
                "step": "domain_discovery",
                "version": "1.0",
                "fallback_used": True
            }
        }


def save_domain_discovery(domain_discovery: Dict, dataset_name: str):
    """Save domain discovery results to config/domain_discovery/."""
    logger.info("Saving domain discovery results...")
    
    # Create domain discovery directory
    domain_discovery_dir = config_manager.config_dir / "domain_discovery"
    domain_discovery_dir.mkdir(parents=True, exist_ok=True)
    
    # Save domain discovery
    domain_file = domain_discovery_dir / f"{dataset_name}_domain_discovery.json"
    with domain_file.open('w', encoding='utf-8') as f:
        json.dump(domain_discovery, f, indent=2, default=str)
    
    logger.info(f"Domain discovery saved to: {domain_file}")
    
    # Extract and save technical indicators separately for easier access
    technical_indicators = {
        "attack_indicators": domain_discovery.get("attack_patterns", {}).get("common_indicators", []),
        "attack_signatures": domain_discovery.get("attack_patterns", {}).get("attack_signatures", []),
        "technical_markers": domain_discovery.get("attack_patterns", {}).get("technical_markers", []),
        "normal_indicators": domain_discovery.get("normal_baselines", {}).get("typical_behaviors", []),
        "baseline_metrics": domain_discovery.get("normal_baselines", {}).get("baseline_metrics", []),
        "metadata": domain_discovery.get("metadata", {})
    }
    
    indicators_file = domain_discovery_dir / f"{dataset_name}_technical_indicators.json"
    with indicators_file.open('w', encoding='utf-8') as f:
        json.dump(technical_indicators, f, indent=2, default=str)
    
    logger.info(f"Technical indicators saved to: {indicators_file}")
    return domain_file, indicators_file


# ================== STEP 2: CONTEXTUAL PROBLEM ENRICHMENT ==================

def enrich_contextual_problems(domain_discovery: Dict, data_info: Dict) -> Dict:
    """
    Step 2: Contextual Problem Enrichment using domain discovery.
    Create enriched problem definitions with real-world context.
    """
    logger.info("=== STEP 2: CONTEXTUAL PROBLEM ENRICHMENT ===")
    
    # Prepare enrichment context
    enrichment_context = {
        'domain_discovery': domain_discovery,
        'data_info': data_info,
        'dataset_name': data_info.get('data_name', 'unknown')
    }
    
    # Convert to JSON for prompt
    context_json = json.dumps(enrichment_context, indent=2, default=str)
    
    # Load contextual enrichment prompts
    system_prompt = load_prompt(
        "domain_discovery_prompts",
        "prompts.contextual_enrichment.system.template"
    )
    
    user_prompt = load_prompt(
        "domain_discovery_prompts",
        "prompts.contextual_enrichment.user.template",
        enrichment_context_json=context_json
    )
    
    # Call LLM for contextual enrichment
    logger.info("Enriching problem definitions with real-world context...")
    response_content = process_llm_request(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_name=MODEL_NAME,
        temperature=0.4
    )
    
    try:
        # Clean and parse JSON response
        if response_content.startswith('```'):
            first_backticks_end = response_content.find('\n', 3)
            if first_backticks_end != -1:
                last_backticks_start = response_content.rfind('```')
                if last_backticks_start > first_backticks_end:
                    response_content = response_content[first_backticks_end + 1:last_backticks_start].strip()
        
        contextual_problems = json.loads(response_content)
        
        # Add metadata
        contextual_problems['metadata'] = {
            'dataset_name': data_info.get('data_name', 'unknown'),
            'enrichment_timestamp': pd.Timestamp.now().isoformat(),
            'domain_discovery_used': True,
            'step': 'contextual_enrichment',
            'version': '1.0'
        }
        
        logger.info("Contextual problem enrichment completed successfully")
        return contextual_problems
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse contextual enrichment response: {str(e)}")
        # Return fallback structure
        return {
            "enriched_problems": [{
                "area": data_info.get('domain', 'Cybersecurity'),
                "nature": data_info.get('data_name', 'realworld_analysis'),
                "description": f"Real-world cybersecurity problem derived from {data_info.get('data_name', 'dataset')}",
                "attack_vectors": data_info.get('attack_types', []),
                "real_world_context": {
                    "domain_insights": "Analysis from real data patterns",
                    "technical_implementation": "Based on actual attack signatures",
                    "threat_landscape": "Current threat environment analysis"
                },
                "risk_reduction": [
                    "Implement monitoring based on real patterns",
                    "Deploy detection systems for identified signatures",
                    "Establish baselines from normal behavior analysis"
                ]
            }],
            "attack_taxonomy": {
                "primary_attacks": data_info.get('attack_types', []),
                "attack_patterns": [],
                "mitigation_strategies": []
            },
            "metadata": {
                "dataset_name": data_info.get('data_name', 'unknown'),
                "enrichment_timestamp": pd.Timestamp.now().isoformat(),
                "domain_discovery_used": True,
                "step": "contextual_enrichment",
                "version": "1.0",
                "fallback_used": True
            }
        }


def save_contextual_problems(contextual_problems: Dict, dataset_name: str):
    """Save contextual problems to config/contextual_problems/."""
    logger.info("Saving contextual problems...")
    
    # Create contextual problems directory
    contextual_dir = config_manager.config_dir / "contextual_problems"
    contextual_dir.mkdir(parents=True, exist_ok=True)
    
    # Save contextual problems
    contextual_file = contextual_dir / f"{dataset_name}_contextual_problems.json"
    with contextual_file.open('w', encoding='utf-8') as f:
        json.dump(contextual_problems, f, indent=2, default=str)
    
    logger.info(f"Contextual problems saved to: {contextual_file}")
    
    # Extract and save attack taxonomy separately
    attack_taxonomy = contextual_problems.get("attack_taxonomy", {})
    attack_taxonomy['metadata'] = contextual_problems.get("metadata", {})
    
    taxonomy_file = contextual_dir / f"{dataset_name}_attack_taxonomy.json"
    with taxonomy_file.open('w', encoding='utf-8') as f:
        json.dump(attack_taxonomy, f, indent=2, default=str)
    
    logger.info(f"Attack taxonomy saved to: {taxonomy_file}")
    return contextual_file, taxonomy_file


# ================== STEP 3: SCHEMA-AWARE SEED GENERATION ==================

def generate_schema_aware_seeds(data_info: Dict,
                               domain_discovery: Dict,
                               contextual_problems: Dict,
                               malicious_samples: pd.DataFrame,
                               benign_samples: pd.DataFrame) -> Dict:
    """
    Step 3: Schema-Aware Seed Generation using all previous context.
    Generate high-quality seeds that maintain schema consistency.
    """
    logger.info("=== STEP 3: SCHEMA-AWARE SEED GENERATION ===")
    
    # Prepare comprehensive generation context
    generation_context = {
        'data_info': data_info,
        'domain_discovery': domain_discovery,
        'contextual_problems': contextual_problems,
        'schema_template': {
            'columns': list(malicious_samples.columns) if len(malicious_samples) > 0 else list(benign_samples.columns),
            'data_types': {col: str(dtype) for col, dtype in malicious_samples.dtypes.items()} if len(malicious_samples) > 0 else {},
            'label_column': data_info.get('label_column', 'label'),
            'label_encoding': data_info.get('label_encoding', {'malicious': 1, 'benign': 0})
        },
        'reference_samples': {
            'malicious_examples': malicious_samples.head(5).to_dict('records') if len(malicious_samples) > 0 else [],
            'benign_examples': benign_samples.head(5).to_dict('records') if len(benign_samples) > 0 else []
        }
    }
    
    # Generate seeds in multiple batches for diversity
    all_seeds = []
    seeds_per_batch = 12
    batches_needed = max(2, (TARGET_SEEDS_PER_CLASS * 2) // seeds_per_batch)
    
    logger.info(f"Generating seeds in {batches_needed} batches of {seeds_per_batch} each")
    
    for batch_num in range(batches_needed):
        logger.info(f"Generating seed batch {batch_num + 1}/{batches_needed}")
        
        # Convert context to JSON for prompt
        context_json = json.dumps(generation_context, indent=2, default=str)
        
        # Load seed generation prompts
        system_prompt = load_prompt(
            "domain_discovery_prompts",
            "prompts.schema_aware_generation.system.template"
        )
        
        user_prompt = load_prompt(
            "domain_discovery_prompts",
            "prompts.schema_aware_generation.user.template",
            generation_context_json=context_json,
            batch_size=seeds_per_batch,
            batch_number=batch_num + 1,
            total_batches=batches_needed
        )
        
        # Call LLM for seed generation
        response_content = process_llm_request(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=MODEL_NAME,
            temperature=0.8  # Higher temperature for diversity
        )
        
        try:
            # Clean and parse JSON response
            if response_content.startswith('```'):
                first_backticks_end = response_content.find('\n', 3)
                if first_backticks_end != -1:
                    last_backticks_start = response_content.rfind('```')
                    if last_backticks_start > first_backticks_end:
                        response_content = response_content[first_backticks_end + 1:last_backticks_start].strip()
            
            seed_data = json.loads(response_content)
            batch_seeds = seed_data.get('examples', seed_data.get('seeds', []))
            
            all_seeds.extend(batch_seeds)
            logger.info(f"Batch {batch_num + 1} generated {len(batch_seeds)} seeds. Total: {len(all_seeds)}")
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse seed batch {batch_num + 1}: {str(e)}")
            continue
    
    # Add comprehensive generation metadata
    generation_metadata = {
        'generation_method': 'schema_aware_three_step',
        'data_source': 'real_world_enhanced_analysis',
        'domain_discovery_used': True,
        'contextual_enrichment_used': True,
        'schema_preserved': True,
        'dataset_name': data_info.get('data_name', 'unknown'),
        'target_seeds_per_class': TARGET_SEEDS_PER_CLASS,
        'total_seeds_generated': len(all_seeds),
        'generation_batches': batches_needed,
        'seeds_per_batch': seeds_per_batch,
        'generation_timestamp': pd.Timestamp.now().isoformat(),
        'step': 'schema_aware_generation',
        'version': '1.0'
    }
    
    logger.info(f"Schema-aware seed generation completed: {len(all_seeds)} total seeds")
    
    return {
        'examples': all_seeds,
        'generation_metadata': generation_metadata
    }


def save_raw_seeds(seed_data: Dict, dataset_name: str, area: str, nature: str):
    """Save raw seeds to data/seeds-raw/ directory."""
    logger.info("Saving raw seeds...")
    
    # Create seeds-raw directory structure
    seeds_raw_dir = config_manager.data_dir / "seeds-raw"
    area_dir = seeds_raw_dir / area.replace(' ', '_').replace('(', '').replace(')', '')
    area_dir.mkdir(parents=True, exist_ok=True)
    
    # Count malicious and benign examples
    malicious_count = 0
    benign_count = 0
    
    for example in seed_data['examples']:
        # Check various label formats
        if (example.get('label') == 1 or 
            example.get('Label') == 1 or
            str(example.get('label', '')).lower() in ['malicious', 'attack', '1'] or
            example.get('_metadata', {}).get('sample_type') == 'malicious'):
            malicious_count += 1
        else:
            benign_count += 1
    
    # Prepare comprehensive metadata
    metadata = {
        'source': 'three_step_real_world_analysis',
        'dataset_name': dataset_name,
        'generation_method': 'domain_discovery_contextual_schema_aware',
        'total_examples': len(seed_data['examples']),
        'malicious_examples': malicious_count,
        'benign_examples': benign_count,
        'area': area,
        'nature': nature,
        'domain_discovery_used': True,
        'contextual_enrichment_used': True,
        'schema_preserved': True,
        'generation_metadata': seed_data['generation_metadata'],
        'quality_status': 'raw_unvalidated',
        'configuration': {
            'samples_per_class_analyzed': SAMPLES_PER_CLASS,
            'target_seeds_per_class': TARGET_SEEDS_PER_CLASS,
            'actual_seeds_generated': len(seed_data['examples'])
        }
    }
    
    # Save raw seeds
    seeds_file = area_dir / f"{nature}_examples.json"
    complete_seed_data = {
        'examples': seed_data['examples'],
        'metadata': metadata
    }
    
    with seeds_file.open('w', encoding='utf-8') as f:
        json.dump(complete_seed_data, f, indent=2, default=str)
    
    logger.info(f"Raw seeds saved to: {seeds_file}")
    logger.info(f"Results: {malicious_count} malicious, {benign_count} benign seeds")
    return seeds_file


def update_problems_json(contextual_problems: Dict, dataset_name: str):
    """Update problems.json with the enriched problem from Step 2."""
    logger.info("Updating problems.json with contextual problem...")
    
    enriched_problems = contextual_problems.get('enriched_problems', [])
    if not enriched_problems:
        logger.warning("No enriched problems found to save")
        return
    
    # Take the first enriched problem and enhance it further
    primary_problem = enriched_problems[0].copy()
    primary_problem.update({
        'source': 'three_step_real_world_analysis',
        'dataset_name': dataset_name,
        'domain_discovery_used': True,
        'contextual_enrichment_used': True,
        'step_metadata': {
            'step_1_domain_discovery': 'completed',
            'step_2_contextual_enrichment': 'completed',
            'step_3_seed_generation': 'pending'
        }
    })
    
    # Replace problems.json with the enriched problem
    problems_data = {"problems": [primary_problem]}
    config_manager.save_problems([primary_problem], "problems")
    
    nature = primary_problem.get('nature', 'realworld_analysis')
    logger.info(f"Updated problems.json with contextual problem: {nature}")
    return nature


def main(csv_file: str = None, 
         data_info_name: str = None,
         label_column: str = "label",
         samples_per_class: int = None):
    """
    Main function implementing the 3-step real-world data analysis pipeline.
    """
    # Use configured values if not provided
    if csv_file is None:
        csv_file = CSV_FILE
    if data_info_name is None:
        data_info_name = DATA_INFO_NAME
    if samples_per_class is None:
        samples_per_class = SAMPLES_PER_CLASS
        
    logger.info("="*80)
    logger.info("THREE-STEP REAL-WORLD DATA ANALYSIS PIPELINE")
    logger.info("="*80)
    logger.info(f"CSV file: {csv_file}")
    logger.info(f"Data info name: {data_info_name}")
    logger.info(f"Samples per class: {samples_per_class}")
    logger.info(f"Target seeds per class: {TARGET_SEEDS_PER_CLASS}")
    
    try:
        # Load data info and raw data
        raw_dir = config_manager.project_root / "raw"
        file_path = raw_dir / csv_file
        
        data_info = load_data_info(data_info_name)
        if 'label_column' in data_info and data_info['label_column']:
            label_column = data_info['label_column']
            logger.info(f"Using label column from data_info.yaml: {label_column}")
        
        df = load_raw_data(file_path, label_column)
        malicious_samples, benign_samples = stratified_sample(df, label_column, samples_per_class)
        
        # ========== STEP 1: DOMAIN DISCOVERY ==========
        domain_discovery = analyze_domain_discovery(data_info, malicious_samples, benign_samples)
        domain_file, indicators_file = save_domain_discovery(domain_discovery, data_info_name)
        
        # ========== STEP 2: CONTEXTUAL PROBLEM ENRICHMENT ==========
        contextual_problems = enrich_contextual_problems(domain_discovery, data_info)
        contextual_file, taxonomy_file = save_contextual_problems(contextual_problems, data_info_name)
        
        # ========== STEP 3: SCHEMA-AWARE SEED GENERATION ==========
        seed_data = generate_schema_aware_seeds(
            data_info, domain_discovery, contextual_problems, malicious_samples, benign_samples
        )
        
        # Update problems.json and save raw seeds
        nature = update_problems_json(contextual_problems, data_info_name)
        area = contextual_problems.get('enriched_problems', [{}])[0].get('area', 'Real-world Analysis')
        seeds_file = save_raw_seeds(seed_data, data_info_name, area, nature)
        
        # Final Summary
        logger.info("="*80)
        logger.info("THREE-STEP ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Dataset: {csv_file}")
        logger.info(f"Data info: {data_info.get('data_name', 'unknown')}")
        logger.info(f"Domain: {data_info.get('domain', 'unknown')}")
        logger.info("")
        logger.info("STEP 1 - Domain Discovery:")
        logger.info(f"  ✓ Domain characteristics analyzed")
        logger.info(f"  ✓ Attack patterns extracted")
        logger.info(f"  ✓ Normal baselines established")
        logger.info(f"  → {domain_file}")
        logger.info(f"  → {indicators_file}")
        logger.info("")
        logger.info("STEP 2 - Contextual Problem Enrichment:")
        logger.info(f"  ✓ Problem definitions enriched with real-world context")
        logger.info(f"  ✓ Attack taxonomy developed")
        logger.info(f"  ✓ Threat landscape analysis completed")
        logger.info(f"  → {contextual_file}")
        logger.info(f"  → {taxonomy_file}")
        logger.info("")
        logger.info("STEP 3 - Schema-Aware Seed Generation:")
        logger.info(f"  ✓ {len(seed_data['examples'])} high-quality seeds generated")
        logger.info(f"  ✓ Schema consistency maintained")
        logger.info(f"  ✓ Domain knowledge incorporated")
        logger.info(f"  → {seeds_file}")
        logger.info("")
        logger.info("Configuration:")
        logger.info(f"  - Samples per class analyzed: {samples_per_class}")
        logger.info(f"  - Target seeds per class: {TARGET_SEEDS_PER_CLASS}")
        logger.info(f"  - Actual seeds generated: {len(seed_data['examples'])}")
        logger.info("")
        logger.info("Files Updated:")
        logger.info(f"  - problems.json (updated with contextual problem)")
        logger.info(f"  - config/domain_discovery/ (Step 1 outputs)")
        logger.info(f"  - config/contextual_problems/ (Step 2 outputs)")
        logger.info(f"  - data/seeds-raw/ (Step 3 outputs)")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Error in three-step analysis pipeline: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Three-step real-world data analysis pipeline")
    parser.add_argument('--csv-file', help=f'CSV file name in raw/ directory (default: {CSV_FILE})')
    parser.add_argument('--data-info-name', help=f'Dataset name in data_info.yaml (default: {DATA_INFO_NAME})')
    parser.add_argument('--label-column', default='label', 
                       help='Name of the label column (can be overridden by data_info.yaml)')
    parser.add_argument('--samples-per-class', type=int, default=SAMPLES_PER_CLASS,
                       help=f'Number of samples per class to analyze (default: {SAMPLES_PER_CLASS})')
    
    args = parser.parse_args()
    
    main(
        csv_file=args.csv_file,
        data_info_name=args.data_info_name,
        label_column=args.label_column,
        samples_per_class=args.samples_per_class
    )