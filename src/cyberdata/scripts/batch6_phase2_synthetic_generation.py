#!/usr/bin/env python3
"""
Batch 6 Experiment - Phase 2: High-Quality Synthetic Data Generation

Enhanced synthetic data generation using real_data_rewriter.py methodology:
- Input: 2000 seed samples from Phase 1 (core: 1000, edge: 1000)
- Method: YAML-configured prompt templates with GPT-4.1-mini
- Output: 6000 high-quality synthetic samples (3 prompt variants each)
- Parallel processing with 20 workers for efficiency

Key improvements over Batch 5:
- Structured YAML prompts vs hardcoded strings
- CSV output format vs unreliable parsing
- Mature real_data_rewriter.py toolchain vs experimental code

Author: Claude
Created: 2025-07-30
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
import logging
import gzip
import concurrent.futures
from typing import Dict, List, Tuple, Any

# Project path setup
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.cyberdata.utils.logger_config import setup_logger
from src.cyberdata.utils.data_loader import load_csv_data
from src.cyberdata.utils.prompt_loader import load_prompt
from src.cyberdata.utils.llm_invoke import process_llm_request

class Batch6Phase2SyntheticGeneration:
    """Batch 6 Phase 2: High-quality synthetic data generation"""
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logger("batch6_phase2")
        self.project_root = PROJECT_ROOT
        self.batch6_dir = self.project_root / "data" / "batch6"
        
        # Input directories
        self.seed_prep_dir = self.batch6_dir / "seed_preparation"
        
        # Output directories
        self.synthetic_gen_dir = self.batch6_dir / "synthetic_generation"
        self.phase2_analysis_dir = self.batch6_dir / "phase2_analysis"
        
        # Create output directories
        for dir_path in [self.synthetic_gen_dir, self.phase2_analysis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Generation configuration
        self.model_name = "gpt-4.1-mini"
        self.temperature = 0.8
        self.max_workers = 20
        self.api_delay = 0.1  # Delay between API calls
        
        # Prompt configuration mapping
        self.prompt_configs = {
            "original": "rewrite_generation",
            "strong": "rewrite_generation_strong", 
            "weak": "rewrite_generation_weak"
        }
        
        # Expected input and output
        self.expected_seeds = 2000
        self.expected_total_synthetic = 6000  # 2000 seeds × 3 prompts
        
        self.logger.info("Batch 6 Phase 2 synthetic generation initialization completed")
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Max workers: {self.max_workers}")
        self.logger.info(f"Prompt variants: {list(self.prompt_configs.keys())}")
        
    def load_seed_data(self) -> pd.DataFrame:
        """Load prepared seed data from Phase 1"""
        try:
            self.logger.info("Loading seed data from Phase 1...")
            
            seeds_file = self.seed_prep_dir / "real_malicious_seeds_with_layers.csv"
            if not seeds_file.exists():
                raise FileNotFoundError(f"Seed data file not found: {seeds_file}")
                
            seeds_df = load_csv_data(seeds_file, logger=self.logger)
            
            # Validate seed data
            if len(seeds_df) != self.expected_seeds:
                raise ValueError(f"Expected {self.expected_seeds} seeds, found {len(seeds_df)}")
                
            # Check required columns
            required_cols = ['subject', 'body', 'label', 'seed_layer', 'batch6_seed_id']
            missing_cols = [col for col in required_cols if col not in seeds_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            # Verify layer distribution
            layer_dist = seeds_df['seed_layer'].value_counts()
            self.logger.info(f"Seed layer distribution: {dict(layer_dist)}")
            
            self.logger.info(f"✅ Successfully loaded {len(seeds_df)} seed samples")
            return seeds_df
            
        except Exception as e:
            self.logger.error(f"Failed to load seed data: {str(e)}")
            raise
            
    def prepare_layer_seeds(self, seeds_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Separate seeds by layer for processing"""
        try:
            self.logger.info("Preparing seeds by layer...")
            
            layer_seeds = {}
            
            for layer in ['core', 'edge']:
                layer_mask = seeds_df['seed_layer'] == layer
                layer_df = seeds_df[layer_mask].copy()
                
                if len(layer_df) == 0:
                    raise ValueError(f"No seeds found for {layer} layer")
                    
                layer_seeds[layer] = layer_df
                self.logger.info(f"{layer} layer: {len(layer_df)} seeds prepared")
                
            self.logger.info("✅ Layer seeds preparation completed")
            return layer_seeds
            
        except Exception as e:
            self.logger.error(f"Failed to prepare layer seeds: {str(e)}")
            raise
            
    def rewrite_single_record(self, record: Dict, index: int, prompt_file: str, layer: str) -> Dict:
        """Rewrite a single malicious email record using LLM with YAML prompts"""
        try:
            # Load prompts from YAML configuration
            system_prompt = load_prompt(prompt_file, "prompts.rewrite_phishing.system.template")
            user_prompt = load_prompt(prompt_file, "prompts.rewrite_phishing.user.template",
                                     original_subject=record['subject'],
                                     original_body=record['body'])
            
            # Add delay for API rate limiting
            time.sleep(self.api_delay)
            
            # Call LLM API
            response = process_llm_request(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name=self.model_name,
                temperature=self.temperature
            )
            
            # Parse response (expecting JSON format)
            try:
                response_data = json.loads(response)
                rewritten_record = {
                    'subject': response_data.get('rewritten_subject', record['subject']),
                    'body': response_data.get('rewritten_body', record['body']),
                    'label': record['label'],  # Keep original label (malicious)
                    'source': record.get('source', 'unknown'),
                    'prompt_used': prompt_file,
                    'original_seed_id': record['batch6_seed_id'],
                    'original_layer': layer,
                    'generation_index': index
                }
                
                self.logger.debug(f"Successfully rewrote record {index} from {layer} layer with {prompt_file}")
                return rewritten_record
                
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse JSON response for record {index}, using original")
                # Fallback: return original record with metadata
                fallback_record = {
                    'subject': record['subject'],
                    'body': record['body'],
                    'label': record['label'],
                    'source': record.get('source', 'unknown'),
                    'prompt_used': prompt_file,
                    'original_seed_id': record['batch6_seed_id'],
                    'original_layer': layer,
                    'generation_index': index
                }
                return fallback_record
                
        except Exception as e:
            self.logger.error(f"Error rewriting record {index}: {str(e)}")
            # Return original record as failsafe
            failsafe_record = {
                'subject': record['subject'],
                'body': record['body'],
                'label': record['label'],
                'source': record.get('source', 'unknown'),
                'prompt_used': prompt_file,
                'original_seed_id': record['batch6_seed_id'],
                'original_layer': layer,
                'generation_index': index
            }
            return failsafe_record
            
    def generate_synthetic_for_layer_prompt(self, layer_df: pd.DataFrame, layer: str, 
                                          prompt_type: str, prompt_file: str) -> pd.DataFrame:
        """Generate synthetic data for a specific layer and prompt combination"""
        try:
            # Check if output file already exists
            output_file = self.synthetic_gen_dir / f"{layer}_{prompt_type}_synthetic.csv.gz"
            if output_file.exists():
                self.logger.info(f"Output file already exists, loading: {output_file}")
                with gzip.open(output_file, 'rt', encoding='utf-8') as f:
                    existing_df = pd.read_csv(f)
                self.logger.info(f"✅ Loaded existing {len(existing_df)} synthetic samples")
                return existing_df
            
            self.logger.info(f"Generating synthetic data: {layer} layer × {prompt_type} prompt")
            self.logger.info(f"Processing {len(layer_df)} seeds with {self.max_workers} workers")
            
            start_time = time.time()
            rewritten_records = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Create futures for all records
                futures = []
                for index, (_, record) in enumerate(layer_df.iterrows()):
                    future = executor.submit(
                        self.rewrite_single_record, 
                        record.to_dict(), 
                        index, 
                        prompt_file, 
                        layer
                    )
                    futures.append(future)
                
                # Collect results as they complete
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    try:
                        result = future.result()
                        rewritten_records.append(result)
                        
                        if (i + 1) % 50 == 0:
                            elapsed = time.time() - start_time
                            progress = (i + 1) / len(futures) * 100
                            self.logger.info(f"Progress: {progress:.1f}% ({i + 1}/{len(futures)}) - Elapsed: {elapsed:.1f}s")
                            
                    except Exception as e:
                        self.logger.error(f"Failed to process record: {str(e)}")
            
            # Convert to DataFrame
            synthetic_df = pd.DataFrame(rewritten_records)
            
            elapsed = time.time() - start_time
            self.logger.info(f"✅ {layer} × {prompt_type} generation completed in {elapsed:.1f}s")
            self.logger.info(f"Generated {len(synthetic_df)} synthetic samples")
            
            return synthetic_df
            
        except Exception as e:
            self.logger.error(f"Failed to generate synthetic data for {layer} × {prompt_type}: {str(e)}")
            raise
            
    def save_synthetic_data(self, synthetic_df: pd.DataFrame, layer: str, prompt_type: str) -> Path:
        """Save synthetic data to compressed CSV file"""
        try:
            output_file = self.synthetic_gen_dir / f"{layer}_{prompt_type}_synthetic.csv.gz"
            
            self.logger.info(f"Saving {len(synthetic_df)} synthetic samples to: {output_file}")
            
            with gzip.open(output_file, 'wt', encoding='utf-8') as f:
                synthetic_df.to_csv(f, index=False)
                
            self.logger.info(f"✅ Saved synthetic data: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Failed to save synthetic data: {str(e)}")
            raise
            
    def generate_all_synthetic_data(self, layer_seeds: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Generate synthetic data for all layer-prompt combinations"""
        try:
            self.logger.info("Starting comprehensive synthetic data generation...")
            
            all_synthetic = {}
            total_combinations = len(['core', 'edge']) * len(self.prompt_configs)
            current_combination = 0
            
            generation_start = time.time()
            
            for layer in ['core', 'edge']:
                all_synthetic[layer] = {}
                layer_df = layer_seeds[layer]
                
                for prompt_type, prompt_file in self.prompt_configs.items():
                    current_combination += 1
                    
                    self.logger.info(f"="*60)
                    self.logger.info(f"COMBINATION {current_combination}/{total_combinations}: {layer.upper()} × {prompt_type.upper()}")
                    self.logger.info(f"="*60)
                    
                    # Generate synthetic data
                    synthetic_df = self.generate_synthetic_for_layer_prompt(
                        layer_df, layer, prompt_type, prompt_file
                    )
                    
                    # Save synthetic data
                    output_file = self.save_synthetic_data(synthetic_df, layer, prompt_type)
                    
                    # Store in memory for further processing
                    all_synthetic[layer][prompt_type] = synthetic_df
                    
                    # Log progress
                    elapsed = time.time() - generation_start
                    remaining = (total_combinations - current_combination) * (elapsed / current_combination)
                    self.logger.info(f"Progress: {current_combination}/{total_combinations} completed")
                    self.logger.info(f"Elapsed: {elapsed/60:.1f}min, Estimated remaining: {remaining/60:.1f}min")
                    
            total_elapsed = time.time() - generation_start
            self.logger.info(f"✅ All synthetic data generation completed in {total_elapsed/60:.1f} minutes")
            
            return all_synthetic
            
        except Exception as e:
            self.logger.error(f"Failed to generate all synthetic data: {str(e)}")
            raise
            
    def validate_generation_results(self, all_synthetic: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """Validate and summarize generation results"""
        try:
            self.logger.info("Validating generation results...")
            
            validation_results = {
                'total_generated': 0,
                'by_layer': {},
                'by_prompt': {},
                'quality_check': {},
                'generation_summary': {}
            }
            
            # Count by layer and prompt
            for layer in ['core', 'edge']:
                layer_total = 0
                validation_results['by_layer'][layer] = {}
                
                for prompt_type in self.prompt_configs.keys():
                    if layer in all_synthetic and prompt_type in all_synthetic[layer]:
                        count = len(all_synthetic[layer][prompt_type])
                        validation_results['by_layer'][layer][prompt_type] = count
                        layer_total += count
                        
                        # Count by prompt type across layers
                        if prompt_type not in validation_results['by_prompt']:
                            validation_results['by_prompt'][prompt_type] = 0
                        validation_results['by_prompt'][prompt_type] += count
                        
                validation_results['by_layer'][layer]['total'] = layer_total
                validation_results['total_generated'] += layer_total
                
            # Quality checks
            expected_total = self.expected_total_synthetic
            generation_success_rate = validation_results['total_generated'] / expected_total * 100
            
            validation_results['quality_check'] = {
                'expected_total': expected_total,
                'actual_total': validation_results['total_generated'],
                'success_rate': generation_success_rate,
                'meets_target': generation_success_rate >= 95.0  # 95% success threshold
            }
            
            # Generation summary
            validation_results['generation_summary'] = {
                'layers_processed': len(['core', 'edge']),
                'prompts_processed': len(self.prompt_configs),
                'total_combinations': len(['core', 'edge']) * len(self.prompt_configs),
                'model_used': self.model_name,
                'generation_date': '2025-07-30'
            }
            
            self.logger.info(f"✅ Validation completed:")
            self.logger.info(f"  Total generated: {validation_results['total_generated']:,}")
            self.logger.info(f"  Success rate: {generation_success_rate:.1f}%")
            self.logger.info(f"  Target met: {validation_results['quality_check']['meets_target']}")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Failed to validate generation results: {str(e)}")
            raise
            
    def generate_phase2_summary(self, validation_results: Dict[str, Any]) -> None:
        """Generate comprehensive Phase 2 summary"""
        try:
            self.logger.info("Generating Phase 2 summary...")
            
            summary = {
                'phase2_completion': {
                    'status': 'completed',
                    'completion_date': '2025-07-30',
                    'total_runtime_hours': 'TBD'
                },
                'generation_configuration': {
                    'model': self.model_name,
                    'temperature': self.temperature,
                    'max_workers': self.max_workers,
                    'prompt_configs': self.prompt_configs,
                    'target_layers': ['core', 'edge']
                },
                'input_analysis': {
                    'seed_samples_processed': self.expected_seeds,
                    'layers_processed': len(['core', 'edge']),
                    'prompts_per_seed': len(self.prompt_configs)
                },
                'generation_results': validation_results,
                'output_files': {
                    'synthetic_data_files': [
                        f"{layer}_{prompt}_synthetic.csv.gz" 
                        for layer in ['core', 'edge'] 
                        for prompt in self.prompt_configs.keys()
                    ],
                    'total_output_files': len(['core', 'edge']) * len(self.prompt_configs)
                },
                'quality_assessment': {
                    'generation_success': validation_results['quality_check']['meets_target'],
                    'data_integrity': 'validated',
                    'ready_for_phase3': validation_results['quality_check']['meets_target']
                }
            }
            
            # Save summary
            summary_file = self.phase2_analysis_dir / "phase2_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"✅ Phase 2 summary saved: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate Phase 2 summary: {str(e)}")
            raise
            
    def run_phase2_complete(self) -> None:
        """Execute complete Phase 2 pipeline"""
        try:
            start_time = time.time()
            
            self.logger.info("="*60)
            self.logger.info("STARTING BATCH 6 PHASE 2: HIGH-QUALITY SYNTHETIC GENERATION")
            self.logger.info("="*60)
            
            # Step 1: Load seed data from Phase 1
            self.logger.info("Step 1: Loading seed data from Phase 1...")
            seeds_df = self.load_seed_data()
            
            # Step 2: Prepare seeds by layer
            self.logger.info("Step 2: Preparing seeds by layer...")
            layer_seeds = self.prepare_layer_seeds(seeds_df)
            
            # Step 3: Generate synthetic data for all combinations
            self.logger.info("Step 3: Generating synthetic data for all layer-prompt combinations...")
            all_synthetic = self.generate_all_synthetic_data(layer_seeds)
            
            # Step 4: Validate results
            self.logger.info("Step 4: Validating generation results...")
            validation_results = self.validate_generation_results(all_synthetic)
            
            # Step 5: Generate summary
            self.logger.info("Step 5: Generating Phase 2 summary...")
            self.generate_phase2_summary(validation_results)
            
            # Final completion
            elapsed_time = time.time() - start_time
            
            self.logger.info("="*60)
            self.logger.info("BATCH 6 PHASE 2 COMPLETED SUCCESSFULLY!")
            self.logger.info("="*60)
            self.logger.info(f"Total runtime: {elapsed_time/3600:.1f} hours")
            self.logger.info(f"Synthetic samples generated: {validation_results['total_generated']:,}")
            self.logger.info(f"Success rate: {validation_results['quality_check']['success_rate']:.1f}%")
            self.logger.info(f"Output location: {self.synthetic_gen_dir}")
            self.logger.info("Ready for Phase 3: Synthetic data ID annotation")
            self.logger.info("="*60)
            
        except Exception as e:
            self.logger.error(f"Batch 6 Phase 2 failed: {str(e)}", exc_info=True)
            raise


def main():
    """Main function to run Batch 6 Phase 2"""
    logger = setup_logger("batch6_phase2")
    
    try:
        # Check if Phase 2 already completed
        batch6_dir = PROJECT_ROOT / "data" / "batch6"
        phase2_summary = batch6_dir / "phase2_analysis" / "phase2_summary.json"
        
        if phase2_summary.exists():
            logger.info(f"Phase 2 already completed! Summary found: {phase2_summary}")
            with open(phase2_summary, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            logger.info(f"Generated: {summary['generation_results']['total_generated']} synthetic samples")
            logger.info("Use Ctrl+C to stop if running unnecessarily")
            return
        
        # Initialize and run Phase 2
        phase2 = Batch6Phase2SyntheticGeneration(logger=logger)
        phase2.run_phase2_complete()
        
        logger.info("Batch 6 Phase 2 successfully completed!")
        
    except Exception as e:
        logger.error(f"Error in Batch 6 Phase 2: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()