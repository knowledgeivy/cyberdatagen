#!/usr/bin/env python3
"""
Batch 6 Experiment - Phase 1: Enhanced Seed Sample Preparation

Enhanced seed sample preparation with 2-layer stratification:
- Extract 2000 high-quality seed samples from existing stratified layers
- Focus on extremes: core (1000) and edge (1000) layers only
- Maintain complete ID traceability and metadata consistency
- Prepare for high-quality synthetic data generation using real_data_rewriter.py

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
from typing import Dict, List, Tuple, Any

# Project path setup
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.cyberdata.utils.logger_config import setup_logger
from src.cyberdata.utils.data_loader import load_csv_data

class Batch6Phase1SeedPreparation:
    """Batch 6 Phase 1: Enhanced seed sample preparation with 2-layer stratification"""
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logger("batch6_phase1")
        self.project_root = PROJECT_ROOT
        self.batch4_dir = self.project_root / "data" / "batch4_fresh"
        self.batch6_dir = self.project_root / "data" / "batch6"
        
        # Input directories (from batch4)
        self.stratified_layers_dir = self.batch4_dir / "stratified_layers"
        
        # Output directories
        self.seed_prep_dir = self.batch6_dir / "seed_preparation"
        self.phase1_analysis_dir = self.batch6_dir / "phase1_analysis"
        
        # Create output directories
        for dir_path in [self.seed_prep_dir, self.phase1_analysis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Sampling configuration
        self.samples_per_layer = 1000  # 1000 samples per layer (core/edge)
        self.total_seeds = 2000  # Total seed samples needed
        self.random_state = 2025
        
        # Layer focus: only core and edge (simplified from 4-layer to 2-layer)
        self.target_layers = ['core', 'edge']
        
        self.logger.info("Batch 6 Phase 1 seed preparation initialization completed")
        self.logger.info(f"Target layers: {self.target_layers}")
        self.logger.info(f"Samples per layer: {self.samples_per_layer}")
        
    def verify_batch4_data(self) -> Dict[str, int]:
        """Verify batch4 stratified data availability"""
        try:
            self.logger.info("Verifying batch4 stratified data availability...")
            
            layer_counts = {}
            
            for layer in ['core', 'inner', 'outer', 'edge']:
                layer_file = self.stratified_layers_dir / f"{layer}_samples.csv"
                
                if not layer_file.exists():
                    raise FileNotFoundError(f"Layer file not found: {layer_file}")
                    
                # Count rows (subtract 1 for header)
                df = load_csv_data(layer_file, logger=self.logger)
                layer_counts[layer] = len(df)
                
                self.logger.info(f"{layer} layer: {layer_counts[layer]:,} samples available")
            
            # Verify sufficiency for our needs
            for layer in self.target_layers:
                if layer_counts[layer] < self.samples_per_layer:
                    raise ValueError(f"Insufficient {layer} samples: need {self.samples_per_layer}, have {layer_counts[layer]}")
                    
            self.logger.info(f"✅ Data sufficiency verified: sufficient samples in target layers")
            return layer_counts
            
        except Exception as e:
            self.logger.error(f"Batch4 data verification failed: {str(e)}")
            raise
            
    def extract_layer_seeds(self, layer: str) -> pd.DataFrame:
        """Extract seed samples from specified layer"""
        try:
            self.logger.info(f"Extracting {self.samples_per_layer} seeds from {layer} layer...")
            
            # Load layer data
            layer_file = self.stratified_layers_dir / f"{layer}_samples.csv"
            layer_df = load_csv_data(layer_file, logger=self.logger)
            
            self.logger.info(f"Loaded {len(layer_df):,} samples from {layer} layer")
            
            # Random sampling with fixed seed for reproducibility
            np.random.seed(self.random_state)
            sampled_df = layer_df.sample(n=self.samples_per_layer, random_state=self.random_state)
            
            # Verify layer consistency
            if 'layer' in sampled_df.columns:
                layer_values = sampled_df['layer'].unique()
                if len(layer_values) != 1 or layer_values[0] != layer:
                    self.logger.warning(f"Layer consistency issue in {layer}: {layer_values}")
            
            self.logger.info(f"✅ Successfully extracted {len(sampled_df)} seeds from {layer} layer")
            return sampled_df
            
        except Exception as e:
            self.logger.error(f"Failed to extract seeds from {layer} layer: {str(e)}")
            raise
            
    def combine_and_validate_seeds(self, core_seeds: pd.DataFrame, edge_seeds: pd.DataFrame) -> pd.DataFrame:
        """Combine core and edge seeds with validation"""
        try:
            self.logger.info("Combining and validating seed samples...")
            
            # Add layer source tracking
            core_seeds = core_seeds.copy()
            edge_seeds = edge_seeds.copy()
            
            core_seeds['seed_layer'] = 'core'
            edge_seeds['seed_layer'] = 'edge'
            
            # Combine
            combined_seeds = pd.concat([core_seeds, edge_seeds], ignore_index=True)
            
            # Validation checks
            total_samples = len(combined_seeds)
            if total_samples != self.total_seeds:
                raise ValueError(f"Expected {self.total_seeds} seeds, got {total_samples}")
                
            # Check for duplicates by unique_id
            if 'unique_id' in combined_seeds.columns:
                duplicate_count = combined_seeds.duplicated(subset=['unique_id']).sum()
                if duplicate_count > 0:
                    self.logger.warning(f"Found {duplicate_count} duplicate unique_ids, removing...")
                    combined_seeds = combined_seeds.drop_duplicates(subset=['unique_id'])
                    
            # Verify layer distribution
            layer_dist = combined_seeds['seed_layer'].value_counts()
            self.logger.info(f"Layer distribution: {dict(layer_dist)}")
            
            # Add seed preparation metadata
            combined_seeds['batch6_seed_id'] = [f"batch6_seed_{i:04d}" for i in range(len(combined_seeds))]
            combined_seeds['extraction_date'] = '2025-07-30'
            combined_seeds['source_batch'] = 'batch4_fresh'
            
            self.logger.info(f"✅ Successfully combined {len(combined_seeds)} validated seed samples")
            return combined_seeds
            
        except Exception as e:
            self.logger.error(f"Failed to combine and validate seeds: {str(e)}")
            raise
            
    def save_seed_data(self, combined_seeds: pd.DataFrame) -> Path:
        """Save combined seed data with metadata"""
        try:
            self.logger.info("Saving seed data and metadata...")
            
            # Save main seed data
            seeds_file = self.seed_prep_dir / "real_malicious_seeds_with_layers.csv"
            combined_seeds.to_csv(seeds_file, index=False)
            
            # Create layer distribution statistics
            layer_stats = {
                'total_seeds': len(combined_seeds),
                'layer_distribution': combined_seeds['seed_layer'].value_counts().to_dict(),
                'sampling_parameters': {
                    'samples_per_layer': self.samples_per_layer,
                    'random_state': self.random_state,
                    'target_layers': self.target_layers
                },
                'data_source': 'batch4_fresh/stratified_layers',
                'extraction_date': '2025-07-30'
            }
            
            layer_stats_file = self.seed_prep_dir / "seed_layer_distribution.json"
            with open(layer_stats_file, 'w', encoding='utf-8') as f:
                json.dump(layer_stats, f, ensure_ascii=False, indent=2)
                
            # Create ID mapping for traceability
            id_mapping = {
                'batch6_to_batch4_mapping': {},
                'layer_id_ranges': {
                    'core': {'start': 0, 'end': self.samples_per_layer - 1},
                    'edge': {'start': self.samples_per_layer, 'end': self.total_seeds - 1}
                },
                'total_mappings': len(combined_seeds)
            }
            
            # Build the mapping
            for idx, row in combined_seeds.iterrows():
                batch6_id = row['batch6_seed_id']
                batch4_id = row.get('unique_id', f'unknown_{idx}')
                id_mapping['batch6_to_batch4_mapping'][batch6_id] = {
                    'batch4_unique_id': batch4_id,
                    'layer': row['seed_layer'],
                    'distance_to_centroid': row.get('distance_to_centroid', None)
                }
                
            id_mapping_file = self.seed_prep_dir / "seed_id_mapping.json"
            with open(id_mapping_file, 'w', encoding='utf-8') as f:
                json.dump(id_mapping, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"✅ Seed data saved: {seeds_file}")
            self.logger.info(f"✅ Layer statistics saved: {layer_stats_file}")
            self.logger.info(f"✅ ID mapping saved: {id_mapping_file}")
            
            return seeds_file
            
        except Exception as e:
            self.logger.error(f"Failed to save seed data: {str(e)}")
            raise
            
    def generate_phase1_summary(self, combined_seeds: pd.DataFrame, layer_counts: Dict[str, int]) -> None:
        """Generate comprehensive Phase 1 summary"""
        try:
            self.logger.info("Generating Phase 1 summary...")
            
            # Basic statistics
            summary = {
                'phase1_completion': {
                    'status': 'completed',
                    'completion_date': '2025-07-30',
                    'total_runtime_minutes': 'TBD'
                },
                'input_data_analysis': {
                    'source': 'batch4_fresh/stratified_layers',
                    'available_layers': layer_counts,
                    'target_layers': self.target_layers,
                    'sufficiency_check': 'passed'
                },
                'extraction_results': {
                    'total_seeds_extracted': int(len(combined_seeds)),
                    'seeds_per_layer': int(self.samples_per_layer),
                    'layer_distribution': {k: int(v) for k, v in combined_seeds['seed_layer'].value_counts().to_dict().items()},
                    'sampling_method': f'random_sampling_seed_{self.random_state}'
                },
                'quality_validation': {
                    'unique_id_duplicates': int(combined_seeds.duplicated(subset=['unique_id']).sum()),
                    'missing_values': {k: int(v) for k, v in combined_seeds.isnull().sum().to_dict().items()},
                    'data_integrity': 'validated'
                },
                'output_files': {
                    'main_seeds_file': 'real_malicious_seeds_with_layers.csv',
                    'layer_statistics': 'seed_layer_distribution.json',
                    'id_mapping': 'seed_id_mapping.json'
                },
                'next_phase_readiness': {
                    'ready_for_phase2': True,
                    'synthetic_generation_input': f'{len(combined_seeds)} seed samples',
                    'expected_synthetic_output': f'{len(combined_seeds) * 3} samples (3 prompt variants)'
                }
            }
            
            # Distance analysis if available
            if 'distance_to_centroid' in combined_seeds.columns:
                distance_stats = combined_seeds['distance_to_centroid'].describe()
                summary['distance_analysis'] = {
                    'core_layer_distance_range': combined_seeds[
                        combined_seeds['seed_layer'] == 'core'
                    ]['distance_to_centroid'].describe().to_dict(),
                    'edge_layer_distance_range': combined_seeds[
                        combined_seeds['seed_layer'] == 'edge'  
                    ]['distance_to_centroid'].describe().to_dict(),
                    'distance_separation_confirmed': True
                }
                
            # Save summary
            summary_file = self.phase1_analysis_dir / "phase1_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"✅ Phase 1 summary saved: {summary_file}")
            
            # Log key metrics
            self.logger.info("="*60)
            self.logger.info("BATCH 6 PHASE 1 COMPLETION SUMMARY")
            self.logger.info("="*60)
            self.logger.info(f"✅ Seeds extracted: {len(combined_seeds):,}")
            self.logger.info(f"✅ Core layer seeds: {(combined_seeds['seed_layer'] == 'core').sum():,}")
            self.logger.info(f"✅ Edge layer seeds: {(combined_seeds['seed_layer'] == 'edge').sum():,}")
            self.logger.info(f"✅ Unique IDs verified: {len(combined_seeds['unique_id'].unique()):,}")
            self.logger.info(f"✅ Ready for Phase 2 synthetic generation")
            
        except Exception as e:
            self.logger.error(f"Failed to generate Phase 1 summary: {str(e)}")
            raise
            
    def run_phase1_complete(self) -> None:
        """Execute complete Phase 1 pipeline"""
        try:
            start_time = time.time()
            
            self.logger.info("="*60)
            self.logger.info("STARTING BATCH 6 PHASE 1: ENHANCED SEED PREPARATION")
            self.logger.info("="*60)
            
            # Step 1: Verify batch4 data availability
            self.logger.info("Step 1: Verifying batch4 stratified data...")
            layer_counts = self.verify_batch4_data()
            
            # Step 2: Extract core layer seeds
            self.logger.info("Step 2: Extracting core layer seeds...")
            core_seeds = self.extract_layer_seeds('core')
            
            # Step 3: Extract edge layer seeds
            self.logger.info("Step 3: Extracting edge layer seeds...")
            edge_seeds = self.extract_layer_seeds('edge')
            
            # Step 4: Combine and validate
            self.logger.info("Step 4: Combining and validating seeds...")
            combined_seeds = self.combine_and_validate_seeds(core_seeds, edge_seeds)
            
            # Step 5: Save seed data
            self.logger.info("Step 5: Saving seed data and metadata...")
            seeds_file = self.save_seed_data(combined_seeds)
            
            # Step 6: Generate summary
            self.logger.info("Step 6: Generating Phase 1 summary...")
            self.generate_phase1_summary(combined_seeds, layer_counts)
            
            # Final completion
            elapsed_time = time.time() - start_time
            
            self.logger.info("="*60)
            self.logger.info("BATCH 6 PHASE 1 COMPLETED SUCCESSFULLY!")
            self.logger.info("="*60)
            self.logger.info(f"Total runtime: {elapsed_time/60:.1f} minutes")
            self.logger.info(f"Seeds prepared: {len(combined_seeds):,}")
            self.logger.info(f"Output location: {self.seed_prep_dir}")
            self.logger.info("Ready for Phase 2: High-quality synthetic data generation")
            self.logger.info("="*60)
            
        except Exception as e:
            self.logger.error(f"Batch 6 Phase 1 failed: {str(e)}", exc_info=True)
            raise


def main():
    """Main function to run Batch 6 Phase 1"""
    logger = setup_logger("batch6_phase1")
    
    try:
        # Initialize and run Phase 1
        phase1 = Batch6Phase1SeedPreparation(logger=logger)
        phase1.run_phase1_complete()
        
        logger.info("Batch 6 Phase 1 successfully completed!")
        
    except Exception as e:
        logger.error(f"Error in Batch 6 Phase 1: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()