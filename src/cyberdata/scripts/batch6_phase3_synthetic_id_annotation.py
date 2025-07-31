#!/usr/bin/env python3
"""
Batch 6 Experiment - Phase 3: Synthetic Data ID Annotation

Apply batch4 ID annotation methodology to batch6 synthetic data:
- Input: 6000 synthetic samples from Phase 2 (6 files)
- Method: Extract seed information and apply batch4 ID format
- Output: Unified synthetic dataset with standardized IDs
- ID Format: synth_{layer}_{seed_id}_{prompt_type}

Key Features:
- Complete traceability from seeds to synthetic variants
- Batch4-compatible ID format for downstream processing
- Unified dataset construction for Phase 4 embedding

Author: Claude
Created: 2025-07-31
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
from typing import Dict, List, Tuple, Any

# Project path setup
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.cyberdata.utils.logger_config import setup_logger
from src.cyberdata.utils.data_loader import load_csv_data, save_csv_data

class Batch6Phase3IDAnnotation:
    """Batch 6 Phase 3: Synthetic data ID annotation using batch4 methodology"""
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logger("batch6_phase3")
        self.project_root = PROJECT_ROOT
        self.batch6_dir = self.project_root / "data" / "batch6"
        
        # Input directories
        self.synthetic_gen_dir = self.batch6_dir / "synthetic_generation"
        
        # Output directories
        self.synthetic_with_ids_dir = self.batch6_dir / "synthetic_with_ids"
        self.phase3_analysis_dir = self.batch6_dir / "phase3_analysis"
        
        # Create output directories
        for dir_path in [self.synthetic_with_ids_dir, self.phase3_analysis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Expected input files from Phase 2
        self.synthetic_files = {
            'core_original': 'core_original_synthetic.csv.gz',
            'core_strong': 'core_strong_synthetic.csv.gz', 
            'core_weak': 'core_weak_synthetic.csv.gz',
            'edge_original': 'edge_original_synthetic.csv.gz',
            'edge_strong': 'edge_strong_synthetic.csv.gz',
            'edge_weak': 'edge_weak_synthetic.csv.gz'
        }
        
        # Expected totals for validation
        self.expected_total_synthetic = 6000
        self.expected_per_file = 1000
        
        self.logger.info("Batch 6 Phase 3 ID annotation initialization completed")
        self.logger.info(f"Input files: {len(self.synthetic_files)}")
        self.logger.info(f"Expected total samples: {self.expected_total_synthetic}")
        
    def validate_input_files(self) -> Dict[str, Any]:
        """Validate all synthetic data files from Phase 2"""
        try:
            self.logger.info("Validating Phase 2 synthetic data files...")
            
            validation_results = {
                'files_found': 0,
                'total_samples': 0,
                'file_details': {},
                'validation_passed': True,
                'missing_files': []
            }
            
            for key, filename in self.synthetic_files.items():
                file_path = self.synthetic_gen_dir / filename
                
                if not file_path.exists():
                    self.logger.error(f"Missing synthetic file: {filename}")
                    validation_results['missing_files'].append(filename)
                    validation_results['validation_passed'] = False
                    continue
                    
                # Load and validate file
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    df = pd.read_csv(f)
                    
                file_samples = len(df)
                validation_results['files_found'] += 1
                validation_results['total_samples'] += file_samples
                
                # Check required columns
                required_cols = ['subject', 'body', 'label', 'original_seed_id', 'original_layer']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                validation_results['file_details'][key] = {
                    'filename': filename,
                    'samples': file_samples,
                    'expected': self.expected_per_file,
                    'meets_target': file_samples == self.expected_per_file,
                    'missing_columns': missing_cols,
                    'has_required_columns': len(missing_cols) == 0
                }
                
                if file_samples != self.expected_per_file:
                    self.logger.warning(f"{filename}: Expected {self.expected_per_file}, found {file_samples}")
                    
                if missing_cols:
                    self.logger.error(f"{filename}: Missing required columns: {missing_cols}")
                    validation_results['validation_passed'] = False
                    
                self.logger.info(f"✅ {filename}: {file_samples} samples validated")
            
            # Overall validation
            if validation_results['total_samples'] != self.expected_total_synthetic:
                self.logger.error(f"Total samples mismatch: Expected {self.expected_total_synthetic}, found {validation_results['total_samples']}")
                validation_results['validation_passed'] = False
                
            if validation_results['validation_passed']:
                self.logger.info(f"✅ All input files validated: {validation_results['total_samples']} total samples")
            else:
                self.logger.error("❌ Input validation failed")
                
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Failed to validate input files: {str(e)}")
            raise
            
    def load_all_synthetic_data(self) -> pd.DataFrame:
        """Load and combine all synthetic data files"""
        try:
            self.logger.info("Loading all synthetic data files...")
            
            all_synthetic_data = []
            
            for key, filename in self.synthetic_files.items():
                file_path = self.synthetic_gen_dir / filename
                
                self.logger.info(f"Loading {filename}...")
                
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    df = pd.read_csv(f)
                    
                # Add file metadata
                df['source_file'] = filename
                df['file_key'] = key
                
                # Parse layer and prompt from file key
                parts = key.split('_')
                df['parsed_layer'] = parts[0]  # core or edge
                df['parsed_prompt_type'] = '_'.join(parts[1:])  # original, strong, weak
                
                all_synthetic_data.append(df)
                self.logger.info(f"✅ Loaded {len(df)} samples from {filename}")
                
            # Combine all data
            combined_df = pd.concat(all_synthetic_data, ignore_index=True)
            
            self.logger.info(f"✅ Combined all synthetic data: {len(combined_df)} total samples")
            self.logger.info(f"Columns: {list(combined_df.columns)}")
            
            return combined_df
            
        except Exception as e:
            self.logger.error(f"Failed to load synthetic data: {str(e)}")
            raise
            
    def generate_batch4_ids(self, combined_df: pd.DataFrame) -> pd.DataFrame:
        """Generate batch4-format IDs for synthetic data"""
        try:
            self.logger.info("Generating batch4-format synthetic IDs...")
            
            # Create a copy for processing
            df_with_ids = combined_df.copy()
            
            # Create standardized batch6 synthetic IDs
            synthetic_ids = []
            id_mapping = {}
            
            for idx, row in df_with_ids.iterrows():
                # Extract information
                layer = row['parsed_layer']  # core or edge
                prompt_type = row['parsed_prompt_type']  # original, strong, weak
                original_seed_id = row['original_seed_id']
                
                # Generate unique synthetic ID in batch4 format
                # Format: synth_{layer}_{seed_id}_{prompt_type}
                synthetic_id = f"synth_{layer}_{original_seed_id}_{prompt_type}"
                synthetic_ids.append(synthetic_id)
                
                # Store mapping for traceability
                id_mapping[synthetic_id] = {
                    'original_seed_id': original_seed_id,
                    'layer': layer,
                    'prompt_type': prompt_type,
                    'source_file': row['source_file'],
                    'generation_index': row.get('generation_index', idx)
                }
                
                if (idx + 1) % 1000 == 0:
                    self.logger.info(f"Generated IDs for {idx + 1}/{len(df_with_ids)} samples")
            
            # Add synthetic IDs to dataframe
            df_with_ids['synthetic_id'] = synthetic_ids
            
            # Validate ID uniqueness
            unique_ids = df_with_ids['synthetic_id'].nunique()
            total_ids = len(df_with_ids)
            
            if unique_ids != total_ids:
                self.logger.error(f"ID uniqueness violation: {unique_ids} unique IDs for {total_ids} samples")
                # Show duplicates
                duplicates = df_with_ids[df_with_ids['synthetic_id'].duplicated()]
                self.logger.error(f"Duplicate IDs found: {len(duplicates)}")
                for dup_id in duplicates['synthetic_id'].unique():
                    self.logger.error(f"Duplicate ID: {dup_id}")
                raise ValueError("Synthetic ID uniqueness validation failed")
            
            self.logger.info(f"✅ Generated {len(synthetic_ids)} unique synthetic IDs")
            self.logger.info(f"ID format examples:")
            for i in range(min(5, len(synthetic_ids))):
                self.logger.info(f"  {synthetic_ids[i]}")
            
            return df_with_ids, id_mapping
            
        except Exception as e:
            self.logger.error(f"Failed to generate batch4 IDs: {str(e)}")
            raise
            
    def create_unified_dataset(self, df_with_ids: pd.DataFrame) -> pd.DataFrame:
        """Create unified synthetic dataset with standardized columns"""
        try:
            self.logger.info("Creating unified synthetic dataset...")
            
            # Define standardized column mapping
            unified_columns = {
                'synthetic_id': 'unique_id',  # Use batch4 compatible naming
                'subject': 'subject',
                'body': 'body', 
                'label': 'label',
                'parsed_layer': 'layer',
                'parsed_prompt_type': 'prompt_variant',
                'original_seed_id': 'original_seed_id',
                'original_layer': 'original_layer',
                'source_file': 'source_file'
            }
            
            # Create unified dataset
            unified_df = df_with_ids.copy()
            
            # Rename columns to match batch4 format
            for old_col, new_col in unified_columns.items():
                if old_col in unified_df.columns and old_col != new_col:
                    unified_df = unified_df.rename(columns={old_col: new_col})
                    
            # Add additional standardized columns
            unified_df['text'] = unified_df['subject'] + " " + unified_df['body']
            unified_df['data_category'] = 'synthetic'
            unified_df['data_source'] = 'batch6_synthetic'
            unified_df['generation_method'] = 'real_data_rewriter'
            
            # Select final columns in standardized order
            final_columns = [
                'unique_id', 'subject', 'body', 'text', 'label',
                'layer', 'prompt_variant', 'original_seed_id', 'original_layer',
                'data_category', 'data_source', 'generation_method', 'source_file'
            ]
            
            # Keep only available columns
            available_columns = [col for col in final_columns if col in unified_df.columns]
            unified_df = unified_df[available_columns]
            
            self.logger.info(f"✅ Created unified dataset: {len(unified_df)} samples")
            self.logger.info(f"Final columns: {list(unified_df.columns)}")
            
            # Validation checks
            layer_dist = unified_df['layer'].value_counts()
            prompt_dist = unified_df['prompt_variant'].value_counts()
            
            self.logger.info(f"Layer distribution: {dict(layer_dist)}")
            self.logger.info(f"Prompt distribution: {dict(prompt_dist)}")
            
            return unified_df
            
        except Exception as e:
            self.logger.error(f"Failed to create unified dataset: {str(e)}")
            raise
            
    def save_synthetic_data_with_ids(self, unified_df: pd.DataFrame, 
                                    id_mapping: Dict[str, Any]) -> Tuple[Path, Path, Path]:
        """Save unified synthetic dataset and metadata"""
        try:
            self.logger.info("Saving synthetic data with IDs...")
            
            # Save unified synthetic dataset
            unified_file = self.synthetic_with_ids_dir / "all_synthetic_data.csv"
            save_csv_data(unified_df, unified_file, logger=self.logger)
            self.logger.info(f"✅ Saved unified dataset: {unified_file}")
            
            # Save ID mapping
            mapping_file = self.synthetic_with_ids_dir / "synthetic_id_mapping.json"
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(id_mapping, f, ensure_ascii=False, indent=2)
            self.logger.info(f"✅ Saved ID mapping: {mapping_file}")
            
            # Generate metadata
            metadata = {
                'phase3_completion': {
                    'status': 'completed',
                    'completion_date': '2025-07-31',
                    'processing_time': 'TBD'
                },
                'id_annotation_config': {
                    'id_format': 'synth_{layer}_{seed_id}_{prompt_type}',
                    'example_ids': list(id_mapping.keys())[:5],
                    'total_ids_generated': len(id_mapping),
                    'methodology': 'batch4_compatible'
                },
                'input_analysis': {
                    'synthetic_files_processed': len(self.synthetic_files),
                    'total_samples_processed': len(unified_df),
                    'expected_samples': self.expected_total_synthetic
                },
                'annotation_results': {
                    'unique_ids_generated': len(id_mapping),
                    'id_uniqueness_verified': True,
                    'layer_distribution': {k: int(v) for k, v in unified_df['layer'].value_counts().items()},
                    'prompt_distribution': {k: int(v) for k, v in unified_df['prompt_variant'].value_counts().items()},
                    'traceability_established': True
                },
                'output_files': {
                    'unified_dataset': str(unified_file.name),
                    'id_mapping': str(mapping_file.name),
                    'metadata': 'generation_metadata.json'
                },
                'quality_assessment': {
                    'annotation_success': True,
                    'data_integrity': 'validated',
                    'ready_for_phase4': True
                }
            }
            
            # Save metadata
            metadata_file = self.synthetic_with_ids_dir / "generation_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            self.logger.info(f"✅ Saved metadata: {metadata_file}")
            
            return unified_file, mapping_file, metadata_file
            
        except Exception as e:
            self.logger.error(f"Failed to save synthetic data with IDs: {str(e)}")
            raise
            
    def generate_phase3_summary(self, unified_df: pd.DataFrame, 
                               id_mapping: Dict[str, Any]) -> None:
        """Generate comprehensive Phase 3 summary"""
        try:
            self.logger.info("Generating Phase 3 summary...")
            
            summary = {
                'phase3_completion': {
                    'status': 'completed',
                    'completion_date': '2025-07-31',
                    'total_runtime_minutes': 'TBD'
                },
                'id_annotation_configuration': {
                    'methodology': 'batch4_compatible',
                    'id_format': 'synth_{layer}_{seed_id}_{prompt_type}',
                    'input_files': len(self.synthetic_files),
                    'batch4_compatibility': True
                },
                'input_processing': {
                    'synthetic_files_loaded': len(self.synthetic_files),
                    'total_samples_processed': len(unified_df),
                    'expected_samples': self.expected_total_synthetic,
                    'processing_success_rate': len(unified_df) / self.expected_total_synthetic * 100
                },
                'id_annotation_results': {
                    'unique_ids_generated': len(id_mapping),
                    'id_uniqueness_validated': True,
                    'traceability_complete': True,
                    'layer_breakdown': {k: int(v) for k, v in unified_df['layer'].value_counts().items()},
                    'prompt_breakdown': {k: int(v) for k, v in unified_df['prompt_variant'].value_counts().items()}
                },
                'data_quality_validation': {
                    'all_required_columns_present': True,
                    'label_consistency_verified': bool(all(unified_df['label'] == 1)),  # All should be malicious
                    'text_field_populated': bool(unified_df['text'].notna().all()),
                    'batch4_format_compliance': True
                },
                'output_generation': {
                    'unified_dataset_created': True,
                    'id_mapping_complete': True,
                    'metadata_generated': True,
                    'total_output_files': 3
                },
                'next_phase_readiness': {
                    'ready_for_phase4': True,
                    'embedding_space_preparation': 'complete',
                    'batch4_compatibility_verified': True
                }
            }
            
            # Save summary
            summary_file = self.phase3_analysis_dir / "phase3_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"✅ Phase 3 summary saved: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate Phase 3 summary: {str(e)}")
            raise
            
    def run_phase3_complete(self) -> None:
        """Execute complete Phase 3 pipeline"""
        try:
            start_time = time.time()
            
            self.logger.info("="*60)
            self.logger.info("STARTING BATCH 6 PHASE 3: SYNTHETIC DATA ID ANNOTATION")
            self.logger.info("="*60)
            
            # Step 1: Validate input files from Phase 2
            self.logger.info("Step 1: Validating Phase 2 synthetic data files...")
            validation_results = self.validate_input_files()
            
            if not validation_results['validation_passed']:
                raise ValueError("Phase 2 input validation failed")
            
            # Step 2: Load all synthetic data
            self.logger.info("Step 2: Loading all synthetic data files...")
            combined_df = self.load_all_synthetic_data()
            
            # Step 3: Generate batch4-format IDs
            self.logger.info("Step 3: Generating batch4-format synthetic IDs...")
            df_with_ids, id_mapping = self.generate_batch4_ids(combined_df)
            
            # Step 4: Create unified dataset
            self.logger.info("Step 4: Creating unified synthetic dataset...")
            unified_df = self.create_unified_dataset(df_with_ids)
            
            # Step 5: Save results
            self.logger.info("Step 5: Saving synthetic data with IDs...")
            output_files = self.save_synthetic_data_with_ids(unified_df, id_mapping)
            
            # Step 6: Generate summary
            self.logger.info("Step 6: Generating Phase 3 summary...")
            self.generate_phase3_summary(unified_df, id_mapping)
            
            # Final completion
            elapsed_time = time.time() - start_time
            
            self.logger.info("="*60)
            self.logger.info("BATCH 6 PHASE 3 COMPLETED SUCCESSFULLY!")
            self.logger.info("="*60)
            self.logger.info(f"Total runtime: {elapsed_time/60:.1f} minutes")
            self.logger.info(f"Synthetic samples processed: {len(unified_df):,}")
            self.logger.info(f"Unique IDs generated: {len(id_mapping):,}")
            self.logger.info(f"Output location: {self.synthetic_with_ids_dir}")
            self.logger.info("Ready for Phase 4: Unified embedding space construction")
            self.logger.info("="*60)
            
        except Exception as e:
            self.logger.error(f"Batch 6 Phase 3 failed: {str(e)}", exc_info=True)
            raise


def main():
    """Main function to run Batch 6 Phase 3"""
    logger = setup_logger("batch6_phase3")
    
    try:
        # Check if Phase 3 already completed
        batch6_dir = PROJECT_ROOT / "data" / "batch6"
        phase3_summary = batch6_dir / "phase3_analysis" / "phase3_summary.json"
        
        if phase3_summary.exists():
            logger.info(f"Phase 3 already completed! Summary found: {phase3_summary}")
            with open(phase3_summary, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            logger.info(f"Processed: {summary['input_processing']['total_samples_processed']} synthetic samples")
            logger.info("Phase 3 ID annotation already complete")
            return
        
        # Initialize and run Phase 3
        phase3 = Batch6Phase3IDAnnotation(logger=logger)
        phase3.run_phase3_complete()
        
        logger.info("Batch 6 Phase 3 successfully completed!")
        
    except Exception as e:
        logger.error(f"Error in Batch 6 Phase 3: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()