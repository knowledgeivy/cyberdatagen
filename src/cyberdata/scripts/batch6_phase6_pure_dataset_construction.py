#!/usr/bin/env python3
"""
Batch 6 Phase 6: Pure Dataset Construction
=========================================

Builds pure datasets using improved synthetic data for performance validation.
Replicates batch5 methodology with batch6's high-quality synthetic data.

Key configurations:
- baseline_real: 100% real malicious data + fixed benign data
- pure_original: 100% original rewrite synthetic data + fixed benign data  
- pure_strong: 100% strong rewrite synthetic data + fixed benign data
- pure_weak: 100% weak rewrite synthetic data + fixed benign data
- Ratios: 5%, 10%, 15%, 20% malicious data proportion
- Dataset size: 10,000 samples each
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
import random
from collections import defaultdict

class Batch6Phase6PureDatasetConstructor:
    """Pure dataset constructor for batch6 improved synthetic data validation"""
    
    def __init__(self):
        self.setup_paths()
        self.setup_logging()
        self.load_configurations()
        
    def setup_paths(self) -> None:
        """Setup directory structure for phase 6"""
        # Use absolute paths to avoid working directory issues
        current_dir = Path(__file__).parent.parent.parent.parent  # Go up to project root
        self.base_dir = current_dir / "data" / "batch6"
        self.input_dirs = {
            'metadata': self.base_dir / "unified_embeddings",
            'seeds': self.base_dir / "seed_preparation", 
            'synthetic': self.base_dir / "synthetic_with_ids",
            'batch4_data': current_dir / "data" / "batch4_fresh"
        }
        
        # Output directories
        self.output_dir = self.base_dir / "pure_datasets"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectory structure
        self.dataset_types = ['baseline_real', 'pure_original', 'pure_strong', 'pure_weak']
        self.malicious_ratios = [5, 10, 15, 20]  # Percentage ratios
        
        for dataset_type in self.dataset_types:
            for ratio in self.malicious_ratios:
                dataset_dir = self.output_dir / f"{dataset_type}_{ratio}pct"
                dataset_dir.mkdir(parents=True, exist_ok=True)
                
    def setup_logging(self) -> None:
        """Setup logging configuration"""
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"batch6_phase6_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_configurations(self) -> None:
        """Load dataset construction configurations"""
        self.config = {
            'dataset_size': 10000,  # Fixed size per dataset
            'malicious_ratios': [5, 10, 15, 20],  # Percentage ratios
            'random_seed': 42,
            'validation_checks': True
        }
        
        # Set random seeds for reproducibility
        random.seed(self.config['random_seed'])
        np.random.seed(self.config['random_seed'])
        
        self.logger.info(f"Phase 6 configuration loaded: {self.config}")
        
        # Log paths for debugging
        self.logger.info(f"Base directory: {self.base_dir}")
        self.logger.info(f"Metadata directory: {self.input_dirs['metadata']}")
        for name, path in self.input_dirs.items():
            exists = path.exists()
            self.logger.info(f"Input path {name}: {path} (exists: {exists})")
            if exists and path.is_dir():
                files = list(path.glob("*"))[:5]  # Show first 5 files
                self.logger.info(f"  Sample files: {[f.name for f in files]}")
        
    def load_source_data(self) -> Dict[str, pd.DataFrame]:
        """Load all source data for dataset construction"""
        try:
            self.logger.info("Loading source data for pure dataset construction...")
            
            source_data = {}
            
            # 1. Load metadata with unified IDs and embeddings
            metadata_file = self.input_dirs['metadata'] / "embedding_metadata.csv.gz"
            if metadata_file.exists():
                source_data['metadata'] = pd.read_csv(metadata_file, compression='gzip')
                self.logger.info(f"✅ Loaded metadata: {len(source_data['metadata'])} records")
            else:
                raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
            
            # 2. Load synthetic data with IDs
            synthetic_file = self.input_dirs['synthetic'] / "all_synthetic_data.csv"
            if synthetic_file.exists():
                source_data['synthetic'] = pd.read_csv(synthetic_file)
                self.logger.info(f"✅ Loaded synthetic data: {len(source_data['synthetic'])} records")
            else:
                raise FileNotFoundError(f"Synthetic data file not found: {synthetic_file}")
            
            # 3. Load batch4 benign data (for fixed benign component)
            benign_file = self.input_dirs['batch4_data'] / "train_benign.csv.gz"
            if benign_file.exists():
                source_data['benign'] = pd.read_csv(benign_file, compression='gzip')
                self.logger.info(f"✅ Loaded benign data: {len(source_data['benign'])} records")
            else:
                # Try uncompressed version as fallback
                benign_file_uncompressed = self.input_dirs['batch4_data'] / "train_benign.csv"
                if benign_file_uncompressed.exists():
                    source_data['benign'] = pd.read_csv(benign_file_uncompressed)
                    self.logger.info(f"✅ Loaded benign data (uncompressed): {len(source_data['benign'])} records")
                else:
                    raise FileNotFoundError(f"Benign data file not found: {benign_file} or {benign_file_uncompressed}")
            
            # 4. Load real malicious background data (for baseline_real)
            malicious_file = self.input_dirs['batch4_data'] / "train_malicious.csv"
            if malicious_file.exists():
                source_data['real_malicious'] = pd.read_csv(malicious_file)
                self.logger.info(f"✅ Loaded real malicious data: {len(source_data['real_malicious'])} records")
            else:
                raise FileNotFoundError(f"Real malicious data file not found: {malicious_file}")
                
            return source_data
            
        except Exception as e:
            self.logger.error(f"Failed to load source data: {str(e)}")
            raise
            
    def prepare_data_pools(self, source_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Prepare categorized data pools for dataset construction"""
        try:
            self.logger.info("Preparing categorized data pools...")
            
            data_pools = {}
            
            # 1. Fixed benign pool (shared across all datasets)
            benign_pool = source_data['benign'].copy()
            benign_pool['data_category'] = 'benign'
            benign_pool['data_source'] = 'real_benign'
            data_pools['benign'] = benign_pool
            self.logger.info(f"✅ Benign pool: {len(data_pools['benign'])} samples")
            
            # 2. Real malicious pool (for baseline_real datasets)
            real_malicious_pool = source_data['real_malicious'].copy()
            real_malicious_pool['data_category'] = 'malicious'
            real_malicious_pool['data_source'] = 'real_malicious'
            data_pools['real_malicious'] = real_malicious_pool
            self.logger.info(f"✅ Real malicious pool: {len(data_pools['real_malicious'])} samples")
            
            # 3. Synthetic data pools by prompt type
            synthetic_data = source_data['synthetic']
            
            # Parse synthetic IDs to extract prompt information
            synthetic_data['prompt_variant'] = synthetic_data['unique_id'].str.extract(r'synth_\w+_\d+_(\w+)')
            
            for prompt_type in ['original', 'strong', 'weak']:
                prompt_mask = synthetic_data['prompt_variant'] == prompt_type
                prompt_pool = synthetic_data[prompt_mask].copy()
                prompt_pool['data_category'] = 'malicious'
                prompt_pool['data_source'] = f'synthetic_{prompt_type}'
                data_pools[f'synthetic_{prompt_type}'] = prompt_pool
                self.logger.info(f"✅ Synthetic {prompt_type} pool: {len(prompt_pool)} samples")
            
            return data_pools
            
        except Exception as e:
            self.logger.error(f"Failed to prepare data pools: {str(e)}")
            raise
            
    def construct_single_dataset(self, malicious_pool: pd.DataFrame, benign_pool: pd.DataFrame, 
                               malicious_ratio: int, dataset_name: str) -> pd.DataFrame:
        """Construct a single pure dataset with specified malicious ratio"""
        try:
            dataset_size = self.config['dataset_size']
            malicious_count = int(dataset_size * malicious_ratio / 100)
            benign_count = dataset_size - malicious_count
            
            self.logger.info(f"Constructing {dataset_name}: {malicious_count} malicious + {benign_count} benign")
            
            # Sample malicious data
            if len(malicious_pool) < malicious_count:
                # Sample with replacement if needed
                malicious_sample = malicious_pool.sample(n=malicious_count, replace=True, random_state=self.config['random_seed'])
                self.logger.warning(f"Sampling with replacement for {dataset_name} malicious data")
            else:
                malicious_sample = malicious_pool.sample(n=malicious_count, random_state=self.config['random_seed'])
            
            # Sample benign data
            if len(benign_pool) < benign_count:
                benign_sample = benign_pool.sample(n=benign_count, replace=True, random_state=self.config['random_seed'])
                self.logger.warning(f"Sampling with replacement for {dataset_name} benign data")
            else:
                benign_sample = benign_pool.sample(n=benign_count, random_state=self.config['random_seed'])
            
            # Combine and shuffle
            combined_dataset = pd.concat([malicious_sample, benign_sample], ignore_index=True)
            combined_dataset = combined_dataset.sample(frac=1, random_state=self.config['random_seed']).reset_index(drop=True)
            
            # Add dataset metadata
            combined_dataset['dataset_name'] = dataset_name
            combined_dataset['malicious_ratio'] = malicious_ratio
            
            self.logger.info(f"✅ {dataset_name} constructed: {len(combined_dataset)} total samples")
            return combined_dataset
            
        except Exception as e:
            self.logger.error(f"Failed to construct dataset {dataset_name}: {str(e)}")
            raise
            
    def construct_all_pure_datasets(self, data_pools: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Construct all pure datasets according to batch6 specifications"""
        try:
            self.logger.info("Constructing all pure datasets...")
            
            all_datasets = defaultdict(dict)
            
            # Dataset configurations
            dataset_configs = {
                'baseline_real': data_pools['real_malicious'],
                'pure_original': data_pools['synthetic_original'],
                'pure_strong': data_pools['synthetic_strong'],
                'pure_weak': data_pools['synthetic_weak']
            }
            
            for dataset_type, malicious_pool in dataset_configs.items():
                self.logger.info(f"Processing {dataset_type} datasets...")
                
                for ratio in self.malicious_ratios:
                    dataset_name = f"{dataset_type}_{ratio}pct"
                    
                    dataset = self.construct_single_dataset(
                        malicious_pool=malicious_pool,
                        benign_pool=data_pools['benign'],
                        malicious_ratio=ratio,
                        dataset_name=dataset_name
                    )
                    
                    all_datasets[dataset_type][f"{ratio}pct"] = dataset
            
            self.logger.info(f"✅ Constructed {len(all_datasets)} dataset types × {len(self.malicious_ratios)} ratios = {len(all_datasets) * len(self.malicious_ratios)} total datasets")
            return dict(all_datasets)
            
        except Exception as e:
            self.logger.error(f"Failed to construct all datasets: {str(e)}")
            raise
            
    def validate_datasets(self, all_datasets: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """Validate constructed datasets for quality and consistency"""
        try:
            self.logger.info("Validating constructed datasets...")
            
            validation_results = {
                'total_datasets': 0,
                'validation_passed': 0,
                'validation_failed': 0,
                'size_validation': {},
                'ratio_validation': {},
                'issues': []
            }
            
            for dataset_type, ratio_datasets in all_datasets.items():
                for ratio_name, dataset in ratio_datasets.items():
                    dataset_name = f"{dataset_type}_{ratio_name}"
                    validation_results['total_datasets'] += 1
                    
                    try:
                        # 1. Size validation
                        expected_size = self.config['dataset_size']
                        actual_size = len(dataset)
                        size_valid = (actual_size == expected_size)
                        validation_results['size_validation'][dataset_name] = {
                            'expected': expected_size,
                            'actual': actual_size,
                            'valid': size_valid
                        }
                        
                        # 2. Ratio validation
                        ratio_value = int(ratio_name.replace('pct', ''))
                        malicious_count = len(dataset[dataset['data_category'] == 'malicious'])
                        expected_malicious = int(expected_size * ratio_value / 100)
                        ratio_valid = (malicious_count == expected_malicious)
                        validation_results['ratio_validation'][dataset_name] = {
                            'expected_malicious': expected_malicious,
                            'actual_malicious': malicious_count,
                            'valid': ratio_valid
                        }
                        
                        # 3. Required columns validation
                        required_columns = ['subject', 'body', 'label', 'data_category', 'data_source']
                        columns_valid = all(col in dataset.columns for col in required_columns)
                        
                        if size_valid and ratio_valid and columns_valid:
                            validation_results['validation_passed'] += 1
                            self.logger.info(f"✅ {dataset_name}: PASSED validation")
                        else:
                            validation_results['validation_failed'] += 1
                            issues = []
                            if not size_valid:
                                issues.append(f"Size mismatch: expected {expected_size}, got {actual_size}")
                            if not ratio_valid:
                                issues.append(f"Ratio mismatch: expected {expected_malicious} malicious, got {malicious_count}")
                            if not columns_valid:
                                issues.append("Missing required columns")
                            validation_results['issues'].append(f"{dataset_name}: {'; '.join(issues)}")
                            self.logger.warning(f"⚠️ {dataset_name}: FAILED validation - {'; '.join(issues)}")
                        
                    except Exception as e:
                        validation_results['validation_failed'] += 1
                        validation_results['issues'].append(f"{dataset_name}: Validation error - {str(e)}")
                        self.logger.error(f"❌ {dataset_name}: Validation error - {str(e)}")
            
            self.logger.info(f"Dataset validation completed: {validation_results['validation_passed']}/{validation_results['total_datasets']} passed")
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Failed to validate datasets: {str(e)}")
            raise
            
    def save_datasets(self, all_datasets: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, List[str]]:
        """Save all constructed datasets to disk"""
        try:
            self.logger.info("Saving all constructed datasets...")
            
            saved_files = defaultdict(list)
            
            for dataset_type, ratio_datasets in all_datasets.items():
                for ratio_name, dataset in ratio_datasets.items():
                    dataset_name = f"{dataset_type}_{ratio_name}"
                    
                    # Save directory
                    save_dir = self.output_dir / dataset_name
                    save_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save main dataset
                    dataset_file = save_dir / f"{dataset_name}_dataset.csv"
                    dataset.to_csv(dataset_file, index=False)
                    saved_files[dataset_type].append(str(dataset_file))
                    
                    # Save dataset statistics  
                    stats = {
                        'dataset_name': dataset_name,
                        'total_samples': len(dataset),
                        'malicious_samples': len(dataset[dataset['data_category'] == 'malicious']),
                        'benign_samples': len(dataset[dataset['data_category'] == 'benign']),
                        'malicious_ratio_actual': len(dataset[dataset['data_category'] == 'malicious']) / len(dataset) * 100,
                        'data_sources': dataset['data_source'].value_counts().to_dict(),
                        'creation_timestamp': datetime.now().isoformat()
                    }
                    
                    stats_file = save_dir / f"{dataset_name}_statistics.json"
                    with open(stats_file, 'w') as f:
                        json.dump(stats, f, indent=2)
                    saved_files[dataset_type].append(str(stats_file))
                    
                    self.logger.info(f"✅ Saved {dataset_name}: {len(dataset)} samples")
            
            # Save overall summary
            summary = {
                'phase6_completion': {
                    'status': 'completed',
                    'completion_date': datetime.now().isoformat(),
                    'total_datasets_created': sum(len(files) // 2 for files in saved_files.values())  # Divide by 2 (dataset + stats)
                },
                'dataset_configuration': {
                    'dataset_types': list(all_datasets.keys()),
                    'malicious_ratios': self.malicious_ratios,
                    'dataset_size': self.config['dataset_size'],
                    'total_datasets': len(self.dataset_types) * len(self.malicious_ratios)
                },
                'file_structure': dict(saved_files)
            }
            
            summary_file = self.output_dir / "phase6_construction_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"✅ Phase 6 dataset construction completed: {summary['phase6_completion']['total_datasets_created']} datasets created")
            return dict(saved_files)
            
        except Exception as e:
            self.logger.error(f"Failed to save datasets: {str(e)}")
            raise
            
    def run_phase6(self) -> None:
        """Execute complete Phase 6 pure dataset construction"""
        try:
            self.logger.info("🚀 Starting Batch 6 Phase 6: Pure Dataset Construction")
            start_time = datetime.now()
            
            # Step 1: Load source data
            source_data = self.load_source_data()
            
            # Step 2: Prepare data pools
            data_pools = self.prepare_data_pools(source_data)
            
            # Step 3: Construct all pure datasets
            all_datasets = self.construct_all_pure_datasets(data_pools)
            
            # Step 4: Validate datasets
            validation_results = self.validate_datasets(all_datasets)
            
            # Step 5: Save datasets
            saved_files = self.save_datasets(all_datasets)
            
            # Phase completion
            end_time = datetime.now()
            runtime = (end_time - start_time).total_seconds() / 60
            
            self.logger.info(f"✅ Batch 6 Phase 6 completed successfully!")
            self.logger.info(f"Runtime: {runtime:.2f} minutes")
            self.logger.info(f"Datasets created: {validation_results['validation_passed']}/{validation_results['total_datasets']}")
            self.logger.info(f"Output directory: {self.output_dir}")
            
            if validation_results['validation_failed'] > 0:
                self.logger.warning(f"⚠️ {validation_results['validation_failed']} datasets failed validation")
                for issue in validation_results['issues']:
                    self.logger.warning(f"   - {issue}")
            
        except Exception as e:
            self.logger.error(f"❌ Batch 6 Phase 6 failed: {str(e)}")
            raise

def main():
    """Main execution function"""
    constructor = Batch6Phase6PureDatasetConstructor()
    constructor.run_phase6()

if __name__ == "__main__":
    main()