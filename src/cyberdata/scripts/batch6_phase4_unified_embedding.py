#!/usr/bin/env python3
"""
Batch 6 Experiment - Phase 4: Unified Embedding Space Construction

Build unified embedding space for all data using batch5 methodology:
- Input: All batch6 data sources (seeds, synthetic, background, benign, test)
- Method: Replicate batch5 Phase 1 approach with 384-dim embeddings
- Output: Unified embedding space with improved synthetic data quality
- Model: all-MiniLM-L6-v2 (consistent with previous experiments)

Key Features:
- Complete data integration across all sources
- Improved synthetic data quality from Phase 2-3
- Batch5-compatible processing pipeline
- Enhanced layer analysis with 2-layer simplification

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
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

# Project path setup
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.cyberdata.utils.logger_config import setup_logger
from src.cyberdata.utils.data_loader import load_csv_data, save_csv_data

class Batch6Phase4UnifiedEmbedding:
    """Batch 6 Phase 4: Unified embedding space construction"""
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logger("batch6_phase4")
        self.project_root = PROJECT_ROOT
        self.batch4_dir = self.project_root / "data" / "batch4_fresh"
        self.batch6_dir = self.project_root / "data" / "batch6"
        
        # Input directories
        self.seed_prep_dir = self.batch6_dir / "seed_preparation"
        self.synthetic_with_ids_dir = self.batch6_dir / "synthetic_with_ids"
        
        # Output directories
        self.unified_embeddings_dir = self.batch6_dir / "unified_embeddings"
        self.phase4_analysis_dir = self.batch6_dir / "phase4_analysis"
        
        # Create output directories
        for dir_path in [self.unified_embeddings_dir, self.phase4_analysis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Model configuration (consistent with batch5)
        self.model_name = 'all-MiniLM-L6-v2'
        self.embedding_dim = 384
        self.random_state = 2025
        
        # Expected data sizes for validation
        self.expected_data_sizes = {
            'real_malicious_background': 15000,  # Background malicious samples
            'real_malicious_seeds': 2000,        # Seed samples
            'batch6_synthetic': 6000,             # High-quality synthetic samples
            'real_benign': 13837,                 # Benign training samples
            'test_set': 7826                      # Fixed test set
        }
        
        self.logger.info("Batch 6 Phase 4 unified embedding initialization completed")
        self.logger.info(f"Embedding model: {self.model_name}")
        self.logger.info(f"Embedding dimension: {self.embedding_dim}")
        
    def load_all_source_data(self) -> Dict[str, pd.DataFrame]:
        """Load all source data for unified embedding space"""
        try:
            self.logger.info("Loading all source data for unified embedding...")
            
            data = {}
            
            # 1. Load real malicious background data (batch4)
            self.logger.info("Loading real malicious background data...")
            data['real_malicious_background'] = load_csv_data(
                self.batch4_dir / "train_malicious.csv", logger=self.logger
            )
            
            # 2. Load real malicious seed data (batch6 Phase 1)
            self.logger.info("Loading real malicious seed data...")
            data['real_malicious_seeds'] = load_csv_data(
                self.seed_prep_dir / "real_malicious_seeds_with_layers.csv", logger=self.logger
            )
            
            # 3. Load batch6 synthetic data (Phase 3)
            self.logger.info("Loading batch6 high-quality synthetic data...")
            data['batch6_synthetic'] = load_csv_data(
                self.synthetic_with_ids_dir / "all_synthetic_data.csv", logger=self.logger
            )
            
            # 4. Load real benign data (batch4)
            self.logger.info("Loading real benign training data...")
            data['real_benign'] = load_csv_data(
                self.batch4_dir / "train_benign.csv.gz", logger=self.logger
            )
            
            # 5. Load fixed test set (batch4)
            self.logger.info("Loading fixed test set...")
            data['test_set'] = load_csv_data(
                self.batch4_dir / "raw_test_set.csv", logger=self.logger
            )
            
            # Validate data sizes
            self.logger.info("Validating loaded data sizes...")
            validation_passed = True
            
            for data_type, expected_size in self.expected_data_sizes.items():
                if data_type in data:
                    actual_size = len(data[data_type])
                    if actual_size != expected_size:
                        self.logger.warning(f"{data_type}: Expected {expected_size}, got {actual_size}")
                        # Don't fail validation for minor differences
                        if abs(actual_size - expected_size) > expected_size * 0.05:  # 5% tolerance
                            validation_passed = False
                    else:
                        self.logger.info(f"✅ {data_type}: {actual_size} samples validated")
                else:
                    self.logger.error(f"Missing data type: {data_type}")
                    validation_passed = False
            
            if not validation_passed:
                self.logger.warning("Data size validation had warnings but continuing...")
            
            # Log summary
            total_samples = sum(len(df) for df in data.values())
            self.logger.info(f"✅ All source data loaded: {total_samples:,} total samples")
            self.logger.info(f"Data sources: {list(data.keys())}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load source data: {str(e)}")
            raise
            
    def create_unified_dataset(self, source_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create unified dataset with standardized columns and metadata"""
        try:
            self.logger.info("Creating unified dataset...")
            
            unified_samples = []
            
            # Process each data source
            for data_type, df in source_data.items():
                self.logger.info(f"Processing {data_type}: {len(df)} samples")
                
                # Standardize columns based on data type
                if data_type == 'real_malicious_background':
                    # Background malicious samples
                    processed_df = df.copy()
                    processed_df['data_category'] = 'real_malicious'
                    processed_df['data_source'] = 'background'
                    processed_df['layer'] = 'background'  # No layer classification for background
                    
                elif data_type == 'real_malicious_seeds':
                    # Seed samples with layer information
                    processed_df = df.copy()
                    processed_df['data_category'] = 'real_malicious'
                    processed_df['data_source'] = 'seeds'
                    # Keep original seed_layer as layer
                    if 'seed_layer' in processed_df.columns:
                        processed_df['layer'] = processed_df['seed_layer']
                    else:
                        processed_df['layer'] = 'unknown'
                    
                elif data_type == 'batch6_synthetic':
                    # High-quality synthetic samples (already processed in Phase 3)
                    processed_df = df.copy()
                    # Keep existing metadata from Phase 3
                    if 'data_category' not in processed_df.columns:
                        processed_df['data_category'] = 'synthetic'
                    if 'data_source' not in processed_df.columns:
                        processed_df['data_source'] = 'batch6_synthetic'
                    
                elif data_type == 'real_benign':
                    # Benign training samples
                    processed_df = df.copy()
                    processed_df['data_category'] = 'real_benign'
                    processed_df['data_source'] = 'training'
                    processed_df['layer'] = 'benign'  # Benign samples don't have layers
                    
                elif data_type == 'test_set':
                    # Fixed test set
                    processed_df = df.copy()
                    # Preserve original test set categories
                    if 'label' in processed_df.columns:
                        processed_df['data_category'] = processed_df['label'].apply(
                            lambda x: 'test_malicious' if x == 1 else 'test_benign'
                        )
                    else:
                        processed_df['data_category'] = 'test_unknown'
                    processed_df['data_source'] = 'test_set'
                    processed_df['layer'] = 'test'
                
                # Ensure required columns exist
                required_columns = ['unique_id', 'subject', 'body', 'text', 'label']
                
                for col in required_columns:
                    if col not in processed_df.columns:
                        if col == 'unique_id':
                            # Generate unique IDs if missing
                            processed_df['unique_id'] = [
                                f"{data_type}_{idx:06d}" for idx in range(len(processed_df))
                            ]
                        elif col == 'text':
                            # Combine subject and body if text is missing
                            processed_df['text'] = (
                                processed_df.get('subject', '').fillna('').astype(str) + ' ' +
                                processed_df.get('body', '').fillna('').astype(str)
                            )
                        else:
                            self.logger.warning(f"Missing required column {col} in {data_type}")
                            processed_df[col] = ''
                
                # Add batch and phase metadata
                processed_df['batch'] = 'batch6'
                processed_df['phase'] = 'phase4'
                processed_df['processing_timestamp'] = '2025-07-31'
                
                unified_samples.append(processed_df)
                self.logger.info(f"✅ Processed {data_type}: {len(processed_df)} samples")
            
            # Combine all data
            unified_df = pd.concat(unified_samples, ignore_index=True)
            
            # Final standardization
            standard_columns = [
                'unique_id', 'subject', 'body', 'text', 'label',
                'data_category', 'data_source', 'layer', 'batch', 'phase'
            ]
            
            # Keep only standard columns plus any additional metadata
            available_standard = [col for col in standard_columns if col in unified_df.columns]
            metadata_columns = [col for col in unified_df.columns if col not in standard_columns]
            
            final_columns = available_standard + metadata_columns
            unified_df = unified_df[final_columns]
            
            # Log final statistics
            self.logger.info(f"✅ Unified dataset created: {len(unified_df)} total samples")
            self.logger.info(f"Final columns: {list(unified_df.columns)}")
            
            # Data distribution analysis
            category_dist = unified_df['data_category'].value_counts()
            source_dist = unified_df['data_source'].value_counts()
            layer_dist = unified_df['layer'].value_counts()
            
            self.logger.info(f"Category distribution: {dict(category_dist)}")
            self.logger.info(f"Source distribution: {dict(source_dist)}")
            self.logger.info(f"Layer distribution: {dict(layer_dist)}")
            
            return unified_df
            
        except Exception as e:
            self.logger.error(f"Failed to create unified dataset: {str(e)}")
            raise
            
    def generate_unified_embeddings(self, unified_df: pd.DataFrame) -> np.ndarray:
        """Generate unified embedding vectors using SentenceTransformer"""
        try:
            self.logger.info("Starting unified embedding generation...")
            
            # Load embedding model
            self.logger.info(f"Loading embedding model: {self.model_name}")
            model = SentenceTransformer(self.model_name)
            
            # Prepare text data
            texts = unified_df['text'].fillna('').astype(str).tolist()
            total_texts = len(texts)
            
            self.logger.info(f"Preparing embedding computation: {total_texts:,} texts")
            
            # Batch processing for memory efficiency
            batch_size = 1000
            embeddings = []
            
            for i in range(0, total_texts, batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = model.encode(batch_texts, show_progress_bar=True)
                embeddings.append(batch_embeddings)
                
                progress = min(i + batch_size, total_texts)
                self.logger.info(f"Embedding progress: {progress:,}/{total_texts:,} ({progress/total_texts*100:.1f}%)")
            
            # Combine all embeddings
            all_embeddings = np.vstack(embeddings)
            
            self.logger.info(f"✅ Embedding generation completed: {all_embeddings.shape}")
            self.logger.info(f"Embedding dimensions: {all_embeddings.shape[1]} (expected: {self.embedding_dim})")
            
            return all_embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to generate unified embeddings: {str(e)}")
            raise
            
    def calculate_centroids_and_distances(self, unified_df: pd.DataFrame, 
                                        embeddings: np.ndarray) -> Dict[str, Any]:
        """Calculate centroids and distance statistics for layer analysis"""
        try:
            self.logger.info("Calculating centroids and distance statistics...")
            
            stats = {
                'centroids': {},
                'distance_statistics': {},
                'layer_analysis': {},
                'data_quality_metrics': {}
            }
            
            # Calculate centroids for each layer (core/edge for batch6)
            layers = ['core', 'edge']  # Simplified to 2 layers for batch6
            
            for layer in layers:
                layer_mask = unified_df['layer'] == layer
                if layer_mask.sum() > 0:
                    layer_embeddings = embeddings[layer_mask]
                    centroid = np.mean(layer_embeddings, axis=0)
                    stats['centroids'][layer] = centroid
                    
                    self.logger.info(f"Calculated centroid for {layer} layer: {layer_mask.sum()} samples")
                else:
                    self.logger.warning(f"No samples found for {layer} layer")
            
            # Calculate distances to centroids for all samples
            if 'core' in stats['centroids'] and 'edge' in stats['centroids']:
                # Calculate distances to both centroids
                core_centroid = stats['centroids']['core']
                edge_centroid = stats['centroids']['edge']
                
                # Distance to core centroid
                distances_to_core = cosine_distances(embeddings, core_centroid.reshape(1, -1)).flatten()
                # Distance to edge centroid
                distances_to_edge = cosine_distances(embeddings, edge_centroid.reshape(1, -1)).flatten()
                
                # Add distance information to unified_df
                unified_df['distance_to_core'] = distances_to_core
                unified_df['distance_to_edge'] = distances_to_edge
                
                # Determine closest centroid
                unified_df['closest_centroid'] = np.where(
                    distances_to_core < distances_to_edge, 'core', 'edge'
                )
                
                # Distance statistics
                stats['distance_statistics'] = {
                    'core_centroid_distances': {
                        'mean': float(np.mean(distances_to_core)),
                        'std': float(np.std(distances_to_core)),
                        'min': float(np.min(distances_to_core)),
                        'max': float(np.max(distances_to_core))
                    },
                    'edge_centroid_distances': {
                        'mean': float(np.mean(distances_to_edge)),
                        'std': float(np.std(distances_to_edge)),
                        'min': float(np.min(distances_to_edge)),
                        'max': float(np.max(distances_to_edge))
                    }
                }
                
                self.logger.info("✅ Centroid distance calculations completed")
            
            # Layer analysis for different data categories
            for category in unified_df['data_category'].unique():
                category_mask = unified_df['data_category'] == category
                category_data = unified_df[category_mask]
                
                if len(category_data) > 0:
                    layer_dist = category_data['layer'].value_counts()
                    stats['layer_analysis'][category] = {
                        'total_samples': len(category_data),
                        'layer_distribution': {k: int(v) for k, v in layer_dist.items()}
                    }
                    
                    if 'distance_to_core' in category_data.columns:
                        stats['layer_analysis'][category]['distance_stats'] = {
                            'mean_distance_to_core': float(category_data['distance_to_core'].mean()),
                            'mean_distance_to_edge': float(category_data['distance_to_edge'].mean())
                        }
            
            # Data quality metrics
            synthetic_mask = unified_df['data_category'] == 'synthetic'
            if synthetic_mask.sum() > 0:
                synthetic_data = unified_df[synthetic_mask]
                
                # Analyze synthetic data quality
                if 'distance_to_core' in synthetic_data.columns:
                    stats['data_quality_metrics']['synthetic_quality'] = {
                        'total_synthetic_samples': len(synthetic_data),
                        'core_synthetic_samples': len(synthetic_data[synthetic_data['layer'] == 'core']),
                        'edge_synthetic_samples': len(synthetic_data[synthetic_data['layer'] == 'edge']),
                        'avg_distance_to_core': float(synthetic_data['distance_to_core'].mean()),
                        'avg_distance_to_edge': float(synthetic_data['distance_to_edge'].mean())
                    }
            
            self.logger.info("✅ Centroids and distance statistics calculated")
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to calculate centroids and distances: {str(e)}")
            raise
            
    def save_unified_embedding_data(self, unified_df: pd.DataFrame, embeddings: np.ndarray,
                                   stats: Dict[str, Any]) -> Tuple[Path, Path, Path, Path]:
        """Save unified embedding data and metadata"""
        try:
            self.logger.info("Saving unified embedding data...")
            
            # Save embeddings as numpy array
            embeddings_file = self.unified_embeddings_dir / "all_embeddings.npy"
            np.save(embeddings_file, embeddings)
            self.logger.info(f"✅ Saved embeddings: {embeddings_file}")
            
            # Save metadata (compressed CSV)
            metadata_file = self.unified_embeddings_dir / "embedding_metadata.csv.gz"
            save_csv_data(unified_df, metadata_file, logger=self.logger)
            self.logger.info(f"✅ Saved metadata: {metadata_file}")
            
            # Save centroids
            centroids_file = self.unified_embeddings_dir / "layer_centroids.json"
            # Convert numpy arrays to lists for JSON serialization
            centroids_for_json = {
                layer: centroid.tolist() for layer, centroid in stats['centroids'].items()
            }
            with open(centroids_file, 'w', encoding='utf-8') as f:
                json.dump(centroids_for_json, f, ensure_ascii=False, indent=2)
            self.logger.info(f"✅ Saved centroids: {centroids_file}")
            
            # Save distance statistics
            distance_stats_file = self.unified_embeddings_dir / "distance_statistics.json"
            with open(distance_stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats['distance_statistics'], f, ensure_ascii=False, indent=2)
            self.logger.info(f"✅ Saved distance statistics: {distance_stats_file}")
            
            return embeddings_file, metadata_file, centroids_file, distance_stats_file
            
        except Exception as e:
            self.logger.error(f"Failed to save unified embedding data: {str(e)}")
            raise
            
    def generate_phase4_summary(self, unified_df: pd.DataFrame, embeddings: np.ndarray,
                               stats: Dict[str, Any]) -> None:
        """Generate comprehensive Phase 4 summary"""
        try:
            self.logger.info("Generating Phase 4 summary...")
            
            summary = {
                'phase4_completion': {
                    'status': 'completed',
                    'completion_date': '2025-07-31',
                    'total_runtime_minutes': 'TBD'
                },
                'embedding_configuration': {
                    'model_name': self.model_name,
                    'embedding_dimension': self.embedding_dim,
                    'batch_size': 1000,
                    'methodology': 'batch5_compatible'
                },
                'input_data_summary': {
                    'total_samples_processed': len(unified_df),
                    'data_sources': {k: int(v) for k, v in unified_df['data_source'].value_counts().items()},
                    'data_categories': {k: int(v) for k, v in unified_df['data_category'].value_counts().items()},
                    'layer_distribution': {k: int(v) for k, v in unified_df['layer'].value_counts().items()}
                },
                'embedding_results': {
                    'embeddings_generated': len(embeddings),
                    'embedding_shape': list(embeddings.shape),
                    'centroids_calculated': len(stats['centroids']),
                    'distance_calculations_complete': bool(stats['distance_statistics'])
                },
                'data_quality_analysis': stats.get('data_quality_metrics', {}),
                'layer_analysis': stats.get('layer_analysis', {}),
                'output_files': {
                    'embeddings_array': 'all_embeddings.npy',
                    'metadata_file': 'embedding_metadata.csv.gz',
                    'centroids_file': 'layer_centroids.json',
                    'distance_stats_file': 'distance_statistics.json'
                },
                'next_phase_readiness': {
                    'ready_for_phase5': True,
                    'visualization_data_prepared': True,
                    'improved_synthetic_quality_expected': True
                }
            }
            
            # Save summary
            summary_file = self.phase4_analysis_dir / "phase4_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"✅ Phase 4 summary saved: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate Phase 4 summary: {str(e)}")
            raise
            
    def run_phase4_complete(self) -> None:
        """Execute complete Phase 4 pipeline"""
        try:
            start_time = time.time()
            
            self.logger.info("="*60)
            self.logger.info("STARTING BATCH 6 PHASE 4: UNIFIED EMBEDDING SPACE CONSTRUCTION")
            self.logger.info("="*60)
            
            # Step 1: Load all source data
            self.logger.info("Step 1: Loading all source data...")
            source_data = self.load_all_source_data()
            
            # Step 2: Create unified dataset
            self.logger.info("Step 2: Creating unified dataset...")
            unified_df = self.create_unified_dataset(source_data)
            
            # Step 3: Generate unified embeddings
            self.logger.info("Step 3: Generating unified embeddings...")
            embeddings = self.generate_unified_embeddings(unified_df)
            
            # Step 4: Calculate centroids and distances
            self.logger.info("Step 4: Calculating centroids and distance statistics...")
            stats = self.calculate_centroids_and_distances(unified_df, embeddings)
            
            # Step 5: Save results
            self.logger.info("Step 5: Saving unified embedding data...")
            output_files = self.save_unified_embedding_data(unified_df, embeddings, stats)
            
            # Step 6: Generate summary
            self.logger.info("Step 6: Generating Phase 4 summary...")
            self.generate_phase4_summary(unified_df, embeddings, stats)
            
            # Final completion
            elapsed_time = time.time() - start_time
            
            self.logger.info("="*60)
            self.logger.info("BATCH 6 PHASE 4 COMPLETED SUCCESSFULLY!")
            self.logger.info("="*60)
            self.logger.info(f"Total runtime: {elapsed_time/60:.1f} minutes")
            self.logger.info(f"Total samples processed: {len(unified_df):,}")
            self.logger.info(f"Embeddings generated: {embeddings.shape}")
            self.logger.info(f"Output location: {self.unified_embeddings_dir}")
            self.logger.info("Ready for Phase 5: Enhanced visualization analysis")
            self.logger.info("="*60)
            
        except Exception as e:
            self.logger.error(f"Batch 6 Phase 4 failed: {str(e)}", exc_info=True)
            raise


def main():
    """Main function to run Batch 6 Phase 4"""
    logger = setup_logger("batch6_phase4")
    
    try:
        # Check if Phase 4 already completed
        batch6_dir = PROJECT_ROOT / "data" / "batch6"
        phase4_summary = batch6_dir / "phase4_analysis" / "phase4_summary.json"
        
        if phase4_summary.exists():
            logger.info(f"Phase 4 already completed! Summary found: {phase4_summary}")
            with open(phase4_summary, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            logger.info(f"Processed: {summary['input_data_summary']['total_samples_processed']} samples")
            logger.info("Phase 4 unified embedding already complete")
            return
        
        # Initialize and run Phase 4
        phase4 = Batch6Phase4UnifiedEmbedding(logger=logger)
        phase4.run_phase4_complete()
        
        logger.info("Batch 6 Phase 4 successfully completed!")
        
    except Exception as e:
        logger.error(f"Error in Batch 6 Phase 4: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()