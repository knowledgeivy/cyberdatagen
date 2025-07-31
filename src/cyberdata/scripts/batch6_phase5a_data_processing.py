#!/usr/bin/env python3
"""
Batch 6 Phase 5a: Data Processing and Dimensionality Reduction
==============================================================

Data processing stage - loads embeddings, performs dimensionality reduction and clustering.
Saves processed data for Phase 5b visualization stage.

Key features:
- Load unified embeddings from Phase 4
- Perform PCA 2D/3D and t-SNE 2D dimensionality reduction
- Execute K-means clustering analysis (K = 2-19)
- Save all processed data for visualization reuse
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Setup project root path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

def setup_logger(name: str) -> logging.Logger:
    """Setup logger for the phase"""
    log_dir = PROJECT_ROOT / "data" / "batch6" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    file_handler = logging.FileHandler(
        log_dir / f"batch6_phase5a_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class Batch6Phase5aDataProcessor:
    """Data processing and dimensionality reduction for batch6 enhanced analysis"""
    
    def __init__(self):
        self.logger = setup_logger("batch6_phase5a")
        self.setup_paths()
        self.setup_configurations()
        
    def setup_paths(self) -> None:
        """Setup directory paths"""
        self.batch6_dir = PROJECT_ROOT / "data" / "batch6"
        self.input_dir = self.batch6_dir / "unified_embeddings"
        
        # Output directory for processed data
        self.output_dir = self.batch6_dir / "phase5a_processed_data"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        
    def setup_configurations(self) -> None:
        """Setup processing configurations"""
        self.config = {
            'dimensionality_reduction': {
                'pca_2d': {'n_components': 2},
                'pca_3d': {'n_components': 3},
                'tsne_2d': {
                    'n_components': 2, 
                    'perplexity': 30, 
                    'random_state': 42,
                    'max_iter': 1000
                }
            },
            'clustering': {
                'kmeans': {
                    'k_range': list(range(2, 30)),  # K = 2 to 29 (wider range)
                    'random_state': 42,             # Fixed seed for reproducibility
                    'n_init': 20,                   # More initialization attempts
                    'init': 'k-means++',            # Better initialization method
                    'max_iter': 500,                # More iterations if needed
                    'tol': 1e-6                     # Tighter convergence tolerance
                }
            },
            'random_seed': 42
        }
        
        # Set random seeds
        np.random.seed(self.config['random_seed'])
        
        self.logger.info("Phase 5a configuration loaded")
        
    def load_embedding_data(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Load embeddings and metadata from Phase 4"""
        try:
            self.logger.info("Loading embedding data from Phase 4...")
            
            # Load embeddings
            embeddings_file = self.input_dir / "all_embeddings.npy"
            if not embeddings_file.exists():
                raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
            
            embeddings = np.load(embeddings_file)
            
            # Load metadata
            metadata_file = self.input_dir / "embedding_metadata.csv.gz"
            if not metadata_file.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
            
            metadata = pd.read_csv(metadata_file, compression='gzip')
            
            # Verify data consistency
            if len(embeddings) != len(metadata):
                raise ValueError(f"Embedding-metadata size mismatch: {len(embeddings)} vs {len(metadata)}")
            
            self.logger.info(f"✅ Loaded embeddings: {embeddings.shape}")
            self.logger.info(f"✅ Loaded metadata: {len(metadata)} records")
            self.logger.info(f"Data sources: {metadata['data_source'].value_counts().to_dict()}")
            
            return embeddings, metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load embedding data: {str(e)}")
            raise
            
    def perform_dimensionality_reduction(self, embeddings: np.ndarray) -> Dict[str, np.ndarray]:
        """Perform PCA and t-SNE dimensionality reduction"""
        try:
            self.logger.info("Performing dimensionality reduction...")
            
            dim_results = {}
            
            # PCA 2D
            self.logger.info("Running PCA 2D...")
            pca_2d = PCA(**self.config['dimensionality_reduction']['pca_2d'])
            dim_results['pca_2d'] = pca_2d.fit_transform(embeddings)
            dim_results['pca_2d_variance'] = pca_2d.explained_variance_ratio_
            
            # PCA 3D
            self.logger.info("Running PCA 3D...")
            pca_3d = PCA(**self.config['dimensionality_reduction']['pca_3d'])
            dim_results['pca_3d'] = pca_3d.fit_transform(embeddings)
            dim_results['pca_3d_variance'] = pca_3d.explained_variance_ratio_
            
            # t-SNE 2D
            self.logger.info("Running t-SNE 2D...")
            tsne_2d = TSNE(**self.config['dimensionality_reduction']['tsne_2d'])
            dim_results['tsne_2d'] = tsne_2d.fit_transform(embeddings)
            
            self.logger.info("✅ Dimensionality reduction completed")
            self.logger.info(f"PCA 2D variance explained: {dim_results['pca_2d_variance'].sum():.3f}")
            self.logger.info(f"PCA 3D variance explained: {dim_results['pca_3d_variance'].sum():.3f}")
            
            return dim_results
            
        except Exception as e:
            self.logger.error(f"Failed dimensionality reduction: {str(e)}")
            raise
            
    def perform_clustering_analysis(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Perform K-means clustering analysis"""
        try:
            self.logger.info("Performing K-means clustering analysis...")
            
            cluster_results = {
                'kmeans': {},
                'optimal_k': None,
                'best_silhouette': -1
            }
            
            k_values = self.config['clustering']['kmeans']['k_range']
            self.logger.info(f"Testing K values: {k_values}")
            
            for k in k_values:
                try:
                    kmeans = KMeans(
                        n_clusters=k, 
                        random_state=self.config['clustering']['kmeans']['random_state'],
                        n_init=self.config['clustering']['kmeans']['n_init'],
                        init=self.config['clustering']['kmeans']['init'],
                        max_iter=self.config['clustering']['kmeans']['max_iter'],
                        tol=self.config['clustering']['kmeans']['tol']
                    )
                    clusters = kmeans.fit_predict(embeddings)
                    
                    # Calculate silhouette score if we have enough samples
                    if len(np.unique(clusters)) > 1 and len(embeddings) > k:
                        silhouette = silhouette_score(embeddings, clusters)
                    else:
                        silhouette = -1
                    
                    cluster_results['kmeans'][k] = {
                        'labels': clusters,
                        'centroids': kmeans.cluster_centers_,
                        'inertia': kmeans.inertia_,
                        'silhouette_score': silhouette
                    }
                    
                    # Track best silhouette score
                    if silhouette > cluster_results['best_silhouette']:
                        cluster_results['best_silhouette'] = silhouette
                        cluster_results['optimal_k'] = k
                    
                    self.logger.info(f"K={k}: Silhouette={silhouette:.4f}, Inertia={kmeans.inertia_:.2f}")
                    
                except Exception as e:
                    self.logger.error(f"Failed clustering with K={k}: {str(e)}")
                    continue
            
            if cluster_results['optimal_k'] is not None:
                self.logger.info(f"✅ Optimal K: {cluster_results['optimal_k']} (Silhouette: {cluster_results['best_silhouette']:.4f})")
            else:
                self.logger.warning("⚠️ No valid clustering results obtained")
            
            return cluster_results
            
        except Exception as e:
            self.logger.error(f"Failed clustering analysis: {str(e)}")
            raise
            
    def calculate_distance_statistics(self, embeddings: np.ndarray, metadata: pd.DataFrame) -> Dict[str, Any]:
        """Calculate distance statistics from centroids"""
        try:
            self.logger.info("Calculating distance statistics...")
            
            # Load centroid information from Phase 4
            centroids_file = self.input_dir / "layer_centroids.json"
            if centroids_file.exists():
                with open(centroids_file, 'r') as f:
                    centroids_data = json.load(f)
                
                distance_stats = centroids_data.copy()
                
                # Add distance calculations to metadata if not already present
                if 'distance_to_core' not in metadata.columns:
                    if 'core_centroid' in centroids_data:
                        core_centroid = np.array(centroids_data['core_centroid'])
                        distances_to_core = np.linalg.norm(embeddings - core_centroid, axis=1)
                        metadata['distance_to_core'] = distances_to_core
                
                if 'distance_to_edge' not in metadata.columns:
                    if 'edge_centroid' in centroids_data:
                        edge_centroid = np.array(centroids_data['edge_centroid'])
                        distances_to_edge = np.linalg.norm(embeddings - edge_centroid, axis=1)
                        metadata['distance_to_edge'] = distances_to_edge
                
                self.logger.info("✅ Distance statistics calculated")
                return distance_stats
                
            else:
                self.logger.warning("⚠️ Centroids file not found, skipping distance calculations")
                return {}
                
        except Exception as e:
            self.logger.error(f"Failed distance calculations: {str(e)}")
            return {}
            
    def save_processed_data(self, embeddings: np.ndarray, metadata: pd.DataFrame, 
                           dim_results: Dict[str, np.ndarray], cluster_results: Dict[str, Any],
                           distance_stats: Dict[str, Any]) -> None:
        """Save all processed data for Phase 5b"""
        try:
            self.logger.info("Saving processed data...")
            
            # Save embeddings and metadata
            np.save(self.output_dir / "embeddings.npy", embeddings)
            metadata.to_csv(self.output_dir / "metadata.csv.gz", compression='gzip', index=False)
            
            # Save dimensionality reduction results
            with open(self.output_dir / "dimensionality_reduction_results.pkl", 'wb') as f:
                pickle.dump(dim_results, f)
            
            # Save clustering results (convert numpy arrays to lists for JSON)
            cluster_results_json = {}
            for key, value in cluster_results.items():
                if key == 'kmeans':
                    cluster_results_json[key] = {}
                    for k, k_results in value.items():
                        cluster_results_json[key][k] = {
                            'labels': k_results['labels'].tolist(),
                            'centroids': k_results['centroids'].tolist(),
                            'inertia': float(k_results['inertia']),
                            'silhouette_score': float(k_results['silhouette_score'])
                        }
                else:
                    cluster_results_json[key] = value
            
            with open(self.output_dir / "clustering_results.json", 'w') as f:
                json.dump(cluster_results_json, f, indent=2)
            
            # Save distance statistics
            with open(self.output_dir / "distance_statistics.json", 'w') as f:
                json.dump(distance_stats, f, indent=2)
            
            # Save processing summary
            processing_summary = {
                'phase5a_completion': {
                    'status': 'completed',
                    'completion_date': datetime.now().isoformat(),
                    'data_shape': {
                        'embeddings': embeddings.shape,
                        'metadata_records': len(metadata)
                    }
                },
                'dimensionality_reduction': {
                    'pca_2d_variance_explained': float(dim_results['pca_2d_variance'].sum()),
                    'pca_3d_variance_explained': float(dim_results['pca_3d_variance'].sum()),
                    'methods_completed': ['PCA_2D', 'PCA_3D', 't-SNE_2D']
                },
                'clustering_analysis': {
                    'optimal_k': cluster_results['optimal_k'],
                    'best_silhouette_score': float(cluster_results['best_silhouette']),
                    'k_values_tested': len(cluster_results['kmeans']),
                    'k_range': f"{min(cluster_results['kmeans'].keys())}-{max(cluster_results['kmeans'].keys())}"
                },
                'data_sources': metadata['data_source'].value_counts().to_dict(),
                'output_files': {
                    'embeddings': str(self.output_dir / "embeddings.npy"),
                    'metadata': str(self.output_dir / "metadata.csv.gz"),
                    'dimensionality_reduction': str(self.output_dir / "dimensionality_reduction_results.pkl"),
                    'clustering_results': str(self.output_dir / "clustering_results.json"),
                    'distance_statistics': str(self.output_dir / "distance_statistics.json")
                }
            }
            
            with open(self.output_dir / "phase5a_processing_summary.json", 'w') as f:
                json.dump(processing_summary, f, indent=2)
            
            self.logger.info("✅ All processed data saved successfully")
            self.logger.info(f"Output directory: {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save processed data: {str(e)}")
            raise
            
    def run_phase5a(self) -> None:
        """Execute complete Phase 5a data processing"""
        try:
            self.logger.info("🚀 Starting Batch 6 Phase 5a: Data Processing & Dimensionality Reduction")
            start_time = datetime.now()
            
            # Step 1: Load embedding data
            embeddings, metadata = self.load_embedding_data()
            
            # Step 2: Perform dimensionality reduction
            dim_results = self.perform_dimensionality_reduction(embeddings)
            
            # Step 3: Perform clustering analysis
            cluster_results = self.perform_clustering_analysis(embeddings)
            
            # Step 4: Calculate distance statistics
            distance_stats = self.calculate_distance_statistics(embeddings, metadata)
            
            # Step 5: Save all processed data
            self.save_processed_data(embeddings, metadata, dim_results, cluster_results, distance_stats)
            
            # Phase completion
            end_time = datetime.now()
            runtime = (end_time - start_time).total_seconds() / 60
            
            self.logger.info(f"✅ Batch 6 Phase 5a completed successfully!")
            self.logger.info(f"Runtime: {runtime:.2f} minutes")
            self.logger.info(f"Processed {len(embeddings)} samples")
            if cluster_results['optimal_k']:
                self.logger.info(f"Optimal K-means clusters: {cluster_results['optimal_k']}")
            self.logger.info("Ready for Phase 5b (Visualization Stage)")
            
        except Exception as e:
            self.logger.error(f"❌ Batch 6 Phase 5a failed: {str(e)}")
            raise

def main():
    """Main function to run Batch 6 Phase 5a"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Batch 6 Phase 5a: Data Processing & Dimensionality Reduction')
    parser.add_argument('--clean', action='store_true', help='Clean previous processing results before running')
    args = parser.parse_args()
    
    logger = setup_logger("batch6_phase5a")
    
    try:
        # Handle clean option
        batch6_dir = PROJECT_ROOT / "data" / "batch6"
        if args.clean:
            logger.info("🧹 Cleaning previous Phase 5a results...")
            
            dirs_to_clean = [
                batch6_dir / "phase5a_processed_data"
            ]
            
            for dir_path in dirs_to_clean:
                if dir_path.exists():
                    import shutil
                    shutil.rmtree(dir_path)
                    logger.info(f"   Removed: {dir_path}")
                    
            logger.info("✅ Cleanup completed!")
        
        # Run Phase 5a
        processor = Batch6Phase5aDataProcessor()
        processor.run_phase5a()
        
    except Exception as e:
        logger.error(f"Phase 5a execution failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()