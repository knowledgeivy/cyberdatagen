#!/usr/bin/env python3
"""
Batch 6 Individual Paper Plots Generator
=======================================

Based on batch6_phase5_enhanced_visualization.py, this script generates individual
plots instead of subplot combinations. All data processing logic remains identical,
only the plotting output is modified to create separate figures.

Output: data/batch6/paper/ directory with individual PNG files
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

# Visualization libraries
import matplotlib
try:
    matplotlib.use('Agg')  # Set backend to Agg to avoid display issues
except Exception as e:
    print(f"Warning: Could not set matplotlib backend to Agg: {e}")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Project path setup
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.cyberdata.utils.logger_config import setup_logger

class Batch6IndividualPaperPlots:
    """Batch 6 Individual Paper Plots Generator - Extract subplots as individual figures"""

    def __init__(self, logger=None):
        self.logger = logger or setup_logger("batch6_paper_plots")
        self.project_root = PROJECT_ROOT
        self.batch6_dir = self.project_root / "data" / "batch6"

        # Input directories (same as original)
        self.unified_embeddings_dir = self.batch6_dir / "unified_embeddings"

        # Output directory for paper figures
        self.paper_output_dir = self.batch6_dir / "paper"
        self.paper_output_dir.mkdir(parents=True, exist_ok=True)

        # Visualization configuration (same as original)
        self.random_state = 2025
        self.figure_size = (10, 8)  # Individual figure size

        # Color palette (identical to original batch6)
        self.color_palette = {
            # Data sources
            'background': '#FF6B6B',
            'seeds': '#4ECDC4',
            'batch6_synthetic': '#45B7D1',
            'training': '#96CEB4',
            'test_set': '#FECA57',

            # Layers
            'core': '#E74C3C',
            'edge': '#3498DB',
            'background': '#95A5A6',
            'benign': '#27AE60',
            'test': '#F39C12',

            # Prompt types
            'original': '#9B59B6',
            'strong': '#E67E22',
            'weak': '#2ECC71'
        }

        self.logger.info("Batch 6 Individual Paper Plots initialization completed")

    def load_embedding_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load unified embedding data from Phase 4 (identical to original)"""
        try:
            self.logger.info("Loading unified embedding data from Phase 4...")

            # Load metadata
            metadata_file = self.unified_embeddings_dir / "embedding_metadata.csv.gz"
            if not metadata_file.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

            metadata_df = pd.read_csv(metadata_file, compression='gzip')
            self.logger.info(f"✅ Loaded metadata: {len(metadata_df)} samples")

            # Load embeddings
            embeddings_file = self.unified_embeddings_dir / "all_embeddings.npy"
            if not embeddings_file.exists():
                raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")

            embeddings = np.load(embeddings_file)
            self.logger.info(f"✅ Loaded embeddings: {embeddings.shape}")

            # Validation
            if len(metadata_df) != len(embeddings):
                raise ValueError(f"Metadata and embeddings size mismatch: {len(metadata_df)} vs {len(embeddings)}")

            return metadata_df, embeddings

        except Exception as e:
            self.logger.error(f"Failed to load embedding data: {str(e)}")
            raise

    def perform_dimensionality_reduction(self, embeddings: np.ndarray) -> Dict[str, np.ndarray]:
        """Perform PCA and t-SNE dimensionality reduction (identical to original)"""
        try:
            self.logger.info("Performing dimensionality reduction...")

            results = {}

            # PCA 2D
            pca_2d = PCA(n_components=2, random_state=self.random_state)
            results['pca_2d'] = pca_2d.fit_transform(embeddings)
            results['pca_2d_variance'] = pca_2d.explained_variance_ratio_

            # t-SNE 2D
            try:
                pca_50 = PCA(n_components=50, random_state=self.random_state)
                embeddings_50d = pca_50.fit_transform(embeddings)

                tsne_2d = TSNE(n_components=2, random_state=self.random_state,
                              perplexity=min(30, len(embeddings)//4), max_iter=500, learning_rate=200.0,
                              n_jobs=1)
                results['tsne_2d'] = tsne_2d.fit_transform(embeddings_50d)
            except Exception as e:
                self.logger.error(f"t-SNE computation failed: {str(e)}")
                results['tsne_2d'] = results['pca_2d']

            return results

        except Exception as e:
            self.logger.error(f"Failed to perform dimensionality reduction: {str(e)}")
            raise

    def perform_clustering_analysis(self, embeddings: np.ndarray,
                                   metadata_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform clustering analysis (identical to original)"""
        try:
            self.logger.info("Performing clustering analysis...")

            results = {
                'kmeans': {},
                'quality_metrics': {}
            }

            # K-means clustering with different k values
            k_values = list(range(2, 20))
            max_k = min(len(embeddings) - 1, max(k_values))
            k_values = [k for k in k_values if k <= max_k]

            for k in k_values:
                try:
                    kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                    clusters = kmeans.fit_predict(embeddings)

                    if len(set(clusters)) < 2:
                        continue

                    silhouette = silhouette_score(embeddings, clusters)
                except Exception as e:
                    continue

                results['kmeans'][k] = {
                    'clusters': clusters,
                    'centroids': kmeans.cluster_centers_,
                    'silhouette_score': silhouette,
                    'inertia': kmeans.inertia_
                }

            # Find optimal k
            best_k = max(results['kmeans'].keys(),
                        key=lambda k: results['kmeans'][k]['silhouette_score'])
            results['optimal_k'] = best_k

            results['quality_metrics'] = {
                'best_kmeans_silhouette': results['kmeans'][best_k]['silhouette_score'],
                'optimal_k': best_k
            }

            return results

        except Exception as e:
            self.logger.error(f"Failed to perform clustering analysis: {str(e)}")
            raise

    def create_individual_plots(self, metadata_df: pd.DataFrame, dim_results: Dict[str, np.ndarray],
                               cluster_results: Dict[str, Any], embeddings: np.ndarray) -> None:
        """Create individual plots by extracting each subplot from original combinations"""
        try:
            self.logger.info("Creating individual paper plots...")

            # Set matplotlib style
            plt.style.use('default')
            sns.set_palette("husl")

            # ========================================
            # FROM COMPREHENSIVE OVERVIEW (2x2 subplot) -> 4 individual plots
            # ========================================

            # Plot 1: PCA by data source
            plt.figure(figsize=self.figure_size)
            for source in metadata_df['data_source'].unique():
                mask = metadata_df['data_source'] == source
                if mask.sum() > 0:
                    color = self.color_palette.get(source, '#888888')
                    plt.scatter(dim_results['pca_2d'][mask, 0], dim_results['pca_2d'][mask, 1],
                               c=color, label=source, alpha=0.6, s=20)

            plt.xlabel(f'PC1 ({dim_results["pca_2d_variance"][0]:.1%} variance)')
            plt.ylabel(f'PC2 ({dim_results["pca_2d_variance"][1]:.1%} variance)')
            plt.title('PCA 2D - By Data Source')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.paper_output_dir / "01_pca_by_data_source.png", dpi=300, bbox_inches='tight')
            plt.close()

            # Plot 2: PCA by layer
            plt.figure(figsize=self.figure_size)
            for layer in ['core', 'edge', 'background', 'benign', 'test']:
                mask = metadata_df['layer'] == layer
                if mask.sum() > 0:
                    color = self.color_palette.get(layer, '#888888')
                    plt.scatter(dim_results['pca_2d'][mask, 0], dim_results['pca_2d'][mask, 1],
                               c=color, label=layer, alpha=0.6, s=20)

            plt.xlabel(f'PC1 ({dim_results["pca_2d_variance"][0]:.1%} variance)')
            plt.ylabel(f'PC2 ({dim_results["pca_2d_variance"][1]:.1%} variance)')
            plt.title('PCA 2D - By Layer (Batch6 Simplified)')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.paper_output_dir / "02_pca_by_layer.png", dpi=300, bbox_inches='tight')
            plt.close()

            # Plot 3: t-SNE by data category
            plt.figure(figsize=self.figure_size)
            for category in metadata_df['data_category'].unique():
                mask = metadata_df['data_category'] == category
                if mask.sum() > 0:
                    plt.scatter(dim_results['tsne_2d'][mask, 0], dim_results['tsne_2d'][mask, 1],
                               label=category, alpha=0.6, s=20)

            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.title('t-SNE 2D - By Data Category')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.paper_output_dir / "03_tsne_by_category.png", dpi=300, bbox_inches='tight')
            plt.close()

            # Plot 4: K-means clustering
            plt.figure(figsize=self.figure_size)
            best_k = cluster_results['optimal_k']
            clusters = cluster_results['kmeans'][best_k]['clusters']
            scatter = plt.scatter(dim_results['pca_2d'][:, 0], dim_results['pca_2d'][:, 1],
                                 c=clusters, cmap='tab10', alpha=0.6, s=20)
            plt.xlabel(f'PC1 ({dim_results["pca_2d_variance"][0]:.1%} variance)')
            plt.ylabel(f'PC2 ({dim_results["pca_2d_variance"][1]:.1%} variance)')
            plt.title(f'K-means Clustering (k={best_k})')
            plt.colorbar(scatter)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.paper_output_dir / "04_kmeans_clustering.png", dpi=300, bbox_inches='tight')
            plt.close()

            # ========================================
            # FROM SYNTHETIC QUALITY ANALYSIS (2x2 subplot) -> 4 individual plots
            # ========================================

            synthetic_mask = metadata_df['data_category'] == 'synthetic'
            if synthetic_mask.sum() > 0:

                # Plot 5: PCA by prompt variant
                plt.figure(figsize=self.figure_size)
                synthetic_data = metadata_df[synthetic_mask]
                if 'prompt_variant' in synthetic_data.columns:
                    for prompt in ['original', 'strong', 'weak']:
                        prompt_mask = (synthetic_mask) & (metadata_df['prompt_variant'] == prompt)
                        if prompt_mask.sum() > 0:
                            color = self.color_palette.get(prompt, '#888888')
                            plt.scatter(dim_results['pca_2d'][prompt_mask, 0], dim_results['pca_2d'][prompt_mask, 1],
                                       c=color, label=f'{prompt} ({prompt_mask.sum()})', alpha=0.7, s=30)

                plt.xlabel(f'PC1 ({dim_results["pca_2d_variance"][0]:.1%} variance)')
                plt.ylabel(f'PC2 ({dim_results["pca_2d_variance"][1]:.1%} variance)')
                plt.title('Synthetic Data - By Prompt Strategy')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.paper_output_dir / "05_synthetic_by_prompt.png", dpi=300, bbox_inches='tight')
                plt.close()

                # Plot 6: Seeds vs Synthetic by Prompt Strategy
                plt.figure(figsize=self.figure_size)
                seeds_mask = metadata_df['data_source'] == 'seeds'
                if seeds_mask.sum() > 0:
                    plt.scatter(dim_results['pca_2d'][seeds_mask, 0], dim_results['pca_2d'][seeds_mask, 1],
                               c='#FF4444', label=f'Real Malicious Seeds ({seeds_mask.sum()})',
                               alpha=0.8, s=60, marker='s', edgecolors='black', linewidth=0.5)

                if 'prompt_variant' in metadata_df.columns:
                    for prompt in ['original', 'strong', 'weak']:
                        prompt_mask = (synthetic_mask) & (metadata_df['prompt_variant'] == prompt)
                        if prompt_mask.sum() > 0:
                            color = self.color_palette.get(prompt, '#888888')
                            plt.scatter(dim_results['pca_2d'][prompt_mask, 0], dim_results['pca_2d'][prompt_mask, 1],
                                       c=color, label=f'Synthetic {prompt.title()} ({prompt_mask.sum()})',
                                       alpha=0.7, s=30, marker='o')

                plt.xlabel(f'PC1 ({dim_results["pca_2d_variance"][0]:.1%} variance)')
                plt.ylabel(f'PC2 ({dim_results["pca_2d_variance"][1]:.1%} variance)')
                plt.title('Real Seeds vs Synthetic Data by Prompt Strategy')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.paper_output_dir / "06_seeds_vs_synthetic_prompt.png", dpi=300, bbox_inches='tight')
                plt.close()

                # Plot 7: t-SNE synthetic data distribution
                plt.figure(figsize=self.figure_size)
                if 'prompt_variant' in synthetic_data.columns:
                    for prompt in ['original', 'strong', 'weak']:
                        prompt_mask = (synthetic_mask) & (metadata_df['prompt_variant'] == prompt)
                        if prompt_mask.sum() > 0:
                            color = self.color_palette.get(prompt, '#888888')
                            plt.scatter(dim_results['tsne_2d'][prompt_mask, 0], dim_results['tsne_2d'][prompt_mask, 1],
                                       c=color, label=f'{prompt}', alpha=0.7, s=30)

                plt.xlabel('t-SNE 1')
                plt.ylabel('t-SNE 2')
                plt.title('t-SNE - Synthetic Data Prompt Strategies')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.paper_output_dir / "07_tsne_synthetic_prompt.png", dpi=300, bbox_inches='tight')
                plt.close()

            # ========================================
            # FROM CLUSTERING QUALITY ANALYSIS (1x2 subplot) -> 2 individual plots
            # ========================================

            # Plot 8: K-means silhouette scores
            plt.figure(figsize=self.figure_size)
            k_values = sorted(list(cluster_results['kmeans'].keys()))
            silhouette_scores = [cluster_results['kmeans'][k]['silhouette_score'] for k in k_values]

            plt.plot(k_values, silhouette_scores, 'bo-', linewidth=2, markersize=6)

            # Add annotation for best K
            if silhouette_scores:
                best_k_idx = np.argmax(silhouette_scores)
                best_k = k_values[best_k_idx]
                best_score = silhouette_scores[best_k_idx]
                plt.annotate(f'Best K={best_k}\\n({best_score:.3f})',
                           xy=(best_k, best_score), xytext=(10, 10),
                           textcoords='offset points', fontsize=10,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Silhouette Score')
            plt.title('K-means Clustering Quality')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.paper_output_dir / "08_silhouette_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()

            # Plot 9: K-means inertia (elbow method)
            plt.figure(figsize=self.figure_size)
            inertia_values = [cluster_results['kmeans'][k]['inertia'] for k in k_values]

            plt.plot(k_values, inertia_values, 'go-', linewidth=2, markersize=6)
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Inertia')
            plt.title('K-means Inertia (Elbow Method)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.paper_output_dir / "09_elbow_method.png", dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info("✅ Individual paper plots created")

        except Exception as e:
            self.logger.error(f"Failed to create individual plots: {str(e)}")
            raise

    def run_individual_plot_generation(self) -> None:
        """Execute individual plot generation"""
        try:
            start_time = time.time()

            self.logger.info("="*60)
            self.logger.info("STARTING BATCH 6 INDIVIDUAL PAPER PLOTS GENERATION")
            self.logger.info("="*60)

            # Step 1: Load embedding data
            metadata_df, embeddings = self.load_embedding_data()

            # Step 2: Perform dimensionality reduction
            dim_results = self.perform_dimensionality_reduction(embeddings)

            # Step 3: Perform clustering analysis
            cluster_results = self.perform_clustering_analysis(embeddings, metadata_df)

            # Step 4: Create individual plots
            self.create_individual_plots(metadata_df, dim_results, cluster_results, embeddings)

            # Final completion
            elapsed_time = time.time() - start_time
            png_files = list(self.paper_output_dir.glob('*.png'))

            self.logger.info("="*60)
            self.logger.info("BATCH 6 INDIVIDUAL PAPER PLOTS COMPLETED!")
            self.logger.info("="*60)
            self.logger.info(f"Total runtime: {elapsed_time/60:.1f} minutes")
            self.logger.info(f"Individual plots created: {len(png_files)}")
            self.logger.info(f"Output location: {self.paper_output_dir}")
            for f in sorted(png_files):
                self.logger.info(f"  - {f.name}")
            self.logger.info("="*60)

        except Exception as e:
            self.logger.error(f"Individual plot generation failed: {str(e)}", exc_info=True)
            raise

def main():
    """Main function"""
    logger = setup_logger("batch6_paper_plots")

    try:
        generator = Batch6IndividualPaperPlots(logger=logger)
        generator.run_individual_plot_generation()

        logger.info("Batch 6 Individual Paper Plots generation successfully completed!")

    except Exception as e:
        logger.error(f"Error in Batch 6 Individual Paper Plots generation: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()