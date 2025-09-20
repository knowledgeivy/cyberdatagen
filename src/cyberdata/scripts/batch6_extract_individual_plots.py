#!/usr/bin/env python3
"""
Batch 6 Extract Individual Plots
===============================

Extract individual plots from batch6 processed data, avoiding re-computation.
Uses existing phase5a processed data to create individual figures.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

class Batch6ExtractIndividualPlots:
    """Extract individual plots using batch6 processed data"""

    def __init__(self):
        self.setup_paths()
        self.setup_styling()
        self.setup_logging()

    def setup_paths(self):
        """Setup paths"""
        self.batch6_dir = PROJECT_ROOT / "data" / "batch6"
        self.input_dir = self.batch6_dir / "phase5a_processed_data"
        self.output_dir = self.batch6_dir / "paper"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup_styling(self):
        """Setup academic styling identical to batch6"""
        plt.rcParams.update({
            'figure.figsize': [10, 8],
            'font.size': 12,
            'font.family': 'serif',
            'axes.linewidth': 1.0,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'legend.frameon': True,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })

        # Batch6 original color palette
        self.color_palette = {
            'background': '#FF6B6B',
            'seeds': '#4ECDC4',
            'batch6_synthetic': '#45B7D1',
            'training': '#96CEB4',
            'test_set': '#FECA57',
            'core': '#E74C3C',
            'edge': '#3498DB',
            'benign': '#27AE60',
            'test': '#F39C12',
            'original': '#9B59B6',
            'strong': '#E67E22',
            'weak': '#2ECC71'
        }

    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def load_processed_data(self):
        """Load batch6 processed data"""
        self.logger.info("Loading processed data from batch6...")

        try:
            # Load metadata
            metadata = pd.read_csv(self.input_dir / "metadata.csv.gz", compression='gzip')

            # Load dimensionality reduction results
            with open(self.input_dir / "dimensionality_reduction_results.pkl", 'rb') as f:
                dim_results = pickle.load(f)

            # Load clustering results
            with open(self.input_dir / "clustering_results.json", 'r') as f:
                cluster_results_raw = json.load(f)

            # Convert clustering results to proper format
            cluster_results = {
                'optimal_k': cluster_results_raw['optimal_k'],
                'kmeans': {}
            }

            for k, k_results in cluster_results_raw['kmeans'].items():
                cluster_results['kmeans'][int(k)] = {
                    'clusters': np.array(k_results['labels']),
                    'silhouette_score': k_results['silhouette_score'],
                    'inertia': k_results['inertia']
                }

            self.logger.info(f"Loaded: {len(metadata)} samples")
            return metadata, dim_results, cluster_results

        except Exception as e:
            self.logger.error(f"Failed to load processed data: {str(e)}")
            raise

    def create_individual_plots(self, metadata, dim_results, cluster_results):
        """Create individual plots"""
        self.logger.info("Creating individual plots...")

        # Plot 1: PCA by data source
        plt.figure(figsize=(10, 8))
        for source in metadata['data_source'].unique():
            mask = metadata['data_source'] == source
            if mask.sum() > 0:
                color = self.color_palette.get(source, '#888888')
                plt.scatter(dim_results['pca_2d'][mask, 0], dim_results['pca_2d'][mask, 1],
                           c=color, label=f'{source} ({mask.sum()})', alpha=0.6, s=20)

        plt.xlabel(f'PC1 ({dim_results["pca_2d_variance"][0]:.1%} variance)')
        plt.ylabel(f'PC2 ({dim_results["pca_2d_variance"][1]:.1%} variance)')
        plt.title('PCA 2D - By Data Source')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "01_pca_by_data_source.png", dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info("✅ PCA by data source")

        # Plot 2: PCA by layer
        plt.figure(figsize=(10, 8))
        for layer in ['core', 'edge', 'background', 'benign', 'test']:
            mask = metadata['layer'] == layer
            if mask.sum() > 0:
                color = self.color_palette.get(layer, '#888888')
                plt.scatter(dim_results['pca_2d'][mask, 0], dim_results['pca_2d'][mask, 1],
                           c=color, label=f'{layer} ({mask.sum()})', alpha=0.6, s=20)

        plt.xlabel(f'PC1 ({dim_results["pca_2d_variance"][0]:.1%} variance)')
        plt.ylabel(f'PC2 ({dim_results["pca_2d_variance"][1]:.1%} variance)')
        plt.title('PCA 2D - By Layer')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "02_pca_by_layer.png", dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info("✅ PCA by layer")

        # Plot 3: t-SNE by data category
        plt.figure(figsize=(10, 8))
        for category in metadata['data_category'].unique():
            mask = metadata['data_category'] == category
            if mask.sum() > 0:
                plt.scatter(dim_results['tsne_2d'][mask, 0], dim_results['tsne_2d'][mask, 1],
                           label=f'{category} ({mask.sum()})', alpha=0.6, s=20)

        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title('t-SNE 2D - By Data Category')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "03_tsne_by_category.png", dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info("✅ t-SNE by category")

        # Plot 4: K-means clustering
        plt.figure(figsize=(10, 8))
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
        plt.savefig(self.output_dir / "04_kmeans_clustering.png", dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info("✅ K-means clustering")

        # Plot 5: Seeds vs Synthetic
        plt.figure(figsize=(12, 8))
        seeds_mask = metadata['data_source'] == 'seeds'
        synthetic_mask = metadata['data_category'] == 'synthetic'

        # Real seeds
        if seeds_mask.sum() > 0:
            plt.scatter(dim_results['pca_2d'][seeds_mask, 0], dim_results['pca_2d'][seeds_mask, 1],
                       c='#FF4444', label=f'Real Malicious Seeds ({seeds_mask.sum()})',
                       alpha=0.8, s=60, marker='s', edgecolors='black', linewidth=0.5)

        # Synthetic by prompt
        if 'prompt_variant' in metadata.columns:
            for prompt in ['original', 'strong', 'weak']:
                prompt_mask = synthetic_mask & (metadata['prompt_variant'] == prompt)
                if prompt_mask.sum() > 0:
                    color = self.color_palette.get(prompt, '#888888')
                    plt.scatter(dim_results['pca_2d'][prompt_mask, 0], dim_results['pca_2d'][prompt_mask, 1],
                               c=color, label=f'Synthetic {prompt.title()} ({prompt_mask.sum()})',
                               alpha=0.7, s=30, marker='o')

        plt.xlabel(f'PC1 ({dim_results["pca_2d_variance"][0]:.1%} variance)')
        plt.ylabel(f'PC2 ({dim_results["pca_2d_variance"][1]:.1%} variance)')
        plt.title('Real Seeds vs Synthetic Data by Prompt Strategy')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "05_seeds_vs_synthetic.png", dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info("✅ Seeds vs synthetic")

        # Plot 6: Silhouette analysis
        plt.figure(figsize=(10, 8))
        k_values = sorted(cluster_results['kmeans'].keys())
        silhouette_scores = [cluster_results['kmeans'][k]['silhouette_score'] for k in k_values]

        plt.plot(k_values, silhouette_scores, 'o-', linewidth=2, markersize=6, color='#E74C3C')

        # Highlight optimal K
        optimal_k = cluster_results['optimal_k']
        optimal_score = cluster_results['kmeans'][optimal_k]['silhouette_score']
        plt.plot(optimal_k, optimal_score, 'o', markersize=10, color='#F39C12',
                markeredgecolor='black', markeredgewidth=2)

        plt.annotate(f'Optimal K={optimal_k}\\n(Score={optimal_score:.3f})',
                    xy=(optimal_k, optimal_score),
                    xytext=(optimal_k + 2, optimal_score - 0.01),
                    arrowprops=dict(arrowstyle='->', color='black'),
                    fontsize=10, ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Silhouette Score')
        plt.title('K-means Clustering Quality - Silhouette Analysis')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "06_silhouette_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info("✅ Silhouette analysis")

    def run_extraction(self):
        """Run extraction"""
        try:
            self.logger.info("🚀 Starting Batch6 Individual Plot Extraction")

            # Load processed data
            metadata, dim_results, cluster_results = self.load_processed_data()

            # Create individual plots
            self.create_individual_plots(metadata, dim_results, cluster_results)

            # Summary
            png_files = list(self.output_dir.glob("*.png"))
            self.logger.info("="*60)
            self.logger.info(f"✅ Generated {len(png_files)} individual plots in {self.output_dir}")
            for f in sorted(png_files):
                self.logger.info(f"  - {f.name}")
            self.logger.info("="*60)

        except Exception as e:
            self.logger.error(f"Extraction failed: {str(e)}")
            raise

def main():
    extractor = Batch6ExtractIndividualPlots()
    extractor.run_extraction()

if __name__ == "__main__":
    main()