#!/usr/bin/env python3
"""
Extract Batch6 Target Plots as Individual Figures
================================================

Extract specific target figures from batch6 analysis as individual plots:
1. batch6_seeds_vs_synthetic_core_detailed.png -> 4 individual plots
2. batch6_seeds_vs_synthetic_edge_detailed.png -> 4 individual plots
3. batch6_performance_curves_datasets.png -> 4 individual plots
4. batch6_performance_curves_models.png -> 4 individual plots
5. batch6_performance_distributions.png -> individual plot

Based on actual batch6 phase5b and phase7b visualization code.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

class ExtractBatch6TargetPlots:
    """Extract individual plots from batch6 target multi-subplot figures"""

    def __init__(self):
        self.setup_paths()
        self.setup_styling()
        self.setup_logging()

    def setup_paths(self):
        """Setup paths"""
        self.batch6_dir = PROJECT_ROOT / "data" / "batch6"

        # Input directories
        self.phase5a_dir = self.batch6_dir / "phase5a_processed_data"
        self.phase7b_dir = self.batch6_dir / "phase7b_analysis"

        # Output directory
        self.output_dir = self.batch6_dir / "paper"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup_styling(self):
        """Setup batch6 styling"""
        plt.style.use('default')

        # Batch6 color palettes from original code
        self.colors = {
            'real_seeds': '#B22222',
            'synthetic_original': '#4682B4',
            'synthetic_strong': '#CD853F',
            'synthetic_weak': '#9370DB',
            'real_data': '#2C3E50'
        }

        # Performance curve colors
        self.perf_colors = {
            'Real (Baseline)': '#2C3E50',
            'Rewrite Original': '#4682B4',
            'Rewrite Strong': '#CD853F',
            'Rewrite Weak': '#9370DB'
        }

    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def load_phase5a_data(self):
        """Load phase5a processed data"""
        self.logger.info("Loading phase5a processed data...")

        # Load metadata
        metadata = pd.read_csv(self.phase5a_dir / "metadata.csv.gz", compression='gzip')

        # Load dimensionality reduction results
        with open(self.phase5a_dir / "dimensionality_reduction_results.pkl", 'rb') as f:
            dim_results = pickle.load(f)

        return metadata, dim_results

    def load_phase7b_data(self):
        """Load phase7b performance data"""
        self.logger.info("Loading phase7b performance data...")

        # Load performance results
        perf_file = self.phase7b_dir / "csv_export" / "batch6_performance_results_publication.csv"
        if not perf_file.exists():
            # Try alternative location
            perf_file = self.phase7b_dir / "batch6_performance_results.csv"

        performance_data = pd.read_csv(perf_file)
        return performance_data

    def extract_seeds_vs_synthetic_plots(self, metadata, dim_results):
        """Extract individual plots from seeds vs synthetic detailed analysis"""
        self.logger.info("Extracting Seeds vs Synthetic individual plots...")

        for layer in ['core', 'edge']:
            # Filter data for this layer
            layer_mask = metadata['layer'] == layer
            layer_data = metadata[layer_mask]

            if len(layer_data) == 0:
                continue

            # Extract indices for this layer
            layer_indices = layer_data.index

            # Get seeds and synthetic data for this layer
            seeds_mask = layer_data['data_source'] == 'seeds'
            synthetic_mask = layer_data['data_category'] == 'synthetic'

            # Plot 1: PCA Analysis
            plt.figure(figsize=(10, 8))

            # Real seeds
            if seeds_mask.sum() > 0:
                seeds_indices = layer_data[seeds_mask].index
                plt.scatter(dim_results['pca_2d'][seeds_indices, 0],
                           dim_results['pca_2d'][seeds_indices, 1],
                           c=self.colors['real_seeds'],
                           label=f'Real Malicious Seeds ({seeds_mask.sum()})',
                           alpha=0.9, s=50, marker='s', edgecolors='black', linewidth=0.5)

            # Synthetic variants
            if 'prompt_variant' in layer_data.columns:
                for prompt, color in [('original', self.colors['synthetic_original']),
                                    ('strong', self.colors['synthetic_strong']),
                                    ('weak', self.colors['synthetic_weak'])]:
                    prompt_mask = synthetic_mask & (layer_data['prompt_variant'] == prompt)
                    if prompt_mask.sum() > 0:
                        prompt_indices = layer_data[prompt_mask].index
                        plt.scatter(dim_results['pca_2d'][prompt_indices, 0],
                                   dim_results['pca_2d'][prompt_indices, 1],
                                   c=color,
                                   label=f'Synthetic {prompt.title()} ({prompt_mask.sum()})',
                                   alpha=0.7, s=25, marker='o')

            plt.xlabel(f'PC1 ({dim_results["pca_2d_variance"][0]:.1%} variance)')
            plt.ylabel(f'PC2 ({dim_results["pca_2d_variance"][1]:.1%} variance)')
            plt.title(f'{layer.title()} Layer: PCA Analysis')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{layer}_layer_pca_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()

            # Plot 2: t-SNE Analysis
            plt.figure(figsize=(10, 8))

            # Real seeds
            if seeds_mask.sum() > 0:
                seeds_indices = layer_data[seeds_mask].index
                plt.scatter(dim_results['tsne_2d'][seeds_indices, 0],
                           dim_results['tsne_2d'][seeds_indices, 1],
                           c=self.colors['real_seeds'],
                           label=f'Real Malicious Seeds ({seeds_mask.sum()})',
                           alpha=0.9, s=50, marker='s', edgecolors='black', linewidth=0.5)

            # Synthetic variants
            if 'prompt_variant' in layer_data.columns:
                for prompt, color in [('original', self.colors['synthetic_original']),
                                    ('strong', self.colors['synthetic_strong']),
                                    ('weak', self.colors['synthetic_weak'])]:
                    prompt_mask = synthetic_mask & (layer_data['prompt_variant'] == prompt)
                    if prompt_mask.sum() > 0:
                        prompt_indices = layer_data[prompt_mask].index
                        plt.scatter(dim_results['tsne_2d'][prompt_indices, 0],
                                   dim_results['tsne_2d'][prompt_indices, 1],
                                   c=color,
                                   label=f'Synthetic {prompt.title()} ({prompt_mask.sum()})',
                                   alpha=0.7, s=25, marker='o')

            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.title(f'{layer.title()} Layer: t-SNE Analysis')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{layer}_layer_tsne_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()

            # Plot 3: Distance Distribution
            distance_col = f'distance_to_{layer}'
            if distance_col in layer_data.columns:
                plt.figure(figsize=(10, 8))

                distance_data = []
                labels = []
                colors_list = []

                # Real seeds
                if seeds_mask.sum() > 0:
                    distance_data.append(layer_data[seeds_mask][distance_col].values)
                    labels.append(f'Seeds\\n({seeds_mask.sum()})')
                    colors_list.append(self.colors['real_seeds'])

                # Synthetic variants
                if 'prompt_variant' in layer_data.columns:
                    for prompt, color in [('original', self.colors['synthetic_original']),
                                        ('strong', self.colors['synthetic_strong']),
                                        ('weak', self.colors['synthetic_weak'])]:
                        prompt_mask = synthetic_mask & (layer_data['prompt_variant'] == prompt)
                        if prompt_mask.sum() > 0:
                            distance_data.append(layer_data[prompt_mask][distance_col].values)
                            labels.append(f'Synthetic {prompt.title()}\\n({prompt_mask.sum()})')
                            colors_list.append(color)

                if distance_data:
                    bp = plt.boxplot(distance_data, tick_labels=labels, patch_artist=True)

                    for patch, color in zip(bp['boxes'], colors_list):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)

                plt.ylabel(f'Distance to {layer.title()} Centroid')
                plt.title(f'{layer.title()} Layer: Distance Distribution')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.output_dir / f"{layer}_layer_distance_distribution.png", dpi=300, bbox_inches='tight')
                plt.close()

            # Plot 4: Sample Distribution
            plt.figure(figsize=(10, 8))

            sample_counts = []
            sample_labels = []
            sample_colors = []

            # Real seeds
            if seeds_mask.sum() > 0:
                sample_counts.append(seeds_mask.sum())
                sample_labels.append('Real Seeds')
                sample_colors.append(self.colors['real_seeds'])

            # Synthetic variants
            if 'prompt_variant' in layer_data.columns:
                for prompt, color in [('original', self.colors['synthetic_original']),
                                    ('strong', self.colors['synthetic_strong']),
                                    ('weak', self.colors['synthetic_weak'])]:
                    prompt_mask = synthetic_mask & (layer_data['prompt_variant'] == prompt)
                    if prompt_mask.sum() > 0:
                        sample_counts.append(prompt_mask.sum())
                        sample_labels.append(f'Synthetic {prompt.title()}')
                        sample_colors.append(color)

            if sample_counts:
                bars = plt.bar(sample_labels, sample_counts, color=sample_colors, alpha=0.8, edgecolor='black', linewidth=0.5)

                # Add value labels on bars
                for bar, count in zip(bars, sample_counts):
                    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(sample_counts)*0.01,
                            f'{count}', ha='center', va='bottom')

            plt.ylabel('Sample Count')
            plt.title(f'{layer.title()} Layer: Sample Distribution')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{layer}_layer_sample_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"✅ {layer.title()} layer plots extracted")

    def extract_performance_plots(self, performance_data):
        """Extract individual plots from performance analysis"""
        self.logger.info("Extracting performance individual plots...")

        # Prepare data grouping
        dataset_order = ['baseline_real', 'pure_original', 'pure_strong', 'pure_weak']
        model_order = ['RandomForest', 'SVM', 'DeepLearning']
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']

        # Extract dataset performance curves (4 individual plots)
        for metric in metrics:
            plt.figure(figsize=(10, 8))

            for dataset in dataset_order:
                dataset_data = performance_data[performance_data['Dataset_Type'] == dataset]
                if len(dataset_data) > 0:
                    # Group by malicious ratio and calculate mean
                    grouped = dataset_data.groupby('Malicious_Ratio_Percent')[metric].agg(['mean', 'std']).reset_index()

                    color = self.perf_colors.get(dataset.replace('baseline_', '').replace('pure_', 'Rewrite ').replace('_', ' ').title(), '#888888')

                    plt.plot(grouped['Malicious_Ratio_Percent'], grouped['mean'],
                            'o-', color=color, linewidth=2, markersize=6,
                            label=dataset.replace('baseline_', '').replace('pure_', 'Rewrite ').replace('_', ' ').title())

                    # Add confidence intervals
                    plt.fill_between(grouped['Malicious_Ratio_Percent'],
                                   grouped['mean'] - grouped['std'],
                                   grouped['mean'] + grouped['std'],
                                   alpha=0.2, color=color)

            plt.xlabel('Malicious Data Ratio (%)')
            plt.ylabel(metric)
            plt.title(f'{metric} vs Malicious Ratio')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / f"performance_{metric.lower()}_vs_ratio.png", dpi=300, bbox_inches='tight')
            plt.close()

        self.logger.info("✅ Performance curve plots extracted")

        # Extract model comparison plots (4 individual plots)
        for metric in metrics:
            plt.figure(figsize=(10, 8))

            for model in model_order:
                model_data = performance_data[performance_data['Model'] == model]
                if len(model_data) > 0:
                    # Group by malicious ratio and calculate mean
                    grouped = model_data.groupby('Malicious_Ratio_Percent')[metric].agg(['mean', 'std']).reset_index()

                    plt.plot(grouped['Malicious_Ratio_Percent'], grouped['mean'],
                            'o-', linewidth=2, markersize=6, label=model)

                    # Add confidence intervals
                    plt.fill_between(grouped['Malicious_Ratio_Percent'],
                                   grouped['mean'] - grouped['std'],
                                   grouped['mean'] + grouped['std'],
                                   alpha=0.2)

            plt.xlabel('Malicious Data Ratio (%)')
            plt.ylabel(metric)
            plt.title(f'{metric} by Model Type')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / f"model_comparison_{metric.lower()}.png", dpi=300, bbox_inches='tight')
            plt.close()

        self.logger.info("✅ Model comparison plots extracted")

    def run_extraction(self):
        """Run the complete extraction"""
        try:
            self.logger.info("🚀 Starting Batch6 Target Plot Extraction")

            # Load data
            metadata, dim_results = self.load_phase5a_data()
            performance_data = self.load_phase7b_data()

            # Extract Seeds vs Synthetic plots (Core + Edge: 8 plots total)
            self.extract_seeds_vs_synthetic_plots(metadata, dim_results)

            # Extract Performance plots (8 plots total)
            self.extract_performance_plots(performance_data)

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
    extractor = ExtractBatch6TargetPlots()
    extractor.run_extraction()

if __name__ == "__main__":
    main()