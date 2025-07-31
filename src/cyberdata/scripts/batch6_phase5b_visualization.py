#!/usr/bin/env python3
"""
Batch 6 Phase 5b: Academic Visualization Generation
===================================================

Visualization stage - creates publication-quality static and interactive visualizations.
Loads processed data from Phase 5a and generates academic-standard figures.

Key features:
- Academic color schemes and styling (suitable for ACM/IEEE publications)
- Clear, professional legend naming
- Separate Core/Edge analysis for Seeds vs Synthetic comparison
- Static and interactive visualization options
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.offline as pyo
from sklearn.cluster import KMeans
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
        log_dir / f"batch6_phase5b_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

class Batch6Phase5bVisualizer:
    """Academic visualization generator for batch6 analysis"""
    
    def __init__(self):
        self.logger = setup_logger("batch6_phase5b")
        self.setup_paths()
        self.setup_academic_styling()
        
    def setup_paths(self) -> None:
        """Setup directory paths"""
        self.batch6_dir = PROJECT_ROOT / "data" / "batch6"
        self.input_dir = self.batch6_dir / "phase5a_processed_data"
        
        # Output directories
        self.output_dir = self.batch6_dir / "phase5b_analysis"
        self.static_plots_dir = self.output_dir / "static_plots"
        self.interactive_plots_dir = self.output_dir / "interactive_plots"
        self.cluster_analysis_dir = self.output_dir / "cluster_analysis"
        self.distribution_analysis_dir = self.output_dir / "distribution_analysis"
        
        # Create directories
        for dir_path in [self.output_dir, self.static_plots_dir, self.interactive_plots_dir, 
                        self.cluster_analysis_dir, self.distribution_analysis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        
    def setup_academic_styling(self) -> None:
        """Setup academic publication styling"""
        # Academic color schemes (colorblind-friendly, print-friendly)
        self.academic_colors = {
            # Data source colors (muted, professional)
            'data_sources': {
                'dataset': '#2E8B57',          # Sea Green (background/dataset)
                'real_malicious_seeds': '#B22222',  # Fire Brick (seeds)
                'synthetic': '#4682B4',        # Steel Blue (synthetic)
                'real_benign': '#708090',      # Slate Gray (benign)
                'testing_data': '#8B4513'     # Saddle Brown (test)
            },
            
            # Synthetic variants (distinct but harmonious)
            'synthetic_variants': {
                'original': '#4682B4',         # Steel Blue
                'strong': '#CD853F',          # Peru
                'weak': '#9370DB'             # Medium Purple
            },
            
            # Layer colors (consistent with above)
            'layers': {
                'core': '#B22222',            # Fire Brick
                'edge': '#4682B4',           # Steel Blue
                'background': '#2E8B57',     # Sea Green
                'benign': '#708090',         # Slate Gray
                'test': '#8B4513'           # Saddle Brown
            },
            
            # Performance curve colors (for Phase 7b)
            'performance_curves': {
                'real_baseline': '#2C3E50',   # Dark Gray-Blue
                'rewrite_original': '#3498DB', # Blue
                'rewrite_strong': '#E67E22',   # Orange
                'rewrite_weak': '#9B59B6'      # Purple
            }
        }
        
        # Academic legend names (clear, professional)
        self.academic_legends = {
            'data_sources': {
                'background': 'Dataset',
                'seeds': 'Real Malicious Samples (Seeds)',
                'batch6_synthetic': 'Synthetic (Original+Strong+Weak)',
                'real_benign': 'Real Benign',
                'test_set': 'Testing Data',
                'real_malicious': 'Real Malicious (Seeds)',
                'synthetic': 'Synthetic',
                'test_benign': 'Test Benign',  # Will be hidden
                'test_malicious': 'Test Malicious'
            },
            
            'performance_curves': {
                'baseline_real': 'Real (Baseline)',
                'pure_original': 'Rewrite Original',
                'pure_strong': 'Rewrite Strong', 
                'pure_weak': 'Rewrite Weak'
            },
            
            'synthetic_detailed': {
                'original_core': 'Original Core',
                'original_edge': 'Original Edge',
                'strong_core': 'Strong Core',
                'strong_edge': 'Strong Edge',
                'weak_core': 'Weak Core',
                'weak_edge': 'Weak Edge'
            }
        }
        
        # Set matplotlib academic style
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'axes.linewidth': 1.2,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.8,
            'legend.frameon': True,
            'legend.fancybox': False,
            'legend.shadow': False,
            'legend.framealpha': 1.0,
            'legend.edgecolor': 'black',
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
        
        self.logger.info("Academic styling configured")
        
    def load_processed_data(self) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, np.ndarray], Dict[str, Any]]:
        """Load processed data from Phase 5a"""
        try:
            self.logger.info("Loading processed data from Phase 5a...")
            
            # Load embeddings
            embeddings = np.load(self.input_dir / "embeddings.npy")
            
            # Load metadata
            metadata = pd.read_csv(self.input_dir / "metadata.csv.gz", compression='gzip')
            
            # Load dimensionality reduction results
            with open(self.input_dir / "dimensionality_reduction_results.pkl", 'rb') as f:
                dim_results = pickle.load(f)
            
            # Load clustering results
            with open(self.input_dir / "clustering_results.json", 'r') as f:
                cluster_results_raw = json.load(f)
            
            # Convert clustering results back to proper format
            cluster_results = {
                'optimal_k': cluster_results_raw['optimal_k'],
                'best_silhouette': cluster_results_raw['best_silhouette'],
                'kmeans': {}
            }
            
            for k, k_results in cluster_results_raw['kmeans'].items():
                cluster_results['kmeans'][int(k)] = {
                    'labels': np.array(k_results['labels']),
                    'centroids': np.array(k_results['centroids']),
                    'inertia': k_results['inertia'],
                    'silhouette_score': k_results['silhouette_score']
                }
            
            self.logger.info(f"✅ Loaded processed data:")
            self.logger.info(f"   Embeddings: {embeddings.shape}")
            self.logger.info(f"   Metadata: {len(metadata)} records")
            self.logger.info(f"   Dimensionality reduction: {list(dim_results.keys())}")
            self.logger.info(f"   Clustering K range: {min(cluster_results['kmeans'].keys())}-{max(cluster_results['kmeans'].keys())}")
            
            return embeddings, metadata, dim_results, cluster_results
            
        except Exception as e:
            self.logger.error(f"Failed to load processed data: {str(e)}")
            raise
            
    def standardize_data_labels(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Standardize data source labels for academic presentation"""
        try:
            self.logger.info("Standardizing data labels for academic presentation...")
            
            # Create standardized labels
            metadata = metadata.copy()
            
            # Standardize data_source column
            source_mapping = {
                'background': 'dataset',
                'seeds': 'real_malicious_seeds',
                'batch6_synthetic': 'synthetic',
                'real_benign': 'real_benign',
                'test_set': 'testing_data',
                'real_malicious': 'real_malicious_seeds',
                'test_benign': 'test_benign',
                'test_malicious': 'test_malicious'
            }
            
            metadata['data_source'] = metadata['data_source'].replace(source_mapping)
            
            # Add academic display names
            metadata['data_source_display'] = metadata['data_source'].map(self.academic_legends['data_sources'])
            
            # Fill any missing display names
            metadata['data_source_display'] = metadata['data_source_display'].fillna(metadata['data_source'])
            
            self.logger.info("✅ Data labels standardized")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to standardize data labels: {str(e)}")
            raise
            
    def create_comprehensive_overview(self, metadata: pd.DataFrame, dim_results: Dict[str, np.ndarray]) -> None:
        """Create comprehensive data overview with academic styling"""
        try:
            self.logger.info("Creating comprehensive data overview...")
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Batch 6 Data Distribution Analysis', fontsize=16, fontweight='bold')
            
            # Filter out less important categories for cleaner visualization
            important_sources = ['dataset', 'real_malicious_seeds', 'synthetic', 'testing_data']
            filtered_metadata = metadata[metadata['data_source'].isin(important_sources)]
            
            # 1. PCA 2D by data source
            ax1 = axes[0, 0]
            for source in important_sources:
                mask = filtered_metadata['data_source'] == source
                if mask.sum() > 0:
                    indices = filtered_metadata[mask].index
                    color = self.academic_colors['data_sources'][source]
                    display_name = self.academic_legends['data_sources'][source]
                    
                    ax1.scatter(dim_results['pca_2d'][indices, 0], dim_results['pca_2d'][indices, 1], 
                               c=color, label=f'{display_name} ({mask.sum()})', 
                               alpha=0.7, s=20, edgecolors='white', linewidth=0.5)
            
            ax1.set_xlabel(f'PC1 ({dim_results["pca_2d_variance"][0]:.1%} variance)')
            ax1.set_ylabel(f'PC2 ({dim_results["pca_2d_variance"][1]:.1%} variance)')
            ax1.set_title('PCA 2D by Data Source')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # 2. t-SNE 2D by data category
            ax2 = axes[0, 1]
            
            # Create cleaner categories for t-SNE
            display_categories = {}
            for source in important_sources:
                mask = filtered_metadata['data_source'] == source
                if mask.sum() > 0:
                    indices = filtered_metadata[mask].index
                    color = self.academic_colors['data_sources'][source]
                    display_name = self.academic_legends['data_sources'][source]
                    
                    # Skip less important categories
                    if source in ['test_benign']:
                        continue
                        
                    ax2.scatter(dim_results['tsne_2d'][indices, 0], dim_results['tsne_2d'][indices, 1], 
                               c=color, label=display_name, alpha=0.7, s=20, 
                               edgecolors='white', linewidth=0.5)
            
            ax2.set_xlabel('t-SNE Dimension 1')
            ax2.set_ylabel('t-SNE Dimension 2')
            ax2.set_title('t-SNE 2D by Data Category')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            # 3. Layer distribution (if available)
            ax3 = axes[1, 0]
            if 'layer' in metadata.columns:
                layer_counts = metadata['layer'].value_counts()
                colors = [self.academic_colors['layers'].get(layer, '#808080') for layer in layer_counts.index]
                
                bars = ax3.bar(layer_counts.index, layer_counts.values, color=colors, alpha=0.8, edgecolor='black')
                ax3.set_xlabel('Layer')
                ax3.set_ylabel('Sample Count')
                ax3.set_title('Sample Distribution by Layer')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 50,
                            f'{int(height)}', ha='center', va='bottom')
            else:
                ax3.text(0.5, 0.5, 'Layer information not available', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Layer Distribution')
            
            # 4. Data source distribution
            ax4 = axes[1, 1]
            source_counts = filtered_metadata['data_source'].value_counts()
            display_names = [self.academic_legends['data_sources'][source] for source in source_counts.index]
            colors = [self.academic_colors['data_sources'][source] for source in source_counts.index]
            
            bars = ax4.bar(range(len(source_counts)), source_counts.values, color=colors, alpha=0.8, edgecolor='black')
            ax4.set_xticks(range(len(source_counts)))
            ax4.set_xticklabels(display_names, rotation=45, ha='right')
            ax4.set_ylabel('Sample Count')
            ax4.set_title('Sample Distribution by Data Source')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 50,
                        f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.static_plots_dir / "batch6_comprehensive_overview.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("✅ Comprehensive overview created")
            
        except Exception as e:
            self.logger.error(f"Failed to create comprehensive overview: {str(e)}")
            
    def create_seeds_vs_synthetic_detailed_analysis(self, metadata: pd.DataFrame, 
                                                   dim_results: Dict[str, np.ndarray]) -> None:
        """Create detailed Core/Edge analysis for Seeds vs Synthetic comparison"""
        try:
            self.logger.info("Creating detailed Seeds vs Synthetic analysis (Core/Edge separated)...")
            
            # Create separate plots for Core and Edge
            for layer in ['core', 'edge']:
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle(f'Seeds vs Synthetic Analysis - {layer.title()} Layer', fontsize=16, fontweight='bold')
                
                # Filter data for this layer
                layer_mask = metadata['layer'] == layer
                layer_metadata = metadata[layer_mask]
                layer_indices = layer_metadata.index
                
                if len(layer_indices) == 0:
                    self.logger.warning(f"No data found for {layer} layer")
                    plt.close()
                    continue
                
                # Get seeds and synthetic data for this layer
                seeds_mask = layer_metadata['data_source'] == 'real_malicious_seeds'
                synthetic_mask = layer_metadata['data_category'] == 'synthetic'
                
                # 1. PCA: Seeds vs Synthetic by Prompt Strategy (for this layer)
                ax1 = axes[0, 0]
                
                # Plot seeds first (so they appear on top)
                if seeds_mask.sum() > 0:
                    seeds_indices = layer_metadata[seeds_mask].index
                    ax1.scatter(dim_results['pca_2d'][seeds_indices, 0], dim_results['pca_2d'][seeds_indices, 1], 
                               c=self.academic_colors['data_sources']['real_malicious_seeds'], 
                               label=f'Real Malicious Seeds ({seeds_mask.sum()})', 
                               alpha=0.9, s=60, marker='s', edgecolors='black', linewidth=0.8)
                
                # Plot synthetic variants
                if 'prompt_variant' in layer_metadata.columns:
                    for prompt in ['original', 'strong', 'weak']:
                        prompt_mask = synthetic_mask & (layer_metadata['prompt_variant'] == prompt)
                        if prompt_mask.sum() > 0:
                            prompt_indices = layer_metadata[prompt_mask].index
                            color = self.academic_colors['synthetic_variants'][prompt]
                            ax1.scatter(dim_results['pca_2d'][prompt_indices, 0], dim_results['pca_2d'][prompt_indices, 1], 
                                       c=color, label=f'Synthetic {prompt.title()} ({prompt_mask.sum()})', 
                                       alpha=0.7, s=30, marker='o')
                
                ax1.set_xlabel(f'PC1 ({dim_results["pca_2d_variance"][0]:.1%} variance)')
                ax1.set_ylabel(f'PC2 ({dim_results["pca_2d_variance"][1]:.1%} variance)')
                ax1.set_title(f'{layer.title()} Layer: PCA Analysis')
                ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax1.grid(True, alpha=0.3)
                
                # 2. t-SNE: Seeds vs Synthetic (for this layer)
                ax2 = axes[0, 1]
                
                # Plot seeds
                if seeds_mask.sum() > 0:
                    seeds_indices = layer_metadata[seeds_mask].index
                    ax2.scatter(dim_results['tsne_2d'][seeds_indices, 0], dim_results['tsne_2d'][seeds_indices, 1], 
                               c=self.academic_colors['data_sources']['real_malicious_seeds'], 
                               label=f'Real Malicious Seeds ({seeds_mask.sum()})', 
                               alpha=0.9, s=60, marker='s', edgecolors='black', linewidth=0.8)
                
                # Plot synthetic variants
                if 'prompt_variant' in layer_metadata.columns:
                    for prompt in ['original', 'strong', 'weak']:
                        prompt_mask = synthetic_mask & (layer_metadata['prompt_variant'] == prompt)
                        if prompt_mask.sum() > 0:
                            prompt_indices = layer_metadata[prompt_mask].index
                            color = self.academic_colors['synthetic_variants'][prompt]
                            ax2.scatter(dim_results['tsne_2d'][prompt_indices, 0], dim_results['tsne_2d'][prompt_indices, 1], 
                                       c=color, label=f'Synthetic {prompt.title()} ({prompt_mask.sum()})', 
                                       alpha=0.7, s=30, marker='o')
                
                ax2.set_xlabel('t-SNE Dimension 1')
                ax2.set_ylabel('t-SNE Dimension 2')
                ax2.set_title(f'{layer.title()} Layer: t-SNE Analysis')
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax2.grid(True, alpha=0.3)
                
                # 3. Distance Distribution (if available)
                ax3 = axes[1, 0]
                distance_col = f'distance_to_{layer}'
                if distance_col in layer_metadata.columns:
                    # Box plot of distances
                    distance_data = []
                    labels = []
                    
                    if seeds_mask.sum() > 0:
                        distance_data.append(layer_metadata[seeds_mask][distance_col].values)
                        labels.append(f'Seeds\n({seeds_mask.sum()})')
                    
                    if 'prompt_variant' in layer_metadata.columns:
                        for prompt in ['original', 'strong', 'weak']:
                            prompt_mask = synthetic_mask & (layer_metadata['prompt_variant'] == prompt)
                            if prompt_mask.sum() > 0:
                                distance_data.append(layer_metadata[prompt_mask][distance_col].values)
                                labels.append(f'Synthetic {prompt.title()}\n({prompt_mask.sum()})')
                    
                    if distance_data:
                        bp = ax3.boxplot(distance_data, tick_labels=labels, patch_artist=True)
                        
                        # Rotate x-axis labels to prevent overlap
                        ax3.tick_params(axis='x', rotation=45)
                        
                        # Color the boxes
                        colors = [self.academic_colors['data_sources']['real_malicious_seeds']]
                        colors.extend([self.academic_colors['synthetic_variants'][p] for p in ['original', 'strong', 'weak'] 
                                     if any(synthetic_mask & (layer_metadata['prompt_variant'] == p))])
                        
                        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                            patch.set_facecolor(color)
                            patch.set_alpha(0.7)
                
                ax3.set_ylabel(f'Distance to {layer.title()} Centroid')
                ax3.set_title(f'{layer.title()} Layer: Distance Distribution')
                ax3.grid(True, alpha=0.3)
                
                # 4. Sample Statistics
                ax4 = axes[1, 1]
                
                stats_data = []
                if seeds_mask.sum() > 0:
                    stats_data.append(('Real Seeds', seeds_mask.sum()))
                
                if 'prompt_variant' in layer_metadata.columns:
                    for prompt in ['original', 'strong', 'weak']:
                        prompt_mask = synthetic_mask & (layer_metadata['prompt_variant'] == prompt)
                        if prompt_mask.sum() > 0:
                            stats_data.append((f'Synthetic {prompt.title()}', prompt_mask.sum()))
                
                if stats_data:
                    categories, counts = zip(*stats_data)
                    colors = [self.academic_colors['data_sources']['real_malicious_seeds']]
                    colors.extend([self.academic_colors['synthetic_variants'][p] for p in ['original', 'strong', 'weak']
                                 if f'Synthetic {p.title()}' in categories])
                    
                    bars = ax4.bar(categories, counts, color=colors[:len(categories)], alpha=0.8, edgecolor='black')
                    ax4.set_ylabel('Sample Count')
                    ax4.set_title(f'{layer.title()} Layer: Sample Distribution')
                    ax4.tick_params(axis='x', rotation=45)
                    
                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        ax4.text(bar.get_x() + bar.get_width()/2., height + 5,
                                f'{int(height)}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(self.static_plots_dir / f"batch6_seeds_vs_synthetic_{layer}_detailed.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                self.logger.info(f"✅ {layer.title()} layer detailed analysis created")
            
        except Exception as e:
            self.logger.error(f"Failed to create detailed Seeds vs Synthetic analysis: {str(e)}")
            
    def create_clustering_quality_analysis(self, cluster_results: Dict[str, Any]) -> None:
        """Create K-means clustering quality analysis with academic styling"""
        try:
            self.logger.info("Creating clustering quality analysis...")
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle('K-means Clustering Quality Analysis', fontsize=16, fontweight='bold')
            
            k_values = sorted(list(cluster_results['kmeans'].keys()))
            silhouette_scores = [cluster_results['kmeans'][k]['silhouette_score'] for k in k_values]
            inertia_values = [cluster_results['kmeans'][k]['inertia'] for k in k_values]
            
            # 1. Silhouette score analysis
            ax1 = axes[0]
            ax1.plot(k_values, silhouette_scores, 'o-', linewidth=3, markersize=8, 
                    color=self.academic_colors['data_sources']['real_malicious_seeds'], markerfacecolor='white', 
                    markeredgewidth=2, markeredgecolor=self.academic_colors['data_sources']['real_malicious_seeds'])
            ax1.set_xlabel('Number of Clusters (K)')
            ax1.set_ylabel('Silhouette Score')
            ax1.set_title('Clustering Quality (Silhouette Method)')
            ax1.grid(True, alpha=0.3)
            
            # Highlight optimal K
            if cluster_results['optimal_k'] is not None:
                optimal_k = cluster_results['optimal_k']
                optimal_score = cluster_results['kmeans'][optimal_k]['silhouette_score']
                ax1.plot(optimal_k, optimal_score, 'o', markersize=12, 
                        color=self.academic_colors['synthetic_variants']['strong'], 
                        markeredgecolor='black', markeredgewidth=2,
                        label=f'Optimal K={optimal_k} (Score={optimal_score:.3f})')
                ax1.legend()
            
            # 2. Inertia analysis (Elbow method)
            ax2 = axes[1]
            ax2.plot(k_values, inertia_values, 's-', linewidth=3, markersize=8, 
                    color=self.academic_colors['data_sources']['synthetic'], markerfacecolor='white', 
                    markeredgewidth=2, markeredgecolor=self.academic_colors['data_sources']['synthetic'])
            ax2.set_xlabel('Number of Clusters (K)')
            ax2.set_ylabel('Within-Cluster Sum of Squares (Inertia)')
            ax2.set_title('Clustering Quality (Elbow Method)')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.cluster_analysis_dir / "batch6_clustering_quality.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("✅ Clustering quality analysis created")
            
        except Exception as e:
            self.logger.error(f"Failed to create clustering quality analysis: {str(e)}")
            
    def create_interactive_seeds_vs_synthetic_6combinations(self, metadata: pd.DataFrame, 
                                                           dim_results: Dict[str, np.ndarray]) -> None:
        """Create interactive visualization for Real Seeds vs 6 Synthetic combinations"""
        try:
            self.logger.info("Creating interactive Seeds vs 6 Synthetic combinations...")
            
            # Define colors for 6 combinations
            combination_colors = {
                'original-core': self.academic_colors['synthetic_variants']['original'],
                'original-edge': '#87CEEB',  # Light version of original
                'strong-core': self.academic_colors['synthetic_variants']['strong'],
                'strong-edge': '#DEB887',   # Light version of strong  
                'weak-core': self.academic_colors['synthetic_variants']['weak'],
                'weak-edge': '#DDA0DD'      # Light version of weak
            }
            
            # Create 2D PCA interactive plot
            fig = go.Figure()
            
            # Add seeds data (all combined)
            seeds_mask = metadata['data_source'] == 'real_malicious_seeds'
            if seeds_mask.sum() > 0:
                seeds_data = metadata[seeds_mask]
                fig.add_trace(go.Scatter(
                    x=dim_results['pca_2d'][seeds_mask, 0],
                    y=dim_results['pca_2d'][seeds_mask, 1],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=self.academic_colors['data_sources']['real_malicious_seeds'],
                        opacity=0.9,
                        symbol='square',
                        line=dict(width=2, color='black')
                    ),
                    text=[f'ID: {uid}<br>Source: Real Malicious Seeds<br>Layer: {layer}' 
                          for uid, layer in zip(seeds_data['unique_id'], seeds_data['layer'])],
                    name=f'Real Malicious Seeds ({seeds_mask.sum()})',
                    hovertemplate='<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>'
                ))
            
            # Add 6 synthetic combinations
            synthetic_mask = metadata['data_category'] == 'synthetic'
            if 'prompt_variant' in metadata.columns:
                for layer in ['core', 'edge']:
                    for prompt in ['original', 'strong', 'weak']:
                        combination_key = f'{prompt}-{layer}'
                        layer_prompt_mask = (synthetic_mask & 
                                           (metadata['layer'] == layer) & 
                                           (metadata['prompt_variant'] == prompt))
                        
                        if layer_prompt_mask.sum() > 0:
                            color = combination_colors.get(combination_key, '#888888')
                            prompt_data = metadata[layer_prompt_mask]
                            
                            display_name = f'{self.academic_legends["synthetic_detailed"][f"{prompt}_{layer}"]}'
                            
                            fig.add_trace(go.Scatter(
                                x=dim_results['pca_2d'][layer_prompt_mask, 0],
                                y=dim_results['pca_2d'][layer_prompt_mask, 1],
                                mode='markers',
                                marker=dict(
                                    size=8,
                                    color=color,
                                    opacity=0.7
                                ),
                                text=[f'ID: {uid}<br>Type: {display_name}<br>Seed: {seed}' 
                                      for uid, seed in zip(prompt_data['unique_id'], 
                                                         prompt_data.get('original_seed_id', 'N/A'))],
                                name=f'{display_name} ({layer_prompt_mask.sum()})',
                                hovertemplate='<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>'
                            ))
            
            fig.update_layout(
                title='Interactive 2D PCA: Real Seeds vs 6 Synthetic Combinations',
                xaxis_title=f'PC1 ({dim_results["pca_2d_variance"][0]:.1%} variance)',
                yaxis_title=f'PC2 ({dim_results["pca_2d_variance"][1]:.1%} variance)',
                height=700,
                font=dict(family="Times New Roman, serif", size=12),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            pyo.plot(fig, filename=str(self.interactive_plots_dir / "batch6_2d_pca_seeds_vs_6synthetic.html"), 
                    auto_open=False)
            
            self.logger.info("✅ Interactive Seeds vs 6 Synthetic combinations created")
            
        except Exception as e:
            self.logger.error(f"Failed to create interactive visualization: {str(e)}")
            
    def run_phase5b(self) -> None:
        """Execute complete Phase 5b visualization generation"""
        try:
            self.logger.info("🚀 Starting Batch 6 Phase 5b: Academic Visualization Generation")
            start_time = datetime.now()
            
            # Step 1: Load processed data from Phase 5a
            embeddings, metadata, dim_results, cluster_results = self.load_processed_data()
            
            # Step 2: Standardize data labels for academic presentation
            metadata = self.standardize_data_labels(metadata)
            
            # Step 3: Create comprehensive overview
            self.create_comprehensive_overview(metadata, dim_results)
            
            # Step 4: Create detailed Core/Edge Seeds vs Synthetic analysis
            self.create_seeds_vs_synthetic_detailed_analysis(metadata, dim_results)
            
            # Step 5: Create clustering quality analysis
            self.create_clustering_quality_analysis(cluster_results)
            
            # Step 6: Create interactive visualizations
            self.create_interactive_seeds_vs_synthetic_6combinations(metadata, dim_results)
            
            # Phase completion
            end_time = datetime.now()
            runtime = (end_time - start_time).total_seconds() / 60
            
            # Save completion summary
            completion_summary = {
                'phase5b_completion': {
                    'status': 'completed',
                    'completion_date': datetime.now().isoformat(),
                    'runtime_minutes': runtime,
                    'visualizations_created': {
                        'static_plots': len(list(self.static_plots_dir.glob("*.png"))),
                        'interactive_plots': len(list(self.interactive_plots_dir.glob("*.html"))),
                        'cluster_analysis': len(list(self.cluster_analysis_dir.glob("*.png"))),
                        'distribution_analysis': len(list(self.distribution_analysis_dir.glob("*.png")))
                    }
                },
                'academic_styling': {
                    'color_scheme': 'academic_professional',
                    'legend_standardization': 'completed',
                    'core_edge_separation': 'completed'
                },
                'output_directories': {
                    'static_plots': str(self.static_plots_dir),
                    'interactive_plots': str(self.interactive_plots_dir),
                    'cluster_analysis': str(self.cluster_analysis_dir),
                    'distribution_analysis': str(self.distribution_analysis_dir)
                }
            }
            
            with open(self.output_dir / "phase5b_completion_summary.json", 'w') as f:
                json.dump(completion_summary, f, indent=2)
            
            self.logger.info(f"✅ Batch 6 Phase 5b completed successfully!")
            self.logger.info(f"Runtime: {runtime:.2f} minutes")
            self.logger.info(f"Academic visualizations created: {completion_summary['phase5b_completion']['visualizations_created']}")
            self.logger.info(f"Output directory: {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"❌ Batch 6 Phase 5b failed: {str(e)}")
            raise

def main():
    """Main function to run Batch 6 Phase 5b"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Batch 6 Phase 5b: Academic Visualization Generation')
    parser.add_argument('--clean', action='store_true', help='Clean previous visualization results before running')
    args = parser.parse_args()
    
    logger = setup_logger("batch6_phase5b")
    
    try:
        # Handle clean option
        batch6_dir = PROJECT_ROOT / "data" / "batch6"
        if args.clean:
            logger.info("🧹 Cleaning previous Phase 5b results...")
            
            dirs_to_clean = [
                batch6_dir / "phase5b_analysis"
            ]
            
            for dir_path in dirs_to_clean:
                if dir_path.exists():
                    import shutil
                    shutil.rmtree(dir_path)
                    logger.info(f"   Removed: {dir_path}")
                    
            logger.info("✅ Cleanup completed!")
        
        # Run Phase 5b
        visualizer = Batch6Phase5bVisualizer()
        visualizer.run_phase5b()
        
    except Exception as e:
        logger.error(f"Phase 5b execution failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()