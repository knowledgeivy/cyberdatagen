#!/usr/bin/env python3
"""
Batch 6 Phase 7b: Performance Visualization and Analysis
========================================================

Pure visualization stage - reads ML results and creates comprehensive visualizations.
Separate from ML training to allow independent execution and flexible analysis.

Key features:
- Read structured ML results from Phase 7a
- Create static and interactive performance curves
- Export publication-ready CSV tables
- Generate comprehensive analysis reports
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.offline as pyo
import warnings
warnings.filterwarnings('ignore')

class Batch6Phase7bVisualizer:
    """Visualization stage for batch6 ML training results analysis"""
    
    def __init__(self):
        self.setup_paths()
        self.setup_logging()
        self.load_configurations()
        
    def setup_paths(self) -> None:
        """Setup directory structure for phase 7b"""
        # Use absolute paths to avoid working directory issues
        current_dir = Path(__file__).parent.parent.parent.parent  # Go up to project root
        self.base_dir = current_dir / "data" / "batch6"
        
        # Input directory - ML training results
        self.input_dir = self.base_dir / "phase7a_training"
        
        # Output directory - visualization results
        self.output_dir = self.base_dir / "phase7b_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for subdir in ['plots', 'interactive', 'csv_export', 'reports']:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)
            
    def setup_logging(self) -> None:
        """Setup logging configuration"""
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"batch6_phase7b_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_configurations(self) -> None:
        """Load visualization configurations"""
        self.config = {
            'color_schemes': {
                'performance_curves': {
                    'baseline_real': '#2C3E50',      # Dark Gray-Blue (professional)
                    'pure_original': '#3498DB',       # Blue (academic standard)
                    'pure_strong': '#E67E22',         # Orange (distinct)
                    'pure_weak': '#9B59B6'            # Purple (colorblind-friendly)
                },
                'dataset_types': {
                    'baseline_real': '#2C3E50',
                    'pure_original': '#3498DB', 
                    'pure_strong': '#E67E22',
                    'pure_weak': '#9B59B6'
                },
                'models': {
                    'RandomForest': '#34495E',        # Dark Slate
                    'SVM': '#E67E22',                 # Orange
                    'DeepLearning': '#27AE60'         # Green
                }
            },
            'academic_legends': {
                'performance_curves': {
                    'baseline_real': 'Real (Baseline)',
                    'pure_original': 'Rewrite Original',
                    'pure_strong': 'Rewrite Strong',
                    'pure_weak': 'Rewrite Weak'
                }
            },
            'reference_lines': {
                'batch6_target': 0.75,
                'batch5_baseline': 0.60
            },
            'metrics': ['accuracy', 'precision', 'recall', 'f1'],
            'metric_titles': ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        }
        
        self.logger.info("Phase 7b visualization configuration loaded")
        
    def load_training_results(self) -> Dict[str, Any]:
        """Load structured training results from Phase 7a"""
        try:
            self.logger.info("Loading ML training results from Phase 7a...")
            
            # Load structured results
            results_file = self.input_dir / "results" / "structured_results.json"
            if not results_file.exists():
                raise FileNotFoundError(f"Structured results not found: {results_file}")
            
            with open(results_file, 'r') as f:
                structured_results = json.load(f)
            
            # Load detailed results
            detailed_file = self.input_dir / "results" / "detailed_results.csv"
            if not detailed_file.exists():
                raise FileNotFoundError(f"Detailed results not found: {detailed_file}")
            
            detailed_results = pd.read_csv(detailed_file)
            
            self.logger.info(f"✅ Loaded training results:")
            self.logger.info(f"   Total experiments: {len(detailed_results)}")
            self.logger.info(f"   Average F1: {detailed_results['f1'].mean():.4f}")
            self.logger.info(f"   Best F1: {detailed_results['f1'].max():.4f}")
            
            return {
                'structured': structured_results,
                'detailed': detailed_results
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load training results: {str(e)}")
            raise
            
    def prepare_curve_data(self, detailed_results: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for curve visualizations"""
        try:
            self.logger.info("Preparing curve visualization data...")
            
            # Parse malicious ratio to numeric
            detailed_results['malicious_ratio_numeric'] = detailed_results['malicious_ratio'].str.replace('pct', '').astype(int)
            
            # Add derived columns for analysis
            curve_data = detailed_results.copy()
            
            # Add layer information if available (placeholder for future enhancement)
            curve_data['layer'] = 'combined'  # For now, treat as combined
            
            self.logger.info(f"✅ Curve data prepared: {len(curve_data)} data points")
            return curve_data
            
        except Exception as e:
            self.logger.error(f"Failed to prepare curve data: {str(e)}")
            raise
            
    def export_publication_csv(self, curve_data: pd.DataFrame) -> None:
        """Export results to publication-ready CSV format"""
        try:
            self.logger.info("Exporting publication-ready CSV tables...")
            
            # 1. Main results table
            pub_data = []
            for _, row in curve_data.iterrows():
                pub_data.append({
                    'Dataset_Type': row['dataset_type'],
                    'Model': row['model_name'],
                    'Malicious_Ratio_Percent': row['malicious_ratio_numeric'],
                    'Accuracy': round(row['accuracy'], 4),
                    'Precision': round(row['precision'], 4),
                    'Recall': round(row['recall'], 4),
                    'F1_Score': round(row['f1'], 4),
                    'Training_Samples': int(row['training_samples']),
                    'Test_Samples': int(row['test_samples'])
                })
            
            pub_df = pd.DataFrame(pub_data)
            pub_df = pub_df.sort_values(['Dataset_Type', 'Model', 'Malicious_Ratio_Percent'])
            
            # Export main results
            main_csv = self.output_dir / "csv_export" / "batch6_performance_results_publication.csv"
            pub_df.to_csv(main_csv, index=False)
            
            # 2. Summary statistics table
            summary_data = []
            for dataset_type in pub_df['Dataset_Type'].unique():
                for metric in ['Accuracy', 'Precision', 'Recall', 'F1_Score']:
                    subset = pub_df[pub_df['Dataset_Type'] == dataset_type]
                    best_idx = subset[metric].idxmax()
                    
                    summary_data.append({
                        'Dataset_Type': dataset_type,
                        'Metric': metric,
                        'Mean': round(subset[metric].mean(), 4),
                        'Std': round(subset[metric].std(), 4),
                        'Min': round(subset[metric].min(), 4),
                        'Max': round(subset[metric].max(), 4),
                        'Best_Model': subset.loc[best_idx, 'Model'],
                        'Best_Ratio': subset.loc[best_idx, 'Malicious_Ratio_Percent']
                    })
            
            summary_df = pd.DataFrame(summary_data)
            summary_csv = self.output_dir / "csv_export" / "batch6_performance_summary_publication.csv"
            summary_df.to_csv(summary_csv, index=False)
            
            # 3. Model comparison table
            model_comp_data = []
            for model in pub_df['Model'].unique():
                for metric in ['Accuracy', 'Precision', 'Recall', 'F1_Score']:
                    subset = pub_df[pub_df['Model'] == model]
                    model_comp_data.append({
                        'Model': model,
                        'Metric': metric,
                        'Mean': round(subset[metric].mean(), 4),
                        'Std': round(subset[metric].std(), 4),
                        'Best_Dataset': subset.loc[subset[metric].idxmax(), 'Dataset_Type'],
                        'Best_Ratio': subset.loc[subset[metric].idxmax(), 'Malicious_Ratio_Percent']
                    })
            
            model_comp_df = pd.DataFrame(model_comp_data)
            model_csv = self.output_dir / "csv_export" / "batch6_model_comparison_publication.csv"
            model_comp_df.to_csv(model_csv, index=False)
            
            self.logger.info(f"✅ CSV export completed:")
            self.logger.info(f"   Main results: {main_csv}")
            self.logger.info(f"   Summary stats: {summary_csv}")
            self.logger.info(f"   Model comparison: {model_csv}")
            
        except Exception as e:
            self.logger.error(f"Failed to export CSV: {str(e)}")
            raise
            
    def create_static_performance_curves(self, curve_data: pd.DataFrame) -> None:
        """Create static performance curve visualizations"""
        try:
            self.logger.info("Creating static performance curves...")
            
            # 1. Dataset type comparison curves
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Batch 6 Performance Curves: Dataset Type Comparison', fontsize=16, fontweight='bold')
            
            colors = self.config['color_schemes']['performance_curves']
            legends = self.config['academic_legends']['performance_curves']
            
            for i, (metric, title) in enumerate(zip(self.config['metrics'], self.config['metric_titles'])):
                ax = axes[i//2, i%2]
                
                for dataset_type in ['baseline_real', 'pure_original', 'pure_strong', 'pure_weak']:
                    subset = curve_data[curve_data['dataset_type'] == dataset_type]
                    if len(subset) > 0:
                        # Average across models for cleaner visualization
                        curve = subset.groupby('malicious_ratio_numeric')[metric].agg(['mean', 'std']).reset_index()
                        
                        ax.plot(curve['malicious_ratio_numeric'], curve['mean'], 
                               color=colors[dataset_type], linewidth=3, marker='o', markersize=8,
                               label=legends[dataset_type])
                        
                        # Add error bars (std)
                        ax.fill_between(curve['malicious_ratio_numeric'], 
                                       curve['mean'] - curve['std'].fillna(0),
                                       curve['mean'] + curve['std'].fillna(0),
                                       color=colors[dataset_type], alpha=0.15)
                
                ax.set_xlabel('Malicious Data Ratio (%)', fontsize=12)
                ax.set_ylabel(title, fontsize=12)
                ax.set_title(f'{title} vs Malicious Ratio', fontsize=12)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # Set consistent y-axis range for better comparison
                ax.set_ylim(0.45, 1.05)  # 0.5-1.0 with margins
                
                # Add reference lines for F1
                if metric == 'f1':
                    ax.axhline(y=self.config['reference_lines']['batch6_target'], 
                              color='red', linestyle='--', alpha=0.8, linewidth=2,
                              label=f"Target ({self.config['reference_lines']['batch6_target']})")
                    ax.axhline(y=self.config['reference_lines']['batch5_baseline'], 
                              color='orange', linestyle='--', alpha=0.8, linewidth=2,
                              label=f"Batch5 Baseline ({self.config['reference_lines']['batch5_baseline']})")
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'plots' / 'batch6_performance_curves_datasets.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Model comparison curves
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Batch 6 Performance Curves: Model Comparison', fontsize=16, fontweight='bold')
            
            model_colors = self.config['color_schemes']['models']
            
            for i, (metric, title) in enumerate(zip(self.config['metrics'], self.config['metric_titles'])):
                ax = axes[i//2, i%2]
                
                for model in ['RandomForest', 'SVM', 'DeepLearning']:
                    model_data = curve_data[curve_data['model_name'] == model]
                    if len(model_data) > 0:
                        # Average across dataset types
                        curve = model_data.groupby('malicious_ratio_numeric')[metric].agg(['mean', 'std']).reset_index()
                        
                        ax.plot(curve['malicious_ratio_numeric'], curve['mean'], 
                               color=model_colors[model], linewidth=3, marker='s', markersize=8,
                               label=model)
                        
                        ax.fill_between(curve['malicious_ratio_numeric'], 
                                       curve['mean'] - curve['std'].fillna(0),
                                       curve['mean'] + curve['std'].fillna(0),
                                       color=model_colors[model], alpha=0.2)
                
                ax.set_xlabel('Malicious Data Ratio (%)', fontsize=12)
                ax.set_ylabel(title, fontsize=12)
                ax.set_title(f'{title} vs Malicious Ratio (All Models)', fontsize=12)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # Set consistent y-axis range for better comparison
                ax.set_ylim(0.45, 1.05)  # 0.5-1.0 with margins
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'plots' / 'batch6_performance_curves_models.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("✅ Static performance curves created")
            
        except Exception as e:
            self.logger.error(f"Failed to create static curves: {str(e)}")
            
    def create_interactive_performance_curves(self, curve_data: pd.DataFrame) -> None:
        """Create interactive performance curve visualizations"""
        try:
            self.logger.info("Creating interactive performance curves...")
            
            # 1. Interactive dataset comparison
            fig = sp.make_subplots(
                rows=2, cols=2,
                subplot_titles=self.config['metric_titles'],
                vertical_spacing=0.12,
                horizontal_spacing=0.10
            )
            
            colors = self.config['color_schemes']['performance_curves']
            legends = self.config['academic_legends']['performance_curves']
            
            for i, metric in enumerate(self.config['metrics']):
                row = i // 2 + 1
                col = i % 2 + 1
                
                for dataset_type in ['baseline_real', 'pure_original', 'pure_strong', 'pure_weak']:
                    subset = curve_data[curve_data['dataset_type'] == dataset_type]
                    if len(subset) > 0:
                        # Average curve with error bars
                        curve = subset.groupby('malicious_ratio_numeric')[metric].agg(['mean', 'std', 'count']).reset_index()
                        
                        # Main curve line
                        fig.add_trace(
                            go.Scatter(
                                x=curve['malicious_ratio_numeric'],
                                y=curve['mean'],
                                mode='lines+markers',
                                name=legends[dataset_type],
                                line=dict(color=colors[dataset_type], width=3),
                                marker=dict(size=10),
                                showlegend=(i == 0),  # Only show legend for first subplot
                                legendgroup=dataset_type,
                                hovertemplate=f'<b>{legends[dataset_type]}</b><br>' +\
                                            f'Ratio: %{{x}}%<br>' +\
                                            f'{metric.title()}: %{{y:.4f}}<br>' +\
                                            f'Std: %{{customdata:.4f}}<br>' +\
                                            f'Count: {int(curve["count"].iloc[0]) if len(curve) > 0 else 0}<extra></extra>',
                                customdata=curve['std'].fillna(0)
                            ),
                            row=row, col=col
                        )
                        
                        # Remove individual scattered points - only show averaged curves for cleaner visualization
                
                # Add reference lines for F1
                if metric == 'f1':
                    fig.add_hline(y=self.config['reference_lines']['batch6_target'], 
                                 line_dash="dash", line_color="red", line_width=2,
                                 annotation_text="Target (0.75)", row=row, col=col)
                    fig.add_hline(y=self.config['reference_lines']['batch5_baseline'], 
                                 line_dash="dash", line_color="orange", line_width=2,
                                 annotation_text="Batch5 Baseline (0.60)", row=row, col=col)
            
            fig.update_xaxes(title_text="Malicious Data Ratio (%)")
            fig.update_yaxes(title_text="Score")
            fig.update_layout(
                title="Batch 6 Interactive Performance Curves: Dataset Type Comparison",
                height=800,
                hovermode='closest',
                font=dict(size=12)
            )
            
            pyo.plot(fig, filename=str(self.output_dir / 'interactive' / 'batch6_interactive_curves_datasets.html'), 
                    auto_open=False)
            
            # 2. Interactive model comparison
            fig_models = sp.make_subplots(
                rows=2, cols=2,
                subplot_titles=self.config['metric_titles'],
                vertical_spacing=0.12,
                horizontal_spacing=0.10
            )
            
            model_colors = self.config['color_schemes']['models']
            
            for i, metric in enumerate(self.config['metrics']):
                row = i // 2 + 1
                col = i % 2 + 1
                
                for model in ['RandomForest', 'SVM', 'DeepLearning']:
                    model_data = curve_data[curve_data['model_name'] == model]
                    if len(model_data) > 0:
                        curve = model_data.groupby('malicious_ratio_numeric')[metric].agg(['mean', 'std']).reset_index()
                        
                        fig_models.add_trace(
                            go.Scatter(
                                x=curve['malicious_ratio_numeric'],
                                y=curve['mean'],
                                mode='lines+markers',
                                name=model,
                                line=dict(color=model_colors[model], width=3),
                                marker=dict(size=10, symbol='square'),
                                showlegend=(i == 0),
                                legendgroup=model,
                                hovertemplate=f'<b>{model}</b><br>' +\
                                            f'Ratio: %{{x}}%<br>' +\
                                            f'{metric.title()}: %{{y:.4f}}<br>' +\
                                            f'Std: %{{customdata:.4f}}<extra></extra>',
                                customdata=curve['std'].fillna(0)
                            ),
                            row=row, col=col
                        )
            
            fig_models.update_xaxes(title_text="Malicious Data Ratio (%)")
            fig_models.update_yaxes(title_text="Score")
            fig_models.update_layout(
                title="Batch 6 Interactive Performance Curves: Model Comparison",
                height=800,
                hovermode='closest',
                font=dict(size=12)
            )
            
            pyo.plot(fig_models, filename=str(self.output_dir / 'interactive' / 'batch6_interactive_curves_models.html'), 
                    auto_open=False)
            
            self.logger.info("✅ Interactive performance curves created")
            
        except Exception as e:
            self.logger.error(f"Failed to create interactive curves: {str(e)}")
            
    def create_supplementary_visualizations(self, curve_data: pd.DataFrame, 
                                          structured_results: Dict[str, Any]) -> None:
        """Create supplementary visualizations (heatmaps, distributions, etc.)"""
        try:
            self.logger.info("Creating supplementary visualizations...")
            
            # 1. Performance heatmap
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            pivot_table = curve_data.pivot_table(values='f1', index='dataset_type', 
                                                columns='model_name', aggfunc='mean')
            sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, 
                       cbar_kws={'label': 'F1 Score'})
            ax.set_title('F1 Score Heatmap: Dataset Type × Model', fontsize=14)
            ax.set_xlabel('Model', fontsize=12)
            ax.set_ylabel('Dataset Type', fontsize=12)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'plots' / 'batch6_performance_heatmap.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Distribution plots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Batch 6 Performance Distributions', fontsize=16, fontweight='bold')
            
            for i, (metric, title) in enumerate(zip(self.config['metrics'], self.config['metric_titles'])):
                ax = axes[i//2, i%2]
                
                # Box plot by dataset type
                curve_data.boxplot(column=metric, by='dataset_type', ax=ax)
                ax.set_title(f'{title} Distribution by Dataset Type')
                ax.set_xlabel('Dataset Type')
                ax.set_ylabel(title)
                ax.tick_params(axis='x', rotation=45)
                
                # Add reference lines for F1
                if metric == 'f1':
                    ax.axhline(y=self.config['reference_lines']['batch6_target'], 
                              color='red', linestyle='--', alpha=0.8)
                    ax.axhline(y=self.config['reference_lines']['batch5_baseline'], 
                              color='orange', linestyle='--', alpha=0.8)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'plots' / 'batch6_performance_distributions.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("✅ Supplementary visualizations created")
            
        except Exception as e:
            self.logger.error(f"Failed to create supplementary visualizations: {str(e)}")
            
    def generate_analysis_report(self, curve_data: pd.DataFrame, 
                               structured_results: Dict[str, Any]) -> None:
        """Generate comprehensive analysis report"""
        try:
            self.logger.info("Generating analysis report...")
            
            # Analyze results
            best_f1 = curve_data['f1'].max()
            best_result = curve_data.loc[curve_data['f1'].idxmax()]
            avg_f1 = curve_data['f1'].mean()
            
            # Target achievement analysis
            target_achieved = best_f1 >= self.config['reference_lines']['batch6_target']
            improvement_over_batch5 = best_f1 - self.config['reference_lines']['batch5_baseline']
            
            # Create report
            report_lines = [
                "# Batch 6 Phase 7 Performance Analysis Report",
                f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
                "",
                "## Executive Summary",
                f"- **Total Experiments**: {len(curve_data)}",
                f"- **Average F1 Score**: {avg_f1:.4f}",
                f"- **Best F1 Score**: {best_f1:.4f}",
                f"- **Target Achievement**: {'✅ YES' if target_achieved else '❌ NO'} (Target: {self.config['reference_lines']['batch6_target']})",
                f"- **Improvement over Batch5**: {improvement_over_batch5:.4f} (+{improvement_over_batch5/self.config['reference_lines']['batch5_baseline']*100:.1f}%)",
                "",
                "## Best Performance",
                f"- **Dataset**: {best_result['dataset_name']}",
                f"- **Model**: {best_result['model_name']}",
                f"- **Malicious Ratio**: {best_result['malicious_ratio']}",
                f"- **F1 Score**: {best_result['f1']:.4f}",
                f"- **Accuracy**: {best_result['accuracy']:.4f}",
                "",
                "## Performance by Dataset Type"
            ]
            
            # Add dataset type analysis
            for dataset_type in ['baseline_real', 'pure_original', 'pure_strong', 'pure_weak']:
                subset = curve_data[curve_data['dataset_type'] == dataset_type]
                if len(subset) > 0:
                    report_lines.append(f"- **{dataset_type.replace('_', ' ').title()}**: "
                                      f"Avg F1 = {subset['f1'].mean():.4f}, "
                                      f"Best F1 = {subset['f1'].max():.4f}")
            
            report_lines.extend([
                "",
                "## Performance by Model"
            ])
            
            # Add model analysis
            for model in ['RandomForest', 'SVM', 'DeepLearning']:
                subset = curve_data[curve_data['model_name'] == model]
                if len(subset) > 0:
                    report_lines.append(f"- **{model}**: "
                                      f"Avg F1 = {subset['f1'].mean():.4f}, "
                                      f"Best F1 = {subset['f1'].max():.4f}")
            
            report_lines.extend([
                "",
                "## Key Findings",
                "### Dataset Type Performance Ranking"
            ])
            
            # Rank dataset types by average F1
            dataset_ranking = curve_data.groupby('dataset_type')['f1'].mean().sort_values(ascending=False)
            for i, (dataset_type, avg_f1) in enumerate(dataset_ranking.items(), 1):
                report_lines.append(f"{i}. {dataset_type.replace('_', ' ').title()}: {avg_f1:.4f}")
            
            report_lines.extend([
                "",
                "### Model Performance Ranking"
            ])
            
            # Rank models by average F1
            model_ranking = curve_data.groupby('model_name')['f1'].mean().sort_values(ascending=False)
            for i, (model, avg_f1) in enumerate(model_ranking.items(), 1):
                report_lines.append(f"{i}. {model}: {avg_f1:.4f}")
            
            report_lines.extend([
                "",
                "## Files Generated",
                "### Static Visualizations",
                "- `plots/batch6_performance_curves_datasets.png` - Dataset comparison curves",
                "- `plots/batch6_performance_curves_models.png` - Model comparison curves", 
                "- `plots/batch6_performance_heatmap.png` - Performance heatmap",
                "- `plots/batch6_performance_distributions.png` - Distribution analysis",
                "",
                "### Interactive Visualizations",
                "- `interactive/batch6_interactive_curves_datasets.html` - Interactive dataset curves",
                "- `interactive/batch6_interactive_curves_models.html` - Interactive model curves",
                "",
                "### Publication Data",
                "- `csv_export/batch6_performance_results_publication.csv` - Detailed results",
                "- `csv_export/batch6_performance_summary_publication.csv` - Summary statistics",
                "- `csv_export/batch6_model_comparison_publication.csv` - Model comparison",
                "",
                "---",
                f"*Report generated by Batch 6 Phase 7b on {datetime.now().isoformat()}*"
            ])
            
            # Save report
            report_content = "\n".join(report_lines)
            report_file = self.output_dir / "reports" / "batch6_performance_analysis_report.md"
            with open(report_file, 'w') as f:
                f.write(report_content)
            
            self.logger.info(f"✅ Analysis report generated: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate analysis report: {str(e)}")
            
    def run_phase7b(self) -> None:
        """Execute complete Phase 7b visualization and analysis"""
        try:
            self.logger.info("🚀 Starting Batch 6 Phase 7b: Visualization & Analysis Stage")
            start_time = datetime.now()
            
            # Step 1: Load training results
            training_results = self.load_training_results()
            
            # Step 2: Prepare curve data
            curve_data = self.prepare_curve_data(training_results['detailed'])
            
            # Step 3: Export publication CSV
            self.export_publication_csv(curve_data)
            
            # Step 4: Create static visualizations
            self.create_static_performance_curves(curve_data)
            
            # Step 5: Create interactive visualizations
            self.create_interactive_performance_curves(curve_data)
            
            # Step 6: Create supplementary visualizations
            self.create_supplementary_visualizations(curve_data, training_results['structured'])
            
            # Step 7: Generate analysis report
            self.generate_analysis_report(curve_data, training_results['structured'])
            
            # Phase completion
            end_time = datetime.now()
            runtime = (end_time - start_time).total_seconds() / 60
            
            self.logger.info(f"✅ Batch 6 Phase 7b completed successfully!")
            self.logger.info(f"Runtime: {runtime:.2f} minutes")
            self.logger.info(f"Best F1 Score: {curve_data['f1'].max():.4f}")
            self.logger.info(f"Average F1 Score: {curve_data['f1'].mean():.4f}")
            self.logger.info(f"Output directory: {self.output_dir}")
            
            # Final summary
            target_achieved = curve_data['f1'].max() >= self.config['reference_lines']['batch6_target']
            self.logger.info(f"🎯 Batch 6 Target Achievement: {'✅ SUCCESS' if target_achieved else '❌ NOT MET'}")
            
        except Exception as e:
            self.logger.error(f"❌ Batch 6 Phase 7b failed: {str(e)}")
            raise

def main():
    """Main execution function"""
    visualizer = Batch6Phase7bVisualizer()
    visualizer.run_phase7b()

if __name__ == "__main__":
    main()