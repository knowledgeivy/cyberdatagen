#!/usr/bin/env python3
"""
批次4增强版 - Phase 2: Interactive可视化

基于batch3框架，实现：
1. 统一2D scatter plot
2. 多层对比静态图
3. 基础交互功能
4. 为Phase 3的3D可视化做准备

作者: Claude
创建时间: 2025-07-27
基于: 批次4embedding可视化改进设计方案.md + batch3_visualize_results.py
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import random
from typing import Dict, List, Tuple, Any, Optional
import logging
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# 项目路径设置
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.cyberdata.utils.logger_config import setup_logger
from src.cyberdata.utils.data_loader import load_csv_data

class Batch4InteractiveVisualizer:
    """批次4交互式可视化器"""
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logger("batch4_interactive")
        self.project_root = PROJECT_ROOT
        self.batch4_dir = self.project_root / "data" / "batch4_fresh"
        self.enhanced_dir = self.batch4_dir / "enhanced_visualizations"
        
        # 创建输出目录
        self.vis_dir = self.enhanced_dir / "phase2_visualizations"
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置可视化样式
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 设置随机种子
        self.random_state = 2025
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        
        # 颜色配置（基于设计方案）
        self.color_config = {
            'real_malicious': '#1f77b4',    # 蓝色
            'real_benign': '#2ca02c',       # 绿色
            'core_synthetic': '#d62728',    # 红色
            'inner_synthetic': '#ff7f0e',   # 橙色
            'outer_synthetic': '#9467bd',   # 紫色
            'edge_synthetic': '#bcbd22'     # 黄色
        }
        
        self.logger.info("Phase 2交互式可视化器初始化完成")
    
    def load_unified_data(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """加载Phase 1的统一数据"""
        try:
            self.logger.info("加载Phase 1统一数据...")
            
            # 加载主数据
            data_file = self.enhanced_dir / "unified_embedding_data.csv"
            if not data_file.exists():
                raise FileNotFoundError(f"统一数据文件不存在: {data_file}")
            
            df = pd.read_csv(data_file)
            self.logger.info(f"✅ 加载统一数据: {len(df)} 样本")
            
            # 加载元数据
            metadata_file = self.enhanced_dir / "unified_data_metadata.json"
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            self.logger.info(f"✅ 加载元数据: {metadata['total_samples']} 样本")
            
            # 数据验证
            required_columns = ['pca_x', 'pca_y', 'tsne_x', 'tsne_y', 
                              'data_source', 'distance_to_centroid']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"缺失必要列: {missing_cols}")
            
            self.logger.info("数据验证通过")
            return df, metadata
            
        except Exception as e:
            self.logger.error(f"加载统一数据失败: {str(e)}")
            raise
    
    def create_unified_2d_scatter(self, df: pd.DataFrame) -> None:
        """创建统一的2D散点图 (静态版本)"""
        try:
            self.logger.info("创建统一2D散点图...")
            
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))
            
            # PCA散点图
            ax1 = axes[0]
            for source in df['data_source'].unique():
                source_data = df[df['data_source'] == source]
                color = self.color_config.get(source, '#777777')
                ax1.scatter(source_data['pca_x'], source_data['pca_y'], 
                          c=color, label=source.replace('_', ' ').title(), 
                          alpha=0.6, s=20)
            
            ax1.set_xlabel('PCA Component 1')
            ax1.set_ylabel('PCA Component 2')
            ax1.set_title('Unified Embedding Space - PCA Visualization')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # t-SNE散点图
            ax2 = axes[1]
            for source in df['data_source'].unique():
                source_data = df[df['data_source'] == source]
                color = self.color_config.get(source, '#777777')
                ax2.scatter(source_data['tsne_x'], source_data['tsne_y'], 
                          c=color, label=source.replace('_', ' ').title(), 
                          alpha=0.6, s=20)
            
            ax2.set_xlabel('t-SNE Component 1')
            ax2.set_ylabel('t-SNE Component 2')
            ax2.set_title('Unified Embedding Space - t-SNE Visualization')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图片
            static_file = self.vis_dir / "unified_2d_scatter_static.png"
            plt.savefig(static_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ 统一2D散点图保存到: {static_file}")
            
        except Exception as e:
            self.logger.error(f"创建统一2D散点图失败: {str(e)}")
            raise
    
    def create_layer_comparison_matrix(self, df: pd.DataFrame) -> None:
        """创建层间对比矩阵图"""
        try:
            self.logger.info("创建层间对比矩阵...")
            
            # 准备数据
            layer_stats = []
            
            for source in df['data_source'].unique():
                source_data = df[df['data_source'] == source]
                
                stats = {
                    'data_source': source,
                    'count': len(source_data),
                    'mean_distance': source_data['distance_to_centroid'].mean(),
                    'std_distance': source_data['distance_to_centroid'].std(),
                    'mean_pca_x': source_data['pca_x'].mean(),
                    'mean_pca_y': source_data['pca_y'].mean(),
                    'mean_tsne_x': source_data['tsne_x'].mean(),
                    'mean_tsne_y': source_data['tsne_y'].mean()
                }
                layer_stats.append(stats)
            
            stats_df = pd.DataFrame(layer_stats)
            
            # 创建热力图
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 距离统计热力图
            distance_matrix = stats_df[['data_source', 'mean_distance', 'std_distance']].set_index('data_source')
            sns.heatmap(distance_matrix.T, annot=True, fmt='.3f', cmap='viridis', ax=axes[0, 0])
            axes[0, 0].set_title('Distance Statistics by Layer')
            
            # 样本计数
            count_data = stats_df[['data_source', 'count']].set_index('data_source')
            sns.barplot(data=stats_df, x='data_source', y='count', ax=axes[0, 1])
            axes[0, 1].set_title('Sample Count by Layer')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # PCA中心位置
            pca_centers = stats_df[['data_source', 'mean_pca_x', 'mean_pca_y']].set_index('data_source')
            sns.heatmap(pca_centers.T, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 0])
            axes[1, 0].set_title('PCA Center Positions')
            
            # t-SNE中心位置
            tsne_centers = stats_df[['data_source', 'mean_tsne_x', 'mean_tsne_y']].set_index('data_source')
            sns.heatmap(tsne_centers.T, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 1])
            axes[1, 1].set_title('t-SNE Center Positions')
            
            plt.tight_layout()
            
            # 保存图片
            matrix_file = self.vis_dir / "layer_comparison_matrix.png"
            plt.savefig(matrix_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ 层间对比矩阵保存到: {matrix_file}")
            
            # 保存统计数据
            stats_file = self.vis_dir / "layer_statistics.csv"
            stats_df.to_csv(stats_file, index=False)
            self.logger.info(f"✅ 层间统计数据保存到: {stats_file}")
            
        except Exception as e:
            self.logger.error(f"创建层间对比矩阵失败: {str(e)}")
            raise
    
    def create_interactive_2d_plot(self, df: pd.DataFrame) -> go.Figure:
        """创建交互式2D可视化（基于batch3框架）"""
        try:
            self.logger.info("创建交互式2D可视化...")
            
            # 创建子图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('PCA Visualization', 't-SNE Visualization', 
                              'Distance Distribution', 'Prompt Type Analysis'),
                specs=[[{"type": "scatter"}, {"type": "scatter"}],
                       [{"type": "histogram"}, {"type": "scatter"}]]
            )
            
            # PCA可视化
            for source in df['data_source'].unique():
                source_data = df[df['data_source'] == source]
                color = self.color_config.get(source, '#777777')
                
                fig.add_trace(
                    go.Scatter(
                        x=source_data['pca_x'],
                        y=source_data['pca_y'],
                        mode='markers',
                        name=f'{source.replace("_", " ").title()}',
                        marker=dict(color=color, size=5, opacity=0.6),
                        hovertemplate='<b>%{text}</b><br>' +
                                    'PCA: (%{x:.2f}, %{y:.2f})<br>' +
                                    'Distance: %{customdata:.3f}<br>' +
                                    '<extra></extra>',
                        text=[f'{source}<br>ID: {oid}' for oid in source_data['original_id']],
                        customdata=source_data['distance_to_centroid'],
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            # t-SNE可视化  
            for source in df['data_source'].unique():
                source_data = df[df['data_source'] == source]
                color = self.color_config.get(source, '#777777')
                
                fig.add_trace(
                    go.Scatter(
                        x=source_data['tsne_x'],
                        y=source_data['tsne_y'],
                        mode='markers',
                        name=f'{source.replace("_", " ").title()}',
                        marker=dict(color=color, size=5, opacity=0.6),
                        hovertemplate='<b>%{text}</b><br>' +
                                    't-SNE: (%{x:.2f}, %{y:.2f})<br>' +
                                    'Distance: %{customdata:.3f}<br>' +
                                    '<extra></extra>',
                        text=[f'{source}<br>ID: {oid}' for oid in source_data['original_id']],
                        customdata=source_data['distance_to_centroid'],
                        showlegend=False
                    ),
                    row=1, col=2
                )
            
            # 距离分布直方图
            for source in df['data_source'].unique():
                source_data = df[df['data_source'] == source]
                color = self.color_config.get(source, '#777777')
                
                fig.add_trace(
                    go.Histogram(
                        x=source_data['distance_to_centroid'],
                        name=f'{source.replace("_", " ").title()}',
                        marker_color=color,
                        opacity=0.7,
                        showlegend=False
                    ),
                    row=2, col=1
                )
            
            # Prompt类型分析散点图
            prompt_colors = {'real': '#1f77b4', 'rewrite': '#ff7f0e', 
                           'rewrite_strong': '#d62728', 'rewrite_weak': '#2ca02c'}
            
            for prompt_type in df['prompt_type'].unique():
                prompt_data = df[df['prompt_type'] == prompt_type]
                color = prompt_colors.get(prompt_type, '#777777')
                
                fig.add_trace(
                    go.Scatter(
                        x=prompt_data['distance_to_centroid'],
                        y=prompt_data['pca_x'],  # 使用PCA第一维作为Y轴
                        mode='markers',
                        name=f'Prompt: {prompt_type.title()}',
                        marker=dict(color=color, size=4, opacity=0.6),
                        hovertemplate='<b>%{text}</b><br>' +
                                    'Distance: %{x:.3f}<br>' +
                                    'PCA-1: %{y:.2f}<br>' +
                                    '<extra></extra>',
                        text=[f'{prompt_type}<br>{src}' for src in prompt_data['data_source']],
                        showlegend=False
                    ),
                    row=2, col=2
                )
            
            # 更新布局
            fig.update_layout(
                title="Batch4 Enhanced Interactive Visualization",
                height=800,
                showlegend=True,
                hovermode='closest'
            )
            
            # 更新坐标轴标签
            fig.update_xaxes(title_text="PCA Component 1", row=1, col=1)
            fig.update_yaxes(title_text="PCA Component 2", row=1, col=1)
            fig.update_xaxes(title_text="t-SNE Component 1", row=1, col=2)
            fig.update_yaxes(title_text="t-SNE Component 2", row=1, col=2)
            fig.update_xaxes(title_text="Distance to Centroid", row=2, col=1)
            fig.update_yaxes(title_text="Frequency", row=2, col=1)
            fig.update_xaxes(title_text="Distance to Centroid", row=2, col=2)
            fig.update_yaxes(title_text="PCA Component 1", row=2, col=2)
            
            self.logger.info("✅ 交互式2D可视化创建完成")
            return fig
            
        except Exception as e:
            self.logger.error(f"创建交互式2D可视化失败: {str(e)}")
            raise
    
    def create_distance_analysis_plot(self, df: pd.DataFrame) -> go.Figure:
        """创建距离分析图表"""
        try:
            self.logger.info("创建距离分析图表...")
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Distance by Layer', 'Density Group Distribution',
                              'Distance vs PCA Scatter', 'Distance vs t-SNE Scatter'),
                specs=[[{"type": "box"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "scatter"}]]
            )
            
            # 1. 按层的距离箱线图
            for source in df['data_source'].unique():
                source_data = df[df['data_source'] == source]
                color = self.color_config.get(source, '#777777')
                
                fig.add_trace(
                    go.Box(
                        y=source_data['distance_to_centroid'],
                        name=source.replace('_', ' ').title(),
                        marker_color=color,
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # 2. 密度组分布
            density_counts = df['density_group'].value_counts()
            fig.add_trace(
                go.Bar(
                    x=density_counts.index,
                    y=density_counts.values,
                    name='Density Groups',
                    marker_color='lightblue',
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # 3. 距离 vs PCA散点图
            fig.add_trace(
                go.Scatter(
                    x=df['distance_to_centroid'],
                    y=df['pca_x'],
                    mode='markers',
                    marker=dict(
                        color=df['pca_y'],
                        colorscale='viridis',
                        size=3,
                        opacity=0.6,
                        colorbar=dict(title="PCA-Y")
                    ),
                    name='Distance vs PCA',
                    hovertemplate='Distance: %{x:.3f}<br>PCA-X: %{y:.2f}<br><extra></extra>',
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # 4. 距离 vs t-SNE散点图
            fig.add_trace(
                go.Scatter(
                    x=df['distance_to_centroid'],
                    y=df['tsne_x'],
                    mode='markers',
                    marker=dict(
                        color=df['tsne_y'],
                        colorscale='plasma',
                        size=3,
                        opacity=0.6,
                        colorbar=dict(title="t-SNE-Y")
                    ),
                    name='Distance vs t-SNE',
                    hovertemplate='Distance: %{x:.3f}<br>t-SNE-X: %{y:.2f}<br><extra></extra>',
                    showlegend=False
                ),
                row=2, col=2
            )
            
            # 更新布局
            fig.update_layout(
                title="Distance Analysis - Multi-dimensional View",
                height=800,
                showlegend=False
            )
            
            # 更新轴标签
            fig.update_yaxes(title_text="Distance to Centroid", row=1, col=1)
            fig.update_xaxes(title_text="Density Group", row=1, col=2)
            fig.update_yaxes(title_text="Count", row=1, col=2)
            fig.update_xaxes(title_text="Distance to Centroid", row=2, col=1)
            fig.update_yaxes(title_text="PCA Component 1", row=2, col=1)
            fig.update_xaxes(title_text="Distance to Centroid", row=2, col=2)
            fig.update_yaxes(title_text="t-SNE Component 1", row=2, col=2)
            
            self.logger.info("✅ 距离分析图表创建完成")
            return fig
            
        except Exception as e:
            self.logger.error(f"创建距离分析图表失败: {str(e)}")
            raise
    
    def save_interactive_plots(self, main_fig: go.Figure, 
                             distance_fig: go.Figure) -> Dict[str, str]:
        """保存交互式图表"""
        try:
            self.logger.info("保存交互式图表...")
            
            output_files = {}
            
            # 保存主要交互图表
            main_file = self.vis_dir / "batch4_interactive_2d.html"
            main_fig.write_html(main_file)
            output_files['main_interactive'] = str(main_file)
            self.logger.info(f"✅ 主交互图表保存到: {main_file}")
            
            # 保存距离分析图表
            distance_file = self.vis_dir / "batch4_distance_analysis.html"
            distance_fig.write_html(distance_file)
            output_files['distance_analysis'] = str(distance_file)
            self.logger.info(f"✅ 距离分析图表保存到: {distance_file}")
            
            return output_files
            
        except Exception as e:
            self.logger.error(f"保存交互式图表失败: {str(e)}")
            raise
    
    def generate_phase2_summary(self, df: pd.DataFrame, 
                               output_files: Dict[str, str]) -> Dict[str, Any]:
        """生成Phase 2总结报告"""
        try:
            self.logger.info("生成Phase 2总结报告...")
            
            # 数据统计
            data_summary = {
                'total_samples': len(df),
                'data_sources': df['data_source'].value_counts().to_dict(),
                'prompt_types': df['prompt_type'].value_counts().to_dict(),
                'density_groups': df['density_group'].value_counts().to_dict()
            }
            
            # 距离统计
            distance_stats = {}
            for source in df['data_source'].unique():
                source_data = df[df['data_source'] == source]
                distance_stats[source] = {
                    'count': len(source_data),
                    'mean': float(source_data['distance_to_centroid'].mean()),
                    'std': float(source_data['distance_to_centroid'].std()),
                    'min': float(source_data['distance_to_centroid'].min()),
                    'max': float(source_data['distance_to_centroid'].max())
                }
            
            # 可视化质量指标
            vis_quality = {
                'pca_variance_explained': {
                    'component_1': 0.115,  # 从Phase 1获得
                    'component_2': 0.105,
                    'total': 0.220
                },
                'embedding_space_coverage': {
                    'pca_x_range': [float(df['pca_x'].min()), float(df['pca_x'].max())],
                    'pca_y_range': [float(df['pca_y'].min()), float(df['pca_y'].max())],
                    'tsne_x_range': [float(df['tsne_x'].min()), float(df['tsne_x'].max())],
                    'tsne_y_range': [float(df['tsne_y'].min()), float(df['tsne_y'].max())]
                }
            }
            
            # 综合报告
            summary = {
                'phase2_info': {
                    'date': time.strftime('%Y-%m-%d'),
                    'visualization_type': 'Interactive 2D + Static Analysis',
                    'framework_base': 'batch3_visualize_results.py',
                    'total_visualizations': len(output_files) + 2  # +2 for static plots
                },
                'data_summary': data_summary,
                'distance_analysis': distance_stats,
                'visualization_quality': vis_quality,
                'output_files': output_files,
                'interactive_features': {
                    'hover_information': True,
                    'layer_filtering': False,  # Phase 3功能
                    'dynamic_color_coding': True,
                    'multi_plot_coordination': True
                },
                'next_phase_readiness': {
                    'unified_data_available': True,
                    'distance_calculated': True,
                    'embedding_reduced': True,
                    'metadata_complete': True
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"生成Phase 2总结报告失败: {str(e)}")
            raise
    
    def run_phase2_complete(self) -> Dict[str, Any]:
        """执行完整的Phase 2流程"""
        try:
            self.logger.info("="*60)
            self.logger.info("开始执行批次4增强版 Phase 2: Interactive可视化")
            self.logger.info("="*60)
            
            start_time = time.time()
            
            # Step 1: 加载统一数据
            self.logger.info("Step 1: 加载Phase 1统一数据")
            df, metadata = self.load_unified_data()
            
            # Step 2: 创建静态可视化
            self.logger.info("Step 2: 创建静态可视化")
            self.create_unified_2d_scatter(df)
            self.create_layer_comparison_matrix(df)
            
            # Step 3: 创建交互式可视化
            self.logger.info("Step 3: 创建交互式可视化")
            main_interactive_fig = self.create_interactive_2d_plot(df)
            distance_analysis_fig = self.create_distance_analysis_plot(df)
            
            # Step 4: 保存交互式图表
            self.logger.info("Step 4: 保存交互式图表")
            output_files = self.save_interactive_plots(main_interactive_fig, distance_analysis_fig)
            
            # Step 5: 生成总结报告
            self.logger.info("Step 5: 生成Phase 2总结报告")
            summary = self.generate_phase2_summary(df, output_files)
            
            # 保存总结报告
            summary_file = self.vis_dir / "phase2_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            elapsed_time = time.time() - start_time
            
            self.logger.info("="*60)
            self.logger.info("批次4增强版 Phase 2 执行完成!")
            self.logger.info(f"总耗时: {elapsed_time/60:.1f} 分钟")
            self.logger.info(f"处理样本数: {len(df):,}")
            self.logger.info(f"生成可视化: {len(output_files) + 2}")
            self.logger.info(f"输出目录: {self.vis_dir}")
            
            # 显示关键发现
            self.logger.info("关键发现:")
            self.logger.info(f"- 数据源: {len(df['data_source'].unique())} 种")
            self.logger.info(f"- Prompt类型: {len(df['prompt_type'].unique())} 种")
            self.logger.info(f"- 距离范围: [{df['distance_to_centroid'].min():.3f}, {df['distance_to_centroid'].max():.3f}]")
            
            # 层间距离差异
            layer_distances = df.groupby('data_source')['distance_to_centroid'].mean()
            self.logger.info("层间平均距离:")
            for source, dist in layer_distances.items():
                self.logger.info(f"  - {source}: {dist:.3f}")
            
            self.logger.info("="*60)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Phase 2执行失败: {str(e)}")
            raise

def main():
    """主函数"""
    # 设置日志
    logger = setup_logger("batch4_phase2", log_level=logging.INFO)
    
    try:
        # 创建可视化器
        visualizer = Batch4InteractiveVisualizer(logger)
        
        # 执行Phase 2
        summary = visualizer.run_phase2_complete()
        
        logger.info("批次4增强版 Phase 2 successfully completed!")
        return 0
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())