#!/usr/bin/env python3
"""
批次5实验 - Phase 2: 增强可视化分析

全面可视化不同数据源在embedding空间的分布，包括：
1. 数据源对比: Real vs Synthetic (按层)
2. Prompt策略对比: rewrite vs rewrite_strong vs rewrite_weak  
3. 分层效果对比: core vs inner vs outer vs edge
4. 聚类特征分析: 自动识别cluster数量和边界

可视化类型:
- 2D静态图: PCA/t-SNE降维可视化
- 3D交互图: 支持旋转和缩放的立体展示
- 聚类分析图: DBSCAN/K-means聚类结果
- 距离分布图: 各层到质心的距离分布对比

作者: Claude
创建时间: 2025-07-30
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import random
from typing import Dict, List, Tuple, Any
import logging
import json
import time

# 可视化库
import matplotlib
matplotlib.use('Agg')  # 设置后端为Agg，避免显示问题
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

# 项目路径设置
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.cyberdata.utils.logger_config import setup_logger
from src.cyberdata.utils.data_loader import load_csv_data

class Batch5Phase2EnhancedVisualization:
    """批次5 Phase 2: 增强可视化分析器"""
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logger("batch5_phase2")
        self.project_root = PROJECT_ROOT
        self.batch5_dir = self.project_root / "data" / "batch5"
        
        # 输入和输出目录
        self.unified_dir = self.batch5_dir / "unified_embeddings"
        self.viz_dir = self.batch5_dir / "visualizations"
        self.analysis_dir = self.batch5_dir / "phase2_analysis"
        
        # 创建输出目录
        for dir_path in [self.viz_dir, self.analysis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.static_dir = self.viz_dir / "static_plots"
        self.interactive_dir = self.viz_dir / "interactive_plots"
        self.cluster_dir = self.viz_dir / "cluster_analysis"
        self.dist_dir = self.viz_dir / "distribution_analysis"
        
        for dir_path in [self.static_dir, self.interactive_dir, self.cluster_dir, self.dist_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 可视化配置
        self.color_palette = {
            'real_malicious_background': '#FFB6C1',     # 浅粉 - 真实恶意背景数据
            'real_malicious_seeds': '#FF1493',          # 深粉 - LLM种子样本
            'synthetic_rewrite': '#4ECDC4',             # 青色 - 基础重写
            'synthetic_rewrite_strong': '#C44569',      # 深红 - 强化重写
            'synthetic_rewrite_weak': '#F8B500',        # 橙色 - 弱化重写
            'real_benign': '#45B7D1',                   # 蓝色 - 真实良性
            'test_set': '#96CEB4',                      # 绿色 - 测试集
            'core': '#FF8C94',                          # 粉红 - core层
            'inner': '#FFD93D',                         # 黄色 - inner层
            'outer': '#6BCF7F',                         # 浅绿 - outer层
            'edge': '#A8E6CF',                          # 淡绿 - edge层
            # 保持向后兼容
            'real_malicious': '#FF6B6B',
            'real_malicious_sample': '#FFB6C1',         # 映射到背景数据
            'synthetic': '#4ECDC4',
            'rewrite': '#4ECDC4',
            'rewrite_strong': '#C44569',
            'rewrite_weak': '#F8B500'
        }
        
        # 随机种子
        self.random_state = 2025
        
        self.logger.info("Phase 2增强可视化初始化完成")
    
    def load_unified_data(self) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
        """加载统一embedding数据"""
        try:
            self.logger.info("加载统一embedding数据...")
            
            # 加载元数据 (压缩格式)
            metadata_file = self.unified_dir / "embedding_metadata.csv.gz"
            metadata_df = load_csv_data(metadata_file, logger=self.logger)
            
            # 加载embedding矩阵
            embeddings_file = self.unified_dir / "all_embeddings.npy"
            embeddings = np.load(embeddings_file)
            
            # 加载质心信息
            centroids_file = self.unified_dir / "layer_centroids.json"
            with open(centroids_file, 'r', encoding='utf-8') as f:
                centroids = json.load(f)
            
            self.logger.info(f"数据加载完成: {len(metadata_df):,} 样本, {embeddings.shape[1]}维embedding")
            
            return metadata_df, embeddings, centroids
            
        except Exception as e:
            self.logger.error(f"加载统一embedding数据失败: {str(e)}")
            raise
    
    def perform_dimensionality_reduction(self, embeddings: np.ndarray) -> Dict[str, np.ndarray]:
        """执行降维分析"""
        try:
            self.logger.info("执行降维分析...")
            
            results = {}
            
            # PCA降维
            self.logger.info("执行PCA降维...")
            pca_2d = PCA(n_components=2, random_state=self.random_state)
            pca_3d = PCA(n_components=3, random_state=self.random_state)
            
            results['pca_2d'] = pca_2d.fit_transform(embeddings)
            results['pca_3d'] = pca_3d.fit_transform(embeddings)
            
            # 记录解释方差比
            results['pca_2d_variance'] = pca_2d.explained_variance_ratio_
            results['pca_3d_variance'] = pca_3d.explained_variance_ratio_
            
            self.logger.info(f"PCA-2D 解释方差: {results['pca_2d_variance'].sum():.3f}")
            self.logger.info(f"PCA-3D 解释方差: {results['pca_3d_variance'].sum():.3f}")
            
            # t-SNE降维（仅对采样数据，避免计算过慢）
            if len(embeddings) > 5000:
                sample_indices = np.random.choice(len(embeddings), 5000, replace=False)
                sample_embeddings = embeddings[sample_indices]
                self.logger.info(f"t-SNE采样分析: {len(sample_embeddings):,} 样本")
            else:
                sample_embeddings = embeddings
                sample_indices = np.arange(len(embeddings))
            
            self.logger.info("执行t-SNE降维...")
            tsne_2d = TSNE(n_components=2, random_state=self.random_state, perplexity=30)
            tsne_3d = TSNE(n_components=3, random_state=self.random_state, perplexity=30)
            
            tsne_2d_result = tsne_2d.fit_transform(sample_embeddings)
            tsne_3d_result = tsne_3d.fit_transform(sample_embeddings)
            
            # 为完整数据集填充t-SNE结果
            results['tsne_2d'] = np.full((len(embeddings), 2), np.nan)
            results['tsne_3d'] = np.full((len(embeddings), 3), np.nan)
            results['tsne_2d'][sample_indices] = tsne_2d_result
            results['tsne_3d'][sample_indices] = tsne_3d_result
            results['tsne_sample_indices'] = sample_indices
            
            self.logger.info("降维分析完成")
            
            return results
            
        except Exception as e:
            self.logger.error(f"降维分析失败: {str(e)}")
            raise
    
    def perform_clustering_analysis(self, embeddings: np.ndarray, metadata_df: pd.DataFrame) -> Dict[str, Any]:
        """执行聚类分析"""
        try:
            self.logger.info("执行聚类分析...")
            
            results = {}
            
            # 采样进行聚类分析（提高效率）
            if len(embeddings) > 10000:
                sample_indices = np.random.choice(len(embeddings), 10000, replace=False)
                sample_embeddings = embeddings[sample_indices]
                sample_metadata = metadata_df.iloc[sample_indices].copy()
                self.logger.info(f"聚类分析采样: {len(sample_embeddings):,} 样本")
            else:
                sample_embeddings = embeddings
                sample_metadata = metadata_df.copy()
                sample_indices = np.arange(len(embeddings))
            
            # K-means聚类 - 寻找最佳k值
            self.logger.info("K-means聚类分析...")
            k_range = range(2, 11)
            silhouette_scores = []
            inertias = []
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                cluster_labels = kmeans.fit_predict(sample_embeddings)
                silhouette_avg = silhouette_score(sample_embeddings, cluster_labels)
                silhouette_scores.append(silhouette_avg)
                inertias.append(kmeans.inertia_)
                
                self.logger.info(f"K={k}: 轮廓系数={silhouette_avg:.3f}, 惯性={kmeans.inertia_:.0f}")
            
            # 选择最佳k值
            best_k = k_range[np.argmax(silhouette_scores)]
            results['best_k'] = best_k
            results['silhouette_scores'] = dict(zip(k_range, silhouette_scores))
            results['inertias'] = dict(zip(k_range, inertias))
            
            # 使用最佳k值进行聚类
            final_kmeans = KMeans(n_clusters=best_k, random_state=self.random_state, n_init=10)
            kmeans_labels = final_kmeans.fit_predict(sample_embeddings)
            results['kmeans_labels'] = kmeans_labels
            results['kmeans_centers'] = final_kmeans.cluster_centers_
            
            self.logger.info(f"最佳K-means聚类: k={best_k}, 轮廓系数={max(silhouette_scores):.3f}")
            
            # DBSCAN聚类
            self.logger.info("DBSCAN聚类分析...")
            dbscan = DBSCAN(eps=0.5, min_samples=10)
            dbscan_labels = dbscan.fit_predict(sample_embeddings)
            
            n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            n_noise = list(dbscan_labels).count(-1)
            
            results['dbscan_labels'] = dbscan_labels
            results['dbscan_n_clusters'] = n_clusters
            results['dbscan_n_noise'] = n_noise
            
            self.logger.info(f"DBSCAN聚类: {n_clusters}个簇, {n_noise}个噪声点")
            
            # 聚类质量评估
            if n_clusters > 1:
                valid_mask = dbscan_labels != -1
                if valid_mask.sum() > 0:
                    dbscan_silhouette = silhouette_score(
                        sample_embeddings[valid_mask], 
                        dbscan_labels[valid_mask]
                    )
                    results['dbscan_silhouette'] = dbscan_silhouette
                    self.logger.info(f"DBSCAN轮廓系数: {dbscan_silhouette:.3f}")
            
            results['sample_indices'] = sample_indices
            results['sample_metadata'] = sample_metadata
            
            return results
            
        except Exception as e:
            self.logger.error(f"聚类分析失败: {str(e)}")
            raise
    
    def create_static_visualizations(self, metadata_df: pd.DataFrame, dim_results: Dict[str, np.ndarray],
                                   cluster_results: Dict[str, Any], embeddings: np.ndarray) -> None:
        """创建静态可视化图表"""
        try:
            self.logger.info("创建静态可视化图表...")
            
            # 设置matplotlib样式
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. PCA 2D散点图 - 按数据源着色
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Batch 5 Unified Embedding Space Visualization Analysis', fontsize=16, fontweight='bold')
            
            # 1.1 按数据源着色
            ax1 = axes[0, 0]
            for source in metadata_df['data_source'].unique():
                mask = metadata_df['data_source'] == source
                color = self.color_palette.get(source, '#888888')
                ax1.scatter(dim_results['pca_2d'][mask, 0], dim_results['pca_2d'][mask, 1], 
                           c=color, label=source, alpha=0.6, s=20)
            
            ax1.set_xlabel(f'PC1 ({dim_results["pca_2d_variance"][0]:.1%} variance)')
            ax1.set_ylabel(f'PC2 ({dim_results["pca_2d_variance"][1]:.1%} variance)')
            ax1.set_title('PCA Dimensionality Reduction - By Data Source')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 1.2 按合成数据层着色（仅合成数据）
            ax2 = axes[0, 1]
            synthetic_sources = ['synthetic_rewrite', 'synthetic_rewrite_strong', 'synthetic_rewrite_weak']
            synthetic_mask = metadata_df['data_source'].isin(synthetic_sources)
            if synthetic_mask.any():
                synthetic_df = metadata_df[synthetic_mask]
                pca_synthetic = dim_results['pca_2d'][synthetic_mask]
                
                for layer in synthetic_df['layer'].unique():
                    layer_mask = synthetic_df['layer'] == layer
                    color = self.color_palette.get(layer, '#888888')
                    ax2.scatter(pca_synthetic[layer_mask, 0], pca_synthetic[layer_mask, 1], 
                               c=color, label=f'{layer} layer', alpha=0.7, s=25)
            
            ax2.set_xlabel(f'PC1 ({dim_results["pca_2d_variance"][0]:.1%} variance)')
            ax2.set_ylabel(f'PC2 ({dim_results["pca_2d_variance"][1]:.1%} variance)')
            ax2.set_title('PCA Dimensionality Reduction - Synthetic Data Layer Analysis')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 1.3 按prompt变体着色（仅合成数据）
            ax3 = axes[1, 0]
            if synthetic_mask.any():
                for variant in synthetic_df['prompt_variant'].unique():
                    if pd.notna(variant):
                        variant_mask = synthetic_df['prompt_variant'] == variant
                        color = self.color_palette.get(variant, '#888888')
                        ax3.scatter(pca_synthetic[variant_mask, 0], pca_synthetic[variant_mask, 1], 
                                   c=color, label=variant, alpha=0.7, s=25)
            
            ax3.set_xlabel(f'PC1 ({dim_results["pca_2d_variance"][0]:.1%} variance)')
            ax3.set_ylabel(f'PC2 ({dim_results["pca_2d_variance"][1]:.1%} variance)')
            ax3.set_title('PCA Dimensionality Reduction - Prompt Variant Analysis')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 1.4 K-means聚类结果
            ax4 = axes[1, 1]
            sample_indices = cluster_results['sample_indices']
            sample_pca = dim_results['pca_2d'][sample_indices]
            kmeans_labels = cluster_results['kmeans_labels']
            
            scatter = ax4.scatter(sample_pca[:, 0], sample_pca[:, 1], 
                                 c=kmeans_labels, cmap='tab10', alpha=0.6, s=20)
            
            # 绘制聚类中心（需要用原始PCA转换器变换到2D空间）
            # 创建临时PCA来转换聚类中心到2D空间
            temp_pca = PCA(n_components=2, random_state=self.random_state)
            # 先用采样数据拟合PCA
            sample_indices = cluster_results['sample_indices']
            sample_embeddings = embeddings[sample_indices]  # 需要传入embeddings参数
            temp_pca.fit(sample_embeddings)
            centers_pca = temp_pca.transform(cluster_results['kmeans_centers'])
            ax4.scatter(centers_pca[:, 0], centers_pca[:, 1], 
                       c='red', marker='x', s=200, linewidths=3, label='Cluster Centers')
            
            ax4.set_xlabel(f'PC1 ({dim_results["pca_2d_variance"][0]:.1%} variance)')  
            ax4.set_ylabel(f'PC2 ({dim_results["pca_2d_variance"][1]:.1%} variance)')
            ax4.set_title(f'K-means Clustering (k={cluster_results["best_k"]})')
            ax4.legend(loc='upper right', framealpha=0.9)
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            static_overview_file = self.static_dir / "embedding_overview_analysis.png"
            plt.savefig(static_overview_file, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"静态总览图已保存: {static_overview_file}")
            
            # 2. 新增：分层对比分析
            self.create_layer_comparison_plots(metadata_df, dim_results)
            
            # 3. 距离分布分析
            self.create_distance_distribution_plots(metadata_df, dim_results)
            
            # 4. 聚类质量分析
            self.create_clustering_quality_plots(cluster_results)
            
        except Exception as e:
            self.logger.error(f"创建静态可视化失败: {str(e)}")
            raise
    
    def create_layer_comparison_plots(self, metadata_df: pd.DataFrame, 
                                    dim_results: Dict[str, np.ndarray]) -> None:
        """创建分层对比分析图 - Synthetic vs Real Malicious"""
        try:
            self.logger.info("创建分层对比分析图...")
            
            # 定义统一的形状映射 (matplotlib markers)
            marker_mapping = {
                'real_malicious_background': 'o',      # circle
                'real_malicious_seeds': '*',           # star
                'synthetic_rewrite': 'o',              # circle
                'synthetic_rewrite_strong': 's',       # square
                'synthetic_rewrite_weak': 'D',         # diamond
                'core': 'o',                           # circle
                'inner': 's',                          # square
                'outer': 'D',                          # diamond
                'edge': '^'                            # triangle
            }
            
            # 准备数据
            pca_2d = dim_results['pca_2d']
            
            # 创建大图 (2x2)
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Synthetic Data Layer vs Real Malicious Comparison Analysis', 
                        fontsize=16, fontweight='bold')
            
            # 获取real malicious数据 (seeds + background)
            real_seeds_mask = metadata_df['data_source'] == 'real_malicious_seeds'
            real_bg_mask = metadata_df['data_source'] == 'real_malicious_background'
            
            # 1. 各层与种子样本对比
            ax1 = axes[0, 0]
            
            # 先画背景样本 (浅色)
            if real_bg_mask.any():
                ax1.scatter(pca_2d[real_bg_mask, 0], pca_2d[real_bg_mask, 1], 
                           c=self.color_palette['real_malicious_background'], 
                           alpha=0.2, s=8, label='Real Background',
                           marker=marker_mapping['real_malicious_background'])
            
            # 再画种子样本
            if real_seeds_mask.any():
                ax1.scatter(pca_2d[real_seeds_mask, 0], pca_2d[real_seeds_mask, 1], 
                           c=self.color_palette['real_malicious_seeds'], 
                           alpha=0.9, s=50, label='Real Seeds', 
                           marker=marker_mapping['real_malicious_seeds'], 
                           edgecolors='black', linewidth=0.8)
            
            # 画各层合成数据
            synthetic_sources = ['synthetic_rewrite', 'synthetic_rewrite_strong', 'synthetic_rewrite_weak']
            synthetic_mask = metadata_df['data_source'].isin(synthetic_sources)
            
            if synthetic_mask.any():
                synthetic_df = metadata_df[synthetic_mask]
                pca_synthetic = pca_2d[synthetic_mask]
                
                for layer in ['core', 'inner', 'outer', 'edge']:
                    layer_mask = synthetic_df['layer'] == layer
                    if layer_mask.any():
                        color = self.color_palette.get(layer, '#888888')
                        marker = marker_mapping.get(layer, 'o')
                        ax1.scatter(pca_synthetic[layer_mask, 0], pca_synthetic[layer_mask, 1], 
                                   c=color, alpha=0.8, s=35, 
                                   label=f'Synthetic {layer.title()}',
                                   marker=marker, edgecolors='white', linewidth=0.5)
            
            ax1.set_xlabel(f'PC1 ({dim_results["pca_2d_variance"][0]:.1%} variance)')
            ax1.set_ylabel(f'PC2 ({dim_results["pca_2d_variance"][1]:.1%} variance)')
            ax1.set_title('Layer Distribution vs Real Malicious Data')
            ax1.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=1, fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # 2. Prompt策略对比
            ax2 = axes[0, 1]
            
            # 种子样本 (参考点)
            if real_seeds_mask.any():
                ax2.scatter(pca_2d[real_seeds_mask, 0], pca_2d[real_seeds_mask, 1], 
                           c=self.color_palette['real_malicious_seeds'], 
                           alpha=0.9, s=50, label='Real Seeds', 
                           marker=marker_mapping['real_malicious_seeds'], 
                           edgecolors='black', linewidth=0.8)
            
            # 各种prompt策略
            for variant in ['rewrite', 'rewrite_strong', 'rewrite_weak']:
                variant_mask = metadata_df['data_source'] == f'synthetic_{variant}'
                if variant_mask.any():
                    color = self.color_palette[f'synthetic_{variant}']
                    marker = marker_mapping[f'synthetic_{variant}']
                    ax2.scatter(pca_2d[variant_mask, 0], pca_2d[variant_mask, 1], 
                               c=color, alpha=0.8, s=35, 
                               label=f'Synthetic {variant.replace("_", " ").title()}',
                               marker=marker, edgecolors='white', linewidth=0.5)
            
            ax2.set_xlabel(f'PC1 ({dim_results["pca_2d_variance"][0]:.1%} variance)')
            ax2.set_ylabel(f'PC2 ({dim_results["pca_2d_variance"][1]:.1%} variance)')
            ax2.set_title('Prompt Strategy Comparison vs Real Data')
            ax2.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=1, fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # 3. 核心层详细对比
            ax3 = axes[1, 0]
            
            # 种子样本
            if real_seeds_mask.any():
                ax3.scatter(pca_2d[real_seeds_mask, 0], pca_2d[real_seeds_mask, 1], 
                           c=self.color_palette['real_malicious_seeds'], 
                           alpha=0.9, s=60, label='Real Seeds', 
                           marker=marker_mapping['real_malicious_seeds'], 
                           edgecolors='black', linewidth=1.0)
            
            # 只显示core层的合成数据
            if synthetic_mask.any():
                synthetic_df = metadata_df[synthetic_mask]
                pca_synthetic = pca_2d[synthetic_mask]
                core_mask = synthetic_df['layer'] == 'core'
                
                if core_mask.any():
                    core_df = synthetic_df[core_mask]
                    pca_core = pca_synthetic[core_mask]
                    
                    for variant in ['rewrite', 'rewrite_strong', 'rewrite_weak']:
                        variant_mask = core_df['prompt_variant'] == variant
                        if variant_mask.any():
                            color = self.color_palette[f'synthetic_{variant}']
                            marker = marker_mapping[f'synthetic_{variant}']
                            ax3.scatter(pca_core[variant_mask, 0], pca_core[variant_mask, 1], 
                                       c=color, alpha=0.8, s=40, 
                                       label=f'Core {variant.replace("_", " ").title()}',
                                       marker=marker, edgecolors='white', linewidth=0.5)
            
            ax3.set_xlabel(f'PC1 ({dim_results["pca_2d_variance"][0]:.1%} variance)')
            ax3.set_ylabel(f'PC2 ({dim_results["pca_2d_variance"][1]:.1%} variance)')
            ax3.set_title('Core Layer Detailed Analysis')
            ax3.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=1, fontsize=9)
            ax3.grid(True, alpha=0.3)
            
            # 4. 数量统计对比
            ax4 = axes[1, 1]
            
            # 统计各类别数量
            category_counts = {}
            category_counts['Real Seeds'] = real_seeds_mask.sum()
            category_counts['Real Background'] = real_bg_mask.sum()
            
            for variant in ['rewrite', 'rewrite_strong', 'rewrite_weak']:
                variant_mask = metadata_df['data_source'] == f'synthetic_{variant}'
                category_counts[f'Synthetic\n{variant.replace("_", " ").title()}'] = variant_mask.sum()
            
            colors = [self.color_palette['real_malicious_seeds'], 
                     self.color_palette['real_malicious_background'],
                     self.color_palette['synthetic_rewrite'],
                     self.color_palette['synthetic_rewrite_strong'], 
                     self.color_palette['synthetic_rewrite_weak']]
            
            bars = ax4.bar(range(len(category_counts)), list(category_counts.values()), 
                          color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax4.set_xticks(range(len(category_counts)))
            ax4.set_xticklabels(list(category_counts.keys()), rotation=45, ha='right', fontsize=8)
            ax4.set_ylabel('Sample Count')
            ax4.set_title('Data Source Sample Distribution')
            ax4.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for bar, count in zip(bars, category_counts.values()):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                        f'{count:,}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            plt.tight_layout()
            
            # 保存图片
            layer_comparison_file = self.static_dir / "layer_comparison_analysis.png"
            plt.savefig(layer_comparison_file, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"分层对比图已保存: {layer_comparison_file}")
            
        except Exception as e:
            self.logger.error(f"创建分层对比图失败: {str(e)}")
            raise
    
    def create_distance_distribution_plots(self, metadata_df: pd.DataFrame, 
                                         dim_results: Dict[str, np.ndarray]) -> None:
        """创建距离分布分析图"""
        try:
            self.logger.info("创建距离分布分析图...")
            
            # 读取距离统计数据
            stats_file = self.unified_dir / "distance_statistics.json"
            with open(stats_file, 'r', encoding='utf-8') as f:
                distance_stats = json.load(f)
            
            if 'layer_analysis' not in distance_stats:
                self.logger.warning("距离统计数据不完整，跳过距离分布图")
                return
            
            layer_analysis = distance_stats['layer_analysis']
            
            # 创建距离分布对比图
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Synthetic Data Layer Distance Distribution Analysis', fontsize=16, fontweight='bold')
            
            layers = ['core', 'inner', 'outer', 'edge']
            colors = [self.color_palette.get(layer, '#888888') for layer in layers]
            
            # 1. 平均距离对比
            ax1 = axes[0, 0]
            mean_distances = [layer_analysis[layer]['mean_distance'] for layer in layers]
            bars1 = ax1.bar(layers, mean_distances, color=colors, alpha=0.7)
            ax1.set_ylabel('Average Cosine Distance')
            ax1.set_title('Average Distance to Centroid by Layer')
            ax1.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, distance in zip(bars1, mean_distances):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{distance:.3f}', ha='center', va='bottom')
            
            # 2. 距离标准差对比
            ax2 = axes[0, 1]
            std_distances = [layer_analysis[layer]['std_distance'] for layer in layers]
            bars2 = ax2.bar(layers, std_distances, color=colors, alpha=0.7)
            ax2.set_ylabel('Distance Standard Deviation')
            ax2.set_title('Standard Deviation of Distance Distribution by Layer')
            ax2.grid(True, alpha=0.3)
            
            for bar, std in zip(bars2, std_distances):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{std:.3f}', ha='center', va='bottom')
            
            # 3. 距离范围对比
            ax3 = axes[1, 0]
            min_distances = [layer_analysis[layer]['min_distance'] for layer in layers]
            max_distances = [layer_analysis[layer]['max_distance'] for layer in layers]
            
            x_pos = np.arange(len(layers))
            ax3.bar(x_pos - 0.2, min_distances, 0.4, label='Min Distance', alpha=0.7)
            ax3.bar(x_pos + 0.2, max_distances, 0.4, label='Max Distance', alpha=0.7)
            ax3.set_xlabel('Layer')
            ax3.set_ylabel('Cosine Distance')
            ax3.set_title('Distance Range Comparison by Layer')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(layers)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. 样本数量对比
            ax4 = axes[1, 1]
            sample_counts = [layer_analysis[layer]['sample_count'] for layer in layers]
            bars4 = ax4.bar(layers, sample_counts, color=colors, alpha=0.7)
            ax4.set_ylabel('Sample Count')
            ax4.set_title('Sample Count Distribution by Layer')
            ax4.grid(True, alpha=0.3)
            
            for bar, count in zip(bars4, sample_counts):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f'{count}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            distance_file = self.dist_dir / "layer_distance_analysis.png"
            plt.savefig(distance_file, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"距离分布图已保存: {distance_file}")
            
        except Exception as e:
            self.logger.error(f"创建距离分布图失败: {str(e)}")
    
    def create_clustering_quality_plots(self, cluster_results: Dict[str, Any]) -> None:
        """创建聚类质量分析图"""
        try:
            self.logger.info("创建聚类质量分析图...")
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Clustering Analysis Quality Assessment', fontsize=16, fontweight='bold')
            
            # 1. K-means肘部法则
            ax1 = axes[0]
            k_values = list(cluster_results['inertias'].keys())
            inertias = list(cluster_results['inertias'].values())
            
            ax1.plot(k_values, inertias, 'bo-', markersize=8, linewidth=2)
            ax1.set_xlabel('Number of Clusters (k)')
            ax1.set_ylabel('Inertia')
            ax1.set_title('K-means Elbow Method')
            ax1.grid(True, alpha=0.3)
            
            # 标记最佳k值
            best_k = cluster_results['best_k']
            best_inertia = cluster_results['inertias'][best_k]
            ax1.plot(best_k, best_inertia, 'ro', markersize=12, label=f'Optimal k={best_k}')
            ax1.legend()
            
            # 2. 轮廓系数
            ax2 = axes[1]
            silhouette_scores = list(cluster_results['silhouette_scores'].values())
            
            ax2.plot(k_values, silhouette_scores, 'go-', markersize=8, linewidth=2)
            ax2.set_xlabel('Number of Clusters (k)')
            ax2.set_ylabel('Silhouette Score')
            ax2.set_title('K-means Silhouette Score')
            ax2.grid(True, alpha=0.3)
            
            # 标记最佳轮廓系数
            best_silhouette = max(silhouette_scores)
            ax2.plot(best_k, best_silhouette, 'ro', markersize=12, 
                    label=f'Best Silhouette Score={best_silhouette:.3f}')
            ax2.legend()
            
            # 3. DBSCAN结果
            ax3 = axes[2]
            dbscan_data = [
                cluster_results['dbscan_n_clusters'],
                cluster_results['dbscan_n_noise'],
                len(cluster_results['dbscan_labels']) - cluster_results['dbscan_n_clusters'] - cluster_results['dbscan_n_noise']
            ]
            labels = ['Valid Clusters', 'Noise Points', 'Clustered Points']
            colors = ['#2E8B57', '#DC143C', '#4682B4']
            
            ax3.pie(dbscan_data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax3.set_title(f'DBSCAN Clustering Results\n({cluster_results["dbscan_n_clusters"]} clusters)')
            
            plt.tight_layout()
            
            cluster_quality_file = self.cluster_dir / "clustering_quality_analysis.png"
            plt.savefig(cluster_quality_file, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"聚类质量图已保存: {cluster_quality_file}")
            
        except Exception as e:
            self.logger.error(f"创建聚类质量图失败: {str(e)}")
    
    def create_interactive_visualizations(self, metadata_df: pd.DataFrame, dim_results: Dict[str, np.ndarray],
                                        cluster_results: Dict[str, Any]) -> None:
        """创建交互式可视化"""
        try:
            self.logger.info("创建交互式可视化...")
            
            # 1. 3D PCA交互图
            self.create_3d_pca_interactive(metadata_df, dim_results)
            
            # 2. 分层对比交互图
            self.create_layer_comparison_interactive(metadata_df, dim_results)
            
            # 3. 数据源对比交互仪表板
            self.create_comparison_dashboard(metadata_df, dim_results, cluster_results)
            
            # 4. t-SNE交互可视化
            self.create_tsne_interactive(metadata_df, dim_results)
            
        except Exception as e:
            self.logger.error(f"创建交互式可视化失败: {str(e)}")
    
    def create_3d_pca_interactive(self, metadata_df: pd.DataFrame, dim_results: Dict[str, np.ndarray]) -> None:
        """创建3D PCA交互图"""
        try:
            # 创建3D散点图
            fig = go.Figure()
            
            # 按数据源分组
            for source in metadata_df['data_source'].unique():
                mask = metadata_df['data_source'] == source
                color = self.color_palette.get(source, '#888888')
                
                fig.add_trace(go.Scatter3d(
                    x=dim_results['pca_3d'][mask, 0],
                    y=dim_results['pca_3d'][mask, 1], 
                    z=dim_results['pca_3d'][mask, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=color,
                        opacity=0.6
                    ),
                    name=source,
                    text=metadata_df[mask]['unique_id'],
                    hovertemplate="<b>%{text}</b><br>" +
                                "PC1: %{x:.3f}<br>" +
                                "PC2: %{y:.3f}<br>" +
                                "PC3: %{z:.3f}<br>" +
                                "<extra></extra>"
                ))
            
            # 更新布局
            fig.update_layout(
                title='Batch 5 Unified Embedding Space - 3D PCA Visualization',
                scene=dict(
                    xaxis_title=f'PC1 ({dim_results["pca_3d_variance"][0]:.1%})',
                    yaxis_title=f'PC2 ({dim_results["pca_3d_variance"][1]:.1%})',
                    zaxis_title=f'PC3 ({dim_results["pca_3d_variance"][2]:.1%})'
                ),
                width=1000,
                height=800
            )
            
            # 保存交互图
            interactive_3d_file = self.interactive_dir / "batch5_3d_pca_interactive.html"
            pyo.plot(fig, filename=str(interactive_3d_file), auto_open=False)
            self.logger.info(f"3D PCA交互图已保存: {interactive_3d_file}")
            
        except Exception as e:
            self.logger.error(f"创建3D PCA交互图失败: {str(e)}")
    
    def create_layer_comparison_interactive(self, metadata_df: pd.DataFrame, dim_results: Dict[str, np.ndarray]) -> None:
        """创建交互式分层对比图"""
        try:
            # 定义统一的形状映射
            shape_mapping = {
                'real_malicious_background': 'circle',
                'real_malicious_seeds': 'star',
                'synthetic_rewrite': 'circle',
                'synthetic_rewrite_strong': 'square', 
                'synthetic_rewrite_weak': 'diamond',
                'core': 'circle',
                'inner': 'square',
                'outer': 'diamond',
                'edge': 'triangle-up'
            }
            
            # 创建Dash应用风格的交互图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Layer Distribution vs Real Malicious Data',
                    'Prompt Strategy Comparison', 
                    'Seeds vs Core Layer Synthetic',
                    'Sample Count Distribution'
                ),
                specs=[[{"type": "scatter"}, {"type": "scatter"}],
                       [{"type": "scatter"}, {"type": "scatter"}]]
            )
            
            pca_2d = dim_results['pca_2d']
            
            # 准备数据掩码
            real_bg_mask = metadata_df['data_source'] == 'real_malicious_background'
            real_seeds_mask = metadata_df['data_source'] == 'real_malicious_seeds'
            synthetic_sources = ['synthetic_rewrite', 'synthetic_rewrite_strong', 'synthetic_rewrite_weak']
            synthetic_mask = metadata_df['data_source'].isin(synthetic_sources)
            
            # 1. 分层分布对比 (左上) - 显示layer分布
            
            # 背景数据 (透明显示)
            if real_bg_mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=pca_2d[real_bg_mask, 0],
                        y=pca_2d[real_bg_mask, 1],
                        mode='markers',
                        marker=dict(
                            color=self.color_palette['real_malicious_background'],
                            size=4,
                            opacity=0.2,
                            symbol=shape_mapping['real_malicious_background']
                        ),
                        name='Real Background',
                        text=metadata_df.loc[real_bg_mask, 'unique_id'],
                        hovertemplate="<b>Real Background</b><br>ID: %{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>"
                    ),
                    row=1, col=1
                )
            
            # 种子样本
            if real_seeds_mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=pca_2d[real_seeds_mask, 0],
                        y=pca_2d[real_seeds_mask, 1],
                        mode='markers',
                        marker=dict(
                            color=self.color_palette['real_malicious_seeds'],
                            size=8,
                            opacity=0.9,
                            symbol=shape_mapping['real_malicious_seeds'],
                            line=dict(width=1, color='black')
                        ),
                        name='Real Seeds',
                        text=metadata_df.loc[real_seeds_mask, 'unique_id'],
                        hovertemplate="<b>Real Seeds</b><br>ID: %{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>"
                    ),
                    row=1, col=1
                )
            
            # 各层合成数据
            if synthetic_mask.any():
                synthetic_df = metadata_df[synthetic_mask].copy()
                for layer in ['core', 'inner', 'outer', 'edge']:
                    layer_mask = synthetic_df['layer'] == layer
                    if layer_mask.any():
                        layer_indices = synthetic_df[layer_mask].index
                        fig.add_trace(
                            go.Scatter(
                                x=pca_2d[layer_indices, 0],
                                y=pca_2d[layer_indices, 1],
                                mode='markers',
                                marker=dict(
                                    color=self.color_palette.get(layer, '#888888'),
                                    size=6,
                                    opacity=0.8,
                                    symbol=shape_mapping.get(layer, 'circle'),
                                    line=dict(width=0.5, color='white')
                                ),
                                name=f'Synthetic {layer.title()}',
                                text=synthetic_df.loc[layer_mask, 'unique_id'],
                                hovertemplate=f"<b>Synthetic {layer.title()}</b><br>ID: %{{text}}<br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<extra></extra>"
                            ),
                            row=1, col=1
                        )
            
            # 2. Prompt策略对比 (右上) - 显示prompt变体
            
            # 种子样本 (参考点)
            if real_seeds_mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=pca_2d[real_seeds_mask, 0],
                        y=pca_2d[real_seeds_mask, 1],
                        mode='markers',
                        marker=dict(
                            color=self.color_palette['real_malicious_seeds'],
                            size=8,
                            opacity=0.9,
                            symbol=shape_mapping['real_malicious_seeds'],
                            line=dict(width=1, color='black')
                        ),
                        name='Real Seeds',
                        text=metadata_df.loc[real_seeds_mask, 'unique_id'],
                        hovertemplate="<b>Real Seeds</b><br>ID: %{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>"
                    ),
                    row=1, col=2
                )
            
            # 各种prompt策略
            for variant in ['rewrite', 'rewrite_strong', 'rewrite_weak']:
                variant_mask = metadata_df['data_source'] == f'synthetic_{variant}'
                if variant_mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=pca_2d[variant_mask, 0],
                            y=pca_2d[variant_mask, 1],
                            mode='markers',
                            marker=dict(
                                color=self.color_palette[f'synthetic_{variant}'],
                                size=6,
                                opacity=0.8,
                                symbol=shape_mapping[f'synthetic_{variant}'],
                                line=dict(width=0.5, color='white')
                            ),
                            name=f'Synthetic {variant.replace("_", " ").title()}',
                            text=metadata_df.loc[variant_mask, 'unique_id'],
                            hovertemplate=f"<b>Synthetic {variant.replace('_', ' ').title()}</b><br>ID: %{{text}}<br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<extra></extra>"
                        ),
                        row=1, col=2
                    )
            
            # 3. 种子vs核心层详细对比 (左下) - 专注core层
            
            # 种子样本
            if real_seeds_mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=pca_2d[real_seeds_mask, 0],
                        y=pca_2d[real_seeds_mask, 1],
                        mode='markers',
                        marker=dict(
                            color=self.color_palette['real_malicious_seeds'],
                            size=10,
                            opacity=0.9,
                            symbol=shape_mapping['real_malicious_seeds'],
                            line=dict(width=2, color='black')
                        ),
                        name='Real Seeds',
                        text=metadata_df.loc[real_seeds_mask, 'unique_id'],
                        hovertemplate="<b>Real Seeds</b><br>ID: %{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>"
                    ),
                    row=2, col=1
                )
            
            # 只显示core层的合成数据
            if synthetic_mask.any():
                core_mask = (metadata_df['data_source'].isin(synthetic_sources)) & (metadata_df['layer'] == 'core')
                if core_mask.any():
                    core_df = metadata_df[core_mask]
                    for variant in ['rewrite', 'rewrite_strong', 'rewrite_weak']:
                        variant_core_mask = core_df['prompt_variant'] == variant
                        if variant_core_mask.any():
                            variant_indices = core_df[variant_core_mask].index
                            fig.add_trace(
                                go.Scatter(
                                    x=pca_2d[variant_indices, 0],
                                    y=pca_2d[variant_indices, 1],
                                    mode='markers',
                                    marker=dict(
                                        color=self.color_palette[f'synthetic_{variant}'],
                                        size=7,
                                        opacity=0.8,
                                        symbol=shape_mapping[f'synthetic_{variant}'],
                                        line=dict(width=0.5, color='white')
                                    ),
                                    name=f'Core {variant.replace("_", " ").title()}',
                                    text=core_df.loc[variant_core_mask, 'unique_id'],
                                    hovertemplate=f"<b>Core {variant.replace('_', ' ').title()}</b><br>ID: %{{text}}<br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<extra></extra>"
                                ),
                                row=2, col=1
                            )
            
            # 4. 数量统计 (右下) - 使用bar图
            categories = ['Real Seeds', 'Real Background', 'Synthetic\nRewrite', 'Synthetic\nStrong', 'Synthetic\nWeak']
            counts = [
                real_seeds_mask.sum(),
                real_bg_mask.sum(),
                (metadata_df['data_source'] == 'synthetic_rewrite').sum(),
                (metadata_df['data_source'] == 'synthetic_rewrite_strong').sum(),
                (metadata_df['data_source'] == 'synthetic_rewrite_weak').sum()
            ]
            colors = [
                self.color_palette['real_malicious_seeds'],
                self.color_palette['real_malicious_background'],
                self.color_palette['synthetic_rewrite'],
                self.color_palette['synthetic_rewrite_strong'],
                self.color_palette['synthetic_rewrite_weak']
            ]
            
            fig.add_trace(
                go.Bar(
                    x=categories,
                    y=counts,
                    marker=dict(color=colors, opacity=0.8, line=dict(width=1, color='black')),
                    name='Sample Counts',
                    showlegend=False,
                    text=[f'{count:,}' for count in counts],
                    textposition='auto',
                    textfont=dict(color='white', size=10, family='Arial Black')
                ),
                row=2, col=2
            )
            
            # 更新布局
            fig.update_layout(
                title='Interactive Layer Comparison Analysis - Synthetic vs Real Malicious',
                height=800,
                showlegend=True,
                legend=dict(x=1.02, y=1, xanchor='left', yanchor='top')
            )
            
            # 更新各子图轴标签
            fig.update_xaxes(title_text=f'PC1 ({dim_results["pca_2d_variance"][0]:.1%})', row=1, col=1)
            fig.update_yaxes(title_text=f'PC2 ({dim_results["pca_2d_variance"][1]:.1%})', row=1, col=1)
            fig.update_xaxes(title_text=f'PC1 ({dim_results["pca_2d_variance"][0]:.1%})', row=1, col=2)
            fig.update_yaxes(title_text=f'PC2 ({dim_results["pca_2d_variance"][1]:.1%})', row=1, col=2)
            fig.update_xaxes(title_text=f'PC1 ({dim_results["pca_2d_variance"][0]:.1%})', row=2, col=1)
            fig.update_yaxes(title_text=f'PC2 ({dim_results["pca_2d_variance"][1]:.1%})', row=2, col=1)
            fig.update_xaxes(title_text='Data Source', row=2, col=2)
            fig.update_yaxes(title_text='Sample Count', row=2, col=2)
            
            # 保存
            layer_interactive_file = self.interactive_dir / "batch5_layer_comparison_dashboard.html"
            pyo.plot(fig, filename=str(layer_interactive_file), auto_open=False)
            self.logger.info(f"交互式分层对比图已保存: {layer_interactive_file}")
            
        except Exception as e:
            self.logger.error(f"创建交互式分层对比图失败: {str(e)}")
    
    def create_comparison_dashboard(self, metadata_df: pd.DataFrame, dim_results: Dict[str, np.ndarray],
                                  cluster_results: Dict[str, Any]) -> None:
        """创建对比分析仪表板"""
        try:
            # 创建子图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('By Data Source', 'By Synthetic Data Layer', 'By Prompt Variant', 'K-means Clustering'),
                specs=[[{"type": "scatter"}, {"type": "scatter"}],
                       [{"type": "scatter"}, {"type": "scatter"}]]
            )
            
            # 1. 按数据源
            for source in metadata_df['data_source'].unique():
                mask = metadata_df['data_source'] == source
                color = self.color_palette.get(source, '#888888')
                
                fig.add_trace(
                    go.Scatter(
                        x=dim_results['pca_2d'][mask, 0],
                        y=dim_results['pca_2d'][mask, 1],
                        mode='markers',
                        marker=dict(color=color, size=5, opacity=0.6),
                        name=source,
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            # 2. 按合成数据层（仅合成数据）
            synthetic_mask = metadata_df['data_source'] == 'synthetic'
            if synthetic_mask.any():
                synthetic_df = metadata_df[synthetic_mask]
                pca_synthetic = dim_results['pca_2d'][synthetic_mask]
                
                for layer in synthetic_df['layer'].unique():
                    layer_mask = synthetic_df['layer'] == layer
                    color = self.color_palette.get(layer, '#888888')
                    
                    fig.add_trace(
                        go.Scatter(
                            x=pca_synthetic[layer_mask, 0],
                            y=pca_synthetic[layer_mask, 1],
                            mode='markers',
                            marker=dict(color=color, size=6, opacity=0.7),
                            name=f'{layer}层',
                            showlegend=False
                        ),
                        row=1, col=2
                    )
            
            # 3. 按Prompt变体（仅合成数据）
            if synthetic_mask.any():
                for variant in synthetic_df['prompt_variant'].unique():
                    if pd.notna(variant):
                        variant_mask = synthetic_df['prompt_variant'] == variant
                        color = self.color_palette.get(variant, '#888888')
                        
                        fig.add_trace(
                            go.Scatter(
                                x=pca_synthetic[variant_mask, 0],
                                y=pca_synthetic[variant_mask, 1],
                                mode='markers',
                                marker=dict(color=color, size=6, opacity=0.7),
                                name=variant,
                                showlegend=False
                            ),
                            row=2, col=1
                        )
            
            # 4. K-means聚类
            sample_indices = cluster_results['sample_indices']
            sample_pca = dim_results['pca_2d'][sample_indices]
            kmeans_labels = cluster_results['kmeans_labels']
            
            fig.add_trace(
                go.Scatter(
                    x=sample_pca[:, 0],
                    y=sample_pca[:, 1],
                    mode='markers',
                    marker=dict(
                        color=kmeans_labels,
                        colorscale='tab10',
                        size=5,
                        opacity=0.6
                    ),
                    name='聚类',
                    showlegend=False
                ),
                row=2, col=2
            )
            
            # 更新布局
            fig.update_layout(
                title_text="Batch 5 Unified Embedding Space - Multi-dimensional Comparative Analysis",
                height=800,
                width=1200
            )
            
            # 保存仪表板
            dashboard_file = self.interactive_dir / "batch5_comparison_dashboard.html"
            pyo.plot(fig, filename=str(dashboard_file), auto_open=False)
            self.logger.info(f"对比分析仪表板已保存: {dashboard_file}")
            
        except Exception as e:
            self.logger.error(f"创建对比分析仪表板失败: {str(e)}")
    
    def create_tsne_interactive(self, metadata_df: pd.DataFrame, dim_results: Dict[str, np.ndarray]) -> None:
        """创建t-SNE交互可视化"""
        try:
            # 获取有效的t-SNE数据
            valid_mask = ~np.isnan(dim_results['tsne_2d'][:, 0])
            if not valid_mask.any():
                self.logger.warning("没有有效的t-SNE数据，跳过t-SNE可视化")
                return
            
            valid_df = metadata_df[valid_mask].copy()
            tsne_data = dim_results['tsne_2d'][valid_mask]
            
            # 创建t-SNE散点图
            fig = px.scatter(
                x=tsne_data[:, 0],
                y=tsne_data[:, 1],
                color=valid_df['data_source'],
                title='Batch 5 Unified Embedding Space - t-SNE Visualization',
                labels={'x': 't-SNE 1', 'y': 't-SNE 2', 'color': 'Data Source'},
                hover_data={'unique_id': valid_df['unique_id']},
                color_discrete_map=self.color_palette
            )
            
            fig.update_traces(marker_size=5, marker_opacity=0.7)
            fig.update_layout(width=1000, height=800)
            
            # 保存t-SNE图
            tsne_file = self.interactive_dir / "batch5_tsne_interactive.html"
            pyo.plot(fig, filename=str(tsne_file), auto_open=False)
            self.logger.info(f"t-SNE交互图已保存: {tsne_file}")
            
        except Exception as e:
            self.logger.error(f"创建t-SNE交互图失败: {str(e)}")
    
    def generate_phase2_summary(self, metadata_df: pd.DataFrame, dim_results: Dict[str, np.ndarray],
                              cluster_results: Dict[str, Any]) -> None:
        """生成Phase 2统计摘要"""
        try:
            self.logger.info("生成Phase 2统计摘要...")
            
            summary = {
                'phase2_info': {
                    'date': '2025-07-30',
                    'total_samples_visualized': len(metadata_df),
                    'embedding_dimension': dim_results['pca_2d'].shape[1] if 'pca_2d' in dim_results else 384,
                    'random_state': self.random_state
                },
                'dimensionality_reduction': {
                    'pca_2d_variance_explained': float(dim_results['pca_2d_variance'].sum()) if 'pca_2d_variance' in dim_results else None,
                    'pca_3d_variance_explained': float(dim_results['pca_3d_variance'].sum()) if 'pca_3d_variance' in dim_results else None,
                    'tsne_samples': len(dim_results.get('tsne_sample_indices', []))
                },
                'clustering_analysis': {
                    'kmeans_best_k': cluster_results.get('best_k'),
                    'kmeans_best_silhouette': max(cluster_results.get('silhouette_scores', {}).values()) if cluster_results.get('silhouette_scores') else None,
                    'dbscan_clusters': cluster_results.get('dbscan_n_clusters'),
                    'dbscan_noise_points': cluster_results.get('dbscan_n_noise'),
                    'dbscan_silhouette': cluster_results.get('dbscan_silhouette')
                },
                'data_distribution': {
                    'by_source': metadata_df['data_source'].value_counts().to_dict(),
                    'synthetic_by_layer': metadata_df[metadata_df['data_source'].str.startswith('synthetic_')]['layer'].value_counts().to_dict() if metadata_df['data_source'].str.startswith('synthetic_').any() else {},
                    'synthetic_by_prompt': metadata_df[metadata_df['data_source'].str.startswith('synthetic_')]['prompt_variant'].value_counts().to_dict() if metadata_df['data_source'].str.startswith('synthetic_').any() else {}
                },
                'visualization_files': {
                    'static_plots': {
                        'overview': 'static_plots/embedding_overview_analysis.png',
                        'distance_analysis': 'distribution_analysis/layer_distance_analysis.png',
                        'clustering_quality': 'cluster_analysis/clustering_quality_analysis.png'
                    },
                    'interactive_plots': {
                        '3d_pca': 'interactive_plots/batch5_3d_pca_interactive.html',
                        'dashboard': 'interactive_plots/batch5_comparison_dashboard.html',
                        'tsne': 'interactive_plots/batch5_tsne_interactive.html'
                    }
                }
            }
            
            # 保存摘要
            summary_file = self.analysis_dir / "phase2_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Phase 2统计摘要已保存: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"生成Phase 2统计摘要失败: {str(e)}")
            raise
    
    def run_phase2_complete(self) -> bool:
        """执行完整的Phase 2流程"""
        try:
            self.logger.info("="*50)
            self.logger.info("开始执行批次5 Phase 2: 增强可视化分析")
            self.logger.info("="*50)
            
            start_time = time.time()
            
            # Step 1: 加载统一embedding数据
            self.logger.info("Step 1: 加载统一embedding数据")
            metadata_df, embeddings, centroids = self.load_unified_data()
            
            # Step 2: 执行降维分析
            self.logger.info("Step 2: 执行降维分析")
            dim_results = self.perform_dimensionality_reduction(embeddings)
            
            # Step 3: 执行聚类分析
            self.logger.info("Step 3: 执行聚类分析")
            cluster_results = self.perform_clustering_analysis(embeddings, metadata_df)
            
            # Step 4: 创建静态可视化
            self.logger.info("Step 4: 创建静态可视化")
            self.create_static_visualizations(metadata_df, dim_results, cluster_results, embeddings)
            
            # Step 5: 创建交互式可视化
            self.logger.info("Step 5: 创建交互式可视化")
            self.create_interactive_visualizations(metadata_df, dim_results, cluster_results)
            
            # Step 6: 生成统计摘要
            self.logger.info("Step 6: 生成统计摘要")
            self.generate_phase2_summary(metadata_df, dim_results, cluster_results)
            
            elapsed_time = time.time() - start_time
            
            self.logger.info("="*50)
            self.logger.info("批次5 Phase 2 执行完成!")
            self.logger.info(f"总耗时: {elapsed_time/60:.1f} 分钟")
            self.logger.info(f"可视化样本数: {len(metadata_df):,}")
            self.logger.info(f"最佳聚类数: {cluster_results.get('best_k', 'N/A')}")
            self.logger.info("可视化文件已保存到: data/batch5/visualizations/")
            self.logger.info("="*50)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Phase 2执行失败: {str(e)}")
            return False

def main():
    """主函数"""
    # 设置随机种子
    random.seed(2025)
    np.random.seed(2025)
    
    # 设置日志
    logger = setup_logger("batch5_phase2", log_level=logging.INFO)
    
    try:
        # 创建可视化器
        visualizer = Batch5Phase2EnhancedVisualization(logger)
        
        # 执行Phase 2
        success = visualizer.run_phase2_complete()
        
        if success:
            logger.info("批次5 Phase 2 successfully completed!")
            return 0
        else:
            logger.error("批次5 Phase 2 failed!")
            return 1
            
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())