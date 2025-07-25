#!/usr/bin/env python3
"""
批次4实验 - Phase 4: 统一embedding和可视化

对所有20个数据集配置进行统一的embedding分析和可视化：
1. 为所有数据集生成embedding向量
2. 分析不同数据组和比例的embedding分布
3. 可视化合成数据与真实数据的对比
4. 生成综合分析报告

作者: Claude
创建时间: 2025-07-24
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# 项目路径设置
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.cyberdata.utils.logger_config import setup_logger
from src.cyberdata.utils.data_loader import load_csv_data

class Batch4Phase4EmbeddingAnalyzer:
    """批次4 Phase 4: 统一embedding和可视化分析器"""
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logger("batch4_phase4")
        self.project_root = PROJECT_ROOT
        self.batch4_dir = self.project_root / "data" / "batch4_fresh"
        
        # 创建输出目录
        self.embeddings_dir = self.batch4_dir / "phase4_embeddings"
        self.visualizations_dir = self.batch4_dir / "phase4_visualizations"
        self.analysis_dir = self.batch4_dir / "phase4_analysis"
        
        for dir_path in [self.embeddings_dir, self.visualizations_dir, self.analysis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 加载embedding模型
        self.model_name = "all-MiniLM-L6-v2"
        self.logger.info(f"初始化embedding模型: {self.model_name}")
        self.embedding_model = SentenceTransformer(self.model_name)
        
        # 实验参数
        self.data_groups = ['real_only', 'core', 'inner', 'outer', 'edge']
        self.malicious_ratios = [0.05, 0.10, 0.15, 0.20]
        
        # 设置matplotlib字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 随机种子
        self.random_state = 2025
        
        self.logger.info("Phase 4初始化完成")
    
    def load_all_datasets(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """加载所有数据集配置"""
        try:
            self.logger.info("加载所有数据集配置...")
            
            datasets = {}
            datasets_dir = self.batch4_dir / "datasets"
            
            for group in self.data_groups:
                datasets[group] = {}
                for ratio in self.malicious_ratios:
                    config_name = f"{group}_{int(ratio*100)}pct"
                    config_dir = datasets_dir / config_name
                    
                    if config_dir.exists():
                        train_file = config_dir / "train.csv"
                        test_file = config_dir / "test.csv"
                        
                        if train_file.exists() and test_file.exists():
                            datasets[group][f"{int(ratio*100)}pct"] = {
                                'train': load_csv_data(train_file, logger=self.logger),
                                'test': load_csv_data(test_file, logger=self.logger),
                                'config_name': config_name
                            }
                            
                            train_mal = len(datasets[group][f"{int(ratio*100)}pct"]['train'][
                                datasets[group][f"{int(ratio*100)}pct"]['train']['label'] == 1])
                            train_ben = len(datasets[group][f"{int(ratio*100)}pct"]['train'][
                                datasets[group][f"{int(ratio*100)}pct"]['train']['label'] == 0])
                            
                            self.logger.info(f"✅ {config_name}: 训练集{train_mal}恶意+{train_ben}良性")
                        else:
                            self.logger.warning(f"❌ {config_name}: 数据文件缺失")
                    else:
                        self.logger.warning(f"❌ {config_name}: 配置目录不存在")
            
            total_configs = sum(len(group_data) for group_data in datasets.values())
            self.logger.info(f"成功加载 {total_configs} 个数据集配置")
            
            return datasets
            
        except Exception as e:
            self.logger.error(f"加载数据集失败: {str(e)}")
            raise
    
    def generate_embeddings_for_dataset(self, dataset: Dict[str, pd.DataFrame], 
                                      config_name: str) -> Dict[str, np.ndarray]:
        """为单个数据集生成embedding"""
        try:
            self.logger.info(f"生成{config_name}的embedding...")
            
            embeddings = {}
            
            for split_name, df in dataset.items():
                if split_name == 'config_name':
                    continue
                    
                # 获取文本数据
                texts = df['text'].tolist()
                
                # 生成embedding
                self.logger.info(f"  - {split_name}: {len(texts)} 个样本")
                batch_embeddings = self.embedding_model.encode(
                    texts, 
                    batch_size=64, 
                    show_progress_bar=False
                )
                
                embeddings[split_name] = batch_embeddings
                
                # 保存embedding
                embedding_file = self.embeddings_dir / f"{config_name}_{split_name}_embeddings.npy"
                np.save(embedding_file, batch_embeddings)
            
            self.logger.info(f"✅ {config_name}: embedding生成完成")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"生成{config_name}的embedding失败: {str(e)}")
            raise
    
    def analyze_embedding_distribution(self, embeddings: Dict[str, np.ndarray], 
                                     dataset: Dict[str, pd.DataFrame], 
                                     config_name: str) -> Dict[str, Any]:
        """分析embedding分布特征"""
        try:
            analysis = {
                'config_name': config_name,
                'embedding_stats': {},
                'similarity_analysis': {},
                'centroid_analysis': {}
            }
            
            for split_name, embs in embeddings.items():
                if split_name == 'config_name':
                    continue
                    
                df = dataset[split_name]
                
                # 基础统计
                stats = {
                    'dimension': embs.shape[1],
                    'sample_count': embs.shape[0],
                    'mean_norm': float(np.mean(np.linalg.norm(embs, axis=1))),
                    'std_norm': float(np.std(np.linalg.norm(embs, axis=1))),
                    'mean_vector': embs.mean(axis=0).tolist(),
                    'std_vector': embs.std(axis=0).tolist()
                }
                
                # 分类别分析
                malicious_mask = df['label'] == 1
                benign_mask = df['label'] == 0
                
                if malicious_mask.sum() > 0:
                    mal_embs = embs[malicious_mask]
                    stats['malicious'] = {
                        'count': int(malicious_mask.sum()),
                        'mean_norm': float(np.mean(np.linalg.norm(mal_embs, axis=1))),
                        'centroid': mal_embs.mean(axis=0).tolist()
                    }
                
                if benign_mask.sum() > 0:
                    ben_embs = embs[benign_mask]
                    stats['benign'] = {
                        'count': int(benign_mask.sum()),
                        'mean_norm': float(np.mean(np.linalg.norm(ben_embs, axis=1))),
                        'centroid': ben_embs.mean(axis=0).tolist()
                    }
                
                # 计算类别间余弦相似度
                if malicious_mask.sum() > 0 and benign_mask.sum() > 0:
                    mal_centroid = mal_embs.mean(axis=0).reshape(1, -1)
                    ben_centroid = ben_embs.mean(axis=0).reshape(1, -1)
                    
                    inter_class_similarity = float(cosine_similarity(mal_centroid, ben_centroid)[0, 0])
                    stats['inter_class_similarity'] = inter_class_similarity
                
                analysis['embedding_stats'][split_name] = stats
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"分析{config_name}的embedding分布失败: {str(e)}")
            raise
    
    def create_embedding_visualization(self, embeddings: Dict[str, np.ndarray], 
                                     dataset: Dict[str, pd.DataFrame], 
                                     config_name: str) -> None:
        """创建embedding可视化"""
        try:
            self.logger.info(f"创建{config_name}的可视化...")
            
            # 只可视化训练集
            if 'train' not in embeddings:
                return
                
            train_embs = embeddings['train']
            train_df = dataset['train']
            
            # 采样以提高可视化速度
            max_samples = 2000
            if len(train_embs) > max_samples:
                sample_indices = np.random.choice(len(train_embs), max_samples, replace=False)
                train_embs = train_embs[sample_indices]
                train_df = train_df.iloc[sample_indices]
            
            # PCA降维
            pca = PCA(n_components=2, random_state=self.random_state)
            pca_embs = pca.fit_transform(train_embs)
            
            # t-SNE降维
            tsne = TSNE(n_components=2, random_state=self.random_state, perplexity=30)
            tsne_embs = tsne.fit_transform(train_embs)
            
            # 创建可视化
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            labels = train_df['label'].values
            colors = ['blue' if label == 0 else 'red' for label in labels]
            label_names = ['Benign' if label == 0 else 'Malicious' for label in labels]
            
            # PCA可视化
            for label in [0, 1]:
                mask = labels == label
                name = 'Benign' if label == 0 else 'Malicious'
                color = 'blue' if label == 0 else 'red'
                axes[0].scatter(pca_embs[mask, 0], pca_embs[mask, 1], 
                              c=color, label=name, alpha=0.6, s=20)
            
            axes[0].set_title(f'{config_name} - PCA Visualization')
            axes[0].set_xlabel(f'PC1 (explained variance: {pca.explained_variance_ratio_[0]:.3f})')
            axes[0].set_ylabel(f'PC2 (explained variance: {pca.explained_variance_ratio_[1]:.3f})')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # t-SNE可视化
            for label in [0, 1]:
                mask = labels == label
                name = 'Benign' if label == 0 else 'Malicious'
                color = 'blue' if label == 0 else 'red'
                axes[1].scatter(tsne_embs[mask, 0], tsne_embs[mask, 1], 
                              c=color, label=name, alpha=0.6, s=20)
            
            axes[1].set_title(f'{config_name} - t-SNE Visualization')
            axes[1].set_xlabel('t-SNE 1')
            axes[1].set_ylabel('t-SNE 2')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图片
            vis_file = self.visualizations_dir / f"{config_name}_embedding_visualization.png"
            plt.savefig(vis_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ {config_name}: 可视化已保存到 {vis_file}")
            
        except Exception as e:
            self.logger.error(f"创建{config_name}的可视化失败: {str(e)}")
            # 不抛出异常，继续处理其他配置
    
    def create_comparative_analysis(self, all_analyses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """创建对比分析"""
        try:
            self.logger.info("创建对比分析...")
            
            comparative_analysis = {
                'summary': {
                    'total_configs': len(all_analyses),
                    'data_groups': self.data_groups,
                    'malicious_ratios': [f"{int(r*100)}%" for r in self.malicious_ratios]
                },
                'group_comparison': {},
                'ratio_comparison': {},
                'similarity_matrix': {},
                'statistical_summary': {}
            }
            
            # 按组比较
            for group in self.data_groups:
                group_stats = []
                for ratio in self.malicious_ratios:
                    config_name = f"{group}_{int(ratio*100)}pct"
                    if config_name in all_analyses:
                        train_stats = all_analyses[config_name]['embedding_stats'].get('train', {})
                        if 'inter_class_similarity' in train_stats:
                            group_stats.append({
                                'ratio': f"{int(ratio*100)}%",
                                'inter_class_similarity': train_stats['inter_class_similarity'],
                                'malicious_count': train_stats.get('malicious', {}).get('count', 0),
                                'benign_count': train_stats.get('benign', {}).get('count', 0)
                            })
                
                if group_stats:
                    comparative_analysis['group_comparison'][group] = {
                        'configs': group_stats,
                        'avg_similarity': np.mean([s['inter_class_similarity'] for s in group_stats]),
                        'std_similarity': np.std([s['inter_class_similarity'] for s in group_stats])
                    }
            
            # 按比例比较
            for ratio in self.malicious_ratios:
                ratio_str = f"{int(ratio*100)}pct"
                ratio_stats = []
                for group in self.data_groups:
                    config_name = f"{group}_{int(ratio*100)}pct"
                    if config_name in all_analyses:
                        train_stats = all_analyses[config_name]['embedding_stats'].get('train', {})
                        if 'inter_class_similarity' in train_stats:
                            ratio_stats.append({
                                'group': group,
                                'inter_class_similarity': train_stats['inter_class_similarity'],
                                'malicious_count': train_stats.get('malicious', {}).get('count', 0)
                            })
                
                if ratio_stats:
                    comparative_analysis['ratio_comparison'][ratio_str] = {
                        'configs': ratio_stats,
                        'avg_similarity': np.mean([s['inter_class_similarity'] for s in ratio_stats]),
                        'std_similarity': np.std([s['inter_class_similarity'] for s in ratio_stats])
                    }
            
            # 生成统计摘要
            all_similarities = []
            for analysis in all_analyses.values():
                train_stats = analysis['embedding_stats'].get('train', {})
                if 'inter_class_similarity' in train_stats:
                    all_similarities.append(train_stats['inter_class_similarity'])
            
            if all_similarities:
                comparative_analysis['statistical_summary'] = {
                    'overall_mean_similarity': float(np.mean(all_similarities)),
                    'overall_std_similarity': float(np.std(all_similarities)),
                    'similarity_range': {
                        'min': float(np.min(all_similarities)),
                        'max': float(np.max(all_similarities))
                    },
                    'total_analyzed_configs': len(all_similarities)
                }
            
            return comparative_analysis
            
        except Exception as e:
            self.logger.error(f"创建对比分析失败: {str(e)}")
            raise
    
    def create_summary_visualizations(self, comparative_analysis: Dict[str, Any]) -> None:
        """创建总结性可视化"""
        try:
            self.logger.info("创建总结性可视化...")
            
            # 1. 组间对比
            if comparative_analysis.get('group_comparison'):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                groups = list(comparative_analysis['group_comparison'].keys())
                avg_sims = [comparative_analysis['group_comparison'][g]['avg_similarity'] for g in groups]
                std_sims = [comparative_analysis['group_comparison'][g]['std_similarity'] for g in groups]
                
                bars = ax1.bar(groups, avg_sims, yerr=std_sims, capsize=5, alpha=0.7)
                ax1.set_title('Inter-class Similarity by Data Group')
                ax1.set_ylabel('Average Cosine Similarity')
                ax1.set_xlabel('Data Group')
                ax1.grid(True, alpha=0.3)
                
                # 添加数值标签
                for bar, avg_sim in zip(bars, avg_sims):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                           f'{avg_sim:.3f}', ha='center', va='bottom')
            
                # 2. 比例间对比
                if comparative_analysis.get('ratio_comparison'):
                    ratios = list(comparative_analysis['ratio_comparison'].keys())
                    ratio_avg_sims = [comparative_analysis['ratio_comparison'][r]['avg_similarity'] for r in ratios]
                    ratio_std_sims = [comparative_analysis['ratio_comparison'][r]['std_similarity'] for r in ratios]
                    
                    bars2 = ax2.bar(ratios, ratio_avg_sims, yerr=ratio_std_sims, capsize=5, alpha=0.7, color='orange')
                    ax2.set_title('Inter-class Similarity by Malicious Ratio')
                    ax2.set_ylabel('Average Cosine Similarity')
                    ax2.set_xlabel('Malicious Ratio')
                    ax2.grid(True, alpha=0.3)
                    
                    # 添加数值标签
                    for bar, avg_sim in zip(bars2, ratio_avg_sims):
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                               f'{avg_sim:.3f}', ha='center', va='bottom')
                
                plt.tight_layout()
                summary_file = self.visualizations_dir / "phase4_summary_comparison.png"
                plt.savefig(summary_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                self.logger.info(f"✅ 总结可视化已保存到 {summary_file}")
            
        except Exception as e:
            self.logger.error(f"创建总结性可视化失败: {str(e)}")
    
    def run_phase4_complete(self) -> bool:
        """执行完整的Phase 4流程"""
        try:
            self.logger.info("="*50)
            self.logger.info("开始执行批次4 Phase 4: 统一embedding和可视化")
            self.logger.info("="*50)
            
            start_time = time.time()
            
            # Step 1: 加载所有数据集
            self.logger.info("Step 1: 加载所有数据集")
            all_datasets = self.load_all_datasets()
            
            # Step 2: 生成embedding和分析
            self.logger.info("Step 2: 生成embedding和分析")
            all_analyses = {}
            
            total_configs = sum(len(group_data) for group_data in all_datasets.values())
            processed_count = 0
            
            for group, group_datasets in all_datasets.items():
                for ratio_str, dataset_info in group_datasets.items():
                    processed_count += 1
                    config_name = dataset_info['config_name']
                    
                    self.logger.info(f"处理 {processed_count}/{total_configs}: {config_name}")
                    
                    # 生成embedding
                    embeddings = self.generate_embeddings_for_dataset(
                        {'train': dataset_info['train'], 'test': dataset_info['test']}, 
                        config_name
                    )
                    
                    # 分析embedding分布
                    analysis = self.analyze_embedding_distribution(
                        embeddings, 
                        {'train': dataset_info['train'], 'test': dataset_info['test']}, 
                        config_name
                    )
                    
                    all_analyses[config_name] = analysis
                    
                    # 创建可视化
                    self.create_embedding_visualization(
                        embeddings, 
                        {'train': dataset_info['train'], 'test': dataset_info['test']}, 
                        config_name
                    )
            
            # Step 3: 对比分析
            self.logger.info("Step 3: 创建对比分析")
            comparative_analysis = self.create_comparative_analysis(all_analyses)
            
            # Step 4: 总结性可视化
            self.logger.info("Step 4: 创建总结性可视化")
            self.create_summary_visualizations(comparative_analysis)
            
            # Step 5: 保存分析结果
            self.logger.info("Step 5: 保存分析结果")
            
            # 保存所有分析结果
            all_analyses_file = self.analysis_dir / "all_embedding_analyses.json"
            with open(all_analyses_file, 'w', encoding='utf-8') as f:
                json.dump(all_analyses, f, ensure_ascii=False, indent=2)
            
            # 保存对比分析
            comparative_file = self.analysis_dir / "comparative_analysis.json"
            with open(comparative_file, 'w', encoding='utf-8') as f:
                json.dump(comparative_analysis, f, ensure_ascii=False, indent=2)
            
            # 生成Phase 4摘要
            phase4_summary = {
                'phase4_info': {
                    'date': '2025-07-24',
                    'embedding_model': self.model_name,
                    'total_configs_processed': len(all_analyses),
                    'visualizations_created': len(all_analyses) + 1,  # +1 for summary
                    'analysis_files': [
                        str(all_analyses_file),
                        str(comparative_file)
                    ]
                },
                'processing_summary': comparative_analysis['summary'],
                'key_findings': comparative_analysis.get('statistical_summary', {}),
                'output_locations': {
                    'embeddings': str(self.embeddings_dir),
                    'visualizations': str(self.visualizations_dir),
                    'analyses': str(self.analysis_dir)
                }
            }
            
            summary_file = self.analysis_dir / "phase4_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(phase4_summary, f, ensure_ascii=False, indent=2)
            
            elapsed_time = time.time() - start_time
            
            self.logger.info("="*50)
            self.logger.info("批次4 Phase 4 执行完成!")
            self.logger.info(f"总耗时: {elapsed_time/60:.1f} 分钟")
            self.logger.info(f"处理配置数: {len(all_analyses)}")
            self.logger.info(f"生成可视化: {len(all_analyses) + 1}")
            
            # 显示关键发现
            if 'statistical_summary' in comparative_analysis:
                stats = comparative_analysis['statistical_summary']
                self.logger.info("关键发现:")
                self.logger.info(f"- 平均类间相似度: {stats.get('overall_mean_similarity', 0):.3f}")
                self.logger.info(f"- 相似度标准差: {stats.get('overall_std_similarity', 0):.3f}")
                if 'similarity_range' in stats:
                    self.logger.info(f"- 相似度范围: {stats['similarity_range']['min']:.3f} - {stats['similarity_range']['max']:.3f}")
            
            self.logger.info("输出位置:")
            self.logger.info(f"- Embeddings: {self.embeddings_dir}")
            self.logger.info(f"- 可视化: {self.visualizations_dir}")
            self.logger.info(f"- 分析结果: {self.analysis_dir}")
            self.logger.info("="*50)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Phase 4执行失败: {str(e)}")
            return False

def main():
    """主函数"""
    # 设置随机种子
    random.seed(2025)
    np.random.seed(2025)
    
    # 设置日志
    logger = setup_logger("batch4_phase4", log_level=logging.INFO)
    
    try:
        # 创建分析器
        analyzer = Batch4Phase4EmbeddingAnalyzer(logger)
        
        # 执行Phase 4
        success = analyzer.run_phase4_complete()
        
        if success:
            logger.info("批次4 Phase 4 successfully completed!")
            return 0
        else:
            logger.error("批次4 Phase 4 failed!")
            return 1
            
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())