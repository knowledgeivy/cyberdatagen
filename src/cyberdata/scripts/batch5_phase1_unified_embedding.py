#!/usr/bin/env python3
"""
批次5实验 - Phase 1: 统一Embedding空间构建

为所有数据建立统一的embedding表示空间，包括：
- 真实恶意样本（用于采样）
- 1200个分层合成样本 (core/inner/outer/edge × 300)
- 固定测试集
- 真实良性样本（用于构建训练集）

目标：解决批次4数据混合污染问题，建立纯净数据源的科学对比基础

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
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

# 项目路径设置
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.cyberdata.utils.logger_config import setup_logger
from src.cyberdata.utils.data_loader import load_csv_data, save_csv_data

class Batch5Phase1UnifiedEmbedding:
    """批次5 Phase 1: 统一embedding空间构建器"""
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logger("batch5_phase1")
        self.project_root = PROJECT_ROOT
        self.batch4_dir = self.project_root / "data" / "batch4_fresh"
        self.batch5_dir = self.project_root / "data" / "batch5"
        
        # 创建输出目录
        self.unified_dir = self.batch5_dir / "unified_embeddings"
        self.analysis_dir = self.batch5_dir / "phase1_analysis"
        
        for dir_path in [self.unified_dir, self.analysis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 模型配置
        self.model_name = 'all-MiniLM-L6-v2'  # 与批次4保持一致
        self.embedding_dim = 384
        
        # 随机种子
        self.random_state = 2025
        
        self.logger.info(f"Phase 1初始化完成，使用模型: {self.model_name}")
    
    def load_all_source_data(self) -> Dict[str, pd.DataFrame]:
        """加载所有源数据"""
        try:
            self.logger.info("加载所有源数据...")
            
            data = {}
            
            # 1. 加载真实训练数据
            self.logger.info("加载真实训练数据...")
            data['train_malicious'] = load_csv_data(
                self.batch4_dir / "train_malicious.csv", logger=self.logger
            )
            data['train_benign'] = load_csv_data(
                self.batch4_dir / "train_benign.csv.gz", logger=self.logger
            )
            
            # 2. 加载固定测试集
            self.logger.info("加载固定测试集...")
            data['test_set'] = load_csv_data(
                self.batch4_dir / "raw_test_set.csv", logger=self.logger
            )
            
            # 3. 加载所有合成数据
            self.logger.info("加载分层合成数据...")
            synthetic_dir = self.batch4_dir / "synthetic"
            for layer in ['core', 'inner', 'outer', 'edge']:
                data[f'{layer}_synthetic'] = load_csv_data(
                    synthetic_dir / f"{layer}_synthetic.csv", logger=self.logger
                )
            
            # 统计信息
            total_samples = 0
            for name, df in data.items():
                malicious_count = len(df[df['label'] == 1]) if 'label' in df.columns else 0
                benign_count = len(df[df['label'] == 0]) if 'label' in df.columns else 0
                total_count = len(df)
                total_samples += total_count
                self.logger.info(f"{name}: {total_count:,} 样本 ({malicious_count:,} 恶意, {benign_count:,} 良性)")
            
            self.logger.info(f"总计加载: {total_samples:,} 样本")
            
            return data
            
        except Exception as e:
            self.logger.error(f"加载源数据失败: {str(e)}")
            raise
    
    def prepare_unified_dataset(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """准备统一数据集用于embedding"""
        try:
            self.logger.info("准备统一数据集...")
            
            unified_data = []
            
            # 1. 真实恶意样本（采样用于embedding，避免内存问题）
            real_malicious = data['train_malicious'].copy()
            
            # 采样策略：保持代表性的同时控制数据量
            if len(real_malicious) > 50000:
                real_malicious_sample = real_malicious.sample(
                    n=50000, random_state=self.random_state
                ).copy()
                self.logger.info(f"真实恶意数据采样: {len(real_malicious_sample):,} / {len(real_malicious):,}")
            else:
                real_malicious_sample = real_malicious.copy()
            
            real_malicious_sample['data_source'] = 'real_malicious'
            real_malicious_sample['prompt_variant'] = 'original'
            real_malicious_sample['layer'] = 'original'
            unified_data.append(real_malicious_sample)
            
            # 2. 所有合成数据
            for layer in ['core', 'inner', 'outer', 'edge']:
                synthetic_df = data[f'{layer}_synthetic'].copy()
                synthetic_df['data_source'] = 'synthetic'
                synthetic_df['layer'] = layer
                
                # 确保有prompt_variant列，如果没有从unique_id解析
                if 'prompt_variant' not in synthetic_df.columns:
                    synthetic_df['prompt_variant'] = synthetic_df['unique_id'].str.split('_').str[-1]
                
                unified_data.append(synthetic_df)
                self.logger.info(f"{layer}层合成数据: {len(synthetic_df):,} 样本")
            
            # 3. 真实良性样本（采样）
            real_benign = data['train_benign'].copy()
            if len(real_benign) > 20000:
                real_benign_sample = real_benign.sample(
                    n=20000, random_state=self.random_state
                ).copy()
                self.logger.info(f"真实良性数据采样: {len(real_benign_sample):,} / {len(real_benign):,}")
            else:
                real_benign_sample = real_benign.copy()
            
            real_benign_sample['data_source'] = 'real_benign'
            real_benign_sample['prompt_variant'] = 'original'
            real_benign_sample['layer'] = 'original'
            unified_data.append(real_benign_sample)
            
            # 4. 测试集（采样）
            test_set = data['test_set'].copy()
            if len(test_set) > 10000:
                test_sample = test_set.sample(
                    n=10000, random_state=self.random_state
                ).copy()
                self.logger.info(f"测试数据采样: {len(test_sample):,} / {len(test_set):,}")
            else:
                test_sample = test_set.copy()
            
            test_sample['data_source'] = 'test_set'
            test_sample['prompt_variant'] = 'original'
            test_sample['layer'] = 'original'
            unified_data.append(test_sample)
            
            # 合并所有数据
            unified_df = pd.concat(unified_data, ignore_index=True)
            
            # 确保必要列存在
            required_columns = ['unique_id', 'text', 'label', 'data_source', 'prompt_variant', 'layer']
            for col in required_columns:
                if col not in unified_df.columns:
                    if col == 'unique_id':
                        unified_df[col] = unified_df.index.astype(str)
                    else:
                        unified_df[col] = 'unknown'
            
            self.logger.info(f"统一数据集构建完成: {len(unified_df):,} 样本")
            
            # 数据源分布统计
            source_stats = unified_df['data_source'].value_counts()
            self.logger.info("数据源分布:")
            for source, count in source_stats.items():
                self.logger.info(f"  {source}: {count:,} 样本")
            
            return unified_df
            
        except Exception as e:
            self.logger.error(f"准备统一数据集失败: {str(e)}")
            raise
    
    def generate_unified_embeddings(self, unified_df: pd.DataFrame) -> np.ndarray:
        """生成统一embedding向量"""
        try:
            self.logger.info("开始生成统一embedding向量...")
            
            # 加载embedding模型
            self.logger.info(f"加载embedding模型: {self.model_name}")
            model = SentenceTransformer(self.model_name)
            
            # 准备文本数据
            texts = unified_df['text'].fillna('').astype(str).tolist()
            total_texts = len(texts)
            
            self.logger.info(f"准备embedding计算: {total_texts:,} 个文本")
            
            # 批量计算embedding
            batch_size = 1000
            embeddings = []
            
            for i in range(0, total_texts, batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = model.encode(batch_texts, show_progress_bar=True)
                embeddings.append(batch_embeddings)
                
                progress = min(i + batch_size, total_texts)
                self.logger.info(f"Embedding进度: {progress:,}/{total_texts:,} ({progress/total_texts*100:.1f}%)")
            
            # 合并所有embedding
            all_embeddings = np.vstack(embeddings)
            
            self.logger.info(f"Embedding生成完成: {all_embeddings.shape}")
            
            return all_embeddings
            
        except Exception as e:
            self.logger.error(f"生成统一embedding失败: {str(e)}")
            raise
    
    def calculate_centroids_and_distances(self, unified_df: pd.DataFrame, 
                                        embeddings: np.ndarray) -> Dict[str, Any]:
        """计算质心和距离统计"""
        try:
            self.logger.info("计算质心和距离统计...")
            
            stats = {
                'centroids': {},
                'distance_statistics': {},
                'layer_analysis': {}
            }
            
            # 1. 计算各数据源的质心
            for source in unified_df['data_source'].unique():
                source_mask = unified_df['data_source'] == source
                source_embeddings = embeddings[source_mask]
                centroid = np.mean(source_embeddings, axis=0)
                stats['centroids'][source] = centroid.tolist()
                
                self.logger.info(f"{source} 质心计算完成: {len(source_embeddings):,} 样本")
            
            # 2. 计算合成数据各层的质心和距离分布
            synthetic_mask = unified_df['data_source'] == 'synthetic'
            if synthetic_mask.any():
                synthetic_df = unified_df[synthetic_mask].copy()
                synthetic_embeddings = embeddings[synthetic_mask]
                
                # 总体合成数据质心
                synthetic_centroid = np.mean(synthetic_embeddings, axis=0)
                stats['centroids']['all_synthetic'] = synthetic_centroid.tolist()
                
                # 各层质心和统计
                for layer in ['core', 'inner', 'outer', 'edge']:
                    layer_mask = synthetic_df['layer'] == layer
                    if layer_mask.any():
                        layer_embeddings = synthetic_embeddings[layer_mask]
                        layer_centroid = np.mean(layer_embeddings, axis=0)
                        stats['centroids'][f'{layer}_synthetic'] = layer_centroid.tolist()
                        
                        # 计算层内距离分布
                        distances_to_centroid = cosine_distances(
                            layer_embeddings, layer_centroid.reshape(1, -1)
                        ).flatten()
                        
                        stats['layer_analysis'][layer] = {
                            'sample_count': len(layer_embeddings),
                            'mean_distance': float(np.mean(distances_to_centroid)),
                            'std_distance': float(np.std(distances_to_centroid)),
                            'min_distance': float(np.min(distances_to_centroid)),
                            'max_distance': float(np.max(distances_to_centroid)),
                            'quartiles': [
                                float(np.percentile(distances_to_centroid, 25)),
                                float(np.percentile(distances_to_centroid, 50)),
                                float(np.percentile(distances_to_centroid, 75))
                            ]
                        }
                        
                        self.logger.info(f"{layer}层: {len(layer_embeddings)} 样本, "
                                       f"平均距离: {np.mean(distances_to_centroid):.4f}")
            
            # 3. 计算prompt变体的统计
            if 'prompt_variant' in unified_df.columns:
                for variant in ['rewrite', 'rewrite_strong', 'rewrite_weak']:
                    variant_mask = (unified_df['data_source'] == 'synthetic') & \
                                 (unified_df['prompt_variant'] == variant)
                    if variant_mask.any():
                        variant_embeddings = embeddings[variant_mask]
                        variant_centroid = np.mean(variant_embeddings, axis=0)
                        stats['centroids'][f'{variant}_synthetic'] = variant_centroid.tolist()
                        
                        self.logger.info(f"{variant}变体: {len(variant_embeddings)} 样本")
            
            return stats
            
        except Exception as e:
            self.logger.error(f"计算质心和距离统计失败: {str(e)}")
            raise
    
    def save_unified_embedding_data(self, unified_df: pd.DataFrame, embeddings: np.ndarray,
                                  stats: Dict[str, Any]) -> None:
        """保存统一embedding数据"""
        try:
            self.logger.info("保存统一embedding数据...")
            
            # 1. 保存embedding矩阵
            embeddings_file = self.unified_dir / "all_embeddings.npy"
            np.save(embeddings_file, embeddings)
            self.logger.info(f"Embedding矩阵已保存: {embeddings_file}")
            
            # 2. 保存元数据 (压缩格式以适应GitHub限制)
            metadata_file = self.unified_dir / "embedding_metadata.csv.gz"
            save_csv_data(unified_df, metadata_file, compress=True, logger=self.logger)
            self.logger.info(f"元数据已保存: {metadata_file}")
            
            # 3. 保存质心信息
            centroids_file = self.unified_dir / "layer_centroids.json"
            with open(centroids_file, 'w', encoding='utf-8') as f:
                json.dump(stats['centroids'], f, ensure_ascii=False, indent=2)
            self.logger.info(f"质心信息已保存: {centroids_file}")
            
            # 4. 保存距离统计
            distance_stats_file = self.unified_dir / "distance_statistics.json"
            distance_stats = {
                'layer_analysis': stats['layer_analysis'],
                'generation_info': {
                    'date': '2025-07-30',
                    'model': self.model_name,
                    'embedding_dim': self.embedding_dim,
                    'total_samples': len(unified_df),
                    'random_state': self.random_state
                }
            }
            
            with open(distance_stats_file, 'w', encoding='utf-8') as f:
                json.dump(distance_stats, f, ensure_ascii=False, indent=2)
            self.logger.info(f"距离统计已保存: {distance_stats_file}")
            
        except Exception as e:
            self.logger.error(f"保存统一embedding数据失败: {str(e)}")
            raise
    
    def generate_phase1_summary(self, unified_df: pd.DataFrame, embeddings: np.ndarray,
                              stats: Dict[str, Any]) -> None:
        """生成Phase 1统计摘要"""
        try:
            self.logger.info("生成Phase 1统计摘要...")
            
            # 数据源统计
            source_stats = unified_df['data_source'].value_counts().to_dict()
            
            # 标签分布
            label_stats = unified_df['label'].value_counts().to_dict()
            
            # 合成数据分析
            synthetic_analysis = {}
            if 'synthetic' in source_stats:
                synthetic_df = unified_df[unified_df['data_source'] == 'synthetic']
                
                # 按层统计
                layer_distribution = synthetic_df['layer'].value_counts().to_dict()
                
                # 按prompt变体统计
                variant_distribution = synthetic_df['prompt_variant'].value_counts().to_dict()
                
                synthetic_analysis = {
                    'total_synthetic': source_stats.get('synthetic', 0),
                    'layer_distribution': layer_distribution,
                    'variant_distribution': variant_distribution
                }
            
            summary = {
                'phase1_info': {
                    'date': '2025-07-30',
                    'model': self.model_name,
                    'embedding_dimension': self.embedding_dim,
                    'random_state': self.random_state,
                    'total_samples': len(unified_df),
                    'embedding_shape': list(embeddings.shape)
                },
                'data_distribution': {
                    'by_source': source_stats,
                    'by_label': label_stats,
                    'synthetic_analysis': synthetic_analysis
                },
                'quality_metrics': {
                    'layer_analysis': stats['layer_analysis'],
                    'centroids_computed': len(stats['centroids'])
                },
                'file_outputs': {
                    'embeddings': 'all_embeddings.npy',
                    'metadata': 'embedding_metadata.csv.gz',
                    'centroids': 'layer_centroids.json',
                    'statistics': 'distance_statistics.json'
                }
            }
            
            # 保存摘要
            summary_file = self.analysis_dir / "phase1_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Phase 1统计摘要已保存: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"生成Phase 1统计摘要失败: {str(e)}")
            raise
    
    def run_phase1_complete(self) -> bool:
        """执行完整的Phase 1流程"""
        try:
            self.logger.info("="*50)
            self.logger.info("开始执行批次5 Phase 1: 统一Embedding空间构建")
            self.logger.info("="*50)
            
            start_time = time.time()
            
            # Step 1: 加载所有源数据
            self.logger.info("Step 1: 加载所有源数据")
            data = self.load_all_source_data()
            
            # Step 2: 准备统一数据集
            self.logger.info("Step 2: 准备统一数据集")
            unified_df = self.prepare_unified_dataset(data)
            
            # Step 3: 生成统一embedding
            self.logger.info("Step 3: 生成统一embedding")
            embeddings = self.generate_unified_embeddings(unified_df)
            
            # Step 4: 计算质心和距离统计
            self.logger.info("Step 4: 计算质心和距离统计")
            stats = self.calculate_centroids_and_distances(unified_df, embeddings)
            
            # Step 5: 保存所有数据
            self.logger.info("Step 5: 保存embedding数据")
            self.save_unified_embedding_data(unified_df, embeddings, stats)
            
            # Step 6: 生成统计摘要
            self.logger.info("Step 6: 生成统计摘要")
            self.generate_phase1_summary(unified_df, embeddings, stats)
            
            elapsed_time = time.time() - start_time
            
            self.logger.info("="*50)
            self.logger.info("批次5 Phase 1 执行完成!")
            self.logger.info(f"总耗时: {elapsed_time/60:.1f} 分钟")
            self.logger.info(f"处理样本数: {len(unified_df):,}")
            self.logger.info(f"Embedding维度: {embeddings.shape}")
            self.logger.info("数据已保存到: data/batch5/unified_embeddings/")
            self.logger.info("="*50)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Phase 1执行失败: {str(e)}")
            return False

def main():
    """主函数"""
    # 设置随机种子
    random.seed(2025)
    np.random.seed(2025)
    
    # 设置日志
    logger = setup_logger("batch5_phase1", log_level=logging.INFO)
    
    try:
        # 创建构建器
        builder = Batch5Phase1UnifiedEmbedding(logger)
        
        # 执行Phase 1
        success = builder.run_phase1_complete()
        
        if success:
            logger.info("批次5 Phase 1 successfully completed!")
            return 0
        else:
            logger.error("批次5 Phase 1 failed!")
            return 1
            
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())