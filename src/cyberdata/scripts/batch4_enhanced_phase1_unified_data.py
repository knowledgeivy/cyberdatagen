#!/usr/bin/env python3
"""
批次4增强版 - Phase 1: 统一数据整合和Embedding生成

基于设计方案实现：
1. 数据源整合和清洗
2. 统一embedding生成  
3. 统一降维处理
4. 元数据标注和ID追踪

作者: Claude
创建时间: 2025-07-27
基于: 批次4embedding可视化改进设计方案.md
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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import warnings
warnings.filterwarnings('ignore')

# 项目路径设置
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.cyberdata.utils.logger_config import setup_logger
from src.cyberdata.utils.data_loader import load_csv_data

class Batch4UnifiedDataProcessor:
    """批次4统一数据处理器"""
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logger("batch4_unified")
        self.project_root = PROJECT_ROOT
        self.batch4_dir = self.project_root / "data" / "batch4_fresh"
        
        # 创建输出目录
        self.enhanced_dir = self.batch4_dir / "enhanced_visualizations"
        self.enhanced_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载embedding模型
        self.model_name = "all-MiniLM-L6-v2"
        self.logger.info(f"初始化embedding模型: {self.model_name}")
        self.embedding_model = SentenceTransformer(self.model_name)
        
        # 设置随机种子
        self.random_state = 2025
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        
        # 采样参数（按设计方案）
        self.sampling_config = {
            'real_benign': 5000,      # 从1M+中采样
            'real_malicious': 5000,   # 从17K中采样  
            'core_synthetic': 300,    # 全部
            'inner_synthetic': 300,   # 全部
            'outer_synthetic': 300,   # 全部
            'edge_synthetic': 300     # 全部
        }
        
        self.logger.info("Phase 1统一数据处理器初始化完成")
    
    def load_malicious_centroid(self) -> np.ndarray:
        """加载恶意样本质心"""
        try:
            centroid_file = self.batch4_dir / "malicious_centroid.json"
            with open(centroid_file, 'r') as f:
                centroid_data = json.load(f)
            
            centroid = np.array(centroid_data['centroid'])
            self.logger.info(f"✅ 加载恶意样本质心: {centroid.shape}")
            return centroid
            
        except Exception as e:
            self.logger.error(f"加载质心失败: {str(e)}")
            raise
    
    def load_raw_data_sources(self) -> Dict[str, pd.DataFrame]:
        """加载所有原始数据源"""
        try:
            self.logger.info("加载原始数据源...")
            
            data_sources = {}
            
            # 1. 真实恶意数据
            mal_file = self.batch4_dir / "train_malicious.csv"
            if mal_file.exists():
                data_sources['real_malicious'] = load_csv_data(mal_file, logger=self.logger)
                self.logger.info(f"✅ 真实恶意数据: {len(data_sources['real_malicious'])} 样本")
            else:
                self.logger.error(f"❌ 真实恶意数据文件不存在: {mal_file}")
                
            # 2. 真实良性数据
            ben_file = self.batch4_dir / "train_benign.csv.gz"
            if ben_file.exists():
                data_sources['real_benign'] = load_csv_data(ben_file, logger=self.logger)
                self.logger.info(f"✅ 真实良性数据: {len(data_sources['real_benign'])} 样本")
            else:
                self.logger.error(f"❌ 真实良性数据文件不存在: {ben_file}")
            
            # 3. 合成数据 (4个距离层)
            synthetic_dir = self.batch4_dir / "synthetic"
            for layer in ['core', 'inner', 'outer', 'edge']:
                layer_file = synthetic_dir / f"{layer}_synthetic.csv"
                if layer_file.exists():
                    data_sources[f'{layer}_synthetic'] = load_csv_data(layer_file, logger=self.logger)
                    self.logger.info(f"✅ {layer}合成数据: {len(data_sources[f'{layer}_synthetic'])} 样本")
                else:
                    self.logger.error(f"❌ {layer}合成数据文件不存在: {layer_file}")
            
            total_samples = sum(len(df) for df in data_sources.values())
            self.logger.info(f"数据源加载完成，总计: {total_samples:,} 样本")
            
            return data_sources
            
        except Exception as e:
            self.logger.error(f"加载数据源失败: {str(e)}")
            raise
    
    def apply_sampling_strategy(self, data_sources: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """应用采样策略"""
        try:
            self.logger.info("应用采样策略...")
            
            sampled_data = {}
            
            for source_name, df in data_sources.items():
                target_size = self.sampling_config.get(source_name, len(df))
                
                if len(df) > target_size:
                    # 需要采样
                    sampled_df = df.sample(n=target_size, random_state=self.random_state)
                    sampled_data[source_name] = sampled_df
                    self.logger.info(f"✅ {source_name}: {len(df)} → {target_size} (采样)")
                else:
                    # 不需要采样
                    sampled_data[source_name] = df
                    self.logger.info(f"✅ {source_name}: {len(df)} (全部)")
            
            total_sampled = sum(len(df) for df in sampled_data.values())
            self.logger.info(f"采样完成，最终样本数: {total_sampled:,}")
            
            return sampled_data
            
        except Exception as e:
            self.logger.error(f"采样失败: {str(e)}")
            raise
    
    def create_unified_dataframe(self, sampled_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """创建统一的DataFrame"""
        try:
            self.logger.info("创建统一DataFrame...")
            
            unified_rows = []
            
            for source_name, df in sampled_data.items():
                self.logger.info(f"处理 {source_name}: {len(df)} 样本")
                
                for idx, row in df.iterrows():
                    # 基础字段
                    unified_row = {
                        'text': row.get('text', ''),
                        'label': row.get('label', 0),
                        'original_id': row.get('unique_id', f"{source_name}_{idx}"),
                        'data_source': source_name
                    }
                    
                    # 数据来源分类
                    if 'real_malicious' in source_name:
                        unified_row.update({
                            'data_category': 'real',
                            'is_malicious': True,
                            'prompt_type': 'real',
                            'distance_layer': 'mixed'  # 真实数据包含所有距离
                        })
                    elif 'real_benign' in source_name:
                        unified_row.update({
                            'data_category': 'real', 
                            'is_malicious': False,
                            'prompt_type': 'real',
                            'distance_layer': 'benign'
                        })
                    else:
                        # 合成数据
                        layer = source_name.replace('_synthetic', '')
                        prompt_type = 'rewrite'  # 默认
                        
                        # 从unique_id解析prompt类型
                        if 'unique_id' in row and pd.notna(row['unique_id']):
                            uid = str(row['unique_id'])
                            if '_rewrite_strong' in uid:
                                prompt_type = 'rewrite_strong'
                            elif '_rewrite_weak' in uid:
                                prompt_type = 'rewrite_weak'
                            elif '_rewrite' in uid:
                                prompt_type = 'rewrite'
                        
                        unified_row.update({
                            'data_category': 'synthetic',
                            'is_malicious': True,  # 所有合成数据都是恶意的
                            'prompt_type': prompt_type,
                            'distance_layer': layer
                        })
                    
                    # 添加其他有用字段
                    for field in ['subject', 'body', 'sender', 'receiver', 'date']:
                        if field in row:
                            unified_row[field] = row[field]
                    
                    unified_rows.append(unified_row)
            
            unified_df = pd.DataFrame(unified_rows)
            
            # 添加索引ID
            unified_df['sample_index'] = range(len(unified_df))
            
            self.logger.info(f"✅ 统一DataFrame创建完成: {len(unified_df)} 样本")
            self.logger.info(f"数据源分布:")
            for source in unified_df['data_source'].value_counts().items():
                self.logger.info(f"  - {source[0]}: {source[1]}")
            
            return unified_df
            
        except Exception as e:
            self.logger.error(f"创建统一DataFrame失败: {str(e)}")
            raise
    
    def generate_unified_embeddings(self, unified_df: pd.DataFrame) -> np.ndarray:
        """生成统一的embedding向量"""
        try:
            self.logger.info("生成统一embedding向量...")
            
            texts = unified_df['text'].tolist()
            
            # 检查是否已存在embedding
            embedding_file = self.enhanced_dir / "unified_embeddings.npy"
            if embedding_file.exists():
                self.logger.info("发现已存在的embedding文件，加载中...")
                embeddings = np.load(embedding_file)
                if embeddings.shape[0] == len(texts):
                    self.logger.info(f"✅ 复用已有embedding: {embeddings.shape}")
                    return embeddings
                else:
                    self.logger.warning("embedding文件样本数不匹配，重新生成...")
            
            # 生成新的embedding
            self.logger.info(f"开始生成embedding: {len(texts)} 个样本")
            
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = self.embedding_model.encode(
                    batch_texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                all_embeddings.append(batch_embeddings)
                
                if (i // batch_size + 1) % 10 == 0:
                    self.logger.info(f"  已处理: {i+len(batch_texts)}/{len(texts)} 样本")
            
            embeddings = np.vstack(all_embeddings)
            
            # 保存embedding
            np.save(embedding_file, embeddings)
            self.logger.info(f"✅ Embedding生成完成并保存: {embeddings.shape}")
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"生成embedding失败: {str(e)}")
            raise
    
    def calculate_distances_to_centroid(self, embeddings: np.ndarray, 
                                      malicious_centroid: np.ndarray) -> np.ndarray:
        """计算到恶意质心的距离"""
        try:
            self.logger.info("计算到恶意质心的距离...")
            
            # 使用余弦距离
            centroid_reshaped = malicious_centroid.reshape(1, -1)
            distances = cosine_distances(embeddings, centroid_reshaped).flatten()
            
            self.logger.info(f"✅ 距离计算完成: {len(distances)} 个样本")
            self.logger.info(f"距离范围: [{distances.min():.3f}, {distances.max():.3f}]")
            self.logger.info(f"平均距离: {distances.mean():.3f} ± {distances.std():.3f}")
            
            return distances
            
        except Exception as e:
            self.logger.error(f"计算距离失败: {str(e)}")
            raise
    
    def perform_unified_dimensionality_reduction(self, embeddings: np.ndarray) -> Dict[str, np.ndarray]:
        """执行统一的降维处理"""
        try:
            self.logger.info("执行统一降维处理...")
            
            # 检查是否已存在降维结果
            pca_file = self.enhanced_dir / "unified_pca_embeddings.npy"
            tsne_file = self.enhanced_dir / "unified_tsne_embeddings.npy"
            
            results = {}
            
            # PCA降维
            if pca_file.exists():
                self.logger.info("加载已有PCA结果...")
                results['pca_2d'] = np.load(pca_file)
            else:
                self.logger.info("执行PCA降维...")
                pca = PCA(n_components=50, random_state=self.random_state)
                pca_intermediate = pca.fit_transform(embeddings)
                
                pca_2d = PCA(n_components=2, random_state=self.random_state)
                results['pca_2d'] = pca_2d.fit_transform(pca_intermediate)
                
                np.save(pca_file, results['pca_2d'])
                self.logger.info(f"✅ PCA降维完成: {results['pca_2d'].shape}")
                self.logger.info(f"解释方差比: {pca_2d.explained_variance_ratio_}")
            
            # t-SNE降维
            if tsne_file.exists():
                self.logger.info("加载已有t-SNE结果...")
                results['tsne_2d'] = np.load(tsne_file)
            else:
                self.logger.info("执行t-SNE降维...")
                # 对大数据集，先PCA到50维再t-SNE
                pca_for_tsne = PCA(n_components=50, random_state=self.random_state)
                embeddings_pca = pca_for_tsne.fit_transform(embeddings)
                
                tsne = TSNE(
                    n_components=2, 
                    random_state=self.random_state, 
                    perplexity=min(30, len(embeddings)//4),
                    max_iter=1000,
                    learning_rate='auto'
                )
                results['tsne_2d'] = tsne.fit_transform(embeddings_pca)
                
                np.save(tsne_file, results['tsne_2d'])
                self.logger.info(f"✅ t-SNE降维完成: {results['tsne_2d'].shape}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"降维处理失败: {str(e)}")
            raise
    
    def assign_density_groups(self, unified_df: pd.DataFrame, 
                            distances: np.ndarray) -> pd.Series:
        """基于embedding密度分配虚拟malicious ratio组"""
        try:
            self.logger.info("分配密度组...")
            
            # 只对恶意样本分组（真实+合成）
            malicious_mask = unified_df['is_malicious']
            malicious_distances = distances[malicious_mask]
            
            # 按距离四分位数分组
            q25, q50, q75 = np.percentile(malicious_distances, [25, 50, 75])
            
            density_groups = np.full(len(unified_df), 'benign', dtype=object)
            
            malicious_indices = np.where(malicious_mask)[0]
            
            for i, mal_idx in enumerate(malicious_indices):
                dist = malicious_distances[i]
                if dist <= q25:
                    density_groups[mal_idx] = '5%_dense_core'
                elif dist <= q50:
                    density_groups[mal_idx] = '10%_medium_dense'
                elif dist <= q75:
                    density_groups[mal_idx] = '15%_lower_dense'
                else:
                    density_groups[mal_idx] = '20%_sparse_edge'
            
            self.logger.info(f"✅ 密度组分配完成")
            density_counts = pd.Series(density_groups).value_counts()
            for group, count in density_counts.items():
                self.logger.info(f"  - {group}: {count}")
            
            return pd.Series(density_groups)
            
        except Exception as e:
            self.logger.error(f"分配密度组失败: {str(e)}")
            raise
    
    def create_final_visualization_dataframe(self, unified_df: pd.DataFrame,
                                           embeddings: np.ndarray,
                                           reduced_embeddings: Dict[str, np.ndarray],
                                           distances: np.ndarray) -> pd.DataFrame:
        """创建最终的可视化DataFrame"""
        try:
            self.logger.info("创建最终可视化DataFrame...")
            
            # 复制原始DataFrame
            final_df = unified_df.copy()
            
            # 添加embedding坐标
            final_df['pca_x'] = reduced_embeddings['pca_2d'][:, 0]
            final_df['pca_y'] = reduced_embeddings['pca_2d'][:, 1]
            final_df['tsne_x'] = reduced_embeddings['tsne_2d'][:, 0]
            final_df['tsne_y'] = reduced_embeddings['tsne_2d'][:, 1]
            
            # 添加距离信息
            final_df['distance_to_centroid'] = distances
            
            # 添加密度组
            final_df['density_group'] = self.assign_density_groups(unified_df, distances)
            
            # 添加可视化用的标签
            final_df['label_name'] = final_df['label'].map({0: 'Benign', 1: 'Malicious'})
            final_df['source_display'] = final_df['data_source'].str.replace('_', ' ').str.title()
            
            # 添加文本预览
            final_df['text_preview'] = final_df['text'].str[:100] + '...'
            
            # 保存完整数据
            output_file = self.enhanced_dir / "unified_embedding_data.csv"
            final_df.to_csv(output_file, index=False)
            self.logger.info(f"✅ 最终DataFrame保存到: {output_file}")
            
            # 保存metadata
            metadata = {
                'total_samples': len(final_df),
                'data_sources': final_df['data_source'].value_counts().to_dict(),
                'prompt_types': final_df['prompt_type'].value_counts().to_dict(),
                'distance_layers': final_df['distance_layer'].value_counts().to_dict(),
                'density_groups': final_df['density_group'].value_counts().to_dict(),
                'distance_stats': {
                    'min': float(distances.min()),
                    'max': float(distances.max()),
                    'mean': float(distances.mean()),
                    'std': float(distances.std())
                },
                'embedding_model': self.model_name,
                'random_state': self.random_state,
                'creation_date': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            metadata_file = self.enhanced_dir / "unified_data_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"✅ 元数据保存到: {metadata_file}")
            
            return final_df
            
        except Exception as e:
            self.logger.error(f"创建最终DataFrame失败: {str(e)}")
            raise
    
    def run_phase1_complete(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """执行完整的Phase 1流程"""
        try:
            self.logger.info("="*60)
            self.logger.info("开始执行批次4增强版 Phase 1: 统一数据整合")
            self.logger.info("="*60)
            
            start_time = time.time()
            
            # Step 1: 加载恶意质心
            self.logger.info("Step 1: 加载恶意质心")
            malicious_centroid = self.load_malicious_centroid()
            
            # Step 2: 加载原始数据源
            self.logger.info("Step 2: 加载原始数据源")
            data_sources = self.load_raw_data_sources()
            
            # Step 3: 应用采样策略
            self.logger.info("Step 3: 应用采样策略")
            sampled_data = self.apply_sampling_strategy(data_sources)
            
            # Step 4: 创建统一DataFrame
            self.logger.info("Step 4: 创建统一DataFrame")
            unified_df = self.create_unified_dataframe(sampled_data)
            
            # Step 5: 生成统一embedding
            self.logger.info("Step 5: 生成统一embedding")
            embeddings = self.generate_unified_embeddings(unified_df)
            
            # Step 6: 计算距离
            self.logger.info("Step 6: 计算到质心距离")
            distances = self.calculate_distances_to_centroid(embeddings, malicious_centroid)
            
            # Step 7: 统一降维
            self.logger.info("Step 7: 统一降维处理")
            reduced_embeddings = self.perform_unified_dimensionality_reduction(embeddings)
            
            # Step 8: 创建最终DataFrame
            self.logger.info("Step 8: 创建最终可视化DataFrame")
            final_df = self.create_final_visualization_dataframe(
                unified_df, embeddings, reduced_embeddings, distances
            )
            
            elapsed_time = time.time() - start_time
            
            # 汇总结果
            summary = {
                'phase1_info': {
                    'date': time.strftime('%Y-%m-%d'),
                    'elapsed_time_minutes': elapsed_time/60,
                    'embedding_model': self.model_name,
                    'random_state': self.random_state
                },
                'data_summary': {
                    'total_samples': len(final_df),
                    'data_sources': final_df['data_source'].value_counts().to_dict(),
                    'prompt_types': final_df['prompt_type'].value_counts().to_dict(),
                    'distance_layers': final_df['distance_layer'].value_counts().to_dict()
                },
                'technical_details': {
                    'embedding_dimension': embeddings.shape[1],
                    'pca_shape': reduced_embeddings['pca_2d'].shape,
                    'tsne_shape': reduced_embeddings['tsne_2d'].shape,
                    'distance_range': [float(distances.min()), float(distances.max())]
                },
                'output_files': {
                    'unified_data': str(self.enhanced_dir / "unified_embedding_data.csv"),
                    'metadata': str(self.enhanced_dir / "unified_data_metadata.json"),
                    'embeddings': str(self.enhanced_dir / "unified_embeddings.npy"),
                    'pca_2d': str(self.enhanced_dir / "unified_pca_embeddings.npy"),
                    'tsne_2d': str(self.enhanced_dir / "unified_tsne_embeddings.npy")
                }
            }
            
            self.logger.info("="*60)
            self.logger.info("批次4增强版 Phase 1 执行完成!")
            self.logger.info(f"总耗时: {elapsed_time/60:.1f} 分钟")
            self.logger.info(f"处理样本数: {len(final_df):,}")
            self.logger.info(f"输出目录: {self.enhanced_dir}")
            
            # 显示关键统计
            self.logger.info("关键统计:")
            self.logger.info(f"- 数据源: {len(final_df['data_source'].unique())} 种")
            self.logger.info(f"- Prompt类型: {len(final_df['prompt_type'].unique())} 种")
            self.logger.info(f"- 距离范围: [{distances.min():.3f}, {distances.max():.3f}]")
            self.logger.info(f"- 恶意样本比例: {final_df['is_malicious'].mean():.1%}")
            self.logger.info("="*60)
            
            return final_df, summary
            
        except Exception as e:
            self.logger.error(f"Phase 1执行失败: {str(e)}")
            raise

def main():
    """主函数"""
    # 设置日志
    logger = setup_logger("batch4_enhanced_phase1", log_level=logging.INFO)
    
    try:
        # 创建处理器
        processor = Batch4UnifiedDataProcessor(logger)
        
        # 执行Phase 1
        final_df, summary = processor.run_phase1_complete()
        
        logger.info("批次4增强版 Phase 1 successfully completed!")
        return 0
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())