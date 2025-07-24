#!/usr/bin/env python3
"""
批次4实验 - Phase 1: 恶意数据embedding和质心分层

基于训练集恶意数据生成embedding，计算质心，
按余弦距离四分位数分层为Core/Inner/Outer/Edge。

作者: Claude
创建时间: 2025-07-24
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import random
from typing import Dict, List, Tuple
import logging
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns

# 项目路径设置
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.cyberdata.utils.logger_config import setup_logger

class Batch4Phase1Stratifier:
    """批次4 Phase 1: 恶意数据embedding和分层器"""
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logger("batch4_phase1")
        self.project_root = PROJECT_ROOT
        self.batch4_dir = self.project_root / "data" / "batch4_fresh"
        
        # 创建输出目录
        self.embeddings_dir = self.batch4_dir / "embeddings"
        self.stratified_dir = self.batch4_dir / "stratified_layers"
        self.analysis_dir = self.batch4_dir / "phase1_analysis"
        
        for dir_path in [self.embeddings_dir, self.stratified_dir, self.analysis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 设置embedding模型
        self.embedding_model = None
        self.model_name = "all-MiniLM-L6-v2"  # 384维embedding
        
        # 随机种子
        self.random_state = 2025
        
    def load_malicious_data(self) -> pd.DataFrame:
        """加载训练集恶意数据"""
        try:
            self.logger.info("加载训练集恶意数据...")
            
            malicious_file = self.batch4_dir / "train_malicious.csv"
            if not malicious_file.exists():
                raise FileNotFoundError(f"恶意数据文件不存在: {malicious_file}")
            
            df = pd.read_csv(malicious_file)
            self.logger.info(f"加载恶意数据: {len(df)} 个样本")
            
            # 验证必要字段
            required_fields = ['unique_id', 'subject', 'body', 'text', 'label']
            missing_fields = [field for field in required_fields if field not in df.columns]
            if missing_fields:
                raise ValueError(f"缺少必要字段: {missing_fields}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"加载恶意数据失败: {str(e)}")
            raise
    
    def initialize_embedding_model(self) -> None:
        """初始化embedding模型"""
        try:
            self.logger.info(f"初始化embedding模型: {self.model_name}")
            
            # 加载sentence transformer模型
            self.embedding_model = SentenceTransformer(self.model_name)
            
            # 获取embedding维度
            test_embedding = self.embedding_model.encode(["test"])
            embedding_dim = test_embedding.shape[1]
            self.logger.info(f"Embedding维度: {embedding_dim}")
            
        except Exception as e:
            self.logger.error(f"初始化embedding模型失败: {str(e)}")
            raise
    
    def generate_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """生成文本embedding"""
        try:
            self.logger.info("生成文本embedding...")
            
            # 提取文本内容
            texts = df['text'].tolist()
            
            # 批量生成embedding
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            self.logger.info(f"Embedding生成完成: {embeddings.shape}")
            
            # 保存embedding
            embedding_file = self.embeddings_dir / "malicious_embeddings.npy"
            np.save(embedding_file, embeddings)
            self.logger.info(f"Embedding已保存: {embedding_file}")
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"生成embedding失败: {str(e)}")
            raise
    
    def calculate_centroid(self, embeddings: np.ndarray) -> np.ndarray:
        """计算embedding质心"""
        try:
            self.logger.info("计算embedding质心...")
            
            # 计算质心（平均向量）
            centroid = np.mean(embeddings, axis=0)
            self.logger.info(f"质心向量维度: {centroid.shape}")
            
            # 保存质心
            centroid_data = {
                'centroid': centroid.tolist(),
                'dimension': int(centroid.shape[0]),
                'sample_count': int(embeddings.shape[0]),
                'calculation_date': '2025-07-24'
            }
            
            centroid_file = self.batch4_dir / "malicious_centroid.json"
            with open(centroid_file, 'w', encoding='utf-8') as f:
                json.dump(centroid_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"质心已保存: {centroid_file}")
            
            return centroid
            
        except Exception as e:
            self.logger.error(f"计算质心失败: {str(e)}")
            raise
    
    def calculate_distances_to_centroid(self, embeddings: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        """计算所有样本到质心的余弦距离"""
        try:
            self.logger.info("计算样本到质心的余弦距离...")
            
            # 计算余弦距离
            distances = np.array([cosine(embedding, centroid) for embedding in embeddings])
            
            # 基本统计
            self.logger.info(f"距离统计:")
            self.logger.info(f"  最小距离: {distances.min():.4f}")
            self.logger.info(f"  最大距离: {distances.max():.4f}")
            self.logger.info(f"  平均距离: {distances.mean():.4f}")
            self.logger.info(f"  标准差: {distances.std():.4f}")
            
            return distances
            
        except Exception as e:
            self.logger.error(f"计算距离失败: {str(e)}")
            raise
    
    def perform_quartile_stratification(self, df: pd.DataFrame, distances: np.ndarray) -> Dict[str, pd.DataFrame]:
        """按四分位数进行分层"""
        try:
            self.logger.info("按四分位数进行分层...")
            
            # 计算四分位数
            q25 = np.percentile(distances, 25)
            q50 = np.percentile(distances, 50)
            q75 = np.percentile(distances, 75)
            
            self.logger.info(f"四分位数:")
            self.logger.info(f"  Q1 (25%): {q25:.4f}")
            self.logger.info(f"  Q2 (50%): {q50:.4f}")
            self.logger.info(f"  Q3 (75%): {q75:.4f}")
            
            # 添加距离到数据框
            df_with_distance = df.copy()
            df_with_distance['distance_to_centroid'] = distances
            
            # 分层
            layers = {}
            
            # Core层: 距离最小的25%（最接近质心）
            core_mask = distances <= q25
            layers['core'] = df_with_distance[core_mask].copy()
            layers['core']['layer'] = 'core'
            
            # Inner层: 25%-50%
            inner_mask = (distances > q25) & (distances <= q50)
            layers['inner'] = df_with_distance[inner_mask].copy()
            layers['inner']['layer'] = 'inner'
            
            # Outer层: 50%-75%
            outer_mask = (distances > q50) & (distances <= q75)
            layers['outer'] = df_with_distance[outer_mask].copy()
            layers['outer']['layer'] = 'outer'
            
            # Edge层: 距离最大的25%（最远离质心）
            edge_mask = distances > q75
            layers['edge'] = df_with_distance[edge_mask].copy()
            layers['edge']['layer'] = 'edge'
            
            # 统计信息
            for layer_name, layer_df in layers.items():
                count = len(layer_df)
                min_dist = layer_df['distance_to_centroid'].min()
                max_dist = layer_df['distance_to_centroid'].max()
                mean_dist = layer_df['distance_to_centroid'].mean()
                
                self.logger.info(f"{layer_name.capitalize()}层: {count} 样本, "
                               f"距离范围 [{min_dist:.4f}, {max_dist:.4f}], "
                               f"平均距离 {mean_dist:.4f}")
            
            return layers
            
        except Exception as e:
            self.logger.error(f"分层失败: {str(e)}")
            raise
    
    def update_layer_ids(self, layers: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """更新分层数据的ID标注"""
        try:
            self.logger.info("更新分层数据的ID标注...")
            
            updated_layers = {}
            
            for layer_name, layer_df in layers.items():
                updated_df = layer_df.copy()
                
                # 更新unique_id格式
                new_ids = []
                for i, old_id in enumerate(updated_df['unique_id']):
                    new_id = f"train_real_mal_{layer_name}_{i:06d}"
                    new_ids.append(new_id)
                
                updated_df['unique_id'] = new_ids
                updated_layers[layer_name] = updated_df
                
                self.logger.info(f"{layer_name}层ID更新完成: {len(updated_df)} 个样本")
            
            return updated_layers
            
        except Exception as e:
            self.logger.error(f"更新ID标注失败: {str(e)}")
            raise
    
    def save_stratified_layers(self, layers: Dict[str, pd.DataFrame]) -> None:
        """保存分层数据"""
        try:
            self.logger.info("保存分层数据...")
            
            for layer_name, layer_df in layers.items():
                layer_file = self.stratified_dir / f"{layer_name}_samples.csv"
                layer_df.to_csv(layer_file, index=False, encoding='utf-8')
                self.logger.info(f"保存{layer_name}层: {layer_file} ({len(layer_df)} 样本)")
            
        except Exception as e:
            self.logger.error(f"保存分层数据失败: {str(e)}")
            raise
    
    def create_distance_visualization(self, distances: np.ndarray, layers: Dict[str, pd.DataFrame]) -> None:
        """创建距离分布可视化"""
        try:
            self.logger.info("创建距离分布可视化...")
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False\n            \n            # 创建子图\n            fig, axes = plt.subplots(2, 2, figsize=(15, 12))\n            fig.suptitle('批次4 恶意数据质心距离分层分析', fontsize=16, fontweight='bold')\n            \n            # 1. 距离分布直方图\n            axes[0, 0].hist(distances, bins=50, alpha=0.7, color='skyblue', edgecolor='black')\n            axes[0, 0].axvline(distances.mean(), color='red', linestyle='--', label=f'平均距离: {distances.mean():.4f}')\n            axes[0, 0].set_xlabel('到质心的余弦距离')\n            axes[0, 0].set_ylabel('样本数量')\n            axes[0, 0].set_title('距离分布直方图')\n            axes[0, 0].legend()\n            axes[0, 0].grid(True, alpha=0.3)\n            \n            # 2. 分层距离箱线图\n            layer_distances = [layers[layer]['distance_to_centroid'].values for layer in ['core', 'inner', 'outer', 'edge']]\n            box_plot = axes[0, 1].boxplot(layer_distances, labels=['Core', 'Inner', 'Outer', 'Edge'], \n                                         patch_artist=True)\n            \n            # 设置箱线图颜色\n            colors = ['lightcoral', 'lightblue', 'lightgreen', 'wheat']\n            for patch, color in zip(box_plot['boxes'], colors):\n                patch.set_facecolor(color)\n            \n            axes[0, 1].set_xlabel('分层')\n            axes[0, 1].set_ylabel('到质心的余弦距离')\n            axes[0, 1].set_title('各层距离分布箱线图')\n            axes[0, 1].grid(True, alpha=0.3)\n            \n            # 3. 分层样本数量条形图\n            layer_counts = [len(layers[layer]) for layer in ['core', 'inner', 'outer', 'edge']]\n            bars = axes[1, 0].bar(['Core', 'Inner', 'Outer', 'Edge'], layer_counts, \n                                 color=colors, edgecolor='black')\n            \n            # 添加数值标签\n            for bar, count in zip(bars, layer_counts):\n                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, \n                               str(count), ha='center', va='bottom', fontweight='bold')\n            \n            axes[1, 0].set_xlabel('分层')\n            axes[1, 0].set_ylabel('样本数量')\n            axes[1, 0].set_title('各层样本数量')\n            axes[1, 0].grid(True, alpha=0.3)\n            \n            # 4. 累积分布函数\n            sorted_distances = np.sort(distances)\n            cumulative = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)\n            axes[1, 1].plot(sorted_distances, cumulative, linewidth=2, color='navy')\n            \n            # 添加四分位数线\n            quartiles = [np.percentile(distances, q) for q in [25, 50, 75]]\n            quartile_labels = ['Q1 (25%)', 'Q2 (50%)', 'Q3 (75%)']\n            colors_q = ['red', 'orange', 'purple']\n            \n            for q_val, q_label, q_color in zip(quartiles, quartile_labels, colors_q):\n                axes[1, 1].axvline(q_val, color=q_color, linestyle='--', \n                                  label=f'{q_label}: {q_val:.4f}')\n            \n            axes[1, 1].set_xlabel('到质心的余弦距离')\n            axes[1, 1].set_ylabel('累积概率')\n            axes[1, 1].set_title('距离累积分布函数')\n            axes[1, 1].legend()\n            axes[1, 1].grid(True, alpha=0.3)\n            \n            # 调整布局\n            plt.tight_layout()\n            \n            # 保存图像\n            viz_file = self.analysis_dir / \"distance_stratification_analysis.png\"\n            plt.savefig(viz_file, dpi=300, bbox_inches='tight')\n            self.logger.info(f\"可视化图像已保存: {viz_file}\")\n            \n            plt.close()\n            \n        except Exception as e:\n            self.logger.error(f\"创建可视化失败: {str(e)}\")\n            # 不抛出异常，因为可视化失败不应影响主流程\n    \n    def generate_stratification_summary(self, layers: Dict[str, pd.DataFrame], \n                                      distances: np.ndarray, centroid: np.ndarray) -> None:\n        \"\"\"生成分层总结报告\"\"\"\n        try:\n            self.logger.info(\"生成分层总结报告...\")\n            \n            summary = {\n                'stratification_info': {\n                    'date': '2025-07-24',\n                    'method': 'cosine_distance_quartiles',\n                    'embedding_model': self.model_name,\n                    'embedding_dimension': int(centroid.shape[0]),\n                    'total_samples': int(len(distances))\n                },\n                'distance_statistics': {\n                    'min_distance': float(distances.min()),\n                    'max_distance': float(distances.max()),\n                    'mean_distance': float(distances.mean()),\n                    'std_distance': float(distances.std()),\n                    'quartiles': {\n                        'q25': float(np.percentile(distances, 25)),\n                        'q50': float(np.percentile(distances, 50)),\n                        'q75': float(np.percentile(distances, 75))\n                    }\n                },\n                'layer_statistics': {},\n                'id_format': {\n                    'pattern': 'train_real_mal_{layer}_{index:06d}',\n                    'example': 'train_real_mal_core_000001'\n                }\n            }\n            \n            # 各层统计信息\n            for layer_name, layer_df in layers.items():\n                layer_stats = {\n                    'sample_count': len(layer_df),\n                    'percentage': len(layer_df) / len(distances) * 100,\n                    'distance_range': {\n                        'min': float(layer_df['distance_to_centroid'].min()),\n                        'max': float(layer_df['distance_to_centroid'].max()),\n                        'mean': float(layer_df['distance_to_centroid'].mean()),\n                        'std': float(layer_df['distance_to_centroid'].std())\n                    }\n                }\n                summary['layer_statistics'][layer_name] = layer_stats\n            \n            # 保存总结\n            summary_file = self.analysis_dir / \"stratification_summary.json\"\n            with open(summary_file, 'w', encoding='utf-8') as f:\n                json.dump(summary, f, ensure_ascii=False, indent=2)\n            \n            self.logger.info(f\"分层总结已保存: {summary_file}\")\n            \n        except Exception as e:\n            self.logger.error(f\"生成总结报告失败: {str(e)}\")\n            raise\n    \n    def run_phase1_complete(self) -> bool:\n        \"\"\"执行完整的Phase 1流程\"\"\"\n        try:\n            self.logger.info(\"=\"*50)\n            self.logger.info(\"开始执行批次4 Phase 1: 恶意数据embedding和质心分层\")\n            self.logger.info(\"=\"*50)\n            \n            # Step 1: 加载恶意数据\n            self.logger.info(\"Step 1: 加载训练集恶意数据\")\n            malicious_df = self.load_malicious_data()\n            \n            # Step 2: 初始化embedding模型\n            self.logger.info(\"Step 2: 初始化embedding模型\")\n            self.initialize_embedding_model()\n            \n            # Step 3: 生成embedding\n            self.logger.info(\"Step 3: 生成文本embedding\")\n            embeddings = self.generate_embeddings(malicious_df)\n            \n            # Step 4: 计算质心\n            self.logger.info(\"Step 4: 计算embedding质心\")\n            centroid = self.calculate_centroid(embeddings)\n            \n            # Step 5: 计算距离\n            self.logger.info(\"Step 5: 计算样本到质心的距离\")\n            distances = self.calculate_distances_to_centroid(embeddings, centroid)\n            \n            # Step 6: 四分位数分层\n            self.logger.info(\"Step 6: 按四分位数进行分层\")\n            layers = self.perform_quartile_stratification(malicious_df, distances)\n            \n            # Step 7: 更新ID标注\n            self.logger.info(\"Step 7: 更新分层数据的ID标注\")\n            layers_with_ids = self.update_layer_ids(layers)\n            \n            # Step 8: 保存分层数据\n            self.logger.info(\"Step 8: 保存分层数据\")\n            self.save_stratified_layers(layers_with_ids)\n            \n            # Step 9: 创建可视化\n            self.logger.info(\"Step 9: 创建距离分布可视化\")\n            self.create_distance_visualization(distances, layers_with_ids)\n            \n            # Step 10: 生成总结报告\n            self.logger.info(\"Step 10: 生成分层总结报告\")\n            self.generate_stratification_summary(layers_with_ids, distances, centroid)\n            \n            self.logger.info(\"=\"*50)\n            self.logger.info(\"批次4 Phase 1 执行完成!\")\n            self.logger.info(f\"总样本数: {len(malicious_df)}\")\n            for layer_name, layer_df in layers_with_ids.items():\n                self.logger.info(f\"{layer_name.capitalize()}层: {len(layer_df)} 样本\")\n            self.logger.info(\"分层数据已保存到: data/batch4_fresh/stratified_layers/\")\n            self.logger.info(\"=\"*50)\n            \n            return True\n            \n        except Exception as e:\n            self.logger.error(f\"Phase 1执行失败: {str(e)}\")\n            return False\n\ndef main():\n    \"\"\"主函数\"\"\"\n    # 设置随机种子\n    random.seed(2025)\n    np.random.seed(2025)\n    \n    # 设置日志\n    logger = setup_logger(\"batch4_phase1\", log_level=logging.INFO)\n    \n    try:\n        # 创建分层器\n        stratifier = Batch4Phase1Stratifier(logger)\n        \n        # 执行Phase 1\n        success = stratifier.run_phase1_complete()\n        \n        if success:\n            logger.info(\"批次4 Phase 1 successfully completed!\")\n            return 0\n        else:\n            logger.error(\"批次4 Phase 1 failed!\")\n            return 1\n            \n    except Exception as e:\n        logger.error(f\"程序执行失败: {str(e)}\")\n        return 1\n\nif __name__ == \"__main__\":\n    exit(main())