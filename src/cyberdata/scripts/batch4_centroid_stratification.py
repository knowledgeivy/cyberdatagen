#!/usr/bin/env python3
"""
阶段4实验 - Phase 1: 质心计算和内部距离分层

基于真实恶意数据的embedding质心，按距离四分位数分层，
为后续的不平衡数据实验做准备。

作者: Claude
创建时间: 2025-07-24
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial.distance import cosine
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import json
from typing import Dict, List, Tuple, Any
import logging

# 项目路径设置
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.cyberdata.utils.logger_config import setup_logger

class Batch4CentroidAnalyzer:
    """阶段4质心分层分析器"""
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logger("batch4_centroid")
        self.project_root = PROJECT_ROOT
        self.batch4_dir = self.project_root / "data" / "batch4"
        self.stratification_dir = self.batch4_dir / "layer_stratification"
        
        # 确保输出目录存在
        self.stratification_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据存储
        self.malicious_data = None
        self.malicious_embeddings = None
        self.centroid = None
        self.distances = None
        self.layers = None
        
    def load_real_malicious_data(self) -> bool:
        """加载真实恶意数据和embeddings"""
        try:
            # 加载真实恶意数据
            malicious_file = self.project_root / "data" / "batch1" / "base_samples" / "malicious_5k.csv"
            self.logger.info(f"加载真实恶意数据: {malicious_file}")
            
            self.malicious_data = pd.read_csv(malicious_file)
            self.logger.info(f"加载了 {len(self.malicious_data)} 个真实恶意样本")
            
            # 加载embeddings
            embedding_file = self.project_root / "data" / "batch1" / "embeddings" / "batch1_embeddings.pkl"
            self.logger.info(f"加载embeddings: {embedding_file}")
            
            with open(embedding_file, 'rb') as f:
                embedding_data = pickle.load(f)
            
            # 检查embedding数据结构
            self.logger.info(f"Embedding数据结构: {list(embedding_data.keys())}")
            
            # 提取embeddings、data_ids和data_types
            embeddings = embedding_data['embeddings']  # numpy array
            data_ids = embedding_data['data_ids']      # list of IDs
            data_types = embedding_data['data_types']  # list of types
            
            self.logger.info(f"总embedding数量: {len(embeddings)}")
            self.logger.info(f"数据类型分布: {set(data_types)}")
            
            # 提取真实恶意数据的embeddings
            real_malicious_embeddings = []
            malicious_ids = set(self.malicious_data['data_id'].tolist())
            
            for i, (data_id, data_type) in enumerate(zip(data_ids, data_types)):
                if data_id in malicious_ids and data_type == 'real':
                    real_malicious_embeddings.append(embeddings[i])
            
            self.malicious_embeddings = np.array(real_malicious_embeddings)
            self.logger.info(f"提取了 {len(self.malicious_embeddings)} 个恶意数据embeddings")
            self.logger.info(f"Embedding维度: {self.malicious_embeddings.shape[1]}")
            
            # 验证数据完整性
            if len(self.malicious_data) != len(self.malicious_embeddings):
                self.logger.warning(f"数据不匹配: CSV {len(self.malicious_data)}, Embeddings {len(self.malicious_embeddings)}")
                # 取较小的数量
                min_len = min(len(self.malicious_data), len(self.malicious_embeddings))
                self.malicious_data = self.malicious_data.iloc[:min_len]
                self.malicious_embeddings = self.malicious_embeddings[:min_len]
                self.logger.info(f"调整为 {min_len} 个匹配样本")
            
            return True
            
        except Exception as e:
            self.logger.error(f"加载数据失败: {str(e)}")
            return False
    
    def calculate_centroid(self) -> np.ndarray:
        """计算真实恶意数据的embedding质心"""
        try:
            self.logger.info("计算恶意数据embedding质心...")
            
            # 计算平均向量作为质心
            self.centroid = np.mean(self.malicious_embeddings, axis=0)
            
            self.logger.info(f"质心计算完成，维度: {self.centroid.shape}")
            self.logger.info(f"质心向量范围: [{self.centroid.min():.4f}, {self.centroid.max():.4f}]")
            
            # 保存质心向量
            centroid_file = self.stratification_dir / "centroid_vector.npy"
            np.save(centroid_file, self.centroid)
            self.logger.info(f"质心向量保存到: {centroid_file}")
            
            return self.centroid
            
        except Exception as e:
            self.logger.error(f"质心计算失败: {str(e)}")
            raise
    
    def calculate_distances_to_centroid(self) -> np.ndarray:
        """计算每个样本到质心的余弦距离"""
        try:
            self.logger.info("计算样本到质心的余弦距离...")
            
            distances = []
            for embedding in self.malicious_embeddings:
                # 使用余弦距离
                distance = cosine(embedding, self.centroid)
                distances.append(distance)
            
            self.distances = np.array(distances)
            
            self.logger.info(f"距离计算完成，样本数: {len(self.distances)}")
            self.logger.info(f"距离统计 - 最小: {self.distances.min():.4f}, "
                           f"最大: {self.distances.max():.4f}, "
                           f"平均: {self.distances.mean():.4f}, "
                           f"标准差: {self.distances.std():.4f}")
            
            return self.distances
            
        except Exception as e:
            self.logger.error(f"距离计算失败: {str(e)}")
            raise
    
    def stratify_by_quartiles(self) -> Dict[str, List[Dict]]:
        """按四分位数将样本分层"""
        try:
            self.logger.info("按距离四分位数分层...")
            
            # 计算四分位数
            q25, q50, q75 = np.percentile(self.distances, [25, 50, 75])
            
            self.logger.info(f"四分位数: Q1={q25:.4f}, Q2={q50:.4f}, Q3={q75:.4f}")
            
            # 初始化层次字典
            self.layers = {
                'core': [],    # 0-25%: 最接近质心
                'inner': [],   # 25-50%: 内层
                'outer': [],   # 50-75%: 外层
                'edge': []     # 75-100%: 最远离质心
            }
            
            # 分层逻辑
            for i, distance in enumerate(self.distances):
                # 组合subject和body作为text
                row = self.malicious_data.iloc[i]
                text = f"{row['subject']} {row['body']}" if pd.notna(row['subject']) and pd.notna(row['body']) else str(row.get('subject', '')) + str(row.get('body', ''))
                
                sample_info = {
                    'index': int(i),
                    'data_id': str(row['data_id']),
                    'distance': float(distance),
                    'text': text[:200] + '...' if len(text) > 200 else text,  # 截断长文本用于存储
                    'subject': str(row['subject']) if pd.notna(row['subject']) else '',
                    'body': str(row['body'])[:100] + '...' if len(str(row['body'])) > 100 else str(row['body']),
                    'label': int(row['label']) if pd.notna(row['label']) else 0
                }
                
                if distance <= q25:
                    self.layers['core'].append(sample_info)
                elif distance <= q50:
                    self.layers['inner'].append(sample_info)
                elif distance <= q75:
                    self.layers['outer'].append(sample_info)
                else:
                    self.layers['edge'].append(sample_info)
            
            # 报告分层结果
            self.logger.info("分层结果:")
            for layer_name, samples in self.layers.items():
                distances_in_layer = [s['distance'] for s in samples]
                self.logger.info(f"  {layer_name}: {len(samples)} 样本, "
                               f"距离范围: [{min(distances_in_layer):.4f}, {max(distances_in_layer):.4f}]")
            
            # 保存分层结果
            layers_file = self.stratification_dir / "layered_samples.json"
            
            # 转换为可序列化格式
            serializable_layers = {}
            for layer_name, samples in self.layers.items():
                serializable_layers[layer_name] = []
                for sample in samples:
                    serializable_sample = sample.copy()
                    serializable_sample['distance'] = float(serializable_sample['distance'])
                    serializable_layers[layer_name].append(serializable_sample)
            
            with open(layers_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_layers, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"分层结果保存到: {layers_file}")
            
            return self.layers
            
        except Exception as e:
            self.logger.error(f"分层失败: {str(e)}")
            raise
    
    def visualize_stratification(self) -> None:
        """可视化分层结果"""
        try:
            self.logger.info("生成分层可视化图表...")
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('阶段4: 真实恶意数据内部距离分层分析', fontsize=16, fontweight='bold')
            
            # 1. 距离分布直方图
            ax1 = axes[0, 0]
            ax1.hist(self.distances, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            
            # 添加四分位数线
            q25, q50, q75 = np.percentile(self.distances, [25, 50, 75])
            ax1.axvline(q25, color='red', linestyle='--', alpha=0.8, label=f'Q1={q25:.3f}')
            ax1.axvline(q50, color='orange', linestyle='--', alpha=0.8, label=f'Q2={q50:.3f}')
            ax1.axvline(q75, color='purple', linestyle='--', alpha=0.8, label=f'Q3={q75:.3f}')
            
            ax1.set_xlabel('余弦距离到质心')
            ax1.set_ylabel('样本数量')
            ax1.set_title('距离分布直方图')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 分层样本数量柱状图
            ax2 = axes[0, 1]
            layer_names = ['Core', 'Inner', 'Outer', 'Edge']
            layer_counts = [len(self.layers[name.lower()]) for name in layer_names]
            colors = ['#2E8B57', '#4682B4', '#DAA520', '#DC143C']
            
            bars = ax2.bar(layer_names, layer_counts, color=colors, alpha=0.8)
            ax2.set_ylabel('样本数量')
            ax2.set_title('各层样本数量分布')
            ax2.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, count in zip(bars, layer_counts):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                        str(count), ha='center', va='bottom', fontweight='bold')
            
            # 3. 距离箱线图
            ax3 = axes[1, 0]
            layer_distances = []
            for layer_name in ['core', 'inner', 'outer', 'edge']:
                distances_in_layer = [s['distance'] for s in self.layers[layer_name]]
                layer_distances.append(distances_in_layer)
            
            bp = ax3.boxplot(layer_distances, labels=['Core', 'Inner', 'Outer', 'Edge'], 
                            patch_artist=True, showmeans=True)
            
            # 设置箱线图颜色
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)
            
            ax3.set_ylabel('余弦距离到质心')
            ax3.set_title('各层距离分布箱线图')
            ax3.grid(True, alpha=0.3)
            
            # 4. t-SNE可视化（如果样本不太多）
            ax4 = axes[1, 1]
            if len(self.malicious_embeddings) <= 3000:  # 避免t-SNE计算过慢
                self.logger.info("执行t-SNE降维可视化...")
                tsne = TSNE(n_components=2, random_state=42, perplexity=30)
                embeddings_2d = tsne.fit_transform(self.malicious_embeddings)
                
                # 为每个层次分配颜色
                layer_colors = {'core': '#2E8B57', 'inner': '#4682B4', 'outer': '#DAA520', 'edge': '#DC143C'}
                
                for layer_name, samples in self.layers.items():
                    indices = [s['index'] for s in samples]
                    points = embeddings_2d[indices]
                    ax4.scatter(points[:, 0], points[:, 1], 
                              c=layer_colors[layer_name], label=layer_name.title(), 
                              alpha=0.6, s=20)
                
                ax4.set_xlabel('t-SNE 维度 1')
                ax4.set_ylabel('t-SNE 维度 2')
                ax4.set_title('t-SNE可视化分层结果')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, '样本数量过多\n跳过t-SNE可视化', 
                        ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                ax4.set_title('t-SNE可视化 (跳过)')
            
            plt.tight_layout()
            
            # 保存图表
            plot_file = self.stratification_dir / "distance_distribution.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"可视化图表保存到: {plot_file}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"可视化失败: {str(e)}")
            raise
    
    def generate_stratification_report(self) -> Dict[str, Any]:
        """生成分层分析报告"""
        try:
            self.logger.info("生成分层分析报告...")
            
            # 计算统计信息
            q25, q50, q75 = np.percentile(self.distances, [25, 50, 75])
            
            report = {
                "experiment_info": {
                    "experiment_name": "batch4_centroid_stratification",
                    "date": "2025-07-24",
                    "description": "基于真实恶意数据质心的内部距离分层分析"
                },
                "data_summary": {
                    "total_samples": len(self.malicious_data),
                    "embedding_dimension": self.malicious_embeddings.shape[1],
                    "data_source": "batch1/base_samples/malicious_5k.csv"
                },
                "centroid_analysis": {
                    "centroid_dimension": len(self.centroid),
                    "centroid_norm": float(np.linalg.norm(self.centroid)),
                    "centroid_mean": float(self.centroid.mean()),
                    "centroid_std": float(self.centroid.std())
                },
                "distance_statistics": {
                    "min_distance": float(self.distances.min()),
                    "max_distance": float(self.distances.max()),
                    "mean_distance": float(self.distances.mean()),
                    "std_distance": float(self.distances.std()),
                    "quartiles": {
                        "q25": float(q25),
                        "q50": float(q50),
                        "q75": float(q75)
                    }
                },
                "stratification_results": {}
            }
            
            # 各层详细统计
            for layer_name, samples in self.layers.items():
                distances_in_layer = [s['distance'] for s in samples]
                report["stratification_results"][layer_name] = {
                    "sample_count": len(samples),
                    "percentage": round(len(samples) / len(self.malicious_data) * 100, 2),
                    "distance_range": {
                        "min": float(min(distances_in_layer)),
                        "max": float(max(distances_in_layer)),
                        "mean": float(np.mean(distances_in_layer)),
                        "std": float(np.std(distances_in_layer))
                    }
                }
            
            # 保存报告
            report_file = self.stratification_dir / "stratification_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"分层报告保存到: {report_file}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"报告生成失败: {str(e)}")
            raise
    
    def run_phase1_complete(self) -> bool:
        """执行完整的Phase 1流程"""
        try:
            self.logger.info("="*50)
            self.logger.info("开始执行阶段4 Phase 1: 质心计算和内部距离分层")
            self.logger.info("="*50)
            
            # Step 1: 加载数据
            self.logger.info("Step 1: 加载真实恶意数据和embeddings")
            if not self.load_real_malicious_data():
                return False
            
            # Step 2: 计算质心
            self.logger.info("Step 2: 计算embedding质心")
            self.calculate_centroid()
            
            # Step 3: 计算距离
            self.logger.info("Step 3: 计算样本到质心的距离")
            self.calculate_distances_to_centroid()
            
            # Step 4: 四分位数分层
            self.logger.info("Step 4: 按四分位数分层")
            self.stratify_by_quartiles()
            
            # Step 5: 可视化
            self.logger.info("Step 5: 生成可视化图表")
            self.visualize_stratification()
            
            # Step 6: 生成报告
            self.logger.info("Step 6: 生成分层分析报告")
            report = self.generate_stratification_report()
            
            self.logger.info("="*50)
            self.logger.info("阶段4 Phase 1 执行完成!")
            self.logger.info(f"- 总样本数: {len(self.malicious_data)}")
            self.logger.info(f"- Core层: {len(self.layers['core'])} 样本")
            self.logger.info(f"- Inner层: {len(self.layers['inner'])} 样本")
            self.logger.info(f"- Outer层: {len(self.layers['outer'])} 样本")  
            self.logger.info(f"- Edge层: {len(self.layers['edge'])} 样本")
            self.logger.info("="*50)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Phase 1执行失败: {str(e)}")
            return False

def main():
    """主函数"""
    # 设置日志
    logger = setup_logger("batch4_phase1", log_level=logging.INFO)
    
    try:
        # 创建分析器
        analyzer = Batch4CentroidAnalyzer(logger)
        
        # 执行Phase 1
        success = analyzer.run_phase1_complete()
        
        if success:
            logger.info("阶段4 Phase 1 successfully completed!")
            return 0
        else:
            logger.error("阶段4 Phase 1 failed!")
            return 1
            
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())