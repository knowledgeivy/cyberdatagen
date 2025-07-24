#!/usr/bin/env python3
"""
阶段4实验 - Phase 3: 数据集构建和baseline对照组

构建5个数据组(real_only + 4个synthetic层) × 4个不平衡比例的训练数据集

作者: Claude
创建时间: 2025-07-24
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import random
from typing import Dict, List, Any, Tuple
import logging

# 项目路径设置
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.cyberdata.utils.logger_config import setup_logger

class Batch4DatasetBuilder:
    """阶段4数据集构建器"""
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logger("batch4_phase3")
        self.project_root = PROJECT_ROOT
        self.batch4_dir = self.project_root / "data" / "batch4"
        self.datasets_dir = self.batch4_dir / "datasets"
        
        # 确保输出目录存在
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # 实验配置
        self.ratios = [0.05, 0.10, 0.15, 0.20]  # 恶意数据比例
        self.data_groups = ['real_only', 'core', 'inner', 'outer', 'edge']  # 5个数据组
        self.total_samples = 10000  # 每个数据集的总样本数
        
        # 数据存储
        self.real_malicious_data = None
        self.real_benign_data = None
        self.synthetic_data = {}
        self.test_set = None
        
    def load_fixed_datasets(self) -> bool:
        """加载固定的数据集（确保可比性）"""
        try:
            self.logger.info("加载固定的真实数据和测试集...")
            
            # 1. 加载真实恶意数据
            malicious_file = self.project_root / "data" / "batch1" / "base_samples" / "malicious_5k.csv"
            self.real_malicious_data = pd.read_csv(malicious_file)
            self.logger.info(f"加载真实恶意数据: {len(self.real_malicious_data)} 个样本")
            
            # 2. 加载真实良性数据
            benign_file = self.project_root / "data" / "batch1" / "base_samples" / "benign_5k.csv"
            self.real_benign_data = pd.read_csv(benign_file)
            self.logger.info(f"加载真实良性数据: {len(self.real_benign_data)} 个样本")
            
            # 3. 加载固定测试集
            test_file = self.project_root / "data" / "batch1" / "ml_results" / "fixed_test_set.csv"
            self.test_set = pd.read_csv(test_file)
            self.logger.info(f"加载固定测试集: {len(self.test_set)} 个样本")
            
            return True
            
        except Exception as e:
            self.logger.error(f"加载固定数据集失败: {str(e)}")
            return False
    
    def load_synthetic_data(self) -> bool:
        """加载各层合成数据"""
        try:
            self.logger.info("加载各层合成数据...")
            
            synthetic_dir = self.batch4_dir / "synthetic"
            
            for layer in ['core', 'inner', 'outer', 'edge']:
                layer_file = synthetic_dir / f"{layer}_layer_synthetic.csv"
                if layer_file.exists():
                    self.synthetic_data[layer] = pd.read_csv(layer_file)
                    self.logger.info(f"加载{layer}层合成数据: {len(self.synthetic_data[layer])} 个样本")
                else:
                    self.logger.error(f"未找到{layer}层合成数据文件: {layer_file}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"加载合成数据失败: {str(e)}")
            return False
    
    def build_real_only_dataset(self, ratio: float) -> pd.DataFrame:
        """构建纯真实数据集"""
        try:
            n_malicious = int(self.total_samples * ratio)
            n_benign = self.total_samples - n_malicious
            
            # 从真实数据中采样
            if len(self.real_malicious_data) >= n_malicious:
                sampled_malicious = self.real_malicious_data.sample(n=n_malicious, random_state=42)
            else:
                # 如果数据不足，使用重复采样
                sampled_malicious = self.real_malicious_data.sample(n=n_malicious, replace=True, random_state=42)
            
            if len(self.real_benign_data) >= n_benign:
                sampled_benign = self.real_benign_data.sample(n=n_benign, random_state=42)
            else:
                sampled_benign = self.real_benign_data.sample(n=n_benign, replace=True, random_state=42)
            
            # 合并数据集
            dataset = pd.concat([sampled_malicious, sampled_benign], ignore_index=True)
            
            # 随机打乱
            dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
            
            self.logger.info(f"构建real_only数据集 (比例{ratio:.0%}): 恶意{n_malicious}, 良性{n_benign}")
            
            return dataset
            
        except Exception as e:
            self.logger.error(f"构建real_only数据集失败: {str(e)}")
            raise
    
    def build_synthetic_dataset(self, layer: str, ratio: float) -> pd.DataFrame:
        """构建包含合成数据的数据集"""
        try:
            n_malicious = int(self.total_samples * ratio)
            n_benign = self.total_samples - n_malicious
            
            # 恶意样本：一半真实，一半合成
            n_real_malicious = n_malicious // 2
            n_synthetic_malicious = n_malicious - n_real_malicious
            
            # 采样真实恶意数据
            if len(self.real_malicious_data) >= n_real_malicious:
                sampled_real_malicious = self.real_malicious_data.sample(n=n_real_malicious, random_state=42)
            else:
                sampled_real_malicious = self.real_malicious_data.sample(n=n_real_malicious, replace=True, random_state=42)
            
            # 采样合成恶意数据
            layer_synthetic = self.synthetic_data[layer]
            if len(layer_synthetic) >= n_synthetic_malicious:
                sampled_synthetic = layer_synthetic.sample(n=n_synthetic_malicious, random_state=42)
            else:
                sampled_synthetic = layer_synthetic.sample(n=n_synthetic_malicious, replace=True, random_state=42)
            
            # 采样良性数据
            if len(self.real_benign_data) >= n_benign:
                sampled_benign = self.real_benign_data.sample(n=n_benign, random_state=42)
            else:
                sampled_benign = self.real_benign_data.sample(n=n_benign, replace=True, random_state=42)
            
            # 合并数据集
            dataset = pd.concat([sampled_real_malicious, sampled_synthetic, sampled_benign], ignore_index=True)
            
            # 随机打乱
            dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
            
            self.logger.info(f"构建{layer}层数据集 (比例{ratio:.0%}): "
                           f"真实恶意{n_real_malicious}, 合成恶意{n_synthetic_malicious}, 良性{n_benign}")
            
            return dataset
            
        except Exception as e:
            self.logger.error(f"构建{layer}层数据集失败: {str(e)}")
            raise
    
    def save_dataset_configuration(self, group: str, ratio: float, dataset: pd.DataFrame) -> None:
        """保存数据集配置"""
        try:
            # 创建目录
            config_dir = self.datasets_dir / f"{group}_{ratio:.0%}".replace('%', 'pct')
            config_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存训练集
            train_file = config_dir / "train.csv"
            dataset.to_csv(train_file, index=False, encoding='utf-8')
            
            # 复制测试集
            test_file = config_dir / "test.csv"
            self.test_set.to_csv(test_file, index=False, encoding='utf-8')
            
            # 生成配置元数据
            config_metadata = {
                'experiment_name': f'batch4_{group}_{ratio:.0%}'.replace('%', 'pct'),
                'data_group': group,
                'malicious_ratio': ratio,
                'total_samples': len(dataset),
                'train_samples': len(dataset),
                'test_samples': len(self.test_set),
                'malicious_count': len(dataset[dataset['label'] == 1]),
                'benign_count': len(dataset[dataset['label'] == 0]),
                'creation_date': '2025-07-24',
                'data_sources': {
                    'real_malicious': 'batch1/base_samples/malicious_5k.csv',
                    'real_benign': 'batch1/base_samples/benign_5k.csv',
                    'synthetic': f'batch4/synthetic/{group}_layer_synthetic.csv' if group != 'real_only' else None,
                    'test_set': 'batch1/ml_results/fixed_test_set.csv'
                }
            }
            
            metadata_file = config_dir / "config.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(config_metadata, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"保存数据集配置: {config_dir}")
            
        except Exception as e:
            self.logger.error(f"保存数据集配置失败: {str(e)}")
            raise
    
    def validate_dataset_balance(self, dataset: pd.DataFrame, expected_ratio: float) -> bool:
        """验证数据集平衡性"""
        try:
            total_count = len(dataset)
            malicious_count = len(dataset[dataset['label'] == 1])
            actual_ratio = malicious_count / total_count
            
            # 允许1%的误差
            if abs(actual_ratio - expected_ratio) <= 0.01:
                self.logger.info(f"数据集平衡性验证通过: 预期{expected_ratio:.1%}, 实际{actual_ratio:.1%}")
                return True
            else:
                self.logger.warning(f"数据集平衡性偏差: 预期{expected_ratio:.1%}, 实际{actual_ratio:.1%}")
                return False
                
        except Exception as e:
            self.logger.error(f"验证数据集平衡性失败: {str(e)}")
            return False
    
    def generate_dataset_summary(self) -> None:
        """生成数据集构建总结"""
        try:
            summary = {
                'experiment_info': {
                    'name': 'batch4_dataset_construction',
                    'date': '2025-07-24',
                    'description': '阶段4不平衡数据实验的数据集构建'
                },
                'configuration': {
                    'ratios': self.ratios,
                    'data_groups': self.data_groups,
                    'total_samples_per_dataset': self.total_samples,
                    'total_configurations': len(self.ratios) * len(self.data_groups)
                },
                'data_sources': {
                    'real_malicious_samples': len(self.real_malicious_data),
                    'real_benign_samples': len(self.real_benign_data),
                    'test_samples': len(self.test_set),
                    'synthetic_layers': {layer: len(data) for layer, data in self.synthetic_data.items()}
                },
                'dataset_matrix': {}
            }
            
            # 构建数据集矩阵
            for group in self.data_groups:
                summary['dataset_matrix'][group] = {}
                for ratio in self.ratios:
                    ratio_str = f"{ratio:.0%}".replace('%', 'pct')
                    config_dir = self.datasets_dir / f"{group}_{ratio_str}"
                    
                    if config_dir.exists():
                        summary['dataset_matrix'][group][f"{ratio:.0%}"] = {
                            'status': 'completed',
                            'path': str(config_dir.relative_to(self.project_root))
                        }
                    else:
                        summary['dataset_matrix'][group][f"{ratio:.0%}"] = {
                            'status': 'missing',
                            'path': None
                        }
            
            # 保存总结
            summary_file = self.datasets_dir / "dataset_construction_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"数据集构建总结保存到: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"生成数据集总结失败: {str(e)}")
            raise
    
    def run_phase3_complete(self) -> bool:
        """执行完整的Phase 3流程"""
        try:
            self.logger.info("="*50)
            self.logger.info("开始执行阶段4 Phase 3: 数据集构建和baseline对照组")
            self.logger.info("="*50)
            
            # Step 1: 加载固定数据集
            self.logger.info("Step 1: 加载固定数据集")
            if not self.load_fixed_datasets():
                return False
                
            # Step 2: 加载合成数据
            self.logger.info("Step 2: 加载各层合成数据")
            if not self.load_synthetic_data():
                return False
            
            # Step 3: 构建所有数据集配置
            self.logger.info("Step 3: 构建所有数据集配置")
            total_configs = len(self.data_groups) * len(self.ratios)
            completed_configs = 0
            
            for group in self.data_groups:
                for ratio in self.ratios:
                    self.logger.info(f"构建数据集: {group}, 比例{ratio:.0%}")
                    
                    # 构建数据集
                    if group == 'real_only':
                        dataset = self.build_real_only_dataset(ratio)
                    else:
                        dataset = self.build_synthetic_dataset(group, ratio)
                    
                    # 验证平衡性
                    if self.validate_dataset_balance(dataset, ratio):
                        # 保存数据集
                        self.save_dataset_configuration(group, ratio, dataset)
                        completed_configs += 1
                    else:
                        self.logger.warning(f"数据集{group}_{ratio:.0%}验证失败，但仍然保存")
                        self.save_dataset_configuration(group, ratio, dataset)
                        completed_configs += 1
                    
                    self.logger.info(f"进度: {completed_configs}/{total_configs} 配置完成")
            
            # Step 4: 生成总结报告
            self.logger.info("Step 4: 生成数据集构建总结")
            self.generate_dataset_summary()
            
            self.logger.info("="*50)
            self.logger.info("阶段4 Phase 3 执行完成!")
            self.logger.info(f"成功构建 {completed_configs} 个数据集配置")
            self.logger.info(f"数据集矩阵: {len(self.data_groups)} 数据组 × {len(self.ratios)} 比例")
            for group in self.data_groups:
                group_datasets = [f"{ratio:.0%}" for ratio in self.ratios]
                self.logger.info(f"- {group}: {', '.join(group_datasets)}")
            self.logger.info("="*50)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Phase 3执行失败: {str(e)}")
            return False

def main():
    """主函数"""
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    # 设置日志
    logger = setup_logger("batch4_phase3", log_level=logging.INFO)
    
    try:
        # 创建数据集构建器
        builder = Batch4DatasetBuilder(logger)
        
        # 执行Phase 3
        success = builder.run_phase3_complete()
        
        if success:
            logger.info("阶段4 Phase 3 successfully completed!")
            return 0
        else:
            logger.error("阶段4 Phase 3 failed!")
            return 1
            
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())