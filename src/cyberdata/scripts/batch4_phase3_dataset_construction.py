#!/usr/bin/env python3
"""
批次4实验 - Phase 3: 不平衡数据集构建

基于真实数据和Phase 2生成的合成数据，构建20个不平衡数据集配置：
- 5个数据组: real_only, core, inner, outer, edge
- 4个恶意比例: 5%, 10%, 15%, 20%
- 总共20个训练集配置，使用相同的独立测试集

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

# 项目路径设置
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.cyberdata.utils.logger_config import setup_logger
from src.cyberdata.utils.data_loader import load_csv_data, save_csv_data

class Batch4Phase3DatasetBuilder:
    """批次4 Phase 3: 不平衡数据集构建器"""
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logger("batch4_phase3")
        self.project_root = PROJECT_ROOT
        self.batch4_dir = self.project_root / "data" / "batch4_fresh"
        
        # 创建输出目录
        self.datasets_dir = self.batch4_dir / "datasets"
        self.analysis_dir = self.batch4_dir / "phase3_analysis"
        
        for dir_path in [self.datasets_dir, self.analysis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 实验参数
        self.data_groups = ['real_only', 'core', 'inner', 'outer', 'edge']
        self.malicious_ratios = [0.05, 0.10, 0.15, 0.20]  # 5%, 10%, 15%, 20%
        self.total_configs = len(self.data_groups) * len(self.malicious_ratios)
        
        # 随机种子
        self.random_state = 2025
        
        self.logger.info(f"Phase 3初始化完成，将构建 {self.total_configs} 个数据集配置")
    
    def load_base_data(self) -> Dict[str, pd.DataFrame]:
        """加载所有基础数据"""
        try:
            self.logger.info("加载基础数据...")
            
            data = {}
            
            # 加载真实训练数据
            data['train_malicious'] = load_csv_data(
                self.batch4_dir / "train_malicious.csv", logger=self.logger
            )
            data['train_benign'] = load_csv_data(
                self.batch4_dir / "train_benign.csv", logger=self.logger
            )
            
            # 加载固定测试集
            data['test_set'] = load_csv_data(
                self.batch4_dir / "raw_test_set.csv", logger=self.logger
            )
            
            # 加载合成数据
            synthetic_dir = self.batch4_dir / "synthetic"
            for layer in ['core', 'inner', 'outer', 'edge']:
                data[f'{layer}_synthetic'] = load_csv_data(
                    synthetic_dir / f"{layer}_synthetic.csv", logger=self.logger
                )
            
            # 记录数据统计
            for name, df in data.items():
                malicious_count = len(df[df['label'] == 1]) if 'label' in df.columns else 0
                benign_count = len(df[df['label'] == 0]) if 'label' in df.columns else 0
                total_count = len(df)
                self.logger.info(f"{name}: {total_count} 总样本 ({malicious_count} 恶意, {benign_count} 良性)")
            
            return data
            
        except Exception as e:
            self.logger.error(f"加载基础数据失败: {str(e)}")
            raise
    
    def create_malicious_pool(self, data: Dict[str, pd.DataFrame], group_name: str) -> pd.DataFrame:
        """创建指定组的恶意数据池"""
        try:
            if group_name == 'real_only':
                # 只使用真实恶意数据
                malicious_pool = data['train_malicious'].copy()
                self.logger.info(f"real_only组: 使用 {len(malicious_pool)} 个真实恶意样本")
                
            elif group_name in ['core', 'inner', 'outer', 'edge']:
                # 真实数据 + 指定层合成数据
                real_malicious = data['train_malicious'].copy()
                synthetic_malicious = data[f'{group_name}_synthetic'].copy()
                
                # 合并数据
                malicious_pool = pd.concat([real_malicious, synthetic_malicious], ignore_index=True)
                
                self.logger.info(f"{group_name}组: {len(real_malicious)} 真实 + {len(synthetic_malicious)} 合成 = {len(malicious_pool)} 总恶意样本")
                
            else:
                raise ValueError(f"未知的数据组: {group_name}")
            
            return malicious_pool
            
        except Exception as e:
            self.logger.error(f"创建{group_name}组恶意数据池失败: {str(e)}")
            raise
    
    def sample_balanced_dataset(self, malicious_pool: pd.DataFrame, benign_pool: pd.DataFrame, 
                              malicious_ratio: float, target_size: int = 10000) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """采样生成平衡数据集"""
        try:
            # 计算样本数量
            malicious_count = int(target_size * malicious_ratio)
            benign_count = target_size - malicious_count
            
            # 检查数据池大小
            available_malicious = len(malicious_pool)
            available_benign = len(benign_pool)
            
            if available_malicious < malicious_count:
                self.logger.warning(f"恶意样本不足: 需要{malicious_count}, 可用{available_malicious}")
                malicious_count = available_malicious
                
            if available_benign < benign_count:
                self.logger.warning(f"良性样本不足: 需要{benign_count}, 可用{available_benign}")
                benign_count = available_benign
            
            # 重新计算实际比例
            actual_total = malicious_count + benign_count
            actual_ratio = malicious_count / actual_total if actual_total > 0 else 0
            
            # 随机采样
            sampled_malicious = malicious_pool.sample(
                n=malicious_count, random_state=self.random_state
            ).copy()
            
            sampled_benign = benign_pool.sample(
                n=benign_count, random_state=self.random_state
            ).copy()
            
            # 合并数据集
            dataset = pd.concat([sampled_malicious, sampled_benign], ignore_index=True)
            
            # 打乱顺序
            dataset = dataset.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
            
            # 统计信息
            stats = {
                'target_size': target_size,
                'actual_size': len(dataset),
                'target_malicious_ratio': malicious_ratio,
                'actual_malicious_ratio': actual_ratio,
                'malicious_count': malicious_count,
                'benign_count': benign_count,
                'available_malicious': available_malicious,
                'available_benign': available_benign
            }
            
            return dataset, stats
            
        except Exception as e:
            self.logger.error(f"采样平衡数据集失败: {str(e)}")
            raise
    
    def build_single_dataset(self, data: Dict[str, pd.DataFrame], group_name: str, 
                           malicious_ratio: float) -> Tuple[str, Dict[str, Any]]:
        """构建单个数据集配置"""
        try:
            # 创建配置名称
            config_name = f"{group_name}_{int(malicious_ratio*100)}pct"
            self.logger.info(f"构建数据集: {config_name}")
            
            # 创建恶意数据池
            malicious_pool = self.create_malicious_pool(data, group_name)
            benign_pool = data['train_benign'].copy()
            
            # 采样训练集
            train_dataset, train_stats = self.sample_balanced_dataset(
                malicious_pool, benign_pool, malicious_ratio
            )
            
            # 使用固定测试集
            test_dataset = data['test_set'].copy()
            
            # 创建配置目录
            config_dir = self.datasets_dir / config_name
            config_dir.mkdir(exist_ok=True)
            
            # 保存数据集
            train_file = config_dir / "train.csv"
            test_file = config_dir / "test.csv"
            
            save_csv_data(train_dataset, train_file, compress=False, logger=self.logger)
            save_csv_data(test_dataset, test_file, compress=False, logger=self.logger)
            
            # 生成配置信息
            config_info = {
                'config_name': config_name,
                'group_name': group_name,
                'malicious_ratio': malicious_ratio,
                'creation_date': '2025-07-24',
                'train_file': str(train_file),
                'test_file': str(test_file),
                'train_stats': train_stats,
                'test_stats': {
                    'total_size': len(test_dataset),
                    'malicious_count': len(test_dataset[test_dataset['label'] == 1]),
                    'benign_count': len(test_dataset[test_dataset['label'] == 0]),
                    'malicious_ratio': len(test_dataset[test_dataset['label'] == 1]) / len(test_dataset)
                }
            }
            
            # 保存配置信息
            config_file = config_dir / "config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_info, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"✅ {config_name}: 训练集{len(train_dataset)}样本, 测试集{len(test_dataset)}样本")
            
            return config_name, config_info
            
        except Exception as e:
            self.logger.error(f"构建数据集{group_name}_{int(malicious_ratio*100)}pct失败: {str(e)}")
            raise
    
    def build_all_datasets(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """构建所有20个数据集配置"""
        try:
            self.logger.info("开始构建所有数据集配置...")
            
            all_configs = {}
            config_count = 0
            
            for group_name in self.data_groups:
                for malicious_ratio in self.malicious_ratios:
                    config_count += 1
                    self.logger.info(f"进度: {config_count}/{self.total_configs}")
                    
                    config_name, config_info = self.build_single_dataset(
                        data, group_name, malicious_ratio
                    )
                    
                    all_configs[config_name] = config_info
            
            self.logger.info(f"成功构建 {len(all_configs)} 个数据集配置")
            return all_configs
            
        except Exception as e:
            self.logger.error(f"构建所有数据集失败: {str(e)}")
            raise
    
    def generate_phase3_summary(self, all_configs: Dict[str, Dict[str, Any]]) -> None:
        """生成Phase 3统计摘要"""
        try:
            self.logger.info("生成Phase 3统计摘要...")
            
            summary = {
                'phase3_info': {
                    'date': '2025-07-24',
                    'total_configs': len(all_configs),
                    'data_groups': self.data_groups,
                    'malicious_ratios': self.malicious_ratios,
                    'random_state': self.random_state
                },
                'group_statistics': {},
                'ratio_statistics': {},
                'config_details': all_configs,
                'experiment_matrix': {
                    'groups': self.data_groups,
                    'ratios': [f"{int(r*100)}%" for r in self.malicious_ratios],
                    'total_experiments': self.total_configs
                }
            }
            
            # 按组统计
            for group in self.data_groups:
                group_configs = [c for c in all_configs.keys() if c.startswith(group)]
                summary['group_statistics'][group] = {
                    'config_count': len(group_configs),
                    'config_names': group_configs
                }
            
            # 按比例统计
            for ratio in self.malicious_ratios:
                ratio_str = f"{int(ratio*100)}pct"
                ratio_configs = [c for c in all_configs.keys() if c.endswith(ratio_str)]
                summary['ratio_statistics'][ratio_str] = {
                    'config_count': len(ratio_configs),
                    'config_names': ratio_configs
                }
            
            # 保存摘要
            summary_file = self.analysis_dir / "phase3_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Phase 3统计摘要已保存: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"生成Phase 3统计摘要失败: {str(e)}")
            raise
    
    def run_phase3_complete(self) -> bool:
        """执行完整的Phase 3流程"""
        try:
            self.logger.info("="*50)
            self.logger.info("开始执行批次4 Phase 3: 不平衡数据集构建")
            self.logger.info("="*50)
            
            start_time = time.time()
            
            # Step 1: 加载基础数据
            self.logger.info("Step 1: 加载基础数据")
            data = self.load_base_data()
            
            # Step 2: 构建所有数据集配置
            self.logger.info("Step 2: 构建所有数据集配置")
            all_configs = self.build_all_datasets(data)
            
            # Step 3: 生成统计摘要
            self.logger.info("Step 3: 生成统计摘要")
            self.generate_phase3_summary(all_configs)
            
            elapsed_time = time.time() - start_time
            
            self.logger.info("="*50)
            self.logger.info("批次4 Phase 3 执行完成!")
            self.logger.info(f"总耗时: {elapsed_time/60:.1f} 分钟")
            self.logger.info(f"构建数据集配置数: {len(all_configs)}")
            
            # 显示配置清单
            self.logger.info("构建的数据集配置:")
            for group in self.data_groups:
                group_configs = [c for c in all_configs.keys() if c.startswith(group)]
                self.logger.info(f"- {group}: {', '.join(group_configs)}")
            
            self.logger.info("数据集已保存到: data/batch4_fresh/datasets/")
            self.logger.info("="*50)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Phase 3执行失败: {str(e)}")
            return False

def main():
    """主函数"""
    # 设置随机种子
    random.seed(2025)
    np.random.seed(2025)
    
    # 设置日志
    logger = setup_logger("batch4_phase3", log_level=logging.INFO)
    
    try:
        # 创建构建器
        builder = Batch4Phase3DatasetBuilder(logger)
        
        # 执行Phase 3
        success = builder.run_phase3_complete()
        
        if success:
            logger.info("批次4 Phase 3 successfully completed!")
            return 0
        else:
            logger.error("批次4 Phase 3 failed!")
            return 1
            
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())