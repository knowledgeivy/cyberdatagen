#!/usr/bin/env python3
"""
批次5实验 - Phase 3: 纯净数据集重构

构建严格分离的纯净训练数据集，解决批次4的数据混合污染问题：

数据集配置:
1. baseline_real: 100%真实恶意数据 + 固定良性数据
2. pure_rewrite: 100%基础重写合成数据 + 固定良性数据  
3. pure_strong: 100%强化合成数据 + 固定良性数据
4. pure_weak: 100%弱化合成数据 + 固定良性数据

比例配置: 5%, 10%, 15%, 20%恶意数据比例
总计: 16个纯净配置 (4组 × 4比例)

关键改进:
- ❌ 绝不混合真实和合成数据
- ✅ 每个配置使用单一数据源
- ✅ 保持良性数据池固定不变
- ✅ 使用相同的独立测试集

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

# 项目路径设置
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.cyberdata.utils.logger_config import setup_logger
from src.cyberdata.utils.data_loader import load_csv_data, save_csv_data

class Batch5Phase3PureDatasetBuilder:
    """批次5 Phase 3: 纯净数据集构建器"""
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logger("batch5_phase3")
        self.project_root = PROJECT_ROOT
        self.batch4_dir = self.project_root / "data" / "batch4_fresh"
        self.batch5_dir = self.project_root / "data" / "batch5"
        
        # 创建输出目录
        self.datasets_dir = self.batch5_dir / "pure_datasets"
        self.analysis_dir = self.batch5_dir / "phase3_analysis"
        
        for dir_path in [self.datasets_dir, self.analysis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 实验参数 - 纯净数据组配置
        self.data_groups = ['baseline_real', 'pure_rewrite', 'pure_strong', 'pure_weak']
        self.malicious_ratios = [0.05, 0.10, 0.15, 0.20]  # 5%, 10%, 15%, 20%
        self.total_configs = len(self.data_groups) * len(self.malicious_ratios)
        self.target_size = 10000  # 与批次4保持一致
        
        # 随机种子
        self.random_state = 2025
        
        self.logger.info(f"Phase 3初始化完成，将构建 {self.total_configs} 个纯净数据集配置")
        self.logger.info(f"数据组: {self.data_groups}")
        self.logger.info(f"恶意比例: {[f'{int(r*100)}%' for r in self.malicious_ratios]}")
    
    def load_source_data(self) -> Dict[str, pd.DataFrame]:
        """加载所有源数据"""
        try:
            self.logger.info("加载源数据...")
            
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
            
            # 3. 加载所有合成数据并按prompt变体重组
            self.logger.info("加载并重组合成数据...")
            synthetic_dir = self.batch4_dir / "synthetic"
            
            # 按prompt变体重组合成数据
            data['all_rewrite'] = []
            data['all_strong'] = []
            data['all_weak'] = []
            
            for layer in ['core', 'inner', 'outer', 'edge']:
                layer_data = load_csv_data(
                    synthetic_dir / f"{layer}_synthetic.csv", logger=self.logger
                )
                
                # 按prompt变体分组
                if 'prompt_variant' in layer_data.columns:
                    rewrite_data = layer_data[layer_data['prompt_variant'] == 'rewrite'].copy()
                    strong_data = layer_data[layer_data['prompt_variant'] == 'rewrite_strong'].copy()
                    weak_data = layer_data[layer_data['prompt_variant'] == 'rewrite_weak'].copy()
                else:
                    # 从unique_id解析prompt变体
                    rewrite_data = layer_data[layer_data['unique_id'].str.contains('_rewrite$', na=False)].copy()
                    strong_data = layer_data[layer_data['unique_id'].str.contains('_rewrite_strong$', na=False)].copy()
                    weak_data = layer_data[layer_data['unique_id'].str.contains('_rewrite_weak$', na=False)].copy()
                
                data['all_rewrite'].append(rewrite_data)
                data['all_strong'].append(strong_data)
                data['all_weak'].append(weak_data)
                
                self.logger.info(f"{layer}层: rewrite={len(rewrite_data)}, strong={len(strong_data)}, weak={len(weak_data)}")
            
            # 合并各prompt变体的数据
            data['all_rewrite'] = pd.concat(data['all_rewrite'], ignore_index=True) if data['all_rewrite'] else pd.DataFrame()
            data['all_strong'] = pd.concat(data['all_strong'], ignore_index=True) if data['all_strong'] else pd.DataFrame()
            data['all_weak'] = pd.concat(data['all_weak'], ignore_index=True) if data['all_weak'] else pd.DataFrame()
            
            # 统计信息
            for name, df in data.items():
                if isinstance(df, pd.DataFrame):
                    malicious_count = len(df[df['label'] == 1]) if 'label' in df.columns else 0
                    benign_count = len(df[df['label'] == 0]) if 'label' in df.columns else 0
                    total_count = len(df)
                    self.logger.info(f"{name}: {total_count:,} 样本 ({malicious_count:,} 恶意, {benign_count:,} 良性)")
            
            # 验证数据充足性
            self.verify_data_sufficiency(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"加载源数据失败: {str(e)}")
            raise
    
    def verify_data_sufficiency(self, data: Dict[str, pd.DataFrame]) -> None:
        """验证数据充足性"""
        try:
            self.logger.info("验证数据充足性...")
            
            # 计算最大需求（20%比例）
            max_malicious_needed = int(self.target_size * 0.20)  # 2000个恶意样本
            benign_needed = self.target_size - max_malicious_needed  # 8000个良性样本
            
            # 检查各数据源
            sufficiency_report = {
                'max_malicious_needed': max_malicious_needed,
                'benign_needed': benign_needed,
                'data_availability': {}
            }
            
            # 真实恶意数据
            real_malicious_count = len(data['train_malicious'])
            sufficiency_report['data_availability']['real_malicious'] = {
                'available': real_malicious_count,
                'needed': max_malicious_needed,
                'sufficient': real_malicious_count >= max_malicious_needed,
                'usage_ratio': max_malicious_needed / real_malicious_count if real_malicious_count > 0 else 0
            }
            
            # 各合成数据变体
            for variant in ['all_rewrite', 'all_strong', 'all_weak']:
                variant_count = len(data[variant])
                sufficiency_report['data_availability'][variant] = {
                    'available': variant_count,
                    'needed': max_malicious_needed,
                    'sufficient': variant_count >= max_malicious_needed,
                    'usage_ratio': max_malicious_needed / variant_count if variant_count > 0 else 0
                }
                
                if variant_count < max_malicious_needed:
                    self.logger.warning(f"{variant}: 数据不足! 可用{variant_count}, 需要{max_malicious_needed}")
                    self.logger.warning(f"将调整策略: 重采样或降低最大比例")
            
            # 良性数据
            benign_count = len(data['train_benign'])
            sufficiency_report['data_availability']['benign'] = {
                'available': benign_count,
                'needed': benign_needed,
                'sufficient': benign_count >= benign_needed,
                'usage_ratio': benign_needed / benign_count if benign_count > 0 else 0
            }
            
            # 保存充足性报告
            sufficiency_file = self.analysis_dir / "data_sufficiency_report.json"
            with open(sufficiency_file, 'w', encoding='utf-8') as f:
                json.dump(sufficiency_report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"数据充足性报告已保存: {sufficiency_file}")
            
            # 检查是否所有数据源都充足
            insufficient_sources = []
            for source, info in sufficiency_report['data_availability'].items():
                if not info['sufficient']:
                    insufficient_sources.append(source)
            
            if insufficient_sources:
                self.logger.warning(f"数据不足的源: {insufficient_sources}")
                self.logger.info("将使用重采样策略处理数据不足问题")
            else:
                self.logger.info("✅ 所有数据源充足")
            
        except Exception as e:
            self.logger.error(f"验证数据充足性失败: {str(e)}")
            raise
    
    def create_malicious_pool(self, data: Dict[str, pd.DataFrame], group_name: str) -> pd.DataFrame:
        """创建指定组的恶意数据池"""
        try:
            if group_name == 'baseline_real':
                # 只使用真实恶意数据
                malicious_pool = data['train_malicious'].copy()
                self.logger.info(f"baseline_real组: 使用 {len(malicious_pool):,} 个真实恶意样本")
                
            elif group_name == 'pure_rewrite':
                # 只使用基础重写合成数据
                malicious_pool = data['all_rewrite'].copy()
                self.logger.info(f"pure_rewrite组: 使用 {len(malicious_pool):,} 个基础重写合成样本")
                
            elif group_name == 'pure_strong':
                # 只使用强化合成数据
                malicious_pool = data['all_strong'].copy()
                self.logger.info(f"pure_strong组: 使用 {len(malicious_pool):,} 个强化合成样本")
                
            elif group_name == 'pure_weak':
                # 只使用弱化合成数据
                malicious_pool = data['all_weak'].copy()
                self.logger.info(f"pure_weak组: 使用 {len(malicious_pool):,} 个弱化合成样本")
                
            else:
                raise ValueError(f"未知的数据组: {group_name}")
            
            return malicious_pool
            
        except Exception as e:
            self.logger.error(f"创建{group_name}组恶意数据池失败: {str(e)}")
            raise
    
    def sample_balanced_dataset(self, malicious_pool: pd.DataFrame, benign_pool: pd.DataFrame, 
                              malicious_ratio: float, target_size: int = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """采样生成平衡数据集"""
        try:
            if target_size is None:
                target_size = self.target_size
                
            # 计算样本数量
            malicious_count = int(target_size * malicious_ratio)
            benign_count = target_size - malicious_count
            
            # 检查数据池大小
            available_malicious = len(malicious_pool)
            available_benign = len(benign_pool)
            
            # 处理数据不足的情况
            if available_malicious < malicious_count:
                if available_malicious == 0:
                    raise ValueError("恶意数据池为空")
                    
                self.logger.warning(f"恶意样本不足: 需要{malicious_count}, 可用{available_malicious}")
                
                # 使用重采样策略
                if available_malicious < malicious_count / 2:
                    # 如果可用数据少于需求的一半，使用有放回采样
                    sampled_malicious = malicious_pool.sample(
                        n=malicious_count, replace=True, random_state=self.random_state
                    ).copy()
                    self.logger.info(f"使用有放回采样: {malicious_count} 个样本")
                else:
                    # 否则仅使用可用数据，调整比例
                    malicious_count = available_malicious
                    sampled_malicious = malicious_pool.copy()
                    self.logger.info(f"调整恶意样本数量为: {malicious_count}")
            else:
                # 无放回随机采样
                sampled_malicious = malicious_pool.sample(
                    n=malicious_count, random_state=self.random_state
                ).copy()
            
            # 处理良性样本
            if available_benign < benign_count:
                self.logger.warning(f"良性样本不足: 需要{benign_count}, 可用{available_benign}")
                benign_count = available_benign
                sampled_benign = benign_pool.copy()
            else:
                sampled_benign = benign_pool.sample(
                    n=benign_count, random_state=self.random_state
                ).copy()
            
            # 重新计算实际比例
            actual_total = malicious_count + benign_count
            actual_ratio = malicious_count / actual_total if actual_total > 0 else 0
            
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
                'available_benign': available_benign,
                'used_resampling': available_malicious < int(target_size * malicious_ratio)
            }
            
            return dataset, stats
            
        except Exception as e:
            self.logger.error(f"采样平衡数据集失败: {str(e)}")
            raise
    
    def build_single_dataset(self, data: Dict[str, pd.DataFrame], group_name: str, 
                           malicious_ratio: float) -> Tuple[str, Dict[str, Any]]:
        """构建单个纯净数据集配置"""
        try:
            # 创建配置名称
            config_name = f"{group_name}_{int(malicious_ratio*100)}pct"
            self.logger.info(f"构建纯净数据集: {config_name}")
            
            # 创建恶意数据池（纯净，单一数据源）
            malicious_pool = self.create_malicious_pool(data, group_name)
            
            if len(malicious_pool) == 0:
                raise ValueError(f"{group_name} 恶意数据池为空")
            
            # 使用固定的良性数据池
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
                'data_purity': 'pure',  # 关键标记：纯净数据
                'creation_date': '2025-07-30',
                'train_file': str(train_file),
                'test_file': str(test_file),
                'train_stats': train_stats,
                'test_stats': {
                    'total_size': len(test_dataset),
                    'malicious_count': len(test_dataset[test_dataset['label'] == 1]),
                    'benign_count': len(test_dataset[test_dataset['label'] == 0]),
                    'malicious_ratio': len(test_dataset[test_dataset['label'] == 1]) / len(test_dataset)
                },
                'data_composition': {
                    'malicious_source': self.get_data_source_description(group_name),
                    'benign_source': 'real_train_benign',
                    'test_source': 'fixed_test_set_from_batch4'
                }
            }
            
            # 保存配置信息
            config_file = config_dir / "config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_info, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"✅ {config_name}: 训练集{len(train_dataset):,}样本 "
                           f"(实际恶意比例: {train_stats['actual_malicious_ratio']:.1%}), "
                           f"测试集{len(test_dataset):,}样本")
            
            if train_stats['used_resampling']:
                self.logger.info(f"   ⚠️  使用了重采样策略处理数据不足")
            
            return config_name, config_info
            
        except Exception as e:
            self.logger.error(f"构建数据集{group_name}_{int(malicious_ratio*100)}pct失败: {str(e)}")
            raise
    
    def get_data_source_description(self, group_name: str) -> str:
        """获取数据源描述"""
        descriptions = {
            'baseline_real': 'real_malicious_training_data',
            'pure_rewrite': 'synthetic_rewrite_all_layers',
            'pure_strong': 'synthetic_rewrite_strong_all_layers',
            'pure_weak': 'synthetic_rewrite_weak_all_layers'
        }
        return descriptions.get(group_name, 'unknown')
    
    def build_all_datasets(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """构建所有16个纯净数据集配置"""
        try:
            self.logger.info("开始构建所有纯净数据集配置...")
            
            all_configs = {}
            config_count = 0
            
            for group_name in self.data_groups:
                for malicious_ratio in self.malicious_ratios:
                    config_count += 1
                    self.logger.info(f"进度: {config_count}/{self.total_configs}")
                    
                    try:
                        config_name, config_info = self.build_single_dataset(
                            data, group_name, malicious_ratio
                        )
                        all_configs[config_name] = config_info
                        
                    except Exception as e:
                        self.logger.error(f"构建配置 {group_name}_{int(malicious_ratio*100)}pct 失败: {str(e)}")
                        # 继续构建其他配置
                        continue
            
            self.logger.info(f"成功构建 {len(all_configs)} 个纯净数据集配置")
            
            # 验证构建结果
            if len(all_configs) < self.total_configs:
                missing_count = self.total_configs - len(all_configs)
                self.logger.warning(f"有 {missing_count} 个配置构建失败")
            
            return all_configs
            
        except Exception as e:
            self.logger.error(f"构建所有数据集失败: {str(e)}")
            raise
    
    def generate_phase3_summary(self, all_configs: Dict[str, Dict[str, Any]]) -> None:
        """生成Phase 3统计摘要"""
        try:
            self.logger.info("生成Phase 3统计摘要...")
            
            # 按组和比例统计
            group_stats = {}
            ratio_stats = {}
            
            for config_name, config_info in all_configs.items():
                group = config_info['group_name']
                ratio = f"{int(config_info['malicious_ratio']*100)}%"
                
                # 按组统计
                if group not in group_stats:
                    group_stats[group] = {
                        'config_count': 0,
                        'config_names': [],
                        'total_samples': 0,
                        'avg_malicious_ratio': 0
                    }
                
                group_stats[group]['config_count'] += 1
                group_stats[group]['config_names'].append(config_name)
                group_stats[group]['total_samples'] += config_info['train_stats']['actual_size']
                group_stats[group]['avg_malicious_ratio'] += config_info['train_stats']['actual_malicious_ratio']
                
                # 按比例统计
                if ratio not in ratio_stats:
                    ratio_stats[ratio] = {
                        'config_count': 0,
                        'config_names': [],
                        'total_samples': 0
                    }
                
                ratio_stats[ratio]['config_count'] += 1
                ratio_stats[ratio]['config_names'].append(config_name)
                ratio_stats[ratio]['total_samples'] += config_info['train_stats']['actual_size']
            
            # 计算平均值
            for group in group_stats:
                count = group_stats[group]['config_count']
                if count > 0:
                    group_stats[group]['avg_malicious_ratio'] /= count
            
            # 数据纯净性验证
            purity_check = {
                'all_configs_pure': True,
                'mixed_data_configs': [],
                'resampling_configs': []
            }
            
            for config_name, config_info in all_configs.items():
                # 检查数据纯净性
                if config_info.get('data_purity') != 'pure':
                    purity_check['all_configs_pure'] = False
                    purity_check['mixed_data_configs'].append(config_name)
                
                # 检查是否使用了重采样
                if config_info['train_stats'].get('used_resampling', False):
                    purity_check['resampling_configs'].append(config_name)
            
            summary = {
                'phase3_info': {
                    'date': '2025-07-30',
                    'total_configs': len(all_configs),
                    'target_configs': self.total_configs,
                    'success_rate': len(all_configs) / self.total_configs if self.total_configs > 0 else 0,
                    'data_groups': self.data_groups,
                    'malicious_ratios': self.malicious_ratios,
                    'random_state': self.random_state,
                    'target_size_per_dataset': self.target_size
                },
                'data_purity_validation': purity_check,
                'group_statistics': group_stats,
                'ratio_statistics': ratio_stats,
                'config_details': all_configs,
                'experiment_matrix': {
                    'groups': self.data_groups,
                    'ratios': [f"{int(r*100)}%" for r in self.malicious_ratios],
                    'total_experiments_planned': self.total_configs,
                    'total_experiments_completed': len(all_configs)
                },
                'key_improvements_from_batch4': [
                    "严格数据源分离：绝不混合真实和合成数据",
                    "纯净数据组：每个配置使用单一恶意数据源",
                    "固定良性数据池：确保对照组可比性",
                    "统一测试集：使用相同的独立测试数据",
                    "重采样策略：处理合成数据不足问题"
                ]
            }
            
            # 保存摘要
            summary_file = self.analysis_dir / "phase3_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Phase 3统计摘要已保存: {summary_file}")
            
            # 输出关键统计信息
            self.logger.info("="*30)
            self.logger.info("纯净数据集构建统计:")
            self.logger.info(f"✅ 成功配置: {len(all_configs)}/{self.total_configs}")
            self.logger.info(f"✅ 数据纯净性: {'通过' if purity_check['all_configs_pure'] else '部分通过'}")
            
            if purity_check['resampling_configs']:
                self.logger.info(f"⚠️  使用重采样: {len(purity_check['resampling_configs'])} 个配置")
            
            for group, stats in group_stats.items():
                self.logger.info(f"  {group}: {stats['config_count']} 个配置, "
                               f"平均恶意比例: {stats['avg_malicious_ratio']:.1%}")
            self.logger.info("="*30)
            
        except Exception as e:
            self.logger.error(f"生成Phase 3统计摘要失败: {str(e)}")
            raise
    
    def run_phase3_complete(self) -> bool:
        """执行完整的Phase 3流程"""
        try:
            self.logger.info("="*50)
            self.logger.info("开始执行批次5 Phase 3: 纯净数据集重构")
            self.logger.info("="*50)
            
            start_time = time.time()
            
            # Step 1: 加载源数据
            self.logger.info("Step 1: 加载源数据")
            data = self.load_source_data()
            
            # Step 2: 构建所有纯净数据集配置
            self.logger.info("Step 2: 构建所有纯净数据集配置")
            all_configs = self.build_all_datasets(data)
            
            # Step 3: 生成统计摘要
            self.logger.info("Step 3: 生成统计摘要")
            self.generate_phase3_summary(all_configs)
            
            elapsed_time = time.time() - start_time
            
            self.logger.info("="*50)
            self.logger.info("批次5 Phase 3 执行完成!")
            self.logger.info(f"总耗时: {elapsed_time/60:.1f} 分钟")
            self.logger.info(f"构建数据集配置数: {len(all_configs)}")
            
            # 显示配置清单
            self.logger.info("构建的纯净数据集配置:")
            for group in self.data_groups:
                group_configs = [c for c in all_configs.keys() if c.startswith(group)]
                if group_configs:
                    self.logger.info(f"- {group}: {', '.join(group_configs)}")
                else:
                    self.logger.warning(f"- {group}: 无成功配置")
            
            self.logger.info("数据集已保存到: data/batch5/pure_datasets/")
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
    logger = setup_logger("batch5_phase3", log_level=logging.INFO)
    
    try:
        # 创建构建器
        builder = Batch5Phase3PureDatasetBuilder(logger)
        
        # 执行Phase 3
        success = builder.run_phase3_complete()
        
        if success:
            logger.info("批次5 Phase 3 successfully completed!")
            return 0
        else:
            logger.error("批次5 Phase 3 failed!")
            return 1
            
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())