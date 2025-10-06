"""
数据集构建模块
负责根据不同的synthetic ratio构建训练和测试数据集
"""

import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from loguru import logger
from sklearn.model_selection import train_test_split

from ..config.config_manager import ExperimentConfig


class DatasetBuilder:
    """数据集构建器"""

    def __init__(self, config: ExperimentConfig):
        """
        初始化构建器

        Args:
            config: 实验配置
        """
        self.config = config
        self.random_state = np.random.RandomState(config.random_seed)

    def load_processed_data(self, processed_data_dir: str) -> Dict[str, Any]:
        """
        加载预处理后的数据

        Args:
            processed_data_dir: 预处理数据目录

        Returns:
            Dict[str, Any]: 包含groups和metadata的字典
        """
        logger.info(f"加载预处理数据: {processed_data_dir}")

        dataset_name = self.config.dataset.lower().replace('-', '')

        # 加载元数据
        metadata_file = os.path.join(processed_data_dir, f"{dataset_name}_metadata.json")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"元数据文件不存在: {metadata_file}")

        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # 加载分组数据
        groups_dir = os.path.join(processed_data_dir, 'groups')
        groups = {}

        for group_name in metadata['groups'].keys():
            group_file = os.path.join(groups_dir, f"{dataset_name}_{group_name}.csv.gz")
            if os.path.exists(group_file):
                groups[group_name] = pd.read_csv(group_file, compression='gzip')
            else:
                logger.warning(f"分组文件不存在: {group_file}")

        # 加载测试集
        test_file = os.path.join(processed_data_dir, f"{dataset_name}_test_set.csv.gz")
        test_set = None
        if os.path.exists(test_file):
            test_set = pd.read_csv(test_file, compression='gzip')

        logger.info(f"加载完成: {len(groups)} 组, 测试集大小: {len(test_set) if test_set is not None else 0}")

        return {
            'groups': groups,
            'metadata': metadata,
            'test_set': test_set
        }

    def load_synthetic_data(self, synthetic_data_dir: str, prompt_name: str, llm_engine: str) -> pd.DataFrame:
        """
        加载synthetic数据

        Args:
            synthetic_data_dir: synthetic数据目录
            prompt_name: prompt名称
            llm_engine: LLM引擎

        Returns:
            pd.DataFrame: synthetic数据
        """
        dataset_name = self.config.dataset.lower().replace('-', '')
        synthetic_file = os.path.join(
            synthetic_data_dir,
            f"{dataset_name}_synthetic_{prompt_name}_{llm_engine}.csv.gz"
        )

        if not os.path.exists(synthetic_file):
            raise FileNotFoundError(f"Synthetic数据文件不存在: {synthetic_file}")

        synthetic_data = pd.read_csv(synthetic_file, compression='gzip')
        logger.info(f"加载synthetic数据: {len(synthetic_data)} 条")

        return synthetic_data

    def create_mixed_training_set(
        self,
        group_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        synthetic_ratio: int,
        strategy: str = "within_group",
        all_groups_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """
        创建混合训练集

        Args:
            group_data: 组数据
            synthetic_data: synthetic数据
            synthetic_ratio: synthetic比例 (0-100)
            strategy: 混合策略 ("within_group", "cross_group", "real_fixed_random_synthetic", "full_random")
            all_groups_data: 所有组数据 (用于full_random策略)

        Returns:
            pd.DataFrame: 混合训练集
        """
        # 获取当前组ID
        current_group_id = group_data['group_id'].iloc[0] if 'group_id' in group_data.columns else None

        # 根据策略选择real spam和synthetic数据
        if strategy == "within_group":
            # Strategy 1: Real Group i + Synthetic Group i
            real_spam_pool = group_data[group_data['label'] == 1].copy()
            real_non_spam = group_data[group_data['label'] == 0].copy()

            if current_group_id is not None:
                available_synthetic = synthetic_data[
                    synthetic_data['group_id'] == current_group_id
                ].copy()
            else:
                available_synthetic = synthetic_data.copy()

        elif strategy == "cross_group":
            # Strategy 2: Real Group i + Synthetic Group j≠i
            real_spam_pool = group_data[group_data['label'] == 1].copy()
            real_non_spam = group_data[group_data['label'] == 0].copy()

            if current_group_id is not None:
                available_synthetic = synthetic_data[
                    synthetic_data['group_id'] != current_group_id
                ].copy()
            else:
                available_synthetic = synthetic_data.copy()

        elif strategy == "real_fixed_random_synthetic":
            # Strategy 3: Real Group i + Random Synthetic (from all groups)
            real_spam_pool = group_data[group_data['label'] == 1].copy()
            real_non_spam = group_data[group_data['label'] == 0].copy()

            # Use all synthetic data randomly
            available_synthetic = synthetic_data.copy()

        elif strategy == "full_random":
            # Strategy 4: Random Real + Random Synthetic
            if all_groups_data is None:
                raise ValueError("all_groups_data required for full_random strategy")

            # Combine all real spam data from all groups
            all_real_spam = []
            all_real_non_spam = []
            for group_name, group_df in all_groups_data.items():
                all_real_spam.append(group_df[group_df['label'] == 1])
                all_real_non_spam.append(group_df[group_df['label'] == 0])

            real_spam_pool = pd.concat(all_real_spam, ignore_index=True)
            # For non-spam, still use current group to maintain test consistency
            real_non_spam = group_data[group_data['label'] == 0].copy()

            # Use all synthetic data randomly
            available_synthetic = synthetic_data.copy()

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # 计算需要的synthetic和real数量
        total_spam_needed = len(real_spam_pool) if strategy != "full_random" else len(group_data[group_data['label'] == 1])
        synthetic_count = int(total_spam_needed * synthetic_ratio / 100)
        real_count = total_spam_needed - synthetic_count

        logger.debug(f"Spam混合比例 ({strategy}): real={real_count}, synthetic={synthetic_count}")

        # 选择real spam
        if real_count > 0:
            if real_count <= len(real_spam_pool):
                selected_real_spam = real_spam_pool.sample(
                    n=real_count,
                    random_state=self.random_state,
                    replace=False  # No replacement sampling
                )
            else:
                # 如果需要的real数量超过可用数量，使用所有可用的
                selected_real_spam = real_spam_pool.copy()
                logger.warning(f"Real spam数据不足: 需要{real_count}, 可用{len(real_spam_pool)}")
        else:
            selected_real_spam = pd.DataFrame()

        # 选择synthetic spam
        if synthetic_count > 0:
            if synthetic_count <= len(available_synthetic):
                selected_synthetic = available_synthetic.sample(
                    n=synthetic_count,
                    random_state=self.random_state,
                    replace=False  # No replacement sampling
                )
            else:
                # 如果需要的synthetic数量超过可用数量，使用所有可用的
                selected_synthetic = available_synthetic.copy()
                logger.warning(f"Synthetic spam数据不足: 需要{synthetic_count}, 可用{len(available_synthetic)}")

            # 转换synthetic数据格式以匹配real数据
            synthetic_for_training = []
            for _, row in selected_synthetic.iterrows():
                synthetic_row = {
                    'unique_id': f"synthetic_{row['original_id']}_{synthetic_ratio}",
                    'subject': row['synthetic_subject'],
                    'body': row['synthetic_body'],
                    'label': 1,  # synthetic spam
                    'data_source': f"synthetic_{row['llm_engine']}",
                    'group_id': row.get('group_id', -1),
                    'is_synthetic': True,
                    'original_id': row['original_id'],
                    'prompt_name': row['prompt_name'],
                    'llm_engine': row['llm_engine']
                }
                synthetic_for_training.append(synthetic_row)

            synthetic_spam_df = pd.DataFrame(synthetic_for_training)
        else:
            synthetic_spam_df = pd.DataFrame()

        # 为real spam添加标识
        if not selected_real_spam.empty:
            selected_real_spam = selected_real_spam.copy()
            selected_real_spam['is_synthetic'] = False
            selected_real_spam['original_id'] = selected_real_spam['unique_id']
            selected_real_spam['prompt_name'] = 'real'
            selected_real_spam['llm_engine'] = 'none'

        # 为non-spam添加标识
        real_non_spam['is_synthetic'] = False
        real_non_spam['original_id'] = real_non_spam['unique_id']
        real_non_spam['prompt_name'] = 'real'
        real_non_spam['llm_engine'] = 'none'

        # 合并所有数据
        training_data_parts = []

        if not selected_real_spam.empty:
            training_data_parts.append(selected_real_spam)

        if not synthetic_spam_df.empty:
            training_data_parts.append(synthetic_spam_df)

        training_data_parts.append(real_non_spam)

        if not training_data_parts:
            raise ValueError("无法创建训练集：所有数据部分都为空")

        training_set = pd.concat(training_data_parts, ignore_index=True)

        # 随机打乱
        training_set = training_set.sample(
            n=len(training_set),
            random_state=self.random_state
        ).reset_index(drop=True)

        # 添加数据集标识
        training_set['synthetic_ratio'] = synthetic_ratio
        training_set['strategy'] = strategy
        training_set['dataset_type'] = 'training'

        logger.debug(f"训练集构建完成: {len(training_set)} 条, spam比例: {(training_set['label'] == 1).mean():.3f}")

        return training_set

    def build_datasets_for_ratios(
        self,
        processed_data: Dict[str, Any],
        synthetic_data: pd.DataFrame,
        prompt_name: str,
        llm_engine: str,
        strategy: str = "within_group",
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        为所有synthetic ratio构建数据集

        Args:
            processed_data: 预处理数据
            synthetic_data: synthetic数据
            prompt_name: prompt名称
            llm_engine: LLM引擎
            strategy: 混合策略
            output_dir: 输出目录

        Returns:
            Dict[str, Any]: 构建结果信息
        """
        if output_dir is None:
            output_dir = self.config.datasets_path

        logger.info(f"为所有synthetic ratio构建数据集: 策略={strategy}")

        groups = processed_data['groups']
        test_set = processed_data['test_set']
        build_results = {
            'strategy': strategy,
            'prompt_name': prompt_name,
            'llm_engine': llm_engine,
            'datasets': {},
            'metadata': {}
        }

        # 创建输出目录
        strategy_dir = os.path.join(output_dir, strategy)
        os.makedirs(strategy_dir, exist_ok=True)

        # 为每个synthetic ratio构建数据集
        for synthetic_ratio in self.config.synthetic_ratios:
            logger.info(f"构建synthetic ratio {synthetic_ratio}% 的数据集")

            ratio_datasets = {}
            ratio_metadata = {
                'synthetic_ratio': synthetic_ratio,
                'groups': {},
                'trials': {}
            }

            # 为每个组构建数据集
            for group_name, group_data in groups.items():
                group_datasets = {}
                group_metadata = {
                    'group_name': group_name,
                    'group_id': group_data['group_id'].iloc[0] if 'group_id' in group_data.columns else None,
                    'trials': {}
                }

                # 为每次试验构建数据集
                for trial in range(self.config.trials_per_config):
                    try:
                        # 设置试验特定的随机种子
                        trial_seed = self.config.random_seed + trial
                        self.random_state = np.random.RandomState(trial_seed)

                        # 创建混合训练集
                        training_set = self.create_mixed_training_set(
                            group_data, synthetic_data, synthetic_ratio, strategy,
                            all_groups_data=groups if strategy == "full_random" else None
                        )

                        # 保存训练集
                        dataset_name = self.config.dataset.lower().replace('-', '')
                        train_filename = f"train_{prompt_name}_r{synthetic_ratio}_g{group_metadata['group_id']}_t{trial}.csv.gz"
                        train_path = os.path.join(strategy_dir, train_filename)

                        training_set.to_csv(train_path, compression='gzip', index=False)

                        # 记录数据集信息
                        trial_info = {
                            'trial_id': trial,
                            'train_file': train_path,
                            'train_size': len(training_set),
                            'train_spam_count': (training_set['label'] == 1).sum(),
                            'train_spam_ratio': (training_set['label'] == 1).mean(),
                            'synthetic_count': (training_set.get('is_synthetic', False) == True).sum(),
                            'real_count': (training_set.get('is_synthetic', False) == False).sum(),
                            'random_seed': trial_seed
                        }

                        group_datasets[f'trial_{trial}'] = trial_info
                        group_metadata['trials'][f'trial_{trial}'] = trial_info

                    except Exception as e:
                        logger.error(f"构建失败: ratio={synthetic_ratio}, group={group_name}, trial={trial}, error={e}")
                        continue

                ratio_datasets[group_name] = group_datasets
                ratio_metadata['groups'][group_name] = group_metadata

            # 保存测试集（所有ratio和prompt共享，测试集不含synthetic数据）
            if test_set is not None and synthetic_ratio == self.config.synthetic_ratios[0]:
                test_filename = "test_set.csv.gz"
                test_path = os.path.join(strategy_dir, test_filename)
                # 只保存一次，避免重复写入
                if not os.path.exists(test_path):
                    test_set.to_csv(test_path, compression='gzip', index=False)

                ratio_metadata['test_file'] = test_path
                ratio_metadata['test_size'] = len(test_set)

            build_results['datasets'][f'ratio_{synthetic_ratio}'] = ratio_datasets
            build_results['metadata'][f'ratio_{synthetic_ratio}'] = ratio_metadata

        # 保存构建元数据
        metadata_filename = f"build_metadata_{strategy}_{prompt_name}_{llm_engine}.json"
        metadata_path = os.path.join(strategy_dir, metadata_filename)

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(build_results, f, indent=2, ensure_ascii=False, default=str)

        build_results['metadata_file'] = metadata_path

        logger.info(f"数据集构建完成: 策略={strategy}, 保存到={strategy_dir}")
        return build_results

    def load_dataset_for_experiment(
        self,
        datasets_dir: str,
        strategy: str,
        prompt_name: str,
        synthetic_ratio: int,
        group_id: int,
        trial: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        加载特定实验的数据集

        Args:
            datasets_dir: 数据集目录
            strategy: 混合策略
            prompt_name: prompt名称
            synthetic_ratio: synthetic比例
            group_id: 组ID
            trial: 试验ID

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (训练集, 测试集)
        """
        strategy_dir = os.path.join(datasets_dir, strategy)

        # 加载训练集
        train_filename = f"train_{prompt_name}_r{synthetic_ratio}_g{group_id}_t{trial}.csv.gz"
        train_path = os.path.join(strategy_dir, train_filename)

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"训练集文件不存在: {train_path}")

        training_set = pd.read_csv(train_path, compression='gzip')

        # 加载测试集（所有prompt共享同一个测试集）
        test_filename = "test_set.csv.gz"
        test_path = os.path.join(strategy_dir, test_filename)

        if not os.path.exists(test_path):
            raise FileNotFoundError(f"测试集文件不存在: {test_path}")

        test_set = pd.read_csv(test_path, compression='gzip')

        return training_set, test_set

    def get_dataset_statistics(self, datasets_dir: str, strategy: str) -> Dict[str, Any]:
        """
        获取数据集统计信息

        Args:
            datasets_dir: 数据集目录
            strategy: 混合策略

        Returns:
            Dict[str, Any]: 统计信息
        """
        strategy_dir = os.path.join(datasets_dir, strategy)

        # 查找元数据文件
        metadata_files = [f for f in os.listdir(strategy_dir) if f.startswith('build_metadata_')]

        if not metadata_files:
            raise FileNotFoundError(f"未找到构建元数据文件: {strategy_dir}")

        metadata_file = os.path.join(strategy_dir, metadata_files[0])

        with open(metadata_file, 'r', encoding='utf-8') as f:
            build_metadata = json.load(f)

        # 计算统计信息
        stats = {
            'strategy': strategy,
            'total_ratios': len(build_metadata['datasets']),
            'total_groups': 0,
            'total_trials': 0,
            'ratio_stats': {}
        }

        for ratio_key, ratio_data in build_metadata['datasets'].items():
            ratio_stats = {
                'groups_count': len(ratio_data),
                'trials_per_group': 0,
                'total_datasets': 0
            }

            for group_name, group_data in ratio_data.items():
                ratio_stats['trials_per_group'] = len(group_data)
                ratio_stats['total_datasets'] += len(group_data)

            stats['ratio_stats'][ratio_key] = ratio_stats
            stats['total_groups'] = max(stats['total_groups'], ratio_stats['groups_count'])
            stats['total_trials'] = max(stats['total_trials'], ratio_stats['total_datasets'])

        return stats


# 便捷函数
def build_all_datasets(
    config: ExperimentConfig,
    processed_data_dir: str,
    synthetic_data_dir: str,
    prompt_name: str,
    llm_engine: str,
    strategy: str = "within_group",
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    便捷函数：构建所有数据集

    Args:
        config: 实验配置
        processed_data_dir: 预处理数据目录
        synthetic_data_dir: synthetic数据目录
        prompt_name: prompt名称
        llm_engine: LLM引擎
        strategy: 混合策略
        output_dir: 输出目录

    Returns:
        Dict[str, Any]: 构建结果
    """
    builder = DatasetBuilder(config)

    # 加载数据
    processed_data = builder.load_processed_data(processed_data_dir)
    synthetic_data = builder.load_synthetic_data(synthetic_data_dir, prompt_name, llm_engine)

    # 构建数据集
    return builder.build_datasets_for_ratios(
        processed_data, synthetic_data, prompt_name, llm_engine, strategy, output_dir
    )