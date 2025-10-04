"""
数据预处理模块
负责加载、清洗、分组原始邮件数据
"""

import pandas as pd
import numpy as np
import json
import gzip
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import StratifiedShuffleSplit
from loguru import logger
import os

from ..config.config_manager import ExperimentConfig


class EmailDataPreprocessor:
    """邮件数据预处理器"""

    def __init__(self, config: ExperimentConfig):
        """
        初始化预处理器

        Args:
            config: 实验配置
        """
        self.config = config
        self.random_state = np.random.RandomState(config.random_seed)

    def load_raw_data(self, dataset_name: str) -> pd.DataFrame:
        """
        加载原始数据

        Args:
            dataset_name: 数据集名称 (e.g., "CEAS-08")

        Returns:
            pd.DataFrame: 原始数据
        """
        logger.info(f"加载原始数据: {dataset_name}")

        # 首先尝试从SevenPhishingEmailDataset目录加载（新数据源）
        seven_dataset_file = Path(self.config.raw_data_path) / "SevenPhishingEmailDataset" / f"{dataset_name}.csv"

        if seven_dataset_file.exists():
            logger.info(f"从SevenPhishingEmailDataset加载: {seven_dataset_file}")
            try:
                data = pd.read_csv(seven_dataset_file)
                logger.info(f"数据加载完成: {len(data)} 条记录")
                logger.info(f"数据列: {list(data.columns)}")
                return data
            except Exception as e:
                logger.error(f"数据加载失败: {e}")
                raise

        # 如果SevenPhishingEmailDataset不存在，回退到旧的分离文件格式（向后兼容）
        logger.info("SevenPhishingEmailDataset不存在，尝试旧格式（train/test分离）")
        train_file = Path(self.config.raw_data_path) / f"email_phishing_{dataset_name}_train.csv.gz"
        test_file = Path(self.config.raw_data_path) / f"email_phishing_{dataset_name}_test.csv.gz"

        if not train_file.exists() or not test_file.exists():
            raise FileNotFoundError(
                f"数据文件不存在。尝试了：\n"
                f"  1. {seven_dataset_file}\n"
                f"  2. {train_file} 和 {test_file}"
            )

        try:
            # 加载训练和测试数据
            train_data = pd.read_csv(train_file, compression='gzip')
            test_data = pd.read_csv(test_file, compression='gzip')

            # 合并数据
            data = pd.concat([train_data, test_data], ignore_index=True)

            logger.info(f"数据加载完成: {len(data)} 条记录")
            logger.info(f"数据列: {list(data.columns)}")

            return data

        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        清洗数据

        Args:
            data: 原始数据

        Returns:
            pd.DataFrame: 清洗后的数据
        """
        logger.info("开始数据清洗")

        # 检查必要的列
        required_columns = ['subject', 'body', 'label']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"缺少必要的列: {missing_columns}")

        # 创建副本
        cleaned_data = data.copy()

        # 处理缺失值
        logger.info(f"处理前缺失值: {cleaned_data.isnull().sum().to_dict()}")

        # 填充缺失的subject和body
        cleaned_data['subject'] = cleaned_data['subject'].fillna('')
        cleaned_data['body'] = cleaned_data['body'].fillna('')

        # 删除subject和body都为空的记录
        before_count = len(cleaned_data)
        cleaned_data = cleaned_data[
            (cleaned_data['subject'].str.len() > 0) |
            (cleaned_data['body'].str.len() > 0)
        ].copy()
        after_count = len(cleaned_data)

        if before_count != after_count:
            logger.info(f"删除空内容记录: {before_count - after_count} 条")

        # 清理文本内容
        cleaned_data['subject'] = cleaned_data['subject'].astype(str).str.strip()
        cleaned_data['body'] = cleaned_data['body'].astype(str).str.strip()

        # 确保标签为数值类型
        cleaned_data['label'] = pd.to_numeric(cleaned_data['label'], errors='coerce')

        # 删除标签缺失的记录
        before_count = len(cleaned_data)
        cleaned_data = cleaned_data.dropna(subset=['label']).copy()
        after_count = len(cleaned_data)

        if before_count != after_count:
            logger.info(f"删除标签缺失记录: {before_count - after_count} 条")

        # 确保标签只有0和1
        valid_labels = cleaned_data['label'].isin([0, 1])
        if not valid_labels.all():
            logger.warning(f"发现无效标签: {cleaned_data[~valid_labels]['label'].unique()}")
            cleaned_data = cleaned_data[valid_labels].copy()

        logger.info(f"数据清洗完成: {len(cleaned_data)} 条记录")
        logger.info(f"标签分布: {cleaned_data['label'].value_counts().to_dict()}")

        return cleaned_data

    def add_unique_ids(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        为每条记录添加唯一ID

        Args:
            data: 清洗后的数据

        Returns:
            pd.DataFrame: 添加ID后的数据
        """
        logger.info("添加唯一ID")

        data_with_ids = data.copy()

        # 创建基于内容的哈希ID
        def create_content_hash(row):
            content = f"{row['subject']}||{row['body']}||{row['label']}"
            return hashlib.md5(content.encode('utf-8')).hexdigest()

        data_with_ids['content_hash'] = data_with_ids.apply(create_content_hash, axis=1)

        # 添加序列ID
        data_with_ids['unique_id'] = [
            f"{self.config.dataset.lower()}_{i:06d}"
            for i in range(len(data_with_ids))
        ]

        # 添加数据源信息
        data_with_ids['data_source'] = self.config.dataset
        data_with_ids['processing_timestamp'] = pd.Timestamp.now()

        logger.info(f"唯一ID添加完成: {len(data_with_ids)} 条记录")

        return data_with_ids

    def create_balanced_groups(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        创建平衡的数据分组

        Args:
            data: 添加ID后的数据

        Returns:
            Dict: 包含分组信息的字典
        """
        logger.info(f"创建 {self.config.n_groups} 个平衡分组")

        # 分离spam和non-spam数据
        spam_data = data[data['label'] == 1].copy()
        non_spam_data = data[data['label'] == 0].copy()

        logger.info(f"可用数据: spam={len(spam_data)}, non-spam={len(non_spam_data)}")

        # 计算每组所需的样本数
        spam_per_group = int(self.config.sample_size_per_group * self.config.spam_ratio)
        non_spam_per_group = self.config.sample_size_per_group - spam_per_group

        logger.info(f"每组配置: spam={spam_per_group}, non-spam={non_spam_per_group}")

        # 检查数据充足性
        total_spam_needed = spam_per_group * self.config.n_groups
        total_non_spam_needed = non_spam_per_group * self.config.n_groups

        if len(spam_data) < total_spam_needed:
            raise ValueError(f"Spam数据不足: 需要{total_spam_needed}, 可用{len(spam_data)}")

        # Non-spam允许重复使用（with replacement）作为固定背景
        non_spam_replacement = len(non_spam_data) < total_non_spam_needed
        if non_spam_replacement:
            logger.warning(f"Non-spam数据不足，将使用replacement采样: 需要{total_non_spam_needed}, 可用{len(non_spam_data)}")
            logger.info("Non-spam作为背景负样本，允许在不同组间重复出现（学术上可接受）")

        # 随机打乱spam数据（严格无放回）
        spam_data = spam_data.sample(n=len(spam_data), random_state=self.config.random_seed, replace=False)

        # 随机打乱non-spam数据，准备分组
        non_spam_data = non_spam_data.sample(n=len(non_spam_data), random_state=self.config.random_seed, replace=False)

        # 创建分组
        groups = {}
        group_metadata = {
            'n_groups': self.config.n_groups,
            'sample_size_per_group': self.config.sample_size_per_group,
            'spam_per_group': spam_per_group,
            'non_spam_per_group': non_spam_per_group,
            'spam_ratio': self.config.spam_ratio,
            'random_seed': self.config.random_seed,
            'non_spam_replacement': non_spam_replacement,  # 记录是否使用了replacement
            'groups': {}
        }

        for i in range(self.config.n_groups):
            # 为当前组选择spam数据（严格无放回）
            spam_start = i * spam_per_group
            spam_end = (i + 1) * spam_per_group
            group_spam = spam_data.iloc[spam_start:spam_end].copy()

            # 为当前组选择non-spam数据（允许循环使用）
            if non_spam_replacement:
                # 使用模运算循环使用non-spam数据
                group_non_spam_indices = []
                for j in range(non_spam_per_group):
                    idx = (i * non_spam_per_group + j) % len(non_spam_data)
                    group_non_spam_indices.append(idx)
                group_non_spam = non_spam_data.iloc[group_non_spam_indices].copy()
            else:
                # 充足时正常分配
                non_spam_start = i * non_spam_per_group
                non_spam_end = (i + 1) * non_spam_per_group
                group_non_spam = non_spam_data.iloc[non_spam_start:non_spam_end].copy()

            # 合并组数据
            group_data = pd.concat([group_spam, group_non_spam], ignore_index=True)

            # 随机打乱组内数据
            group_data = group_data.sample(n=len(group_data), random_state=self.config.random_seed + i)

            # 添加组标识
            group_data['group_id'] = i
            group_data['group_position'] = range(len(group_data))

            groups[f"group_{i}"] = group_data

            # 记录组元数据
            group_metadata['groups'][f"group_{i}"] = {
                'group_id': i,
                'total_samples': len(group_data),
                'spam_samples': len(group_spam),
                'non_spam_samples': len(group_non_spam),
                'spam_ratio_actual': len(group_spam) / len(group_data),
                'spam_ids': group_spam['unique_id'].tolist(),
                'non_spam_ids': group_non_spam['unique_id'].tolist()
            }

        logger.info(f"分组创建完成: {len(groups)} 组")

        # 计算剩余数据
        # Spam严格无放回，所以只有未使用的spam可用于测试
        remaining_spam = spam_data.iloc[total_spam_needed:]

        # Non-spam如果使用了replacement，则所有non-spam都可用于测试集
        # 因为训练集的non-spam是循环使用的，并未"消耗"数据
        if non_spam_replacement:
            remaining_non_spam = non_spam_data.copy()
            logger.info(f"Non-spam使用了replacement，所有{len(remaining_non_spam)}个non-spam可用于测试集")
        else:
            remaining_non_spam = non_spam_data.iloc[total_non_spam_needed:]

        return {
            'groups': groups,
            'metadata': group_metadata,
            'remaining_spam': remaining_spam,
            'remaining_non_spam': remaining_non_spam
        }

    def create_test_set(self, remaining_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        从剩余数据创建测试集

        Args:
            remaining_data: 剩余数据字典

        Returns:
            pd.DataFrame: 测试集
        """
        logger.info("创建测试集")

        remaining_spam = remaining_data['remaining_spam']
        remaining_non_spam = remaining_data['remaining_non_spam']

        # 计算测试集大小
        total_remaining = len(remaining_spam) + len(remaining_non_spam)
        test_size = int(total_remaining * self.config.test_split)

        # 保持spam比例
        test_spam_size = int(test_size * self.config.spam_ratio)
        test_non_spam_size = test_size - test_spam_size

        # 检查数据充足性
        if len(remaining_spam) < test_spam_size:
            test_spam_size = len(remaining_spam)
            logger.warning(f"调整测试集spam大小为: {test_spam_size}")

        if len(remaining_non_spam) < test_non_spam_size:
            test_non_spam_size = len(remaining_non_spam)
            logger.warning(f"调整测试集non-spam大小为: {test_non_spam_size}")

        # 随机选择测试数据
        test_spam = remaining_spam.sample(n=test_spam_size, random_state=self.config.random_seed)
        test_non_spam = remaining_non_spam.sample(n=test_non_spam_size, random_state=self.config.random_seed)

        # 合并测试集
        test_set = pd.concat([test_spam, test_non_spam], ignore_index=True)
        test_set = test_set.sample(n=len(test_set), random_state=self.config.random_seed)

        # 添加测试集标识
        test_set['split'] = 'test'
        test_set['test_position'] = range(len(test_set))

        logger.info(f"测试集创建完成: {len(test_set)} 条记录")
        logger.info(f"测试集spam比例: {(test_set['label'] == 1).mean():.3f}")

        return test_set

    def save_processed_data(self, processed_data: Dict[str, any], output_dir: str) -> Dict[str, str]:
        """
        保存预处理后的数据

        Args:
            processed_data: 预处理后的数据字典
            output_dir: 输出目录

        Returns:
            Dict[str, str]: 保存的文件路径
        """
        logger.info(f"保存预处理数据到: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}

        dataset_name = self.config.dataset.lower().replace('-', '')

        # 保存完整的预处理数据
        all_groups_data = []
        for group_name, group_data in processed_data['groups'].items():
            all_groups_data.append(group_data)

        complete_data = pd.concat(all_groups_data, ignore_index=True)

        # 添加测试集
        if 'test_set' in processed_data:
            complete_data = pd.concat([complete_data, processed_data['test_set']], ignore_index=True)

        complete_file = os.path.join(output_dir, f"{dataset_name}_processed.csv.gz")
        complete_data.to_csv(complete_file, compression='gzip', index=False)
        saved_files['complete_data'] = complete_file

        # 保存分组数据（单独文件）
        groups_dir = os.path.join(output_dir, 'groups')
        os.makedirs(groups_dir, exist_ok=True)

        for group_name, group_data in processed_data['groups'].items():
            group_file = os.path.join(groups_dir, f"{dataset_name}_{group_name}.csv.gz")
            group_data.to_csv(group_file, compression='gzip', index=False)
            saved_files[group_name] = group_file

        # 保存测试集
        if 'test_set' in processed_data:
            test_file = os.path.join(output_dir, f"{dataset_name}_test_set.csv.gz")
            processed_data['test_set'].to_csv(test_file, compression='gzip', index=False)
            saved_files['test_set'] = test_file

        # 保存元数据
        metadata_file = os.path.join(output_dir, f"{dataset_name}_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data['metadata'], f, indent=2, ensure_ascii=False, default=str)
        saved_files['metadata'] = metadata_file

        # 保存统计信息
        stats = self._compute_statistics(complete_data, processed_data['metadata'])
        stats_file = os.path.join(output_dir, f"{dataset_name}_statistics.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
        saved_files['statistics'] = stats_file

        logger.info(f"数据保存完成: {len(saved_files)} 个文件")
        return saved_files

    def _compute_statistics(self, data: pd.DataFrame, metadata: Dict) -> Dict:
        """
        计算数据统计信息

        Args:
            data: 完整数据
            metadata: 元数据

        Returns:
            Dict: 统计信息
        """
        stats = {
            'total_samples': len(data),
            'total_spam': (data['label'] == 1).sum(),
            'total_non_spam': (data['label'] == 0).sum(),
            'overall_spam_ratio': (data['label'] == 1).mean(),
            'subject_length_stats': {
                'mean': data['subject'].str.len().mean(),
                'std': data['subject'].str.len().std(),
                'min': data['subject'].str.len().min(),
                'max': data['subject'].str.len().max()
            },
            'body_length_stats': {
                'mean': data['body'].str.len().mean(),
                'std': data['body'].str.len().std(),
                'min': data['body'].str.len().min(),
                'max': data['body'].str.len().max()
            },
            'groups_stats': {}
        }

        # 按组统计
        if 'group_id' in data.columns:
            for group_id in data['group_id'].unique():
                if pd.isna(group_id):
                    continue
                group_data = data[data['group_id'] == group_id]
                stats['groups_stats'][f'group_{int(group_id)}'] = {
                    'total_samples': len(group_data),
                    'spam_samples': (group_data['label'] == 1).sum(),
                    'non_spam_samples': (group_data['label'] == 0).sum(),
                    'spam_ratio': (group_data['label'] == 1).mean()
                }

        return stats

    def process_dataset(self, dataset_name: str, output_dir: Optional[str] = None) -> Dict[str, any]:
        """
        完整的数据预处理流程

        Args:
            dataset_name: 数据集名称
            output_dir: 输出目录

        Returns:
            Dict: 处理后的数据和元数据
        """
        if output_dir is None:
            output_dir = self.config.processed_data_path

        logger.info(f"开始预处理数据集: {dataset_name}")

        # 1. 加载原始数据
        raw_data = self.load_raw_data(dataset_name)

        # 2. 清洗数据
        cleaned_data = self.clean_data(raw_data)

        # 3. 添加唯一ID
        data_with_ids = self.add_unique_ids(cleaned_data)

        # 4. 创建分组
        groups_result = self.create_balanced_groups(data_with_ids)

        # 5. 创建测试集
        test_set = self.create_test_set(groups_result)

        # 6. 准备最终结果
        processed_data = {
            'groups': groups_result['groups'],
            'metadata': groups_result['metadata'],
            'test_set': test_set,
            'raw_data_stats': {
                'original_size': len(raw_data),
                'cleaned_size': len(cleaned_data),
                'with_ids_size': len(data_with_ids)
            }
        }

        # 7. 保存数据
        saved_files = self.save_processed_data(processed_data, output_dir)
        processed_data['saved_files'] = saved_files

        logger.info("数据预处理完成")
        return processed_data