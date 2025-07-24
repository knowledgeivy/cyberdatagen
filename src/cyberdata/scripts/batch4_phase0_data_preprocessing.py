#!/usr/bin/env python3
"""
批次4实验 - Phase 0: 原始数据预处理和train/test划分

从CEAS-08原始数据开始，进行严格的8:2 train/test划分，
确保训练和测试数据完全独立，消除数据泄露风险。

作者: Claude
创建时间: 2025-07-24
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import random
from typing import Tuple, Dict
import logging
from sklearn.model_selection import train_test_split

# 项目路径设置
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.cyberdata.utils.logger_config import setup_logger
from src.cyberdata.utils.data_loader import save_csv_data, check_file_size_and_recommend_compression

class Batch4Phase0Preprocessor:
    """批次4 Phase 0: 原始数据预处理器"""
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logger("batch4_phase0")
        self.project_root = PROJECT_ROOT
        self.batch4_dir = self.project_root / "data" / "batch4_fresh"
        
        # 原始数据路径
        self.ceas08_file = self.project_root / "raw" / "SevenPhishingEmailDataset" / "CEAS-08.csv"
        
        # 确保输出目录存在
        self.batch4_dir.mkdir(parents=True, exist_ok=True)
        
        # 随机种子设置（确保可重复性）
        self.random_state = 2025
        
    def load_raw_data(self) -> pd.DataFrame:
        """加载CEAS-08原始数据"""
        try:
            self.logger.info(f"加载CEAS-08原始数据: {self.ceas08_file}")
            
            if not self.ceas08_file.exists():
                raise FileNotFoundError(f"CEAS-08文件不存在: {self.ceas08_file}")
            
            # 读取原始数据
            df = pd.read_csv(self.ceas08_file)
            self.logger.info(f"原始数据加载完成: {len(df)} 个样本")
            
            # 显示基本统计信息
            label_counts = df['label'].value_counts()
            self.logger.info(f"标签分布: {dict(label_counts)}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"加载原始数据失败: {str(e)}")
            raise
    
    def clean_and_standardize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗和标准化"""
        try:
            self.logger.info("开始数据清洗和标准化...")
            
            # 1. 处理缺失值
            original_len = len(df)
            df = df.dropna(subset=['subject', 'body', 'label'])
            self.logger.info(f"删除缺失值: {original_len} -> {len(df)} 个样本")
            
            # 2. 标准化标签格式
            df['label'] = df['label'].astype(int)
            
            # 3. 清理文本内容
            df['subject'] = df['subject'].astype(str).str.strip()
            df['body'] = df['body'].astype(str).str.strip()
            
            # 4. 过滤空内容
            df = df[(df['subject'].str.len() > 0) & (df['body'].str.len() > 0)]
            self.logger.info(f"过滤空内容后: {len(df)} 个样本")
            
            # 5. 添加组合文本字段（用于embedding）
            df['text'] = df['subject'] + " " + df['body']
            
            return df
            
        except Exception as e:
            self.logger.error(f"数据清洗失败: {str(e)}")
            raise
    
    def perform_train_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """执行8:2 train/test划分"""
        try:
            self.logger.info("执行8:2 train/test划分...")
            
            # 分层抽样，保持各类别比例
            train_df, test_df = train_test_split(
                df,
                test_size=0.2,
                random_state=self.random_state,
                stratify=df['label'],
                shuffle=True
            )
            
            self.logger.info(f"训练集: {len(train_df)} 个样本")
            self.logger.info(f"测试集: {len(test_df)} 个样本")
            
            # 验证标签分布
            train_label_dist = train_df['label'].value_counts(normalize=True)
            test_label_dist = test_df['label'].value_counts(normalize=True)
            
            self.logger.info("训练集标签分布:")
            for label, ratio in train_label_dist.items():
                self.logger.info(f"  标签{label}: {ratio:.1%}")
                
            self.logger.info("测试集标签分布:")
            for label, ratio in test_label_dist.items():
                self.logger.info(f"  标签{label}: {ratio:.1%}")
            
            return train_df, test_df
            
        except Exception as e:
            self.logger.error(f"train/test划分失败: {str(e)}")
            raise
    
    def extract_malicious_benign_data(self, train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """从训练集中提取恶意和良性数据"""
        try:
            self.logger.info("从训练集提取恶意和良性数据...")
            
            # 提取恶意数据 (label=1)
            train_malicious = train_df[train_df['label'] == 1].copy()
            
            # 提取良性数据 (label=0)
            train_benign = train_df[train_df['label'] == 0].copy()
            
            self.logger.info(f"训练集恶意样本: {len(train_malicious)} 个")
            self.logger.info(f"训练集良性样本: {len(train_benign)} 个")
            
            return train_malicious, train_benign
            
        except Exception as e:
            self.logger.error(f"提取恶意/良性数据失败: {str(e)}")
            raise
    
    def add_unique_ids(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                      train_malicious: pd.DataFrame, train_benign: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """添加唯一ID标注"""
        try:
            self.logger.info("添加唯一ID标注...")
            
            # 为训练集添加ID
            train_df_with_id = train_df.copy()
            train_df_with_id['unique_id'] = [f"train_real_{i:06d}" for i in range(len(train_df))]
            
            # 为测试集添加ID
            test_df_with_id = test_df.copy()
            test_df_with_id['unique_id'] = [f"test_real_{i:06d}" for i in range(len(test_df))]
            
            # 为恶意训练数据添加ID
            train_malicious_with_id = train_malicious.copy()
            train_malicious_with_id['unique_id'] = [f"train_real_mal_{i:06d}" for i in range(len(train_malicious))]
            
            # 为良性训练数据添加ID
            train_benign_with_id = train_benign.copy()
            train_benign_with_id['unique_id'] = [f"train_real_ben_{i:06d}" for i in range(len(train_benign))]
            
            self.logger.info("ID标注完成")
            
            return {
                'train_set': train_df_with_id,
                'test_set': test_df_with_id,
                'train_malicious': train_malicious_with_id,
                'train_benign': train_benign_with_id
            }
            
        except Exception as e:
            self.logger.error(f"添加ID标注失败: {str(e)}")
            raise
    
    def save_processed_data(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        """保存处理后的数据"""
        try:
            self.logger.info("保存处理后的数据...")
            
            # 保存各个数据集，大文件自动压缩
            file_mapping = {
                'train_set': 'raw_train_set.csv',
                'test_set': 'raw_test_set.csv',
                'train_malicious': 'train_malicious.csv',
                'train_benign': 'train_benign.csv'
            }
            
            for key, filename in file_mapping.items():
                file_path = self.batch4_dir / filename
                df = data_dict[key]
                
                # 估算文件大小（每行约300字节）
                estimated_size_mb = len(df) * 300 / (1024 * 1024)
                compress = estimated_size_mb > 50.0
                
                if compress:
                    self.logger.info(f"文件 {filename} 预计大小 {estimated_size_mb:.1f}MB，将压缩保存")
                
                save_csv_data(df, file_path, compress=compress, logger=self.logger)
            
            # 生成数据统计摘要
            self.generate_data_summary(data_dict)
            
        except Exception as e:
            self.logger.error(f"保存数据失败: {str(e)}")
            raise
    
    def generate_data_summary(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        """生成数据处理摘要"""
        try:
            summary = {
                'processing_info': {
                    'source_file': str(self.ceas08_file),
                    'processing_date': '2025-07-24',
                    'random_state': self.random_state,
                    'split_ratio': '8:2 (train:test)'
                },
                'data_statistics': {
                    'train_total': len(data_dict['train_set']),
                    'test_total': len(data_dict['test_set']),
                    'train_malicious': len(data_dict['train_malicious']),
                    'train_benign': len(data_dict['train_benign'])
                },
                'label_distribution': {
                    'train_malicious_ratio': len(data_dict['train_malicious']) / len(data_dict['train_set']),
                    'train_benign_ratio': len(data_dict['train_benign']) / len(data_dict['train_set'])
                },
                'id_format': {
                    'train_samples': 'train_real_{index:06d}',
                    'test_samples': 'test_real_{index:06d}',
                    'train_malicious': 'train_real_mal_{index:06d}',
                    'train_benign': 'train_real_ben_{index:06d}'
                }
            }
            
            # 保存摘要
            import json
            summary_file = self.batch4_dir / "phase0_data_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"数据处理摘要保存到: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"生成数据摘要失败: {str(e)}")
            raise
    
    def run_phase0_complete(self) -> bool:
        """执行完整的Phase 0流程"""
        try:
            self.logger.info("="*50)
            self.logger.info("开始执行批次4 Phase 0: 原始数据预处理和train/test划分")
            self.logger.info("="*50)
            
            # Step 1: 加载原始数据
            self.logger.info("Step 1: 加载CEAS-08原始数据")
            raw_df = self.load_raw_data()
            
            # Step 2: 数据清洗和标准化
            self.logger.info("Step 2: 数据清洗和标准化")
            clean_df = self.clean_and_standardize_data(raw_df)
            
            # Step 3: train/test划分
            self.logger.info("Step 3: 执行8:2 train/test划分")
            train_df, test_df = self.perform_train_test_split(clean_df)
            
            # Step 4: 提取恶意和良性数据
            self.logger.info("Step 4: 从训练集提取恶意和良性数据")
            train_malicious, train_benign = self.extract_malicious_benign_data(train_df)
            
            # Step 5: 添加唯一ID
            self.logger.info("Step 5: 添加唯一ID标注")
            data_dict = self.add_unique_ids(train_df, test_df, train_malicious, train_benign)
            
            # Step 6: 保存处理后的数据
            self.logger.info("Step 6: 保存处理后的数据")
            self.save_processed_data(data_dict)
            
            self.logger.info("="*50)
            self.logger.info("批次4 Phase 0 执行完成!")
            self.logger.info(f"训练集: {len(data_dict['train_set'])} 样本")
            self.logger.info(f"测试集: {len(data_dict['test_set'])} 样本")
            self.logger.info(f"训练集恶意样本: {len(data_dict['train_malicious'])} 个")
            self.logger.info(f"训练集良性样本: {len(data_dict['train_benign'])} 个")
            self.logger.info("数据已保存到: data/batch4_fresh/")
            self.logger.info("="*50)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Phase 0执行失败: {str(e)}")
            return False

def main():
    """主函数"""
    # 设置随机种子
    random.seed(2025)
    np.random.seed(2025)
    
    # 设置日志
    logger = setup_logger("batch4_phase0", log_level=logging.INFO)
    
    try:
        # 创建预处理器
        preprocessor = Batch4Phase0Preprocessor(logger)
        
        # 执行Phase 0
        success = preprocessor.run_phase0_complete()
        
        if success:
            logger.info("批次4 Phase 0 successfully completed!")
            return 0
        else:
            logger.error("批次4 Phase 0 failed!")
            return 1
            
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())