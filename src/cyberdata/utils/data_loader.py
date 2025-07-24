#!/usr/bin/env python3
"""
数据加载工具模块

处理CSV和压缩CSV文件的统一读取接口，
支持自动检测文件格式（.csv, .csv.gz）

作者: Claude
创建时间: 2025-07-24
"""

import pandas as pd
import gzip
from pathlib import Path
from typing import Union, Optional
import logging

def load_csv_data(file_path: Union[str, Path], 
                  logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    统一的CSV数据加载函数，自动处理压缩和非压缩文件
    
    Args:
        file_path: CSV文件路径（支持.csv和.csv.gz）
        logger: 日志记录器（可选）
    
    Returns:
        pd.DataFrame: 加载的数据
    
    Raises:
        FileNotFoundError: 文件不存在
        Exception: 读取失败
    """
    file_path = Path(file_path)
    
    if logger:
        logger.info(f"加载数据文件: {file_path}")
    
    # 检查文件是否存在
    if not file_path.exists():
        # 如果指定的文件不存在，尝试查找压缩版本
        if file_path.suffix == '.csv':
            gz_path = file_path.with_suffix('.csv.gz')
            if gz_path.exists():
                file_path = gz_path
                if logger:
                    logger.info(f"找到压缩版本: {file_path}")
            else:
                raise FileNotFoundError(f"文件不存在: {file_path}")
        else:
            raise FileNotFoundError(f"文件不存在: {file_path}")
    
    try:
        # 根据文件扩展名选择读取方式
        if file_path.suffix == '.gz':
            # 读取压缩文件
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                df = pd.read_csv(f)
            if logger:
                logger.info(f"成功加载压缩数据: {len(df)} 行")
        else:
            # 读取普通CSV文件
            df = pd.read_csv(file_path, encoding='utf-8')
            if logger:
                logger.info(f"成功加载数据: {len(df)} 行")
        
        return df
        
    except Exception as e:
        error_msg = f"读取文件失败 {file_path}: {str(e)}"
        if logger:
            logger.error(error_msg)
        raise Exception(error_msg)

def save_csv_data(df: pd.DataFrame, 
                  file_path: Union[str, Path],
                  compress: bool = False,
                  logger: Optional[logging.Logger] = None) -> None:
    """
    统一的CSV数据保存函数，可选择压缩
    
    Args:
        df: 要保存的DataFrame
        file_path: 保存路径
        compress: 是否压缩保存
        logger: 日志记录器（可选）
    """
    file_path = Path(file_path)
    
    try:
        if compress:
            # 保存为压缩文件
            if not file_path.suffix == '.gz':
                file_path = file_path.with_suffix(file_path.suffix + '.gz')
            
            with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                df.to_csv(f, index=False)
            
            if logger:
                logger.info(f"成功保存压缩数据到: {file_path} ({len(df)} 行)")
        else:
            # 保存为普通CSV文件
            df.to_csv(file_path, index=False, encoding='utf-8')
            
            if logger:
                logger.info(f"成功保存数据到: {file_path} ({len(df)} 行)")
                
    except Exception as e:
        error_msg = f"保存文件失败 {file_path}: {str(e)}"
        if logger:
            logger.error(error_msg)
        raise Exception(error_msg)

def check_file_size_and_recommend_compression(file_path: Union[str, Path],
                                             size_limit_mb: float = 50.0,
                                             logger: Optional[logging.Logger] = None) -> bool:
    """
    检查文件大小并建议是否需要压缩
    
    Args:
        file_path: 文件路径
        size_limit_mb: 大小限制（MB）
        logger: 日志记录器
    
    Returns:
        bool: 是否建议压缩
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return False
    
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    
    if logger:
        logger.info(f"文件 {file_path.name} 大小: {file_size_mb:.1f} MB")
    
    if file_size_mb > size_limit_mb:
        if logger:
            logger.warning(f"文件超过 {size_limit_mb} MB 限制，建议压缩")
        return True
    
    return False