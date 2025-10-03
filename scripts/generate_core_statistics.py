#!/usr/bin/env python3
"""
Core Statistics Generator for Multi-Sample Evaluation Framework
Generates only the essential CSV reports as specified in design documents
"""

import argparse
import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from scipy import stats
from scipy.stats import ttest_rel
from scipy.spatial.distance import jensenshannon
from loguru import logger

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from src.config.config_manager import load_config


def kl_divergence(p, q, epsilon=1e-10):
    """计算KL散度"""
    p = np.array(p) + epsilon
    q = np.array(q) + epsilon
    p = p / np.sum(p)
    q = q / np.sum(q)
    return np.sum(p * np.log(p / q))


def cohens_d(x, y):
    """计算Cohen's d效应量"""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std


def load_and_organize_results(results_file: str) -> Dict:
    """加载并组织结果数据"""
    logger.info(f"加载分类结果: {results_file}")

    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # 按strategy和ratio组织数据
    organized = {}
    for result in results['results']:
        strategy = result.get('strategy', 'unknown')
        ratio = result['synthetic_ratio']

        if strategy not in organized:
            organized[strategy] = {}
        if ratio not in organized[strategy]:
            organized[strategy][ratio] = {}

        for classifier_name, classifier_result in result['classifiers'].items():
            if not classifier_result['success']:
                continue

            if classifier_name not in organized[strategy][ratio]:
                organized[strategy][ratio][classifier_name] = {}

            metrics = classifier_result['metrics']
            for metric_name, metric_value in metrics.items():
                if metric_name not in organized[strategy][ratio][classifier_name]:
                    organized[strategy][ratio][classifier_name][metric_name] = []
                organized[strategy][ratio][classifier_name][metric_name].append(metric_value)

    logger.info(f"组织完成: {len(organized)} 个策略")
    return organized


def generate_core_statistics(organized_results: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    生成核心统计表格

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (分类性能表, 统计检验表)
    """
    logger.info("生成核心统计指标")

    # 1. 分类性能表 (基于设计文档Table 1格式)
    performance_rows = []

    # 2. 统计检验表 (基于设计文档Table 2格式)
    statistical_rows = []

    for strategy, strategy_data in organized_results.items():
        baseline_ratio = 0  # 基线比较

        if baseline_ratio not in strategy_data:
            logger.warning(f"策略 {strategy} 缺少基线ratio {baseline_ratio}")
            continue

        baseline_data = strategy_data[baseline_ratio]

        # 生成分类性能表
        for ratio in sorted(strategy_data.keys()):
            ratio_data = strategy_data[ratio]

            for classifier in ratio_data.keys():
                classifier_data = ratio_data[classifier]

                for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                    if metric in classifier_data:
                        values = classifier_data[metric]
                        performance_rows.append({
                            'strategy': strategy,
                            'synthetic_ratio': ratio,
                            'classifier': classifier,
                            'metric': metric,
                            'mean': np.mean(values),
                            'std': np.std(values, ddof=1),
                            'n': len(values)
                        })

        # 生成统计检验表 (与基线比较)
        for ratio in sorted(strategy_data.keys()):
            if ratio == baseline_ratio:
                continue

            ratio_data = strategy_data[ratio]

            for classifier in baseline_data.keys():
                if classifier not in ratio_data:
                    continue

                baseline_classifier = baseline_data[classifier]
                ratio_classifier = ratio_data[classifier]

                for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                    if metric not in baseline_classifier or metric not in ratio_classifier:
                        continue

                    baseline_values = np.array(baseline_classifier[metric])
                    ratio_values = np.array(ratio_classifier[metric])

                    if len(baseline_values) == 0 or len(ratio_values) == 0:
                        continue

                    # 基本统计
                    baseline_mean = np.mean(baseline_values)
                    ratio_mean = np.mean(ratio_values)
                    performance_change = ratio_mean - baseline_mean

                    # t检验
                    if len(baseline_values) == len(ratio_values):
                        t_stat, p_value = ttest_rel(baseline_values, ratio_values)
                    else:
                        t_stat, p_value = stats.ttest_ind(baseline_values, ratio_values)

                    # Cohen's d
                    try:
                        effect_size = cohens_d(baseline_values, ratio_values)
                    except:
                        effect_size = np.nan

                    # KL散度
                    try:
                        hist_baseline, bins = np.histogram(baseline_values, bins=5, density=True)
                        hist_ratio, _ = np.histogram(ratio_values, bins=bins, density=True)
                        hist_baseline = hist_baseline / np.sum(hist_baseline)
                        hist_ratio = hist_ratio / np.sum(hist_ratio)
                        kl_div = kl_divergence(hist_baseline, hist_ratio)
                    except:
                        kl_div = np.nan

                    statistical_rows.append({
                        'strategy': strategy,
                        'synthetic_ratio': ratio,
                        'classifier': classifier,
                        'metric': metric,
                        'baseline_mean': baseline_mean,
                        'test_mean': ratio_mean,
                        'performance_change': performance_change,
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'cohens_d': effect_size,
                        'kl_divergence': kl_div,
                        'is_significant': p_value < 0.05 if not np.isnan(p_value) else False
                    })

    performance_df = pd.DataFrame(performance_rows)
    statistical_df = pd.DataFrame(statistical_rows)

    logger.info(f"性能表: {len(performance_df)} 行")
    logger.info(f"统计表: {len(statistical_df)} 行")

    return performance_df, statistical_df


def find_best_ratios(statistical_df: pd.DataFrame) -> pd.DataFrame:
    """找到每个分类器和指标的最佳synthetic ratio"""
    logger.info("计算最佳synthetic ratio")

    best_ratios = []

    for classifier in statistical_df['classifier'].unique():
        for metric in statistical_df['metric'].unique():
            # 对每个分类器-指标组合，找到性能衰减最小的ratio
            subset = statistical_df[
                (statistical_df['classifier'] == classifier) &
                (statistical_df['metric'] == metric)
            ]

            if len(subset) == 0:
                continue

            # 找到绝对性能变化最小的ratio
            best_idx = subset['performance_change'].abs().idxmin()
            best_row = subset.loc[best_idx]

            best_ratios.append({
                'classifier': classifier,
                'metric': metric,
                'best_ratio': best_row['synthetic_ratio'],
                'best_strategy': best_row['strategy'],
                'performance_change': best_row['performance_change'],
                'p_value': best_row['p_value'],
                'cohens_d': best_row['cohens_d'],
                'is_significant': best_row['is_significant']
            })

    best_ratios_df = pd.DataFrame(best_ratios)
    logger.info(f"最佳ratio表: {len(best_ratios_df)} 行")

    return best_ratios_df


def main():
    parser = argparse.ArgumentParser(description='Generate core statistics CSV reports')
    parser.add_argument('--config', type=str, required=True, help='Configuration file path')
    parser.add_argument('--results_file', type=str, help='Classification results file path')
    parser.add_argument('--output_dir', type=str, help='Output directory')

    args = parser.parse_args()

    try:
        # 加载配置
        config = load_config(args.config)

        # 设置路径
        results_file = args.results_file or os.path.join(
            config.output.get('results_path', './output/results/'),
            f"{config.name}_classification_results.json"
        )
        output_dir = args.output_dir or config.output.get('reports_path', './output/reports/')

        if not os.path.exists(results_file):
            logger.error(f"结果文件不存在: {results_file}")
            return

        logger.info("开始生成核心统计报告")

        # 加载和组织数据
        organized_results = load_and_organize_results(results_file)

        # 生成核心统计表
        performance_df, statistical_df = generate_core_statistics(organized_results)

        # 生成最佳ratio表
        best_ratios_df = find_best_ratios(statistical_df)

        # 保存CSV文件
        os.makedirs(output_dir, exist_ok=True)

        # 1. 分类性能表 (对应设计文档 Table 1)
        performance_file = os.path.join(output_dir, f"{config.name}_classification_performance.csv")
        performance_df.to_csv(performance_file, index=False)
        logger.info(f"分类性能表已保存: {performance_file}")

        # 2. 统计检验表 (对应设计文档 Table 2)
        statistical_file = os.path.join(output_dir, f"{config.name}_statistical_tests.csv")
        statistical_df.to_csv(statistical_file, index=False)
        logger.info(f"统计检验表已保存: {statistical_file}")

        # 3. 最佳ratio表 (对应设计文档 Table 3)
        best_ratios_file = os.path.join(output_dir, f"{config.name}_best_ratios.csv")
        best_ratios_df.to_csv(best_ratios_file, index=False)
        logger.info(f"最佳ratio表已保存: {best_ratios_file}")

        # 输出核心统计摘要
        logger.info("=" * 50)
        logger.info("核心统计报告生成完成")
        logger.info("=" * 50)

        # 显著性差异汇总
        if len(statistical_df) > 0:
            significant_count = statistical_df['is_significant'].sum()
            total_tests = len(statistical_df)
            logger.info(f"显著性差异: {significant_count}/{total_tests} ({significant_count/total_tests*100:.1f}%)")

            # 平均效应量
            mean_effect_size = statistical_df['cohens_d'].abs().mean()
            logger.info(f"平均效应量 (|Cohen's d|): {mean_effect_size:.3f}")

            # 平均KL散度
            mean_kl = statistical_df['kl_divergence'].mean()
            logger.info(f"平均KL散度: {mean_kl:.3f}")

        logger.info(f"核心CSV报告已保存到: {output_dir}")
        logger.success("核心统计报告生成完成")

    except Exception as e:
        logger.error(f"核心统计报告生成失败: {e}")
        raise


if __name__ == "__main__":
    main()