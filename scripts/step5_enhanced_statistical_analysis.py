#!/usr/bin/env python3
"""
Enhanced Statistical Analysis Script for Multi-Sample Evaluation Framework
Provides comprehensive statistical tests including KL divergence, t-tests, effect sizes
Generates CSV reports suitable for academic paper writing
"""

import argparse
import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, f_oneway, ks_2samp, mannwhitneyu
from scipy.spatial.distance import jensenshannon
from statsmodels.stats.multitest import multipletests
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from src.config.config_manager import load_config


def kl_divergence(p, q, epsilon=1e-10):
    """
    计算KL散度

    Args:
        p, q: 概率分布
        epsilon: 平滑参数避免log(0)

    Returns:
        float: KL散度值
    """
    p = np.array(p) + epsilon
    q = np.array(q) + epsilon

    # 归一化确保为概率分布
    p = p / np.sum(p)
    q = q / np.sum(q)

    return np.sum(p * np.log(p / q))


def js_divergence(p, q):
    """
    计算Jensen-Shannon散度

    Args:
        p, q: 概率分布

    Returns:
        float: JS散度值
    """
    return jensenshannon(p, q) ** 2


def cohens_d(x, y):
    """
    计算Cohen's d效应量

    Args:
        x, y: 两组数据

    Returns:
        float: Cohen's d值
    """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std


def load_classification_results(results_file: str) -> Dict[str, Any]:
    """加载分类结果"""
    logger.info(f"加载分类结果: {results_file}")

    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    logger.info(f"加载完成: {len(results['results'])} 个实验结果")
    return results


def organize_results_by_strategy_and_ratio(results: Dict[str, Any]) -> Dict[str, Dict[int, Dict[str, List[float]]]]:
    """
    按策略和ratio组织结果

    Returns:
        Dict: {strategy: {ratio: {classifier: {metric: [values]}}}}
    """
    logger.info("按策略和ratio组织结果")

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


def compute_enhanced_statistics(organized_results: Dict) -> pd.DataFrame:
    """
    计算增强统计指标

    Returns:
        pd.DataFrame: 包含所有统计指标的表格
    """
    logger.info("计算增强统计指标")

    stats_rows = []

    # 对每个策略计算统计
    for strategy, strategy_data in organized_results.items():
        baseline_ratio = 0  # 基线比较

        if baseline_ratio not in strategy_data:
            logger.warning(f"策略 {strategy} 缺少基线ratio {baseline_ratio}")
            continue

        baseline_data = strategy_data[baseline_ratio]

        for ratio in sorted(strategy_data.keys()):
            if ratio == baseline_ratio:
                continue

            ratio_data = strategy_data[ratio]

            # 对每个分类器计算统计
            for classifier in baseline_data.keys():
                if classifier not in ratio_data:
                    continue

                baseline_classifier = baseline_data[classifier]
                ratio_classifier = ratio_data[classifier]

                # 对每个指标计算统计
                for metric in baseline_classifier.keys():
                    if metric not in ratio_classifier:
                        continue

                    baseline_values = np.array(baseline_classifier[metric])
                    ratio_values = np.array(ratio_classifier[metric])

                    if len(baseline_values) == 0 or len(ratio_values) == 0:
                        continue

                    # 基本统计
                    baseline_mean = np.mean(baseline_values)
                    baseline_std = np.std(baseline_values, ddof=1)
                    ratio_mean = np.mean(ratio_values)
                    ratio_std = np.std(ratio_values, ddof=1)

                    # 性能变化
                    absolute_change = ratio_mean - baseline_mean
                    relative_change = (absolute_change / baseline_mean) * 100 if baseline_mean != 0 else 0

                    # t检验 (配对或独立)
                    if len(baseline_values) == len(ratio_values):
                        # 配对t检验
                        t_stat, t_pvalue = ttest_rel(baseline_values, ratio_values)
                        test_type = "paired_ttest"
                    else:
                        # 独立t检验
                        t_stat, t_pvalue = stats.ttest_ind(baseline_values, ratio_values)
                        test_type = "independent_ttest"

                    # Wilcoxon符号秩检验
                    if len(baseline_values) == len(ratio_values):
                        try:
                            w_stat, w_pvalue = wilcoxon(baseline_values, ratio_values)
                        except:
                            w_stat, w_pvalue = np.nan, np.nan
                    else:
                        # Mann-Whitney U检验
                        try:
                            w_stat, w_pvalue = mannwhitneyu(baseline_values, ratio_values, alternative='two-sided')
                        except:
                            w_stat, w_pvalue = np.nan, np.nan

                    # 效应量 (Cohen's d)
                    try:
                        effect_size = cohens_d(baseline_values, ratio_values)
                    except:
                        effect_size = np.nan

                    # Kolmogorov-Smirnov检验
                    try:
                        ks_stat, ks_pvalue = ks_2samp(baseline_values, ratio_values)
                    except:
                        ks_stat, ks_pvalue = np.nan, np.nan

                    # 分布相似性指标
                    try:
                        # 创建概率分布进行KL散度计算
                        # 使用直方图估计概率密度
                        hist_baseline, bins = np.histogram(baseline_values, bins=10, density=True)
                        hist_ratio, _ = np.histogram(ratio_values, bins=bins, density=True)

                        # 归一化
                        hist_baseline = hist_baseline / np.sum(hist_baseline)
                        hist_ratio = hist_ratio / np.sum(hist_ratio)

                        kl_div = kl_divergence(hist_baseline, hist_ratio)
                        js_div = js_divergence(hist_baseline, hist_ratio)
                    except:
                        kl_div, js_div = np.nan, np.nan

                    # 添加行数据
                    stats_rows.append({
                        'strategy': strategy,
                        'synthetic_ratio': ratio,
                        'classifier': classifier,
                        'metric': metric,
                        'baseline_mean': baseline_mean,
                        'baseline_std': baseline_std,
                        'baseline_n': len(baseline_values),
                        'test_mean': ratio_mean,
                        'test_std': ratio_std,
                        'test_n': len(ratio_values),
                        'absolute_change': absolute_change,
                        'relative_change_pct': relative_change,
                        't_statistic': t_stat,
                        't_pvalue': t_pvalue,
                        't_test_type': test_type,
                        'wilcoxon_statistic': w_stat,
                        'wilcoxon_pvalue': w_pvalue,
                        'cohens_d': effect_size,
                        'ks_statistic': ks_stat,
                        'ks_pvalue': ks_pvalue,
                        'kl_divergence': kl_div,
                        'js_divergence': js_div,
                        'is_significant_t': t_pvalue < 0.05 if not np.isnan(t_pvalue) else False,
                        'is_significant_wilcoxon': w_pvalue < 0.05 if not np.isnan(w_pvalue) else False,
                        'is_significant_ks': ks_pvalue < 0.05 if not np.isnan(ks_pvalue) else False
                    })

    df = pd.DataFrame(stats_rows)

    # 多重比较校正
    if len(df) > 0:
        # 对t检验p值进行校正
        valid_t_pvals = df['t_pvalue'].dropna()
        if len(valid_t_pvals) > 0:
            reject, pvals_corrected, _, _ = multipletests(valid_t_pvals, method='fdr_bh')
            df.loc[df['t_pvalue'].notna(), 't_pvalue_corrected'] = pvals_corrected
            df.loc[df['t_pvalue'].notna(), 'is_significant_t_corrected'] = reject

        # 对Wilcoxon检验p值进行校正
        valid_w_pvals = df['wilcoxon_pvalue'].dropna()
        if len(valid_w_pvals) > 0:
            reject, pvals_corrected, _, _ = multipletests(valid_w_pvals, method='fdr_bh')
            df.loc[df['wilcoxon_pvalue'].notna(), 'wilcoxon_pvalue_corrected'] = pvals_corrected
            df.loc[df['wilcoxon_pvalue'].notna(), 'is_significant_wilcoxon_corrected'] = reject

    logger.info(f"计算完成: {len(df)} 个统计比较")
    return df


def generate_summary_tables(stats_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """生成汇总表格"""
    logger.info("生成汇总表格")

    summary_tables = {}

    # 1. 策略间比较汇总
    strategy_summary = stats_df.groupby(['synthetic_ratio', 'classifier', 'metric']).agg({
        'absolute_change': ['mean', 'std'],
        'relative_change_pct': ['mean', 'std'],
        'cohens_d': ['mean', 'std'],
        'kl_divergence': ['mean', 'std'],
        'is_significant_t_corrected': 'sum'
    }).round(4)

    strategy_summary.columns = ['_'.join(col).strip() for col in strategy_summary.columns]
    summary_tables['strategy_comparison'] = strategy_summary.reset_index()

    # 2. 最佳ratio分析
    best_ratio_list = []
    for classifier in stats_df['classifier'].unique():
        for metric in stats_df['metric'].unique():
            subset = stats_df[(stats_df['classifier'] == classifier) &
                            (stats_df['metric'] == metric)]

            if len(subset) == 0:
                continue

            # 找到绝对变化最小的ratio
            best_idx = subset['absolute_change'].abs().idxmin()
            best_row = subset.loc[best_idx]

            best_ratio_list.append({
                'classifier': classifier,
                'metric': metric,
                'best_ratio': best_row['synthetic_ratio'],
                'best_strategy': best_row['strategy'],
                'absolute_change': best_row['absolute_change'],
                'relative_change_pct': best_row['relative_change_pct'],
                'cohens_d': best_row['cohens_d'],
                'is_significant': best_row.get('is_significant_t_corrected', False)
            })

    summary_tables['best_ratios'] = pd.DataFrame(best_ratio_list)

    # 3. 显著性检验汇总
    significance_summary = stats_df.groupby(['strategy', 'classifier', 'metric']).agg({
        'is_significant_t_corrected': 'sum',
        'is_significant_wilcoxon_corrected': 'sum',
        'cohens_d': ['mean', 'std']
    }).round(4)

    significance_summary.columns = ['_'.join(col).strip() for col in significance_summary.columns]
    summary_tables['significance_summary'] = significance_summary.reset_index()

    # 4. 效应量分析
    effect_size_bins = [-np.inf, 0.2, 0.5, 0.8, np.inf]
    effect_size_labels = ['Negligible', 'Small', 'Medium', 'Large']

    stats_df['effect_size_category'] = pd.cut(
        stats_df['cohens_d'].abs(),
        bins=effect_size_bins,
        labels=effect_size_labels,
        include_lowest=True
    )

    effect_summary = stats_df.groupby(['strategy', 'effect_size_category']).size().unstack(fill_value=0)
    summary_tables['effect_sizes'] = effect_summary

    logger.info(f"生成 {len(summary_tables)} 个汇总表格")
    return summary_tables


def save_csv_reports(stats_df: pd.DataFrame, summary_tables: Dict[str, pd.DataFrame],
                    output_dir: str, experiment_name: str):
    """保存CSV格式报告"""
    logger.info("保存CSV格式报告")

    os.makedirs(output_dir, exist_ok=True)

    # 保存主要统计表
    main_stats_file = os.path.join(output_dir, f"{experiment_name}_detailed_statistics.csv")
    stats_df.to_csv(main_stats_file, index=False)
    logger.info(f"详细统计表已保存: {main_stats_file}")

    # 保存汇总表格
    for table_name, table_df in summary_tables.items():
        table_file = os.path.join(output_dir, f"{experiment_name}_summary_{table_name}.csv")
        table_df.to_csv(table_file, index=False)
        logger.info(f"汇总表已保存: {table_file}")

    # 生成LaTeX就绪的表格
    latex_dir = os.path.join(output_dir, 'latex_tables')
    os.makedirs(latex_dir, exist_ok=True)

    for table_name, table_df in summary_tables.items():
        latex_file = os.path.join(latex_dir, f"{table_name}.tex")
        latex_table = table_df.to_latex(index=False, float_format="{:.4f}".format)

        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(latex_table)

        logger.info(f"LaTeX表格已保存: {latex_file}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced statistical analysis script')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Configuration file path'
    )
    parser.add_argument(
        '--results_file',
        type=str,
        help='Classification results file path'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Output directory for CSV reports'
    )

    args = parser.parse_args()

    try:
        # 加载配置
        logger.info("加载配置文件")
        config = load_config(args.config)

        # 设置路径
        results_file = args.results_file or os.path.join(
            config.output.get('results_path', './output/results/'),
            f"{config.name}_classification_results.json"
        )

        output_dir = args.output_dir or config.output.get('reports_path', './output/reports/')

        # 检查结果文件
        if not os.path.exists(results_file):
            logger.error(f"结果文件不存在: {results_file}")
            return

        logger.info("开始增强统计分析")
        logger.info(f"结果文件: {results_file}")
        logger.info(f"输出目录: {output_dir}")

        # 加载和组织数据
        results = load_classification_results(results_file)
        organized_results = organize_results_by_strategy_and_ratio(results)

        # 计算增强统计指标
        stats_df = compute_enhanced_statistics(organized_results)

        if len(stats_df) == 0:
            logger.warning("没有找到可比较的数据")
            return

        # 生成汇总表格
        summary_tables = generate_summary_tables(stats_df)

        # 保存CSV报告
        save_csv_reports(stats_df, summary_tables, output_dir, config.name)

        # 打印关键统计信息
        logger.info("=" * 50)
        logger.info("增强统计分析完成")
        logger.info("=" * 50)

        logger.info(f"总计算统计比较: {len(stats_df)}")

        # 显著性结果汇总
        if 'is_significant_t_corrected' in stats_df.columns:
            significant_count = stats_df['is_significant_t_corrected'].sum()
            logger.info(f"显著性差异 (t检验校正后): {significant_count}/{len(stats_df)}")

        # KL散度汇总
        if 'kl_divergence' in stats_df.columns:
            mean_kl = stats_df['kl_divergence'].mean()
            logger.info(f"平均KL散度: {mean_kl:.4f}")

        # 效应量汇总
        if 'cohens_d' in stats_df.columns:
            mean_effect = stats_df['cohens_d'].abs().mean()
            logger.info(f"平均效应量 (|Cohen's d|): {mean_effect:.4f}")

        logger.info(f"CSV报告已保存到: {output_dir}")
        logger.success("增强统计分析完成")

    except Exception as e:
        logger.error(f"增强统计分析失败: {e}")
        raise


if __name__ == "__main__":
    main()