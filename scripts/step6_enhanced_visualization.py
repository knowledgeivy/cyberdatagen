#!/usr/bin/env python3
"""
Enhanced Step 6: Advanced Visualization Script
Generates comprehensive visualizations for multi-strategy experiment comparison
"""

import argparse
import sys
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
from src.config.config_manager import load_config


def setup_logging(config):
    """设置日志"""
    log_config = config.logging
    log_file = os.path.join(
        config.output.get('logs_path', './logs/'),
        f"{config.name}_enhanced_visualization.log"
    )

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger.add(
        log_file,
        level=log_config.get('level', 'INFO'),
        format=log_config.get('format', "{time} | {level} | {message}"),
        rotation=log_config.get('rotation', "100 MB"),
        retention=log_config.get('retention', "30 days")
    )


def setup_matplotlib_style():
    """设置学术期刊风格的matplotlib样式"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        # 高分辨率输出，适合打印
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'figure.figsize': [16, 10],

        # 字体设置 - 学术期刊标准
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,

        # 间距设置 - 增加标题间距
        'axes.titlepad': 20,  # 子图标题与图的间距
        'figure.subplot.hspace': 0.6,  # 子图间垂直间距
        'figure.subplot.wspace': 0.35,  # 子图间水平间距

        # 网格和线条
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.linewidth': 0.8,
        'lines.linewidth': 2,
        'lines.markersize': 8,

        # 图例
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'black',
        'legend.fancybox': False
    })


def load_classification_results(results_file: str) -> pd.DataFrame:
    """加载分类结果数据"""
    logger.info(f"加载分类结果: {results_file}")

    with open(results_file, 'r', encoding='utf-8') as f:
        results_data = json.load(f)

    # 转换为DataFrame以便分析
    data = []
    results = results_data.get('results', [])

    for result in results:
        for classifier, classifier_data in result['classifiers'].items():
            metrics = classifier_data.get('metrics', {})
            row = {
                'experiment_id': result['experiment_id'],
                'strategy': result['strategy'],
                'prompt_strategy': result.get('prompt_strategy', 'original'),  # 默认值
                'repetition': result['trial'],
                'group': result['group_id'],
                'synthetic_ratio': result['synthetic_ratio'],
                'classifier': classifier,
                **metrics
            }
            data.append(row)

    df = pd.DataFrame(data)
    logger.info(f"加载了 {len(df)} 条记录")
    return df


def create_strategy_comparison_plots(df: pd.DataFrame, output_dir: str, experiment_name: str, config=None):
    """创建策略对比图表"""
    logger.info("创建策略对比图表")

    # 调试：检查数据
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"Unique strategies: {df['strategy'].unique()}")
    logger.info(f"Unique classifiers: {df['classifier'].unique()}")
    logger.info(f"Unique synthetic ratios: {sorted(df['synthetic_ratio'].unique())}")

    # 检查每个策略的数据量
    for strategy in df['strategy'].unique():
        count = len(df[df['strategy'] == strategy])
        logger.info(f"Strategy '{strategy}': {count} records")

    # 获取模型信息
    model_info = "gpt-4.1-mini"
    if config and hasattr(config, 'llm'):
        if hasattr(config.llm, 'api_config') and 'openai' in config.llm.api_config:
            model_info = config.llm.api_config['openai'].get('model', 'gpt-4.1-mini')

    # 计算策略级别的统计数据
    strategy_stats = df.groupby(['strategy', 'classifier', 'synthetic_ratio']).agg({
        'accuracy': ['mean', 'std', 'count'],
        'precision': ['mean', 'std', 'count'],
        'recall': ['mean', 'std', 'count'],
        'f1_score': ['mean', 'std', 'count'],
        'auc_roc': ['mean', 'std', 'count']
    }).reset_index()

    logger.info(f"Strategy stats shape: {strategy_stats.shape}")

    # 扁平化列名
    strategy_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in strategy_stats.columns.values]
    strategy_stats.columns = [col.replace('__', '_').strip('_') for col in strategy_stats.columns]

    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    classifiers = df['classifier'].unique()
    strategies = df['strategy'].unique()

    # 学术风格的线型、标记和颜色配置（黑白打印友好）
    style_config = {
        'within_group': {
            'linestyle': '-',
            'marker': 'o',
            'color': '#2E86AB',  # 深蓝
            'label': 'Within Group'
        },
        'cross_group': {
            'linestyle': '--',
            'marker': 's',
            'color': '#A23B72',  # 深紫红
            'label': 'Cross Group'
        },
        'real_fixed_random_synthetic': {
            'linestyle': '-.',
            'marker': '^',
            'color': '#F18F01',  # 橙色
            'label': 'Real Fixed Random Synthetic'
        },
        'full_random': {
            'linestyle': ':',
            'marker': 'd',
            'color': '#C73E1D',  # 深红
            'label': 'Full Random'
        }
    }

    # 为每个metric创建对比图
    for metric in metrics:
        fig, axes = plt.subplots(1, len(classifiers), figsize=(20, 6))
        if len(classifiers) == 1:
            axes = [axes]

        for i, classifier in enumerate(classifiers):
            ax = axes[i]

            # 为每个策略绘制曲线
            for strategy in strategies:
                strategy_data = strategy_stats[
                    (strategy_stats['strategy'] == strategy) &
                    (strategy_stats['classifier'] == classifier)
                ].copy()

                if len(strategy_data) == 0:
                    logger.warning(f"No data for strategy={strategy}, classifier={classifier}, metric={metric}")
                    continue

                strategy_data = strategy_data.sort_values('synthetic_ratio')

                ratios = strategy_data['synthetic_ratio'].values
                means = strategy_data[f'{metric}_mean'].values
                stds = strategy_data[f'{metric}_std'].values

                logger.info(f"Plotting {strategy} for {classifier}: {len(ratios)} points")

                # 计算置信区间
                ci_lower = means - 1.96 * stds / np.sqrt(strategy_data[f'{metric}_count'].values)
                ci_upper = means + 1.96 * stds / np.sqrt(strategy_data[f'{metric}_count'].values)

                # 获取样式配置
                style = style_config.get(strategy, {
                    'linestyle': '-',
                    'marker': 'o',
                    'color': 'black',
                    'label': strategy.replace('_', ' ').title()
                })

                # 绘制曲线（学术风格：线型+标记+颜色）
                ax.plot(ratios, means,
                       linestyle=style['linestyle'],
                       marker=style['marker'],
                       color=style['color'],
                       label=style['label'],
                       linewidth=2,
                       markersize=8,
                       markerfacecolor='white',
                       markeredgewidth=1.5,
                       markeredgecolor=style['color'])

                # 置信区间阴影（半透明，与线条同色）
                ax.fill_between(ratios, ci_lower, ci_upper,
                               color=style['color'],
                               alpha=0.15)

            ax.set_title(f'{classifier.replace("_", " ").title()} - {metric.replace("_", " ").title()}',
                        fontsize=11, pad=10)
            ax.set_xlabel('Synthetic Ratio (%)', fontsize=10)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)

            # 设置x轴
            unique_ratios = sorted(df['synthetic_ratio'].unique())
            ax.set_xticks(unique_ratios)
            ax.set_xticklabels(unique_ratios)

        plt.suptitle(f'Strategy Comparison: {metric.replace("_", " ").title()} Performance ({model_info})',
                    fontsize=14, y=0.998)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, bottom=0.08, hspace=0.4)

        # 保存为高分辨率PNG
        plt.savefig(os.path.join(output_dir, f'{experiment_name}_strategy_comparison_{metric}.png'),
                   bbox_inches='tight', dpi=600, facecolor='white')
        plt.close()

    logger.info("策略对比图表创建完成")


def create_baseline_comparison_chart(df: pd.DataFrame, output_dir: str, experiment_name: str, config=None):
    """创建基线对比图表"""
    logger.info("创建基线对比图表")

    # 获取模型信息
    model_info = "gpt-4.1-mini"
    if config and hasattr(config, 'llm'):
        if hasattr(config.llm, 'api_config') and 'openai' in config.llm.api_config:
            model_info = config.llm.api_config['openai'].get('model', 'gpt-4.1-mini')

    # 提取baseline数据 (0% synthetic ratio)
    baseline_data = df[df['synthetic_ratio'] == 0].copy()

    if len(baseline_data) == 0:
        logger.warning("未找到基线数据，跳过基线对比图")
        return

    # 计算每个分类器的平均性能
    baseline_stats = baseline_data.groupby('classifier').agg({
        'accuracy': 'mean',
        'precision': 'mean',
        'recall': 'mean',
        'f1_score': 'mean',
        'auc_roc': 'mean'
    }).reset_index()

    # 创建雷达图
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    metric_labels = [m.replace('_', ' ').title() for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合圆圈

    colors = plt.cm.Set2(np.linspace(0, 1, len(baseline_stats)))

    for i, (_, row) in enumerate(baseline_stats.iterrows()):
        values = [row[metric] for metric in metrics]
        values += values[:1]  # 闭合圆圈

        ax.plot(angles, values, 'o-', linewidth=2, label=row['classifier'].replace('_', ' ').title(),
               color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1)
    ax.set_title(f'Baseline Performance Comparison - 0% Synthetic ({model_info})',
                size=14, fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.15))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{experiment_name}_baseline_radar.png'),
               bbox_inches='tight', dpi=600, facecolor='white')
    plt.close()

    logger.info("基线对比图表创建完成")


def create_degradation_heatmaps(df: pd.DataFrame, output_dir: str, experiment_name: str, config=None):
    """创建性能衰减热力图"""
    logger.info("创建性能衰减热力图")

    # 获取模型信息
    model_info = "gpt-4.1-mini"
    if config and hasattr(config, 'llm'):
        if hasattr(config.llm, 'api_config') and 'openai' in config.llm.api_config:
            model_info = config.llm.api_config['openai'].get('model', 'gpt-4.1-mini')

    # 计算性能衰减
    baseline_performance = df[df['synthetic_ratio'] == 0].groupby(['strategy', 'classifier']).agg({
        'accuracy': 'mean',
        'precision': 'mean',
        'recall': 'mean',
        'f1_score': 'mean',
        'auc_roc': 'mean'
    }).reset_index()

    all_performance = df.groupby(['strategy', 'classifier', 'synthetic_ratio']).agg({
        'accuracy': 'mean',
        'precision': 'mean',
        'recall': 'mean',
        'f1_score': 'mean',
        'auc_roc': 'mean'
    }).reset_index()

    # 合并数据计算衰减
    degradation_data = []

    for _, baseline_row in baseline_performance.iterrows():
        strategy = baseline_row['strategy']
        classifier = baseline_row['classifier']

        strategy_data = all_performance[
            (all_performance['strategy'] == strategy) &
            (all_performance['classifier'] == classifier)
        ]

        for _, perf_row in strategy_data.iterrows():
            if perf_row['synthetic_ratio'] == 0:
                continue

            for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
                degradation = (baseline_row[metric] - perf_row[metric]) / baseline_row[metric] * 100
                degradation_data.append({
                    'strategy': strategy,
                    'classifier': classifier,
                    'synthetic_ratio': perf_row['synthetic_ratio'],
                    'metric': metric,
                    'degradation_percent': degradation
                })

    if not degradation_data:
        logger.warning("未找到衰减数据，跳过热力图")
        return

    degradation_df = pd.DataFrame(degradation_data)

    # 为每个metric创建热力图
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        if i >= len(axes):
            break

        ax = axes[i]

        # 准备热力图数据
        metric_data = degradation_df[degradation_df['metric'] == metric]
        pivot_data = metric_data.pivot_table(
            index=['strategy', 'classifier'],
            columns='synthetic_ratio',
            values='degradation_percent',
            fill_value=0
        )

        # 创建热力图
        sns.heatmap(pivot_data, annot=True, cmap='RdYlBu_r', center=0,
                   ax=ax, fmt='.1f', cbar_kws={'label': 'Degradation (%)'})
        ax.set_title(f'{metric.replace("_", " ").title()} Degradation (%)', fontsize=12, pad=10)
        ax.set_xlabel('Synthetic Ratio (%)', fontsize=10)
        ax.set_ylabel('Strategy & Classifier', fontsize=10)

        # 旋转y轴标签以避免重叠
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=9)

    # 隐藏多余的子图
    for i in range(len(metrics), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f'Performance Degradation Heatmaps ({model_info})', fontsize=16, y=0.998)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.05, hspace=0.5)
    plt.savefig(os.path.join(output_dir, f'{experiment_name}_degradation_heatmaps.png'),
               bbox_inches='tight', dpi=600, facecolor='white')
    plt.close()

    logger.info("性能衰减热力图创建完成")


def create_best_strategy_summary(df: pd.DataFrame, output_dir: str, experiment_name: str, config=None):
    """创建最佳策略汇总表"""
    logger.info("创建最佳策略汇总表")

    # 为每个分类器和合成比例找到最佳策略
    best_strategies = []

    for classifier in df['classifier'].unique():
        for ratio in sorted(df['synthetic_ratio'].unique()):
            if ratio == 0:  # 跳过基线
                continue

            ratio_data = df[
                (df['classifier'] == classifier) &
                (df['synthetic_ratio'] == ratio)
            ].groupby('strategy').agg({
                'f1_score': 'mean',
                'accuracy': 'mean',
                'precision': 'mean',
                'recall': 'mean',
                'auc_roc': 'mean'
            }).reset_index()

            if len(ratio_data) == 0:
                continue

            # 找到F1-score最高的策略
            best_idx = ratio_data['f1_score'].idxmax()
            best_strategy = ratio_data.loc[best_idx]

            best_strategies.append({
                'classifier': classifier,
                'synthetic_ratio': ratio,
                'best_strategy': best_strategy['strategy'],
                'f1_score': best_strategy['f1_score'],
                'accuracy': best_strategy['accuracy'],
                'precision': best_strategy['precision'],
                'recall': best_strategy['recall'],
                'auc_roc': best_strategy['auc_roc']
            })

    if not best_strategies:
        logger.warning("未找到最佳策略数据")
        return

    best_df = pd.DataFrame(best_strategies)

    # 创建可视化
    fig, axes = plt.subplots(1, len(df['classifier'].unique()), figsize=(20, 6))
    if len(df['classifier'].unique()) == 1:
        axes = [axes]

    for i, classifier in enumerate(df['classifier'].unique()):
        ax = axes[i]

        classifier_data = best_df[best_df['classifier'] == classifier]

        if len(classifier_data) == 0:
            continue

        # 创建堆叠条形图显示每个比例的最佳策略
        strategies = classifier_data['best_strategy'].unique()
        strategy_colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
        strategy_color_map = dict(zip(strategies, strategy_colors))

        ratios = sorted(classifier_data['synthetic_ratio'].unique())
        bottoms = np.zeros(len(ratios))

        for strategy in strategies:
            heights = []
            for ratio in ratios:
                if strategy in classifier_data[classifier_data['synthetic_ratio'] == ratio]['best_strategy'].values:
                    heights.append(1)
                else:
                    heights.append(0)

            ax.bar(ratios, heights, bottom=bottoms, label=strategy.replace('_', ' ').title(),
                  color=strategy_color_map[strategy], alpha=0.8)
            bottoms += heights

        ax.set_title(f'{classifier.replace("_", " ").title()}\nBest Strategy by Ratio', fontsize=12)
        ax.set_xlabel('Synthetic Ratio (%)', fontsize=10)
        ax.set_ylabel('Best Strategy', fontsize=10)
        ax.legend(fontsize=9)
        ax.set_xticks(ratios)
        ax.set_xticklabels(ratios)

    plt.suptitle('Best Performing Strategy by Synthetic Ratio', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(os.path.join(output_dir, f'{experiment_name}_best_strategies.png'),
               bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()

    # 保存详细表格
    best_df.to_csv(os.path.join(output_dir, f'{experiment_name}_best_strategies.csv'), index=False)

    logger.info("最佳策略汇总创建完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Enhanced visualization script')
    parser.add_argument('--config', required=True, help='Configuration file path')
    parser.add_argument('--results', required=True, help='Classification results file path')

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    setup_logging(config)
    setup_matplotlib_style()

    logger.info("开始增强可视化分析")

    # 创建输出目录
    output_dir = config.output.get('plots_path', './output/plots/')
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    df = load_classification_results(args.results)

    experiment_name = config.name

    # 创建各种可视化
    create_strategy_comparison_plots(df, output_dir, experiment_name, config)
    create_baseline_comparison_chart(df, output_dir, experiment_name, config)
    create_degradation_heatmaps(df, output_dir, experiment_name, config)
    # create_best_strategy_summary(df, output_dir, experiment_name, config)  # 用户不需要

    logger.info(f"所有可视化已保存到: {output_dir}")
    logger.info("增强可视化分析完成")


if __name__ == "__main__":
    main()