#!/usr/bin/env python3
"""
Step 6: Combined Visualization - 生成prompt对比图
把original, strong, weak三个prompt的性能曲线垂直排列在一起进行对比
"""

import argparse
import sys
import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
from src.config.config_manager import load_config


def setup_logging(config):
    """设置日志"""
    log_config = config.logging
    log_file = os.path.join(
        config.output.get('logs_path', './logs/'),
        f"{config.name}_combined_visualization.log"
    )

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger.add(
        log_file,
        level=log_config.get('level', 'INFO'),
        format=log_config.get('format', "{time} | {level} | {message}"),
        rotation=log_config.get('rotation', "100 MB"),
        retention=log_config.get('retention', "30 days")
    )


def setup_matplotlib_style(config):
    """设置matplotlib样式"""
    viz_config = config.output.get('visualization', {})

    plt.style.use(viz_config.get('style', 'seaborn-v0_8'))
    plt.rcParams['figure.dpi'] = viz_config.get('dpi', 300)
    plt.rcParams['savefig.dpi'] = viz_config.get('dpi', 300)

    # 设置颜色调色板
    palette = viz_config.get('color_palette', 'husl')
    sns.set_palette(palette)


def load_analysis(file_path):
    """加载分析结果"""
    logger.info(f"加载分析结果: {file_path}")
    with open(file_path, 'r') as f:
        return json.load(f)


def create_prompt_comparison_plot(base_dir: str, strategy: str, experiment_name: str, config=None):
    """
    为指定strategy创建prompt对比图

    Args:
        base_dir: 基础目录
        strategy: within_group 或 cross_group
        experiment_name: 实验名称
        config: 配置对象
    """
    logger.info(f"创建 {strategy} 的prompt对比图...")

    prompts = ['original', 'strong', 'weak']
    classifiers = ['svm', 'random_forest']
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']

    metric_labels = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1-Score',
        'auc_roc': 'AUC-ROC'
    }

    classifier_labels = {
        'svm': 'SVM',
        'random_forest': 'Random Forest'
    }

    # 加载所有prompt的数据
    all_data = {}
    reports_dir = os.path.join(base_dir, 'reports')

    for prompt in prompts:
        analysis_file = os.path.join(reports_dir, f'{experiment_name}_{prompt}_{strategy}_statistical_analysis.json')
        if not os.path.exists(analysis_file):
            logger.error(f"文件不存在: {analysis_file}")
            continue
        all_data[prompt] = load_analysis(analysis_file)

    if len(all_data) != 3:
        logger.error(f"无法加载所有prompt的数据 (只加载了 {len(all_data)}/3)")
        return

    # 创建figure - 3行(prompts) × 5列(metrics)
    # 每个subplot中用不同颜色/线型绘制两个classifier
    fig = plt.figure(figsize=(20, 10))

    # 定义classifier的颜色和线型
    clf_styles = {
        'svm': {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o'},
        'random_forest': {'color': '#ff7f0e', 'linestyle': '--', 'marker': 's'}
    }

    # 为每个prompt创建一行
    for prompt_idx, prompt in enumerate(prompts):
        data = all_data[prompt]

        # 为每个metric创建子图
        for metric_idx, metric in enumerate(metrics):
            # 计算subplot位置：3行 × 5列
            subplot_idx = prompt_idx * 5 + metric_idx + 1
            ax = plt.subplot(3, 5, subplot_idx)

            # 为每个classifier绘制曲线
            for clf in classifiers:
                # 提取数据
                ratios = []
                means = []
                ci_lowers = []
                ci_uppers = []

                descriptive_stats = data.get('descriptive_statistics', {})
                for ratio_str, ratio_data in sorted(descriptive_stats.items(),
                                                   key=lambda x: int(x[0])):
                    ratio = int(ratio_str)
                    clf_data = ratio_data.get(clf, {})
                    metric_data = clf_data.get(metric, {})

                    if metric_data:
                        ratios.append(ratio)
                        means.append(metric_data['mean'])
                        ci_lowers.append(metric_data['ci_lower'])
                        ci_uppers.append(metric_data['ci_upper'])

                # 绘制曲线
                if ratios:
                    style = clf_styles[clf]
                    ax.plot(ratios, means,
                           color=style['color'],
                           linestyle=style['linestyle'],
                           marker=style['marker'],
                           linewidth=2,
                           markersize=5,
                           label=classifier_labels[clf],
                           alpha=0.9)
                    ax.fill_between(ratios, ci_lowers, ci_uppers,
                                   color=style['color'],
                                   alpha=0.15)

            # 设置y轴范围为0-1.1（顶部留空间），但只显示0-1.0的刻度
            ax.set_ylim([0, 1.1])
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

            # 设置标题（只在第一行显示metric名称）
            if prompt_idx == 0:
                ax.set_title(metric_labels[metric], fontsize=11, fontweight='bold')

            # 设置y轴标签（只在第一列显示prompt名称）
            if metric_idx == 0:
                ax.set_ylabel(f'{prompt.capitalize()}\n{metric_labels[metric]}',
                             fontsize=10, fontweight='bold')
            else:
                ax.set_ylabel(metric_labels[metric], fontsize=9)

            # 设置x轴标签（只在最后一行显示）
            if prompt_idx == 2:
                ax.set_xlabel('Synthetic Ratio (%)', fontsize=9)
            else:
                ax.set_xlabel('')

            # 添加图例（只在第一行最后一列添加）
            if prompt_idx == 0 and metric_idx == 4:
                ax.legend(loc='lower right', fontsize=9, framealpha=0.9)

            # 网格 - 与step6_visualization.py保持一致
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)

    # 设置总标题
    strategy_title = strategy.replace('_', '-').title()
    fig.suptitle(f'Performance Comparison Across Prompts - {strategy_title} Strategy',
                fontsize=16, fontweight='bold', y=0.995)

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # 保存图片到combined子文件夹
    output_dir = os.path.join(base_dir, 'plots', 'combined')
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f'prompts_comparison_{strategy}_performance_curves.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"已保存: {output_file}")
    file_size = Path(output_file).stat().st_size / 1024
    logger.info(f"文件大小: {file_size:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description='生成prompt对比图（combined visualization）')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='配置文件路径'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        help='实验名称'
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        help='实验基础目录（包含reports和plots子文件夹）'
    )
    parser.add_argument(
        '--strategies',
        type=str,
        nargs='+',
        default=['within_group', 'cross_group'],
        choices=['within_group', 'cross_group'],
        help='要生成的策略'
    )

    args = parser.parse_args()

    try:
        # 加载配置
        logger.info("加载配置文件")
        config = load_config(args.config)

        # 设置日志
        setup_logging(config)

        # 设置matplotlib样式
        setup_matplotlib_style(config)

        # 确定参数
        experiment_name = args.experiment_name or config.name

        if args.base_dir:
            base_dir = args.base_dir
        else:
            # 默认基础目录
            output_base = config.output.get('base_path', './output/full_experiments/')
            base_dir = os.path.join(output_base, experiment_name.replace('_v1', ''))

        logger.info(f"开始创建prompt对比图")
        logger.info(f"基础目录: {base_dir}")
        logger.info(f"策略: {args.strategies}")

        # 为每个strategy创建对比图
        for strategy in args.strategies:
            create_prompt_comparison_plot(base_dir, strategy, experiment_name, config)

        logger.info("=" * 50)
        logger.info("✅ 所有prompt对比图已生成")
        logger.info("=" * 50)

        # 列出生成的文件
        combined_dir = os.path.join(base_dir, 'plots', 'combined')
        if os.path.exists(combined_dir):
            logger.info("生成的文件:")
            for file in sorted(os.listdir(combined_dir)):
                if file.endswith('.png'):
                    logger.info(f"  {file}")

        logger.success("Combined visualization成功完成")

    except Exception as e:
        logger.error(f"生成combined图失败: {e}")
        raise


if __name__ == "__main__":
    main()
