#!/usr/bin/env python3
"""
Improved Step 6: Visualization Script
修复布局问题、移除heatmap、优化学术论文使用的可视化
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

# 设置matplotlib样式
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def load_analysis_results(analysis_file: str) -> dict:
    """加载统计分析结果"""
    logger.info(f"加载统计分析结果: {analysis_file}")

    with open(analysis_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_performance_curves(analysis_results: dict, output_dir: str, experiment_name: str, config=None):
    """创建改进的性能曲线图 - 针对学术论文优化"""
    logger.info("创建性能曲线图")

    descriptive_stats = analysis_results.get('descriptive_statistics', {})
    if not descriptive_stats:
        logger.warning("没有找到描述性统计数据")
        return

    # 获取实验参数
    model_info = "GPT-4.1-mini"
    dataset = "CEAS-08"
    n_groups = 3
    sample_size = 200

    if config:
        if hasattr(config, 'llm') and hasattr(config.llm, 'api_config') and 'openai' in config.llm.api_config:
            model_info = config.llm.api_config['openai'].get('model', 'GPT-4.1-mini')
        if hasattr(config, 'experiment') and hasattr(config.experiment, 'dataset'):
            dataset = config.experiment.dataset
        if hasattr(config, 'data'):
            n_groups = getattr(config.data, 'n_groups', 3)
            sample_size = getattr(config.data, 'sample_size_per_group', 200)

    # 收集分类器和指标
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']

    classifiers = set()
    for ratio_data in descriptive_stats.values():
        classifiers.update(ratio_data.keys())
    classifiers = sorted(list(classifiers))

    # 为每个分类器创建单独的图 - 更清晰的学术展示
    for classifier in classifiers:
        fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4))
        if len(metrics) == 1:
            axes = [axes]

        # 添加包含所有关键信息的总标题（只显示一次）
        exp_info = f"Dataset: {dataset}, Model: {model_info}, Groups: {n_groups}, Sample Size: {sample_size}"
        fig.suptitle(f'{classifier.upper()} Performance Across Synthetic Ratios\n{exp_info}',
                    fontsize=14, fontweight='bold', y=0.95)

        for j, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[j]

            # 收集数据
            ratios = []
            means = []
            ci_lowers = []
            ci_uppers = []

            for ratio in sorted(descriptive_stats.keys(), key=int):
                if (classifier in descriptive_stats[ratio] and
                    metric in descriptive_stats[ratio][classifier]):

                    stats = descriptive_stats[ratio][classifier][metric]
                    ratios.append(int(ratio))
                    means.append(stats['mean'])
                    ci_lowers.append(stats['ci_lower'])
                    ci_uppers.append(stats['ci_upper'])

            if not ratios:
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(metric_label, fontsize=12, fontweight='bold')
                continue

            # 绘制性能曲线 - 使用学术风格
            line_color = plt.cm.Set1(j)
            ax.plot(ratios, means, 'o-', linewidth=2.5, markersize=8,
                   color=line_color, markerfacecolor='white', markeredgecolor=line_color,
                   markeredgewidth=2)
            ax.fill_between(ratios, ci_lowers, ci_uppers, alpha=0.2, color=line_color)

            # 设置标题和标签
            ax.set_title(metric_label, fontsize=12, fontweight='bold')
            ax.set_xlabel('Synthetic Data Ratio (%)', fontsize=11)
            ax.set_ylabel('Score', fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--')

            # 美化坐标轴
            ax.set_xticks(ratios)
            ax.set_xticklabels([f'{r}%' for r in ratios])
            ax.tick_params(axis='both', which='major', labelsize=10)

            # 设置y轴范围以更好地展示变化
            y_min = min(ci_lowers) - 0.05
            y_max = max(ci_uppers) + 0.05
            ax.set_ylim(max(0, y_min), min(1.0, y_max))

        plt.tight_layout()
        plt.subplots_adjust(top=0.80)  # 为标题留空间

        # 保存图片
        filename = f'{experiment_name}_{classifier}_performance_curves.png'
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
        plt.close()

    logger.info("性能曲线图创建完成")


def create_degradation_bar_chart(analysis_results: dict, output_dir: str, experiment_name: str, config=None):
    """创建性能衰减条形图 - 替代heatmap，更适合学术论文"""
    logger.info("创建性能衰减条形图")

    degradation_analysis = analysis_results.get('degradation_analysis', {})
    if not degradation_analysis:
        logger.warning("没有找到性能衰减分析数据")
        return

    # 获取实验参数
    model_info = "GPT-4.1-mini"
    dataset = "CEAS-08"
    if config:
        if hasattr(config, 'llm') and hasattr(config.llm, 'api_config') and 'openai' in config.llm.api_config:
            model_info = config.llm.api_config['openai'].get('model', 'GPT-4.1-mini')
        if hasattr(config, 'experiment') and hasattr(config.experiment, 'dataset'):
            dataset = config.experiment.dataset

    # 准备数据
    data = []
    for classifier, classifier_data in degradation_analysis.items():
        for ratio, ratio_data in classifier_data.items():
            for metric, degradation in ratio_data.items():
                data.append({
                    'Classifier': classifier.upper(),
                    'Ratio': f'{ratio}%',
                    'Metric': metric.replace('_', ' ').title(),
                    'Degradation': degradation
                })

    if not data:
        logger.warning("没有性能衰减数据可视化")
        return

    df = pd.DataFrame(data)

    # 创建条形图
    classifiers = df['Classifier'].unique()
    metrics = df['Metric'].unique()

    fig, axes = plt.subplots(len(classifiers), 1, figsize=(14, 4 * len(classifiers)))
    if len(classifiers) == 1:
        axes = [axes]

    fig.suptitle(f'Performance Degradation Analysis\nDataset: {dataset}, Model: {model_info}',
                fontsize=14, fontweight='bold', y=0.98)

    for i, classifier in enumerate(classifiers):
        ax = axes[i]
        classifier_data = df[df['Classifier'] == classifier]

        # 创建分组条形图
        ratios = classifier_data['Ratio'].unique()
        x = np.arange(len(metrics))
        width = 0.15

        for j, ratio in enumerate(sorted(ratios, key=lambda x: int(x.rstrip('%')))):
            ratio_data = classifier_data[classifier_data['Ratio'] == ratio]
            degradations = []
            for metric in metrics:
                metric_data = ratio_data[ratio_data['Metric'] == metric]
                if not metric_data.empty:
                    degradations.append(metric_data['Degradation'].iloc[0])
                else:
                    degradations.append(0)

            offset = (j - len(ratios)/2 + 0.5) * width
            bars = ax.bar(x + offset, degradations, width, label=ratio, alpha=0.8)

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                if abs(height) > 0.01:  # 只显示有意义的数值
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005 if height >= 0 else height - 0.015,
                           f'{height:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)

        ax.set_title(f'{classifier} Performance Degradation', fontsize=12, fontweight='bold')
        ax.set_xlabel('Metrics', fontsize=11)
        ax.set_ylabel('Degradation', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend(title='Synthetic Ratio', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig(os.path.join(output_dir, f'{experiment_name}_degradation_analysis.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

    logger.info("性能衰减条形图创建完成")


def create_statistical_significance_plot(analysis_results: dict, output_dir: str, experiment_name: str, config=None):
    """创建统计显著性可视化"""
    logger.info("创建统计显著性图")

    hypothesis_tests = analysis_results.get('hypothesis_tests', {})
    if not hypothesis_tests:
        logger.warning("没有找到假设检验数据")
        return

    # 准备数据
    data = []
    for test_key, test_result in hypothesis_tests.items():
        if test_result.get('significant', False):
            parts = test_key.split('_vs_')
            if len(parts) == 2:
                ratio_classifier_metric = parts[1]
                parts2 = ratio_classifier_metric.split('_')
                if len(parts2) >= 3:
                    ratio = parts2[0]
                    classifier = parts2[1]
                    metric = '_'.join(parts2[2:])

                    data.append({
                        'Ratio': f'{ratio}%',
                        'Classifier': classifier.upper(),
                        'Metric': metric.replace('_', ' ').title(),
                        'P_value': test_result.get('p_value', 1.0),
                        'Effect_size': abs(test_result.get('effect_size', 0.0))
                    })

    if not data:
        logger.warning("没有显著的统计结果可视化")
        return

    df = pd.DataFrame(data)

    # 创建散点图
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # 按分类器分组绘制
    classifiers = df['Classifier'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(classifiers)))

    for i, classifier in enumerate(classifiers):
        classifier_data = df[df['Classifier'] == classifier]
        ax.scatter(classifier_data['Effect_size'], -np.log10(classifier_data['P_value']),
                  c=[colors[i]], label=classifier, alpha=0.7, s=60)

    # 添加显著性阈值线
    ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='α = 0.05')

    ax.set_xlabel('Effect Size (|Cohen\'s d|)', fontsize=12)
    ax.set_ylabel('-log₁₀(p-value)', fontsize=12)
    ax.set_title('Statistical Significance vs Effect Size', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{experiment_name}_statistical_significance.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

    logger.info("统计显著性图创建完成")


def create_summary_report(analysis_results: dict, output_dir: str, experiment_name: str, config=None):
    """创建改进的总结报告"""
    logger.info("创建总结报告")

    # 获取实验信息
    model_info = "GPT-4.1-mini"
    dataset = "CEAS-08"
    n_groups = 3
    sample_size = 200

    if config:
        if hasattr(config, 'llm') and hasattr(config.llm, 'api_config') and 'openai' in config.llm.api_config:
            model_info = config.llm.api_config['openai'].get('model', 'GPT-4.1-mini')
        if hasattr(config, 'experiment') and hasattr(config.experiment, 'dataset'):
            dataset = config.experiment.dataset
        if hasattr(config, 'data'):
            n_groups = getattr(config.data, 'n_groups', 3)
            sample_size = getattr(config.data, 'sample_size_per_group', 200)

    # 获取数据
    descriptive_stats = analysis_results.get('descriptive_statistics', {})
    hypothesis_tests = analysis_results.get('hypothesis_tests', {})
    degradation_analysis = analysis_results.get('degradation_analysis', {})

    # 计算总实验数
    total_experiments = sum(len(analysis_results.get('raw_data', {}).get(ratio, {}))
                           for ratio in descriptive_stats.keys()) if 'raw_data' in analysis_results else 0

    # 生成HTML报告
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Experiment Analysis Report - {experiment_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 10px; margin-bottom: 30px; }}
            .section {{ margin: 30px 0; padding: 20px; border-left: 4px solid #007acc; }}
            .metric {{ margin: 15px 0; padding: 15px; background-color: #f9f9f9; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; font-weight: bold; }}
            .significant {{ color: #d9534f; font-weight: bold; }}
            .not-significant {{ color: #5cb85c; }}
            .experiment-info {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Synthetic Spam Email Data Generation - Analysis Report</h1>
            <div class="experiment-info">
                <p><strong>Experiment:</strong> {experiment_name}</p>
                <p><strong>Dataset:</strong> {dataset}</p>
                <p><strong>LLM Model:</strong> {model_info}</p>
                <p><strong>Experimental Design:</strong> R={n_groups} groups, N={sample_size} samples per group</p>
                <p><strong>Analysis Date:</strong> {analysis_results.get('metadata', {}).get('analysis_date', 'N/A')}</p>
                <p><strong>Total Experiments:</strong> {total_experiments if total_experiments > 0 else 'N/A'}</p>
            </div>
        </div>
    """

    # 基线性能表
    if descriptive_stats and '0' in descriptive_stats:
        html_content += """
        <div class="section">
            <h2>Baseline Performance (0% Synthetic Data)</h2>
            <table>
                <tr><th>Classifier</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>AUC-ROC</th></tr>
        """

        baseline_stats = descriptive_stats['0']
        for classifier in sorted(baseline_stats.keys()):
            stats = baseline_stats[classifier]
            html_content += f"""
                <tr>
                    <td>{classifier.upper()}</td>
                    <td>{stats.get('accuracy', {}).get('mean', 0):.4f}</td>
                    <td>{stats.get('precision', {}).get('mean', 0):.4f}</td>
                    <td>{stats.get('recall', {}).get('mean', 0):.4f}</td>
                    <td>{stats.get('f1_score', {}).get('mean', 0):.4f}</td>
                    <td>{stats.get('auc_roc', {}).get('mean', 0):.4f}</td>
                </tr>
            """

        html_content += "</table></div>"

    # 显著性结果表
    significant_results = []
    for test_key, test_result in hypothesis_tests.items():
        if test_result.get('significant', False):
            parts = test_key.split('_vs_')
            if len(parts) == 2:
                ratio_classifier_metric = parts[1]
                parts2 = ratio_classifier_metric.split('_')
                if len(parts2) >= 3:
                    ratio = parts2[0]
                    classifier = parts2[1]
                    metric = '_'.join(parts2[2:])

                    # 获取性能衰减值
                    degradation = 0
                    if classifier in degradation_analysis and ratio in degradation_analysis[classifier]:
                        degradation = degradation_analysis[classifier][ratio].get(metric, 0)

                    significant_results.append({
                        'ratio': ratio,
                        'classifier': classifier,
                        'metric': metric,
                        'degradation': degradation,
                        'p_value': test_result.get('p_value', 1.0),
                        'effect_size': test_result.get('effect_size', 0.0)
                    })

    if significant_results:
        # 按p值排序
        significant_results.sort(key=lambda x: x['p_value'])

        html_content += """
        <div class="section">
            <h2>Significant Performance Degradations</h2>
            <table>
                <tr><th>Ratio</th><th>Classifier</th><th>Metric</th><th>Degradation</th><th>P-value</th><th>Effect Size</th></tr>
        """

        for result in significant_results[:20]:  # 显示前20个
            html_content += f"""
                <tr class="significant">
                    <td>{result['ratio']}%</td>
                    <td>{result['classifier'].upper()}</td>
                    <td>{result['metric'].replace('_', ' ').title()}</td>
                    <td>{result['degradation']:.4f}</td>
                    <td>{result['p_value']:.4f}</td>
                    <td>{result['effect_size']:.4f}</td>
                </tr>
            """

        html_content += "</table></div>"

    # 结论
    html_content += f"""
        <div class="section">
            <h2>Overall Conclusions</h2>
            <ul>
                <li>Baseline performance established across {len(descriptive_stats.get('0', {}))} classifiers</li>
                <li>Found {len(significant_results)} significant performance degradations</li>
                <li>Analysis conducted using {model_info} on {dataset} dataset</li>
                <li>Experimental design: R={n_groups} groups with N={sample_size} samples each</li>
            </ul>
        </div>
    </body>
    </html>
    """

    # 保存报告
    report_file = os.path.join(output_dir, f'{experiment_name}_summary_report.html')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info(f"总结报告创建完成: {report_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='改进的可视化脚本')
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--experiment_name', default='experiment', help='实验名称')
    parser.add_argument('--analysis_file', required=True, help='统计分析结果文件')
    parser.add_argument('--output_dir', default='./output/plots/', help='输出目录')
    parser.add_argument('--plot_types', nargs='+',
                       choices=['curves', 'degradation', 'significance', 'report'],
                       default=['curves', 'degradation', 'significance', 'report'],
                       help='要生成的图表类型')

    args = parser.parse_args()

    # 加载配置
    logger.info("加载配置文件")
    config = load_config(args.config)

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载分析结果
    analysis_results = load_analysis_results(args.analysis_file)

    logger.info("开始创建可视化")
    logger.info(f"分析文件: {args.analysis_file}")
    logger.info(f"图表类型: {args.plot_types}")

    # 生成可视化
    if 'curves' in args.plot_types:
        create_performance_curves(analysis_results, args.output_dir, args.experiment_name, config)

    if 'degradation' in args.plot_types:
        create_degradation_bar_chart(analysis_results, args.output_dir, args.experiment_name, config)

    if 'significance' in args.plot_types:
        create_statistical_significance_plot(analysis_results, args.output_dir, args.experiment_name, config)

    if 'report' in args.plot_types:
        create_summary_report(analysis_results, args.output_dir, args.experiment_name, config)

    logger.info("==================================================")
    logger.info("可视化创建完成")
    logger.info("==================================================")
    logger.info(f"图表保存目录: {args.output_dir}")

    # 列出生成的文件
    generated_files = []
    for file in os.listdir(args.output_dir):
        if args.experiment_name in file:
            generated_files.append(file)

    if generated_files:
        logger.info("生成的文件:")
        for file in sorted(generated_files):
            logger.info(f"  {file}")

    logger.success("可视化成功完成")


if __name__ == "__main__":
    main()