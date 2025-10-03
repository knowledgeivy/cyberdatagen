#!/usr/bin/env python3
"""
Step 6: 可视化脚本
负责生成实验结果的图表和可视化报告
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
        f"{config.name}_visualization.log"
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
    plt.rcParams['figure.figsize'] = viz_config.get('figure_size', [12, 8])

    # 设置颜色调色板
    palette = viz_config.get('color_palette', 'husl')
    sns.set_palette(palette)


def load_analysis_results(analysis_file: str) -> dict:
    """加载统计分析结果"""
    logger.info(f"加载统计分析结果: {analysis_file}")

    with open(analysis_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    return results


def create_performance_curves(analysis_results: dict, output_dir: str, experiment_name: str, config=None):
    """创建性能曲线图"""
    logger.info("创建性能曲线图")

    # 获取模型和prompt信息用于标题
    model_info = "GPT-4o-mini"  # 默认值
    prompt_info = "Original"   # 默认值

    if config and hasattr(config, 'llm'):
        if hasattr(config.llm, 'api_config') and 'openai' in config.llm.api_config:
            model_info = config.llm.api_config['openai'].get('model', 'GPT-4o-mini')

    # 简化标题信息
    title_suffix = f" ({model_info}, {prompt_info} Prompt)"

    descriptive_stats = analysis_results.get('descriptive_statistics', {})
    if not descriptive_stats:
        logger.warning("未找到描述性统计数据，跳过性能曲线图")
        return

    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']

    # 为每个分类器创建子图
    classifiers = set()
    for ratio_data in descriptive_stats.values():
        classifiers.update(ratio_data.keys())

    classifiers = sorted(list(classifiers))

    fig, axes = plt.subplots(len(classifiers), len(metrics), figsize=(20, 5 * len(classifiers)))
    if len(classifiers) == 1 and len(metrics) == 1:
        axes = np.array([[axes]])
    elif len(classifiers) == 1:
        axes = axes.reshape(1, -1)
    elif len(metrics) == 1:
        axes = axes.reshape(-1, 1)

    for i, classifier in enumerate(classifiers):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]

            # 收集数据
            ratios = []
            means = []
            stds = []
            ci_lowers = []
            ci_uppers = []

            for ratio in sorted(descriptive_stats.keys(), key=int):
                if (classifier in descriptive_stats[ratio] and
                        metric in descriptive_stats[ratio][classifier]):
                    stats = descriptive_stats[ratio][classifier][metric]

                    ratios.append(int(ratio))
                    means.append(stats['mean'])
                    stds.append(stats['std'])
                    ci_lowers.append(stats['ci_lower'])
                    ci_uppers.append(stats['ci_upper'])

            if not ratios:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{classifier} - {metric}{title_suffix}')
                continue

            # 绘制曲线
            ax.plot(ratios, means, 'o-', linewidth=2, markersize=6, label=metric)
            ax.fill_between(ratios, ci_lowers, ci_uppers, alpha=0.3)

            # 设置标题和标签
            ax.set_title(f'{classifier} - {metric}{title_suffix}')
            ax.set_xlabel('Synthetic Ratio (%)')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)

            # 设置x轴
            ax.set_xticks(ratios)
            ax.set_xticklabels(ratios)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{experiment_name}_performance_curves.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

    logger.info("性能曲线图创建完成")


def create_degradation_heatmap(analysis_results: dict, output_dir: str, experiment_name: str, config=None):
    """创建性能衰减热力图"""
    logger.info("创建性能衰减热力图")

    # 获取模型信息
    model_info = "GPT-4o-mini"
    if config and hasattr(config, 'llm'):
        if hasattr(config.llm, 'api_config') and 'openai' in config.llm.api_config:
            model_info = config.llm.api_config['openai'].get('model', 'GPT-4o-mini')

    degradation_analysis = analysis_results.get('performance_degradation', {})
    if not degradation_analysis:
        logger.warning("未找到性能衰减数据，跳过热力图")
        return

    # 准备数据
    classifiers = list(degradation_analysis.keys())
    metrics = ['f1_score', 'accuracy', 'precision', 'recall']

    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        # 收集数据
        data_matrix = []
        ratios = []

        # 获取所有ratio
        for classifier in classifiers:
            if metric in degradation_analysis[classifier]:
                classifier_ratios = list(degradation_analysis[classifier][metric]['ratios'].keys())
                ratios.extend(classifier_ratios)

        ratios = sorted(list(set(ratios)), key=int)

        # 构建矩阵
        for classifier in classifiers:
            row = []
            for ratio in ratios:
                if (metric in degradation_analysis[classifier] and
                        str(ratio) in degradation_analysis[classifier][metric]['ratios']):
                    degradation = degradation_analysis[classifier][metric]['ratios'][str(ratio)]['relative_degradation']
                    row.append(degradation * 100)  # 转换为百分比
                else:
                    row.append(np.nan)
            data_matrix.append(row)

        # 创建热力图
        data_df = pd.DataFrame(data_matrix, index=classifiers, columns=[f'{r}%' for r in ratios])

        sns.heatmap(data_df, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0,
                    ax=axes[i], cbar_kws={'label': 'Performance Degradation (%)'})
        axes[i].set_title(f'Performance Degradation - {metric.replace("_", " ").title()} ({model_info})')
        axes[i].set_xlabel('Synthetic Ratio')
        axes[i].set_ylabel('Classifier')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{experiment_name}_degradation_heatmap.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

    logger.info("性能衰减热力图创建完成")


def create_statistical_significance_plot(analysis_results: dict, output_dir: str, experiment_name: str, config=None):
    """创建统计显著性图"""
    logger.info("创建统计显著性图")

    # 获取模型信息
    model_info = "GPT-4o-mini"
    if config and hasattr(config, 'llm'):
        if hasattr(config.llm, 'api_config') and 'openai' in config.llm.api_config:
            model_info = config.llm.api_config['openai'].get('model', 'GPT-4o-mini')

    hypothesis_tests = analysis_results.get('hypothesis_tests', {})
    if not hypothesis_tests:
        logger.warning("未找到假设检验数据，跳过显著性图")
        return

    # 准备数据
    significance_data = []

    for ratio, ratio_tests in hypothesis_tests.items():
        if ratio == '_correction_info':
            continue

        for classifier, classifier_tests in ratio_tests.items():
            if classifier == '_correction_info':
                continue

            for metric, test_result in classifier_tests.items():
                if not isinstance(test_result, dict):
                    continue

                significance_data.append({
                    'ratio': int(ratio),
                    'classifier': classifier,
                    'metric': metric,
                    'p_value': test_result.get('t_p_value_corrected', test_result.get('t_p_value', np.nan)),
                    'significant': test_result.get('significant_t_corrected', test_result.get('significant_t', False)),
                    'effect_size': abs(test_result.get('cohens_d', 0)),
                    'mean_difference': test_result.get('mean_difference', 0)
                })

    if not significance_data:
        logger.warning("未找到有效的假设检验数据")
        return

    df = pd.DataFrame(significance_data)

    # 创建散点图：effect size vs p-value
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 子图1: Effect size vs p-value
    scatter = ax1.scatter(df['effect_size'], -np.log10(df['p_value']),
                          c=df['ratio'], cmap='viridis', alpha=0.6, s=50)
    ax1.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='α = 0.05')
    ax1.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium Effect')
    ax1.set_xlabel('Effect Size (|Cohen\'s d|)')
    ax1.set_ylabel('-log10(p-value)')
    ax1.set_title(f'Statistical Significance vs Effect Size ({model_info})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Synthetic Ratio (%)')

    # 子图2: 显著性计数
    sig_counts = df.groupby(['ratio', 'classifier'])['significant'].sum().reset_index()
    pivot_sig = sig_counts.pivot(index='classifier', columns='ratio', values='significant')

    sns.heatmap(pivot_sig, annot=True, fmt='d', cmap='Reds', ax=ax2)
    ax2.set_title(f'Number of Significant Tests by Ratio and Classifier ({model_info})')
    ax2.set_xlabel('Synthetic Ratio (%)')
    ax2.set_ylabel('Classifier')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{experiment_name}_statistical_significance.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

    logger.info("统计显著性图创建完成")


def create_interactive_dashboard(analysis_results: dict, output_dir: str, experiment_name: str, config=None):
    """创建交互式仪表板"""
    logger.info("创建交互式仪表板")

    # 获取模型信息
    model_info = "GPT-4o-mini"
    if config and hasattr(config, 'llm'):
        if hasattr(config.llm, 'api_config') and 'openai' in config.llm.api_config:
            model_info = config.llm.api_config['openai'].get('model', 'GPT-4o-mini')

    descriptive_stats = analysis_results.get('descriptive_statistics', {})
    if not descriptive_stats:
        logger.warning("未找到描述性统计数据，跳过交互式仪表板")
        return

    # 准备数据
    plot_data = []
    for ratio, ratio_data in descriptive_stats.items():
        for classifier, classifier_data in ratio_data.items():
            for metric, stats in classifier_data.items():
                plot_data.append({
                    'ratio': int(ratio),
                    'classifier': classifier,
                    'metric': metric,
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'ci_lower': stats['ci_lower'],
                    'ci_upper': stats['ci_upper']
                })

    df = pd.DataFrame(plot_data)

    # 创建子图
    metrics = df['metric'].unique()
    classifiers = df['classifier'].unique()

    fig = make_subplots(
        rows=len(metrics), cols=1,
        subplot_titles=[metric.replace('_', ' ').title() for metric in metrics],
        vertical_spacing=0.1
    )

    colors = px.colors.qualitative.Set1[:len(classifiers)]

    for i, metric in enumerate(metrics):
        metric_data = df[df['metric'] == metric]

        for j, classifier in enumerate(classifiers):
            classifier_data = metric_data[metric_data['classifier'] == classifier]

            if len(classifier_data) == 0:
                continue

            # 主线
            fig.add_trace(
                go.Scatter(
                    x=classifier_data['ratio'],
                    y=classifier_data['mean'],
                    mode='lines+markers',
                    name=f'{classifier}' if i == 0 else None,
                    line=dict(color=colors[j % len(colors)]),
                    marker=dict(size=8),
                    showlegend=(i == 0),
                    legendgroup=classifier
                ),
                row=i + 1, col=1
            )

            # 置信区间
            fig.add_trace(
                go.Scatter(
                    x=list(classifier_data['ratio']) + list(classifier_data['ratio'][::-1]),
                    y=list(classifier_data['ci_upper']) + list(classifier_data['ci_lower'][::-1]),
                    fill='toself',
                    fillcolor=colors[j % len(colors)],
                    opacity=0.2,
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=classifier
                ),
                row=i + 1, col=1
            )

        # 更新x轴
        fig.update_xaxes(title_text="Synthetic Ratio (%)", row=i + 1, col=1)
        fig.update_yaxes(title_text=metric.replace('_', ' ').title(), row=i + 1, col=1)

    fig.update_layout(
        height=300 * len(metrics),
        title_text=f"Performance Metrics Dashboard - {experiment_name} ({model_info})",
        hovermode='x unified'
    )

    # 保存为HTML
    html_file = os.path.join(output_dir, f'{experiment_name}_interactive_dashboard.html')
    fig.write_html(html_file)

    logger.info(f"交互式仪表板创建完成: {html_file}")


def create_summary_report(analysis_results: dict, output_dir: str, experiment_name: str, config=None):
    """创建总结报告"""
    logger.info("创建总结报告")

    # 获取模型和prompt信息
    model_info = "GPT-4o-mini"
    prompt_info = "Original"

    if config and hasattr(config, 'llm'):
        if hasattr(config.llm, 'api_config') and 'openai' in config.llm.api_config:
            model_info = config.llm.api_config['openai'].get('model', 'GPT-4o-mini')

    findings = analysis_results.get('summary_findings', {})
    experiment_info = analysis_results.get('experiment_info', {})

    # 创建HTML报告
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Experiment Analysis Report - {experiment_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 10px; }}
            .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; }}
            .metric {{ margin: 10px 0; padding: 10px; background-color: #f9f9f9; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .significant {{ color: #d9534f; font-weight: bold; }}
            .not-significant {{ color: #5cb85c; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Synthetic Spam Email Data Generation - Analysis Report</h1>
            <p><strong>Experiment:</strong> {experiment_info.get('experiment_name', experiment_name)}</p>
            <p><strong>LLM Model:</strong> {model_info}</p>
            <p><strong>Prompt Strategy:</strong> {prompt_info}</p>
            <p><strong>Analysis Date:</strong> {experiment_info.get('analysis_timestamp', 'N/A')}</p>
            <p><strong>Total Experiments:</strong> {experiment_info.get('total_experiments', 'N/A')}</p>
        </div>
    """

    # Baseline性能
    if findings.get('baseline_performance'):
        html_content += """
        <div class="section">
            <h2>Baseline Performance (0% Synthetic)</h2>
            <table>
                <tr><th>Classifier</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>AUC-ROC</th></tr>
        """

        for classifier, metrics in findings['baseline_performance'].items():
            accuracy = metrics.get('accuracy', 'N/A')
            precision = metrics.get('precision', 'N/A')
            recall = metrics.get('recall', 'N/A')
            f1_score = metrics.get('f1_score', 'N/A')
            auc_roc = metrics.get('auc_roc', 'N/A')

            # Format values
            accuracy_str = f"{accuracy:.4f}" if isinstance(accuracy, float) else str(accuracy)
            precision_str = f"{precision:.4f}" if isinstance(precision, float) else str(precision)
            recall_str = f"{recall:.4f}" if isinstance(recall, float) else str(recall)
            f1_score_str = f"{f1_score:.4f}" if isinstance(f1_score, float) else str(f1_score)
            auc_roc_str = f"{auc_roc:.4f}" if isinstance(auc_roc, float) else str(auc_roc)

            html_content += f"""
                <tr>
                    <td>{classifier}</td>
                    <td>{accuracy_str}</td>
                    <td>{precision_str}</td>
                    <td>{recall_str}</td>
                    <td>{f1_score_str}</td>
                    <td>{auc_roc_str}</td>
                </tr>
            """

        html_content += "</table></div>"

    # 显著的性能衰减
    if findings.get('significant_degradations'):
        html_content += """
        <div class="section">
            <h2>Significant Performance Degradations</h2>
            <table>
                <tr><th>Ratio</th><th>Classifier</th><th>Metric</th><th>Degradation</th><th>P-value</th><th>Effect Size</th></tr>
        """

        for degradation in findings['significant_degradations'][:10]:  # 显示前10个
            html_content += f"""
                <tr class="significant">
                    <td>{degradation['ratio']}%</td>
                    <td>{degradation['classifier']}</td>
                    <td>{degradation['metric']}</td>
                    <td>{degradation['degradation']:.4f}</td>
                    <td>{degradation['p_value']:.4f}</td>
                    <td>{degradation['effect_size']:.4f}</td>
                </tr>
            """

        html_content += "</table></div>"

    # 临界比例
    if findings.get('critical_ratios'):
        html_content += """
        <div class="section">
            <h2>Critical Degradation Ratios</h2>
            <p>The synthetic ratio at which performance degradation exceeds acceptable threshold (5%):</p>
        """

        for classifier, metrics in findings['critical_ratios'].items():
            html_content += f"<div class='metric'><strong>{classifier}:</strong> "
            for metric, critical_ratio in metrics.items():
                html_content += f"{metric}: {critical_ratio}%, "
            html_content = html_content.rstrip(', ') + "</div>"

        html_content += "</div>"

    # 最佳比例
    if findings.get('best_synthetic_ratios'):
        html_content += """
        <div class="section">
            <h2>Optimal Synthetic Ratios</h2>
            <p>The synthetic ratio with minimal performance degradation:</p>
        """

        for classifier, metrics in findings['best_synthetic_ratios'].items():
            html_content += f"<div class='metric'><strong>{classifier}:</strong> "
            for metric, best_info in metrics.items():
                html_content += f"{metric}: {best_info['ratio']}% (degradation: {best_info['degradation']:.4f}), "
            html_content = html_content.rstrip(', ') + "</div>"

        html_content += "</div>"

    # 总体结论
    if findings.get('overall_conclusions'):
        html_content += """
        <div class="section">
            <h2>Overall Conclusions</h2>
            <ul>
        """

        for conclusion in findings['overall_conclusions']:
            html_content += f"<li>{conclusion}</li>"

        html_content += "</ul></div>"

    html_content += """
    </body>
    </html>
    """

    # 保存HTML报告
    report_file = os.path.join(output_dir, f'{experiment_name}_summary_report.html')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info(f"总结报告创建完成: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='可视化脚本')
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
        '--analysis_file',
        type=str,
        help='统计分析结果文件路径'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='输出目录'
    )
    parser.add_argument(
        '--plot_types',
        type=str,
        nargs='+',
        default=['curves', 'heatmap', 'significance', 'dashboard', 'report'],
        choices=['curves', 'heatmap', 'significance', 'dashboard', 'report'],
        help='要生成的图表类型'
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

        # 确定文件路径
        experiment_name = args.experiment_name or config.name

        if args.analysis_file:
            analysis_file = args.analysis_file
        else:
            analysis_file = os.path.join(
                config.output.get('results_path', './output/results/'),
                f"{experiment_name}_statistical_analysis.json"
            )

        output_dir = args.output_dir or config.output.get('plots_path', './output/plots/')

        logger.info(f"开始创建可视化")
        logger.info(f"分析文件: {analysis_file}")
        logger.info(f"图表类型: {args.plot_types}")

        # 检查分析文件是否存在
        if not os.path.exists(analysis_file):
            logger.error(f"分析文件不存在: {analysis_file}")
            return

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 加载分析结果
        analysis_results = load_analysis_results(analysis_file)

        # 生成各种图表
        if 'curves' in args.plot_types:
            create_performance_curves(analysis_results, output_dir, experiment_name, config)

        if 'heatmap' in args.plot_types:
            create_degradation_heatmap(analysis_results, output_dir, experiment_name, config)

        if 'significance' in args.plot_types:
            create_statistical_significance_plot(analysis_results, output_dir, experiment_name, config)

        if 'dashboard' in args.plot_types:
            create_interactive_dashboard(analysis_results, output_dir, experiment_name, config)

        if 'report' in args.plot_types:
            create_summary_report(analysis_results, output_dir, experiment_name, config)

        logger.info("=" * 50)
        logger.info("可视化创建完成")
        logger.info("=" * 50)

        logger.info(f"图表保存目录: {output_dir}")

        # 列出生成的文件
        generated_files = []
        for file in os.listdir(output_dir):
            if file.startswith(experiment_name):
                generated_files.append(file)

        if generated_files:
            logger.info("生成的文件:")
            for file in sorted(generated_files):
                logger.info(f"  {file}")

        logger.success("可视化成功完成")

    except Exception as e:
        logger.error(f"可视化失败: {e}")
        raise


if __name__ == "__main__":
    main()