"""
论文表格生成器
生成包含均值±标准差的详细统计表格,用于论文写作
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
from loguru import logger


def extract_group_level_statistics(
    merged_results_file: str,
    prompt: str,
    strategy: str
) -> pd.DataFrame:
    """
    从merged_results中提取group级别的统计数据

    Args:
        merged_results_file: 合并的结果文件路径
        prompt: Prompt名称
        strategy: 策略名称

    Returns:
        DataFrame包含每个ratio×classifier×metric的所有group数据
    """
    logger.info(f"加载结果文件: {merged_results_file}")

    with open(merged_results_file, 'r') as f:
        data = json.load(f)

    results = data['results']

    # 过滤指定的prompt和strategy
    filtered_results = [
        r for r in results
        if r.get('prompt') == prompt and r.get('strategy') == strategy
    ]

    logger.info(f"过滤后的结果数: {len(filtered_results)}")

    # 组织数据
    rows = []

    for result in filtered_results:
        ratio = result.get('synthetic_ratio')
        group_id = result.get('group_id')

        for clf_name, clf_result in result.get('classifiers', {}).items():
            if not clf_result.get('success'):
                continue

            metrics = clf_result.get('metrics', {})

            for metric_name, metric_value in metrics.items():
                rows.append({
                    'prompt': prompt,
                    'strategy': strategy,
                    'synthetic_ratio': ratio,
                    'group_id': group_id,
                    'classifier': clf_name,
                    'metric': metric_name,
                    'value': metric_value
                })

    df = pd.DataFrame(rows)
    logger.info(f"提取了 {len(df)} 行数据")

    return df


def calculate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算汇总统计: 均值 ± 标准差

    Args:
        df: Group级别的数据

    Returns:
        汇总统计DataFrame
    """
    logger.info("计算汇总统计...")

    summary = df.groupby([
        'prompt', 'strategy', 'synthetic_ratio', 'classifier', 'metric'
    ])['value'].agg(['mean', 'std', 'count']).reset_index()

    # 格式化为 mean ± std
    summary['mean_std'] = summary.apply(
        lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}",
        axis=1
    )

    # 添加置信区间 (95% CI)
    # CI = mean ± 1.96 * (std / sqrt(n))
    import numpy as np
    summary['ci_lower'] = summary['mean'] - 1.96 * (summary['std'] / np.sqrt(summary['count']))
    summary['ci_upper'] = summary['mean'] + 1.96 * (summary['std'] / np.sqrt(summary['count']))
    summary['ci_95'] = summary.apply(
        lambda row: f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]",
        axis=1
    )

    logger.info(f"汇总统计完成: {len(summary)} 行")

    return summary


def generate_paper_table_wide_format(
    summary_df: pd.DataFrame,
    metric: str = 'f1_score',
    output_file: str = None
) -> pd.DataFrame:
    """
    生成论文用的宽格式表格

    格式:
    Ratio | SVM (mean±std) | SVM 95%CI | RF (mean±std) | RF 95%CI

    Args:
        summary_df: 汇总统计DataFrame
        metric: 要显示的指标
        output_file: 输出文件路径 (可选)

    Returns:
        宽格式表格
    """
    logger.info(f"生成宽格式表格 - 指标: {metric}")

    # 过滤指定metric
    df = summary_df[summary_df['metric'] == metric].copy()

    # Pivot表格
    table_mean_std = df.pivot_table(
        index=['prompt', 'strategy', 'synthetic_ratio'],
        columns='classifier',
        values='mean_std',
        aggfunc='first'
    ).reset_index()

    table_ci = df.pivot_table(
        index=['prompt', 'strategy', 'synthetic_ratio'],
        columns='classifier',
        values='ci_95',
        aggfunc='first'
    ).reset_index()

    # 合并
    table = table_mean_std.copy()

    # 重命名列
    for clf in ['svm', 'random_forest']:
        if clf in table.columns:
            table[f'{clf}_mean_std'] = table[clf]
            table[f'{clf}_ci_95'] = table_ci[clf]
            table.drop(columns=[clf], inplace=True)

    # 排序
    table = table.sort_values(['prompt', 'strategy', 'synthetic_ratio'])

    if output_file:
        table.to_csv(output_file, index=False)
        logger.success(f"表格已保存: {output_file}")

    return table


def generate_latex_table(
    summary_df: pd.DataFrame,
    prompt: str,
    strategy: str,
    metric: str = 'f1_score',
    output_file: str = None
) -> str:
    """
    生成LaTeX格式的表格

    Args:
        summary_df: 汇总统计DataFrame
        prompt: Prompt名称
        strategy: 策略名称
        metric: 指标名称
        output_file: 输出文件路径 (可选)

    Returns:
        LaTeX代码
    """
    logger.info(f"生成LaTeX表格: {prompt} - {strategy} - {metric}")

    # 过滤数据
    df = summary_df[
        (summary_df['prompt'] == prompt) &
        (summary_df['strategy'] == strategy) &
        (summary_df['metric'] == metric)
    ].copy()

    # 排序
    df = df.sort_values('synthetic_ratio')

    # 构建LaTeX
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append(f"\\caption{{Performance on {metric.upper()} - {prompt.capitalize()} Prompt, {strategy.replace('_', ' ').title()} Strategy}}")
    latex.append("\\label{tab:" + f"{prompt}_{strategy}_{metric}" + "}")
    latex.append("\\begin{tabular}{c|cc|cc}")
    latex.append("\\hline")
    latex.append("\\textbf{Synthetic} & \\multicolumn{2}{c|}{\\textbf{SVM}} & \\multicolumn{2}{c}{\\textbf{Random Forest}} \\\\")
    latex.append("\\textbf{Ratio (\\%)} & Mean $\\pm$ Std & 95\\% CI & Mean $\\pm$ Std & 95\\% CI \\\\")
    latex.append("\\hline")

    # 获取所有唯一的ratio
    unique_ratios = sorted(df['synthetic_ratio'].unique())

    for ratio in unique_ratios:
        ratio_int = int(ratio)

        # 获取SVM和RF的数据
        svm_data = df[(df['synthetic_ratio'] == ratio) & (df['classifier'] == 'svm')]
        rf_data = df[(df['synthetic_ratio'] == ratio) & (df['classifier'] == 'random_forest')]

        if not svm_data.empty and not rf_data.empty:
            svm_mean_std = svm_data.iloc[0]['mean_std'].replace('±', '$\\pm$')
            svm_ci = svm_data.iloc[0]['ci_95']
            rf_mean_std = rf_data.iloc[0]['mean_std'].replace('±', '$\\pm$')
            rf_ci = rf_data.iloc[0]['ci_95']

            latex.append(f"{ratio_int} & {svm_mean_std} & {svm_ci} & {rf_mean_std} & {rf_ci} \\\\")

    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    latex_code = "\n".join(latex)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(latex_code)
        logger.success(f"LaTeX表格已保存: {output_file}")

    return latex_code


def generate_all_paper_tables(
    merged_results_file: str,
    output_dir: str,
    prompts: List[str] = ['original', 'strong', 'weak'],
    strategies: List[str] = ['within_group', 'cross_group'],
    metrics: List[str] = ['f1_score', 'accuracy', 'precision', 'recall', 'auc_roc']
):
    """
    生成所有组合的论文表格

    Args:
        merged_results_file: 合并的结果文件
        output_dir: 输出目录
        prompts: Prompt列表
        strategies: 策略列表
        metrics: 指标列表
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("开始生成论文表格")
    logger.info("=" * 80)

    all_summaries = []

    # 为每个prompt×strategy组合提取数据
    for prompt in prompts:
        for strategy in strategies:
            logger.info(f"\n处理: {prompt} × {strategy}")

            # 提取group级别数据
            df = extract_group_level_statistics(
                merged_results_file,
                prompt,
                strategy
            )

            # 计算汇总统计
            summary = calculate_summary_statistics(df)
            all_summaries.append(summary)

            # 为每个metric生成宽格式表格
            for metric in metrics:
                wide_table = generate_paper_table_wide_format(
                    summary,
                    metric=metric,
                    output_file=str(output_path / f"table_{prompt}_{strategy}_{metric}.csv")
                )

            # 生成LaTeX表格 (只为F1-score)
            latex_code = generate_latex_table(
                summary,
                prompt=prompt,
                strategy=strategy,
                metric='f1_score',
                output_file=str(output_path / f"latex_{prompt}_{strategy}_f1.tex")
            )

    # 合并所有汇总统计
    logger.info("\n合并所有汇总统计...")
    combined_summary = pd.concat(all_summaries, ignore_index=True)

    # 保存完整汇总
    full_summary_file = output_path / "full_summary_statistics.csv"
    combined_summary.to_csv(full_summary_file, index=False)
    logger.success(f"完整汇总已保存: {full_summary_file}")

    # 生成对比表格 (所有prompt×strategy的F1-score)
    logger.info("\n生成对比表格...")
    comparison_df = combined_summary[combined_summary['metric'] == 'f1_score'].copy()

    comparison_table = comparison_df.pivot_table(
        index='synthetic_ratio',
        columns=['prompt', 'strategy', 'classifier'],
        values='mean_std',
        aggfunc='first'
    )

    comparison_file = output_path / "comparison_all_configs_f1.csv"
    comparison_table.to_csv(comparison_file)
    logger.success(f"对比表格已保存: {comparison_file}")

    logger.info("\n" + "=" * 80)
    logger.success("所有论文表格生成完成!")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 80)

    return combined_summary
