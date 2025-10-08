"""
CSV导出工具
将统计分析结果导出为CSV格式，便于查看和分析
"""

import pandas as pd
from pathlib import Path
from typing import Dict
from loguru import logger


def export_sensitivity_to_csv(analysis_file: Path, output_csv: Path):
    """
    导出sensitivity分析结果到CSV

    Args:
        analysis_file: 分析JSON文件路径
        output_csv: 输出CSV文件路径
    """
    import json

    with open(analysis_file, 'r') as f:
        data = json.load(f)

    experiment_info = data.get('experiment_info', {})
    prompt = experiment_info.get('prompt', 'unknown')
    strategy = experiment_info.get('strategy', 'unknown')

    degradation = data.get('performance_degradation', {})
    if not degradation:
        logger.warning(f"无performance_degradation数据: {analysis_file}")
        return

    rows = []

    for classifier in ['svm', 'random_forest']:
        if classifier not in degradation:
            continue

        for metric in ['f1_score', 'accuracy', 'precision', 'recall', 'auc_roc']:
            if metric not in degradation[classifier]:
                continue

            metric_data = degradation[classifier][metric]
            baseline = metric_data.get('baseline_performance', 0)
            sensitivity = metric_data.get('sensitivity', {})

            # 添加总体sensitivity信息
            rows.append({
                'prompt': prompt,
                'strategy': strategy,
                'classifier': classifier,
                'metric': metric,
                'synthetic_ratio': 'sensitivity',
                'performance': baseline,
                'regression_slope': sensitivity.get('regression_slope', ''),
                'regression_r_squared': sensitivity.get('regression_r_squared', ''),
                'regression_p_value': sensitivity.get('regression_p_value', ''),
                'total_change': sensitivity.get('total_change_0_to_max', ''),
                'absolute_degradation': '',
                'relative_degradation': '',
                'p_value': ''
            })

            # 添加每个ratio的详细信息
            ratios = metric_data.get('ratios', {})
            for ratio_str in sorted(ratios.keys(), key=int):
                ratio_data = ratios[ratio_str]

                # 获取对应的hypothesis test p-value
                hypothesis_tests = data.get('hypothesis_tests', {})
                p_value = ''
                if ratio_str in hypothesis_tests:
                    if classifier in hypothesis_tests[ratio_str]:
                        if metric in hypothesis_tests[ratio_str][classifier]:
                            p_value = hypothesis_tests[ratio_str][classifier][metric].get('t_p_value_corrected', '')

                rows.append({
                    'prompt': prompt,
                    'strategy': strategy,
                    'classifier': classifier,
                    'metric': metric,
                    'synthetic_ratio': int(ratio_str),
                    'performance': ratio_data.get('performance', ''),
                    'regression_slope': '',
                    'regression_r_squared': '',
                    'regression_p_value': '',
                    'total_change': '',
                    'absolute_degradation': ratio_data.get('absolute_degradation', ''),
                    'relative_degradation': ratio_data.get('relative_degradation', ''),
                    'p_value': p_value
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    logger.info(f"已导出CSV: {output_csv}")


def export_all_analyses_to_csv(base_dir, experiment_name: str, prompts: list, strategies: list):
    """
    批量导出所有分析结果到CSV

    Args:
        base_dir: 实验基础目录 (str或Path)
        experiment_name: 实验名称
        prompts: prompt列表
        strategies: strategy列表
    """
    base_dir = Path(base_dir)
    reports_dir = base_dir / 'reports'
    csv_dir = base_dir / 'reports' / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)

    logger.info("开始批量导出CSV...")

    for prompt in prompts:
        for strategy in strategies:
            analysis_file = reports_dir / f'{experiment_name}_{prompt}_{strategy}_statistical_analysis.json'

            if not analysis_file.exists():
                logger.warning(f"跳过 - 文件不存在: {analysis_file}")
                continue

            output_csv = csv_dir / f'{experiment_name}_{prompt}_{strategy}_detailed.csv'

            try:
                export_sensitivity_to_csv(analysis_file, output_csv)
            except Exception as e:
                logger.error(f"导出失败 {prompt}-{strategy}: {e}")

    # 合并所有CSV到一个大文件
    all_csvs = list(csv_dir.glob(f'{experiment_name}_*_detailed.csv'))
    if all_csvs:
        dfs = [pd.read_csv(f) for f in all_csvs]
        merged_df = pd.concat(dfs, ignore_index=True)

        merged_csv = csv_dir / f'{experiment_name}_all_results.csv'
        merged_df.to_csv(merged_csv, index=False)
        logger.success(f"已合并所有结果到: {merged_csv}")

    logger.success(f"CSV导出完成，位置: {csv_dir}/")
