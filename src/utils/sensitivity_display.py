"""
Sensitivity分析展示工具
从统计分析结果中提取并展示sensitivity指标
"""

import json
from pathlib import Path
from typing import Dict, List
from loguru import logger


def load_analysis(file_path: Path) -> Dict:
    """加载分析结果"""
    with open(file_path, 'r') as f:
        return json.load(f)


def display_sensitivity_analysis(analysis_file: Path, show_all_metrics: bool = False):
    """
    展示单个文件的sensitivity分析

    Args:
        analysis_file: 分析文件路径
        show_all_metrics: 是否显示所有metrics
    """
    data = load_analysis(analysis_file)

    experiment_info = data.get('experiment_info', {})
    prompt = experiment_info.get('prompt', 'unknown')
    strategy = experiment_info.get('strategy', 'unknown')

    print(f"\n{'='*80}")
    print(f"Sensitivity Analysis: {prompt.upper()} - {strategy.replace('_', '-').title()}")
    print(f"{'='*80}\n")

    degradation = data.get('performance_degradation', {})

    for classifier in ['svm', 'random_forest']:
        clf_label = 'SVM' if classifier == 'svm' else 'Random Forest'

        if classifier not in degradation:
            continue

        print(f"## {clf_label}\n")

        metrics_to_show = degradation[classifier].keys() if show_all_metrics else ['f1_score', 'accuracy']

        for metric in metrics_to_show:
            if metric not in degradation[classifier]:
                continue

            metric_data = degradation[classifier][metric]

            if 'sensitivity' not in metric_data:
                print(f"⚠️  {metric.upper()}: No sensitivity data available")
                continue

            sens = metric_data['sensitivity']
            baseline = metric_data['baseline_performance']

            metric_label = metric.replace('_', '-').upper()
            print(f"### {metric_label}")
            print(f"  Baseline (0% synthetic):     {baseline:.4f}")
            print(f"  Regression slope:            {sens['regression_slope']:+.6f} per 1% synthetic")
            print(f"  R² (linearity):              {sens['regression_r_squared']:.4f}")
            print(f"  p-value:                     {sens['regression_p_value']:.6f}")
            print(f"  Total change (0→{sens['max_ratio_tested']}%):     {sens['total_change_0_to_max']:+.4f}")
            print()

        print()


def create_sensitivity_table(base_dir: Path, prompts: List[str], strategies: List[str]):
    """
    创建sensitivity对比表

    Args:
        base_dir: 实验基础目录
        prompts: prompt列表
        strategies: strategy列表
    """
    print(f"\n{'='*100}")
    print("SENSITIVITY COMPARISON TABLE")
    print(f"{'='*100}\n")

    all_results = {}

    for prompt in prompts:
        all_results[prompt] = {}
        for strategy in strategies:
            analysis_file = base_dir / 'reports' / f'full_ceas08_gpt41mini_v1_{prompt}_{strategy}_statistical_analysis.json'

            if not analysis_file.exists():
                logger.warning(f"文件不存在: {analysis_file}")
                continue

            data = load_analysis(analysis_file)
            degradation = data.get('performance_degradation', {})
            all_results[prompt][strategy] = {}

            for classifier in ['svm', 'random_forest']:
                if classifier not in degradation:
                    continue

                all_results[prompt][strategy][classifier] = {}

                for metric in ['f1_score', 'accuracy']:
                    if metric not in degradation[classifier]:
                        continue

                    metric_data = degradation[classifier][metric]

                    if 'sensitivity' in metric_data:
                        sens = metric_data['sensitivity']
                        all_results[prompt][strategy][classifier][metric] = {
                            'slope': sens['regression_slope'],
                            'r_squared': sens['regression_r_squared']
                        }

    # 打印对比表 - F1 Score
    print("## F1-Score Sensitivity (Regression Slope per 1% Synthetic)\n")
    print("| Prompt   | Strategy     | SVM Slope  | SVM R²  | RF Slope   | RF R²   |")
    print("|----------|--------------|------------|---------|------------|---------|")

    for prompt in prompts:
        for strategy in strategies:
            if strategy in all_results.get(prompt, {}):
                svm_data = all_results[prompt][strategy].get('svm', {}).get('f1_score')
                rf_data = all_results[prompt][strategy].get('random_forest', {}).get('f1_score')

                if svm_data and rf_data:
                    print(f"| {prompt:8s} | {strategy:12s} | "
                          f"{svm_data['slope']:+.6f} | {svm_data['r_squared']:.4f} | "
                          f"{rf_data['slope']:+.6f} | {rf_data['r_squared']:.4f} |")

    print()
    print("## Accuracy Sensitivity (Regression Slope per 1% Synthetic)\n")
    print("| Prompt   | Strategy     | SVM Slope  | SVM R²  | RF Slope   | RF R²   |")
    print("|----------|--------------|------------|---------|------------|---------|")

    for prompt in prompts:
        for strategy in strategies:
            if strategy in all_results.get(prompt, {}):
                svm_data = all_results[prompt][strategy].get('svm', {}).get('accuracy')
                rf_data = all_results[prompt][strategy].get('random_forest', {}).get('accuracy')

                if svm_data and rf_data:
                    print(f"| {prompt:8s} | {strategy:12s} | "
                          f"{svm_data['slope']:+.6f} | {svm_data['r_squared']:.4f} | "
                          f"{rf_data['slope']:+.6f} | {rf_data['r_squared']:.4f} |")

    print()
    print("**Interpretation:**")
    print("  - Slope: Performance change per 1% increase in synthetic data ratio")
    print("  - Negative slope = performance decreases as synthetic ratio increases")
    print("  - Larger absolute value = more sensitive to synthetic data percentage")
    print("  - R² closer to 1.0 = more linear relationship")
    print()
