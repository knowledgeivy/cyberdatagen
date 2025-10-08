"""
统计分析模块
负责对分类实验结果进行统计分析和假设检验
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, f_oneway
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from ..config.config_manager import ExperimentConfig


class StatisticalAnalyzer:
    """统计分析器"""

    def __init__(self, config: ExperimentConfig):
        """
        初始化统计分析器

        Args:
            config: 实验配置
        """
        self.config = config
        self.eval_config = config.evaluation

    def load_classification_results(self, results_file: str) -> Dict[str, Any]:
        """
        加载分类结果

        Args:
            results_file: 结果文件路径

        Returns:
            Dict[str, Any]: 分类结果
        """
        logger.info(f"加载分类结果: {results_file}")

        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

        logger.info(f"加载完成: {len(results['results'])} 个实验结果")
        return results

    def organize_results_by_ratio(self, results: Dict[str, Any], group_by_prompt: bool = True) -> Dict[str, Dict[str, List[float]]]:
        """
        按synthetic ratio组织结果

        Args:
            results: 分类结果
            group_by_prompt: 是否按prompt分组

        Returns:
            Dict: 按ratio组织的结果
                  如果group_by_prompt=True: {prompt: {ratio: {classifier: {metric: [values]}}}}
                  如果group_by_prompt=False: {ratio: {classifier: {metric: [values]}}}
        """
        logger.info(f"按synthetic ratio组织结果 (group_by_prompt={group_by_prompt})")

        organized = {}

        for result in results['results']:
            ratio = result['synthetic_ratio']
            prompt = result.get('prompt', 'original')
            strategy = result.get('strategy', 'unknown')

            if group_by_prompt:
                # 按 prompt -> ratio -> classifier -> metric 组织
                if prompt not in organized:
                    organized[prompt] = {}

                if ratio not in organized[prompt]:
                    organized[prompt][ratio] = {}

                current_level = organized[prompt][ratio]
            else:
                # 按 ratio -> classifier -> metric 组织
                if ratio not in organized:
                    organized[ratio] = {}

                current_level = organized[ratio]

            for classifier_name, classifier_result in result['classifiers'].items():
                if not classifier_result['success']:
                    continue

                if classifier_name not in current_level:
                    current_level[classifier_name] = {}

                metrics = classifier_result['metrics']
                for metric_name, metric_value in metrics.items():
                    if metric_name not in current_level[classifier_name]:
                        current_level[classifier_name][metric_name] = []

                    current_level[classifier_name][metric_name].append(metric_value)

        if group_by_prompt:
            logger.info(f"组织完成: {len(organized)} 个prompt")
        else:
            logger.info(f"组织完成: {len(organized)} 个ratio")
        return organized

    def compute_descriptive_statistics(self, organized_results: Dict, has_prompt_level: bool = True) -> Dict[str, Any]:
        """
        计算描述性统计

        Args:
            organized_results: 按ratio组织的结果
            has_prompt_level: 数据是否包含prompt层级

        Returns:
            Dict[str, Any]: 描述性统计
        """
        logger.info("计算描述性统计")

        descriptive_stats = {}

        def compute_stats_for_ratio_data(ratio_data):
            """计算单个ratio的统计数据"""
            ratio_stats = {}
            for classifier_name, classifier_data in ratio_data.items():
                classifier_stats = {}

                for metric_name, metric_values in classifier_data.items():
                    if len(metric_values) == 0:
                        continue

                    values = np.array(metric_values)
                    classifier_stats[metric_name] = {
                        'count': len(values),
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values, ddof=1)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'median': float(np.median(values)),
                        'q25': float(np.percentile(values, 25)),
                        'q75': float(np.percentile(values, 75)),
                        'sem': float(stats.sem(values)),  # 标准误
                        'ci_lower': float(np.mean(values) - 1.96 * stats.sem(values)),
                        'ci_upper': float(np.mean(values) + 1.96 * stats.sem(values))
                    }

                ratio_stats[classifier_name] = classifier_stats
            return ratio_stats

        if has_prompt_level:
            # 数据结构: {prompt: {ratio: {classifier: {metric: [values]}}}}
            for prompt, prompt_data in organized_results.items():
                descriptive_stats[prompt] = {}
                for ratio, ratio_data in prompt_data.items():
                    descriptive_stats[prompt][ratio] = compute_stats_for_ratio_data(ratio_data)
        else:
            # 数据结构: {ratio: {classifier: {metric: [values]}}}
            for ratio, ratio_data in organized_results.items():
                descriptive_stats[ratio] = compute_stats_for_ratio_data(ratio_data)

        logger.info("描述性统计计算完成")
        return descriptive_stats

    def perform_hypothesis_tests(self, organized_results: Dict) -> Dict[str, Any]:
        """
        执行假设检验

        Args:
            organized_results: 按ratio组织的结果

        Returns:
            Dict[str, Any]: 假设检验结果
        """
        logger.info("执行假设检验")

        hypothesis_tests = {}
        alpha = self.eval_config.get('statistical_tests', {}).get('alpha', 0.05)

        # 获取baseline (ratio=0) 的结果
        baseline_ratio = 0
        if baseline_ratio not in organized_results:
            logger.warning("未找到baseline (ratio=0) 结果，跳过假设检验")
            return {}

        baseline_data = organized_results[baseline_ratio]

        for ratio, ratio_data in organized_results.items():
            if ratio == baseline_ratio:
                continue

            ratio_tests = {}

            for classifier_name in baseline_data.keys():
                if classifier_name not in ratio_data:
                    continue

                classifier_tests = {}

                for metric_name in baseline_data[classifier_name].keys():
                    if metric_name not in ratio_data[classifier_name]:
                        continue

                    baseline_values = np.array(baseline_data[classifier_name][metric_name])
                    ratio_values = np.array(ratio_data[classifier_name][metric_name])

                    if len(baseline_values) == 0 or len(ratio_values) == 0:
                        continue

                    # Paired t-test (假设配对样本)
                    min_length = min(len(baseline_values), len(ratio_values))
                    baseline_paired = baseline_values[:min_length]
                    ratio_paired = ratio_values[:min_length]

                    if min_length > 1:
                        # t-test
                        try:
                            t_stat, t_p_value = ttest_rel(baseline_paired, ratio_paired)
                        except:
                            t_stat, t_p_value = np.nan, np.nan

                        # Wilcoxon signed-rank test (非参数)
                        try:
                            w_stat, w_p_value = wilcoxon(baseline_paired, ratio_paired)
                        except:
                            w_stat, w_p_value = np.nan, np.nan

                        # Effect size (Cohen's d)
                        pooled_std = np.sqrt(
                            ((len(baseline_paired) - 1) * np.var(baseline_paired, ddof=1) +
                             (len(ratio_paired) - 1) * np.var(ratio_paired, ddof=1)) /
                            (len(baseline_paired) + len(ratio_paired) - 2)
                        )

                        if pooled_std > 0:
                            cohens_d = (np.mean(baseline_paired) - np.mean(ratio_paired)) / pooled_std
                        else:
                            cohens_d = 0

                        classifier_tests[metric_name] = {
                            'baseline_mean': float(np.mean(baseline_paired)),
                            'ratio_mean': float(np.mean(ratio_paired)),
                            'mean_difference': float(np.mean(baseline_paired) - np.mean(ratio_paired)),
                            't_statistic': float(t_stat),
                            't_p_value': float(t_p_value),
                            'wilcoxon_statistic': float(w_stat),
                            'wilcoxon_p_value': float(w_p_value),
                            'cohens_d': float(cohens_d),
                            'effect_size_interpretation': self._interpret_effect_size(cohens_d),
                            'significant_t': bool(t_p_value < alpha) if not np.isnan(t_p_value) else False,
                            'significant_wilcoxon': bool(w_p_value < alpha) if not np.isnan(w_p_value) else False
                        }

                ratio_tests[classifier_name] = classifier_tests
            hypothesis_tests[ratio] = ratio_tests

        # 多重比较校正
        corrected_tests = self._correct_multiple_comparisons(hypothesis_tests)

        logger.info("假设检验完成")
        return corrected_tests

    def _interpret_effect_size(self, cohens_d: float) -> str:
        """
        解释效应量大小

        Args:
            cohens_d: Cohen's d值

        Returns:
            str: 效应量解释
        """
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def _correct_multiple_comparisons(self, hypothesis_tests: Dict) -> Dict:
        """
        多重比较校正

        Args:
            hypothesis_tests: 原始假设检验结果

        Returns:
            Dict: 校正后的结果
        """
        logger.info("执行多重比较校正")

        method = self.eval_config.get('statistical_tests', {}).get('multiple_comparison_method', 'fdr_bh')

        # 收集所有p值
        all_p_values = []
        p_value_map = []  # 记录p值对应的位置

        for ratio, ratio_tests in hypothesis_tests.items():
            for classifier_name, classifier_tests in ratio_tests.items():
                for metric_name, test_result in classifier_tests.items():
                    # t-test p值
                    if 't_p_value' in test_result and not np.isnan(test_result['t_p_value']):
                        all_p_values.append(test_result['t_p_value'])
                        p_value_map.append((ratio, classifier_name, metric_name, 't_p_value'))

                    # Wilcoxon p值
                    if 'wilcoxon_p_value' in test_result and not np.isnan(test_result['wilcoxon_p_value']):
                        all_p_values.append(test_result['wilcoxon_p_value'])
                        p_value_map.append((ratio, classifier_name, metric_name, 'wilcoxon_p_value'))

        if len(all_p_values) == 0:
            return hypothesis_tests

        # 执行校正
        rejected, p_adjusted, alpha_sidak, alpha_bonf = multipletests(
            all_p_values, alpha=0.05, method=method
        )

        # 更新结果
        corrected_tests = hypothesis_tests.copy()

        for i, (ratio, classifier_name, metric_name, p_type) in enumerate(p_value_map):
            test_result = corrected_tests[ratio][classifier_name][metric_name]

            if p_type == 't_p_value':
                test_result['t_p_value_corrected'] = float(p_adjusted[i])
                test_result['significant_t_corrected'] = bool(rejected[i])
            else:
                test_result['wilcoxon_p_value_corrected'] = float(p_adjusted[i])
                test_result['significant_wilcoxon_corrected'] = bool(rejected[i])

        # 添加校正信息
        for ratio in corrected_tests:
            corrected_tests[ratio]['_correction_info'] = {
                'method': method,
                'total_tests': len(all_p_values),
                'alpha_bonferroni': float(alpha_bonf),
                'alpha_sidak': float(alpha_sidak)
            }

        return corrected_tests

    def analyze_performance_degradation(self, organized_results: Dict) -> Dict[str, Any]:
        """
        分析性能衰减

        Args:
            organized_results: 按ratio组织的结果

        Returns:
            Dict[str, Any]: 性能衰减分析
        """
        logger.info("分析性能衰减")

        degradation_analysis = {}

        # 获取baseline性能
        baseline_ratio = 0
        if baseline_ratio not in organized_results:
            logger.warning("未找到baseline结果")
            return {}

        baseline_data = organized_results[baseline_ratio]

        for classifier_name, classifier_data in baseline_data.items():
            classifier_analysis = {}

            for metric_name, baseline_values in classifier_data.items():
                baseline_mean = np.mean(baseline_values)

                metric_analysis = {
                    'baseline_performance': baseline_mean,
                    'ratios': {},
                    'degradation_curve': [],
                    'critical_ratio': None,
                    'max_acceptable_degradation': 0.05  # 5%衰减阈值
                }

                # 分析每个ratio的性能
                for ratio in sorted(organized_results.keys()):
                    if ratio == baseline_ratio:
                        continue

                    if (classifier_name in organized_results[ratio] and
                            metric_name in organized_results[ratio][classifier_name]):

                        ratio_values = organized_results[ratio][classifier_name][metric_name]
                        ratio_mean = np.mean(ratio_values)

                        # 计算性能衰减
                        if baseline_mean > 0:
                            degradation = (baseline_mean - ratio_mean) / baseline_mean
                        else:
                            degradation = 0

                        ratio_analysis = {
                            'performance': ratio_mean,
                            'absolute_degradation': baseline_mean - ratio_mean,
                            'relative_degradation': degradation,
                            'degradation_percentage': degradation * 100
                        }

                        metric_analysis['ratios'][ratio] = ratio_analysis
                        metric_analysis['degradation_curve'].append({
                            'ratio': ratio,
                            'performance': ratio_mean,
                            'degradation': degradation
                        })

                        # 检查是否达到临界衰减
                        if (metric_analysis['critical_ratio'] is None and
                                degradation > metric_analysis['max_acceptable_degradation']):
                            metric_analysis['critical_ratio'] = ratio

                # 计算sensitivity指标（性能对synthetic ratio的敏感度）
                metric_analysis['sensitivity'] = self._calculate_sensitivity(
                    baseline_mean,
                    metric_analysis['degradation_curve']
                )

                classifier_analysis[metric_name] = metric_analysis
            degradation_analysis[classifier_name] = classifier_analysis

        return degradation_analysis

    def _calculate_sensitivity(self, baseline: float, degradation_curve: List[Dict]) -> Dict[str, float]:
        """
        计算模型对synthetic data比例的敏感度

        Args:
            baseline: 基线性能（0% synthetic）
            degradation_curve: 性能衰减曲线

        Returns:
            Dict: 包含各种sensitivity指标
        """
        if not degradation_curve:
            return {}

        # 准备数据：ratios和performances
        ratios = [0] + [point['ratio'] for point in degradation_curve]
        performances = [baseline] + [point['performance'] for point in degradation_curve]

        # 线性回归计算斜率
        slope, intercept, r_value, p_value, std_err = stats.linregress(ratios, performances)

        # 计算0%到100%的总变化（如果有100%的数据）
        max_ratio = max(ratios)
        if max_ratio == 100:
            perf_at_max = performances[-1]
            total_change = perf_at_max - baseline
            linear_rate = total_change / 100
            relative_rate = (total_change / baseline) * 100 / 100 if baseline != 0 else 0
        else:
            # 如果没有100%，使用最大ratio估算
            perf_at_max = performances[-1]
            total_change = perf_at_max - baseline
            linear_rate = total_change / max_ratio
            relative_rate = (total_change / baseline) * 100 / max_ratio if baseline != 0 else 0

        return {
            'regression_slope': float(slope),
            'regression_intercept': float(intercept),
            'regression_r_squared': float(r_value ** 2),
            'regression_p_value': float(p_value),
            'regression_stderr': float(std_err),
            'linear_rate_per_percent': float(linear_rate),
            'relative_rate_per_percent': float(relative_rate * 100),
            'max_ratio_tested': int(max_ratio),
            'total_change_0_to_max': float(total_change)
        }

    def compare_prompt_strategies(self, results_by_prompt: Dict[str, Dict]) -> Dict[str, Any]:
        """
        比较不同prompt策略

        Args:
            results_by_prompt: 按prompt组织的结果

        Returns:
            Dict[str, Any]: prompt比较结果
        """
        logger.info("比较prompt策略")

        if len(results_by_prompt) < 2:
            logger.warning("至少需要2个prompt进行比较")
            return {}

        prompt_comparison = {}

        # 获取所有prompt名称
        prompt_names = list(results_by_prompt.keys())

        # 对每个ratio进行比较
        for ratio in results_by_prompt[prompt_names[0]].keys():
            ratio_comparison = {}

            # 对每个分类器进行比较
            for classifier_name in results_by_prompt[prompt_names[0]][ratio].keys():
                classifier_comparison = {}

                # 对每个指标进行比较
                for metric_name in results_by_prompt[prompt_names[0]][ratio][classifier_name].keys():
                    # 收集所有prompt的值
                    prompt_values = {}
                    for prompt_name in prompt_names:
                        if (ratio in results_by_prompt[prompt_name] and
                                classifier_name in results_by_prompt[prompt_name][ratio] and
                                metric_name in results_by_prompt[prompt_name][ratio][classifier_name]):
                            prompt_values[prompt_name] = results_by_prompt[prompt_name][ratio][classifier_name][metric_name]

                    if len(prompt_values) < 2:
                        continue

                    # 执行ANOVA
                    values_list = list(prompt_values.values())
                    try:
                        f_stat, p_value = f_oneway(*values_list)
                    except:
                        f_stat, p_value = np.nan, np.nan

                    # 计算每个prompt的统计信息
                    prompt_stats = {}
                    for prompt_name, values in prompt_values.items():
                        prompt_stats[prompt_name] = {
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values, ddof=1)),
                            'count': len(values)
                        }

                    # 找出最好的prompt
                    best_prompt = max(prompt_stats.keys(), key=lambda x: prompt_stats[x]['mean'])

                    classifier_comparison[metric_name] = {
                        'anova_f_statistic': float(f_stat),
                        'anova_p_value': float(p_value),
                        'significant': bool(p_value < 0.05) if not np.isnan(p_value) else False,
                        'best_prompt': best_prompt,
                        'prompt_stats': prompt_stats
                    }

                ratio_comparison[classifier_name] = classifier_comparison
            prompt_comparison[ratio] = ratio_comparison

        return prompt_comparison

    def generate_comprehensive_report(
        self,
        results_file: str,
        output_dir: str
    ) -> Dict[str, Any]:
        """
        生成综合统计分析报告

        Args:
            results_file: 结果文件路径
            output_dir: 输出目录

        Returns:
            Dict[str, Any]: 综合分析报告
        """
        logger.info("生成综合统计分析报告")

        # 加载结果
        results = self.load_classification_results(results_file)

        # 组织结果
        organized_results = self.organize_results_by_ratio(results)

        # 计算描述性统计
        descriptive_stats = self.compute_descriptive_statistics(organized_results)

        # 执行假设检验
        hypothesis_tests = self.perform_hypothesis_tests(organized_results)

        # 分析性能衰减
        degradation_analysis = self.analyze_performance_degradation(organized_results)

        # 生成综合报告
        comprehensive_report = {
            'experiment_info': {
                'experiment_name': results['experiment_name'],
                'total_experiments': results['summary']['total_experiments'],
                'successful_experiments': results['summary']['successful_experiments'],
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            },
            'descriptive_statistics': descriptive_stats,
            'hypothesis_tests': hypothesis_tests,
            'performance_degradation': degradation_analysis,
            'summary_findings': self._generate_summary_findings(
                descriptive_stats, hypothesis_tests, degradation_analysis
            )
        }

        # 保存报告
        os.makedirs(output_dir, exist_ok=True)
        report_file = os.path.join(output_dir, f"{self.config.name}_statistical_analysis.json")

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_report, f, indent=2, ensure_ascii=False, default=str)

        comprehensive_report['report_file'] = report_file

        logger.info(f"统计分析报告已保存: {report_file}")
        return comprehensive_report

    def _generate_summary_findings(
        self,
        descriptive_stats: Dict,
        hypothesis_tests: Dict,
        degradation_analysis: Dict
    ) -> Dict[str, Any]:
        """
        生成总结性发现

        Args:
            descriptive_stats: 描述性统计
            hypothesis_tests: 假设检验结果
            degradation_analysis: 性能衰减分析

        Returns:
            Dict[str, Any]: 总结性发现
        """
        findings = {
            'baseline_performance': {},
            'significant_degradations': [],
            'critical_ratios': {},
            'best_synthetic_ratios': {},
            'overall_conclusions': []
        }

        # 提取baseline性能
        if 0 in descriptive_stats:
            findings['baseline_performance'] = {
                classifier: {
                    metric: stats['mean']
                    for metric, stats in metrics.items()
                }
                for classifier, metrics in descriptive_stats[0].items()
            }

        # 识别显著的性能衰减
        for ratio, ratio_tests in hypothesis_tests.items():
            for classifier, classifier_tests in ratio_tests.items():
                for metric, test_result in classifier_tests.items():
                    if isinstance(test_result, dict) and test_result.get('significant_t_corrected', False):
                        findings['significant_degradations'].append({
                            'ratio': ratio,
                            'classifier': classifier,
                            'metric': metric,
                            'degradation': test_result['mean_difference'],
                            'p_value': test_result['t_p_value_corrected'],
                            'effect_size': test_result['cohens_d']
                        })

        # 提取临界比例
        for classifier, classifier_analysis in degradation_analysis.items():
            findings['critical_ratios'][classifier] = {}
            for metric, metric_analysis in classifier_analysis.items():
                if metric_analysis['critical_ratio'] is not None:
                    findings['critical_ratios'][classifier][metric] = metric_analysis['critical_ratio']

        # 找出最佳synthetic ratio（性能衰减最小）
        for classifier, classifier_analysis in degradation_analysis.items():
            findings['best_synthetic_ratios'][classifier] = {}
            for metric, metric_analysis in classifier_analysis.items():
                min_degradation_ratio = min(
                    metric_analysis['ratios'].keys(),
                    key=lambda r: abs(metric_analysis['ratios'][r]['relative_degradation'])
                )
                findings['best_synthetic_ratios'][classifier][metric] = {
                    'ratio': min_degradation_ratio,
                    'degradation': metric_analysis['ratios'][min_degradation_ratio]['relative_degradation']
                }

        # 生成总体结论
        findings['overall_conclusions'] = [
            f"Baseline performance established across {len(findings['baseline_performance'])} classifiers",
            f"Found {len(findings['significant_degradations'])} significant performance degradations",
            f"Critical degradation thresholds identified for multiple classifier-metric combinations"
        ]

        return findings


# 便捷函数
def analyze_experiment_results(
    config: ExperimentConfig,
    results_file: str,
    output_dir: str
) -> Dict[str, Any]:
    """
    便捷函数：分析实验结果

    Args:
        config: 实验配置
        results_file: 结果文件路径
        output_dir: 输出目录

    Returns:
        Dict[str, Any]: 分析报告
    """
    analyzer = StatisticalAnalyzer(config)
    return analyzer.generate_comprehensive_report(results_file, output_dir)