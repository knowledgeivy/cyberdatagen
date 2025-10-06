#!/usr/bin/env python3
"""
合并所有group的独立结果文件到一个总文件
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
import sys

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger


def merge_group_results(output_dir: str, experiment_name: str, strategies: list):
    """
    合并所有group的结果文件

    Args:
        output_dir: 输出目录
        experiment_name: 实验名称
        strategies: 策略列表
    """
    all_results = []
    total_groups = 20

    # 检查是否存在旧的合并文件（兼容模式）
    old_merged_file = os.path.join(output_dir, f"{experiment_name}_classification_results.json")
    old_results_by_strategy = {}

    if os.path.exists(old_merged_file):
        logger.info(f"发现旧的合并文件，读取已有数据: {old_merged_file}")
        try:
            with open(old_merged_file, 'r', encoding='utf-8') as f:
                old_data = json.load(f)
                old_results = old_data.get('results', [])
                for result in old_results:
                    strategy = result.get('strategy')
                    if strategy not in old_results_by_strategy:
                        old_results_by_strategy[strategy] = []
                    old_results_by_strategy[strategy].append(result)
        except Exception as e:
            logger.error(f"读取旧文件失败: {e}")

    # 遍历所有策略和group
    for strategy in strategies:
        logger.info(f"处理策略: {strategy}")
        strategy_results = []
        found_group_files = False

        for group_id in range(total_groups):
            group_file = os.path.join(
                output_dir,
                f"{experiment_name}_{strategy}_group{group_id}_results.json"
            )

            if os.path.exists(group_file):
                found_group_files = True
                try:
                    with open(group_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        group_results = data.get('results', [])
                        strategy_results.extend(group_results)
                        logger.info(f"  Group {group_id}: {len(group_results)} 个实验")
                except Exception as e:
                    logger.error(f"  Group {group_id}: 读取失败 - {e}")

        # 如果没有找到group文件，尝试从旧文件中获取
        if not found_group_files and strategy in old_results_by_strategy:
            strategy_results = old_results_by_strategy[strategy]
            logger.info(f"  从旧文件读取: {len(strategy_results)} 个实验")
        elif not found_group_files:
            logger.warning(f"  策略 {strategy}: 没有找到任何数据")

        logger.info(f"策略 {strategy}: 总计 {len(strategy_results)} 个实验")
        all_results.extend(strategy_results)

    # 保存合并后的结果
    merged_file = os.path.join(output_dir, f"{experiment_name}_classification_results.json")

    # 统计分类器性能
    classifier_stats = defaultdict(list)
    for result in all_results:
        for classifier_name, classifier_result in result.get('classifiers', {}).items():
            if classifier_result.get('success'):
                f1_score = classifier_result.get('metrics', {}).get('f1_score')
                if f1_score is not None:
                    classifier_stats[classifier_name].append(f1_score)

    merged_data = {
        'experiment_name': experiment_name,
        'summary': {
            'total_experiments': len(all_results),
            'successful_experiments': len(all_results),
            'failed_experiments': 0,
            'results_file': merged_file,
            'average_f1_scores': {
                name: sum(scores) / len(scores)
                for name, scores in classifier_stats.items()
            }
        },
        'results': all_results,
        'config': {
            'classifiers': list(classifier_stats.keys()),
            'evaluation_metrics': ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        }
    }

    with open(merged_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False, default=str)

    logger.success(f"合并完成: {len(all_results)} 个实验")
    logger.success(f"结果已保存到: {merged_file}")

    # 打印统计信息
    logger.info("\n分类器平均F1分数:")
    for name, avg_f1 in merged_data['summary']['average_f1_scores'].items():
        logger.info(f"  {name}: {avg_f1:.4f}")

    # 按策略统计
    from collections import Counter
    strategy_counts = Counter([r['strategy'] for r in all_results])
    logger.info("\n各策略实验数:")
    for strategy, count in sorted(strategy_counts.items()):
        logger.info(f"  {strategy}: {count}/1100 ({count/1100*100:.1f}%)")

    return merged_file


def main():
    parser = argparse.ArgumentParser(description='合并group结果文件')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output/full_experiments/ceas08_gpt41mini/results/',
        help='结果目录'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='full_ceas08_gpt41mini_v1',
        help='实验名称'
    )
    parser.add_argument(
        '--strategies',
        nargs='+',
        default=['within_group', 'cross_group', 'real_fixed_random_synthetic', 'full_random'],
        help='策略列表'
    )

    args = parser.parse_args()

    merge_group_results(args.output_dir, args.experiment_name, args.strategies)


if __name__ == "__main__":
    main()
