#!/usr/bin/env python3
"""
合并SMOTE group结果文件
"""
import json
import glob
import os
import argparse
from pathlib import Path


def merge_smote_group_results(results_dir, experiment_name, variant, strategy, output_file=None):
    """合并指定variant和strategy的所有SMOTE group结果文件"""

    # 查找所有group文件
    pattern = f"{experiment_name}_{variant}_{strategy}_group*.json"
    group_files = sorted(glob.glob(os.path.join(results_dir, pattern)))

    if not group_files:
        print(f"错误: 未找到匹配的group文件: {pattern}")
        return None

    print(f"找到 {len(group_files)} 个group文件")

    # 加载所有group结果
    all_results = []
    experiment_name_str = None
    smote_variant = None

    for group_file in group_files:
        print(f"  加载: {os.path.basename(group_file)}")
        with open(group_file, 'r') as f:
            data = json.load(f)
            all_results.extend(data['results'])
            if experiment_name_str is None:
                experiment_name_str = data['experiment_name']
                smote_variant = data['smote_variant']

    # 重新计算summary
    # Group by (group_id, synthetic_ratio) to count experiments
    experiment_groups = {}
    for r in all_results:
        key = (r['group_id'], r['synthetic_ratio'])
        if key not in experiment_groups:
            experiment_groups[key] = []
        experiment_groups[key].append(r)

    total_experiments = len(experiment_groups)
    successful_experiments = sum(
        1 for group in experiment_groups.values()
        if all(r['success'] for r in group)
    )
    failed_experiments = total_experiments - successful_experiments

    summary = {
        'total_experiments': total_experiments,
        'successful_experiments': successful_experiments,
        'failed_experiments': failed_experiments
    }

    # 构建合并后的结果 - 转换为GPT/Claude兼容格式
    merged_results = []
    for (group_id, synthetic_ratio), group_results in sorted(experiment_groups.items()):
        # 将同一实验的不同分类器结果合并
        classifiers_dict = {}
        for r in group_results:
            clf_name = r['classifier']
            classifiers_dict[clf_name] = {
                'metrics': r['metrics'],
                'success': r['success']
            }

        # 构建实验记录
        experiment = {
            'experiment_id': f's{strategy}_r{synthetic_ratio}_g{group_id}',
            'group_id': group_id,
            'synthetic_ratio': synthetic_ratio,
            'strategy': strategy,
            'method': group_results[0]['method'],
            'classifiers': classifiers_dict,
            'train_size': group_results[0]['metadata'].get('train_size', 1000),
            'test_size': group_results[0]['metadata'].get('test_size', 7430),
        }
        merged_results.append(experiment)

    merged_data = {
        'experiment_name': experiment_name_str,
        'summary': summary,
        'results': merged_results,
        'config': {
            'smote_variant': smote_variant,
            'strategy': strategy
        }
    }

    # 保存合并结果
    if output_file is None:
        output_file = os.path.join(results_dir, f"{experiment_name}_{variant}_{strategy}_classification_results.json")

    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)

    print(f"\n合并完成:")
    print(f"  总实验数: {total_experiments}")
    print(f"  成功: {successful_experiments}")
    print(f"  失败: {failed_experiments}")
    print(f"  输出文件: {output_file}")

    return output_file


def main():
    parser = argparse.ArgumentParser(description='合并SMOTE group分类结果文件')
    parser.add_argument('--results_dir', type=str, required=True, help='结果文件目录')
    parser.add_argument('--experiment_name', type=str, required=True, help='实验名称')
    parser.add_argument('--variant', type=str, required=True, help='SMOTE变体 (smote/adasyn)')
    parser.add_argument('--strategy', type=str, required=True, help='策略类型')
    parser.add_argument('--output_file', type=str, help='输出文件路径（可选）')

    args = parser.parse_args()

    merge_smote_group_results(
        args.results_dir,
        args.experiment_name,
        args.variant,
        args.strategy,
        args.output_file
    )


if __name__ == '__main__':
    main()
