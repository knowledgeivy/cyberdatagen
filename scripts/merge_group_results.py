#!/usr/bin/env python3
"""
合并group分类结果文件
"""
import json
import glob
import os
import argparse
from pathlib import Path


def merge_group_results(results_dir, experiment_name, prompt, strategy, output_file=None):
    """合并指定prompt和strategy的所有group结果文件"""

    # 查找所有group文件
    pattern = f"{experiment_name}_{prompt}_{strategy}_group*.json"
    group_files = sorted(glob.glob(os.path.join(results_dir, pattern)))

    if not group_files:
        print(f"错误: 未找到匹配的group文件: {pattern}")
        return None

    print(f"找到 {len(group_files)} 个group文件")

    # 加载所有group结果
    all_results = []
    config = None
    experiment_name_str = None

    for group_file in group_files:
        print(f"  加载: {os.path.basename(group_file)}")
        with open(group_file, 'r') as f:
            data = json.load(f)
            all_results.extend(data['results'])
            if config is None:
                config = data['config']
                experiment_name_str = data['experiment_name']

    # 重新计算summary
    total_experiments = len(all_results)
    successful_experiments = sum(1 for r in all_results if all(c['success'] for c in r['classifiers'].values()))
    failed_experiments = total_experiments - successful_experiments

    summary = {
        'total_experiments': total_experiments,
        'successful_experiments': successful_experiments,
        'failed_experiments': failed_experiments
    }

    # 构建合并后的结果
    merged_data = {
        'experiment_name': experiment_name_str,
        'summary': summary,
        'results': all_results,
        'config': config
    }

    # 保存合并结果
    if output_file is None:
        output_file = os.path.join(results_dir, f"{experiment_name}_classification_results.json")

    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)

    print(f"\n合并完成:")
    print(f"  总实验数: {total_experiments}")
    print(f"  成功: {successful_experiments}")
    print(f"  失败: {failed_experiments}")
    print(f"  输出文件: {output_file}")

    return output_file


def main():
    parser = argparse.ArgumentParser(description='合并group分类结果文件')
    parser.add_argument('--results_dir', type=str, required=True, help='结果文件目录')
    parser.add_argument('--experiment_name', type=str, required=True, help='实验名称')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt类型')
    parser.add_argument('--strategy', type=str, required=True, help='策略类型')
    parser.add_argument('--output_file', type=str, help='输出文件路径（可选）')

    args = parser.parse_args()

    merge_group_results(
        args.results_dir,
        args.experiment_name,
        args.prompt,
        args.strategy,
        args.output_file
    )


if __name__ == '__main__':
    main()
