#!/usr/bin/env python3
"""
合并 SVM 和 Random Forest 统计分析数据
将新的 SVM 数据和旧的 Random Forest 数据合并到一个文件中
"""

import json
import sys
from pathlib import Path

def merge_statistical_analysis(svm_file: str, rf_file: str, output_file: str):
    """
    合并两个统计分析文件

    Args:
        svm_file: 包含新 SVM 数据的文件
        rf_file: 包含旧 Random Forest 数据的文件
        output_file: 输出合并后的文件
    """
    # 读取 SVM 数据
    with open(svm_file, 'r', encoding='utf-8') as f:
        svm_data = json.load(f)

    # 读取 Random Forest 数据
    with open(rf_file, 'r', encoding='utf-8') as f:
        rf_data = json.load(f)

    # 创建合并后的数据（基于 SVM 数据）
    merged_data = svm_data.copy()

    # 合并 descriptive_statistics
    if 'descriptive_statistics' in rf_data:
        for ratio, ratio_data in rf_data['descriptive_statistics'].items():
            if 'random_forest' in ratio_data:
                if ratio not in merged_data['descriptive_statistics']:
                    merged_data['descriptive_statistics'][ratio] = {}
                merged_data['descriptive_statistics'][ratio]['random_forest'] = ratio_data['random_forest']

    # 合并 performance_degradation
    if 'performance_degradation' in rf_data:
        if 'random_forest' in rf_data['performance_degradation']:
            if 'performance_degradation' not in merged_data:
                merged_data['performance_degradation'] = {}
            merged_data['performance_degradation']['random_forest'] = rf_data['performance_degradation']['random_forest']

    # 合并 hypothesis_tests
    if 'hypothesis_tests' in rf_data:
        for ratio, ratio_tests in rf_data['hypothesis_tests'].items():
            if ratio == '_correction_info':
                continue
            if 'random_forest' in ratio_tests:
                if ratio not in merged_data['hypothesis_tests']:
                    merged_data['hypothesis_tests'][ratio] = {}
                merged_data['hypothesis_tests'][ratio]['random_forest'] = ratio_tests['random_forest']

    # 保存合并后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)

    print(f"✓ 合并完成: {output_file}")
    print(f"  - SVM 数据来源: {svm_file}")
    print(f"  - Random Forest 数据来源: {rf_file}")


def main():
    # Claude 模型
    claude_base = "output/full_experiments/ceas08_claude35haiku/reports"

    for prompt in ['original', 'strong', 'weak']:
        svm_file = f"{claude_base}/full_ceas08_claude35haiku_v1_{prompt}_cross_group_statistical_analysis.json"
        rf_file = f"{claude_base}/full_ceas08_claude35haiku_v1_{prompt}_cross_group_statistical_analysis_OLD.json"
        output_file = svm_file  # 覆盖原文件

        print(f"\n处理 Claude - {prompt} prompt:")
        merge_statistical_analysis(svm_file, rf_file, output_file)

    # GPT 模型
    gpt_base = "output/full_experiments/ceas08_gpt41mini/reports"

    for prompt in ['original', 'strong', 'weak']:
        svm_file = f"{gpt_base}/full_ceas08_gpt41mini_v1_{prompt}_cross_group_statistical_analysis.json"
        rf_file = f"{gpt_base}/full_ceas08_gpt41mini_v1_{prompt}_cross_group_statistical_analysis_OLD.json"
        output_file = svm_file  # 覆盖原文件

        if Path(svm_file).exists() and Path(rf_file).exists():
            print(f"\n处理 GPT - {prompt} prompt:")
            merge_statistical_analysis(svm_file, rf_file, output_file)
        else:
            print(f"\n跳过 GPT - {prompt} prompt (文件不存在)")

    print("\n" + "="*50)
    print("所有文件合并完成！")


if __name__ == "__main__":
    main()
