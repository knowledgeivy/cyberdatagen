#!/usr/bin/env python3
"""
Sequential execution of all 4 strategies to avoid file conflicts
"""

import argparse
import subprocess
import sys
import os
import json
from pathlib import Path
from loguru import logger

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from src.config.config_manager import load_config


def run_strategy(config_path: str, strategy: str) -> dict:
    """运行单个策略的分类实验"""
    logger.info(f"开始运行策略: {strategy}")

    # 构建输出文件名
    config = load_config(config_path)
    output_dir = config.output.get('results_path', './output/results/')
    strategy_results_file = os.path.join(output_dir, f"{config.name}_{strategy}_results.json")

    # 运行分类实验
    cmd = [
        "poetry", "run", "python", "scripts/step4_classification.py",
        "--config", config_path,
        "--strategy", strategy,
        "--output_dir", output_dir
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"策略 {strategy} 执行失败: {result.stderr}")
            return None

        logger.info(f"策略 {strategy} 执行成功")

        # 读取结果文件
        main_results_file = os.path.join(output_dir, f"{config.name}_classification_results.json")
        if os.path.exists(main_results_file):
            with open(main_results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)

            # 重命名为策略特定文件
            with open(strategy_results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"策略 {strategy} 结果已保存: {strategy_results_file}")
            return results
        else:
            logger.error(f"未找到策略 {strategy} 的结果文件")
            return None

    except Exception as e:
        logger.error(f"策略 {strategy} 执行异常: {e}")
        return None


def merge_results(config_path: str, strategy_results: dict) -> None:
    """合并所有策略的结果"""
    logger.info("开始合并所有策略结果")

    config = load_config(config_path)
    output_dir = config.output.get('results_path', './output/results/')
    merged_file = os.path.join(output_dir, f"{config.name}_all_strategies_results.json")

    # 合并结果
    merged_results = {
        "experiment_name": config.name,
        "strategies": list(strategy_results.keys()),
        "summary": {
            "total_strategies": len(strategy_results),
            "total_experiments": sum(len(results['results']) for results in strategy_results.values()),
            "merged_timestamp": "2025-10-02T20:10:00"
        },
        "results": []
    }

    # 添加所有策略的结果
    for strategy, results in strategy_results.items():
        if results and 'results' in results:
            for result in results['results']:
                # 确保strategy字段正确设置
                result['strategy'] = strategy
                merged_results['results'].append(result)

    # 保存合并结果
    with open(merged_file, 'w', encoding='utf-8') as f:
        json.dump(merged_results, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"合并结果已保存: {merged_file}")
    logger.info(f"总实验数: {len(merged_results['results'])}")


def main():
    parser = argparse.ArgumentParser(description='Run all strategies sequentially')
    parser.add_argument('--config', type=str, required=True, help='Configuration file path')

    args = parser.parse_args()

    # 四种策略
    strategies = ['within_group', 'cross_group', 'real_fixed_random_synthetic', 'full_random']

    logger.info("开始顺序执行所有策略")
    logger.info(f"策略列表: {strategies}")

    strategy_results = {}

    # 顺序执行每个策略
    for strategy in strategies:
        results = run_strategy(args.config, strategy)
        if results:
            strategy_results[strategy] = results
        else:
            logger.warning(f"策略 {strategy} 未返回有效结果")

    # 合并所有结果
    if strategy_results:
        merge_results(args.config, strategy_results)

        # 输出摘要
        logger.info("=" * 50)
        logger.info("所有策略执行完成")
        logger.info("=" * 50)

        for strategy, results in strategy_results.items():
            if results and 'results' in results:
                logger.info(f"{strategy}: {len(results['results'])} 个实验")

        total_experiments = sum(len(results['results']) for results in strategy_results.values())
        logger.info(f"总实验数: {total_experiments}")

        logger.success("所有策略执行和合并完成")
    else:
        logger.error("没有成功的策略结果")


if __name__ == "__main__":
    main()