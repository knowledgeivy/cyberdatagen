"""
批处理工具
提供批量运行分析和可视化的功能
"""

import subprocess
from pathlib import Path
from typing import List, Dict
from loguru import logger


def run_batch_analysis(
    config_file: str,
    results_file: str,
    reports_dir: str,
    experiment_name: str,
    prompts: List[str] = None,
    strategies: List[str] = None
) -> Dict[str, int]:
    """
    批量生成统计分析报告

    Args:
        config_file: 配置文件路径
        results_file: 合并后的结果文件
        reports_dir: 报告输出目录
        experiment_name: 实验名称
        prompts: Prompt列表
        strategies: 策略列表

    Returns:
        Dict: {'success': n, 'failed': n}
    """
    if prompts is None:
        prompts = ['original', 'strong', 'weak']
    if strategies is None:
        strategies = ['within_group', 'cross_group']

    # 检查结果文件
    if not Path(results_file).exists():
        logger.error(f"结果文件不存在: {results_file}")
        raise FileNotFoundError(f"请先合并结果文件: {results_file}")

    logger.info("开始批量生成统计分析报告")
    logger.info(f"  Prompts: {', '.join(prompts)}")
    logger.info(f"  Strategies: {', '.join(strategies)}")

    stats = {'success': 0, 'failed': 0}

    for prompt in prompts:
        for strategy in strategies:
            logger.info(f"分析 {prompt} - {strategy}...")

            output_file = Path(reports_dir) / f"{experiment_name}_{prompt}_{strategy}_statistical_analysis.json"

            cmd = [
                'poetry', 'run', 'python', 'scripts/step5_statistical_analysis.py',
                '--config', config_file,
                '--results_file', results_file,
                '--output_file', str(output_file),
                '--filter_prompt', prompt,
                '--filter_strategy', strategy
            ]

            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                logger.info(f"  ✓ 完成: {output_file}")
                stats['success'] += 1
            except subprocess.CalledProcessError as e:
                logger.error(f"  ✗ 失败: {e.stderr[:200]}")
                stats['failed'] += 1

    total = stats['success'] + stats['failed']
    logger.success(f"批量分析完成: {stats['success']}/{total} 成功")

    return stats


def run_batch_visualization(
    config_file: str,
    base_dir: str,
    experiment_name: str,
    prompts: List[str] = None,
    strategies: List[str] = None
) -> Dict[str, int]:
    """
    批量生成可视化图表

    Args:
        config_file: 配置文件路径
        base_dir: 实验基础目录
        experiment_name: 实验名称
        prompts: Prompt列表
        strategies: 策略列表

    Returns:
        Dict: {'success': n, 'skipped': n, 'failed': n}
    """
    if prompts is None:
        prompts = ['original', 'strong', 'weak']
    if strategies is None:
        strategies = ['within_group', 'cross_group']

    base_path = Path(base_dir)
    reports_dir = base_path / 'reports'
    plots_dir = base_path / 'plots'

    logger.info("开始批量生成可视化图表")
    logger.info(f"  Prompts: {', '.join(prompts)}")
    logger.info(f"  Strategies: {', '.join(strategies)}")

    stats = {'success': 0, 'skipped': 0, 'failed': 0}

    # 生成每个 prompt × strategy 的小图
    for prompt in prompts:
        for strategy in strategies:
            logger.info(f"生成 {prompt} - {strategy}...")

            analysis_file = reports_dir / f"{experiment_name}_{prompt}_{strategy}_statistical_analysis.json"
            output_dir = plots_dir / prompt / strategy

            if not analysis_file.exists():
                logger.warning(f"  ⚠ 跳过 - 分析文件不存在: {analysis_file}")
                stats['skipped'] += 1
                continue

            cmd = [
                'poetry', 'run', 'python', 'scripts/step6_visualization.py',
                '--config', config_file,
                '--experiment_name', f"{experiment_name}_{prompt}_{strategy}",
                '--analysis_file', str(analysis_file),
                '--output_dir', str(output_dir),
                '--plot_types', 'curves'
            ]

            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                logger.info(f"  ✓ 完成")
                stats['success'] += 1
            except subprocess.CalledProcessError as e:
                logger.error(f"  ✗ 失败: {e.stderr[:200]}")
                stats['failed'] += 1

    # 生成combined对比图
    logger.info("生成combined对比图...")

    cmd = [
        'poetry', 'run', 'python', 'scripts/step6_visualization.py',
        '--config', config_file,
        '--mode', 'combined',
        '--experiment_name', experiment_name,
        '--base_dir', base_dir,
        '--strategies'
    ] + strategies

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("  ✓ 完成")
    except subprocess.CalledProcessError as e:
        logger.error(f"  ✗ 失败: {e.stderr[:200]}")

    total = stats['success'] + stats['skipped'] + stats['failed']
    logger.success(f"批量可视化完成: {stats['success']}/{total} 成功, {stats['skipped']}/{total} 跳过")
    logger.info(f"图表位置:")
    logger.info(f"  - 小图: {plots_dir}/{{prompt}}/{{strategy}}/")
    logger.info(f"  - 大图: {plots_dir}/combined/")

    return stats
