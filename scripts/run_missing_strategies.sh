#!/bin/bash
# 重新运行缺失的3个策略（修复并发保存bug后）
# cross_group, real_fixed_random_synthetic, full_random

CONFIG="config/full_ceas08_gpt41mini_v1.yaml"
STRATEGIES=("cross_group" "real_fixed_random_synthetic" "full_random")

echo "=========================================="
echo "重新运行缺失的策略（并行模式）"
echo "日期: $(date)"
echo "配置: $CONFIG"
echo "策略: ${STRATEGIES[@]}"
echo "=========================================="

for strategy in "${STRATEGIES[@]}"; do
    echo ""
    echo "=========================================="
    echo "开始策略: $strategy"
    echo "开始时间: $(date)"
    echo "=========================================="

    bash scripts/run_parallel_strategy.sh "$strategy"

    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "✓ 策略 $strategy 完成"
        echo "完成时间: $(date)"
    else
        echo "✗ 策略 $strategy 有 $exit_code 个任务失败"
        echo "失败时间: $(date)"
        echo "继续下一个策略..."
    fi

    echo "=========================================="
done

echo ""
echo "=========================================="
echo "所有缺失策略完成！正在合并所有结果..."
echo "=========================================="

# 合并所有group的结果文件（包括within_group）
poetry run python scripts/merge_results.py \
    --output_dir "./output/full_experiments/ceas08_gpt41mini/results/" \
    --experiment_name "full_ceas08_gpt41mini_v1" \
    --strategies within_group cross_group real_fixed_random_synthetic full_random

echo ""
echo "=========================================="
echo "实验和合并全部完成！"
echo "结束时间: $(date)"
echo "=========================================="
