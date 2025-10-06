#!/bin/bash
# 并行运行单个策略的分类实验（按group并行）
# 用法: ./run_parallel_strategy.sh <strategy_name>

STRATEGY=$1
CONFIG="config/full_ceas08_gpt41mini_v1.yaml"

if [ -z "$STRATEGY" ]; then
    echo "用法: $0 <strategy_name>"
    echo "示例: $0 within_group"
    exit 1
fi

echo "=========================================="
echo "并行运行策略: $STRATEGY"
echo "开始时间: $(date)"
echo "=========================================="

# 为每个group (0-19) 启动一个独立进程
# 每个group包含: 11 ratios × 5 trials = 55个实验
pids=()

for group in {0..19}; do
    echo "启动 group $group"

    poetry run python scripts/step4_classification.py \
        --config "$CONFIG" \
        --strategy "$STRATEGY" \
        --group_id $group \
        --resume \
        > "logs/${STRATEGY}_group${group}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

    pids+=($!)

    # 避免同时启动太多进程，每启动5个等待0.5秒
    if [ $((group % 5)) -eq 4 ]; then
        sleep 0.5
    fi
done

echo "已启动 ${#pids[@]} 个并行进程 (每个处理1个group)"
echo "进程IDs: ${pids[@]}"

# 等待所有进程完成
echo "等待所有进程完成..."
failed=0
for pid in "${pids[@]}"; do
    if wait $pid; then
        echo "✓ 进程 $pid 完成"
    else
        echo "✗ 进程 $pid 失败"
        ((failed++))
    fi
done

echo "=========================================="
echo "策略 $STRATEGY 完成"
echo "完成时间: $(date)"
echo "失败进程数: $failed"
echo "=========================================="

exit $failed
