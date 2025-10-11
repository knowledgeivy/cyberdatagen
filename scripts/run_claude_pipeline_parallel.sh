#!/bin/bash
# Claude 3.5 Haiku 完整实验 Pipeline (并行版本)
# Steps 3-6: Dataset Construction → Classification → Analysis → Visualization
# 策略：按group并行运行classification，最大化利用多核CPU

set -e  # 遇到错误立即退出

CONFIG="config/full_ceas08_claude35haiku_v1.yaml"
EXP_NAME="full_ceas08_claude35haiku_v1"
LLM_ENGINE="claude-3-5-haiku"

# 并行参数
N_GROUPS=20  # 总共20个groups (0-19)
MAX_PARALLEL=8  # 同时运行的最大group数（可根据CPU核心数调整）

# 定义prompts和strategies
PROMPTS=("original" "strong" "weak")
STRATEGIES=("within_group" "cross_group")

echo "=========================================="
echo "Claude 3.5 Haiku Pipeline 开始执行（并行模式）"
echo "时间: $(date)"
echo "最大并行数: ${MAX_PARALLEL} groups"
echo "=========================================="
echo ""

# 遍历每个prompt
for PROMPT in "${PROMPTS[@]}"; do
    echo "=========================================="
    echo "处理 Prompt: ${PROMPT}"
    echo "=========================================="

    # 遍历每个strategy
    for STRATEGY in "${STRATEGIES[@]}"; do
        echo ""
        echo "----------------------------------------"
        echo "Strategy: ${STRATEGY}"
        echo "----------------------------------------"

        # Step 3: Dataset Construction (串行执行，因为比较快)
        echo "[Step 3/6] Dataset Construction..."
        python scripts/step3_dataset_construction.py \
            --config ${CONFIG} \
            --prompt ${PROMPT} \
            --llm_engine ${LLM_ENGINE} \
            --strategy ${STRATEGY}

        # Step 4: Classification (并行执行，按group)
        echo "[Step 4/6] Classification (parallel by group)..."

        # 并行运行每个group
        PIDS=()
        for GROUP_ID in $(seq 0 $((N_GROUPS - 1))); do
            # 如果已经有MAX_PARALLEL个进程在运行，等待其中一个完成
            while [ ${#PIDS[@]} -ge ${MAX_PARALLEL} ]; do
                for i in "${!PIDS[@]}"; do
                    if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                        unset 'PIDS[$i]'
                    fi
                done
                PIDS=("${PIDS[@]}")  # 重新索引数组
                sleep 0.5
            done

            # 启动新的group分类任务
            (
                echo "  Starting classification for group ${GROUP_ID}..."
                python scripts/step4_classification.py \
                    --config ${CONFIG} \
                    --prompt ${PROMPT} \
                    --strategy ${STRATEGY} \
                    --group_id ${GROUP_ID} \
                    --resume \
                    > "logs/${PROMPT}_${STRATEGY}_group${GROUP_ID}_classification.log" 2>&1
                echo "  ✓ Group ${GROUP_ID} completed"
            ) &

            PIDS+=($!)
        done

        # 等待所有group完成
        echo "  Waiting for all groups to complete..."
        for PID in "${PIDS[@]}"; do
            wait $PID
        done
        echo "  ✓ All groups completed"

        # Step 5: Statistical Analysis
        echo "[Step 5/6] Statistical Analysis..."
        python scripts/step5_statistical_analysis.py \
            --config ${CONFIG} \
            --filter_prompt ${PROMPT} \
            --filter_strategy ${STRATEGY}

        # Step 6: Visualization (Single Mode)
        echo "[Step 6/6] Visualization (Single)..."
        python scripts/step6_visualization.py \
            --config ${CONFIG} \
            --mode single \
            --experiment_name ${EXP_NAME} \
            --analysis_file "output/full_experiments/ceas08_claude35haiku/reports/${EXP_NAME}_${PROMPT}_${STRATEGY}_statistical_analysis.json" \
            --output_dir "output/full_experiments/ceas08_claude35haiku/plots/" \
            --plot_types curves heatmap significance dashboard report

        echo "✓ ${PROMPT} - ${STRATEGY} 完成"
    done
done

echo ""
echo "=========================================="
echo "Combined Visualization"
echo "=========================================="

# Step 6: Combined Mode (跨prompt对比)
for STRATEGY in "${STRATEGIES[@]}"; do
    echo "Creating combined plot for ${STRATEGY}..."
    python scripts/step6_visualization.py \
        --config ${CONFIG} \
        --mode combined \
        --experiment_name ${EXP_NAME} \
        --base_dir "output/full_experiments/ceas08_claude35haiku" \
        --strategies ${STRATEGY}
done

echo ""
echo "=========================================="
echo "Pipeline 执行完成！"
echo "时间: $(date)"
echo "=========================================="
echo ""
echo "结果保存位置:"
echo "  数据集: data/full_experiments/ceas08_claude35haiku/datasets/"
echo "  分类结果: output/full_experiments/ceas08_claude35haiku/results/"
echo "  分析报告: output/full_experiments/ceas08_claude35haiku/reports/"
echo "  可视化图表: output/full_experiments/ceas08_claude35haiku/plots/"
