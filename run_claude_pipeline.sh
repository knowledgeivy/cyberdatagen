#!/bin/bash
# Claude 3.5 Haiku 完整实验 Pipeline
# Steps 3-6: Dataset Construction → Classification → Analysis → Visualization

set -e  # 遇到错误立即退出

CONFIG="config/full_ceas08_claude35haiku_v1.yaml"
EXP_NAME="full_ceas08_claude35haiku_v1"
LLM_ENGINE="claude-3-5-haiku"

# 定义prompts和strategies
PROMPTS=("original" "strong" "weak")
STRATEGIES=("within_group" "cross_group")

echo "=========================================="
echo "Claude 3.5 Haiku Pipeline 开始执行"
echo "时间: $(date)"
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

        # Step 3: Dataset Construction
        echo "[Step 3/6] Dataset Construction..."
        python scripts/step3_dataset_construction.py \
            --config ${CONFIG} \
            --prompt ${PROMPT} \
            --llm_engine ${LLM_ENGINE} \
            --strategy ${STRATEGY}

        # Step 4: Classification
        echo "[Step 4/6] Classification..."
        python scripts/step4_classification.py \
            --config ${CONFIG} \
            --prompt ${PROMPT} \
            --strategy ${STRATEGY}

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
