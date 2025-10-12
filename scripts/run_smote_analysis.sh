#!/bin/bash
# Run complete SMOTE analysis pipeline: merge, stats, visualization

CONFIG="config/full_ceas08_smote_v1.yaml"
EXPERIMENT_NAME="full_ceas08_smote_v1"
VARIANT="smote"
RESULTS_DIR="output/full_experiments/ceas08_smote/results"
REPORTS_DIR="output/full_experiments/ceas08_smote/reports"
PLOTS_DIR="output/full_experiments/ceas08_smote/plots"

mkdir -p "$REPORTS_DIR"
mkdir -p "$PLOTS_DIR"

echo "=========================================="
echo "SMOTE分析流程"
echo "=========================================="
echo ""

for strategy in within_group cross_group; do
    echo "=========================================="
    echo "Processing strategy: $strategy"
    echo "=========================================="

    # Step 1: 合并group结果
    echo "[1/3] 合并group结果..."
    python scripts/merge_smote_results.py \
        --results_dir "$RESULTS_DIR" \
        --experiment_name "$EXPERIMENT_NAME" \
        --variant "$VARIANT" \
        --strategy "$strategy"

    if [ $? -ne 0 ]; then
        echo "❌ 合并失败: $strategy"
        continue
    fi

    # Step 2: 统计分析
    echo "[2/3] 统计分析..."
    python scripts/step5_statistical_analysis.py \
        --config "$CONFIG" \
        --results_file "${RESULTS_DIR}/${EXPERIMENT_NAME}_${VARIANT}_${strategy}_classification_results.json" \
        --filter_strategy "$strategy" \
        --output_file "${REPORTS_DIR}/${EXPERIMENT_NAME}_${VARIANT}_${strategy}_statistical_analysis.json"

    if [ $? -ne 0 ]; then
        echo "❌ 统计分析失败: $strategy"
        continue
    fi

    # Step 3: 可视化
    echo "[3/3] 生成可视化..."
    mkdir -p "${PLOTS_DIR}/${VARIANT}/${strategy}"

    python scripts/step6_visualization.py \
        --config "$CONFIG" \
        --mode single \
        --analysis_file "${REPORTS_DIR}/${EXPERIMENT_NAME}_${VARIANT}_${strategy}_statistical_analysis.json" \
        --output_dir "${PLOTS_DIR}/${VARIANT}/${strategy}" \
        --experiment_name "${EXPERIMENT_NAME}_${VARIANT}_${strategy}" \
        --plot_types curves heatmap significance report

    if [ $? -eq 0 ]; then
        echo "✅ $strategy 完成"
    else
        echo "❌ 可视化失败: $strategy"
    fi

    echo ""
done

echo ""
echo "=========================================="
echo "SMOTE分析完成!"
echo "=========================================="
echo "Results:"
echo "  - Merged: $RESULTS_DIR"
echo "  - Analysis: $REPORTS_DIR"
echo "  - Plots: $PLOTS_DIR"
echo "=========================================="
