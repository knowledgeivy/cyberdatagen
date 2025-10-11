#!/bin/bash
# 重新生成Claude 3.5 Haiku的分析报告和可视化
# 按照正确的目录结构组织：prompt/strategy

set -e

CONFIG="config/full_ceas08_claude35haiku_v1.yaml"
EXP_NAME="full_ceas08_claude35haiku_v1"
BASE_DIR="output/full_experiments/ceas08_claude35haiku"

echo "=========================================="
echo "重新生成Claude分析报告和可视化"
echo "时间: $(date)"
echo "=========================================="
echo ""

# 遍历每个prompt
for PROMPT in original strong weak; do
  echo "=========================================="
  echo "处理 Prompt: ${PROMPT}"
  echo "=========================================="

  # 遍历每个strategy
  for STRATEGY in within_group cross_group; do
    echo ""
    echo "----------------------------------------"
    echo "Strategy: ${STRATEGY}"
    echo "----------------------------------------"

    # 定义路径
    RESULTS_FILE="${BASE_DIR}/results/${EXP_NAME}_${PROMPT}_${STRATEGY}_classification_results.json"
    REPORT_FILE="${BASE_DIR}/reports/${EXP_NAME}_${PROMPT}_${STRATEGY}_statistical_analysis.json"
    PLOTS_DIR="${BASE_DIR}/plots/${PROMPT}/${STRATEGY}"

    echo "Results: ${RESULTS_FILE}"
    echo "Report: ${REPORT_FILE}"
    echo "Plots: ${PLOTS_DIR}"

    # 检查results文件是否存在
    if [ ! -f "${RESULTS_FILE}" ]; then
      echo "错误: Results文件不存在"
      continue
    fi

    # Step 5: Statistical Analysis
    echo "[Step 5] Statistical Analysis..."
    python scripts/step5_statistical_analysis.py \
      --config ${CONFIG} \
      --results_file "${RESULTS_FILE}" \
      --output_file "${REPORT_FILE}" \
      --filter_prompt ${PROMPT} \
      --filter_strategy ${STRATEGY}

    # 创建plots子目录
    mkdir -p "${PLOTS_DIR}"

    # Step 6: Visualization
    echo "[Step 6] Visualization..."
    python scripts/step6_visualization.py \
      --config ${CONFIG} \
      --mode single \
      --experiment_name "${EXP_NAME}_${PROMPT}_${STRATEGY}" \
      --analysis_file "${REPORT_FILE}" \
      --output_dir "${PLOTS_DIR}" \
      --plot_types curves heatmap significance dashboard report

    echo "✓ ${PROMPT} - ${STRATEGY} 完成"
  done
done

echo ""
echo "=========================================="
echo "生成Combined Visualization"
echo "=========================================="

# Combined visualization (放在plots/combined目录)
for STRATEGY in within_group cross_group; do
  echo "Creating combined plot for ${STRATEGY}..."
  python scripts/step6_visualization.py \
    --config ${CONFIG} \
    --mode combined \
    --experiment_name ${EXP_NAME} \
    --base_dir "${BASE_DIR}" \
    --strategies ${STRATEGY}
done

echo ""
echo "=========================================="
echo "分析和可视化生成完成！"
echo "时间: $(date)"
echo "=========================================="
echo ""
echo "结果保存位置:"
echo "  统计报告: ${BASE_DIR}/reports/"
echo "  可视化图表: ${BASE_DIR}/plots/"
echo "    ├── original/"
echo "    │   ├── within_group/"
echo "    │   └── cross_group/"
echo "    ├── strong/"
echo "    │   ├── within_group/"
echo "    │   └── cross_group/"
echo "    ├── weak/"
echo "    │   ├── within_group/"
echo "    │   └── cross_group/"
echo "    └── combined/"
