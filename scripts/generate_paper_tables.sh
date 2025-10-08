#!/bin/bash
# 生成论文用的统计表格
# 包含均值±标准差和95%置信区间

set -e

echo "=========================================="
echo "生成论文统计表格"
echo "=========================================="

# 配置参数
MERGED_RESULTS="output/full_experiments/ceas08_gpt41mini/merged_results.json"
OUTPUT_DIR="output/full_experiments/ceas08_gpt41mini/paper_tables"

# 检查输入文件
if [ ! -f "$MERGED_RESULTS" ]; then
    echo "错误: 找不到合并结果文件: $MERGED_RESULTS"
    exit 1
fi

echo ""
echo "输入文件: $MERGED_RESULTS"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 运行表格生成
poetry run python << EOF
from src.utils import generate_all_paper_tables

# 生成所有表格
summary = generate_all_paper_tables(
    merged_results_file='$MERGED_RESULTS',
    output_dir='$OUTPUT_DIR',
    prompts=['original', 'strong', 'weak'],
    strategies=['within_group', 'cross_group'],
    metrics=['f1_score', 'accuracy', 'precision', 'recall', 'auc_roc']
)

print("\n表格生成完成!")
print(f"共生成 {len(summary)} 行统计数据")
EOF

echo ""
echo "=========================================="
echo "生成完成! 查看结果:"
echo "  输出目录: $OUTPUT_DIR"
echo "  - full_summary_statistics.csv (完整汇总)"
echo "  - comparison_all_configs_f1.csv (所有配置对比)"
echo "  - table_*.csv (各个配置的详细表格)"
echo "  - latex_*.tex (LaTeX格式表格)"
echo "=========================================="
