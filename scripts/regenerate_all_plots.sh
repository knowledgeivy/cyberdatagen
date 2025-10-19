#!/bin/bash
# 重新生成所有用于论文的可视化图表（使用更新后的标题）

set -e

echo "=========================================="
echo "重新生成所有论文图表"
echo "时间: $(date)"
echo "=========================================="
echo ""

# ==========================================
# Track 1: Synthetic-to-Real Detection
# ==========================================
echo "=========================================="
echo "Track 1: Synthetic-to-Real Detection"
echo "=========================================="

# GPT-4.1-mini
echo "[1/2] Generating GPT-4.1-mini combined plot..."
python scripts/step6_visualization.py \
  --config config/full_ceas08_gpt41mini_v1.yaml \
  --mode combined \
  --experiment_name full_ceas08_gpt41mini_v1 \
  --base_dir output/full_experiments/ceas08_gpt41mini \
  --strategies cross_group

# Claude-3.5-Haiku
echo "[2/2] Generating Claude-3.5-Haiku combined plot..."
python scripts/step6_visualization.py \
  --config config/full_ceas08_claude35haiku_v1.yaml \
  --mode combined \
  --experiment_name full_ceas08_claude35haiku_v1 \
  --base_dir output/full_experiments/ceas08_claude35haiku \
  --strategies cross_group

echo "✓ Track 1 plots generated"
echo ""

# ==========================================
# Track 2a: Real-to-Synthetic Detection
# ==========================================
echo "=========================================="
echo "Track 2a: Real-to-Synthetic Detection"
echo "=========================================="

python scripts/visualize_real_enhanced_detection.py \
  --data_file output/real_enhanced_detection/analysis/plotting_data.json \
  --output_dir output/real_enhanced_detection/plots \
  --plot_types combined

echo "✓ Track 2a plots generated"
echo ""

# ==========================================
# Track 2b: Mixed-to-Synthetic Detection
# ==========================================
echo "=========================================="
echo "Track 2b: Mixed-to-Synthetic Detection"
echo "=========================================="

python scripts/visualize_reverse_detection.py \
  --data_file output/reverse_detection/analysis/plotting_data.json \
  --output_dir output/reverse_detection/plots \
  --plot_types combined

echo "✓ Track 2b plots generated"
echo ""

# ==========================================
# Copy plots to paper/pic directory
# ==========================================
echo "=========================================="
echo "复制图片到 paper/pic/"
echo "=========================================="

mkdir -p paper/pic

# Track 1 plots
echo "Copying Track 1 plots..."
cp output/full_experiments/ceas08_gpt41mini/plots/combined/prompts_comparison_cross_group_performance_curves.png \
   paper/pic/prompts_comparison_cross_group_performance_curves_gpt.png

cp output/full_experiments/ceas08_claude35haiku/plots/combined/prompts_comparison_cross_group_performance_curves.png \
   paper/pic/prompts_comparison_cross_group_performance_curves_claude.png

# Track 2a plots
echo "Copying Track 2a plots..."
cp output/real_enhanced_detection/plots/combined/gpt41mini_prompts_comparison_cross_group_real_enhanced_curves.png \
   paper/pic/gpt41mini_prompts_comparison_cross_group_real_enhanced_curves.png

cp output/real_enhanced_detection/plots/combined/claude35haiku_prompts_comparison_cross_group_real_enhanced_curves.png \
   paper/pic/claude35haiku_prompts_comparison_cross_group_real_enhanced_curves.png

# Track 2b plots
echo "Copying Track 2b plots..."
cp output/reverse_detection/plots/combined/gpt41mini_prompts_comparison_cross_group_reverse_performance_curves.png \
   paper/pic/gpt41mini_prompts_comparison_cross_group_reverse_performance_curves.png

cp output/reverse_detection/plots/combined/claude35haiku_prompts_comparison_cross_group_reverse_performance_curves.png \
   paper/pic/claude35haiku_prompts_comparison_cross_group_reverse_performance_curves.png

echo "✓ All plots copied to paper/pic/"
echo ""

echo "=========================================="
echo "所有图表生成完成！"
echo "时间: $(date)"
echo "=========================================="
echo ""
echo "生成的图片："
echo "  Track 1 (Synthetic-to-Real):"
echo "    - paper/pic/prompts_comparison_cross_group_performance_curves_gpt.png"
echo "    - paper/pic/prompts_comparison_cross_group_performance_curves_claude.png"
echo ""
echo "  Track 2a (Real-to-Synthetic):"
echo "    - paper/pic/gpt41mini_prompts_comparison_cross_group_real_enhanced_curves.png"
echo "    - paper/pic/claude35haiku_prompts_comparison_cross_group_real_enhanced_curves.png"
echo ""
echo "  Track 2b (Mixed-to-Synthetic):"
echo "    - paper/pic/gpt41mini_prompts_comparison_cross_group_reverse_performance_curves.png"
echo "    - paper/pic/claude35haiku_prompts_comparison_cross_group_reverse_performance_curves.png"
echo ""
