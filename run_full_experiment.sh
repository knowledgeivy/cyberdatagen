#!/bin/bash
# Full-Scale Experiment Execution Script
# Experiment: full_ceas08_gpt41mini_v1
# Created: 2025-10-03

set -e  # Exit on error

CONFIG="config/full_ceas08_gpt41mini_v1.yaml"
EXPERIMENT="full_ceas08_gpt41mini_v1"

echo "=========================================="
echo "Full-Scale Experiment: $EXPERIMENT"
echo "=========================================="
echo ""

# Step 1: Data Preprocessing
echo "Step 1: Data Preprocessing"
echo "------------------------------------------"
poetry run python scripts/step1_data_preprocessing.py --config $CONFIG
echo "✅ Step 1 completed"
echo ""

# Step 2: LLM Generation (3 prompts)
echo "Step 2: LLM Generation (3 prompts)"
echo "------------------------------------------"
echo "  Generating with 'original' prompt..."
poetry run python scripts/step2_llm_generation.py --config $CONFIG --prompt original
echo "  ✅ Original prompt completed"
echo ""

echo "  Generating with 'strong' prompt..."
poetry run python scripts/step2_llm_generation.py --config $CONFIG --prompt strong
echo "  ✅ Strong prompt completed"
echo ""

echo "  Generating with 'weak' prompt..."
poetry run python scripts/step2_llm_generation.py --config $CONFIG --prompt weak
echo "  ✅ Weak prompt completed"
echo ""
echo "✅ Step 2 completed"
echo ""

# Create backup of synthetic data
echo "Creating backup of synthetic data..."
tar -czf synthetic_backup_$(date +%Y%m%d_%H%M%S).tar.gz \
  data/full_experiments/ceas08_gpt41mini/synthetic/
echo "✅ Backup created"
echo ""

# Step 3: Dataset Construction (4 strategies, only original prompt)
echo "Step 3: Dataset Construction (4 strategies)"
echo "------------------------------------------"
for strategy in within_group cross_group real_fixed_random_synthetic full_random; do
  echo "  Building datasets for strategy: $strategy"
  poetry run python scripts/step3_dataset_construction.py \
    --config $CONFIG \
    --prompt original \
    --strategy $strategy
  echo "  ✅ $strategy completed"
done
echo "✅ Step 3 completed"
echo ""

# Step 4: Classification Experiments (4 strategies, sequential with --resume)
echo "Step 4: Classification Experiments (4 strategies)"
echo "------------------------------------------"
echo "⚠️  This step will take 24-36 hours"
echo ""

for strategy in within_group cross_group real_fixed_random_synthetic full_random; do
  echo "  Running classification for strategy: $strategy"
  poetry run python scripts/step4_classification.py \
    --config $CONFIG \
    --strategy $strategy \
    --resume
  echo "  ✅ $strategy completed"
done
echo "✅ Step 4 completed"
echo ""

# Create backup of results
echo "Creating backup of classification results..."
tar -czf results_backup_$(date +%Y%m%d_%H%M%S).tar.gz \
  output/full_experiments/ceas08_gpt41mini/results/
echo "✅ Results backup created"
echo ""

# Step 5: Statistical Analysis
echo "Step 5: Statistical Analysis"
echo "------------------------------------------"
poetry run python scripts/step5_enhanced_statistical_analysis.py --config $CONFIG
echo "✅ Step 5 completed"
echo ""

# Step 6: Visualization
echo "Step 6: Visualization"
echo "------------------------------------------"
poetry run python scripts/step6_enhanced_visualization.py \
  --config $CONFIG \
  --results output/full_experiments/ceas08_gpt41mini/results/${EXPERIMENT}_classification_results.json
echo "✅ Step 6 completed"
echo ""

echo "=========================================="
echo "🎉 Full Experiment Completed Successfully!"
echo "=========================================="
echo ""
echo "Results location:"
echo "  - Classification: output/full_experiments/ceas08_gpt41mini/results/"
echo "  - Visualizations: output/full_experiments/ceas08_gpt41mini/plots/"
echo "  - Reports: output/full_experiments/ceas08_gpt41mini/reports/"
echo "  - Logs: logs/full_experiments/ceas08_gpt41mini/"
echo ""
echo "Backups created:"
ls -lh synthetic_backup_*.tar.gz results_backup_*.tar.gz 2>/dev/null | tail -2
