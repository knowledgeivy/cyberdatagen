# SMOTE Baseline - Quick Start Guide

## Overview

SMOTE (Synthetic Minority Over-sampling Technique) baseline for comparing with LLM-based synthetic data generation.

**Key Difference**: SMOTE operates in **feature space** (TF-IDF), not text space like LLMs.

## Quick Start

### Prerequisites

Ensure processed data exists:
```bash
ls data/full_experiments/ceas08_gpt41mini/processed/
# Should see: *_processed.csv.gz
```

### Run Complete Pipeline

```bash
# Run all experiments in parallel (3-4 hours)
bash scripts/run_smote_pipeline_parallel.sh
```

This will:
1. Generate SMOTE-augmented datasets (20 groups × 11 ratios × 2 strategies)
2. Train classifiers (SVM, Random Forest)
3. Save results in `output/full_experiments/ceas08_smote/results/`

### Run Single Group (for testing)

```bash
# Test with one group
python scripts/run_smote_experiments.py \
    --config config/full_ceas08_smote_v1.yaml \
    --variant smote \
    --strategy within_group \
    --group_id 0
```

### Run with ADASYN (alternative variant)

```bash
# Edit run_smote_pipeline_parallel.sh
# Change: VARIANT="smote" to VARIANT="adasyn"
bash scripts/run_smote_pipeline_parallel.sh
```

## Experiment Configuration

**Total Configurations**: 1,760
- 2 variants (SMOTE, ADASYN)
- 2 strategies (within_group, cross_group)
- 11 ratios (0%, 10%, ..., 100%)
- 20 groups
- 2 classifiers (SVM, Random Forest)

**Total Training Runs**: 8,800 (1,760 × 5 trials)

**Runtime**: ~3-4 hours (local CPU, no API costs)

## Output Structure

```
output/full_experiments/ceas08_smote/
├── results/
│   ├── full_ceas08_smote_v1_smote_within_group_group0_results.json
│   ├── full_ceas08_smote_v1_smote_within_group_group1_results.json
│   └── ... (40 files total: 20 groups × 2 strategies)
├── reports/
│   └── (generated after merging results)
└── plots/
    └── (generated after visualization)

data/full_experiments/ceas08_smote/
└── datasets/
    ├── smote_within_group/
    │   ├── group_0_ratio_0.pkl
    │   ├── group_0_ratio_10.pkl
    │   └── ... (220 files: 20 groups × 11 ratios)
    └── smote_cross_group/
        └── ... (220 files)
```

## Analysis and Visualization

After experiments complete:

### 1. Merge Results (if needed)
```bash
# Similar to LLM experiments
python scripts/merge_group_results.py \
    --base_dir output/full_experiments/ceas08_smote/results \
    --experiment_name full_ceas08_smote_v1 \
    --strategies within_group cross_group
```

### 2. Statistical Analysis
```bash
# For each strategy
python scripts/step5_statistical_analysis.py \
    --config config/full_ceas08_smote_v1.yaml \
    --results_file output/full_experiments/ceas08_smote/results/full_ceas08_smote_v1_smote_within_group_classification_results.json \
    --output_file output/full_experiments/ceas08_smote/reports/full_ceas08_smote_v1_smote_within_group_statistical_analysis.json \
    --filter_prompt smote \
    --filter_strategy within_group
```

### 3. Visualization
```bash
# Single mode
python scripts/step6_visualization.py \
    --config config/full_ceas08_smote_v1.yaml \
    --mode single \
    --analysis_file output/full_experiments/ceas08_smote/reports/full_ceas08_smote_v1_smote_within_group_statistical_analysis.json \
    --output_dir output/full_experiments/ceas08_smote/plots/smote/within_group/

# Combined mode (compare with LLMs)
python scripts/step6_visualization.py \
    --config config/full_ceas08_smote_v1.yaml \
    --mode combined \
    --experiment_name full_ceas08_smote_v1 \
    --base_dir output/full_experiments/ceas08_smote \
    --strategies within_group cross_group
```

## Comparison with LLMs

After completing SMOTE, GPT-4.1-mini, and Claude-3.5-Haiku experiments, compare:

```python
from pathlib import Path
import pandas as pd

# Load all results
smote_results = pd.read_csv('output/full_experiments/ceas08_smote/reports/csv/...')
gpt_results = pd.read_csv('output/full_experiments/ceas08_gpt41mini/reports/csv/...')
claude_results = pd.read_csv('output/full_experiments/ceas08_claude35haiku/reports/csv/...')

# Compare F1-scores across methods
comparison = pd.DataFrame({
    'SMOTE': smote_results['f1_score'],
    'GPT-4.1-mini': gpt_results['f1_score'],
    'Claude-3.5-Haiku': claude_results['f1_score']
})

print(comparison.describe())
```

## Key Metrics to Compare

1. **Performance**: F1-score, AUC-ROC, Balanced Accuracy
2. **Cost**: SMOTE (free) vs LLM ($50-80)
3. **Time**: SMOTE (3-4 hours) vs LLM (6-8 hours + API time)
4. **Quality**: Feature-space vs Text-space vs Semantic generation

## Troubleshooting

### Error: "No processed data found"
```bash
# Run step1 first to generate processed data
python scripts/step1_data_preprocessing.py \
    --config config/full_ceas08_smote_v1.yaml
```

### Error: "imbalanced-learn not installed"
```bash
# Install required package
pip install imbalanced-learn
# or
poetry add imbalanced-learn
```

### Memory Error
```bash
# Reduce max_parallel in run_smote_pipeline_parallel.sh
MAX_PARALLEL=4  # Instead of 8
```

### Check Progress
```bash
# Count completed results
ls output/full_experiments/ceas08_smote/results/*.json | wc -l
# Should eventually be 40 (20 groups × 2 strategies)
```

## Academic Value

**Research Question**: Is LLM-based semantic generation superior to traditional feature-space augmentation?

**Expected Findings**:
1. SMOTE may perform well at low ratios (10-30%)
2. LLM should outperform at high ratios (70-100%)
3. Cost-benefit tradeoff analysis

**Paper Contribution**:
- First systematic comparison of LLM vs SMOTE for spam generation
- Feature-space vs semantic-space augmentation
- Practical guidance for method selection

## Next Steps

1. ✅ Run SMOTE experiments
2. ⏳ Compare with LLM results
3. ⏳ Update paper with comparative analysis
4. ⏳ Generate final comparison plots
