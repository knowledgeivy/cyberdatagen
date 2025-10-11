# CyberData: LLM-based Synthetic Spam Email Data Generation

A systematic evaluation framework for assessing LLM-generated synthetic spam email data effectiveness on imbalanced datasets.

## Overview

This framework evaluates whether LLM-generated synthetic spam data can effectively improve machine learning model performance on imbalanced email datasets. It implements a rigorous multi-sample evaluation protocol with statistical analysis.

## Research Objectives

1. **Core Question**: Can LLM-generated synthetic spam effectively improve ML model performance on imbalanced datasets?
2. **Specific Questions**:
   - What synthetic spam ratio achieves optimal performance?
   - How do different prompt strategies affect data quality?
   - What are the performance degradation patterns?
   - How do within-group vs cross-group mixing strategies compare?

## Experimental Design

### Multi-Sample Evaluation Protocol
- **Groups**: R = 20
- **Sample Size**: N = 1100 per group
- **Spam Ratio**: Fixed 1:9 (realistic imbalance)
- **Synthetic Ratios**: 0%, 10%, 20%, ..., 100%
- **Prompt Strategies**: Original, Strong, Weak
- **Mixing Strategies**: Within-group, Cross-group
- **LLM Models**: GPT-4.1-mini, Claude-3.5-Haiku

### Statistical Rigor
- Paired t-tests with FDR correction
- Effect size analysis (Cohen's d)
- Performance degradation curve analysis
- Sensitivity analysis (regression slopes)

## Installation

### Requirements
- Python 3.12+
- OpenAI API key (GPT-4.1-mini)
- Anthropic API key (Claude-3.5-Haiku)

### Setup
```bash
# Clone repository
git clone <repository_url>
cd cyberdata

# Install with Poetry
poetry install

# Set up environment
cp .env.example .env
# Add API keys to .env
```

## Quick Start

### Full Experiment Pipeline

#### For GPT-4.1-mini:
```bash
# Run complete pipeline (Steps 3-6)
bash scripts/run_gpt_pipeline_parallel.sh
```

#### For Claude-3.5-Haiku:
```bash
# Run complete pipeline (Steps 3-6)
bash scripts/run_claude_pipeline_parallel.sh
```

#### Regenerate Analysis Only:
```bash
# Regenerate Claude visualizations
bash scripts/regenerate_claude_analysis.sh
```

### Step-by-Step Execution

#### Step 1: Data Preprocessing
```bash
python scripts/step1_data_preprocessing.py \
    --config config/full_ceas08_gpt41mini_v1.yaml
```

#### Step 2: LLM Generation
```bash
python scripts/step2_llm_generation.py \
    --config config/full_ceas08_gpt41mini_v1.yaml \
    --prompt original \
    --llm_engine gpt-4.1-mini
```

#### Step 3: Dataset Construction
```bash
python scripts/step3_dataset_construction.py \
    --config config/full_ceas08_gpt41mini_v1.yaml \
    --prompt original \
    --llm_engine gpt-4.1-mini \
    --strategy within_group
```

#### Step 4: Classification
```bash
python scripts/step4_classification.py \
    --config config/full_ceas08_gpt41mini_v1.yaml \
    --prompt original \
    --strategy within_group \
    --group_id 0 \
    --resume
```

#### Step 5: Statistical Analysis
```bash
python scripts/step5_statistical_analysis.py \
    --config config/full_ceas08_gpt41mini_v1.yaml \
    --results_file output/.../results.json \
    --output_file output/.../analysis.json \
    --filter_prompt original \
    --filter_strategy within_group
```

#### Step 6: Visualization
```bash
# Single mode
python scripts/step6_visualization.py \
    --config config/full_ceas08_gpt41mini_v1.yaml \
    --mode single \
    --analysis_file output/.../analysis.json \
    --output_dir output/.../plots/

# Combined mode (multi-prompt comparison)
python scripts/step6_visualization.py \
    --config config/full_ceas08_gpt41mini_v1.yaml \
    --mode combined \
    --experiment_name full_ceas08_gpt41mini_v1 \
    --base_dir output/full_experiments/ceas08_gpt41mini \
    --strategies within_group cross_group
```

## Project Structure

```
cyberdata/
├── config/                          # Configuration files
│   ├── full_ceas08_gpt41mini_v1.yaml
│   └── full_ceas08_claude35haiku_v1.yaml
├── scripts/                         # Execution scripts
│   ├── step1_data_preprocessing.py
│   ├── step2_llm_generation.py
│   ├── step3_dataset_construction.py
│   ├── step4_classification.py
│   ├── step5_statistical_analysis.py
│   ├── step6_visualization.py
│   ├── run_gpt_pipeline_parallel.sh
│   ├── run_claude_pipeline_parallel.sh
│   └── regenerate_claude_analysis.sh
├── src/                            # Source code
│   ├── config/                     # Configuration management
│   ├── data_processing/            # Data preprocessing
│   ├── llm_generation/             # LLM generation modules
│   ├── classification/             # Classification models
│   ├── analysis/                   # Statistical analysis
│   └── utils/                      # Utility functions
├── data/                           # Data directory
│   └── full_experiments/
│       ├── processed/              # Preprocessed data
│       ├── synthetic/              # LLM-generated data
│       └── datasets/               # Training/testing datasets
├── output/                         # Experiment results
│   └── full_experiments/
│       ├── ceas08_gpt41mini/
│       │   ├── results/           # Classification results
│       │   ├── reports/           # Statistical analysis
│       │   └── plots/             # Visualizations
│       │       ├── original/
│       │       │   ├── within_group/
│       │       │   └── cross_group/
│       │       ├── strong/
│       │       │   ├── within_group/
│       │       │   └── cross_group/
│       │       ├── weak/
│       │       │   ├── within_group/
│       │       │   └── cross_group/
│       │       └── combined/      # Multi-prompt comparisons
│       └── ceas08_claude35haiku/
│           └── (same structure)
└── logs/                          # Log files
```

## Key Features

### Class Imbalance Handling
- Balanced class weights in all classifiers
- Stratified sampling for group creation
- Focus on F1-score, AUC-ROC, AUC-PR, Balanced Accuracy

### Statistical Analysis
- Paired t-tests for synthetic ratio comparison
- Benjamini-Hochberg FDR correction
- Effect size analysis (Cohen's d)
- Sensitivity analysis (performance degradation slopes)
- Critical threshold identification

### Prompt Strategies

#### Original Prompt
Direct rewriting maintaining malicious intent and technical patterns

#### Strong Prompt
Enhanced spam with obvious marketing language and urgency cues

#### Weak Prompt
Subtle spam appearing more legitimate with professional tone

### Mixing Strategies

#### Within-group Mixing
- **Combination**: Real Group i + Synthetic Group i
- **Purpose**: Test synthetic-real compatibility within same distribution

#### Cross-group Mixing
- **Combination**: Real Group i + Synthetic Group j≠i
- **Purpose**: Test generalization across different spam patterns

## Experiment Configurations

### GPT-4.1-mini Experiments
- **Total configs**: 20 groups × 11 ratios × 3 prompts × 2 strategies = 1,320
- **Config file**: `config/full_ceas08_gpt41mini_v1.yaml`
- **Output dir**: `output/full_experiments/ceas08_gpt41mini/`

### Claude-3.5-Haiku Experiments
- **Total configs**: 20 groups × 11 ratios × 3 prompts × 2 strategies = 1,320
- **Config file**: `config/full_ceas08_claude35haiku_v1.yaml`
- **Output dir**: `output/full_experiments/ceas08_claude35haiku/`

## Outputs

### Classification Results
- Performance metrics for all configurations
- Success/failure tracking
- Detailed model predictions

### Statistical Reports
- Descriptive statistics (mean, std, CI)
- Hypothesis test results (t-tests, p-values)
- Performance degradation analysis
- Sensitivity analysis (regression slopes)
- Critical threshold identification

### Visualizations
- Performance curves by synthetic ratio
- Performance degradation heatmaps
- Statistical significance plots
- Interactive dashboards (HTML)
- Summary reports (HTML)
- Multi-prompt comparison plots

## Key Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: Spam detection precision
- **Recall/Sensitivity**: Spam detection recall
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve
- **AUC-PR**: Area under precision-recall curve
- **Balanced Accuracy**: Average of sensitivity and specificity

## Advanced Usage

### Python API for Batch Processing

#### Batch Analysis and Visualization
```python
from src.utils import run_batch_analysis, run_batch_visualization

# Batch analysis for all prompt×strategy combinations
run_batch_analysis(
    config_file='config/full_ceas08_gpt41mini_v1.yaml',
    results_file='output/full_experiments/ceas08_gpt41mini/results/full_ceas08_gpt41mini_v1_classification_results.json',
    reports_dir='output/full_experiments/ceas08_gpt41mini/reports',
    experiment_name='full_ceas08_gpt41mini_v1'
)

# Batch visualization
run_batch_visualization(
    config_file='config/full_ceas08_gpt41mini_v1.yaml',
    base_dir='output/full_experiments/ceas08_gpt41mini',
    experiment_name='full_ceas08_gpt41mini_v1'
)
```

#### Merge Classification Results
```python
from src.utils import merge_classification_results

# Merge group results into unified files
merge_classification_results(
    output_dir='output/full_experiments/ceas08_gpt41mini/results',
    experiment_name='full_ceas08_gpt41mini_v1',
    strategies=['within_group', 'cross_group'],
    total_groups=20
)
```

#### Export to CSV
```python
from src.utils import export_all_analyses_to_csv
from pathlib import Path

# Export detailed metrics to CSV format
export_all_analyses_to_csv(
    base_dir=Path('output/full_experiments/ceas08_gpt41mini'),
    experiment_name='full_ceas08_gpt41mini_v1',
    prompts=['original', 'strong', 'weak'],
    strategies=['within_group', 'cross_group']
)
```

Output CSV location: `output/.../reports/csv/`

CSV columns include:
- `synthetic_ratio`: 0%, 10%, ..., 100%
- `performance`: Metric value at this ratio
- `regression_slope`: Sensitivity slope
- `regression_r_squared`: R² value
- `regression_p_value`: Regression p-value
- `p_value`: Hypothesis test p-value vs baseline
- `absolute_degradation`: Absolute performance drop
- `relative_degradation`: Relative performance drop

#### Sensitivity Analysis Display
```python
from src.utils import display_sensitivity_analysis, create_sensitivity_table
from pathlib import Path

# Display single file sensitivity
display_sensitivity_analysis(
    Path('output/.../analysis.json'),
    show_all_metrics=False
)

# Create comparison table across prompts/strategies
create_sensitivity_table(
    Path('output/full_experiments/ceas08_gpt41mini'),
    prompts=['original', 'strong', 'weak'],
    strategies=['within_group', 'cross_group']
)
```

Sensitivity metrics interpretation:
- **Negative slope**: Performance decreases with more synthetic data
- **Larger absolute value**: More sensitive to synthetic data
- **R² close to 1**: Strong linear relationship

### Complete Workflow Example
```python
from pathlib import Path
from src.utils import (
    merge_classification_results,
    run_batch_analysis,
    run_batch_visualization,
    create_sensitivity_table,
    export_all_analyses_to_csv
)

# 1. Merge results
merge_classification_results(
    output_dir='output/full_experiments/ceas08_gpt41mini/results',
    experiment_name='full_ceas08_gpt41mini_v1',
    strategies=['within_group', 'cross_group']
)

# 2. Batch analysis
run_batch_analysis(
    config_file='config/full_ceas08_gpt41mini_v1.yaml',
    results_file='output/full_experiments/ceas08_gpt41mini/results/full_ceas08_gpt41mini_v1_classification_results.json',
    reports_dir='output/full_experiments/ceas08_gpt41mini/reports',
    experiment_name='full_ceas08_gpt41mini_v1'
)

# 3. Batch visualization
run_batch_visualization(
    config_file='config/full_ceas08_gpt41mini_v1.yaml',
    base_dir='output/full_experiments/ceas08_gpt41mini',
    experiment_name='full_ceas08_gpt41mini_v1'
)

# 4. Export to CSV
export_all_analyses_to_csv(
    base_dir=Path('output/full_experiments/ceas08_gpt41mini'),
    experiment_name='full_ceas08_gpt41mini_v1',
    prompts=['original', 'strong', 'weak'],
    strategies=['within_group', 'cross_group']
)

# 5. View sensitivity table
create_sensitivity_table(
    Path('output/full_experiments/ceas08_gpt41mini'),
    prompts=['original', 'strong', 'weak'],
    strategies=['within_group', 'cross_group']
)
```

## Cost and Timeline

### Full Experiment (per LLM)
- **LLM Generation**: ~$50-80 (main cost)
- **Classification**: 6-8 hours (parallel execution)
- **Analysis**: 30 minutes
- **Storage**: ~20GB per LLM

### Resource Requirements
- **Compute**: 32GB RAM, 8+ CPU cores recommended
- **API**: OpenAI or Anthropic API access

## Version

Current version: 2.2.0

## License

MIT License - see LICENSE file for details

## Contact

knowledgeivy01@gmail.com
