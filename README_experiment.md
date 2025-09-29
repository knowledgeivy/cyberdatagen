# Synthetic Spam Email Data Generation: Multi-Sample Evaluation Framework

## Overview

This framework provides a systematic evaluation methodology for LLM-generated synthetic spam email data effectiveness on imbalanced datasets. It supports step-by-step experimentation, configurable parameters, and comprehensive statistical analysis.

**Based on the experimental design described in `paper/discussion_real.md`**

## Research Objectives

1. **Core Research Question**: Can LLM-generated synthetic spam email data effectively improve machine learning model performance on imbalanced datasets?

2. **Specific Research Questions**:
   - What synthetic spam ratio achieves optimal performance?
   - How do different prompt strategies affect synthetic data quality?
   - What are the performance degradation patterns and critical thresholds?
   - How does within-group vs cross-group generation compare for generalization?

## Experimental Design

### Multi-Sample Evaluation Protocol
- **Groups**: R = 20 (pilot: 10)
- **Sample Size**: N = 9000 per group (pilot: 6000)
- **Spam Ratio**: Fixed 1:9 (realistic imbalance)
- **Synthetic Ratios**: 0%, 25%, 50%, 75%, 100% (pilot), full range 0%-100%
- **Trials**: 5 per configuration

### Statistical Rigor
- Paired t-tests with FDR correction
- Effect size analysis (Cohen's d)
- Performance degradation curve analysis
- Critical threshold identification

## Project Structure

```
./
├── data/paper/              # Experiment data directory
│   ├── processed/           # Preprocessed data
│   ├── synthetic/           # LLM-generated synthetic data
│   └── datasets/            # Built training/testing datasets
├── output/                  # Experiment results
│   ├── results/            # Classification results and metrics
│   ├── plots/              # Visualization charts
│   └── reports/            # Statistical analysis reports
├── src/                    # Source code
│   ├── config/            # Configuration management
│   ├── data_processing/   # Data preprocessing
│   ├── llm_generation/    # LLM generation modules
│   ├── classification/    # Classification models
│   ├── analysis/          # Statistical analysis
│   └── utils/             # Utility functions
├── scripts/               # Execution scripts
├── config/               # Configuration files
└── logs/                # Log files
```

## Quick Start

### 1. Environment Setup

**Note**: This project uses Poetry for dependency management (compatible with existing batch environments)

```bash
# If using existing environment
poetry shell

# Or create new environment
python -m venv pilot_env
source pilot_env/bin/activate  # Linux/Mac

# Set up API keys
echo "OPENAI_API_KEY=your_key_here" > .env

# Create necessary directories
mkdir -p data/paper/{processed,synthetic,datasets} output/{results,plots,reports} logs
```

### 2. Configuration

The pilot study uses `config/pilot_config.yaml` with optimized settings:

```yaml
# Key pilot study parameters
experiment:
  name: "pilot_ceas08_v1"
  dataset: "CEAS-08"
  random_seed: 42

data:
  n_groups: 10                    # Reduced for pilot
  sample_size_per_group: 6000     # Reduced for pilot
  spam_ratio: 0.1                 # Fixed 1:9 ratio
  synthetic_ratios: [0, 25, 50, 75, 100]  # Key points

# Optimized prompts based on previous batch experience
prompts:
  original: "Direct rewrite maintaining malicious intent..."
  strong: "Enhanced spam with obvious marketing language..."
  weak: "Subtle spam appearing more legitimate..."
```

### 3. Step-by-Step Execution

#### Step 1: Data Preprocessing (10 minutes)
```bash
python scripts/step1_data_preprocessing.py --config config/pilot_config.yaml
```

**Expected Output:**
- `data/paper/processed/ceas08_processed.csv.gz`: Complete data with unique IDs
- Balanced groups with stratified sampling
- Test set separation (20%)

#### Step 2: LLM Data Generation (60-90 minutes)
```bash
# Generate for each prompt strategy
python scripts/step2_llm_generation.py --config config/pilot_config.yaml --prompt original
python scripts/step2_llm_generation.py --config config/pilot_config.yaml --prompt strong
python scripts/step2_llm_generation.py --config config/pilot_config.yaml --prompt weak
```

**Expected Output:**
- `data/paper/synthetic/ceas08_synthetic_[prompt]_gpt41mini.csv.gz`
- Quality control metrics and generation statistics

#### Step 3: Dataset Construction (15 minutes)
```bash
python scripts/step3_dataset_construction.py \
    --config config/pilot_config.yaml \
    --prompt original \
    --strategy within_group
```

**Expected Output:**
- Training sets with different synthetic ratios
- Maintained test set consistency
- Data leakage prevention

#### Step 4: Classification Experiments (45-60 minutes)
```bash
python scripts/step4_classification.py \
    --config config/pilot_config.yaml \
    --strategy within_group
```

**Expected Output:**
- `output/results/pilot_ceas08_v1_classification_results.json`
- Performance metrics for all classifiers and configurations

#### Step 5: Statistical Analysis (10 minutes)
```bash
python scripts/step5_statistical_analysis.py \
    --config config/pilot_config.yaml \
    --experiment_name pilot_ceas08_v1
```

**Expected Output:**
- Paired t-tests vs baseline (0% synthetic)
- Multiple comparison correction (FDR)
- Performance degradation analysis
- Critical threshold identification

#### Step 6: Visualization (10 minutes)
```bash
python scripts/step6_visualization.py \
    --config config/pilot_config.yaml \
    --experiment_name pilot_ceas08_v1
```

**Expected Output:**
- Performance degradation curves
- Statistical significance heatmaps
- Prompt strategy comparisons

## Key Features

### Class Imbalance Handling
- Balanced class weights in all classifiers
- Stratified sampling for group creation
- Focus on F1-score, AUC-ROC, AUC-PR metrics

### Statistical Rigor
- Paired t-tests for synthetic ratio comparison
- Benjamini-Hochberg FDR correction
- Effect size analysis (Cohen's d)
- Non-parametric alternatives (Wilcoxon)

### Experimental Strategies
- **Within-group**: Generate synthetic data from same group spam
- **Cross-group**: Generate from other groups (tests generalization)

### Quality Control
- Length similarity checks (30%-300% of original)
- Semantic similarity thresholds (10%-95%)
- Language quality scoring
- Generation success rate monitoring

## Pilot Study Results Interpretation

### Success Criteria
1. **Data Generation Success Rate** > 95%
2. **Baseline Performance** (0% synthetic): F1-score > 0.80
3. **Optimal Synthetic Ratio**: Performance decline < 5%
4. **Prompt Strategy Significance**: p-value < 0.05

### Key Metrics to Monitor
1. **Performance Degradation Curve**: Critical threshold identification
2. **Prompt Strategy Ranking**: Original vs Strong vs Weak effectiveness
3. **Statistical Significance**: Multiple comparison corrected p-values
4. **Effect Sizes**: Cohen's d for practical significance

## Expected Costs and Timeline

### Pilot Study (Total: 3-4 hours)
- **Step 1-3**: 25 minutes (data preparation)
- **Step 4**: 60-90 minutes (LLM generation - main cost)
- **Step 5-6**: 75 minutes (experiments and analysis)

### Resource Requirements
- **API Cost**: ~$20-30 (GPT-4o-mini for pilot)
- **Compute**: 8GB RAM, 4 CPU cores recommended
- **Storage**: ~2GB for pilot study

## Troubleshooting

### Common Issues
1. **API Rate Limits**: Reduce batch_size or add delays
2. **Memory Issues**: Reduce sample_size_per_group
3. **Generation Quality**: Check prompt templates and temperature
4. **Config Path Issues**: Use `config/` not `configs/`

### Log Files
- `logs/pilot_ceas08_v1_preprocessing.log`
- `logs/pilot_ceas08_v1_generation.log`
- `logs/pilot_ceas08_v1_classification.log`

## Next Steps After Pilot

1. **Cross-group Strategy**: Test generalization hypothesis
2. **Parameter Scaling**: Increase R, N to full study values
3. **Multi-LLM Comparison**: Add Claude and Gemini engines
4. **Multi-Dataset Validation**: TREC-07, Assassin, Enron

## Quality Assurance Checklist

- [ ] Data preprocessing completed without errors
- [ ] Synthetic data generation successful for all prompts
- [ ] Classification results reasonable (baseline F1 > 0.8)
- [ ] Statistical analysis shows clear significance patterns
- [ ] Visualization charts are clear and interpretable
- [ ] Experiment is reproducible with same random seed

## Academic Contributions

### Theoretical Contributions
1. **First Systematic Study**: LLM effectiveness in imbalanced cybersecurity data
2. **Statistical Rigor**: Robust multi-sample evaluation framework
3. **Optimization Theory**: Three-dimensional prompt-ratio-model optimization space

### Practical Value
1. **Cost Reduction**: Decreased manual labeling requirements
2. **Performance Enhancement**: Improved model performance in data-scarce scenarios
3. **Reproducibility**: Standardized evaluation protocol

This framework addresses the core research question from `discussion_real.md` with rigorous experimental design, supporting both pilot studies and full-scale experiments for international conference publication.