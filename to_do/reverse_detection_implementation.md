# Reverse Detection Experiment - Implementation Details

**Created**: 2025-10-12
**Status**: Ready for Review - Code prepared but not executed
**Priority**: ⭐⭐⭐⭐⭐ Must Do (Tier 1)

---

## 📋 Experiment Overview

### Objective
Test whether classifiers trained on **pure real data** can detect **LLM-generated synthetic spam** in addition to real spam.

### Hypothesis
If LLM-generated spam contains detectable artifacts or differs systematically from real spam, then classifiers trained solely on real data will:
- Option A: **Successfully detect** synthetic spam (high F1) → LLMs produce realistic spam
- Option B: **Fail to detect** synthetic spam (low F1) → LLMs produce qualitatively different spam

### Real-World Motivation
Attackers increasingly use LLMs to generate spam/phishing emails at scale. This experiment tests operational readiness: can existing real-data-trained classifiers handle LLM-generated threats?

---

## 🔬 Experimental Design

### Training Set
- **Composition**: 0% synthetic (pure real data)
- **Size**: Same as baseline (1,000 samples per group)
- **Spam Ratio**: 10% (100 real spam + 900 real ham)
- **Groups**: 20 independent groups (same as forward experiment)

### Testing Set
- **Composition**: 100% synthetic spam + real ham
- **Synthetic Spam Sources**:
  - GPT-4.1-mini (3 prompts: original, strong, weak)
  - Claude-3.5-Haiku (3 prompts: original, strong, weak)
  - SMOTE (no prompts)
- **Real Ham**: Same fixed test set ham samples
- **Spam Ratio**: ~10% (to match forward experiment)

### Experimental Matrix

**Total Experiments**: 560 classification runs

```
Methods: 3 (GPT, Claude, SMOTE)
├── LLM Methods: 2 models × 3 prompts × 2 strategies = 12 configurations
└── SMOTE: 1 method × 2 strategies = 2 configurations

Per Configuration:
├── Groups: 20
└── Classifiers: 2 (SVM, Random Forest)

Total: 14 configurations × 20 groups × 2 classifiers = 560 experiments
```

**但只测试100% synthetic**, 不需要0-90%的ratios，所以远少于forward experiment的2200个实验。

---

## 📁 File Structure

### Input Data Locations

**Real Training Data** (baseline, already exists):
```
output/prepared_data/groups/
├── group_0/
│   ├── train.pkl  # Pure real data
│   └── test.pkl   # Real ham + real spam
├── group_1/
...
└── group_19/
```

**Synthetic Data** (already generated):
```
output/full_experiments/ceas08_gpt41mini/datasets/
├── original_within_group_100/
│   ├── group_0_train.pkl
│   └── ...
├── original_cross_group_100/
├── strong_within_group_100/
├── strong_cross_group_100/
├── weak_within_group_100/
└── weak_cross_group_100/

output/full_experiments/ceas08_claude35haiku/datasets/
[同样结构]

output/full_experiments/ceas08_smote/datasets/
├── smote_within_group_100/
└── smote_cross_group_100/
```

### Output Paths

**Results Directory**:
```
output/reverse_detection/
├── results/
│   ├── gpt41mini_original_within_group_reverse_results.json
│   ├── gpt41mini_strong_within_group_reverse_results.json
│   ├── ...
│   ├── claude35haiku_original_within_group_reverse_results.json
│   ├── ...
│   └── smote_within_group_reverse_results.json
│
├── analysis/
│   ├── reverse_detection_summary.json
│   ├── method_comparison.json
│   └── statistical_analysis.json
│
├── plots/
│   ├── reverse_detection_f1_comparison.png
│   ├── forward_vs_reverse_comparison.png
│   └── method_specific_plots/
│
└── logs/
    └── reverse_detection.log
```

---

## 🛠️ Implementation Scripts

### Script 1: `reverse_detection_experiment.py`

**Location**: `/Users/tianyu/Notebooks/cyberdata/scripts/reverse_detection_experiment.py`

**Purpose**: Main experiment runner

**Key Functions**:
1. `load_baseline_training_data(group_id)` - Load pure real training data
2. `load_synthetic_test_spam(method, prompt, strategy, group_id)` - Load 100% synthetic spam
3. `construct_reverse_test_set(synthetic_spam, real_ham)` - Combine synthetic spam + real ham
4. `run_reverse_classification(train, test, classifiers)` - Train on real, test on synthetic
5. `save_results(results, output_path)` - Save experiment results

**Command-Line Arguments**:
```bash
--method         # gpt41mini | claude35haiku | smote
--prompt         # original | strong | weak (N/A for SMOTE)
--strategy       # within_group | cross_group
--group_id       # 0-19 or None (run all groups)
--classifiers    # svm | random_forest | both (default: both)
--output_dir     # output/reverse_detection (default)
```

**Example Usage**:
```bash
# Run single group for GPT original within-group
python scripts/reverse_detection_experiment.py \
    --method gpt41mini \
    --prompt original \
    --strategy within_group \
    --group_id 0 \
    --classifiers both

# Run all groups for SMOTE
python scripts/reverse_detection_experiment.py \
    --method smote \
    --strategy within_group \
    --classifiers both
```

### Script 2: `run_all_reverse_detection.sh`

**Location**: `/Users/tianyu/Notebooks/cyberdata/scripts/run_all_reverse_detection.sh`

**Purpose**: Batch runner for all configurations

**Features**:
- Loops through all methods, prompts, strategies
- Runs all 20 groups for each configuration
- Progress tracking
- Error handling and logging

**Estimated Runtime**: 3-4 hours (560 experiments)

### Script 3: `analyze_reverse_detection.py`

**Location**: `/Users/tianyu/Notebooks/cyberdata/scripts/analyze_reverse_detection.py`

**Purpose**: Post-experiment analysis

**Outputs**:
1. **Summary Statistics**: Mean F1, std, by method/prompt/strategy
2. **Comparison Tables**: Forward vs Reverse detection performance
3. **Statistical Tests**: Paired t-tests comparing forward and reverse
4. **Visualizations**: Bar charts, comparison plots

---

## 📊 Expected Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    REVERSE DETECTION FLOW                    │
└─────────────────────────────────────────────────────────────┘

Step 1: Load Baseline Training Data
├── Source: output/prepared_data/groups/group_{i}/train.pkl
├── Content: 100 real spam + 900 real ham
└── Purpose: Train classifiers on pure real data

Step 2: Load Synthetic Test Spam (100%)
├── GPT/Claude: output/full_experiments/ceas08_{method}/datasets/
│               {prompt}_{strategy}_100/group_{i}_train.pkl
│               → Extract only spam samples (label==1)
│
└── SMOTE: output/full_experiments/ceas08_smote/datasets/
           smote_{strategy}_100/group_{i}_train.pkl
           → Extract only spam samples (label==1)

Step 3: Load Real Test Ham
├── Source: output/prepared_data/groups/group_{i}/test.pkl
├── Content: Real ham samples (label==0)
└── Purpose: Provide real ham for balanced test set

Step 4: Construct Reverse Test Set
├── Synthetic Spam: All 100% synthetic spam from Step 2
├── Real Ham: Real ham from Step 3
├── Spam Ratio: ~10% (to match forward experiment ratio)
└── Total Size: ~1000 samples (adjust to match original test set size)

Step 5: Train Classifiers
├── Training Data: Pure real data (Step 1)
├── Classifiers: SVM, Random Forest
└── Feature Extraction: TF-IDF (same config as forward experiment)

Step 6: Test on Reverse Set
├── Test Data: Synthetic spam + real ham (Step 4)
├── Metrics: F1, Precision, Recall, Accuracy, AUC-ROC, etc.
└── Comparison: Compare with forward experiment F1 (real→real)

Step 7: Statistical Analysis
├── Paired t-tests: Forward vs Reverse F1
├── Effect sizes: Cohen's d
├── Group-level variance analysis
└── Method comparison: GPT vs Claude vs SMOTE
```

---

## 🔍 Data Validation Checklist

Before running experiments, validate:

- [ ] **Baseline training data exists** for all 20 groups
- [ ] **Synthetic spam data (100%)** exists for:
  - [ ] GPT-4.1-mini: original/strong/weak × within/cross
  - [ ] Claude-3.5-Haiku: original/strong/weak × within/cross
  - [ ] SMOTE: within/cross
- [ ] **Real test ham** available in test.pkl files
- [ ] **Sample counts** match expectations:
  - Training: 1000 samples (100 spam + 900 ham)
  - Test synthetic spam: ~100-150 samples per group
  - Test real ham: ~900-1350 samples
- [ ] **Label distribution** correct (spam=1, ham=0)

---

## 📈 Analysis Plan

### Primary Metrics
- **F1-Score** (primary metric)
- Precision, Recall, Accuracy
- AUC-ROC, AUC-PR

### Comparisons

**1. Forward vs Reverse Detection**:
```
Forward: Real training → Real test
Reverse: Real training → Synthetic test

Question: Does detection F1 drop when testing on synthetic?
```

**2. Method Comparison**:
```
GPT vs Claude vs SMOTE

Question: Which synthetic method is most/least detectable?
```

**3. Prompt Sensitivity**:
```
Original vs Strong vs Weak (for LLMs)

Question: Do different prompts produce different detectability?
```

**4. Classifier Sensitivity**:
```
SVM vs Random Forest

Question: Which classifier is more robust to synthetic spam?
```

### Statistical Tests
- **Paired t-tests**: Forward F1 vs Reverse F1 for each group
- **Effect sizes**: Cohen's d to quantify magnitude
- **ANOVA**: Compare across methods (GPT/Claude/SMOTE)
- **Correlation**: Baseline performance vs detection performance

### Visualization Plan

**Plot 1**: Bar chart comparing Forward vs Reverse F1
- X-axis: Methods (GPT-Original, GPT-Strong, ..., SMOTE)
- Y-axis: F1-Score
- Two bars per method: Forward (blue), Reverse (red)

**Plot 2**: Heatmap of detection F1 by group
- Rows: Groups (0-19)
- Columns: Methods
- Color: F1-Score (green=high, red=low)

**Plot 3**: Scatter plot Forward vs Reverse
- X-axis: Forward F1 (real→real)
- Y-axis: Reverse F1 (real→synthetic)
- Points: Groups
- Diagonal line: y=x (perfect correspondence)

---

## ⚠️ Implementation Notes

### Key Design Decisions

**Decision 1: Test Set Composition**
- **Chosen**: 100% synthetic spam + real ham
- **Rationale**: Most realistic - attackers generate spam, not ham
- **Alternative**: 100% synthetic spam + 100% synthetic ham (more extreme, less practical)

**Decision 2: Spam Ratio in Test Set**
- **Chosen**: ~10% spam (match forward experiment)
- **Rationale**: Consistent evaluation conditions
- **Implementation**: Downsample ham if needed to match ratio

**Decision 3: Training Data**
- **Chosen**: Use existing baseline (0% synthetic) training data
- **Rationale**: No need to regenerate, already available
- **Benefit**: Faster implementation, consistent with forward experiment

**Decision 4: Synthetic Ratio**
- **Chosen**: Only test 100% synthetic (not 0%, 10%, ..., 90%)
- **Rationale**: Most extreme case, saves time (don't need 11 ratios)
- **Trade-off**: No sensitivity curve, but sufficient for main question

### Potential Issues & Solutions

**Issue 1**: Synthetic spam count mismatch across groups
- **Solution**: Normalize by taking min(available, target_count)

**Issue 2**: Test set size differs from forward experiment
- **Solution**: Match sizes by sampling, report sizes in results

**Issue 3**: Different spam distributions (real vs synthetic)
- **Solution**: Expected! This is the point of the experiment

**Issue 4**: Memory constraints with large datasets
- **Solution**: Process groups sequentially, clear memory between groups

---

## ✅ Pre-Execution Checklist

Before running `reverse_detection_experiment.py`:

### Data Verification
- [ ] Run data availability check script
- [ ] Verify sample counts in each group
- [ ] Check label distributions
- [ ] Confirm file formats (pkl files loadable)

### Configuration
- [ ] TF-IDF parameters match forward experiment
- [ ] Classifier hyperparameters match forward experiment
- [ ] Random seeds set for reproducibility
- [ ] Output directories created

### Code Review
- [ ] Script runs without errors on test group (group_id=0)
- [ ] Results file format correct
- [ ] Metrics calculation validated
- [ ] Error handling tested

### Resources
- [ ] Sufficient disk space for results (~500MB)
- [ ] Estimated runtime acceptable (3-4 hours)
- [ ] Compute resources available (CPU/GPU)

---

## 📝 Post-Execution Tasks

After experiments complete:

1. **Verify Results**:
   - [ ] All 560 experiments completed successfully
   - [ ] No missing result files
   - [ ] Metrics in expected ranges

2. **Run Analysis**:
   - [ ] Execute `analyze_reverse_detection.py`
   - [ ] Generate all plots
   - [ ] Compute statistical tests

3. **Paper Integration**:
   - [ ] Add new evaluation subsection (Section 5.X)
   - [ ] Update discussion with findings
   - [ ] Add plots to paper
   - [ ] Update conclusion if findings significant

---

## 🎯 Success Criteria

The experiment is successful if:

1. ✅ All 560 classification experiments complete
2. ✅ Results files saved in correct format
3. ✅ Statistical analysis produces interpretable findings
4. ✅ Comparison with forward experiment reveals clear patterns
5. ✅ At least 2 publication-quality plots generated

---

**Last Updated**: 2025-10-12
**Status**: Ready for human review → Execution → Analysis
