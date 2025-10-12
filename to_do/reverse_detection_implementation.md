# Reverse Detection Experiment - Cross-Model Implementation

**Version**: 3.0 (Cross-Model 1-vs-Rest Design)
**Date**: 2025-10-12
**Status**: Ready for code review before execution

---

## Executive Summary

### Design: Cross-Model Detection (1-vs-Rest)

We implement **cross-model reverse detection** with **three training synthetic ratios (0%, 50%, 100%)** to answer:

**"Can classifiers trained on one LLM's synthetic data detect another LLM's synthetic spam?"**

### Key Design Principles

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Models** | GPT-4.1-mini ↔ Claude-3.5-Haiku only | Exclude SMOTE (no text-space comparability) |
| **Training-Testing Pairing** | 1-vs-rest (train on Claude → test on GPT, vice versa) | Avoid overfitting, test cross-model generalization |
| **Prompt Matching** | Same prompt type (strong vs strong, weak vs weak) | Control for prompt complexity |
| **Strategy Matching** | Same strategy (within vs within, cross vs cross) | Control for template distribution |
| **Training Ratios** | 0%, 50%, 100% | Test exposure effect with different models |
| **Total Experiments** | 1,440 | 12 configs × 3 ratios × 20 groups × 2 classifiers |
| **Runtime** | 8-10 hours | Can run overnight |

---

## Research Questions

This experiment answers:

1. **Cross-Model Detection**: Can classifiers trained on Claude synthetic detect GPT synthetic spam? (and vice versa)
2. **Cross-Model Exposure Effect**: Does exposure to one LLM's synthetic data improve detection of another LLM's synthetic spam?
3. **Model Detectability Comparison**: Which LLM (GPT vs Claude) generates more detectable spam when trained on the other?
4. **Prompt Impact on Detectability**: Do strong/weak/original prompts affect cross-model detectability?
5. **Strategy Impact**: Does within-group vs cross-group affect cross-model detection?

---

## Experimental Design

### Core Principle: Train on Model A, Test on Model B

**For GPT Testing**:
- Training synthetic: Claude (same prompt, same strategy)
- Testing synthetic: GPT

**For Claude Testing**:
- Training synthetic: GPT (same prompt, same strategy)
- Testing synthetic: Claude

**No overlap between training and testing synthetic sources!**

---

## Experiment Matrix

### Total: 12 Configurations (LLM only, no SMOTE)

```
Testing Models: 2 (GPT, Claude)
├── GPT-4.1-mini testing: 3 prompts × 2 strategies = 6 configs
└── Claude-3.5-Haiku testing: 3 prompts × 2 strategies = 6 configs

Per Configuration:
├── 3 training ratios: 0%, 50%, 100%
├── 20 groups: Independent replications
└── 2 classifiers: SVM, Random Forest

Total: 12 × 3 × 20 × 2 = 1,440 experiments
```

### Configuration Mapping Table

| Testing Target | Testing Prompt | Testing Strategy | Training Synthetic Source | Training Prompt | Training Strategy |
|----------------|----------------|------------------|---------------------------|-----------------|-------------------|
| **GPT-4.1-mini** | original | within_group | Claude-3.5-Haiku | original | within_group |
| **GPT-4.1-mini** | original | cross_group | Claude-3.5-Haiku | original | cross_group |
| **GPT-4.1-mini** | strong | within_group | Claude-3.5-Haiku | strong | within_group |
| **GPT-4.1-mini** | strong | cross_group | Claude-3.5-Haiku | strong | cross_group |
| **GPT-4.1-mini** | weak | within_group | Claude-3.5-Haiku | weak | within_group |
| **GPT-4.1-mini** | weak | cross_group | Claude-3.5-Haiku | weak | cross_group |
| **Claude-3.5-Haiku** | original | within_group | GPT-4.1-mini | original | within_group |
| **Claude-3.5-Haiku** | original | cross_group | GPT-4.1-mini | original | cross_group |
| **Claude-3.5-Haiku** | strong | within_group | GPT-4.1-mini | strong | within_group |
| **Claude-3.5-Haiku** | strong | cross_group | GPT-4.1-mini | strong | cross_group |
| **Claude-3.5-Haiku** | weak | within_group | GPT-4.1-mini | weak | within_group |
| **Claude-3.5-Haiku** | weak | cross_group | GPT-4.1-mini | weak | cross_group |

**Key**: Training and testing always use **different models** but **same prompt and strategy**

---

## Data Flow

### Example: Testing GPT-strong-within (Group 0)

#### Step 1: Load Baseline Data
```python
baseline_train = load('output/prepared_data/groups/group_0/train.pkl')
baseline_test = load('output/prepared_data/groups/group_0/test.pkl')

real_spam = baseline_train[baseline_train['label'] == 1]  # 100 samples
real_ham_train = baseline_train[baseline_train['label'] == 0]  # 900 samples
real_ham_test = baseline_test[baseline_test['label'] == 0]  # 900 samples
```

#### Step 2: Load Testing Synthetic (GPT)
```python
gpt_dataset = load('output/full_experiments/ceas08_gpt41mini/datasets/strong_within_group_100/group_0_train.pkl')
testing_synthetic_spam = gpt_dataset[gpt_dataset['label'] == 1]  # 100 GPT spam
```

#### Step 3: Load Training Synthetic (Claude, different model!)
```python
claude_dataset = load('output/full_experiments/ceas08_claude35haiku/datasets/strong_within_group_100/group_0_train.pkl')
training_synthetic_spam = claude_dataset[claude_dataset['label'] == 1]  # 100 Claude spam
```

#### Step 4: Construct Training Sets (3 Ratios)

**Ratio 0%: Pure Real**
```python
train_0 = baseline_train  # 100 real spam + 900 real ham
```

**Ratio 50%: 50% Real + 50% Claude Synthetic**
```python
train_50 = pd.concat([
    real_spam.sample(50, random_state=42),  # 50 real spam
    training_synthetic_spam.sample(50, random_state=42),  # 50 Claude spam
    real_ham_train  # 900 real ham
])  # Total: 1000 samples
```

**Ratio 100%: 100% Claude Synthetic**
```python
train_100 = pd.concat([
    training_synthetic_spam,  # 100 Claude spam (all)
    real_ham_train  # 900 real ham
])  # Total: 1000 samples
```

#### Step 5: Construct Testing Set (100% GPT Synthetic)
```python
test = pd.concat([
    testing_synthetic_spam,  # 100 GPT spam (all)
    real_ham_test.sample(900, random_state=42)  # 900 real ham
])  # Total: 1000 samples
```

#### Key Point
**Training uses Claude synthetic, Testing uses GPT synthetic → No data leakage!**

---

## Training Set Composition by Ratio

### Ratio 0%: Baseline (No Synthetic Exposure)
```
100 real spam + 900 real ham = 1000 samples
Synthetic ratio: 0%
```

**Purpose**: Baseline performance without any synthetic exposure

---

### Ratio 50%: Mixed Exposure
```
50 real spam + 50 synthetic spam (from opposite model) + 900 real ham = 1000 samples
Synthetic ratio: 50% (of spam portion)
```

**Purpose**: Test partial cross-model exposure effect

---

### Ratio 100%: Full Synthetic Exposure
```
100 synthetic spam (from opposite model) + 900 real ham = 1000 samples
Synthetic ratio: 100% (of spam portion)
```

**Purpose**: Test whether training entirely on one model helps detect the other

---

## Testing Set Composition (Fixed)

**Always 100% synthetic spam from target model**

```
100 synthetic spam (from testing target model) + 900 real ham = 1000 samples
Spam ratio: ~10%
```

**Example**:
- Testing GPT → 100 GPT synthetic spam
- Testing Claude → 100 Claude synthetic spam

---

## Data Requirements

### Baseline Data (All Groups)
```
output/prepared_data/groups/
├── group_0/
│   ├── train.pkl  (1,000 samples: 100 real spam + 900 real ham)
│   └── test.pkl   (1,000 samples: 100 real spam + 900 real ham)
├── ...
└── group_19/
    ├── train.pkl
    └── test.pkl

Total: 40 files
```

---

### GPT-4.1-mini Datasets (6 Configurations)
```
output/full_experiments/ceas08_gpt41mini/datasets/
├── original_within_group_100/group_{0-19}_train.pkl
├── original_cross_group_100/group_{0-19}_train.pkl
├── strong_within_group_100/group_{0-19}_train.pkl
├── strong_cross_group_100/group_{0-19}_train.pkl
├── weak_within_group_100/group_{0-19}_train.pkl
└── weak_cross_group_100/group_{0-19}_train.pkl

Total: 6 configs × 20 groups = 120 files
```

---

### Claude-3.5-Haiku Datasets (6 Configurations)
```
output/full_experiments/ceas08_claude35haiku/datasets/
├── original_within_group_100/group_{0-19}_train.pkl
├── original_cross_group_100/group_{0-19}_train.pkl
├── strong_within_group_100/group_{0-19}_train.pkl
├── strong_cross_group_100/group_{0-19}_train.pkl
├── weak_within_group_100/group_{0-19}_train.pkl
└── weak_cross_group_100/group_{0-19}_train.pkl

Total: 6 configs × 20 groups = 120 files
```

---

### Total Required Files
```
Baseline: 40 files (20 groups × 2 files)
GPT datasets: 120 files (6 configs × 20 groups)
Claude datasets: 120 files (6 configs × 20 groups)

Total: 280 files
```

---

## Output Structure

### Results Directory
```
output/reverse_detection/
├── results/
│   # GPT testing (trained on Claude)
│   ├── gpt41mini_original_within_ratio0_reverse_results.json
│   ├── gpt41mini_original_within_ratio50_reverse_results.json
│   ├── gpt41mini_original_within_ratio100_reverse_results.json
│   ├── gpt41mini_original_cross_ratio0_reverse_results.json
│   ├── gpt41mini_original_cross_ratio50_reverse_results.json
│   ├── gpt41mini_original_cross_ratio100_reverse_results.json
│   ├── gpt41mini_strong_within_ratio0_reverse_results.json
│   ├── gpt41mini_strong_within_ratio50_reverse_results.json
│   ├── gpt41mini_strong_within_ratio100_reverse_results.json
│   ├── gpt41mini_strong_cross_ratio0_reverse_results.json
│   ├── gpt41mini_strong_cross_ratio50_reverse_results.json
│   ├── gpt41mini_strong_cross_ratio100_reverse_results.json
│   ├── gpt41mini_weak_within_ratio0_reverse_results.json
│   ├── gpt41mini_weak_within_ratio50_reverse_results.json
│   ├── gpt41mini_weak_within_ratio100_reverse_results.json
│   ├── gpt41mini_weak_cross_ratio0_reverse_results.json
│   ├── gpt41mini_weak_cross_ratio50_reverse_results.json
│   ├── gpt41mini_weak_cross_ratio100_reverse_results.json
│
│   # Claude testing (trained on GPT)
│   ├── claude35haiku_original_within_ratio0_reverse_results.json
│   ├── claude35haiku_original_within_ratio50_reverse_results.json
│   ├── claude35haiku_original_within_ratio100_reverse_results.json
│   ├── ... (15 more Claude files)
│
│   Total: 36 JSON files (12 configs × 3 ratios)
│
└── logs/
    └── reverse_detection.log
```

---

### Result File Format

```json
{
  "experiment": "reverse_detection_cross_model",
  "testing_method": "gpt41mini",
  "testing_prompt": "strong",
  "testing_strategy": "within_group",
  "training_synthetic_source": "claude35haiku",
  "training_synthetic_prompt": "strong",
  "training_synthetic_strategy": "within_group",
  "training_synthetic_ratio": 50,
  "testing_synthetic_ratio": 100,
  "n_groups": 20,
  "classifiers": ["svm", "random_forest"],
  "results": [
    {
      "group_id": 0,
      "training_ratio": 50,
      "success": true,
      "training_source": "claude35haiku-strong-within_group",
      "testing_target": "gpt41mini-strong-within_group",
      "classifiers": {
        "svm": {
          "classifier": "svm",
          "metrics": {
            "f1_score": 0.65,
            "precision": 0.70,
            "recall": 0.60,
            "accuracy": 0.90,
            "auc_roc": 0.82
          },
          "train_size": 1000,
          "train_real_spam_count": 50,
          "train_synthetic_spam_count": 50,
          "train_synthetic_source": "claude35haiku",
          "train_ham_count": 900,
          "test_size": 1000,
          "test_synthetic_spam_count": 100,
          "test_synthetic_source": "gpt41mini",
          "test_ham_count": 900,
          "test_spam_ratio": 0.10
        },
        "random_forest": { ... }
      }
    },
    ... (19 more groups)
  ]
}
```

---

## Expected Results and Hypotheses

### Hypothesis 1: Cross-Model Exposure Helps

```
Training Ratio 0%   → F1 = 0.55  (no synthetic exposure)
Training Ratio 50%  → F1 = 0.62  (partial exposure to different model)
Training Ratio 100% → F1 = 0.70  (full exposure to different model)
```

**Interpretation**: Even though trained on different model's synthetic, exposure still helps. LLMs share common synthetic patterns.

**Paper Impact**: Strong finding - synthetic data has transferable features across models

---

### Hypothesis 2: Cross-Model Exposure Doesn't Help

```
Training Ratio 0%   → F1 = 0.65  (pure real data generalizes well)
Training Ratio 50%  → F1 = 0.63  (no benefit)
Training Ratio 100% → F1 = 0.58  (worse - overfits to wrong model)
```

**Interpretation**: GPT and Claude generate qualitatively different synthetic spam. Training on one doesn't help detect the other.

**Paper Impact**: Models have distinct synthetic signatures

---

### Hypothesis 3: Asymmetric Detectability

```
Train on Claude → Test on GPT: F1 = 0.70
Train on GPT → Test on Claude: F1 = 0.55
```

**Interpretation**: GPT spam is easier to detect (more stereotypical patterns), while Claude spam is stealthier.

**Paper Impact**: Model-specific deployment recommendations

---

### Hypothesis 4: Prompt Complexity Matters

```
Strong prompt:
  Train(Claude-strong) → Test(GPT-strong): F1 = 0.72

Weak prompt:
  Train(Claude-weak) → Test(GPT-weak): F1 = 0.62
```

**Interpretation**: Strong prompts produce more complex, harder-to-detect spam even across models.

---

## Pre-Execution Checklist

### Environment
- [ ] Python environment activated
- [ ] Required packages: pandas, numpy, sklearn, loguru, torch
- [ ] Disk space: ~700MB for results
- [ ] Compute: 8-10 hours available (overnight run)

### Data Availability
- [ ] Baseline: 40 files verified
- [ ] GPT datasets: 120 files verified (6 configs × 20 groups)
- [ ] Claude datasets: 120 files verified (6 configs × 20 groups)
- [ ] **Total**: 280 files

### Scripts
- [ ] `reverse_detection_experiment.py` updated for cross-model
- [ ] `run_all_reverse_detection.sh` updated with correct pairing
- [ ] `verify_reverse_detection_data.py` checks all required files
- [ ] All scripts executable

### Configuration
- [ ] `configs/ceas08_gpt41mini_config.yaml` exists
- [ ] `configs/ceas08_claude35haiku_config.yaml` exists

---

## Execution Workflow

### Step 1: Verify Data (MANDATORY)
```bash
cd /Users/tianyu/Notebooks/cyberdata
python scripts/verify_reverse_detection_data.py
```

**Expected**: All checks PASS for baseline + GPT + Claude

---

### Step 2: Test Run (RECOMMENDED)
```bash
# Test single configuration: train on Claude, test on GPT
python scripts/reverse_detection_experiment.py \
    --testing_method gpt41mini \
    --testing_prompt original \
    --testing_strategy within_group \
    --training_ratio 50 \
    --group_id 0 \
    --classifiers svm random_forest \
    --config configs/ceas08_gpt41mini_config.yaml \
    --output_dir output/reverse_detection
```

**Runtime**: ~3 minutes
**Expected**: Result file created with cross-model metadata

---

### Step 3: Full Run
```bash
bash scripts/run_all_reverse_detection.sh
```

**Runtime**: 8-10 hours
**Total**: 1,440 experiments
**Output**: 36 JSON files

---

## Post-Execution Verification

### 1. Check File Count
```bash
ls output/reverse_detection/results/*.json | wc -l
# Expected: 36
```

### 2. Validate Cross-Model Pairing
```python
import json
from pathlib import Path

results_dir = Path('output/reverse_detection/results')

for file in results_dir.glob('*.json'):
    with open(file) as f:
        data = json.load(f)

    testing = data['testing_method']
    training = data['training_synthetic_source']

    # Verify they are different
    assert testing != training, f"Error: {file.name} has same model for train and test!"

    print(f"✓ {file.name}: Train({training}) → Test({testing})")
```

### 3. Summary Statistics
```python
import json
import numpy as np
from pathlib import Path

results_dir = Path('output/reverse_detection/results')

# GPT testing (trained on Claude)
print("GPT Testing (trained on Claude):")
for ratio in [0, 50, 100]:
    files = list(results_dir.glob(f'gpt41mini_*_ratio{ratio}_*.json'))

    f1_scores = []
    for file in files:
        with open(file) as f:
            data = json.load(f)
        for result in data['results']:
            if result['success']:
                f1_scores.append(result['classifiers']['svm']['metrics']['f1_score'])

    print(f"  Ratio {ratio}%: F1 = {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

print("\nClaude Testing (trained on GPT):")
for ratio in [0, 50, 100]:
    files = list(results_dir.glob(f'claude35haiku_*_ratio{ratio}_*.json'))

    f1_scores = []
    for file in files:
        with open(file) as f:
            data = json.load(f)
        for result in data['results']:
            if result['success']:
                f1_scores.append(result['classifiers']['svm']['metrics']['f1_score'])

    print(f"  Ratio {ratio}%: F1 = {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
```

---

## Analysis Plan

### Primary Comparisons

**1. Cross-Model Exposure Effect**
```
For each testing target (12 configs):
  Compare F1 across ratios 0% → 50% → 100%
  Test: Does exposure to different model help?
```

**2. Model Detectability Asymmetry**
```
GPT detectability: Train(Claude) → Test(GPT)
Claude detectability: Train(GPT) → Test(Claude)

Compare: Which model is easier to detect?
```

**3. Prompt Impact**
```
Compare: original vs strong vs weak (within same model pair)
Test: Do prompts affect cross-model detectability?
```

**4. Strategy Impact**
```
Compare: within-group vs cross-group
Test: Does template distribution matter for cross-model detection?
```

---

## Paper Integration

### New Section: "5.X Cross-Model Reverse Detection"

**Subsections**:
1. **Motivation** (1 paragraph): Why cross-model detection matters
2. **Experimental Design** (1 paragraph): 1-vs-rest, three ratios
3. **Cross-Model Exposure Effect** (1-2 paragraphs): Ratio 0% → 50% → 100% results
4. **Model Detectability Asymmetry** (1 paragraph): GPT vs Claude
5. **Prompt and Strategy Effects** (1 paragraph): Interaction analysis

**Figures**:
- **Figure A**: Line plot - F1 vs training ratio (separate lines for GPT/Claude testing)
- **Figure B**: Heatmap - Detectability matrix (prompt × strategy × testing model)
- **Figure C**: Bar chart - Asymmetry comparison (Train-Claude/Test-GPT vs Train-GPT/Test-Claude)

**Estimated Length**: 2-2.5 pages

---

## Discussion Integration

**New Subsection**: "Cross-Model Generalization of Synthetic Data"

**Content**:
- If exposure helps: LLMs share common synthetic patterns → transferable defense
- If asymmetric: Model-specific deployment strategies
- If prompt matters: Prompt complexity affects transferability
- Implications for multi-LLM threat landscapes

**Estimated Length**: 0.5-1 page

---

## Timeline

| Task | Duration |
|------|----------|
| Code review | 30 min |
| Data verification | 5 min |
| Test run | 3 min |
| Full run | 8-10 hours (overnight) |
| Analysis | 2-3 hours |
| Plotting | 1-2 hours |
| Writing | 3-4 hours |
| **Total** | ~18 hours (with overnight run) |

---

## Success Criteria

✅ All 36 result files generated
✅ Cross-model pairing verified (training ≠ testing model)
✅ Clear trends in F1 vs training ratio
✅ Statistically significant exposure effect (if present)
✅ Interpretable asymmetry findings

---

## Key Advantages of This Design

1. ✅ **No overfitting**: Training and testing use different model sources
2. ✅ **No data leakage**: Complete separation of synthetic sources
3. ✅ **Practical relevance**: Mirrors real multi-LLM threat landscape
4. ✅ **Novel contribution**: First study of cross-model synthetic spam detection
5. ✅ **Controlled comparison**: Matching prompts and strategies isolate model effect
6. ✅ **Feasible scope**: 1,440 experiments in 8-10 hours

---

**Status**: Implementation specification complete
**Next Step**: Update scripts for cross-model pairing
**Timeline**: Code update (2 hours) → Review → Execution (overnight) → Analysis (1 day)
