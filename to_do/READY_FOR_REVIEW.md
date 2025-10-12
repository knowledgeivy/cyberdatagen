# Reverse Detection Experiment - Ready for Review

**Status**: ✅ Code prepared, awaiting human review before execution
**Date**: 2025-10-12
**Estimated Runtime**: 3-4 hours for all 560 experiments

---

## 📁 Prepared Files

### 1. Implementation Documentation
- **File**: `to_do/reverse_detection_implementation.md`
- **Purpose**: Complete technical specification
- **Contents**:
  - Experimental design
  - Data flow diagrams
  - Input/output paths
  - Success criteria
  - Pre/post-execution checklists

### 2. Main Experiment Script
- **File**: `scripts/reverse_detection_experiment.py`
- **Language**: Python 3
- **Permissions**: ✅ Executable (`chmod +x`)
- **Key Features**:
  - Load 0% synthetic baseline training data
  - Load 100% synthetic spam for testing
  - Construct reverse test sets (synthetic spam + real ham)
  - Train classifiers (SVM, Random Forest) on real data
  - Evaluate on synthetic data
  - Save results in JSON format

### 3. Batch Runner Script
- **File**: `scripts/run_all_reverse_detection.sh`
- **Language**: Bash
- **Permissions**: ✅ Executable (`chmod +x`)
- **Coverage**:
  - 14 configurations (GPT×6, Claude×6, SMOTE×2)
  - 20 groups per configuration
  - 2 classifiers per group
  - **Total**: 560 experiments

### 4. Data Verification Script
- **File**: `scripts/verify_reverse_detection_data.py`
- **Language**: Python 3
- **Permissions**: ✅ Executable (`chmod +x`)
- **Purpose**: Pre-flight check
- **Checks**:
  - Baseline training data (20 groups)
  - GPT synthetic data (6 configurations × 20 groups)
  - Claude synthetic data (6 configurations × 20 groups)
  - SMOTE data (2 configurations × 20 groups)
  - Data integrity and sample counts

---

## 🎯 Execution Workflow

### Step 1: Data Verification (5 minutes)
```bash
cd /Users/tianyu/Notebooks/cyberdata
python scripts/verify_reverse_detection_data.py
```

**Expected Output**:
```
✓ Baseline data: PASS
✓ GPT-4.1-mini data: PASS (6 configurations)
✓ Claude-3.5-Haiku data: PASS (6 configurations)
✓ SMOTE data: PASS (2 configurations)

✓ All data checks PASSED!
Ready to run reverse detection experiments
```

**If any FAIL**: Fix data issues before proceeding

---

### Step 2A: Test Run (10 minutes) - RECOMMENDED
Test on a single group before full run:

```bash
# Test GPT original within-group, group 0
python scripts/reverse_detection_experiment.py \
    --method gpt41mini \
    --prompt original \
    --strategy within_group \
    --group_id 0 \
    --classifiers svm random_forest \
    --config configs/ceas08_gpt41mini_config.yaml \
    --output_dir output/reverse_detection
```

**What to check**:
- Script runs without errors
- Results file created: `output/reverse_detection/results/gpt41mini_original_within_group_reverse_results.json`
- JSON format correct
- Metrics in expected range (F1: 0.0-1.0)

---

### Step 2B: Full Run (3-4 hours)
Run all 560 experiments:

```bash
bash scripts/run_all_reverse_detection.sh
```

**Progress Tracking**:
- Script will print: `[X/14] Method | Prompt | Strategy`
- Each configuration processes all 20 groups automatically
- Errors will cause script to exit (set -e)
- Check logs: `output/reverse_detection/logs/reverse_detection.log`

---

## 📊 Expected Outputs

### Result Files
```
output/reverse_detection/
├── results/
│   ├── gpt41mini_original_within_group_reverse_results.json
│   ├── gpt41mini_original_cross_group_reverse_results.json
│   ├── gpt41mini_strong_within_group_reverse_results.json
│   ├── gpt41mini_strong_cross_group_reverse_results.json
│   ├── gpt41mini_weak_within_group_reverse_results.json
│   ├── gpt41mini_weak_cross_group_reverse_results.json
│   ├── claude35haiku_original_within_group_reverse_results.json
│   ├── claude35haiku_original_cross_group_reverse_results.json
│   ├── claude35haiku_strong_within_group_reverse_results.json
│   ├── claude35haiku_strong_cross_group_reverse_results.json
│   ├── claude35haiku_weak_within_group_reverse_results.json
│   ├── claude35haiku_weak_cross_group_reverse_results.json
│   ├── smote_within_group_reverse_results.json
│   └── smote_cross_group_reverse_results.json  (14 files total)
│
└── logs/
    └── reverse_detection.log
```

### Result File Format
```json
{
  "experiment": "reverse_detection",
  "method": "gpt41mini",
  "prompt": "original",
  "strategy": "within_group",
  "n_groups": 20,
  "classifiers": ["svm", "random_forest"],
  "results": [
    {
      "method": "gpt41mini",
      "prompt": "original",
      "strategy": "within_group",
      "group_id": 0,
      "success": true,
      "classifiers": {
        "svm": {
          "classifier": "svm",
          "metrics": {
            "f1_score": 0.75,
            "precision": 0.80,
            "recall": 0.70,
            "accuracy": 0.92,
            "auc_roc": 0.85,
            ...
          },
          "train_size": 1000,
          "test_size": 1000,
          "test_spam_count": 100,
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

## ✅ Pre-Execution Checklist

Before running experiments, verify:

### Environment
- [ ] Python environment activated (if using virtualenv)
- [ ] Required packages installed: pandas, numpy, sklearn, loguru, torch
- [ ] Sufficient disk space (~500MB for results)
- [ ] Sufficient compute resources (CPU/GPU)

### Data
- [ ] Run `verify_reverse_detection_data.py` → All checks PASS
- [ ] Baseline data exists for all 20 groups
- [ ] Synthetic data exists for all required configurations
- [ ] Config files exist: `configs/ceas08_*.yaml`

### Scripts
- [ ] All scripts executable: `ls -l scripts/reverse_*.py scripts/verify_*.py`
- [ ] Test run successful (Step 2A)

### Output
- [ ] Output directory writable: `output/reverse_detection/`
- [ ] No conflicting processes writing to same directory

---

## 🔍 Post-Execution Verification

After experiments complete:

### 1. Check Completeness
```bash
# Count result files (should be 14)
ls output/reverse_detection/results/*.json | wc -l

# Check all files have content (not empty)
find output/reverse_detection/results -name "*.json" -size 0
```

### 2. Validate Results
Quick validation script:
```python
import json
from pathlib import Path

results_dir = Path('output/reverse_detection/results')

for result_file in results_dir.glob('*.json'):
    with open(result_file) as f:
        data = json.load(f)

    n_groups = len(data['results'])
    successful = sum(1 for r in data['results'] if r['success'])

    print(f"{result_file.name}:")
    print(f"  Groups: {n_groups}/20")
    print(f"  Successful: {successful}/{n_groups}")

    if successful < n_groups:
        print(f"  ⚠️  {n_groups - successful} failed!")
```

### 3. Quick Stats
```bash
# Extract F1 scores from one result file
python -c "
import json
with open('output/reverse_detection/results/gpt41mini_original_within_group_reverse_results.json') as f:
    data = json.load(f)

svm_f1s = [r['classifiers']['svm']['metrics']['f1_score']
           for r in data['results'] if r['success']]
print(f'SVM F1: {sum(svm_f1s)/len(svm_f1s):.4f} ± {(sum((x-sum(svm_f1s)/len(svm_f1s))**2 for x in svm_f1s)/len(svm_f1s))**0.5:.4f}')
"
```

---

## 🚨 Troubleshooting

### Issue: "FileNotFoundError: Baseline training data not found"
**Cause**: Missing baseline training data for some groups
**Fix**: Re-run data preparation step (step1_data_preprocessing.py)

### Issue: "FileNotFoundError: Synthetic dataset not found"
**Cause**: Missing 100% synthetic data for some configurations
**Fix**: Check if forward experiments completed. May need to re-run specific configurations.

### Issue: "MemoryError" during classification
**Cause**: Insufficient RAM for feature extraction
**Fix**: Process fewer groups at once or increase swap space

### Issue: Script hangs/freezes
**Cause**: May be waiting on GPU or long computation
**Fix**: Check process with `top` or `htop`, allow time to complete

### Issue: JSON decode error in results
**Cause**: Results file corrupted or incomplete write
**Fix**: Delete partial file and re-run that configuration

---

## 📈 Next Steps After Execution

Once all experiments complete successfully:

1. **Run Analysis** (to be implemented):
   ```bash
   python scripts/analyze_reverse_detection.py
   ```

2. **Generate Plots** (to be implemented):
   - Forward vs Reverse F1 comparison
   - Method comparison (GPT vs Claude vs SMOTE)
   - Heatmap by group

3. **Paper Integration**:
   - Add Section 5.X "Reverse Detection: Real Data vs Synthetic Spam"
   - Update Discussion with findings
   - Add 2-3 plots to evaluation section
   - Update conclusion if findings significant

---

## 🎓 Key Experimental Questions

This experiment will answer:

1. **Can real-trained classifiers detect synthetic spam?**
   - High F1 → Yes, synthetic spam is detectable
   - Low F1 → No, synthetic spam evades detection

2. **Which method produces most detectable synthetic spam?**
   - Compare F1 across GPT/Claude/SMOTE

3. **Does prompt strategy affect detectability?**
   - Compare original vs strong vs weak for each LLM

4. **Which classifier is more robust?**
   - Compare SVM vs Random Forest detection rates

---

## 📞 Contact & Support

If issues arise during execution:
- Check logs: `output/reverse_detection/logs/reverse_detection.log`
- Review implementation doc: `to_do/reverse_detection_implementation.md`
- Verify data with: `scripts/verify_reverse_detection_data.py`

---

**Last Updated**: 2025-10-12
**Status**: ✅ Ready for human review
**Action Required**: Review code → Run Step 1 (verify data) → Run Step 2A (test) → Run Step 2B (full run)
