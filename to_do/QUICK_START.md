# Reverse Detection Experiment - Quick Start Guide

**⚠️  IMPORTANT**: Do NOT run experiments yet! Review code first!

---

## 📁 All Files Created

```
to_do/
├── final_10_days_plan.md              # Master plan (10-day timeline)
├── reverse_detection_implementation.md # Technical specification
├── READY_FOR_REVIEW.md                # Detailed execution guide
└── QUICK_START.md                     # This file

scripts/
├── reverse_detection_experiment.py    # Main experiment script ✅ Executable
├── run_all_reverse_detection.sh       # Batch runner ✅ Executable
└── verify_reverse_detection_data.py   # Data validation ✅ Executable
```

---

## 🚀 Quick Commands

### 1. Verify Data (ALWAYS RUN FIRST)
```bash
cd /Users/tianyu/Notebooks/cyberdata
python scripts/verify_reverse_detection_data.py
```
**Expected**: All checks PASS ✓

---

### 2. Test Single Group (RECOMMENDED BEFORE FULL RUN)
```bash
python scripts/reverse_detection_experiment.py \
    --method gpt41mini \
    --prompt original \
    --strategy within_group \
    --group_id 0
```
**Runtime**: ~2 minutes
**Check**: `output/reverse_detection/results/gpt41mini_original_within_group_reverse_results.json`

---

### 3. Run All Experiments
```bash
bash scripts/run_all_reverse_detection.sh
```
**Runtime**: 3-4 hours
**Total**: 560 experiments (14 configs × 20 groups × 2 classifiers)

---

## 📊 What Gets Generated

### Input Data (Already Exists)
- Baseline training: `output/prepared_data/groups/group_*/train.pkl` (20 groups)
- Synthetic spam 100%: `output/full_experiments/ceas08_*/datasets/` (14 configs × 20 groups)
- Real ham: `output/prepared_data/groups/group_*/test.pkl` (20 groups)

### Output Data (Will Be Created)
```
output/reverse_detection/
├── results/           # 14 JSON files (one per configuration)
│   ├── gpt41mini_original_within_group_reverse_results.json
│   ├── gpt41mini_original_cross_group_reverse_results.json
│   ├── ... (12 more files)
│   └── smote_cross_group_reverse_results.json
│
└── logs/
    └── reverse_detection.log  # Execution log
```

---

## 🔍 Key Experiment Details

### Training Set
- **Composition**: 0% synthetic (pure real data)
- **Size**: 1,000 samples (100 spam + 900 ham)
- **Source**: Baseline training data from forward experiments

### Testing Set
- **Composition**: 100% synthetic spam + real ham
- **Spam Sources**:
  - GPT-4.1-mini (original/strong/weak × within/cross) = 6 configs
  - Claude-3.5-Haiku (original/strong/weak × within/cross) = 6 configs
  - SMOTE (within/cross) = 2 configs
- **Ham Source**: Real ham from test sets
- **Spam Ratio**: ~10% (matches forward experiments)

### Experiment Matrix
```
14 configurations:
├── GPT-4.1-mini: 3 prompts × 2 strategies = 6
├── Claude-3.5-Haiku: 3 prompts × 2 strategies = 6
└── SMOTE: 1 method × 2 strategies = 2

Per configuration:
├── 20 groups (same as forward experiment)
└── 2 classifiers (SVM, Random Forest)

Total: 14 × 20 × 2 = 560 experiments
```

---

## ✅ Pre-Flight Checklist

Before running experiments:

- [ ] Code reviewed by human (YOU!)
- [ ] Data verification passed: `python scripts/verify_reverse_detection_data.py`
- [ ] Test run successful: Single group test completed
- [ ] Sufficient disk space: ~500MB free
- [ ] Sufficient time: 3-4 hours available
- [ ] Config files exist: `configs/ceas08_*.yaml`

---

## 📈 Expected Results

### Scenarios

**Scenario A: Synthetic spam is detectable**
- F1 > 0.7: Classifiers successfully detect synthetic spam
- Interpretation: LLMs produce realistic spam

**Scenario B: Synthetic spam evades detection**
- F1 < 0.5: Classifiers fail to identify synthetic spam
- Interpretation: LLM-generated spam differs qualitatively from real spam

**Scenario C: Method-dependent**
- GPT: High F1
- Claude: Medium F1
- SMOTE: Low F1
- Interpretation: Different methods produce different detectability

---

## 🎯 Research Questions

This experiment answers:

1. Can real-data-trained classifiers detect synthetic spam?
2. Which method (GPT/Claude/SMOTE) is most/least detectable?
3. Does prompt strategy affect detectability?
4. Which classifier (SVM/RF) is more robust to synthetic data?

---

## 📝 Next Steps After Execution

1. Verify all 14 result files created
2. Check for errors in logs
3. Run analysis script (to be implemented)
4. Generate comparison plots
5. Integrate findings into paper Section 5.X

---

## 🚨 Emergency Stop

If you need to stop experiments:
```bash
# Press Ctrl+C in terminal running the script

# Or find and kill process:
ps aux | grep reverse_detection
kill -9 <PID>
```

Results are saved after each group, so partial progress is preserved.

---

## 📚 More Information

- **Full specification**: `to_do/reverse_detection_implementation.md`
- **Detailed execution guide**: `to_do/READY_FOR_REVIEW.md`
- **Master plan**: `to_do/final_10_days_plan.md`

---

**Status**: ✅ Ready for review
**Created**: 2025-10-12
**Action Required**: **REVIEW CODE FIRST**, then run experiments
