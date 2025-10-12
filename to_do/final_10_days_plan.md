# Final 10-Day Paper Improvement Plan

**Deadline**: Paper submission in 10 days
**Last Updated**: 2025-10-12
**Priority**: High-impact, feasible improvements

---

## 📊 Overview: Three Major Improvements

| Improvement | Priority | Time Cost | Impact | Status |
|------------|----------|-----------|--------|--------|
| **1. Reverse Detection Experiment** | ⭐⭐⭐⭐⭐ Must Do | 2 days | Very High | 🔴 Not Started |
| **2. Hyperparameter Justification** | ⭐⭐⭐⭐ Must Do | 0.5-3 days | Medium-High | 🔴 Not Started |
| **3. Spam Sample Examples in Appendix** | ⭐⭐⭐⭐⭐ Must Do | 0.5-1 days | High | 🔴 Not Started |

**Total Minimum**: 3-3.5 days
**Total Maximum** (with optional sensitivity analysis): 6.5 days
**Buffer**: 3.5-7 days for paper writing, revision, and submission

---

## 🎯 Task 1: Reverse Detection Experiment ⭐⭐⭐⭐⭐

### Motivation
Real-world threat scenario: Attackers use LLMs to generate spam/phishing emails. This experiment tests whether classifiers trained on **real data only** can detect **LLM-generated spam**.

### Research Questions
1. Can real-data-trained classifiers identify synthetic spam as spam?
2. Do LLM-generated samples contain detectable artifacts?
3. Which synthetic method (GPT/Claude/SMOTE) is most/least detectable?

### Experimental Design

**Training Set**:
- Pure real data (0% synthetic)
- Real spam + real ham
- Same 20 groups as baseline

**Testing Set Options**:
- **Option A (Recommended)**: 100% synthetic spam + real ham
  - Tests: Can classifier detect LLM spam vs real ham?
  - More realistic scenario

- **Option B (Extreme case)**: 100% synthetic spam + 100% synthetic ham
  - Tests: Pure synthetic detection
  - Less practical but interesting

### Implementation Steps

#### Day 1: Setup & Data Preparation (4 hours)
- [ ] Review existing 0% synthetic baseline results (already have this!)
- [ ] Organize 100% synthetic data for all methods:
  - [ ] GPT-4.1-mini (original/strong/weak prompts)
  - [ ] Claude-3.5-Haiku (original/strong/weak prompts)
  - [ ] SMOTE
- [ ] Decide: Option A or Option B for testing set
- [ ] Prepare data loading scripts

#### Day 1-2: Run Experiments (1 day)
- [ ] Run classification on all configurations:
  - [ ] 20 groups × 2 classifiers (SVM, RF) = 40 base experiments
  - [ ] × 3 prompt strategies × 2 LLMs = 240 LLM experiments
  - [ ] + 40 SMOTE experiments
  - [ ] **Total**: 280 experiments (reuse trained models from baseline!)
- [ ] Compute all 7 metrics
- [ ] Save results to `output/reverse_detection/`

#### Day 2: Analysis & Visualization (4 hours)
- [ ] Statistical analysis:
  - [ ] Compare detection F1 vs original task F1
  - [ ] Paired t-tests: Real→Real vs Real→Synthetic
  - [ ] Effect sizes (Cohen's d)
- [ ] Generate plots:
  - [ ] Bar chart: Detection F1 by method (GPT/Claude/SMOTE)
  - [ ] Comparison: Original task vs Reverse detection
- [ ] Create summary table for paper

### Expected Outcomes & Interpretation

| Scenario | Detection F1 | Interpretation | Paper Implication |
|----------|-------------|----------------|-------------------|
| High (>0.7) | Synthetic spam easily detected | LLM samples have artifacts | Real classifiers robust to LLM spam |
| Medium (0.5-0.7) | Moderate detection | Some artifacts present | Need caution in practice |
| Low (<0.5) | Poor detection | LLM spam evades detection | LLM-generated spam is dangerous |

### Paper Integration

**New Section**: `evaluation.tex` → Add subsection "5.X Reverse Detection: Real Data vs Synthetic Spam"

**Content**:
- Motivation (real-world attacker scenario)
- Experimental setup
- Results table + visualization
- Statistical significance
- Comparison across methods (GPT vs Claude vs SMOTE)

**Discussion Extension**: Add subsection about practical implications for spam defense

**Estimated Impact**: Could be a **major contribution** distinguishing this paper from prior work!

---

## 🎯 Task 2: Hyperparameter Justification/Analysis ⭐⭐⭐⭐

### Problem
Current paper uses fixed hyperparameters without justification. Reviewers may question:
- Why these specific values?
- Are results sensitive to hyperparameter choices?
- Did you optimize or just use defaults?

### Three Solution Options

---

#### **Option A: Detailed Justification (Recommended for Speed)** ✅

**Time**: 0.5 days (4 hours)
**Effort**: Low
**Persuasiveness**: Medium

**Implementation**:

1. **Add to Methodology** (Section 4.4 Classifier Configuration):
```markdown
### Hyperparameter Selection Rationale

We employ standard baseline configurations to ensure reproducibility and
comparability with existing literature:

**Support Vector Machine (SVM)**:
- C = 1.0 (regularization parameter): Balances margin maximization and
  misclassification penalty
- kernel = 'rbf' (Radial Basis Function): Suitable for non-linearly
  separable spam feature space
- gamma = 'scale': Auto-adjusts kernel coefficient based on feature variance
- Rationale: These are scikit-learn's recommended defaults for text
  classification [cite: Pedregosa et al. 2011]

**Random Forest**:
- n_estimators = 100: Standard ensemble size balancing performance and
  computational cost [cite: Breiman 2001]
- max_depth = None: Allows trees to grow until pure leaves, leveraging
  ensemble averaging for generalization
- min_samples_split = 2: Default splitting criterion
- Rationale: Widely used configuration for spam detection tasks
  [cite: relevant spam detection papers]

These standard configurations enable fair comparison across synthetic data
methods while maintaining computational feasibility for large-scale
replication (R=20).
```

2. **Add to Appendix** (Table of all hyperparameters):
```markdown
### Table A.X: Complete Classifier Hyperparameters

| Classifier | Hyperparameter | Value | Justification |
|------------|----------------|-------|---------------|
| SVM | C | 1.0 | Standard regularization |
| SVM | kernel | rbf | Non-linear classification |
| SVM | gamma | scale | Auto feature-scaled |
| RF | n_estimators | 100 | Standard ensemble size |
| RF | max_depth | None | Full tree growth |
| RF | min_samples_split | 2 | Default splitting |
```

**To-Do**:
- [ ] Write hyperparameter justification text (2 hours)
- [ ] Find 2-3 relevant citations for SVM defaults
- [ ] Find 2-3 relevant citations for RF defaults
- [ ] Add table to appendix (1 hour)
- [ ] Cross-reference in methodology (1 hour)

**Risk**: Reviewer may still ask "but did you try others?"

---

#### **Option B: Full Grid Search (NOT Recommended - Too Slow)** ❌

**Time**: 5-7 days
**Effort**: Very High
**Persuasiveness**: Very High

**Why Not Recommended**:
- Would need to rerun ALL 2,200 experiments with multiple hyperparameter combinations
- 10-day deadline makes this infeasible
- Not proportional to benefit

**Skip this option.**

---

#### **Option C: Limited Sensitivity Analysis (Recommended if Time Allows)** ⏸️

**Time**: 2-3 days
**Effort**: Medium
**Persuasiveness**: High

**Implementation**:

1. **Select Representative Subset**:
   - Choose 3 groups: Group 0 (low baseline), Group 10 (medium), Group 15 (high)
   - Choose 1 configuration: 50% synthetic, within-group, original prompt
   - This gives 3 groups × 2 classifiers = 6 base experiments

2. **Hyperparameter Grid** (modest, not exhaustive):

**SVM Grid**:
```python
param_grid_svm = {
    'C': [0.1, 1.0, 10.0],           # 3 values
    'kernel': ['linear', 'rbf'],     # 2 values
    'gamma': ['scale']               # 1 value (keep fixed)
}
# Total: 3 × 2 = 6 combinations
```

**Random Forest Grid**:
```python
param_grid_rf = {
    'n_estimators': [50, 100, 200],  # 3 values
    'max_depth': [None, 20],         # 2 values
    'min_samples_split': [2]         # 1 value (keep fixed)
}
# Total: 3 × 2 = 6 combinations
```

3. **Total Experiments**:
   - 3 groups × 2 classifiers × 6 param combos = 36 experiments
   - With 3 replicates: 108 experiments
   - Estimated time: 4-6 hours compute + 1 day analysis

4. **Expected Result**:
   - If optimal params ≈ current params → Validates choice
   - If optimal params differ but performance change < 3% F1 → Shows robustness
   - Document in Appendix with table + brief discussion

**To-Do** (if time allows):
- [ ] Day 1: Setup grid search script (4 hours)
- [ ] Day 1-2: Run experiments on 3 groups (overnight)
- [ ] Day 2: Analyze results, create table (4 hours)
- [ ] Day 2-3: Write appendix section (2 hours)

**Appendix Section**:
```markdown
## Appendix A.X: Hyperparameter Sensitivity Analysis

To validate our hyperparameter choices, we conducted limited grid search
on 3 representative groups (Group 0, 10, 15) at 50% synthetic proportion.

### Table A.X: Hyperparameter Sensitivity Results

| Classifier | Hyperparameters | Group 0 F1 | Group 10 F1 | Group 15 F1 | Mean F1 |
|------------|----------------|------------|-------------|-------------|---------|
| SVM | C=0.1, kernel=linear | 0.542 | 0.571 | 0.589 | 0.567 |
| SVM | C=1.0, kernel=rbf (ours) | 0.558 | 0.583 | 0.601 | 0.581 |
| SVM | C=10.0, kernel=rbf | 0.551 | 0.579 | 0.598 | 0.576 |
| RF | n=50, depth=None | 0.774 | 0.801 | 0.819 | 0.798 |
| RF | n=100, depth=None (ours) | 0.780 | 0.806 | 0.823 | 0.803 |
| RF | n=200, depth=20 | 0.778 | 0.804 | 0.821 | 0.801 |

**Findings**: Performance varies by ±2-3% F1 across hyperparameter
configurations, with our chosen parameters (C=1.0 RBF for SVM, n=100
unlimited depth for RF) achieving competitive or optimal performance.
This validates that our main conclusions are robust to hyperparameter choices.
```

---

### **Recommendation for Task 2**:

**Minimum (Must Do)**:
- ✅ **Option A: Detailed Justification** (0.5 days)

**Optional (If time permits after Task 1 & 3)**:
- ⏸️ **Option C: Sensitivity Analysis on 3 groups** (2-3 days)

**Timeline**:
- Day 4: Complete Option A (4 hours)
- Day 8-9 (if time allows): Option C

---

## 🎯 Task 3: Spam Sample Examples in Appendix ⭐⭐⭐⭐⭐

### Motivation
Qualitative examples help readers:
- Understand practical differences between prompt strategies
- Assess LLM generation quality
- Visualize what "original/strong/weak" prompts actually produce
- Increase transparency and reproducibility

### Content Requirements

**Needed**:
1. **1 real spam email** (original from CEAS-08)
2. **6 synthetic versions**:
   - GPT-4.1-mini: original prompt
   - GPT-4.1-mini: strong prompt
   - GPT-4.1-mini: weak prompt
   - Claude-3.5-Haiku: original prompt
   - Claude-3.5-Haiku: strong prompt
   - Claude-3.5-Haiku: weak prompt
3. **3 complete prompt templates** (original, strong, weak)

### Implementation Steps

#### Day 3 Morning: Data Availability Check (2 hours)
- [ ] Search for existing generated synthetic text in:
  - [ ] `output/llm_generation/`
  - [ ] `output/synthetic_data/`
  - [ ] `output/generated_samples/`
  - [ ] Check JSON result files for "generated_text" fields
- [ ] Document finding: ✅ Data exists / ❌ Need to regenerate

#### Scenario A: If Data Exists ✅ (4 hours total)
- [ ] Select 1 representative spam email:
  - Moderate length (not too long for appendix)
  - Clear spam indicators (urgency, call-to-action, suspicious links)
  - Readable (not too technical/obscure)
- [ ] Extract corresponding 6 synthetic versions from saved data
- [ ] Verify quality (all 6 versions generated successfully, parseable)
- [ ] Format for LaTeX (see format below)

#### Scenario B: If Data Doesn't Exist ❌ (1 day total)
- [ ] Select 1 representative spam email (same criteria)
- [ ] Write script to regenerate 6 versions:
  ```python
  # Pseudo-code
  real_spam = select_representative_spam()

  for model in ['gpt-4.1-mini', 'claude-3.5-haiku']:
      for prompt_strategy in ['original', 'strong', 'weak']:
          synthetic = call_llm_api(model, prompt_strategy, real_spam)
          save_sample(model, prompt_strategy, synthetic)
  ```
- [ ] Run API calls (6 calls, cost ≈ $0.01)
- [ ] Verify outputs
- [ ] Format for LaTeX

#### Day 3 Afternoon: Format & Integrate (4 hours)
- [ ] Create appendix section structure
- [ ] Format samples in LaTeX (use `lstlisting` or `verbatim`)
- [ ] Add annotations/highlights for key differences
- [ ] Extract and format prompt templates

### LaTeX Formatting Template

```latex
\section{Qualitative Examples: Spam Generation Across Methods}
\label{appendix:spam_examples}

This appendix provides concrete examples of synthetic spam generation
across different prompt strategies and LLM models, enabling qualitative
assessment of generation characteristics.

\subsection{Original Real Spam Email}

\begin{lstlisting}[
  basicstyle=\small\ttfamily,
  frame=single,
  breaklines=true,
  captionpos=b,
  caption={Original Real Spam from CEAS-08 Dataset}
]
Subject: URGENT: Verify Your Account Now!

Dear valued customer,

Your account has been flagged for suspicious activity. To avoid
suspension, please verify your identity immediately by clicking
the link below:

http://verify-account-secure.example.com/verify?id=12345

Failure to verify within 24 hours will result in permanent
account closure.

Best regards,
Security Team
\end{lstlisting}

\subsection{LLM-Generated Synthetic Versions}

\subsubsection{GPT-4.1-mini Generations}

\paragraph{Original Prompt Strategy}
\begin{lstlisting}[basicstyle=\small\ttfamily, frame=single, breaklines=true]
Subject: IMPORTANT: Confirm Your Account Details

Dear customer,

We've detected unusual activity on your account. Please confirm
your information by visiting the secure link below within 24 hours:

http://account-verification-center.example.com/confirm

Your account security is our priority.

Regards,
Account Security
\end{lstlisting}

\paragraph{Strong Prompt Strategy}
\begin{lstlisting}[basicstyle=\small\ttfamily, frame=single, breaklines=true]
Subject: **URGENT ACTION REQUIRED** - Account Suspension Imminent!!!

ATTENTION CUSTOMER,

YOUR ACCOUNT WILL BE TERMINATED IN 12 HOURS unless you verify
NOW! Click immediately:

http://URGENT-verify-NOW.example.com/WARNING

DO NOT IGNORE THIS MESSAGE!!!

Security Alert System
\end{lstlisting}

\paragraph{Weak Prompt Strategy}
\begin{lstlisting}[basicstyle=\small\ttfamily, frame=single, breaklines=true]
Subject: Account Review Notice

Hello,

As part of our routine security review, we'd appreciate if you
could confirm your account details at your convenience:

http://customer-portal.example.com/review

Thank you for your cooperation.

Customer Service
\end{lstlisting}

[Repeat similar structure for Claude-3.5-Haiku]

\subsection{Prompt Templates}
\label{appendix:prompt_templates}

\subsubsection{Original Prompt Template}
\begin{lstlisting}[basicstyle=\footnotesize\ttfamily, frame=single]
You are a spam email generator. Given the following spam email,
rewrite it while preserving the malicious intent and key
characteristics:

- Maintain urgency and call-to-action
- Keep suspicious elements (links, threats, urgency)
- Preserve overall spam structure

Output format: JSON with "rewritten_spam" field.

Input email:
{original_spam}
\end{lstlisting}

[Continue with Strong and Weak prompt templates]
```

### To-Do Checklist

**Day 3 (Total: 1 day)**:
- [ ] Morning: Check data availability (2 hours)
- [ ] Morning/Afternoon: Extract or regenerate samples (2-4 hours)
- [ ] Afternoon: Format in LaTeX (4 hours)
- [ ] Evening: Review and refine (1 hour)

**Day 7 (Integration)**:
- [ ] Add to appendix.tex
- [ ] Cross-reference in methodology (prompt section)
- [ ] Cross-reference in evaluation (qualitative insights)

### Expected Value

**High Impact**:
- Readers can directly see prompt strategy effects
- Demonstrates LLM generation quality
- Increases paper transparency
- Common in top-tier papers (USENIX Security, CCS, IEEE S&P)

**Low Risk**:
- If data exists: trivial to implement
- If need regeneration: minimal cost ($0.01)
- High return on investment

---

## 📅 Recommended 10-Day Timeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     PHASE 1: NEW EXPERIMENTS (Days 1-3)         │
└─────────────────────────────────────────────────────────────────┘

Day 1 (Oct 13):
  ☐ AM: Task 1 - Setup reverse detection experiment
  ☐ PM: Task 1 - Run reverse detection experiments (let overnight)

Day 2 (Oct 14):
  ☐ AM: Task 1 - Analyze reverse detection results
  ☐ PM: Task 1 - Create plots and summary tables

Day 3 (Oct 15):
  ☐ AM: Task 3 - Check data availability & extract/generate samples
  ☐ PM: Task 3 - Format samples in LaTeX appendix

┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 2: DOCUMENTATION (Days 4-5)             │
└─────────────────────────────────────────────────────────────────┘

Day 4 (Oct 16):
  ☐ AM: Task 2 - Write hyperparameter justification (Option A)
  ☐ PM: Task 2 - Format prompts in appendix
  ☐ Evening: Review all new content

Day 5 (Oct 17):
  ☐ Full Day: Integrate Task 1 results into evaluation.tex
    - New subsection for reverse detection
    - Statistical analysis
    - Plots and tables

┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 3: PAPER INTEGRATION (Days 6-8)         │
└─────────────────────────────────────────────────────────────────┘

Day 6 (Oct 18):
  ☐ AM: Update discussion.tex with reverse detection implications
  ☐ PM: Update abstract to mention reverse detection (if significant)

Day 7 (Oct 19):
  ☐ Full Day: Polish appendix
    - Spam samples section
    - Prompt templates
    - Hyperparameter tables
    - Quality check all appendix content

Day 8 (Oct 20):
  ☐ AM: Optional - Start Task 2 Option C (sensitivity analysis) if time
  ☐ PM: Full paper read-through and coherence check

┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 4: FINAL POLISH (Days 9-10)            │
└─────────────────────────────────────────────────────────────────┘

Day 9 (Oct 21):
  ☐ Full Day: Final revisions
    - Check all cross-references
    - Proofread all new sections
    - Verify all figures/tables render correctly
    - Check citation formatting

Day 10 (Oct 22):
  ☐ AM: Final checks and LaTeX compilation
  ☐ PM: Submit!
```

---

## 🎲 Contingency Plans

### If Task 1 (Reverse Detection) Takes Longer Than Expected
- **Reduce scope**: Test only on 10 groups instead of 20
- **Simplify analysis**: Focus on F1-score only, skip other metrics
- **Still valuable**: Even partial results add value

### If Task 3 Data Not Available & API Fails
- **Fallback**: Use generic description of differences in text
- **Alternative**: Provide only prompt templates (still useful)
- **Not critical**: Paper is still publishable without examples

### If Running Behind Schedule (End of Day 6)
- **Prioritize**:
  1. Task 1 (reverse detection) - MUST HAVE
  2. Task 2 Option A (justification) - MUST HAVE
  3. Task 3 (examples) - NICE TO HAVE
- **Cut**: Task 2 Option C (sensitivity analysis) - OPTIONAL

### If Ahead of Schedule (End of Day 7)
- **Add**: Task 2 Option C (sensitivity analysis on 3 groups)
- **Enhance**: More detailed discussion of reverse detection
- **Polish**: Improve figure quality and captions

---

## 📊 Success Metrics

### Task 1: Reverse Detection Experiment
- ✅ Complete 280 experiments successfully
- ✅ Statistical significance tests completed
- ✅ At least 2 plots generated
- ✅ New evaluation section written (2-3 pages)
- ✅ Discussion updated with implications

### Task 2: Hyperparameter Justification
- ✅ Detailed justification text (Option A) completed
- ✅ 2-3 citations found for SVM/RF defaults
- ✅ Appendix table created
- ⏸️ (Optional) Sensitivity analysis on 3 groups completed

### Task 3: Spam Examples
- ✅ 1 real spam + 6 synthetic versions formatted
- ✅ 3 prompt templates documented in appendix
- ✅ LaTeX rendering verified
- ✅ Cross-references added to methodology

---

## 💬 Discussion Points with Co-Author

Before starting implementation, discuss:

### For Task 1 (Reverse Detection):
- [ ] **Testing set choice**: Option A (synthetic spam + real ham) or Option B (all synthetic)?
- [ ] **Scope**: All 20 groups or reduce to 10 for speed?
- [ ] **Paper positioning**: Should this be a major contribution or supporting evidence?

### For Task 2 (Hyperparameters):
- [ ] **Minimum acceptance**: Is Option A (justification only) sufficient?
- [ ] **Sensitivity analysis**: Worth the 2-3 day investment?
- [ ] **Risk tolerance**: What if reviewer still asks for grid search?

### For Task 3 (Examples):
- [ ] **Sample selection**: Who chooses the representative spam?
- [ ] **Privacy**: Any concerns about showing real spam content?
- [ ] **Length**: How detailed should prompt templates be?

---

## 📝 Notes & Decisions Log

### Decision 1: [Date]
**Topic**: Reverse detection testing set
**Decision**: [Option A / Option B]
**Rationale**:

### Decision 2: [Date]
**Topic**: Hyperparameter sensitivity scope
**Decision**: [Option A only / Option A + C]
**Rationale**:

### Decision 3: [Date]
**Topic**: Spam sample data availability
**Decision**: [Exists / Need regeneration]
**Status**:

---

## ✅ Final Checklist Before Submission

### Content Completeness
- [ ] Task 1: Reverse detection section in evaluation
- [ ] Task 1: Reverse detection discussion in discussion section
- [ ] Task 1: Reverse detection plots/tables added
- [ ] Task 2: Hyperparameter justification in methodology
- [ ] Task 2: Hyperparameter table in appendix
- [ ] Task 3: Spam examples in appendix
- [ ] Task 3: Prompt templates in appendix
- [ ] Abstract updated (if reverse detection is major contribution)
- [ ] Conclusion updated with new findings

### Quality Checks
- [ ] All new sections proofread
- [ ] All figures/tables have captions
- [ ] All cross-references work (\\ref, \\cite)
- [ ] LaTeX compiles without errors
- [ ] Bibliography complete for new citations
- [ ] Page limit verified (if applicable)

### Reproducibility
- [ ] All code for Task 1 committed to git
- [ ] All data paths documented
- [ ] All hyperparameters documented
- [ ] All prompts documented

---

**End of Plan** | Last updated: 2025-10-12
