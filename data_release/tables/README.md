# Supplementary Tables

This directory contains supplementary tables with complete experimental results referenced in the paper.

## Files

- `supplementary_tables.tex`: Complete LaTeX document with all supplementary tables
- `supplementary_tables.pdf`: Compiled PDF

## Contents

The supplementary tables include complete classification metrics not shown in the main paper due to space constraints:

### Track 1: Synthetic-to-Real Detection (6 tables)
- Complete metrics across all synthetic ratios (0%, 10%, 20%, ..., 100%)
- Both LLM models (GPT-4.1-mini, Claude-3.5-Haiku)
- Both classifiers (SVM, Random Forest)
- All three prompt strategies (Original, Strong, Weak)
- All metrics: F1-Score, Accuracy, Precision, Recall, AUC-ROC, Balanced Accuracy

### Track 2a: Real-to-Synthetic Detection Baseline
- Training with pure real spam data
- Detection performance against LLM-generated spam
- Varying training sizes (100, 150, 200 real spam samples)
- All prompt strategies and both classifiers

### Track 2b: Cross-Model Augmentation
- Training with mixed real + cross-model synthetic data
- Detection improvements from synthetic augmentation
- Varying synthetic ratios (0%, 50%, 100%)
- All prompt strategies and both classifiers

## Compilation

To generate the PDF:

```bash
cd data_release/tables
pdflatex supplementary_tables.tex
```

Or use your preferred LaTeX compiler.

## Data Format

All results are reported as:
- Mean ± Standard Deviation
- Computed across 20 independent replications (20 cross-validation groups)
- Following the experimental design described in the main paper
- All experiments use cross-group mixing strategy

## Citation

If you use these tables, please cite the main paper:

```bibtex
@inproceedings{llm-synthetic-spam-2025,
  title={[Paper Title]},
  author={[Authors]},
  booktitle={[Conference]},
  year={2025}
}
```

## Additional Data

Raw experimental data and analysis scripts are available in the `output/` directory of the main repository.
