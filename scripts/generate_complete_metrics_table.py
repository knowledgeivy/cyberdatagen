#!/usr/bin/env python3
"""
Generate complete metrics table for appendix (all metrics with avg ± sd)
"""

import json
import numpy as np
from pathlib import Path

# Paths
PROJECT_ROOT = Path("/Users/tianyu/Notebooks/cyberdata")
GPT_RESULTS_DIR = PROJECT_ROOT / "output/full_experiments/ceas08_gpt41mini/results"
CLAUDE_RESULTS_DIR = PROJECT_ROOT / "output/full_experiments/ceas08_claude35haiku/results"

def load_all_groups(results_dir, model_name, prompt="original", strategy="cross_group"):
    """Load results from all group files"""
    all_data = []

    for group_id in range(20):
        file_path = results_dir / f"full_ceas08_{model_name}_v1_{prompt}_{strategy}_group{group_id}_results.json"

        if not file_path.exists():
            print(f"Warning: {file_path} not found")
            continue

        with open(file_path, 'r') as f:
            data = json.load(f)
            # Extract results array from the JSON structure
            if 'results' in data:
                all_data.extend(data['results'])
            else:
                all_data.extend(data)

    return all_data

def compute_stats(values):
    """Compute mean and std"""
    return np.mean(values), np.std(values, ddof=1)

def extract_metrics(data, synthetic_ratio):
    """Extract all metrics for a specific synthetic ratio"""
    metrics = {
        'f1': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'auc_roc': [],
        'balanced_accuracy': []
    }

    for result in data:
        if result['synthetic_ratio'] == synthetic_ratio:
            # Extract metrics from svm classifier
            if 'classifiers' in result and 'svm' in result['classifiers']:
                svm_metrics = result['classifiers']['svm']['metrics']
                metrics['f1'].append(svm_metrics['f1_score'])
                metrics['accuracy'].append(svm_metrics['accuracy'])
                metrics['precision'].append(svm_metrics['precision'])
                metrics['recall'].append(svm_metrics['recall'])
                metrics['auc_roc'].append(svm_metrics['auc_roc'])
                metrics['balanced_accuracy'].append(svm_metrics['balanced_accuracy'])

    # Compute mean ± sd for each metric
    stats = {}
    for metric, values in metrics.items():
        if values:
            mean, std = compute_stats(values)
            stats[metric] = (mean, std)
        else:
            stats[metric] = (0, 0)

    return stats

def format_metric(mean, std):
    """Format metric as mean ± std"""
    return f"{mean:.3f} $\\pm$ {std:.3f}"

def generate_table():
    """Generate complete metrics table"""
    print("Loading GPT results...")
    gpt_data = load_all_groups(GPT_RESULTS_DIR, "gpt41mini")

    print("Loading Claude results...")
    claude_data = load_all_groups(CLAUDE_RESULTS_DIR, "claude35haiku")

    # Extract metrics for 0% and 100% synthetic
    print("\nExtracting GPT metrics...")
    gpt_0 = extract_metrics(gpt_data, 0)
    gpt_100 = extract_metrics(gpt_data, 100)

    print("Extracting Claude metrics...")
    claude_0 = extract_metrics(claude_data, 0)
    claude_100 = extract_metrics(claude_data, 100)

    # Generate LaTeX table
    latex = r"""\begin{table}[h]
\centering
\caption{Complete Classification Metrics (Track 1, SVM, Original Prompt, Cross-Group)}
\label{tab:complete_metrics}
\resizebox{\columnwidth}{!}{
\begin{tabular}{llcccccc}
\toprule
\textbf{Model} & \textbf{Ratio} & \textbf{F1} & \textbf{Acc} & \textbf{Prec} & \textbf{Rec} & \textbf{AUC} & \textbf{BAcc} \\
\midrule
"""

    # GPT rows
    latex += f"\\multirow{{2}}{{*}}{{GPT-4.1-mini}} & 0\\% & "
    latex += f"{format_metric(*gpt_0['f1'])} & "
    latex += f"{format_metric(*gpt_0['accuracy'])} & "
    latex += f"{format_metric(*gpt_0['precision'])} & "
    latex += f"{format_metric(*gpt_0['recall'])} & "
    latex += f"{format_metric(*gpt_0['auc_roc'])} & "
    latex += f"{format_metric(*gpt_0['balanced_accuracy'])} \\\\\n"

    latex += f"& 100\\% & "
    latex += f"{format_metric(*gpt_100['f1'])} & "
    latex += f"{format_metric(*gpt_100['accuracy'])} & "
    latex += f"{format_metric(*gpt_100['precision'])} & "
    latex += f"{format_metric(*gpt_100['recall'])} & "
    latex += f"{format_metric(*gpt_100['auc_roc'])} & "
    latex += f"{format_metric(*gpt_100['balanced_accuracy'])} \\\\\n"

    latex += "\\midrule\n"

    # Claude rows
    latex += f"\\multirow{{2}}{{*}}{{Claude-3.5-Haiku}} & 0\\% & "
    latex += f"{format_metric(*claude_0['f1'])} & "
    latex += f"{format_metric(*claude_0['accuracy'])} & "
    latex += f"{format_metric(*claude_0['precision'])} & "
    latex += f"{format_metric(*claude_0['recall'])} & "
    latex += f"{format_metric(*claude_0['auc_roc'])} & "
    latex += f"{format_metric(*claude_0['balanced_accuracy'])} \\\\\n"

    latex += f"& 100\\% & "
    latex += f"{format_metric(*claude_100['f1'])} & "
    latex += f"{format_metric(*claude_100['accuracy'])} & "
    latex += f"{format_metric(*claude_100['precision'])} & "
    latex += f"{format_metric(*claude_100['recall'])} & "
    latex += f"{format_metric(*claude_100['auc_roc'])} & "
    latex += f"{format_metric(*claude_100['balanced_accuracy'])} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
}
\end{table}
"""

    return latex

def main():
    print("=" * 80)
    print("Generating Complete Metrics Table for Appendix")
    print("=" * 80)

    latex_table = generate_table()

    print("\n" + "=" * 80)
    print("Generated LaTeX Table:")
    print("=" * 80)
    print(latex_table)

    # Save to file
    output_file = PROJECT_ROOT / "output/latex_tables/complete_metrics_table.tex"
    with open(output_file, 'w') as f:
        f.write(latex_table)

    print(f"\n✅ Table saved to {output_file}")

if __name__ == "__main__":
    main()
