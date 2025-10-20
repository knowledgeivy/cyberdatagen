#!/usr/bin/env python3
"""
Generate supplementary tables with complete experimental results
"""

import json
import csv
from pathlib import Path
from collections import defaultdict


def load_gpt_data():
    """Load GPT experimental data from CSV"""
    gpt_file = Path("output/full_experiments/ceas08_gpt41mini/paper_tables/full_summary_statistics.csv")

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    with open(gpt_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['strategy'] != 'cross_group':
                continue

            prompt = row['prompt']
            ratio = int(row['synthetic_ratio'])
            classifier = row['classifier']
            metric = row['metric']

            data[prompt][ratio][classifier][metric] = {
                'mean': float(row['mean']),
                'std': float(row['std'])
            }

    return data


def load_claude_data():
    """Load Claude experimental data from statistical analysis JSON files"""
    prompts = ['original', 'strong', 'weak']
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for prompt in prompts:
        claude_file = Path(f"output/full_experiments/ceas08_claude35haiku/reports/full_ceas08_claude35haiku_v1_{prompt}_cross_group_statistical_analysis.json")

        if not claude_file.exists():
            print(f"Warning: {claude_file} not found")
            continue

        with open(claude_file, 'r') as f:
            json_data = json.load(f)

        # Extract from descriptive_statistics
        desc_stats = json_data.get('descriptive_statistics', {})

        for ratio_str, ratio_data in desc_stats.items():
            ratio = int(ratio_str)

            for classifier in ['svm', 'random_forest']:
                if classifier not in ratio_data:
                    continue

                classifier_data = ratio_data[classifier]

                for metric_name, metric_values in classifier_data.items():
                    if isinstance(metric_values, dict) and 'mean' in metric_values and 'std' in metric_values:
                        data[prompt][ratio][classifier][metric_name] = {
                            'mean': metric_values['mean'],
                            'std': metric_values['std']
                        }

    return data


def format_metric(mean, std):
    """Format metric as mean ± std"""
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def generate_track1_table(gpt_data, claude_data, prompt, classifier='svm'):
    """Generate Track 1 complete metrics table for a specific prompt and classifier"""

    metrics_order = ['f1_score', 'accuracy', 'precision', 'recall', 'auc_roc', 'balanced_accuracy']
    metric_names = {
        'f1_score': 'F1',
        'accuracy': 'Acc',
        'precision': 'Prec',
        'recall': 'Rec',
        'auc_roc': 'AUC',
        'balanced_accuracy': 'BAcc'
    }

    classifier_name = 'SVM' if classifier == 'svm' else 'Random Forest'

    tex = []
    tex.append("\\begin{table}[ht]")
    tex.append("\\centering")
    tex.append(f"\\caption{{Track 1 Complete Metrics: {prompt.title()} Prompt, {classifier_name}, Cross-Group}}")
    tex.append(f"\\label{{tab:track1_{prompt}_{classifier}}}")
    tex.append("\\small")
    tex.append("\\resizebox{\\columnwidth}{!}{")

    # Generate header
    header = "\\begin{tabular}{c" + "cc" * len(metrics_order) + "}"
    tex.append(header)
    tex.append("\\toprule")

    # Column headers
    headers = "\\multirow{2}{*}{\\textbf{Synth}} & \\multicolumn{" + str(len(metrics_order)) + "}{c}{\\textbf{GPT-4.1-mini}} & \\multicolumn{" + str(len(metrics_order)) + "}{c}{\\textbf{Claude-3.5-Haiku}} \\\\"
    tex.append(headers)

    metric_headers = "\\cmidrule(lr){2-" + str(1 + len(metrics_order)) + "} \\cmidrule(lr){" + str(2 + len(metrics_order)) + "-" + str(1 + 2 * len(metrics_order)) + "}"
    tex.append(metric_headers)

    metric_names_row = "\\textbf{Ratio} & " + " & ".join([f"\\textbf{{{metric_names[m]}}}" for m in metrics_order]) + " & " + " & ".join([f"\\textbf{{{metric_names[m]}}}" for m in metrics_order]) + " \\\\"
    tex.append(metric_names_row)
    tex.append("\\midrule")

    # Data rows
    for ratio in range(0, 101, 10):
        row_data = [f"{ratio}\\%"]

        # GPT metrics
        for metric in metrics_order:
            if ratio in gpt_data[prompt] and classifier in gpt_data[prompt][ratio] and metric in gpt_data[prompt][ratio][classifier]:
                values = gpt_data[prompt][ratio][classifier][metric]
                row_data.append(format_metric(values['mean'], values['std']))
            else:
                row_data.append("--")

        # Claude metrics
        for metric in metrics_order:
            if ratio in claude_data[prompt] and classifier in claude_data[prompt][ratio] and metric in claude_data[prompt][ratio][classifier]:
                values = claude_data[prompt][ratio][classifier][metric]
                row_data.append(format_metric(values['mean'], values['std']))
            else:
                row_data.append("--")

        tex.append(" & ".join(row_data) + " \\\\")

    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("}")
    tex.append("\\end{table}")
    tex.append("")

    return "\n".join(tex)


def main():
    print("Loading experimental data...")

    # Load Track 1 data
    gpt_data = load_gpt_data()
    claude_data = load_claude_data()

    print(f"Loaded GPT data for prompts: {list(gpt_data.keys())}")
    print(f"Loaded Claude data for prompts: {list(claude_data.keys())}")

    # Generate LaTeX document
    tex = []

    # Document header
    tex.append("\\documentclass[11pt]{article}")
    tex.append("\\usepackage[margin=1in]{geometry}")
    tex.append("\\usepackage{booktabs}")
    tex.append("\\usepackage{multirow}")
    tex.append("\\usepackage{graphicx}")
    tex.append("\\usepackage{float}")
    tex.append("")
    tex.append("\\title{Supplementary Tables:\\\\")
    tex.append("Complete Experimental Results}")
    tex.append("\\author{}")
    tex.append("\\date{}")
    tex.append("")
    tex.append("\\begin{document}")
    tex.append("\\maketitle")
    tex.append("")

    # Introduction
    tex.append("\\section*{Overview}")
    tex.append("")
    tex.append("This document contains complete experimental results for all tracks and configurations. ")
    tex.append("All results are reported as mean $\\pm$ standard deviation computed across 20 independent replications.")
    tex.append("")
    tex.append("\\textbf{Metrics:}")
    tex.append("\\begin{itemize}")
    tex.append("\\item \\textbf{F1}: F1-Score (primary metric)")
    tex.append("\\item \\textbf{Acc}: Accuracy")
    tex.append("\\item \\textbf{Prec}: Precision")
    tex.append("\\item \\textbf{Rec}: Recall")
    tex.append("\\item \\textbf{AUC}: AUC-ROC")
    tex.append("\\item \\textbf{BAcc}: Balanced Accuracy")
    tex.append("\\end{itemize}")
    tex.append("")
    tex.append("\\clearpage")
    tex.append("")

    # Track 1 Tables
    tex.append("\\section*{Track 1: Synthetic-to-Real Detection}")
    tex.append("")
    tex.append("Complete metrics across all synthetic training ratios (0\\%-100\\%) for both LLM models, ")
    tex.append("all prompt strategies, and both classifiers using cross-group mixing.")
    tex.append("")

    prompts = ['original', 'strong', 'weak']
    classifiers = ['svm', 'random_forest']

    for prompt in prompts:
        for classifier in classifiers:
            print(f"Generating Track 1 table: {prompt} - {classifier}")
            tex.append(generate_track1_table(gpt_data, claude_data, prompt, classifier))
            if prompt != 'weak' or classifier != 'random_forest':
                tex.append("\\clearpage")
            tex.append("")

    # Track 2 Tables
    tex.append("\\clearpage")
    tex.append("\\section*{Track 2: Real-to-Synthetic Detection}")
    tex.append("")

    # Load existing Track 2 tables
    track2a_file = Path("output/latex_tables/track2a_table.tex")
    track2b_file = Path("output/latex_tables/track2b_table.tex")

    if track2a_file.exists():
        with open(track2a_file, 'r') as f:
            tex.append("\\subsection*{Track 2a: Baseline Performance}")
            tex.append("")
            tex.append(f.read())
            tex.append("")

    if track2b_file.exists():
        with open(track2b_file, 'r') as f:
            tex.append("\\subsection*{Track 2b: Cross-Model Augmentation}")
            tex.append("")
            tex.append(f.read())
            tex.append("")

    # Document footer
    tex.append("\\end{document}")

    # Write output
    output_file = Path("data_release/tables/supplementary_tables.tex")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("\n".join(tex))

    print(f"\nGenerated supplementary tables: {output_file}")
    print("Compile with: pdflatex supplementary_tables.tex")


if __name__ == "__main__":
    main()
