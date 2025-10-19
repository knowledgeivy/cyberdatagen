#!/usr/bin/env python3
"""
Generate LaTeX tables from experiment results
"""

import json
import os
from pathlib import Path
import numpy as np
from collections import defaultdict

def load_track1_data(base_dir, model, prompt):
    """Load Track 1 data for a specific model and prompt"""
    results_dir = f"{base_dir}/output/full_experiments/ceas08_{model}/results"
    data = {0: [], 100: []}

    for group_id in range(20):
        filename = f"full_ceas08_{model}_v1_{prompt}_cross_group_group{group_id}_results.json"
        filepath = os.path.join(results_dir, filename)

        if not os.path.exists(filepath):
            continue

        with open(filepath, 'r') as f:
            file_data = json.load(f)

        for result in file_data['results']:
            ratio = result['synthetic_ratio']
            if ratio in [0, 100]:
                metrics = result['classifiers']['svm']['metrics']
                data[ratio].append(metrics)

    return data

def load_track2a_data(base_dir, model, prompt):
    """Load Track 2a data for a specific model and prompt"""
    results_dir = f"{base_dir}/output/real_enhanced_detection/results"
    data = {100: defaultdict(list), 150: defaultdict(list), 200: defaultdict(list)}

    for count in [100, 150, 200]:
        filename = f"{model}_{prompt}_cross_group_count{count}_real_enhanced_results.json"
        filepath = os.path.join(results_dir, filename)

        if not os.path.exists(filepath):
            continue

        with open(filepath, 'r') as f:
            file_data = json.load(f)

        for result in file_data['results']:
            for clf_name in ['svm', 'random_forest']:
                if clf_name in result['classifiers']:
                    metrics = result['classifiers'][clf_name]['metrics']
                    data[count][clf_name].append(metrics)

    return data

def load_track2b_data(base_dir, model, prompt):
    """Load Track 2b data for a specific model and prompt"""
    results_dir = f"{base_dir}/output/reverse_detection/results"
    data = {0: defaultdict(list), 50: defaultdict(list), 100: defaultdict(list)}

    for ratio in [0, 50, 100]:
        filename = f"{model}_{prompt}_cross_group_ratio{ratio}_reverse_results.json"
        filepath = os.path.join(results_dir, filename)

        if not os.path.exists(filepath):
            continue

        with open(filepath, 'r') as f:
            file_data = json.load(f)

        for result in file_data['results']:
            for clf_name in ['svm', 'random_forest']:
                if clf_name in result['classifiers']:
                    metrics = result['classifiers'][clf_name]['metrics']
                    data[ratio][clf_name].append(metrics)

    return data

def compute_stats(metrics_list, metric_name='f1_score'):
    """Compute mean and std for a list of metrics"""
    values = [m[metric_name] for m in metrics_list]
    return np.mean(values), np.std(values)

def format_metric(mean, std):
    """Format metric as mean ± std"""
    return f"{mean:.3f} $\\pm$ {std:.3f}"

def generate_track1_latex(base_dir):
    """Generate Track 1 LaTeX table"""
    table = r"""\begin{table*}[t]
\centering
\caption{Track 1: Synthetic-to-Real Detection Performance (SVM, Cross-Group)}
\label{tab:track1_summary}
\resizebox{\textwidth}{!}{
\begin{tabular}{llcccccc}
\toprule
\textbf{Model} & \textbf{Prompt} & \multicolumn{3}{c}{\textbf{Baseline (0\% Synthetic)}} & \multicolumn{3}{c}{\textbf{100\% Synthetic}} \\
\cmidrule(lr){3-5} \cmidrule(lr){6-8}
& & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} \\
\midrule
"""

    for model in ['gpt41mini', 'claude35haiku']:
        model_name = 'GPT-4.1-mini' if model == 'gpt41mini' else 'Claude-3.5-Haiku'
        for i, prompt in enumerate(['original', 'strong', 'weak']):
            data = load_track1_data(base_dir, model, prompt)

            if len(data[0]) > 0 and len(data[100]) > 0:
                # Baseline metrics
                baseline_p_mean, baseline_p_std = compute_stats(data[0], 'precision')
                baseline_r_mean, baseline_r_std = compute_stats(data[0], 'recall')
                baseline_f_mean, baseline_f_std = compute_stats(data[0], 'f1_score')

                # 100% synthetic metrics
                synth_p_mean, synth_p_std = compute_stats(data[100], 'precision')
                synth_r_mean, synth_r_std = compute_stats(data[100], 'recall')
                synth_f_mean, synth_f_std = compute_stats(data[100], 'f1_score')

                if i == 0:
                    model_row = f"\\multirow{{3}}{{*}}{{{model_name}}}"
                else:
                    model_row = ""

                table += f"{model_row} & {prompt.capitalize()} & "
                table += f"{format_metric(baseline_p_mean, baseline_p_std)} & "
                table += f"{format_metric(baseline_r_mean, baseline_r_std)} & "
                table += f"{format_metric(baseline_f_mean, baseline_f_std)} & "
                table += f"{format_metric(synth_p_mean, synth_p_std)} & "
                table += f"{format_metric(synth_r_mean, synth_r_std)} & "
                table += f"{format_metric(synth_f_mean, synth_f_std)} \\\\\n"

        if model == 'gpt41mini':
            table += "\\midrule\n"

    table += r"""\bottomrule
\end{tabular}
}
\end{table*}
"""
    return table

def generate_track2a_latex(base_dir):
    """Generate Track 2a LaTeX table"""
    table = r"""\begin{table*}[t]
\centering
\caption{Track 2a: Real-to-Synthetic Detection Performance (Cross-Group)}
\label{tab:track2a_summary}
\resizebox{\textwidth}{!}{
\begin{tabular}{llllccc}
\toprule
\textbf{Testing} & \textbf{Prompt} & \textbf{Classifier} & \multicolumn{3}{c}{\textbf{Training Real Spam Count}} \\
\cmidrule(lr){4-6}
\textbf{Target} & & & \textbf{100 (100\%)} & \textbf{150 (150\%)} & \textbf{200 (200\%)} \\
\midrule
"""

    for model in ['gpt41mini', 'claude35haiku']:
        model_name = 'GPT-4.1-mini' if model == 'gpt41mini' else 'Claude-3.5-Haiku'
        model_row_count = 0

        for prompt in ['original', 'strong', 'weak']:
            data = load_track2a_data(base_dir, model, prompt)

            for clf in ['svm', 'random_forest']:
                clf_name = 'SVM' if clf == 'svm' else 'Random Forest'

                if model_row_count == 0:
                    model_col = f"\\multirow{{6}}{{*}}{{{model_name}}}"
                else:
                    model_col = ""

                table += f"{model_col} & {prompt.capitalize()} & {clf_name} & "

                for count in [100, 150, 200]:
                    if len(data[count][clf]) > 0:
                        mean, std = compute_stats(data[count][clf])
                        table += f"{format_metric(mean, std)} & "
                    else:
                        table += "N/A & "

                table = table.rstrip(" & ") + " \\\\\n"
                model_row_count += 1

        if model == 'gpt41mini':
            table += "\\midrule\n"

    table += r"""\bottomrule
\end{tabular}
}
\end{table*}
"""
    return table

def generate_track2b_latex(base_dir):
    """Generate Track 2b LaTeX table"""
    table = r"""\begin{table*}[t]
\centering
\caption{Track 2b: Mixed-to-Synthetic Detection Performance (Cross-Group)}
\label{tab:track2b_summary}
\resizebox{\textwidth}{!}{
\begin{tabular}{llllccc}
\toprule
\textbf{Testing} & \textbf{Prompt} & \textbf{Classifier} & \multicolumn{3}{c}{\textbf{Training Synthetic Ratio}} \\
\cmidrule(lr){4-6}
\textbf{Target} & & & \textbf{0\% (Baseline)} & \textbf{50\%} & \textbf{100\%} \\
\midrule
"""

    for model in ['gpt41mini', 'claude35haiku']:
        model_name = 'GPT-4.1-mini' if model == 'gpt41mini' else 'Claude-3.5-Haiku'
        model_row_count = 0

        for prompt in ['original', 'strong', 'weak']:
            data = load_track2b_data(base_dir, model, prompt)

            for clf in ['svm', 'random_forest']:
                clf_name = 'SVM' if clf == 'svm' else 'Random Forest'

                if model_row_count == 0:
                    model_col = f"\\multirow{{6}}{{*}}{{{model_name}}}"
                else:
                    model_col = ""

                table += f"{model_col} & {prompt.capitalize()} & {clf_name} & "

                for ratio in [0, 50, 100]:
                    if len(data[ratio][clf]) > 0:
                        mean, std = compute_stats(data[ratio][clf])
                        table += f"{format_metric(mean, std)} & "
                    else:
                        table += "N/A & "

                table = table.rstrip(" & ") + " \\\\\n"
                model_row_count += 1

        if model == 'gpt41mini':
            table += "\\midrule\n"

    table += r"""\bottomrule
\end{tabular}
}
\end{table*}
"""
    return table

if __name__ == "__main__":
    base_dir = "/Users/tianyu/Notebooks/cyberdata"
    output_dir = os.path.join(base_dir, "output/latex_tables")

    os.makedirs(output_dir, exist_ok=True)

    # Generate tables
    track1_table = generate_track1_latex(base_dir)
    track2a_table = generate_track2a_latex(base_dir)
    track2b_table = generate_track2b_latex(base_dir)

    # Save to files
    with open(os.path.join(output_dir, "track1_table.tex"), 'w') as f:
        f.write(track1_table)

    with open(os.path.join(output_dir, "track2a_table.tex"), 'w') as f:
        f.write(track2a_table)

    with open(os.path.join(output_dir, "track2b_table.tex"), 'w') as f:
        f.write(track2b_table)

    # Also save all together
    with open(os.path.join(output_dir, "all_tables.tex"), 'w') as f:
        f.write("% Track 1: Synthetic-to-Real\n")
        f.write(track1_table)
        f.write("\n\n")
        f.write("% Track 2a: Real-to-Synthetic\n")
        f.write(track2a_table)
        f.write("\n\n")
        f.write("% Track 2b: Mixed-to-Synthetic\n")
        f.write(track2b_table)

    print(f"LaTeX tables saved to {output_dir}/")
    print("Files created:")
    print("  - track1_table.tex")
    print("  - track2a_table.tex")
    print("  - track2b_table.tex")
    print("  - all_tables.tex")

    print("\n=== TRACK 1 TABLE ===")
    print(track1_table)
    print("\n" + "="*80 + "\n")

    print("=== TRACK 2A TABLE ===")
    print(track2a_table)
    print("\n" + "="*80 + "\n")

    print("=== TRACK 2B TABLE ===")
    print(track2b_table)
