#!/usr/bin/env python3
"""
Extract data for LaTeX tables from experiment results
"""

import json
import os
from pathlib import Path
import numpy as np
from collections import defaultdict

def load_track1_data(base_dir, model, prompt):
    """Load Track 1 data for a specific model and prompt"""
    results_dir = f"{base_dir}/output/full_experiments/ceas08_{model}/results"

    # Collect results for ratio 0 and 100
    data = {0: [], 100: []}

    for group_id in range(20):
        filename = f"full_ceas08_{model}_v1_{prompt}_cross_group_group{group_id}_results.json"
        filepath = os.path.join(results_dir, filename)

        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
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

    # Collect results for count 100, 150, 200
    data = {100: defaultdict(list), 150: defaultdict(list), 200: defaultdict(list)}

    for count in [100, 150, 200]:
        filename = f"{model}_{prompt}_cross_group_count{count}_real_enhanced_results.json"
        filepath = os.path.join(results_dir, filename)

        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
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

    # Collect results for ratio 0, 50, 100
    data = {0: defaultdict(list), 50: defaultdict(list), 100: defaultdict(list)}

    for ratio in [0, 50, 100]:
        filename = f"{model}_{prompt}_cross_group_ratio{ratio}_reverse_results.json"
        filepath = os.path.join(results_dir, filename)

        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
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

def print_track1_table(base_dir):
    """Generate Track 1 table data"""
    print("\n=== Track 1: Synthetic-to-Real ===")
    print("Model | Prompt | Baseline (0%) | 100% Synth | Degradation")
    print("-" * 80)

    for model in ['gpt41mini', 'claude35haiku']:
        for prompt in ['original', 'strong', 'weak']:
            data = load_track1_data(base_dir, model, prompt)

            if len(data[0]) > 0 and len(data[100]) > 0:
                baseline_mean, baseline_std = compute_stats(data[0])
                synth_mean, synth_std = compute_stats(data[100])
                degradation = ((synth_mean - baseline_mean) / baseline_mean) * 100

                print(f"{model:20s} | {prompt:8s} | {baseline_mean:.3f}±{baseline_std:.3f} | "
                      f"{synth_mean:.3f}±{synth_std:.3f} | {degradation:+.1f}%")

def print_track2a_table(base_dir):
    """Generate Track 2a table data"""
    print("\n=== Track 2a: Real-to-Synthetic ===")
    print("Model | Prompt | Classifier | 100 spam | 150 spam | 200 spam")
    print("-" * 100)

    for model in ['gpt41mini', 'claude35haiku']:
        for prompt in ['original', 'strong', 'weak']:
            data = load_track2a_data(base_dir, model, prompt)

            for clf in ['svm', 'random_forest']:
                row = f"{model:20s} | {prompt:8s} | {clf:13s} |"
                for count in [100, 150, 200]:
                    if len(data[count][clf]) > 0:
                        mean, std = compute_stats(data[count][clf])
                        row += f" {mean:.3f}±{std:.3f} |"
                    else:
                        row += " N/A |"
                print(row)

def print_track2b_table(base_dir):
    """Generate Track 2b table data"""
    print("\n=== Track 2b: Mixed-to-Synthetic ===")
    print("Model | Prompt | Classifier | 0% synth | 50% synth | 100% synth")
    print("-" * 100)

    for model in ['gpt41mini', 'claude35haiku']:
        for prompt in ['original', 'strong', 'weak']:
            data = load_track2b_data(base_dir, model, prompt)

            for clf in ['svm', 'random_forest']:
                row = f"{model:20s} | {prompt:8s} | {clf:13s} |"
                for ratio in [0, 50, 100]:
                    if len(data[ratio][clf]) > 0:
                        mean, std = compute_stats(data[ratio][clf])
                        row += f" {mean:.3f}±{std:.3f} |"
                    else:
                        row += " N/A |"
                print(row)

if __name__ == "__main__":
    base_dir = "/Users/tianyu/Notebooks/cyberdata"

    print_track1_table(base_dir)
    print_track2a_table(base_dir)
    print_track2b_table(base_dir)
