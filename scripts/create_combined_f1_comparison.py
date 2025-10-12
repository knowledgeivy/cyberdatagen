#!/usr/bin/env python3
"""
Create combined F1 comparison charts including GPT, Claude, and SMOTE
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Setup matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_f1_data(analysis_file):
    """Load F1 score data from statistical analysis file"""
    with open(analysis_file, 'r') as f:
        data = json.load(f)

    descriptive_stats = data.get('descriptive_statistics', {})

    f1_data = {'svm': {}, 'random_forest': {}}

    for ratio_str, ratio_data in descriptive_stats.items():
        ratio = int(ratio_str)
        for clf in ['svm', 'random_forest']:
            if clf in ratio_data and 'f1_score' in ratio_data[clf]:
                f1_data[clf][ratio] = {
                    'mean': ratio_data[clf]['f1_score']['mean'],
                    'ci_lower': ratio_data[clf]['f1_score']['ci_lower'],
                    'ci_upper': ratio_data[clf]['f1_score']['ci_upper']
                }

    return f1_data

def create_combined_f1_comparison(strategy):
    """Create combined F1 comparison chart for a strategy"""

    # Define paths
    base_path = Path('./output/full_experiments')

    # Load data from all sources
    sources = {
        'GPT-4.1-mini': base_path / 'ceas08_gpt41mini' / 'reports' / f'full_ceas08_gpt41mini_v1_original_{strategy}_statistical_analysis.json',
        'Claude-3.5-Haiku': base_path / 'ceas08_claude35haiku' / 'reports' / f'full_ceas08_claude35haiku_v1_original_{strategy}_statistical_analysis.json',
        'SMOTE': base_path / 'ceas08_smote' / 'reports' / f'full_ceas08_smote_v1_smote_{strategy}_statistical_analysis.json'
    }

    all_data = {}
    for source_name, source_file in sources.items():
        if source_file.exists():
            all_data[source_name] = load_f1_data(source_file)
            print(f"✓ Loaded {source_name}")
        else:
            print(f"✗ Missing {source_name}: {source_file}")

    if len(all_data) < 2:
        print(f"Error: Not enough data for {strategy}")
        return

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Define colors and styles for each source
    source_styles = {
        'GPT-4.1-mini': {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o'},
        'Claude-3.5-Haiku': {'color': '#ff7f0e', 'linestyle': '--', 'marker': 's'},
        'SMOTE': {'color': '#2ca02c', 'linestyle': '-.', 'marker': '^'}
    }

    # Plot for each classifier
    for ax_idx, (ax, clf) in enumerate([(ax1, 'svm'), (ax2, 'random_forest')]):
        clf_label = 'SVM' if clf == 'svm' else 'Random Forest'

        for source_name, f1_data in all_data.items():
            if clf not in f1_data:
                continue

            # Extract data
            ratios = sorted(f1_data[clf].keys())
            means = [f1_data[clf][r]['mean'] for r in ratios]
            ci_lowers = [f1_data[clf][r]['ci_lower'] for r in ratios]
            ci_uppers = [f1_data[clf][r]['ci_upper'] for r in ratios]

            # Get style
            style = source_styles.get(source_name, {'color': 'gray', 'linestyle': '-', 'marker': 'o'})

            # Plot line with CI
            ax.plot(ratios, means,
                   color=style['color'],
                   linestyle=style['linestyle'],
                   marker=style['marker'],
                   linewidth=2,
                   markersize=6,
                   label=source_name,
                   alpha=0.9)
            ax.fill_between(ratios, ci_lowers, ci_uppers,
                           color=style['color'],
                           alpha=0.15)

        # Configure plot
        ax.set_xlabel('Synthetic Ratio (%)', fontsize=12)
        ax.set_ylabel('F1-Score', fontsize=12)
        ax.set_title(f'{clf_label}', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10, framealpha=0.9)

    # Overall title
    strategy_title = strategy.replace('_', '-').title()
    fig.suptitle(f'F1-Score Comparison: LLM vs Traditional Baseline ({strategy_title})',
                fontsize=16, fontweight='bold')

    plt.tight_layout()

    # Save
    output_file = f'paper/pic/combined_f1_comparison_{strategy}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_file}")

def main():
    """Generate both within_group and cross_group comparisons"""
    print("=" * 60)
    print("Generating Combined F1 Comparison Charts")
    print("=" * 60)
    print()

    for strategy in ['within_group', 'cross_group']:
        print(f"Processing {strategy}...")
        create_combined_f1_comparison(strategy)
        print()

    print("=" * 60)
    print("All charts generated!")
    print("=" * 60)

if __name__ == '__main__':
    main()
