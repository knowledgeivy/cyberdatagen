# src/cyberdata/scripts/compare_resampled_results.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Determine PROJECT_ROOT
current_file = Path(__file__).resolve()
if 'src/cyberdata/scripts' in str(current_file):
    PROJECT_ROOT = current_file.parents[3]
else:
    PROJECT_ROOT = Path.cwd()

print(f"PROJECT_ROOT: {PROJECT_ROOT}")

# Paths
BATCH1_DIR = PROJECT_ROOT / 'data/batch1'
BATCH2_DIR = PROJECT_ROOT / 'data/batch2'
ANALYSIS_DIR = PROJECT_ROOT / 'data/analysis'

# Create directories
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

def load_resampled_results():
    """Load resampled results from both batches"""
    print("Loading resampled results...")
    
    # Load batch1 resampled results
    batch1_ml_file = BATCH1_DIR / 'resampled/batch1_resampled_ml_results.json'
    batch1_embed_file = BATCH1_DIR / 'resampled/batch1_resampled_embedding_results.json'
    
    # Load batch2 resampled results
    batch2_ml_file = BATCH2_DIR / 'resampled/batch2_resampled_ml_results.json'
    batch2_embed_file = BATCH2_DIR / 'resampled/batch2_resampled_embedding_results.json'
    
    results = {}
    
    # Load ML results
    for batch_name, ml_file in [('batch1', batch1_ml_file), ('batch2', batch2_ml_file)]:
        if ml_file.exists():
            with open(ml_file, 'r') as f:
                data = json.load(f)
                results[f'{batch_name}_ml'] = data['ml_results']
                results[f'{batch_name}_ml_info'] = data['resampling_info']
            print(f"Loaded {batch_name} ML results")
        else:
            print(f"Warning: {batch_name} ML results not found: {ml_file}")
    
    # Load embedding results
    for batch_name, embed_file in [('batch1', batch1_embed_file), ('batch2', batch2_embed_file)]:
        if embed_file.exists():
            with open(embed_file, 'r') as f:
                data = json.load(f)
                results[f'{batch_name}_embedding'] = data['embedding_results']
            print(f"Loaded {batch_name} embedding results")
        else:
            print(f"Warning: {batch_name} embedding results not found: {embed_file}")
    
    return results

def create_resampled_ml_comparison(results):
    """Create ML performance comparison for resampled data"""
    print("Creating resampled ML comparison...")
    
    if 'batch1_ml' not in results or 'batch2_ml' not in results:
        print("Cannot create ML comparison - missing results")
        return
    
    batch1_ml = results['batch1_ml']
    batch2_ml = results['batch2_ml']
    
    experiments = ['real_only', 'rewrite', 'rewrite_strong', 'rewrite_weak']
    models = ['RandomForest', 'SVM']
    metrics = ['accuracy', 'f1_macro', 'f1_weighted']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        
        # Prepare data
        batch1_data = []
        batch2_data = []
        exp_labels = []
        
        for exp in experiments:
            for model in models:
                # Get batch1 data
                if exp in batch1_ml and model in batch1_ml[exp]:
                    batch1_data.append(batch1_ml[exp][model][metric])
                else:
                    batch1_data.append(0)
                
                # Get batch2 data
                if exp in batch2_ml and model in batch2_ml[exp]:
                    batch2_data.append(batch2_ml[exp][model][metric])
                else:
                    batch2_data.append(0)
                
                exp_labels.append(f"{exp}\n{model}")
        
        # Create bar plot
        x = np.arange(len(exp_labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, batch1_data, width, label='Batch1 (Random)', alpha=0.8, color='lightblue')
        bars2 = ax.bar(x + width/2, batch2_data, width, label='Batch2 (Enhanced)', alpha=0.8, color='lightcoral')
        
        ax.set_xlabel('Experiment + Model')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison (Resampled)')
        ax.set_xticks(x)
        ax.set_xticklabels(exp_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = ANALYSIS_DIR / 'resampled_ml_comparison.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Resampled ML comparison saved to: {plot_file}")
    plt.show()

def create_resampled_coverage_comparison(results):
    """Create coverage comparison for resampled data"""
    print("Creating resampled coverage comparison...")
    
    if 'batch1_embedding' not in results or 'batch2_embedding' not in results:
        print("Cannot create coverage comparison - missing embedding results")
        return
    
    batch1_embed = results['batch1_embedding']
    batch2_embed = results['batch2_embedding']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Coverage rates comparison
    ax1 = axes[0]
    
    coverage_data = []
    batch_names = ['Batch1\n(Random)', 'Batch2\n(Enhanced)']
    
    for batch_name, embed_data in [('Batch1', batch1_embed), ('Batch2', batch2_embed)]:
        if 'coverage_analysis' in embed_data:
            coverage_rate = embed_data['coverage_analysis']['coverage_rate']
            coverage_data.append(coverage_rate)
        else:
            coverage_data.append(0)
    
    bars = ax1.bar(batch_names, coverage_data, color=['lightblue', 'lightcoral'], alpha=0.7)
    ax1.set_ylabel('Coverage Rate (%)')
    ax1.set_title('Synthetic Data Coverage Rate\n(Resampled Data)')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    
    # Add percentage labels
    for bar, rate in zip(bars, coverage_data):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Sample counts comparison
    ax2 = axes[1]
    
    batch1_counts = batch1_embed.get('sample_counts', {})
    batch2_counts = batch2_embed.get('sample_counts', {})
    
    data_types = ['real', 'rewrite', 'rewrite_strong', 'rewrite_weak']
    batch1_vals = [batch1_counts.get(dt, 0) for dt in data_types]
    batch2_vals = [batch2_counts.get(dt, 0) for dt in data_types]
    
    x = np.arange(len(data_types))
    width = 0.35
    
    ax2.bar(x - width/2, batch1_vals, width, label='Batch1', alpha=0.8, color='lightblue')
    ax2.bar(x + width/2, batch2_vals, width, label='Batch2', alpha=0.8, color='lightcoral')
    
    ax2.set_xlabel('Data Type')
    ax2.set_ylabel('Sample Count')
    ax2.set_title('Sample Counts by Data Type\n(Resampled Data)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(data_types, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = ANALYSIS_DIR / 'resampled_coverage_comparison.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Resampled coverage comparison saved to: {plot_file}")
    plt.show()

def calculate_resampled_improvements(results):
    """Calculate improvements in resampled data"""
    print("Calculating resampled improvements...")
    
    improvements = {
        'ml_improvements': {},
        'coverage_improvement': {},
        'summary': {}
    }
    
    # ML improvements
    if 'batch1_ml' in results and 'batch2_ml' in results:
        batch1_ml = results['batch1_ml']
        batch2_ml = results['batch2_ml']
        
        experiments = ['real_only', 'rewrite', 'rewrite_strong', 'rewrite_weak']
        models = ['RandomForest', 'SVM']
        
        for exp in experiments:
            if exp in batch1_ml and exp in batch2_ml:
                improvements['ml_improvements'][exp] = {}
                for model in models:
                    if model in batch1_ml[exp] and model in batch2_ml[exp]:
                        batch1_acc = batch1_ml[exp][model]['accuracy']
                        batch2_acc = batch2_ml[exp][model]['accuracy']
                        improvement = (batch2_acc - batch1_acc) * 100  # percentage points
                        
                        improvements['ml_improvements'][exp][model] = {
                            'batch1_accuracy': batch1_acc,
                            'batch2_accuracy': batch2_acc,
                            'improvement_pp': improvement
                        }
    
    # Coverage improvements
    if 'batch1_embedding' in results and 'batch2_embedding' in results:
        batch1_embed = results['batch1_embedding']
        batch2_embed = results['batch2_embedding']
        
        if 'coverage_analysis' in batch1_embed and 'coverage_analysis' in batch2_embed:
            batch1_coverage = batch1_embed['coverage_analysis']['coverage_rate']
            batch2_coverage = batch2_embed['coverage_analysis']['coverage_rate']
            coverage_improvement = batch2_coverage - batch1_coverage
            
            improvements['coverage_improvement'] = {
                'batch1_coverage': batch1_coverage,
                'batch2_coverage': batch2_coverage,
                'improvement_pp': coverage_improvement
            }
    
    # Summary statistics
    significant_improvements = []
    if 'ml_improvements' in improvements:
        for exp, exp_data in improvements['ml_improvements'].items():
            for model, model_data in exp_data.items():
                if model_data['improvement_pp'] > 1.0:  # >1% improvement
                    significant_improvements.append(f"{exp}_{model}: +{model_data['improvement_pp']:.1f}pp")
    
    improvements['summary'] = {
        'significant_ml_improvements': significant_improvements,
        'coverage_improved': improvements.get('coverage_improvement', {}).get('improvement_pp', 0) > 0,
        'avg_improvement': np.mean([
            model_data['improvement_pp'] 
            for exp_data in improvements.get('ml_improvements', {}).values()
            for model_data in exp_data.values()
        ]) if improvements.get('ml_improvements') else 0
    }
    
    return improvements

def create_improvement_heatmap_resampled(improvements):
    """Create improvement heatmap for resampled data"""
    print("Creating resampled improvement heatmap...")
    
    if 'ml_improvements' not in improvements:
        print("No ML improvements data available for heatmap")
        return
    
    experiments = ['real_only', 'rewrite', 'rewrite_strong', 'rewrite_weak']
    models = ['RandomForest', 'SVM']
    
    # Create improvement matrix
    improvement_matrix = np.zeros((len(models), len(experiments)))
    
    for i, model in enumerate(models):
        for j, exp in enumerate(experiments):
            if exp in improvements['ml_improvements'] and model in improvements['ml_improvements'][exp]:
                improvement_matrix[i, j] = improvements['ml_improvements'][exp][model]['improvement_pp']
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(improvement_matrix, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=5)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(experiments)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(experiments)
    ax.set_yticklabels(models)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(experiments)):
            text = ax.text(j, i, f'{improvement_matrix[i, j]:.1f}pp',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title("ML Performance Improvement (Resampled Data)\nBatch2 vs Batch1 - Percentage Points")
    fig.tight_layout()
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Improvement (percentage points)', rotation=270, labelpad=20)
    
    # Save plot
    plot_file = ANALYSIS_DIR / 'resampled_improvement_heatmap.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Resampled improvement heatmap saved to: {plot_file}")
    plt.show()

def save_resampled_comparison_results(results, improvements):
    """Save comprehensive resampled comparison results"""
    print("Saving resampled comparison results...")
    
    comprehensive_results = {
        'comparison_type': 'resampled_batch1_vs_batch2',
        'timestamp': pd.Timestamp.now().isoformat(),
        'description': 'Comparison of batch1 vs batch2 using resampled data with reduced sample sizes',
        'resampled_results': results,
        'improvements': improvements
    }
    
    # Save full results
    with open(ANALYSIS_DIR / 'resampled_comparative_results.json', 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    # Create executive summary
    summary = {
        'experiment_summary': 'Resampled Data Analysis: Enhanced vs Random Sampling',
        'sample_sizes': {
            'malicious_per_type': results.get('batch1_ml_info', {}).get('target_malicious_per_type', 'unknown'),
            'benign_count': results.get('batch1_ml_info', {}).get('target_benign_count', 'unknown')
        },
        'key_findings': improvements.get('summary', {}),
        'ml_improvements': improvements.get('ml_improvements', {}),
        'coverage_improvement': improvements.get('coverage_improvement', {})
    }
    
    with open(ANALYSIS_DIR / 'resampled_executive_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Resampled comparison results saved to: {ANALYSIS_DIR}")

def print_resampled_final_summary(improvements, results):
    """Print final summary of resampled improvements"""
    print("\n=== RESAMPLED DATA COMPARISON SUMMARY ===")
    
    # Show sample sizes
    if 'batch1_ml_info' in results:
        info = results['batch1_ml_info']
        print(f"\nSample sizes used:")
        print(f"  Malicious per type: {info.get('target_malicious_per_type', 'unknown')}")
        print(f"  Benign samples: {info.get('target_benign_count', 'unknown')}")
    
    # ML improvements
    if 'ml_improvements' in improvements:
        print(f"\nML Performance Changes (Enhanced vs Random):")
        print(f"{'Experiment':<15} {'Model':<12} {'Random':<8} {'Enhanced':<8} {'Change':<8}")
        print("-" * 60)
        
        for exp, exp_data in improvements['ml_improvements'].items():
            for model, model_data in exp_data.items():
                batch1_acc = model_data['batch1_accuracy']
                batch2_acc = model_data['batch2_accuracy']
                change = model_data['improvement_pp']
                change_str = f"+{change:.1f}pp" if change >= 0 else f"{change:.1f}pp"
                
                print(f"{exp:<15} {model:<12} {batch1_acc:<8.3f} {batch2_acc:<8.3f} {change_str:<8}")
    
    # Coverage improvements
    if 'coverage_improvement' in improvements and improvements['coverage_improvement']:
        cov_data = improvements['coverage_improvement']
        print(f"\nCoverage Analysis:")
        print(f"Random sampling coverage: {cov_data['batch1_coverage']:.1f}%")
        print(f"Enhanced sampling coverage: {cov_data['batch2_coverage']:.1f}%")
        print(f"Coverage improvement: {cov_data['improvement_pp']:+.1f} percentage points")
    
    # Overall summary
    if 'summary' in improvements:
        summary = improvements['summary']
        print(f"\nOverall Assessment:")
        print(f"Average ML improvement: {summary.get('avg_improvement', 0):+.2f} percentage points")
        
        if summary.get('significant_ml_improvements'):
            print(f"Significant improvements ({len(summary['significant_ml_improvements'])}):")
            for imp in summary['significant_ml_improvements']:
                print(f"  • {imp}")
        else:
            print("No significant ML improvements detected (>1pp threshold)")
        
        if summary.get('coverage_improved'):
            print("✓ Coverage improved with enhanced sampling")
        else:
            print("✗ Coverage did not improve with enhanced sampling")

def main():
    print("=== RESAMPLED DATA COMPARISON ANALYSIS ===")
    
    # Load resampled results
    results = load_resampled_results()
    
    if not results:
        print("No resampled results found. Please run resample_for_analysis.py and run_analysis_on_resampled.py first.")
        return
    
    # Calculate improvements
    improvements = calculate_resampled_improvements(results)
    
    # Create visualizations
    create_resampled_ml_comparison(results)
    create_resampled_coverage_comparison(results)
    create_improvement_heatmap_resampled(improvements)
    
    # Save comprehensive results
    save_resampled_comparison_results(results, improvements)
    
    # Print final summary
    print_resampled_final_summary(improvements, results)
    
    print(f"\nResampled comparison analysis complete. Results saved to: {ANALYSIS_DIR}")

if __name__ == "__main__":
    main()