# src/cyberdata/scripts/batch2_comparative_analysis.py

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

def load_ml_results():
    """Load ML results from both batches"""
    print("Loading ML results from both batches...")
    
    # Load batch1 results
    batch1_file = BATCH1_DIR / 'ml_results/batch1_comparison_summary.json'
    batch2_file = BATCH2_DIR / 'ml_results/batch2_comparison_summary.json'
    
    batch1_results = None
    batch2_results = None
    
    if batch1_file.exists():
        with open(batch1_file, 'r') as f:
            batch1_results = json.load(f)
        print(f"Loaded batch1 ML results")
    else:
        print(f"Warning: Batch1 ML results not found: {batch1_file}")
    
    if batch2_file.exists():
        with open(batch2_file, 'r') as f:
            batch2_results = json.load(f)
        print(f"Loaded batch2 ML results")
    else:
        print(f"Warning: Batch2 ML results not found: {batch2_file}")
    
    return batch1_results, batch2_results

def load_embedding_analyses():
    """Load embedding analysis results from both batches"""
    print("Loading embedding analysis results...")
    
    # Load batch1 embedding analysis
    batch1_embed_file = BATCH1_DIR / 'embeddings/batch1_uncovered_analysis.json'
    batch2_embed_file = BATCH2_DIR / 'embeddings/batch2_uncovered_analysis.json'
    
    batch1_embed = None
    batch2_embed = None
    
    if batch1_embed_file.exists():
        with open(batch1_embed_file, 'r') as f:
            batch1_embed = json.load(f)
        print(f"Loaded batch1 embedding analysis")
    else:
        print(f"Warning: Batch1 embedding analysis not found: {batch1_embed_file}")
    
    if batch2_embed_file.exists():
        with open(batch2_embed_file, 'r') as f:
            batch2_embed = json.load(f)
        print(f"Loaded batch2 embedding analysis")
    else:
        print(f"Warning: Batch2 embedding analysis not found: {batch2_embed_file}")
    
    return batch1_embed, batch2_embed

def create_ml_comparison_visualization(batch1_results, batch2_results):
    """Create ML performance comparison visualizations"""
    print("Creating ML performance comparison...")
    
    if not batch1_results or not batch2_results:
        print("Cannot create ML comparison - missing results")
        return
    
    # Prepare data for visualization
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
                if exp in batch1_results and model in batch1_results[exp]:
                    batch1_data.append(batch1_results[exp][model][metric])
                else:
                    batch1_data.append(0)
                
                if exp in batch2_results and model in batch2_results[exp]:
                    batch2_data.append(batch2_results[exp][model][metric])
                else:
                    batch2_data.append(0)
                
                exp_labels.append(f"{exp}\n{model}")
        
        # Create bar plot
        x = np.arange(len(exp_labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, batch1_data, width, label='Batch1', alpha=0.8)
        bars2 = ax.bar(x + width/2, batch2_data, width, label='Batch2', alpha=0.8)
        
        ax.set_xlabel('Experiment + Model')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(exp_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = ANALYSIS_DIR / 'ml_performance_comparison.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"ML comparison visualization saved to {plot_file}")
    plt.show()

def create_coverage_comparison(batch1_embed, batch2_embed):
    """Create coverage comparison visualization"""
    print("Creating coverage comparison...")
    
    if not batch1_embed or not batch2_embed:
        print("Cannot create coverage comparison - missing embedding analysis")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Coverage rates comparison
    ax1 = axes[0]
    
    batch_names = []
    covered_counts = []
    uncovered_counts = []
    coverage_rates = []
    
    for batch_name, embed_data in [('Batch1', batch1_embed), ('Batch2', batch2_embed)]:
        if 'uncovered_analysis' in embed_data and 'real' in embed_data['uncovered_analysis']:
            uncovered_count = embed_data['uncovered_analysis']['real']['count']
            total_real = embed_data['data_type_counts'].get('real', 0)
            covered_count = total_real - uncovered_count
            coverage_rate = (covered_count / total_real * 100) if total_real > 0 else 0
            
            batch_names.append(batch_name)
            covered_counts.append(covered_count)
            uncovered_counts.append(uncovered_count)
            coverage_rates.append(coverage_rate)
    
    if batch_names:
        x = np.arange(len(batch_names))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, covered_counts, width, label='Covered', color='lightgreen', alpha=0.8)
        bars2 = ax1.bar(x + width/2, uncovered_counts, width, label='Uncovered', color='red', alpha=0.8)
        
        ax1.set_xlabel('Batch')
        ax1.set_ylabel('Number of Real Samples')
        ax1.set_title('Real Data Coverage by Synthetic Data')
        ax1.set_xticks(x)
        ax1.set_xticklabels(batch_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add count labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{int(height)}', ha='center', va='bottom')
    
    # Coverage rate percentage comparison
    ax2 = axes[1]
    
    if coverage_rates:
        bars = ax2.bar(batch_names, coverage_rates, color=['blue', 'darkblue'], alpha=0.7)
        ax2.set_xlabel('Batch')
        ax2.set_ylabel('Coverage Rate (%)')
        ax2.set_title('Synthetic Data Coverage Rate')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        # Add percentage labels
        for bar, rate in zip(bars, coverage_rates):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = ANALYSIS_DIR / 'coverage_comparison.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Coverage comparison visualization saved to {plot_file}")
    plt.show()

def calculate_improvement_metrics(batch1_results, batch2_results, batch1_embed, batch2_embed):
    """Calculate improvement metrics between batches"""
    print("Calculating improvement metrics...")
    
    improvements = {
        'ml_improvements': {},
        'coverage_improvements': {},
        'summary': {}
    }
    
    # ML performance improvements
    if batch1_results and batch2_results:
        experiments = ['real_only', 'rewrite', 'rewrite_strong', 'rewrite_weak']
        models = ['RandomForest', 'SVM']
        
        for exp in experiments:
            if exp in batch1_results and exp in batch2_results:
                improvements['ml_improvements'][exp] = {}
                for model in models:
                    if model in batch1_results[exp] and model in batch2_results[exp]:
                        batch1_acc = batch1_results[exp][model]['accuracy']
                        batch2_acc = batch2_results[exp][model]['accuracy']
                        improvement = (batch2_acc - batch1_acc) * 100  # percentage points
                        
                        improvements['ml_improvements'][exp][model] = {
                            'batch1_accuracy': batch1_acc,
                            'batch2_accuracy': batch2_acc,
                            'improvement_pp': improvement  # percentage points
                        }
    
    # Coverage improvements
    if batch1_embed and batch2_embed:
        batch1_uncovered = 0
        batch1_total = 0
        batch2_uncovered = 0
        batch2_total = 0
        
        if 'uncovered_analysis' in batch1_embed and 'real' in batch1_embed['uncovered_analysis']:
            batch1_uncovered = batch1_embed['uncovered_analysis']['real']['count']
            batch1_total = batch1_embed['data_type_counts'].get('real', 0)
        
        if 'uncovered_analysis' in batch2_embed and 'real' in batch2_embed['uncovered_analysis']:
            batch2_uncovered = batch2_embed['uncovered_analysis']['real']['count']
            batch2_total = batch2_embed['data_type_counts'].get('real', 0)
        
        if batch1_total > 0 and batch2_total > 0:
            batch1_coverage = (batch1_total - batch1_uncovered) / batch1_total * 100
            batch2_coverage = (batch2_total - batch2_uncovered) / batch2_total * 100
            coverage_improvement = batch2_coverage - batch1_coverage
            
            improvements['coverage_improvements'] = {
                'batch1_coverage_rate': batch1_coverage,
                'batch2_coverage_rate': batch2_coverage,
                'improvement_pp': coverage_improvement,
                'batch1_uncovered': batch1_uncovered,
                'batch2_uncovered': batch2_uncovered
            }
    
    # Generate summary
    significant_improvements = []
    if 'ml_improvements' in improvements:
        for exp, exp_data in improvements['ml_improvements'].items():
            for model, model_data in exp_data.items():
                if model_data['improvement_pp'] > 1.0:  # >1% improvement
                    significant_improvements.append(f"{exp}_{model}: +{model_data['improvement_pp']:.1f}pp")
    
    improvements['summary'] = {
        'significant_ml_improvements': significant_improvements,
        'coverage_improved': improvements.get('coverage_improvements', {}).get('improvement_pp', 0) > 0
    }
    
    return improvements

def create_improvement_heatmap(improvements):
    """Create heatmap showing improvements across experiments"""
    print("Creating improvement heatmap...")
    
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
    
    ax.set_title("ML Performance Improvement (Batch2 vs Batch1)\nPercentage Points")
    fig.tight_layout()
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Improvement (percentage points)', rotation=270, labelpad=20)
    
    # Save plot
    plot_file = ANALYSIS_DIR / 'improvement_heatmap.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Improvement heatmap saved to {plot_file}")
    plt.show()

def save_comprehensive_comparison(batch1_results, batch2_results, batch1_embed, batch2_embed, improvements):
    """Save comprehensive comparison results"""
    print("Saving comprehensive comparison results...")
    
    comprehensive_results = {
        'comparison_type': 'batch1_vs_batch2_final',
        'timestamp': pd.Timestamp.now().isoformat(),
        'batch1_ml_results': batch1_results,
        'batch2_ml_results': batch2_results,
        'batch1_embedding_analysis': batch1_embed,
        'batch2_embedding_analysis': batch2_embed,
        'improvements': improvements
    }
    
    # Save full results
    with open(ANALYSIS_DIR / 'final_comparative_results.json', 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    # Create executive summary
    summary = {
        'experiment_summary': 'Batch1 vs Batch2 LLM Sampling Strategy Comparison',
        'key_findings': improvements['summary'] if 'summary' in improvements else {},
        'ml_performance_changes': improvements['ml_improvements'],
        'coverage_improvements': improvements['coverage_improvements']
    }
    
    with open(ANALYSIS_DIR / 'executive_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Comprehensive comparison results saved to: {ANALYSIS_DIR}")

def print_final_summary(improvements):
    """Print final summary of improvements"""
    print("\n=== FINAL COMPARATIVE SUMMARY ===")
    
    # ML improvements
    if 'ml_improvements' in improvements:
        print("\nML Performance Changes (Batch2 vs Batch1):")
        print(f"{'Experiment':<15} {'Model':<12} {'Batch1':<8} {'Batch2':<8} {'Change':<8}")
        print("-" * 60)
        
        for exp, exp_data in improvements['ml_improvements'].items():
            for model, model_data in exp_data.items():
                batch1_acc = model_data['batch1_accuracy']
                batch2_acc = model_data['batch2_accuracy']
                change = model_data['improvement_pp']
                change_str = f"+{change:.1f}pp" if change >= 0 else f"{change:.1f}pp"
                
                print(f"{exp:<15} {model:<12} {batch1_acc:<8.3f} {batch2_acc:<8.3f} {change_str:<8}")
    
    # Coverage improvements
    if 'coverage_improvements' in improvements and improvements['coverage_improvements']:
        cov_data = improvements['coverage_improvements']
        print(f"\nCoverage Analysis:")
        print(f"Batch1 coverage rate: {cov_data['batch1_coverage_rate']:.1f}%")
        print(f"Batch2 coverage rate: {cov_data['batch2_coverage_rate']:.1f}%")
        print(f"Coverage improvement: {cov_data['improvement_pp']:+.1f} percentage points")
    else:
        print(f"\nCoverage Analysis: No coverage data available")
    
    # Key insights
    if 'summary' in improvements:
        summary = improvements['summary']
        print(f"\nKey Insights:")
        if summary.get('significant_ml_improvements'):
            print(f"Significant ML improvements: {len(summary['significant_ml_improvements'])}")
            for imp in summary['significant_ml_improvements']:
                print(f"  - {imp}")
        else:
            print("No significant ML improvements detected")
        
        if summary.get('coverage_improved'):
            print("✓ Coverage improved in batch2")
        else:
            print("✗ Coverage did not improve in batch2")

def main():
    print("=== BATCH1 vs BATCH2: FINAL COMPARATIVE ANALYSIS ===")
    
    # Load all results
    batch1_results, batch2_results = load_ml_results()
    batch1_embed, batch2_embed = load_embedding_analyses()
    
    # Calculate improvements
    improvements = calculate_improvement_metrics(batch1_results, batch2_results, batch1_embed, batch2_embed)
    
    # Create visualizations
    create_ml_comparison_visualization(batch1_results, batch2_results)
    create_coverage_comparison(batch1_embed, batch2_embed)
    create_improvement_heatmap(improvements)
    
    # Save comprehensive results
    save_comprehensive_comparison(batch1_results, batch2_results, batch1_embed, batch2_embed, improvements)
    
    # Print final summary
    print_final_summary(improvements)
    
    print(f"\nFinal comparative analysis complete. Results saved to: {ANALYSIS_DIR}")

if __name__ == "__main__":
    main()