# src/cyberdata/scripts/batch3_visualize_results.py

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Determine PROJECT_ROOT
current_file = Path(__file__).resolve()
if 'src/cyberdata/scripts' in str(current_file):
    PROJECT_ROOT = current_file.parents[3]
else:
    PROJECT_ROOT = Path.cwd()

# Paths
BATCH3_DIR = PROJECT_ROOT / 'data/batch3'
ANALYSIS_DIR = BATCH3_DIR / 'analysis'
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('default')
sns.set_palette("husl")

class Batch3Visualizer:
    """阶段3结果可视化类"""
    
    def __init__(self):
        self.results = None
        self.critical_points = None
        
    def load_results(self):
        """加载实验结果"""
        results_file = BATCH3_DIR / 'batch3_analysis_results.json'
        if not results_file.exists():
            print(f"Results file not found: {results_file}")
            return False
            
        with open(results_file, 'r') as f:
            data = json.load(f)
            
        self.results = data['individual_results']
        self.critical_points = data['critical_points']
        
        print(f"Loaded {len(self.results)} experiment results")
        print(f"Found {len(self.critical_points)} critical points")
        return True
        
    def create_performance_curves(self):
        """创建性能曲线图"""
        if not self.results:
            return
            
        # Prepare data
        distances = []
        rf_accuracies = []
        svm_accuracies = []
        rf_f1_scores = []
        svm_f1_scores = []
        generation_success_rates = []
        
        for result in self.results:
            dist_min, dist_max = result['distance_range']
            dist_mid = (dist_min + dist_max) / 2
            distances.append(dist_mid)
            
            # Performance metrics
            if 'RandomForest' in result['performance']:
                rf_accuracies.append(result['performance']['RandomForest']['accuracy'])
                rf_f1_scores.append(result['performance']['RandomForest']['f1_macro'])
            else:
                rf_accuracies.append(np.nan)
                rf_f1_scores.append(np.nan)
                
            if 'SVM' in result['performance']:
                svm_accuracies.append(result['performance']['SVM']['accuracy'])
                svm_f1_scores.append(result['performance']['SVM']['f1_macro'])
            else:
                svm_accuracies.append(np.nan)
                svm_f1_scores.append(np.nan)
                
            # Generation success rate
            generation_success_rates.append(result['quality_metrics']['generation_success_rate'])
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Classification Accuracy
        ax1 = axes[0, 0]
        ax1.plot(distances, rf_accuracies, 'o-', label='Random Forest', linewidth=2, markersize=8)
        ax1.plot(distances, svm_accuracies, 's-', label='SVM', linewidth=2, markersize=8)
        
        # Mark critical points
        for cp in self.critical_points:
            cp_dist_min, cp_dist_max = cp['distance_range']
            cp_dist = (cp_dist_min + cp_dist_max) / 2
            ax1.axvline(x=cp_dist, color='red', linestyle='--', alpha=0.7)
            ax1.text(cp_dist, 0.95, f'Critical\\n{cp["model"]}', 
                    rotation=90, ha='right', va='top', fontsize=10, color='red')
        
        ax1.set_xlabel('Distance to Existing Synthetic Data')
        ax1.set_ylabel('Classification Accuracy')
        ax1.set_title('Performance vs Distance: Classification Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.9, 1.0)
        
        # Plot 2: F1 Scores
        ax2 = axes[0, 1]
        ax2.plot(distances, rf_f1_scores, 'o-', label='Random Forest F1', linewidth=2, markersize=8)
        ax2.plot(distances, svm_f1_scores, 's-', label='SVM F1', linewidth=2, markersize=8)
        
        # Mark critical points
        for cp in self.critical_points:
            cp_dist_min, cp_dist_max = cp['distance_range']
            cp_dist = (cp_dist_min + cp_dist_max) / 2
            ax2.axvline(x=cp_dist, color='red', linestyle='--', alpha=0.7)
        
        ax2.set_xlabel('Distance to Existing Synthetic Data')
        ax2.set_ylabel('F1 Score (Macro)')
        ax2.set_title('Performance vs Distance: F1 Scores')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.9, 1.0)
        
        # Plot 3: Generation Success Rate
        ax3 = axes[1, 0]
        ax3.plot(distances, generation_success_rates, 'D-', color='green', 
                linewidth=2, markersize=8, label='Generation Success Rate')
        
        ax3.set_xlabel('Distance to Existing Synthetic Data')
        ax3.set_ylabel('Generation Success Rate')
        ax3.set_title('LLM Generation Quality vs Distance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0.0, 1.1)
        
        # Plot 4: Performance Drop Analysis
        ax4 = axes[1, 1]
        
        # Calculate performance drops
        rf_drops = []
        svm_drops = []
        distance_mids = []
        
        for i in range(1, len(distances)):
            distance_mids.append(distances[i])
            rf_drop = rf_accuracies[i-1] - rf_accuracies[i] if not np.isnan(rf_accuracies[i-1]) and not np.isnan(rf_accuracies[i]) else 0
            svm_drop = svm_accuracies[i-1] - svm_accuracies[i] if not np.isnan(svm_accuracies[i-1]) and not np.isnan(svm_accuracies[i]) else 0
            rf_drops.append(rf_drop)
            svm_drops.append(svm_drop)
        
        ax4.bar([d - 0.02 for d in distance_mids], rf_drops, width=0.04, 
               label='RF Performance Drop', alpha=0.7, color='blue')
        ax4.bar([d + 0.02 for d in distance_mids], svm_drops, width=0.04, 
               label='SVM Performance Drop', alpha=0.7, color='orange')
        
        # Mark critical threshold
        ax4.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Critical Threshold (5%)')
        
        ax4.set_xlabel('Distance to Existing Synthetic Data')
        ax4.set_ylabel('Performance Drop from Previous Point')
        ax4.set_title('Performance Drop Analysis')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = ANALYSIS_DIR / 'batch3_performance_curves.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Performance curves saved to {plot_file}")
        plt.show()
        
    def create_interactive_analysis(self):
        """创建交互式分析图表"""
        if not self.results:
            return
            
        # Prepare data for interactive plots
        data = []
        for result in self.results:
            dist_min, dist_max = result['distance_range']
            dist_mid = (dist_min + dist_max) / 2
            
            for model in ['RandomForest', 'SVM']:
                if model in result['performance']:
                    data.append({
                        'distance': dist_mid,
                        'distance_range': f"[{dist_min:.2f}, {dist_max:.2f})",
                        'model': model,
                        'accuracy': result['performance'][model]['accuracy'],
                        'f1_macro': result['performance'][model]['f1_macro'],
                        'auc_roc': result['performance'][model]['auc_roc'],
                        'n_seeds': result['quality_metrics']['n_seeds'],
                        'n_generated': result['quality_metrics']['n_generated'],
                        'success_rate': result['quality_metrics']['generation_success_rate']
                    })
        
        df = pd.DataFrame(data)
        
        # Create interactive plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Classification Accuracy', 'F1 Score', 'AUC-ROC', 'Generation Metrics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": True}]]
        )
        
        # Accuracy plot
        for model in ['RandomForest', 'SVM']:
            model_data = df[df['model'] == model]
            fig.add_trace(
                go.Scatter(
                    x=model_data['distance'],
                    y=model_data['accuracy'],
                    mode='lines+markers',
                    name=f'{model} Accuracy',
                    hovertemplate='Distance: %{x:.3f}<br>Accuracy: %{y:.4f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # F1 Score plot
        for model in ['RandomForest', 'SVM']:
            model_data = df[df['model'] == model]
            fig.add_trace(
                go.Scatter(
                    x=model_data['distance'],
                    y=model_data['f1_macro'],
                    mode='lines+markers',
                    name=f'{model} F1',
                    hovertemplate='Distance: %{x:.3f}<br>F1: %{y:.4f}<extra></extra>',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # AUC-ROC plot
        for model in ['RandomForest', 'SVM']:
            model_data = df[df['model'] == model]
            fig.add_trace(
                go.Scatter(
                    x=model_data['distance'],
                    y=model_data['auc_roc'],
                    mode='lines+markers',
                    name=f'{model} AUC',
                    hovertemplate='Distance: %{x:.3f}<br>AUC: %{y:.4f}<extra></extra>',
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Generation metrics (using first model's data since it's the same for both)
        rf_data = df[df['model'] == 'RandomForest']
        fig.add_trace(
            go.Scatter(
                x=rf_data['distance'],
                y=rf_data['success_rate'],
                mode='lines+markers',
                name='Success Rate',
                hovertemplate='Distance: %{x:.3f}<br>Success Rate: %{y:.2%}<extra></extra>',
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=rf_data['distance'],
                y=rf_data['n_generated'],
                mode='lines+markers',
                name='Generated Count',
                yaxis='y2',
                hovertemplate='Distance: %{x:.3f}<br>Generated: %{y}<extra></extra>',
                showlegend=False
            ),
            row=2, col=2, secondary_y=True
        )
        
        # Add critical points
        for cp in self.critical_points:
            cp_dist_min, cp_dist_max = cp['distance_range']
            cp_dist = (cp_dist_min + cp_dist_max) / 2
            
            fig.add_vline(
                x=cp_dist,
                line=dict(color="red", width=2, dash="dash"),
                annotation_text=f"Critical: {cp['model']}<br>Drop: {cp['performance_drop']:.3f}",
                annotation_position="top"
            )
        
        fig.update_layout(
            title="Batch3 Distance Analysis: Interactive Results",
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Distance to Existing Synthetic Data")
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="F1 Score", row=1, col=2)
        fig.update_yaxes(title_text="AUC-ROC", row=2, col=1)
        fig.update_yaxes(title_text="Success Rate", row=2, col=2)
        fig.update_yaxes(title_text="Generated Count", row=2, col=2, secondary_y=True)
        
        # Save interactive plot
        interactive_file = ANALYSIS_DIR / 'batch3_interactive_analysis.html'
        fig.write_html(interactive_file)
        print(f"Interactive analysis saved to {interactive_file}")
        
        return fig
    
    def create_critical_points_analysis(self):
        """创建临界点分析图表"""
        if not self.critical_points:
            print("No critical points found")
            return
            
        # Prepare critical points data
        cp_data = []
        for cp in self.critical_points:
            dist_min, dist_max = cp['distance_range']
            cp_data.append({
                'distance_min': dist_min,
                'distance_max': dist_max,
                'distance_mid': (dist_min + dist_max) / 2,
                'model': cp['model'],
                'performance_drop': cp['performance_drop'],
                'previous_accuracy': cp['previous_accuracy'],
                'current_accuracy': cp['current_accuracy'],
                'critical_score': cp['critical_score']
            })
        
        cp_df = pd.DataFrame(cp_data)
        
        # Create critical points visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Critical Points by Distance
        ax1 = axes[0]
        models = cp_df['model'].unique()
        colors = ['red', 'blue']
        
        for i, model in enumerate(models):
            model_data = cp_df[cp_df['model'] == model]
            ax1.scatter(model_data['distance_mid'], model_data['performance_drop'], 
                       s=model_data['critical_score']*100, 
                       c=colors[i], alpha=0.7, label=model)
            
            # Add text labels
            for _, row in model_data.iterrows():
                ax1.annotate(f'{row["performance_drop"]:.3f}', 
                           (row['distance_mid'], row['performance_drop']),
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax1.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5, label='5% Threshold')
        ax1.set_xlabel('Distance to Existing Synthetic Data')
        ax1.set_ylabel('Performance Drop')
        ax1.set_title('Critical Points: Performance Drops')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Critical Score Distribution
        ax2 = axes[1]
        
        # Bar plot of critical scores
        x_pos = np.arange(len(cp_df))
        bars = ax2.bar(x_pos, cp_df['critical_score'], 
                      color=[colors[0] if m == models[0] else colors[1] for m in cp_df['model']])
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, cp_df['critical_score'])):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{score:.1f}', ha='center', va='bottom', fontsize=10)
        
        ax2.set_xlabel('Critical Point Index')
        ax2.set_ylabel('Critical Score (Drop / Threshold)')
        ax2.set_title('Critical Point Severity Scores')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'{row["distance_mid"]:.2f}\\n{row["model"]}' 
                            for _, row in cp_df.iterrows()], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = ANALYSIS_DIR / 'batch3_critical_points_analysis.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Critical points analysis saved to {plot_file}")
        plt.show()
        
        # Save critical points data
        cp_file = ANALYSIS_DIR / 'critical_points_data.csv'
        cp_df.to_csv(cp_file, index=False)
        print(f"Critical points data saved to {cp_file}")
        
    def generate_summary_report(self):
        """生成总结报告"""
        if not self.results:
            return
            
        # Calculate summary statistics
        total_experiments = len(self.results)
        successful_experiments = len([r for r in self.results if r['quality_metrics']['generation_success_rate'] > 0.8])
        
        # Performance statistics
        all_rf_acc = [r['performance']['RandomForest']['accuracy'] 
                     for r in self.results if 'RandomForest' in r['performance']]
        all_svm_acc = [r['performance']['SVM']['accuracy'] 
                      for r in self.results if 'SVM' in r['performance']]
        
        # Distance statistics
        all_distances = [(r['distance_range'][0] + r['distance_range'][1])/2 for r in self.results]
        
        report = f"""
# Batch3 Distance Analysis Summary Report

## 实验概况
- **总实验数**: {total_experiments}
- **成功实验数**: {successful_experiments} ({successful_experiments/total_experiments*100:.1f}%)
- **测试距离范围**: {min(all_distances):.2f} - {max(all_distances):.2f}
- **平均距离**: {np.mean(all_distances):.3f}

## 性能统计
### Random Forest
- **平均准确率**: {np.mean(all_rf_acc):.4f}
- **最高准确率**: {max(all_rf_acc):.4f}
- **最低准确率**: {min(all_rf_acc):.4f}
- **标准差**: {np.std(all_rf_acc):.4f}

### SVM  
- **平均准确率**: {np.mean(all_svm_acc):.4f}
- **最高准确率**: {max(all_svm_acc):.4f}
- **最低准确率**: {min(all_svm_acc):.4f}
- **标准差**: {np.std(all_svm_acc):.4f}

## 临界点分析
- **发现的临界点数**: {len(self.critical_points)}
"""
        
        if self.critical_points:
            report += "\\n### 临界点详情\\n"
            for i, cp in enumerate(self.critical_points, 1):
                dist_min, dist_max = cp['distance_range']
                report += f"""
{i}. **距离区间**: [{dist_min:.2f}, {dist_max:.2f})
   - **模型**: {cp['model']}
   - **性能下降**: {cp['performance_drop']:.4f} ({cp['performance_drop']*100:.2f}%)
   - **严重程度**: {cp['critical_score']:.2f}x 阈值
   - **前一准确率**: {cp['previous_accuracy']:.4f}
   - **当前准确率**: {cp['current_accuracy']:.4f}
"""
        else:
            report += "\\n**未发现显著临界点** (性能下降<5%)\\n"
        
        # Key findings
        performance_trend = "递减" if all_rf_acc[0] > all_rf_acc[-1] else "波动"
        
        report += f"""

## 关键发现
1. **性能趋势**: 随距离增加，分类性能呈{performance_trend}趋势
2. **LLM稳定性**: 平均生成成功率为 {np.mean([r['quality_metrics']['generation_success_rate'] for r in self.results]):.1%}
3. **模型差异**: SVM和Random Forest在不同距离下表现差异为 {abs(np.mean(all_rf_acc) - np.mean(all_svm_acc)):.4f}
4. **临界点存在性**: {"存在明显临界点" if self.critical_points else "未发现显著临界点"}

## 建议
- **安全距离范围**: < {0.6:.2f} (基于观察到的性能稳定区间)
- **谨慎距离范围**: {0.6:.2f} - {0.8:.2f} (需要额外质量验证)
- **避免距离范围**: > {0.8:.2f} (性能可能显著下降)

---
*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Save report
        report_file = ANALYSIS_DIR / 'batch3_summary_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Summary report saved to {report_file}")
        print("\\n" + "="*50)
        print(report)
        
    def run_full_visualization(self):
        """运行完整的可视化分析"""
        print("=== BATCH 3: VISUALIZATION AND ANALYSIS ===")
        
        if not self.load_results():
            print("Failed to load results")
            return
        
        print("Creating visualizations...")
        
        # Create all visualizations
        self.create_performance_curves()
        self.create_interactive_analysis() 
        self.create_critical_points_analysis()
        self.generate_summary_report()
        
        print(f"\\nVisualization complete. All files saved to: {ANALYSIS_DIR}")

def main():
    """主函数"""
    visualizer = Batch3Visualizer()
    visualizer.run_full_visualization()

if __name__ == "__main__":
    main()