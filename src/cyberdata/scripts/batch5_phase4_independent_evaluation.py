#!/usr/bin/env python3
"""
批次5实验 - Phase 4: 模型训练与独立评估

独立评估不同纯净数据源的检测性能：

训练配置:
- 模型: RandomForest + SVM (与批次4保持一致)
- 评估指标: Accuracy, Precision, Recall, F1-Score
- 测试集: 固定的独立测试集(与批次4相同)

实验矩阵:
- 16个数据集配置 × 2个模型 = 32个实验点

性能对比分析:
1. Baseline Real vs Pure Synthetic 主要对比
2. 不同Prompt策略性能排序
3. 不同Malicious Ratio下的性能曲线
4. 模型敏感性分析 (RF vs SVM)

作者: Claude
创建时间: 2025-07-30
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import random
from typing import Dict, List, Tuple, Any
import logging
import json
import time
import warnings
warnings.filterwarnings('ignore')

# 机器学习库
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib

# 可视化库
import matplotlib.pyplot as plt
import seaborn as sns

# 项目路径设置
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.cyberdata.utils.logger_config import setup_logger
from src.cyberdata.utils.data_loader import load_csv_data

class Batch5Phase4IndependentEvaluation:
    """批次5 Phase 4: 独立评估器"""
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logger("batch5_phase4")
        self.project_root = PROJECT_ROOT
        self.batch5_dir = self.project_root / "data" / "batch5"
        
        # 输入和输出目录
        self.datasets_dir = self.batch5_dir / "pure_datasets"
        self.results_dir = self.batch5_dir / "results"
        self.models_dir = self.batch5_dir / "models"
        self.analysis_dir = self.batch5_dir / "phase4_analysis"
        self.viz_dir = self.batch5_dir / "visualizations" / "performance_analysis"
        
        # 创建输出目录
        for dir_path in [self.results_dir, self.models_dir, self.analysis_dir, self.viz_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 模型配置
        self.models_config = {
            'RandomForest': {
                'class': RandomForestClassifier,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 20,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 2025,
                    'n_jobs': -1
                }
            },
            'SVM': {
                'class': SVC,
                'params': {
                    'kernel': 'rbf',
                    'C': 1.0,
                    'gamma': 'scale',
                    'random_state': 2025,
                    'probability': True  # 为了支持概率预测
                }
            }
        }
        
        # TF-IDF配置
        self.tfidf_config = {
            'max_features': 10000,
            'max_df': 0.95,
            'min_df': 2,
            'stop_words': 'english',
            'ngram_range': (1, 2)
        }
        
        # 随机种子
        self.random_state = 2025
        
        self.logger.info("Phase 4独立评估初始化完成")
        self.logger.info(f"模型: {list(self.models_config.keys())}")
    
    def discover_datasets(self) -> List[str]:
        """发现可用的数据集配置"""
        try:
            self.logger.info("发现可用的数据集配置...")
            
            dataset_configs = []
            
            if not self.datasets_dir.exists():
                raise FileNotFoundError(f"数据集目录不存在: {self.datasets_dir}")
            
            # 扫描数据集目录
            for config_dir in self.datasets_dir.iterdir():
                if config_dir.is_dir():
                    train_file = config_dir / "train.csv"
                    test_file = config_dir / "test.csv"
                    config_file = config_dir / "config.json"
                    
                    if all(f.exists() for f in [train_file, test_file, config_file]):
                        dataset_configs.append(config_dir.name)
                    else:
                        self.logger.warning(f"配置 {config_dir.name} 文件不完整，跳过")
            
            dataset_configs.sort()  # 排序确保一致性
            
            self.logger.info(f"发现 {len(dataset_configs)} 个数据集配置: {dataset_configs}")
            
            if len(dataset_configs) == 0:
                raise ValueError("未发现任何有效的数据集配置")
            
            return dataset_configs
            
        except Exception as e:
            self.logger.error(f"发现数据集配置失败: {str(e)}")
            raise
    
    def load_dataset_config(self, config_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """加载单个数据集配置"""
        try:
            config_dir = self.datasets_dir / config_name
            
            # 加载训练和测试数据
            train_df = load_csv_data(config_dir / "train.csv", logger=self.logger)
            test_df = load_csv_data(config_dir / "test.csv", logger=self.logger)
            
            # 加载配置信息
            with open(config_dir / "config.json", 'r', encoding='utf-8') as f:
                config_info = json.load(f)
            
            self.logger.info(f"加载配置 {config_name}: 训练{len(train_df):,}样本, 测试{len(test_df):,}样本")
            
            return train_df, test_df, config_info
            
        except Exception as e:
            self.logger.error(f"加载数据集配置 {config_name} 失败: {str(e)}")
            raise
    
    def create_ml_pipeline(self, model_name: str) -> Pipeline:
        """创建机器学习pipeline"""
        try:
            model_config = self.models_config[model_name]
            
            # 创建TF-IDF向量化器
            tfidf = TfidfVectorizer(**self.tfidf_config)
            
            # 创建模型
            model = model_config['class'](**model_config['params'])
            
            # 创建pipeline
            pipeline = Pipeline([
                ('tfidf', tfidf),
                ('classifier', model)
            ])
            
            return pipeline
            
        except Exception as e:
            self.logger.error(f"创建{model_name} pipeline失败: {str(e)}")
            raise
    
    def train_and_evaluate_single(self, config_name: str, model_name: str) -> Dict[str, Any]:
        """训练和评估单个配置"""
        try:
            self.logger.info(f"训练评估: {config_name} + {model_name}")
            
            # 加载数据
            train_df, test_df, config_info = self.load_dataset_config(config_name)
            
            # 准备训练数据
            X_train = train_df['text'].fillna('').astype(str)
            y_train = train_df['label'].astype(int)
            
            # 准备测试数据
            X_test = test_df['text'].fillna('').astype(str)
            y_test = test_df['label'].astype(int)
            
            # 创建和训练模型
            pipeline = self.create_ml_pipeline(model_name)
            
            train_start = time.time()
            pipeline.fit(X_train, y_train)
            train_time = time.time() - train_start
            
            # 预测
            predict_start = time.time()
            y_pred = pipeline.predict(X_test)
            predict_time = time.time() - predict_start
            
            # 计算评估指标
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='binary', zero_division=0)
            }
            
            # 详细分类报告
            class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            # 结果汇总
            result = {
                'config_name': config_name,
                'model_name': model_name,
                'config_info': config_info,
                'metrics': metrics,
                'classification_report': class_report,
                'timing': {
                    'train_time': train_time,
                    'predict_time': predict_time,
                    'total_time': train_time + predict_time
                },
                'data_info': {
                    'train_samples': len(train_df),
                    'test_samples': len(test_df),
                    'train_malicious_ratio': config_info['train_stats']['actual_malicious_ratio'],
                    'test_malicious_ratio': config_info['test_stats']['malicious_ratio']
                },
                'evaluation_date': '2025-07-30'
            }
            
            # 保存模型
            model_file = self.models_dir / f"{config_name}_{model_name}_model.pkl"
            joblib.dump(pipeline, model_file)
            result['model_file'] = str(model_file)
            
            # 保存单个结果
            result_file = self.results_dir / f"{config_name}_{model_name}_result.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"✅ {config_name} + {model_name}: "
                           f"F1={metrics['f1_score']:.3f}, "
                           f"Acc={metrics['accuracy']:.3f}, "
                           f"训练耗时={train_time:.1f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"训练评估 {config_name} + {model_name} 失败: {str(e)}")
            raise
    
    def run_all_experiments(self, dataset_configs: List[str]) -> Dict[str, Dict[str, Any]]:
        """运行所有实验"""
        try:
            self.logger.info("开始运行所有实验...")
            
            all_results = {}
            total_experiments = len(dataset_configs) * len(self.models_config)
            current_experiment = 0
            
            for config_name in dataset_configs:
                for model_name in self.models_config.keys():
                    current_experiment += 1
                    self.logger.info(f"实验进度: {current_experiment}/{total_experiments}")
                    
                    try:
                        experiment_key = f"{config_name}_{model_name}"
                        result = self.train_and_evaluate_single(config_name, model_name)
                        all_results[experiment_key] = result
                        
                    except Exception as e:
                        self.logger.error(f"实验 {config_name} + {model_name} 失败: {str(e)}")
                        # 继续其他实验
                        continue
            
            self.logger.info(f"完成 {len(all_results)} 个实验 (总计 {total_experiments} 个)")
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"运行所有实验失败: {str(e)}")
            raise
    
    def create_performance_analysis(self, all_results: Dict[str, Dict[str, Any]]) -> None:
        """创建性能分析图表"""
        try:
            self.logger.info("创建性能分析图表...")
            
            # 准备数据
            results_data = []
            for experiment_key, result in all_results.items():
                config_name = result['config_name']
                model_name = result['model_name']
                
                # 解析配置名称
                parts = config_name.split('_')
                if len(parts) >= 2:
                    group_name = '_'.join(parts[:-1])  # 除了最后一部分的所有部分
                    ratio_str = parts[-1]  # 最后一部分，如 "5pct"
                    ratio = float(ratio_str.replace('pct', '')) / 100
                else:
                    group_name = config_name
                    ratio = 0.0
                
                results_data.append({
                    'experiment_key': experiment_key,
                    'config_name': config_name,
                    'group_name': group_name,
                    'model_name': model_name,
                    'malicious_ratio': ratio,
                    'accuracy': result['metrics']['accuracy'],
                    'precision': result['metrics']['precision'],
                    'recall': result['metrics']['recall'],
                    'f1_score': result['metrics']['f1_score'],
                    'train_time': result['timing']['train_time']
                })
            
            results_df = pd.DataFrame(results_data)
            
            # 1. 性能对比曲线图
            self.create_performance_curves(results_df)
            
            # 2. 热力图对比
            self.create_performance_heatmaps(results_df)
            
            # 3. 模型对比分析
            self.create_model_comparison(results_df)
            
            # 4. 数据组效果对比
            self.create_group_comparison(results_df)
            
        except Exception as e:
            self.logger.error(f"创建性能分析失败: {str(e)}")
    
    def create_performance_curves(self, results_df: pd.DataFrame) -> None:
        """创建性能曲线图"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('批次5纯净数据性能对比曲线', fontsize=16, fontweight='bold')
            
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            metric_names = ['准确率', '精确率', '召回率', 'F1分数']
            
            for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
                ax = axes[idx//2, idx%2]
                
                # 按数据组和模型绘制曲线
                for group in results_df['group_name'].unique():
                    for model in results_df['model_name'].unique():
                        mask = (results_df['group_name'] == group) & (results_df['model_name'] == model)
                        group_data = results_df[mask].sort_values('malicious_ratio')
                        
                        if len(group_data) > 0:
                            line_style = '-' if model == 'RandomForest' else '--'
                            ax.plot(group_data['malicious_ratio'] * 100, group_data[metric], 
                                   line_style, marker='o', linewidth=2, markersize=6,
                                   label=f'{group}_{model}')
                
                ax.set_xlabel('恶意样本比例 (%)')
                ax.set_ylabel(metric_name)
                ax.set_title(f'{metric_name}性能曲线')
                ax.grid(True, alpha=0.3)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
                # 设置x轴刻度
                ax.set_xticks([5, 10, 15, 20])
            
            plt.tight_layout()
            
            curves_file = self.viz_dir / "performance_curves.png"
            plt.savefig(curves_file, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"性能曲线图已保存: {curves_file}")
            
        except Exception as e:
            self.logger.error(f"创建性能曲线图失败: {str(e)}")
    
    def create_performance_heatmaps(self, results_df: pd.DataFrame) -> None:
        """创建性能热力图"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            fig.suptitle('批次5性能热力图对比', fontsize=16, fontweight='bold')
            
            metrics = ['accuracy', 'f1_score']
            models = ['RandomForest', 'SVM']
            
            for model_idx, model in enumerate(models):
                for metric_idx, metric in enumerate(metrics):
                    ax = axes[model_idx, metric_idx]
                    
                    # 准备热力图数据
                    model_data = results_df[results_df['model_name'] == model]
                    
                    # 创建透视表
                    heatmap_data = model_data.pivot(
                        index='group_name', 
                        columns='malicious_ratio', 
                        values=metric
                    )
                    
                    # 转换比例为百分比
                    heatmap_data.columns = [f'{int(col*100)}%' for col in heatmap_data.columns]
                    
                    # 绘制热力图
                    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
                               ax=ax, cbar_kws={'label': metric.replace('_', ' ').title()})
                    
                    ax.set_title(f'{model} - {metric.replace("_", " ").title()}')
                    ax.set_xlabel('恶意样本比例')
                    ax.set_ylabel('数据组')
            
            plt.tight_layout()
            
            heatmap_file = self.viz_dir / "performance_heatmaps.png"
            plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"性能热力图已保存: {heatmap_file}")
            
        except Exception as e:
            self.logger.error(f"创建性能热力图失败: {str(e)}")
    
    def create_model_comparison(self, results_df: pd.DataFrame) -> None:
        """创建模型对比分析"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle('RandomForest vs SVM 模型对比', fontsize=16, fontweight='bold')
            
            # 1. F1分数对比
            ax1 = axes[0]
            for group in results_df['group_name'].unique():
                group_data = results_df[results_df['group_name'] == group]
                
                rf_data = group_data[group_data['model_name'] == 'RandomForest']
                svm_data = group_data[group_data['model_name'] == 'SVM']
                
                if len(rf_data) > 0 and len(svm_data) > 0:
                    ax1.scatter(rf_data['f1_score'], svm_data['f1_score'], 
                               label=group, alpha=0.7, s=100)
            
            # 添加对角线
            max_val = max(results_df['f1_score'].max(), results_df['f1_score'].max())
            ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='相等线')
            
            ax1.set_xlabel('RandomForest F1分数')
            ax1.set_ylabel('SVM F1分数')
            ax1.set_title('F1分数对比')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 训练时间对比
            ax2 = axes[1]
            rf_times = results_df[results_df['model_name'] == 'RandomForest']['train_time']
            svm_times = results_df[results_df['model_name'] == 'SVM']['train_time']
            
            ax2.boxplot([rf_times, svm_times], labels=['RandomForest', 'SVM'])
            ax2.set_ylabel('训练时间 (秒)')
            ax2.set_title('训练时间对比')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            comparison_file = self.viz_dir / "model_comparison.png"
            plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"模型对比图已保存: {comparison_file}")
            
        except Exception as e:
            self.logger.error(f"创建模型对比失败: {str(e)}")
    
    def create_group_comparison(self, results_df: pd.DataFrame) -> None:
        """创建数据组效果对比"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('纯净数据组效果对比分析', fontsize=16, fontweight='bold')
            
            # 1. 各组平均F1分数
            ax1 = axes[0, 0]
            group_f1 = results_df.groupby('group_name')['f1_score'].mean().sort_values(ascending=False)
            bars1 = ax1.bar(group_f1.index, group_f1.values, alpha=0.7)
            ax1.set_ylabel('平均F1分数')
            ax1.set_title('各数据组平均F1分数')
            ax1.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, value in zip(bars1, group_f1.values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            # 2. 按比例的F1分数分布
            ax2 = axes[0, 1]
            results_df.boxplot(column='f1_score', by='malicious_ratio', ax=ax2)
            ax2.set_xlabel('恶意样本比例')
            ax2.set_ylabel('F1分数')
            ax2.set_title('不同比例下F1分数分布')
            
            # 3. 数据组vs比例热力图
            ax3 = axes[1, 0]
            group_ratio_f1 = results_df.groupby(['group_name', 'malicious_ratio'])['f1_score'].mean().unstack()
            group_ratio_f1.columns = [f'{int(col*100)}%' for col in group_ratio_f1.columns]
            sns.heatmap(group_ratio_f1, annot=True, fmt='.3f', cmap='viridis', ax=ax3)
            ax3.set_title('数据组×比例 F1分数热力图')
            
            # 4. baseline_real vs synthetic对比
            ax4 = axes[1, 1]
            baseline_data = results_df[results_df['group_name'] == 'baseline_real']['f1_score']
            synthetic_groups = ['pure_rewrite', 'pure_strong', 'pure_weak']
            
            comparison_data = [baseline_data]
            labels = ['baseline_real']
            
            for group in synthetic_groups:
                if group in results_df['group_name'].values:
                    group_data = results_df[results_df['group_name'] == group]['f1_score']
                    comparison_data.append(group_data)
                    labels.append(group)
            
            ax4.boxplot(comparison_data, labels=labels)
            ax4.set_ylabel('F1分数')
            ax4.set_title('Baseline vs Pure Synthetic对比')
            ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            group_comparison_file = self.viz_dir / "group_comparison.png"
            plt.savefig(group_comparison_file, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"数据组对比图已保存: {group_comparison_file}")
            
        except Exception as e:
            self.logger.error(f"创建数据组对比失败: {str(e)}")
    
    def generate_comprehensive_analysis(self, all_results: Dict[str, Dict[str, Any]]) -> None:
        """生成综合分析报告"""
        try:
            self.logger.info("生成综合分析报告...")
            
            # 统计摘要
            total_experiments = len(all_results)
            successful_experiments = len([r for r in all_results.values() if 'metrics' in r])
            
            # 性能统计
            f1_scores = [r['metrics']['f1_score'] for r in all_results.values() if 'metrics' in r]
            accuracies = [r['metrics']['accuracy'] for r in all_results.values() if 'metrics' in r]
            
            # 按组统计
            group_stats = {}
            model_stats = {}
            
            for result in all_results.values():
                if 'metrics' not in result:
                    continue
                    
                group_name = result['config_info']['group_name']
                model_name = result['model_name']
                f1_score = result['metrics']['f1_score']
                
                # 按组统计
                if group_name not in group_stats:
                    group_stats[group_name] = {'f1_scores': [], 'count': 0}
                group_stats[group_name]['f1_scores'].append(f1_score)
                group_stats[group_name]['count'] += 1
                
                # 按model统计
                if model_name not in model_stats:
                    model_stats[model_name] = {'f1_scores': [], 'count': 0}
                model_stats[model_name]['f1_scores'].append(f1_score)
                model_stats[model_name]['count'] += 1
            
            # 计算平均值
            for group in group_stats:
                group_stats[group]['avg_f1'] = np.mean(group_stats[group]['f1_scores'])
                group_stats[group]['std_f1'] = np.std(group_stats[group]['f1_scores'])
            
            for model in model_stats:
                model_stats[model]['avg_f1'] = np.mean(model_stats[model]['f1_scores'])
                model_stats[model]['std_f1'] = np.std(model_stats[model]['f1_scores'])
            
            # 找出最佳和最差性能
            best_result = max(all_results.values(), key=lambda x: x['metrics']['f1_score'] if 'metrics' in x else 0)
            worst_result = min(all_results.values(), key=lambda x: x['metrics']['f1_score'] if 'metrics' in x else 1)
            
            # 综合分析
            analysis = {
                'phase4_info': {
                    'date': '2025-07-30',
                    'total_experiments': total_experiments,
                    'successful_experiments': successful_experiments,
                    'success_rate': successful_experiments / total_experiments if total_experiments > 0 else 0
                },
                'performance_summary': {
                    'f1_score_range': [float(min(f1_scores)), float(max(f1_scores))] if f1_scores else [0, 0],
                    'f1_score_mean': float(np.mean(f1_scores)) if f1_scores else 0,
                    'f1_score_std': float(np.std(f1_scores)) if f1_scores else 0,
                    'accuracy_range': [float(min(accuracies)), float(max(accuracies))] if accuracies else [0, 0],
                    'accuracy_mean': float(np.mean(accuracies)) if accuracies else 0
                },
                'best_performance': {
                    'config': best_result.get('config_name', 'unknown'),
                    'model': best_result.get('model_name', 'unknown'),
                    'f1_score': best_result['metrics']['f1_score'] if 'metrics' in best_result else 0,
                    'accuracy': best_result['metrics']['accuracy'] if 'metrics' in best_result else 0
                },
                'worst_performance': {
                    'config': worst_result.get('config_name', 'unknown'),
                    'model': worst_result.get('model_name', 'unknown'),
                    'f1_score': worst_result['metrics']['f1_score'] if 'metrics' in worst_result else 0,
                    'accuracy': worst_result['metrics']['accuracy'] if 'metrics' in worst_result else 0
                },
                'group_analysis': {group: {
                    'avg_f1_score': stats['avg_f1'],
                    'std_f1_score': stats['std_f1'],
                    'experiment_count': stats['count']
                } for group, stats in group_stats.items()},
                'model_analysis': {model: {
                    'avg_f1_score': stats['avg_f1'],
                    'std_f1_score': stats['std_f1'],
                    'experiment_count': stats['count']
                } for model, stats in model_stats.items()},
                'key_findings': [],
                'detailed_results': all_results
            }
            
            # 生成关键发现
            if group_stats:
                best_group = max(group_stats.keys(), key=lambda g: group_stats[g]['avg_f1'])
                worst_group = min(group_stats.keys(), key=lambda g: group_stats[g]['avg_f1'])
                
                analysis['key_findings'].extend([
                    f"最佳数据组: {best_group} (平均F1: {group_stats[best_group]['avg_f1']:.3f})",
                    f"最差数据组: {worst_group} (平均F1: {group_stats[worst_group]['avg_f1']:.3f})"
                ])
            
            if model_stats:
                best_model = max(model_stats.keys(), key=lambda m: model_stats[m]['avg_f1'])
                analysis['key_findings'].append(
                    f"最佳模型: {best_model} (平均F1: {model_stats[best_model]['avg_f1']:.3f})"
                )
            
            # 保存综合分析
            analysis_file = self.analysis_dir / "comprehensive_analysis.json"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"综合分析报告已保存: {analysis_file}")
            
            # 输出关键统计
            self.logger.info("="*40)
            self.logger.info("批次5性能评估结果:")
            self.logger.info(f"✅ 成功实验: {successful_experiments}/{total_experiments}")
            self.logger.info(f"📊 F1分数范围: {analysis['performance_summary']['f1_score_range']}")
            self.logger.info(f"🏆 最佳配置: {analysis['best_performance']['config']} + {analysis['best_performance']['model']}")
            self.logger.info(f"    F1分数: {analysis['best_performance']['f1_score']:.3f}")
            self.logger.info("="*40)
            
        except Exception as e:
            self.logger.error(f"生成综合分析失败: {str(e)}")
            raise
    
    def run_phase4_complete(self) -> bool:
        """执行完整的Phase 4流程"""
        try:
            self.logger.info("="*50)
            self.logger.info("开始执行批次5 Phase 4: 模型训练与独立评估")
            self.logger.info("="*50)
            
            start_time = time.time()
            
            # Step 1: 发现数据集配置
            self.logger.info("Step 1: 发现数据集配置")
            dataset_configs = self.discover_datasets()
            
            # Step 2: 运行所有实验
            self.logger.info("Step 2: 运行所有实验")
            all_results = self.run_all_experiments(dataset_configs)
            
            # Step 3: 创建性能分析
            self.logger.info("Step 3: 创建性能分析")
            self.create_performance_analysis(all_results)
            
            # Step 4: 生成综合分析
            self.logger.info("Step 4: 生成综合分析")
            self.generate_comprehensive_analysis(all_results)
            
            elapsed_time = time.time() - start_time
            
            self.logger.info("="*50)
            self.logger.info("批次5 Phase 4 执行完成!")
            self.logger.info(f"总耗时: {elapsed_time/60:.1f} 分钟")
            self.logger.info(f"完成实验数: {len(all_results)}")
            self.logger.info("结果已保存到: data/batch5/results/")
            self.logger.info("模型已保存到: data/batch5/models/")
            self.logger.info("可视化已保存到: data/batch5/visualizations/performance_analysis/")
            self.logger.info("="*50)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Phase 4执行失败: {str(e)}")
            return False

def main():
    """主函数"""
    # 设置随机种子
    random.seed(2025)
    np.random.seed(2025)
    
    # 设置日志
    logger = setup_logger("batch5_phase4", log_level=logging.INFO)
    
    try:
        # 创建评估器
        evaluator = Batch5Phase4IndependentEvaluation(logger)
        
        # 执行Phase 4
        success = evaluator.run_phase4_complete()
        
        if success:
            logger.info("批次5 Phase 4 successfully completed!")
            return 0
        else:
            logger.error("批次5 Phase 4 failed!")
            return 1
            
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())