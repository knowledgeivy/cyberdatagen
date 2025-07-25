#!/usr/bin/env python3
"""
批次4实验 - Phase 5: 模型训练和评估

对所有20个数据集配置进行完整的机器学习模型训练和评估：
- 2个模型: RandomForest, SVM
- 20个数据集配置 = 40个实验点
- 使用不平衡数据专用评估指标
- 生成全面的性能对比分析

作者: Claude
创建时间: 2025-07-24
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# 项目路径设置
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.cyberdata.utils.logger_config import setup_logger
from src.cyberdata.utils.data_loader import load_csv_data

class Batch4Phase5ModelTrainer:
    """批次4 Phase 5: 模型训练和评估器"""
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logger("batch4_phase5")
        self.project_root = PROJECT_ROOT
        self.batch4_dir = self.project_root / "data" / "batch4_fresh"
        
        # 创建输出目录
        self.models_dir = self.batch4_dir / "phase5_models"
        self.results_dir = self.batch4_dir / "phase5_results" 
        self.visualizations_dir = self.batch4_dir / "phase5_visualizations"
        self.analysis_dir = self.batch4_dir / "phase5_analysis"
        
        for dir_path in [self.models_dir, self.results_dir, self.visualizations_dir, self.analysis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 实验参数
        self.data_groups = ['real_only', 'core', 'inner', 'outer', 'edge']
        self.malicious_ratios = [0.05, 0.10, 0.15, 0.20]
        self.models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                random_state=2025,
                n_jobs=-1,
                class_weight='balanced'  # 处理不平衡数据
            ),
            'SVM': SVC(
                kernel='rbf',
                random_state=2025,
                probability=True,
                class_weight='balanced'  # 处理不平衡数据
            )
        }
        
        # 设置matplotlib字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 随机种子
        self.random_state = 2025
        
        # 计算总实验数
        self.total_experiments = len(self.data_groups) * len(self.malicious_ratios) * len(self.models)
        
        self.logger.info(f"Phase 5初始化完成，将进行 {self.total_experiments} 个实验")
    
    def load_dataset_configuration(self, group: str, ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
        """加载指定配置的数据集"""
        try:
            config_name = f"{group}_{int(ratio*100)}pct"
            datasets_dir = self.batch4_dir / "datasets" / config_name
            
            train_file = datasets_dir / "train.csv"
            test_file = datasets_dir / "test.csv"
            
            if not (train_file.exists() and test_file.exists()):
                raise FileNotFoundError(f"数据文件缺失: {config_name}")
            
            train_df = load_csv_data(train_file, logger=self.logger)
            test_df = load_csv_data(test_file, logger=self.logger)
            
            return train_df, test_df, config_name
            
        except Exception as e:
            self.logger.error(f"加载数据集{group}_{int(ratio*100)}pct失败: {str(e)}")
            raise
    
    def load_embeddings(self, config_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载对应的embedding向量"""
        try:
            embeddings_dir = self.batch4_dir / "phase4_embeddings"
            
            train_emb_file = embeddings_dir / f"{config_name}_train_embeddings.npy"
            test_emb_file = embeddings_dir / f"{config_name}_test_embeddings.npy"
            
            if not (train_emb_file.exists() and test_emb_file.exists()):
                raise FileNotFoundError(f"Embedding文件缺失: {config_name}")
            
            train_embeddings = np.load(train_emb_file)
            test_embeddings = np.load(test_emb_file)
            
            return train_embeddings, test_embeddings
            
        except Exception as e:
            self.logger.error(f"加载embedding{config_name}失败: {str(e)}")
            raise
    
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """计算全面的评估指标"""
        try:
            metrics = {}
            
            # 基础指标
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, pos_label=1)
            metrics['recall'] = recall_score(y_true, y_pred, pos_label=1)
            metrics['f1_score'] = f1_score(y_true, y_pred, pos_label=1)
            
            # 不平衡数据专用指标
            metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
            metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
            metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
            
            # 混淆矩阵
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics['true_negative'] = int(tn)
            metrics['false_positive'] = int(fp)
            metrics['false_negative'] = int(fn)
            metrics['true_positive'] = int(tp)
            
            # 导出率
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # 概率相关指标
            if y_pred_proba is not None:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                    
                    # PR-AUC
                    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
                    metrics['pr_auc'] = auc(recall_curve, precision_curve)
                except Exception as e:
                    self.logger.warning(f"计算AUC指标失败: {str(e)}")
                    metrics['roc_auc'] = 0.0
                    metrics['pr_auc'] = 0.0
            
            # 数据集特征
            metrics['total_samples'] = len(y_true)
            metrics['positive_samples'] = int(np.sum(y_true))
            metrics['negative_samples'] = int(len(y_true) - np.sum(y_true))
            metrics['positive_ratio'] = float(np.mean(y_true))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"计算评估指标失败: {str(e)}")
            raise
    
    def train_and_evaluate_model(self, model_name: str, model, X_train: np.ndarray, 
                               y_train: np.ndarray, X_test: np.ndarray, 
                               y_test: np.ndarray, config_name: str) -> Dict[str, Any]:
        """训练和评估单个模型"""
        try:
            self.logger.info(f"  训练{model_name}模型...")
            
            # 记录训练开始时间
            train_start_time = time.time()
            
            # 训练模型
            model.fit(X_train, y_train)
            
            training_time = time.time() - train_start_time
            
            # 预测
            test_start_time = time.time()
            y_pred = model.predict(X_test)
            
            # 获取预测概率
            try:
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_pred_proba = model.decision_function(X_test)
                else:
                    y_pred_proba = None
            except:
                y_pred_proba = None
            
            prediction_time = time.time() - test_start_time
            
            # 计算指标
            metrics = self.calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)
            
            # 添加性能指标
            metrics['training_time'] = training_time
            metrics['prediction_time'] = prediction_time
            metrics['samples_per_second'] = len(X_test) / prediction_time if prediction_time > 0 else 0
            
            # 交叉验证
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
                metrics['cv_f1_mean'] = float(np.mean(cv_scores))
                metrics['cv_f1_std'] = float(np.std(cv_scores))
            except Exception as e:
                self.logger.warning(f"交叉验证失败: {str(e)}")
                metrics['cv_f1_mean'] = 0.0
                metrics['cv_f1_std'] = 0.0
            
            # 模型信息
            result = {
                'model_name': model_name,
                'config_name': config_name,
                'metrics': metrics,
                'model_params': model.get_params(),
                'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.logger.info(f"  ✅ {model_name}: F1={metrics['f1_score']:.3f}, AUC={metrics.get('roc_auc', 0):.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"训练{model_name}模型失败: {str(e)}")
            raise
    
    def run_single_experiment(self, group: str, ratio: float) -> Dict[str, Any]:
        """运行单个数据配置的完整实验"""
        try:
            config_name = f"{group}_{int(ratio*100)}pct"
            self.logger.info(f"开始实验: {config_name}")
            
            # 加载数据
            train_df, test_df, _ = self.load_dataset_configuration(group, ratio)
            
            # 加载embeddings
            X_train, X_test = self.load_embeddings(config_name)
            y_train = train_df['label'].values
            y_test = test_df['label'].values
            
            # 验证数据一致性
            if len(X_train) != len(y_train) or len(X_test) != len(y_test):
                raise ValueError(f"数据维度不匹配: {config_name}")
            
            experiment_results = {
                'config_name': config_name,
                'group': group,
                'malicious_ratio': ratio,
                'data_info': {
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'train_malicious': int(np.sum(y_train)),
                    'test_malicious': int(np.sum(y_test)),
                    'embedding_dimension': X_train.shape[1]
                },
                'model_results': {}
            }
            
            # 训练所有模型
            for model_name, model in self.models.items():
                model_result = self.train_and_evaluate_model(
                    model_name, model, X_train, y_train, X_test, y_test, config_name
                )
                experiment_results['model_results'][model_name] = model_result
            
            # 保存实验结果
            result_file = self.results_dir / f"{config_name}_results.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(experiment_results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"✅ {config_name}: 实验完成")
            
            return experiment_results
            
        except Exception as e:
            self.logger.error(f"实验{group}_{int(ratio*100)}pct失败: {str(e)}")
            raise
    
    def create_performance_visualizations(self, all_results: List[Dict[str, Any]]) -> None:
        """创建性能对比可视化"""
        try:
            self.logger.info("创建性能对比可视化...")
            
            # 准备数据
            vis_data = []
            for result in all_results:
                for model_name, model_result in result['model_results'].items():
                    metrics = model_result['metrics']
                    vis_data.append({
                        'config': result['config_name'],
                        'group': result['group'],
                        'ratio': f"{int(result['malicious_ratio']*100)}%",
                        'model': model_name,
                        'f1_score': metrics['f1_score'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'roc_auc': metrics.get('roc_auc', 0),
                        'accuracy': metrics['accuracy']
                    })
            
            df_vis = pd.DataFrame(vis_data)
            
            # 1. F1 Score对比
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # F1 Score by Group
            pivot_f1_group = df_vis.pivot_table(values='f1_score', index='group', columns='model', aggfunc='mean')
            sns.heatmap(pivot_f1_group, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0,0])
            axes[0,0].set_title('Average F1 Score by Data Group')
            axes[0,0].set_xlabel('Model')
            axes[0,0].set_ylabel('Data Group')
            
            # F1 Score by Ratio
            pivot_f1_ratio = df_vis.pivot_table(values='f1_score', index='ratio', columns='model', aggfunc='mean')
            sns.heatmap(pivot_f1_ratio, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0,1])
            axes[0,1].set_title('Average F1 Score by Malicious Ratio')
            axes[0,1].set_xlabel('Model')
            axes[0,1].set_ylabel('Malicious Ratio')
            
            # ROC AUC对比
            pivot_auc_group = df_vis.pivot_table(values='roc_auc', index='group', columns='model', aggfunc='mean')
            sns.heatmap(pivot_auc_group, annot=True, fmt='.3f', cmap='Blues', ax=axes[1,0])
            axes[1,0].set_title('Average ROC AUC by Data Group')
            axes[1,0].set_xlabel('Model')
            axes[1,0].set_ylabel('Data Group')
            
            # 精确度对比
            pivot_prec_group = df_vis.pivot_table(values='precision', index='group', columns='model', aggfunc='mean')
            sns.heatmap(pivot_prec_group, annot=True, fmt='.3f', cmap='Greens', ax=axes[1,1])
            axes[1,1].set_title('Average Precision by Data Group')
            axes[1,1].set_xlabel('Model')
            axes[1,1].set_ylabel('Data Group')
            
            plt.tight_layout()
            perf_file = self.visualizations_dir / "performance_heatmaps.png"
            plt.savefig(perf_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 详细性能对比
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # F1 Score分布
            sns.boxplot(data=df_vis, x='group', y='f1_score', hue='model', ax=axes[0,0])
            axes[0,0].set_title('F1 Score Distribution by Group')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # 按比例的F1 Score
            sns.lineplot(data=df_vis, x='ratio', y='f1_score', hue='model', marker='o', ax=axes[0,1])
            axes[0,1].set_title('F1 Score vs Malicious Ratio')
            
            # ROC AUC分布
            sns.boxplot(data=df_vis, x='group', y='roc_auc', hue='model', ax=axes[1,0])
            axes[1,0].set_title('ROC AUC Distribution by Group')
            axes[1,0].tick_params(axis='x', rotation=45)
            
            # 精确度vs召回率
            sns.scatterplot(data=df_vis, x='precision', y='recall', hue='model', style='group', s=100, ax=axes[1,1])
            axes[1,1].set_title('Precision vs Recall')
            axes[1,1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            
            plt.tight_layout()
            detail_file = self.visualizations_dir / "performance_details.png"
            plt.savefig(detail_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ 性能可视化已保存: {perf_file}, {detail_file}")
            
        except Exception as e:
            self.logger.error(f"创建性能可视化失败: {str(e)}")
    
    def generate_comprehensive_analysis(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成综合分析报告"""
        try:
            self.logger.info("生成综合分析报告...")
            
            analysis = {
                'experiment_summary': {
                    'total_experiments': len(all_results) * len(self.models),
                    'data_groups': self.data_groups,
                    'malicious_ratios': [f"{int(r*100)}%" for r in self.malicious_ratios],
                    'models': list(self.models.keys()),
                    'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'performance_analysis': {},
                'model_comparison': {},
                'data_group_analysis': {},
                'ratio_effect_analysis': {},
                'key_findings': []
            }
            
            # 收集所有指标
            all_metrics = []
            for result in all_results:
                for model_name, model_result in result['model_results'].items():
                    metrics = model_result['metrics'].copy()
                    metrics.update({
                        'config_name': result['config_name'],
                        'group': result['group'],
                        'malicious_ratio': result['malicious_ratio'],
                        'model_name': model_name
                    })
                    all_metrics.append(metrics)
            
            df_metrics = pd.DataFrame(all_metrics)
            
            # 模型整体性能对比
            model_performance = {}
            for model in self.models.keys():
                model_data = df_metrics[df_metrics['model_name'] == model]
                model_performance[model] = {
                    'avg_f1_score': float(model_data['f1_score'].mean()),
                    'std_f1_score': float(model_data['f1_score'].std()),
                    'avg_roc_auc': float(model_data['roc_auc'].mean()),
                    'avg_precision': float(model_data['precision'].mean()),
                    'avg_recall': float(model_data['recall'].mean()),
                    'avg_accuracy': float(model_data['accuracy'].mean()),
                    'avg_training_time': float(model_data['training_time'].mean())
                }
            
            analysis['model_comparison'] = model_performance
            
            # 数据组效果分析
            group_analysis = {}
            for group in self.data_groups:
                group_data = df_metrics[df_metrics['group'] == group]
                group_analysis[group] = {
                    'avg_f1_score': float(group_data['f1_score'].mean()),
                    'std_f1_score': float(group_data['f1_score'].std()),
                    'best_f1_config': group_data.loc[group_data['f1_score'].idxmax(), 'config_name'],
                    'best_f1_score': float(group_data['f1_score'].max()),
                    'avg_roc_auc': float(group_data['roc_auc'].mean())
                }
            
            analysis['data_group_analysis'] = group_analysis
            
            # 比例效果分析
            ratio_analysis = {}
            for ratio in self.malicious_ratios:
                ratio_str = f"{int(ratio*100)}%"
                ratio_data = df_metrics[df_metrics['malicious_ratio'] == ratio]
                ratio_analysis[ratio_str] = {
                    'avg_f1_score': float(ratio_data['f1_score'].mean()),
                    'avg_precision': float(ratio_data['precision'].mean()),
                    'avg_recall': float(ratio_data['recall'].mean()),
                    'best_config': ratio_data.loc[ratio_data['f1_score'].idxmax(), 'config_name'],
                    'best_f1_score': float(ratio_data['f1_score'].max())
                }
            
            analysis['ratio_effect_analysis'] = ratio_analysis
            
            # 关键发现
            best_overall = df_metrics.loc[df_metrics['f1_score'].idxmax()]
            worst_overall = df_metrics.loc[df_metrics['f1_score'].idxmin()]
            
            analysis['key_findings'] = [
                f"最佳整体性能: {best_overall['config_name']} + {best_overall['model_name']} (F1: {best_overall['f1_score']:.3f})",
                f"最差整体性能: {worst_overall['config_name']} + {worst_overall['model_name']} (F1: {worst_overall['f1_score']:.3f})",
                f"性能范围: F1 Score从 {df_metrics['f1_score'].min():.3f} 到 {df_metrics['f1_score'].max():.3f}",
                f"平均性能: F1 Score平均值 {df_metrics['f1_score'].mean():.3f} ± {df_metrics['f1_score'].std():.3f}"
            ]
            
            # 合成数据效果分析
            real_only_perf = df_metrics[df_metrics['group'] == 'real_only']['f1_score'].mean()
            synthetic_groups = ['core', 'inner', 'outer', 'edge']
            synthetic_perf = df_metrics[df_metrics['group'].isin(synthetic_groups)]['f1_score'].mean()
            
            if synthetic_perf > real_only_perf:
                improvement = ((synthetic_perf - real_only_perf) / real_only_perf) * 100
                analysis['key_findings'].append(f"合成数据平均提升性能 {improvement:.1f}% (F1: {real_only_perf:.3f} → {synthetic_perf:.3f})")
            else:
                decline = ((real_only_perf - synthetic_perf) / real_only_perf) * 100
                analysis['key_findings'].append(f"合成数据平均降低性能 {decline:.1f}% (F1: {real_only_perf:.3f} → {synthetic_perf:.3f})")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"生成综合分析失败: {str(e)}")
            raise
    
    def run_phase5_complete(self) -> bool:
        """执行完整的Phase 5流程"""
        try:
            self.logger.info("="*50)
            self.logger.info("开始执行批次4 Phase 5: 模型训练和评估")
            self.logger.info("="*50)
            
            start_time = time.time()
            
            all_results = []
            experiment_count = 0
            
            # 执行所有实验
            self.logger.info(f"开始执行 {self.total_experiments} 个实验...")
            
            for group in self.data_groups:
                for ratio in self.malicious_ratios:
                    experiment_count += 1
                    self.logger.info(f"实验 {experiment_count}/{len(self.data_groups) * len(self.malicious_ratios)}: {group}_{int(ratio*100)}pct")
                    
                    try:
                        experiment_result = self.run_single_experiment(group, ratio)
                        all_results.append(experiment_result)
                    except Exception as e:
                        self.logger.error(f"实验{group}_{int(ratio*100)}pct失败，跳过: {str(e)}")
                        continue
            
            if not all_results:
                raise ValueError("没有成功完成的实验")
            
            # 创建性能可视化
            self.logger.info("创建性能可视化...")
            self.create_performance_visualizations(all_results)
            
            # 生成综合分析
            self.logger.info("生成综合分析...")
            comprehensive_analysis = self.generate_comprehensive_analysis(all_results)
            
            # 保存所有结果
            all_results_file = self.analysis_dir / "all_experiment_results.json"
            with open(all_results_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
                
            analysis_file = self.analysis_dir / "comprehensive_analysis.json"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_analysis, f, ensure_ascii=False, indent=2)
            
            # 生成Phase 5摘要
            phase5_summary = {
                'phase5_info': {
                    'date': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'total_experiments_planned': self.total_experiments,
                    'total_experiments_completed': len(all_results) * len(self.models),
                    'models_used': list(self.models.keys()),
                    'data_configurations': len(all_results),
                    'success_rate': len(all_results) / (len(self.data_groups) * len(self.malicious_ratios))
                },
                'performance_summary': comprehensive_analysis['model_comparison'],
                'best_configurations': comprehensive_analysis.get('key_findings', []),
                'output_locations': {
                    'models': str(self.models_dir),
                    'results': str(self.results_dir),
                    'visualizations': str(self.visualizations_dir),
                    'analysis': str(self.analysis_dir)
                }
            }
            
            summary_file = self.analysis_dir / "phase5_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(phase5_summary, f, ensure_ascii=False, indent=2)
            
            elapsed_time = time.time() - start_time
            
            self.logger.info("="*50)
            self.logger.info("批次4 Phase 5 执行完成!")
            self.logger.info(f"总耗时: {elapsed_time/60:.1f} 分钟")
            self.logger.info(f"完成实验: {len(all_results) * len(self.models)}/{self.total_experiments}")
            self.logger.info(f"成功率: {len(all_results) / (len(self.data_groups) * len(self.malicious_ratios)) * 100:.1f}%")
            
            # 显示关键发现
            if 'key_findings' in comprehensive_analysis:
                self.logger.info("关键发现:")
                for finding in comprehensive_analysis['key_findings']:
                    self.logger.info(f"- {finding}")
            
            self.logger.info("输出位置:")
            self.logger.info(f"- 模型: {self.models_dir}")
            self.logger.info(f"- 结果: {self.results_dir}")
            self.logger.info(f"- 可视化: {self.visualizations_dir}")
            self.logger.info(f"- 分析: {self.analysis_dir}")
            self.logger.info("="*50)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Phase 5执行失败: {str(e)}")
            return False

def main():
    """主函数"""
    # 设置随机种子
    random.seed(2025)
    np.random.seed(2025)
    
    # 设置日志
    logger = setup_logger("batch4_phase5", log_level=logging.INFO)
    
    try:
        # 创建训练器
        trainer = Batch4Phase5ModelTrainer(logger)
        
        # 执行Phase 5
        success = trainer.run_phase5_complete()
        
        if success:
            logger.info("批次4 Phase 5 successfully completed!")
            return 0
        else:
            logger.error("批次4 Phase 5 failed!")
            return 1
            
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())