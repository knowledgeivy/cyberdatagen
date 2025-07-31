#!/usr/bin/env python3
"""
Batch 6 Phase 7: Model Training and Performance Evaluation
==========================================================

Trains and evaluates models on batch6 pure datasets to quantify synthetic data quality improvements.
Compares batch6 results with batch5 to validate the improved generation methodology.

Key objectives:
- Train RandomForest + SVM models on 16 pure datasets  
- Evaluate on fixed independent test set
- Quantify improvement from batch5 baseline
- Target: Pure Synthetic F1 Score > 0.75 (from 0.6)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class Batch6Phase7ModelEvaluator:
    """Model training and evaluation for batch6 pure dataset performance validation"""
    
    def __init__(self):
        self.setup_paths()
        self.setup_logging()
        self.load_configurations()
        
    def setup_paths(self) -> None:
        """Setup directory structure for phase 7"""
        # Use absolute paths to avoid working directory issues
        current_dir = Path(__file__).parent.parent.parent.parent  # Go up to project root
        self.base_dir = current_dir / "data" / "batch6"
        self.input_dirs = {
            'pure_datasets': self.base_dir / "pure_datasets",
            'test_data': current_dir / "data" / "batch4_fresh",
            'batch5_results': current_dir / "data" / "batch5" / "results"  # For comparison
        }
        
        # Output directories - consistent naming with other phases
        self.output_dir = self.base_dir / "phase7_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for subdir in ['models', 'predictions', 'plots', 'analysis']:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)
            
    def setup_logging(self) -> None:
        """Setup logging configuration"""
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"batch6_phase7_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_configurations(self) -> None:
        """Load model training and evaluation configurations"""
        self.config = {
            'models': {
                'RandomForest': {
                    'classifier': RandomForestClassifier,
                    'params': {
                        'n_estimators': 100,
                        'max_depth': 20,
                        'random_state': 42,
                        'n_jobs': -1  # Already enabled for parallel processing
                    }
                },
                'SVM': {
                    'classifier': SVC,
                    'params': {
                        'kernel': 'rbf',
                        'C': 1.0,
                        'random_state': 42,
                        'probability': True,
                        'cache_size': 1000  # Increase cache for better performance
                    }
                },
                'DeepLearning': {
                    'classifier': MLPClassifier,
                    'params': {
                        'hidden_layer_sizes': (100, 50),
                        'activation': 'relu',
                        'solver': 'adam',
                        'alpha': 0.001,
                        'batch_size': 'auto',
                        'learning_rate': 'constant',
                        'learning_rate_init': 0.001,
                        'max_iter': 500,
                        'random_state': 42,
                        'early_stopping': True,
                        'validation_fraction': 0.1,
                        'n_iter_no_change': 10
                    }
                }
            },
            'vectorizer': {
                'max_features': 10000,
                'min_df': 2,
                'max_df': 0.95,
                'stop_words': 'english',
                'ngram_range': (1, 2)
            },
            'evaluation_metrics': ['accuracy', 'precision', 'recall', 'f1'],
            'dataset_types': ['baseline_real', 'pure_original', 'pure_strong', 'pure_weak'],
            'malicious_ratios': [5, 10, 15, 20],
            'random_seed': 42
        }
        
        self.logger.info(f"Phase 7 configuration loaded")
        
    def load_test_data(self) -> pd.DataFrame:
        """Load fixed independent test set"""
        try:
            self.logger.info("Loading independent test set...")
            
            test_file = self.input_dirs['test_data'] / "raw_test_set.csv"
            if not test_file.exists():
                raise FileNotFoundError(f"Test set not found: {test_file}")
            
            test_data = pd.read_csv(test_file)
            
            # Ensure required columns exist
            required_columns = ['subject', 'body', 'label']
            missing_columns = [col for col in required_columns if col not in test_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in test data: {missing_columns}")
            
            self.logger.info(f"✅ Test data loaded: {len(test_data)} samples")
            self.logger.info(f"Test set distribution: {test_data['label'].value_counts().to_dict()}")
            
            return test_data
            
        except Exception as e:
            self.logger.error(f"Failed to load test data: {str(e)}")
            raise
            
    def load_training_datasets(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load all pure training datasets"""
        try:
            self.logger.info("Loading pure training datasets...")
            
            training_data = defaultdict(dict)
            
            for dataset_type in self.config['dataset_types']:
                for ratio in self.config['malicious_ratios']:
                    dataset_name = f"{dataset_type}_{ratio}pct"
                    dataset_dir = self.input_dirs['pure_datasets'] / dataset_name
                    dataset_file = dataset_dir / f"{dataset_name}_dataset.csv"
                    
                    if dataset_file.exists():
                        dataset = pd.read_csv(dataset_file)
                        training_data[dataset_type][f"{ratio}pct"] = dataset
                        self.logger.info(f"✅ Loaded {dataset_name}: {len(dataset)} samples")
                    else:
                        self.logger.warning(f"⚠️ Dataset not found: {dataset_file}")
            
            total_datasets = sum(len(ratios) for ratios in training_data.values())
            self.logger.info(f"✅ Loaded {total_datasets} training datasets")
            
            return dict(training_data)
            
        except Exception as e:
            self.logger.error(f"Failed to load training datasets: {str(e)}")
            raise
            
    def prepare_text_features(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
        """Prepare TF-IDF features for training and testing"""
        try:
            # Combine subject and body for feature extraction
            train_texts = (train_data['subject'].fillna('') + ' ' + train_data['body'].fillna('')).values
            test_texts = (test_data['subject'].fillna('') + ' ' + test_data['body'].fillna('')).values
            
            # Create and fit vectorizer
            vectorizer = TfidfVectorizer(**self.config['vectorizer'])
            
            # Transform data
            X_train = vectorizer.fit_transform(train_texts)
            X_test = vectorizer.transform(test_texts)
            
            return X_train, X_test, vectorizer
            
        except Exception as e:
            self.logger.error(f"Failed to prepare text features: {str(e)}")
            raise
            
    def train_single_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Train a single model on training data"""
        try:
            model_config = self.config['models'][model_name]
            model = model_config['classifier'](**model_config['params'])
            
            # Train model
            model.fit(X_train, y_train)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to train {model_name} model: {str(e)}")
            raise
            
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str, dataset_name: str) -> Dict[str, float]:
        """Evaluate model performance on test set"""
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            
            # Add additional metadata
            metrics.update({
                'model_name': model_name,
                'dataset_name': dataset_name,
                'test_samples': len(y_test),
                'positive_predictions': int(y_pred.sum()),
                'true_positives': int(y_test.sum())
            })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate model {model_name} on {dataset_name}: {str(e)}")
            raise
            
    def run_training_evaluation_loop(self, training_data: Dict[str, Dict[str, pd.DataFrame]], 
                                   test_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Run complete training and evaluation loop for all datasets and models"""
        try:
            self.logger.info("Starting training and evaluation loop...")
            
            all_results = []
            y_test = test_data['label'].values
            
            total_experiments = len(self.config['dataset_types']) * len(self.config['malicious_ratios']) * len(self.config['models'])
            experiment_count = 0
            
            for dataset_type in self.config['dataset_types']:
                for ratio_name, train_dataset in training_data[dataset_type].items():
                    dataset_name = f"{dataset_type}_{ratio_name}"
                    
                    self.logger.info(f"Processing dataset: {dataset_name}")
                    
                    # Prepare training labels
                    y_train = train_dataset['label'].values
                    
                    # Prepare text features
                    X_train, X_test, vectorizer = self.prepare_text_features(train_dataset, test_data)
                    
                    for model_name in self.config['models'].keys():
                        experiment_count += 1
                        self.logger.info(f"[{experiment_count}/{total_experiments}] Training {model_name} on {dataset_name}")
                        
                        try:
                            # Train model
                            model = self.train_single_model(model_name, X_train, y_train)
                            
                            # Evaluate model
                            metrics = self.evaluate_model(model, X_test, y_test, model_name, dataset_name)
                            
                            # Add experiment metadata
                            metrics.update({
                                'dataset_type': dataset_type,
                                'malicious_ratio': ratio_name,
                                'training_samples': len(train_dataset),
                                'feature_count': X_train.shape[1],
                                'experiment_timestamp': datetime.now().isoformat()
                            })
                            
                            all_results.append(metrics)
                            
                            # Save model
                            model_dir = self.output_dir / "models" / dataset_name
                            model_dir.mkdir(parents=True, exist_ok=True)
                            model_file = model_dir / f"{model_name}_model.joblib"
                            joblib.dump(model, model_file)
                            
                            # Save vectorizer
                            vectorizer_file = model_dir / f"{model_name}_vectorizer.joblib"
                            joblib.dump(vectorizer, vectorizer_file)
                            
                            self.logger.info(f"✅ {model_name} on {dataset_name}: F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}")
                            
                        except Exception as e:
                            self.logger.error(f"❌ Failed experiment: {model_name} on {dataset_name} - {str(e)}")
                            continue
            
            self.logger.info(f"✅ Training and evaluation completed: {len(all_results)}/{total_experiments} experiments successful")
            return all_results
            
        except Exception as e:
            self.logger.error(f"Failed training evaluation loop: {str(e)}")
            raise
            
    def analyze_results(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze and summarize experimental results"""
        try:
            self.logger.info("Analyzing experimental results...")
            
            results_df = pd.DataFrame(all_results)
            
            analysis = {
                'overall_summary': {
                    'total_experiments': len(results_df),
                    'successful_experiments': len(results_df),
                    'average_f1': results_df['f1'].mean(),
                    'average_accuracy': results_df['accuracy'].mean()
                },
                'by_dataset_type': {},
                'by_model': {},
                'best_performers': {},
                'batch6_targets': {}
            }
            
            # Analysis by dataset type
            for dataset_type in self.config['dataset_types']:
                type_results = results_df[results_df['dataset_type'] == dataset_type]
                if len(type_results) > 0:
                    analysis['by_dataset_type'][dataset_type] = {
                        'experiments': len(type_results),
                        'avg_f1': type_results['f1'].mean(),
                        'avg_accuracy': type_results['accuracy'].mean(),
                        'best_f1': type_results['f1'].max(),
                        'worst_f1': type_results['f1'].min()
                    }
            
            # Analysis by model
            for model_name in self.config['models'].keys():
                model_results = results_df[results_df['model_name'] == model_name]
                if len(model_results) > 0:
                    analysis['by_model'][model_name] = {
                        'experiments': len(model_results),
                        'avg_f1': model_results['f1'].mean(),
                        'avg_accuracy': model_results['accuracy'].mean(),
                        'best_f1': model_results['f1'].max()
                    }
            
            # Best performers
            best_f1_idx = results_df['f1'].idxmax()
            best_acc_idx = results_df['accuracy'].idxmax()
            
            analysis['best_performers'] = {
                'best_f1': {
                    'dataset': results_df.loc[best_f1_idx, 'dataset_name'],
                    'model': results_df.loc[best_f1_idx, 'model_name'],
                    'f1_score': results_df.loc[best_f1_idx, 'f1'],
                    'accuracy': results_df.loc[best_f1_idx, 'accuracy']
                },
                'best_accuracy': {
                    'dataset': results_df.loc[best_acc_idx, 'dataset_name'],
                    'model': results_df.loc[best_acc_idx, 'model_name'],
                    'f1_score': results_df.loc[best_acc_idx, 'f1'],
                    'accuracy': results_df.loc[best_acc_idx, 'accuracy']
                }
            }
            
            # Batch6 target analysis
            synthetic_results = results_df[results_df['dataset_type'].str.contains('pure_')]
            real_baseline = results_df[results_df['dataset_type'] == 'baseline_real']
            
            if len(synthetic_results) > 0 and len(real_baseline) > 0:
                synthetic_best_f1 = synthetic_results['f1'].max()
                real_avg_f1 = real_baseline['f1'].mean()
                
                analysis['batch6_targets'] = {
                    'synthetic_best_f1': synthetic_best_f1,
                    'real_baseline_avg_f1': real_avg_f1,
                    'improvement_gap': real_avg_f1 - synthetic_best_f1,
                    'target_75_achieved': synthetic_best_f1 >= 0.75,
                    'gap_reduction': (0.8 - synthetic_best_f1) / (0.8 - 0.6) if synthetic_best_f1 < 0.8 else 1.0  # Progress from batch5 baseline
                }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze results: {str(e)}")
            raise
            
    def create_performance_visualizations(self, all_results: List[Dict[str, Any]], 
                                        analysis: Dict[str, Any]) -> None:
        """Create comprehensive performance visualization plots"""
        try:
            self.logger.info("Creating performance visualizations...")
            
            results_df = pd.DataFrame(all_results)
            
            # Set up the plotting style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. F1 Score Comparison by Dataset Type and Model
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Batch 6 Model Performance Analysis', fontsize=16, fontweight='bold')
            
            # F1 by dataset type
            ax1 = axes[0, 0]
            dataset_f1 = results_df.groupby('dataset_type')['f1'].agg(['mean', 'max', 'min'])
            dataset_f1.plot(kind='bar', ax=ax1)
            ax1.set_title('F1 Score by Dataset Type')
            ax1.set_ylabel('F1 Score')
            ax1.set_xlabel('Dataset Type')
            ax1.legend(['Mean', 'Max', 'Min'])
            ax1.tick_params(axis='x', rotation=45)
            
            # Add target line
            ax1.axhline(y=0.75, color='red', linestyle='--', alpha=0.7, label='Target (0.75)')
            
            # F1 by model
            ax2 = axes[0, 1]
            model_f1 = results_df.groupby('model_name')['f1'].agg(['mean', 'max', 'min'])
            model_f1.plot(kind='bar', ax=ax2)
            ax2.set_title('F1 Score by Model Type')
            ax2.set_ylabel('F1 Score')
            ax2.set_xlabel('Model Type')
            ax2.legend(['Mean', 'Max', 'Min'])
            ax2.tick_params(axis='x', rotation=0)
            
            # Performance by malicious ratio
            ax3 = axes[1, 0]
            ratio_performance = results_df.groupby('malicious_ratio')['f1'].mean()
            ratio_performance.plot(kind='bar', ax=ax3, color='skyblue')
            ax3.set_title('Average F1 Score by Malicious Data Ratio')
            ax3.set_ylabel('F1 Score')
            ax3.set_xlabel('Malicious Ratio')
            ax3.tick_params(axis='x', rotation=0)
            
            # Heatmap of performance matrix
            ax4 = axes[1, 1]
            pivot_table = results_df.pivot_table(values='f1', index='dataset_type', 
                                                columns='model_name', aggfunc='mean')
            sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax4)
            ax4.set_title('F1 Score Heatmap: Dataset × Model')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'plots' / 'batch6_performance_overview.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Detailed comparison plot
            fig, ax = plt.subplots(1, 1, figsize=(14, 8))
            
            # Box plot showing F1 distribution by dataset type
            results_df.boxplot(column='f1', by='dataset_type', ax=ax)
            ax.set_title('F1 Score Distribution by Dataset Type', fontsize=14)
            ax.set_ylabel('F1 Score')
            ax.set_xlabel('Dataset Type')
            
            # Add target and baseline lines
            ax.axhline(y=0.75, color='red', linestyle='--', alpha=0.8, label='Batch6 Target (0.75)')
            ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.8, label='Batch5 Baseline (0.60)')
            ax.legend()
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'plots' / 'batch6_f1_distribution.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("✅ Performance visualizations created")
            
        except Exception as e:
            self.logger.error(f"Failed to create visualizations: {str(e)}")
            
    def convert_to_json_serializable(self, obj: Any) -> Any:
        """Convert numpy/pandas types to JSON serializable types"""
        if isinstance(obj, dict):
            return {key: self.convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def save_results(self, all_results: List[Dict[str, Any]], analysis: Dict[str, Any]) -> None:
        """Save all results and analysis to disk"""
        try:
            self.logger.info("Saving results and analysis...")
            
            # Save detailed results
            results_df = pd.DataFrame(all_results)
            results_file = self.output_dir / "performance_matrix.csv"
            results_df.to_csv(results_file, index=False)
            
            # Convert analysis to JSON serializable format
            json_safe_analysis = self.convert_to_json_serializable(analysis)
            
            # Save analysis summary
            analysis_file = self.output_dir / "analysis_summary.json"
            with open(analysis_file, 'w') as f:
                json.dump(json_safe_analysis, f, indent=2)
            
            # Create comprehensive report
            report_lines = [
                "# Batch 6 Phase 7: Model Training and Evaluation Results",
                f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
                "",
                "## Executive Summary",
                f"- **Total Experiments**: {analysis['overall_summary']['total_experiments']}",
                f"- **Average F1 Score**: {analysis['overall_summary']['average_f1']:.4f}",
                f"- **Average Accuracy**: {analysis['overall_summary']['average_accuracy']:.4f}",
                "",
                "## Key Findings"
            ]
            
            # Add batch6 target analysis
            if 'batch6_targets' in analysis:
                targets = analysis['batch6_targets']
                report_lines.extend([
                    f"- **Best Synthetic F1**: {targets['synthetic_best_f1']:.4f}",
                    f"- **Real Baseline Avg F1**: {targets['real_baseline_avg_f1']:.4f}",
                    f"- **Target 0.75 Achieved**: {'✅ YES' if targets['target_75_achieved'] else '❌ NO'}",
                    f"- **Improvement Gap**: {targets['improvement_gap']:.4f}",
                    ""
                ])
            
            # Add performance by dataset type
            report_lines.append("## Performance by Dataset Type")
            for dataset_type, stats in analysis['by_dataset_type'].items():
                report_lines.append(f"- **{dataset_type}**: Avg F1 = {stats['avg_f1']:.4f}, Best F1 = {stats['best_f1']:.4f}")
            
            report_lines.extend([
                "",
                "## Best Performers",
                f"- **Best F1**: {analysis['best_performers']['best_f1']['dataset']} + {analysis['best_performers']['best_f1']['model']} (F1: {analysis['best_performers']['best_f1']['f1_score']:.4f})",
                f"- **Best Accuracy**: {analysis['best_performers']['best_accuracy']['dataset']} + {analysis['best_performers']['best_accuracy']['model']} (Acc: {analysis['best_performers']['best_accuracy']['accuracy']:.4f})",
                "",
                f"## Files Generated",
                f"- Performance Matrix: `performance_matrix.csv`",
                f"- Analysis Summary: `analysis_summary.json`",
                f"- Model Files: `models/` directory",
                f"- Visualizations: `plots/` directory"
            ])
            
            report_content = "\n".join(report_lines)
            report_file = self.output_dir / "comprehensive_report.md"
            with open(report_file, 'w') as f:
                f.write(report_content)
            
            # Save phase completion summary
            completion_summary = {
                'phase7_completion': {
                    'status': 'completed',
                    'completion_date': datetime.now().isoformat(),
                    'total_experiments': len(all_results),
                    'successful_experiments': len([r for r in all_results if r.get('f1', 0) > 0])
                },
                'results_summary': analysis['overall_summary'],
                'batch6_performance': analysis.get('batch6_targets', {}),
                'output_files': {
                    'performance_matrix': str(results_file),
                    'analysis_summary': str(analysis_file),
                    'comprehensive_report': str(report_file)
                }
            }
            
            # Convert completion summary to JSON serializable format
            json_safe_completion_summary = self.convert_to_json_serializable(completion_summary)
            
            summary_file = self.output_dir / "phase7_completion_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(json_safe_completion_summary, f, indent=2)
            
            self.logger.info(f"✅ Results saved to: {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            raise
            
    def run_phase7(self) -> None:
        """Execute complete Phase 7 model training and evaluation"""
        try:
            self.logger.info("🚀 Starting Batch 6 Phase 7: Model Training and Evaluation")
            start_time = datetime.now()
            
            # Step 1: Load test data
            test_data = self.load_test_data()
            
            # Step 2: Load training datasets
            training_data = self.load_training_datasets()
            
            # Step 3: Run training and evaluation loop
            all_results = self.run_training_evaluation_loop(training_data, test_data)
            
            # Step 4: Analyze results
            analysis = self.analyze_results(all_results)
            
            # Step 5: Create visualizations
            self.create_performance_visualizations(all_results, analysis)
            
            # Step 6: Save results
            self.save_results(all_results, analysis)
            
            # Phase completion
            end_time = datetime.now()
            runtime = (end_time - start_time).total_seconds() / 60
            
            self.logger.info(f"✅ Batch 6 Phase 7 completed successfully!")
            self.logger.info(f"Runtime: {runtime:.2f} minutes")
            self.logger.info(f"Experiments completed: {len(all_results)}")
            self.logger.info(f"Average F1 Score: {analysis['overall_summary']['average_f1']:.4f}")
            
            # Report on batch6 targets
            if 'batch6_targets' in analysis:
                targets = analysis['batch6_targets']
                self.logger.info(f"🎯 Target Analysis:")
                self.logger.info(f"   Best Synthetic F1: {targets['synthetic_best_f1']:.4f}")
                self.logger.info(f"   Target 0.75 Achieved: {'✅ YES' if targets['target_75_achieved'] else '❌ NO'}")
                if targets['improvement_gap'] > 0:
                    self.logger.info(f"   Gap to Real Baseline: {targets['improvement_gap']:.4f}")
            
            self.logger.info(f"Results directory: {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"❌ Batch 6 Phase 7 failed: {str(e)}")
            raise

def main():
    """Main execution function"""
    evaluator = Batch6Phase7ModelEvaluator()
    evaluator.run_phase7()

if __name__ == "__main__":
    main()