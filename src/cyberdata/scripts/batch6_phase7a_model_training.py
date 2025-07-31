#!/usr/bin/env python3
"""
Batch 6 Phase 7a: Model Training and Performance Evaluation
===========================================================

Pure ML training stage - trains models and saves results for later visualization.
Separate from visualization to allow independent execution and result reuse.

Key features:
- Train 3 models (RandomForest, SVM, DeepLearning) on 16 pure datasets
- Save trained models and vectorizers for reuse
- Export performance results in structured format
- No visualization - focused on ML execution efficiency
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
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

class Batch6Phase7aModelTrainer:
    """ML training stage for batch6 pure dataset performance evaluation"""
    
    def __init__(self):
        self.setup_paths()
        self.setup_logging()
        self.load_configurations()
        
    def setup_paths(self) -> None:
        """Setup directory structure for phase 7a"""
        # Use absolute paths to avoid working directory issues
        current_dir = Path(__file__).parent.parent.parent.parent  # Go up to project root
        self.base_dir = current_dir / "data" / "batch6"
        self.input_dirs = {
            'pure_datasets': self.base_dir / "pure_datasets",
            'test_data': current_dir / "data" / "batch4_fresh"
        }
        
        # Output directories - ML training results
        self.output_dir = self.base_dir / "phase7a_training"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for subdir in ['models', 'vectorizers', 'results', 'predictions']:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)
            
    def setup_logging(self) -> None:
        """Setup logging configuration"""
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"batch6_phase7a_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_configurations(self) -> None:
        """Load model training configurations"""
        self.config = {
            'models': {
                'RandomForest': {
                    'classifier': RandomForestClassifier,
                    'params': {
                        'n_estimators': 100,
                        'max_depth': 20,
                        'random_state': 42,
                        'n_jobs': -1  # Use all CPU cores
                    }
                },
                'SVM': {
                    'classifier': SVC,
                    'params': {
                        'kernel': 'rbf',
                        'C': 1.0,
                        'random_state': 42,
                        'probability': True,
                        'cache_size': 1000  # Optimize performance
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
            'dataset_types': ['baseline_real', 'pure_original', 'pure_strong', 'pure_weak'],
            'malicious_ratios': [5, 10, 15, 20],
            'random_seed': 42
        }
        
        self.logger.info("Phase 7a ML training configuration loaded")
        
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
            
            training_data = {}
            
            for dataset_type in self.config['dataset_types']:
                training_data[dataset_type] = {}
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
            
            return training_data
            
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
                      model_name: str, dataset_name: str) -> Dict[str, Any]:
        """Evaluate model performance on test set"""
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            }
            
            # Add metadata
            metrics.update({
                'model_name': model_name,
                'dataset_name': dataset_name,
                'test_samples': len(y_test),
                'positive_predictions': int(y_pred.sum()),
                'true_positives': int(y_test.sum()),
                'experiment_timestamp': datetime.now().isoformat()
            })
            
            # Save predictions for detailed analysis
            predictions = {
                'y_true': y_test.tolist(),
                'y_pred': y_pred.tolist(),
                'y_pred_proba': y_pred_proba.tolist() if y_pred_proba is not None else None
            }
            
            pred_file = self.output_dir / "predictions" / f"{dataset_name}_{model_name}_predictions.json"
            with open(pred_file, 'w') as f:
                json.dump(predictions, f, indent=2)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate model {model_name} on {dataset_name}: {str(e)}")
            raise
            
    def save_model_and_vectorizer(self, model: Any, vectorizer: TfidfVectorizer, 
                                 model_name: str, dataset_name: str) -> Dict[str, str]:
        """Save trained model and vectorizer"""
        try:
            # Save model
            model_file = self.output_dir / "models" / f"{dataset_name}_{model_name}_model.joblib"
            joblib.dump(model, model_file)
            
            # Save vectorizer
            vectorizer_file = self.output_dir / "vectorizers" / f"{dataset_name}_{model_name}_vectorizer.joblib"
            joblib.dump(vectorizer, vectorizer_file)
            
            return {
                'model_file': str(model_file),
                'vectorizer_file': str(vectorizer_file)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to save model/vectorizer for {model_name} on {dataset_name}: {str(e)}")
            raise
            
    def run_training_loop(self, training_data: Dict[str, Dict[str, pd.DataFrame]], 
                         test_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Run complete training loop for all datasets and models"""
        try:
            self.logger.info("Starting ML training loop...")
            
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
                            
                            # Save model and vectorizer
                            file_paths = self.save_model_and_vectorizer(model, vectorizer, model_name, dataset_name)
                            
                            # Add experiment metadata
                            metrics.update({
                                'dataset_type': dataset_type,
                                'malicious_ratio': ratio_name,
                                'training_samples': len(train_dataset),
                                'feature_count': X_train.shape[1],
                                'model_file': file_paths['model_file'],
                                'vectorizer_file': file_paths['vectorizer_file']
                            })
                            
                            all_results.append(metrics)
                            
                            self.logger.info(f"✅ {model_name} on {dataset_name}: F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}")
                            
                        except Exception as e:
                            self.logger.error(f"❌ Failed experiment: {model_name} on {dataset_name} - {str(e)}")
                            continue
            
            self.logger.info(f"✅ ML training completed: {len(all_results)}/{total_experiments} experiments successful")
            return all_results
            
        except Exception as e:
            self.logger.error(f"Failed training loop: {str(e)}")
            raise
            
    def save_training_results(self, all_results: List[Dict[str, Any]]) -> None:
        """Save training results in structured format for visualization stage"""
        try:
            self.logger.info("Saving training results...")
            
            # Convert to DataFrame for easier manipulation
            results_df = pd.DataFrame(all_results)
            
            # Save detailed results
            results_file = self.output_dir / "results" / "detailed_results.csv"
            results_df.to_csv(results_file, index=False)
            
            # Save structured results for visualization
            structured_results = {
                'experiment_metadata': {
                    'total_experiments': len(all_results),
                    'completion_date': datetime.now().isoformat(),
                    'dataset_types': self.config['dataset_types'],
                    'malicious_ratios': self.config['malicious_ratios'],
                    'models': list(self.config['models'].keys()),
                    'experiment_matrix': f"{len(self.config['dataset_types'])} × {len(self.config['malicious_ratios'])} × {len(self.config['models'])} = {len(self.config['dataset_types']) * len(self.config['malicious_ratios']) * len(self.config['models'])}"
                },
                'performance_data': all_results,
                'summary_statistics': {
                    'overall_avg_f1': float(results_df['f1'].mean()),
                    'overall_avg_accuracy': float(results_df['accuracy'].mean()),
                    'best_f1_experiment': {
                        'dataset': results_df.loc[results_df['f1'].idxmax(), 'dataset_name'],
                        'model': results_df.loc[results_df['f1'].idxmax(), 'model_name'],
                        'f1_score': float(results_df['f1'].max())
                    },
                    'by_dataset_type': {},
                    'by_model': {}
                }
            }
            
            # Add summary statistics by dataset type
            for dataset_type in self.config['dataset_types']:
                type_results = results_df[results_df['dataset_type'] == dataset_type]
                if len(type_results) > 0:
                    structured_results['summary_statistics']['by_dataset_type'][dataset_type] = {
                        'avg_f1': float(type_results['f1'].mean()),
                        'avg_accuracy': float(type_results['accuracy'].mean()),
                        'best_f1': float(type_results['f1'].max()),
                        'worst_f1': float(type_results['f1'].min())
                    }
            
            # Add summary statistics by model
            for model_name in self.config['models'].keys():
                model_results = results_df[results_df['model_name'] == model_name]
                if len(model_results) > 0:
                    structured_results['summary_statistics']['by_model'][model_name] = {
                        'avg_f1': float(model_results['f1'].mean()),
                        'avg_accuracy': float(model_results['accuracy'].mean()),
                        'best_f1': float(model_results['f1'].max())
                    }
            
            # Save structured results
            structured_file = self.output_dir / "results" / "structured_results.json"
            with open(structured_file, 'w') as f:
                json.dump(structured_results, f, indent=2)
            
            # Save phase completion summary
            completion_summary = {
                'phase7a_training_completion': {
                    'status': 'completed',
                    'completion_date': datetime.now().isoformat(),
                    'total_experiments': len(all_results),
                    'successful_experiments': len(all_results),
                    'models_trained': len(all_results),
                    'output_files': {
                        'detailed_results': str(results_file),
                        'structured_results': str(structured_file),
                        'models_directory': str(self.output_dir / "models"),
                        'vectorizers_directory': str(self.output_dir / "vectorizers"),
                        'predictions_directory': str(self.output_dir / "predictions")
                    }
                },
                'performance_summary': structured_results['summary_statistics']
            }
            
            summary_file = self.output_dir / "phase7a_completion_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(completion_summary, f, indent=2)
            
            self.logger.info(f"✅ Training results saved:")
            self.logger.info(f"   Detailed results: {results_file}")
            self.logger.info(f"   Structured results: {structured_file}")
            self.logger.info(f"   Models: {len(all_results)} trained models saved")
            self.logger.info(f"   Summary: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save training results: {str(e)}")
            raise
            
    def run_phase7a(self) -> None:
        """Execute complete Phase 7a ML training"""
        try:
            self.logger.info("🚀 Starting Batch 6 Phase 7a: ML Training Stage")
            start_time = datetime.now()
            
            # Step 1: Load test data
            test_data = self.load_test_data()
            
            # Step 2: Load training datasets
            training_data = self.load_training_datasets()
            
            # Step 3: Run training loop
            all_results = self.run_training_loop(training_data, test_data)
            
            # Step 4: Save results
            self.save_training_results(all_results)
            
            # Phase completion
            end_time = datetime.now()
            runtime = (end_time - start_time).total_seconds() / 60
            
            self.logger.info(f"✅ Batch 6 Phase 7a completed successfully!")
            self.logger.info(f"Runtime: {runtime:.2f} minutes")
            self.logger.info(f"Experiments completed: {len(all_results)}")
            self.logger.info(f"Average F1 Score: {np.mean([r['f1'] for r in all_results]):.4f}")
            
            # Report key findings
            if all_results:
                best_result = max(all_results, key=lambda x: x['f1'])
                self.logger.info(f"🎯 Best Performance:")
                self.logger.info(f"   Dataset: {best_result['dataset_name']}")
                self.logger.info(f"   Model: {best_result['model_name']}")
                self.logger.info(f"   F1 Score: {best_result['f1']:.4f}")
            
            self.logger.info(f"Output directory: {self.output_dir}")
            self.logger.info("Ready for Phase 7b (Visualization Stage)")
            
        except Exception as e:
            self.logger.error(f"❌ Batch 6 Phase 7a failed: {str(e)}")
            raise

def main():
    """Main execution function"""
    trainer = Batch6Phase7aModelTrainer()
    trainer.run_phase7a()

if __name__ == "__main__":
    main()