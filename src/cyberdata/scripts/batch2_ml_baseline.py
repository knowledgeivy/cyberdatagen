# src/cyberdata/scripts/batch2_ml_baseline.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
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
ENHANCED_SEEDS_DIR = BATCH2_DIR / 'enhanced_seeds'
SYNTHETIC_DIR = BATCH2_DIR / 'synthetic'
ML_RESULTS_DIR = BATCH2_DIR / 'ml_results'

# Create directories
ML_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_fixed_test_set():
    """Load the same fixed test set used in batch1"""
    print("Loading fixed test set from batch1...")
    
    test_file = BATCH1_DIR / 'ml_results/fixed_test_set.csv'
    if not test_file.exists():
        print(f"Error: Fixed test set not found: {test_file}")
        print("Please run batch1_ml_baseline.py first to create the fixed test set")
        return None
    
    test_data = pd.read_csv(test_file)
    print(f"Loaded fixed test set: {len(test_data)} samples")
    print(f"Test distribution: {test_data['label'].value_counts().to_dict()}")
    
    return test_data

def load_experiment_data(experiment_type):
    """Load training data for specific experiment using batch2 data"""
    print(f"Loading batch2 training data for experiment: {experiment_type}")
    
    datasets = []
    
    # Load benign seeds (same as batch1 for consistency)
    benign_seeds = pd.read_csv(ENHANCED_SEEDS_DIR / 'benign_seeds_1k.csv')
    benign_seeds['data_source'] = 'real_benign'
    datasets.append(benign_seeds[['subject', 'body', 'label', 'data_source', 'data_id']])
    
    if experiment_type == 'real_only':
        # Only enhanced real malicious seeds
        enhanced_seeds = pd.read_csv(ENHANCED_SEEDS_DIR / 'malicious_enhanced_seeds.csv')
        enhanced_seeds['data_source'] = 'real_malicious_enhanced'
        datasets.append(enhanced_seeds[['subject', 'body', 'label', 'data_source', 'data_id']])
        
    elif experiment_type == 'rewrite':
        # Enhanced real + rewrite synthetic
        enhanced_seeds = pd.read_csv(ENHANCED_SEEDS_DIR / 'malicious_enhanced_seeds.csv')
        enhanced_seeds['data_source'] = 'real_malicious_enhanced'
        datasets.append(enhanced_seeds[['subject', 'body', 'label', 'data_source', 'data_id']])
        
        rewrite_file = SYNTHETIC_DIR / 'malicious_rewrite_enhanced.csv'
        if rewrite_file.exists():
            rewrite_data = pd.read_csv(rewrite_file)
            rewrite_data['data_source'] = 'rewrite_enhanced'
            datasets.append(rewrite_data[['subject', 'body', 'label', 'data_source', 'data_id']])
        
    elif experiment_type == 'rewrite_strong':
        # Enhanced real + rewrite_strong synthetic
        enhanced_seeds = pd.read_csv(ENHANCED_SEEDS_DIR / 'malicious_enhanced_seeds.csv')
        enhanced_seeds['data_source'] = 'real_malicious_enhanced'
        datasets.append(enhanced_seeds[['subject', 'body', 'label', 'data_source', 'data_id']])
        
        rewrite_strong_file = SYNTHETIC_DIR / 'malicious_rewrite_strong_enhanced.csv'
        if rewrite_strong_file.exists():
            rewrite_strong_data = pd.read_csv(rewrite_strong_file)
            rewrite_strong_data['data_source'] = 'rewrite_strong_enhanced'
            datasets.append(rewrite_strong_data[['subject', 'body', 'label', 'data_source', 'data_id']])
        
    elif experiment_type == 'rewrite_weak':
        # Enhanced real + rewrite_weak synthetic
        enhanced_seeds = pd.read_csv(ENHANCED_SEEDS_DIR / 'malicious_enhanced_seeds.csv')
        enhanced_seeds['data_source'] = 'real_malicious_enhanced'
        datasets.append(enhanced_seeds[['subject', 'body', 'label', 'data_source', 'data_id']])
        
        rewrite_weak_file = SYNTHETIC_DIR / 'malicious_rewrite_weak_enhanced.csv'
        if rewrite_weak_file.exists():
            rewrite_weak_data = pd.read_csv(rewrite_weak_file)
            rewrite_weak_data['data_source'] = 'rewrite_weak_enhanced'
            datasets.append(rewrite_weak_data[['subject', 'body', 'label', 'data_source', 'data_id']])
    
    training_data = pd.concat(datasets, ignore_index=True)
    
    print(f"Training data loaded: {len(training_data)} samples")
    print("Data source distribution:")
    print(training_data['data_source'].value_counts())
    
    return training_data

def prepare_features(train_data, test_data):
    """Prepare TF-IDF features"""
    print("Preparing TF-IDF features...")
    
    # Combine subject and body
    train_text = train_data['subject'].fillna('') + ' ' + train_data['body'].fillna('')
    test_text = test_data['subject'].fillna('') + ' ' + test_data['body'].fillna('')
    
    # TF-IDF vectorization (same parameters as batch1)
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.8,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)
    
    y_train = train_data['label'].values
    y_test = test_data['label'].values
    
    print(f"Feature matrix: Train {X_train.shape}, Test {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, vectorizer

def train_and_evaluate_models(X_train, X_test, y_train, y_test, experiment_name):
    """Train and evaluate models"""
    print(f"Training and evaluating models for {experiment_name}...")
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'accuracy': float(accuracy),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist(),
            'probabilities': y_prob.tolist() if y_prob is not None else None
        }
        
        print(f"    {name} Accuracy: {accuracy:.4f}")
    
    return results

def run_all_experiments(test_data):
    """Run all experimental configurations for batch2"""
    print("Running all batch2 experimental configurations...")
    
    experiments = [
        'real_only',
        'rewrite', 
        'rewrite_strong',
        'rewrite_weak'
    ]
    
    all_results = {}
    
    for experiment in experiments:
        print(f"\n--- BATCH2 EXPERIMENT: {experiment} ---")
        
        # Load training data for this experiment
        training_data = load_experiment_data(experiment)
        
        # Prepare features
        X_train, X_test, y_train, y_test, vectorizer = prepare_features(training_data, test_data)
        
        # Train and evaluate
        results = train_and_evaluate_models(X_train, X_test, y_train, y_test, experiment)
        
        # Store results
        all_results[experiment] = {
            'results': results,
            'training_distribution': training_data['data_source'].value_counts().to_dict(),
            'training_size': len(training_data)
        }
    
    return all_results

def save_comprehensive_results(all_results, test_data):
    """Save comprehensive results"""
    print("Saving comprehensive batch2 results...")
    
    # Create summary comparison
    summary = {
        'experiment': 'batch2_ml_comprehensive',
        'timestamp': pd.Timestamp.now().isoformat(),
        'test_data_info': {
            'size': len(test_data),
            'malicious_count': int((test_data['label'] == 1).sum()),
            'benign_count': int((test_data['label'] == 0).sum()),
            'test_source': 'same_fixed_test_as_batch1'
        },
        'experiments': all_results
    }
    
    # Save full results
    with open(ML_RESULTS_DIR / 'batch2_comprehensive_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create comparison summary
    comparison = {}
    for exp_name, exp_data in all_results.items():
        comparison[exp_name] = {}
        for model_name, model_results in exp_data['results'].items():
            comparison[exp_name][model_name] = {
                'accuracy': model_results['accuracy'],
                'f1_macro': model_results['classification_report']['macro avg']['f1-score'],
                'f1_weighted': model_results['classification_report']['weighted avg']['f1-score']
            }
    
    with open(ML_RESULTS_DIR / 'batch2_comparison_summary.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"Results saved to: {ML_RESULTS_DIR}")

def print_comparison_summary(all_results):
    """Print summary comparison across experiments"""
    print("\n=== BATCH2 RESULTS SUMMARY ===")
    
    print(f"{'Experiment':<15} {'Model':<18} {'Accuracy':<10} {'F1-Macro':<10} {'F1-Weighted'}")
    print("-" * 70)
    
    for exp_name, exp_data in all_results.items():
        for i, (model_name, model_results) in enumerate(exp_data['results'].items()):
            exp_display = exp_name if i == 0 else ""
            accuracy = model_results['accuracy']
            f1_macro = model_results['classification_report']['macro avg']['f1-score']
            f1_weighted = model_results['classification_report']['weighted avg']['f1-score']
            
            print(f"{exp_display:<15} {model_name:<18} {accuracy:<10.4f} {f1_macro:<10.4f} {f1_weighted:.4f}")

def main():
    print("=== BATCH 2: COMPREHENSIVE ML BASELINE ===")
    
    # Load fixed test set (same as batch1)
    test_data = load_fixed_test_set()
    if test_data is None:
        return
    
    # Run all experiments
    all_results = run_all_experiments(test_data)
    
    # Save results
    save_comprehensive_results(all_results, test_data)
    
    # Print summary
    print_comparison_summary(all_results)
    
    print(f"\nBatch2 ML evaluation complete. Results saved to: {ML_RESULTS_DIR}")

if __name__ == "__main__":
    main()