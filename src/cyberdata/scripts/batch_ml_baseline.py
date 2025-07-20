# src/cyberdata/scripts/batch1_ml_baseline.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
SEED_SAMPLES_DIR = BATCH1_DIR / 'seed_samples'
SYNTHETIC_DIR = BATCH1_DIR / 'synthetic'
BASE_SAMPLES_DIR = BATCH1_DIR / 'base_samples'
ML_RESULTS_DIR = BATCH1_DIR / 'ml_results'

# Create directories
ML_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def prepare_training_test_split():
    """Prepare fixed training and test sets from base samples"""
    print("Preparing training/test split from base samples...")
    
    # Load all base samples
    malicious_5k = pd.read_csv(BASE_SAMPLES_DIR / 'malicious_5k.csv')
    benign_5k = pd.read_csv(BASE_SAMPLES_DIR / 'benign_5k.csv')
    
    # Load seed samples (these are used for training)
    malicious_seeds = pd.read_csv(SEED_SAMPLES_DIR / 'malicious_seeds_1k.csv')
    benign_seeds = pd.read_csv(SEED_SAMPLES_DIR / 'benign_seeds_1k.csv')
    
    # Get seed IDs for exclusion
    malicious_seed_ids = set(malicious_seeds['data_id'].values)
    benign_seed_ids = set(benign_seeds['data_id'].values)
    
    # Exclude seeds from base samples to create test sets
    malicious_test = malicious_5k[~malicious_5k['data_id'].isin(malicious_seed_ids)]
    benign_test = benign_5k[~benign_5k['data_id'].isin(benign_seed_ids)]
    
    print(f"Available for testing:")
    print(f"  Malicious: {len(malicious_test)} (excluded {len(malicious_seed_ids)} seeds)")
    print(f"  Benign: {len(benign_test)} (excluded {len(benign_seed_ids)} seeds)")
    
    # Sample balanced test set (1000 each)
    np.random.seed(42)
    test_size = 1000
    
    malicious_test_sample = malicious_test.sample(n=min(test_size, len(malicious_test)), random_state=42)
    benign_test_sample = benign_test.sample(n=min(test_size, len(benign_test)), random_state=42)
    
    # Combine test data
    test_data = pd.concat([malicious_test_sample, benign_test_sample], ignore_index=True)
    test_data['data_source'] = 'real_test'
    
    # Save fixed test set
    test_data.to_csv(ML_RESULTS_DIR / 'fixed_test_set.csv', index=False)
    
    print(f"Fixed test set created: {len(test_data)} samples ({len(malicious_test_sample)} malicious, {len(benign_test_sample)} benign)")
    
    return test_data

def load_experiment_data(experiment_type):
    """Load training data for specific experiment"""
    print(f"Loading training data for experiment: {experiment_type}")
    
    datasets = []
    
    # Load benign seeds (always included)
    benign_seeds = pd.read_csv(SEED_SAMPLES_DIR / 'benign_seeds_1k.csv')
    benign_seeds['data_source'] = 'real_benign'
    datasets.append(benign_seeds[['subject', 'body', 'label', 'data_source', 'data_id']])
    
    if experiment_type == 'real_only':
        # Only real malicious seeds
        malicious_seeds = pd.read_csv(SEED_SAMPLES_DIR / 'malicious_seeds_1k.csv')
        malicious_seeds['data_source'] = 'real_malicious'
        datasets.append(malicious_seeds[['subject', 'body', 'label', 'data_source', 'data_id']])
        
    elif experiment_type == 'rewrite':
        # Real + rewrite synthetic
        malicious_seeds = pd.read_csv(SEED_SAMPLES_DIR / 'malicious_seeds_1k.csv')
        malicious_seeds['data_source'] = 'real_malicious'
        datasets.append(malicious_seeds[['subject', 'body', 'label', 'data_source', 'data_id']])
        
        rewrite_data = pd.read_csv(SYNTHETIC_DIR / 'malicious_rewrite_1k.csv')
        rewrite_data['data_source'] = 'rewrite'
        datasets.append(rewrite_data[['subject', 'body', 'label', 'data_source', 'data_id']])
        
    elif experiment_type == 'rewrite_strong':
        # Real + rewrite_strong synthetic
        malicious_seeds = pd.read_csv(SEED_SAMPLES_DIR / 'malicious_seeds_1k.csv')
        malicious_seeds['data_source'] = 'real_malicious'
        datasets.append(malicious_seeds[['subject', 'body', 'label', 'data_source', 'data_id']])
        
        rewrite_strong_data = pd.read_csv(SYNTHETIC_DIR / 'malicious_rewrite_strong_1k.csv')
        rewrite_strong_data['data_source'] = 'rewrite_strong'
        datasets.append(rewrite_strong_data[['subject', 'body', 'label', 'data_source', 'data_id']])
        
    elif experiment_type == 'rewrite_weak':
        # Real + rewrite_weak synthetic
        malicious_seeds = pd.read_csv(SEED_SAMPLES_DIR / 'malicious_seeds_1k.csv')
        malicious_seeds['data_source'] = 'real_malicious'
        datasets.append(malicious_seeds[['subject', 'body', 'label', 'data_source', 'data_id']])
        
        rewrite_weak_data = pd.read_csv(SYNTHETIC_DIR / 'malicious_rewrite_weak_1k.csv')
        rewrite_weak_data['data_source'] = 'rewrite_weak'
        datasets.append(rewrite_weak_data[['subject', 'body', 'label', 'data_source', 'data_id']])
        
    elif experiment_type == 'all_synthetic':
        # Real + all 3 synthetic variants
        malicious_seeds = pd.read_csv(SEED_SAMPLES_DIR / 'malicious_seeds_1k.csv')
        malicious_seeds['data_source'] = 'real_malicious'
        datasets.append(malicious_seeds[['subject', 'body', 'label', 'data_source', 'data_id']])
        
        for variant in ['rewrite', 'rewrite_strong', 'rewrite_weak']:
            file_path = SYNTHETIC_DIR / f'malicious_{variant}_1k.csv'
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['data_source'] = variant
                datasets.append(df[['subject', 'body', 'label', 'data_source', 'data_id']])
    
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
    
    # TF-IDF vectorization
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
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
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
    """Run all experimental configurations"""
    print("Running all experimental configurations...")
    
    experiments = [
        'real_only',
        'rewrite', 
        'rewrite_strong',
        'rewrite_weak',
        'all_synthetic'
    ]
    
    all_results = {}
    
    for experiment in experiments:
        print(f"\n--- EXPERIMENT: {experiment} ---")
        
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
    print("Saving comprehensive results...")
    
    # Create summary comparison
    summary = {
        'experiment': 'batch1_ml_comprehensive',
        'timestamp': pd.Timestamp.now().isoformat(),
        'test_data_info': {
            'size': len(test_data),
            'malicious_count': int((test_data['label'] == 1).sum()),
            'benign_count': int((test_data['label'] == 0).sum())
        },
        'experiments': all_results
    }
    
    # Save full results
    with open(ML_RESULTS_DIR / 'batch1_comprehensive_results.json', 'w') as f:
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
    
    with open(ML_RESULTS_DIR / 'batch1_comparison_summary.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"Results saved to: {ML_RESULTS_DIR}")

def print_comparison_summary(all_results):
    """Print summary comparison across experiments"""
    print("\n=== COMPREHENSIVE RESULTS SUMMARY ===")
    
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
    print("=== BATCH 1: COMPREHENSIVE ML BASELINE ===")
    
    # Prepare fixed test set (unseen real data)
    test_data = prepare_training_test_split()
    
    # Run all experiments
    all_results = run_all_experiments(test_data)
    
    # Save results
    save_comprehensive_results(all_results, test_data)
    
    # Print summary
    print_comparison_summary(all_results)
    
    print(f"\nComprehensive evaluation complete. Results saved to: {ML_RESULTS_DIR}")

if __name__ == "__main__":
    main()