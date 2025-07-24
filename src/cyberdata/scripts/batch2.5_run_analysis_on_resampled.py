# src/cyberdata/scripts/run_analysis_on_resampled.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
import pickle
import json
import argparse
from pathlib import Path

# Determine PROJECT_ROOT
current_file = Path(__file__).resolve()
if 'src/cyberdata/scripts' in str(current_file):
    PROJECT_ROOT = current_file.parents[3]
else:
    PROJECT_ROOT = Path.cwd()

print(f"PROJECT_ROOT: {PROJECT_ROOT}")

# Configuration
MODEL_NAME = 'all-MiniLM-L6-v2'
plt.style.use('default')

def load_resampled_data(batch_name):
    """Load resampled data for a batch"""
    print(f"Loading resampled data for {batch_name}...")
    
    batch_dir = PROJECT_ROOT / f'data/{batch_name}'
    resampled_dir = batch_dir / 'resampled'
    
    if not resampled_dir.exists():
        print(f"Error: Resampled directory not found: {resampled_dir}")
        print("Please run resample_for_analysis.py first")
        return None
    
    # Load metadata to get file info
    metadata_file = resampled_dir / 'resampling_metadata.json'
    if not metadata_file.exists():
        print(f"Error: Resampling metadata not found: {metadata_file}")
        return None
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"Resampling metadata: {metadata['actual_counts']}")
    
    # Load combined datasets for each experiment
    experiments = ['real_only', 'rewrite', 'rewrite_strong', 'rewrite_weak']
    experiment_data = {}
    
    for exp in experiments:
        # Find the combined file for this experiment
        combined_files = list(resampled_dir.glob(f'{exp}_combined_*.csv'))
        if combined_files:
            combined_file = combined_files[0]  # Take the first match
            df = pd.read_csv(combined_file)
            experiment_data[exp] = df
            print(f"  {exp}: {len(df)} samples loaded")
        else:
            print(f"  Warning: No combined file found for {exp}")
    
    return experiment_data, metadata

def load_fixed_test_set():
    """Load the same fixed test set used in original analysis"""
    test_file = PROJECT_ROOT / 'data/batch1/ml_results/fixed_test_set.csv'
    if not test_file.exists():
        print(f"Error: Fixed test set not found: {test_file}")
        return None
    
    test_data = pd.read_csv(test_file)
    print(f"Loaded fixed test set: {len(test_data)} samples")
    return test_data

def run_ml_experiments(experiment_data, test_data, batch_name):
    """Run ML experiments on resampled data"""
    print(f"\nRunning ML experiments for {batch_name} (resampled)...")
    
    results = {}
    
    for exp_name, train_data in experiment_data.items():
        print(f"\n--- {exp_name.upper()} ---")
        
        # Prepare features
        train_text = train_data['subject'].fillna('') + ' ' + train_data['body'].fillna('')
        test_text = test_data['subject'].fillna('') + ' ' + test_data['body'].fillna('')
        
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
        
        # Train models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True)
        }
        
        exp_results = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            exp_results[model_name] = {
                'accuracy': float(accuracy),
                'f1_macro': float(report['macro avg']['f1-score']),
                'f1_weighted': float(report['weighted avg']['f1-score']),
                'training_size': len(train_data)
            }
            
            print(f"  {model_name}: Accuracy = {accuracy:.4f}, F1-macro = {report['macro avg']['f1-score']:.4f}")
        
        results[exp_name] = exp_results
    
    return results

def run_embedding_analysis(experiment_data, batch_name):
    """Run embedding analysis on resampled data"""
    print(f"\nRunning embedding analysis for {batch_name} (resampled)...")
    
    # Combine all data for embedding analysis
    all_datasets = []
    for exp_name, data in experiment_data.items():
        if exp_name == 'real_only':
            # For embedding analysis, we want to see real + all synthetic variants
            continue
    
    # Load individual resampled files for embedding analysis
    batch_dir = PROJECT_ROOT / f'data/{batch_name}'
    resampled_dir = batch_dir / 'resampled'
    
    # Load metadata to get file names
    with open(resampled_dir / 'resampling_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    embedding_datasets = []
    data_types = []
    
    # Load each type separately to maintain data_type labels
    malicious_types = ['real', 'rewrite', 'rewrite_strong', 'rewrite_weak']
    
    for mal_type in malicious_types:
        file_pattern = f'{mal_type}_malicious_*.csv'
        files = list(resampled_dir.glob(file_pattern))
        if files:
            df = pd.read_csv(files[0])
            df['data_type'] = mal_type
            embedding_datasets.append(df)
            data_types.extend([mal_type] * len(df))
            print(f"  Loaded {len(df)} {mal_type} samples for embedding")
    
    if not embedding_datasets:
        print("No data available for embedding analysis")
        return None
    
    combined_data = pd.concat(embedding_datasets, ignore_index=True)
    
    # Generate embeddings
    print("Generating embeddings...")
    combined_text = combined_data['subject'].fillna('') + ' ' + combined_data['body'].fillna('')
    
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(combined_text.tolist(), show_progress_bar=True)
    
    # Dimensionality reduction
    print("Performing dimensionality reduction...")
    pca = PCA(n_components=2, random_state=42)
    pca_embeddings = pca.fit_transform(embeddings)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_embeddings = tsne.fit_transform(embeddings)
    
    # Analyze coverage
    coverage_info = analyze_coverage(embeddings, np.array(data_types))
    
    # Create visualization
    create_resampled_visualization(pca_embeddings, tsne_embeddings, np.array(data_types), 
                                 pca.explained_variance_ratio_, batch_name, coverage_info)
    
    return {
        'embeddings_shape': embeddings.shape,  # Only save shape, not the actual embeddings
        'data_types': data_types,
        'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
        'coverage_analysis': coverage_info,
        'sample_counts': {dt: len([x for x in data_types if x == dt]) for dt in set(data_types)}
    }

def analyze_coverage(embeddings, data_types):
    """Analyze how well synthetic data covers real data"""
    print("Analyzing coverage...")
    
    real_mask = data_types == 'real'
    synthetic_masks = {
        'rewrite': data_types == 'rewrite',
        'rewrite_strong': data_types == 'rewrite_strong',
        'rewrite_weak': data_types == 'rewrite_weak'
    }
    
    # Combine all synthetic data
    all_synthetic_mask = np.zeros(len(embeddings), dtype=bool)
    for mask in synthetic_masks.values():
        all_synthetic_mask |= mask
    
    if not np.any(real_mask) or not np.any(all_synthetic_mask):
        return {}
    
    real_embeddings = embeddings[real_mask]
    synthetic_embeddings = embeddings[all_synthetic_mask]
    
    # Find nearest synthetic neighbor for each real data point
    nbrs = NearestNeighbors(n_neighbors=1, metric='cosine')
    nbrs.fit(synthetic_embeddings)
    
    distances, _ = nbrs.kneighbors(real_embeddings)
    distances = distances.flatten()
    
    # Identify uncovered real data (top 20% most distant)
    threshold = np.percentile(distances, 80)
    uncovered_mask = distances > threshold
    uncovered_count = np.sum(uncovered_mask)
    
    coverage_rate = (len(real_embeddings) - uncovered_count) / len(real_embeddings) * 100
    
    return {
        'total_real': len(real_embeddings),
        'uncovered_count': int(uncovered_count),
        'coverage_rate': float(coverage_rate),
        'mean_distance': float(np.mean(distances)),
        'threshold': float(threshold)
    }

def create_resampled_visualization(pca_embeddings, tsne_embeddings, data_types, 
                                 explained_variance, batch_name, coverage_info):
    """Create visualization for resampled data"""
    print("Creating visualization...")
    
    data_type_list = ['real', 'rewrite', 'rewrite_strong', 'rewrite_weak']
    colors = ['blue', 'orange', 'green', 'red']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # PCA plot
    ax1 = axes[0]
    for i, data_type in enumerate(data_type_list):
        if data_type in data_types:
            type_mask = data_types == data_type
            ax1.scatter(pca_embeddings[type_mask, 0], pca_embeddings[type_mask, 1],
                       label=data_type, alpha=0.7, s=30, c=colors[i])
    
    ax1.set_xlabel(f'PC1 ({explained_variance[0]:.1%})')
    ax1.set_ylabel(f'PC2 ({explained_variance[1]:.1%})')
    ax1.set_title(f'PCA: {batch_name.title()} Resampled Data')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # t-SNE plot
    ax2 = axes[1]
    for i, data_type in enumerate(data_type_list):
        if data_type in data_types:
            type_mask = data_types == data_type
            ax2.scatter(tsne_embeddings[type_mask, 0], tsne_embeddings[type_mask, 1],
                       label=data_type, alpha=0.7, s=30, c=colors[i])
    
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.set_title(f't-SNE: {batch_name.title()} Resampled Data')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Coverage summary
    ax3 = axes[2]
    if coverage_info:
        covered = coverage_info['total_real'] - coverage_info['uncovered_count']
        uncovered = coverage_info['uncovered_count']
        
        bars = ax3.bar(['Covered', 'Uncovered'], [covered, uncovered], 
                      color=['lightgreen', 'red'], alpha=0.7)
        
        ax3.set_title(f'Coverage Analysis\n{coverage_info["coverage_rate"]:.1f}% covered')
        ax3.set_ylabel('Number of Real Samples')
        
        # Add count labels
        for bar, count in zip(bars, [covered, uncovered]):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(count), ha='center', va='bottom')
    else:
        ax3.text(0.5, 0.5, 'No coverage data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Coverage Analysis')
    
    plt.tight_layout()
    
    # Save plot
    batch_dir = PROJECT_ROOT / f'data/{batch_name}'
    plot_file = batch_dir / 'resampled' / f'{batch_name}_resampled_analysis.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {plot_file}")
    plt.show()

def save_resampled_results(ml_results, embedding_results, batch_name, metadata):
    """Save results from resampled analysis"""
    print(f"Saving resampled results for {batch_name}...")
    
    batch_dir = PROJECT_ROOT / f'data/{batch_name}'
    resampled_dir = batch_dir / 'resampled'
    
    # Save ML results
    ml_summary = {
        'experiment': f'{batch_name}_resampled_ml',
        'timestamp': pd.Timestamp.now().isoformat(),
        'resampling_info': metadata,
        'ml_results': ml_results
    }
    
    with open(resampled_dir / f'{batch_name}_resampled_ml_results.json', 'w') as f:
        json.dump(ml_summary, f, indent=2)
    
    # Save embedding results
    if embedding_results:
        embedding_summary = {
            'experiment': f'{batch_name}_resampled_embedding',
            'timestamp': pd.Timestamp.now().isoformat(),
            'resampling_info': metadata,
            'embedding_results': embedding_results
        }
        
        with open(resampled_dir / f'{batch_name}_resampled_embedding_results.json', 'w') as f:
            json.dump(embedding_summary, f, indent=2)
    
    print(f"Results saved to: {resampled_dir}")

def print_resampled_summary(ml_results, embedding_results, batch_name):
    """Print summary of resampled analysis"""
    print(f"\n=== {batch_name.upper()} RESAMPLED ANALYSIS SUMMARY ===")
    
    # ML results
    print("\nML Performance:")
    print(f"{'Experiment':<15} {'Model':<12} {'Accuracy':<10} {'F1-Macro':<10}")
    print("-" * 50)
    
    for exp_name, exp_data in ml_results.items():
        for model_name, model_data in exp_data.items():
            print(f"{exp_name:<15} {model_name:<12} {model_data['accuracy']:<10.4f} {model_data['f1_macro']:<10.4f}")
    
    # Embedding results
    if embedding_results and 'coverage_analysis' in embedding_results:
        cov = embedding_results['coverage_analysis']
        print(f"\nCoverage Analysis:")
        print(f"Coverage rate: {cov['coverage_rate']:.1f}%")
        print(f"Uncovered samples: {cov['uncovered_count']}/{cov['total_real']}")

def main():
    parser = argparse.ArgumentParser(description='Run analysis on resampled data')
    parser.add_argument('--batch', choices=['batch1', 'batch2', 'both'], default='both',
                       help='Which batch to analyze (default: both)')
    parser.add_argument('--skip-ml', action='store_true',
                       help='Skip ML analysis')
    parser.add_argument('--skip-embedding', action='store_true',
                       help='Skip embedding analysis')
    
    args = parser.parse_args()
    
    print("=== ANALYSIS ON RESAMPLED DATA ===")
    
    # Load fixed test set
    test_data = load_fixed_test_set()
    if test_data is None:
        return
    
    batches_to_process = ['batch1', 'batch2'] if args.batch == 'both' else [args.batch]
    
    for batch_name in batches_to_process:
        print(f"\n{'='*50}")
        print(f"PROCESSING {batch_name.upper()}")
        print(f"{'='*50}")
        
        # Load resampled data
        experiment_data, metadata = load_resampled_data(batch_name)
        if not experiment_data:
            continue
        
        ml_results = None
        embedding_results = None
        
        # Run ML experiments
        if not args.skip_ml:
            ml_results = run_ml_experiments(experiment_data, test_data, batch_name)
        
        # Run embedding analysis
        if not args.skip_embedding:
            embedding_results = run_embedding_analysis(experiment_data, batch_name)
        
        # Save results
        save_resampled_results(ml_results, embedding_results, batch_name, metadata)
        
        # Print summary
        if ml_results:
            print_resampled_summary(ml_results, embedding_results, batch_name)
    
    print(f"\n=== RESAMPLED ANALYSIS COMPLETE ===")
    print(f"Next step: Run comparison analysis on resampled results")

if __name__ == "__main__":
    main()