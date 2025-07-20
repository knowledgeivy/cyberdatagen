# src/cyberdata/scripts/batch2_comparative_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
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
ANALYSIS_DIR = PROJECT_ROOT / 'data/analysis'

# Create directories
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
MODEL_NAME = 'all-MiniLM-L6-v2'
plt.style.use('default')

def load_batch_data(batch_name):
    """Load data for a specific batch"""
    print(f"Loading {batch_name} data...")
    
    if batch_name == 'batch1':
        base_dir = BATCH1_DIR
        seed_file = base_dir / 'seed_samples/malicious_seeds_1k.csv'
        synthetic_files = {
            'rewrite': base_dir / 'synthetic/malicious_rewrite_1k.csv',
            'rewrite_strong': base_dir / 'synthetic/malicious_rewrite_strong_1k.csv',
            'rewrite_weak': base_dir / 'synthetic/malicious_rewrite_weak_1k.csv'
        }
    else:  # batch2
        base_dir = BATCH2_DIR
        seed_file = base_dir / 'enhanced_seeds/malicious_enhanced_seeds.csv'
        synthetic_files = {
            'rewrite': base_dir / 'synthetic/malicious_rewrite_enhanced.csv',
            'rewrite_strong': base_dir / 'synthetic/malicious_rewrite_strong_enhanced.csv',
            'rewrite_weak': base_dir / 'synthetic/malicious_rewrite_weak_enhanced.csv'
        }
    
    # Load real seeds
    if not seed_file.exists():
        print(f"Warning: Seed file not found: {seed_file}")
        return None
    
    real_seeds = pd.read_csv(seed_file)
    real_seeds['data_type'] = f'{batch_name}_real'
    real_seeds['batch'] = batch_name
    
    # Load synthetic variants
    datasets = [real_seeds]
    
    for variant, file_path in synthetic_files.items():
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['data_type'] = f'{batch_name}_{variant}'
            df['batch'] = batch_name
            datasets.append(df)
        else:
            print(f"Warning: {file_path} not found")
    
    combined_data = pd.concat(datasets, ignore_index=True)
    print(f"Loaded {len(combined_data)} {batch_name} samples")
    
    return combined_data

def generate_comparative_embeddings(batch1_data, batch2_data):
    """Generate embeddings for both batches"""
    print("Generating comparative embeddings...")
    
    # Check if embeddings already exist
    embeddings_file = ANALYSIS_DIR / 'comparative_embeddings.pkl'
    
    if embeddings_file.exists():
        print("Loading existing comparative embeddings...")
        with open(embeddings_file, 'rb') as f:
            return pickle.load(f)
    
    # Combine all data
    all_data = pd.concat([batch1_data, batch2_data], ignore_index=True)
    
    # Create text for embedding
    combined_text = all_data['subject'].fillna('') + ' ' + all_data['body'].fillna('')
    
    # Generate embeddings
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(combined_text.tolist(), show_progress_bar=True)
    
    # Save embedding data
    embedding_data = {
        'embeddings': embeddings,
        'data_types': all_data['data_type'].values,
        'batches': all_data['batch'].values,
        'data_ids': all_data['data_id'].values,
        'model_name': MODEL_NAME,
        'data_shape': embeddings.shape,
        'batch1_count': len(batch1_data),
        'batch2_count': len(batch2_data)
    }
    
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embedding_data, f)
    
    print(f"Comparative embeddings saved to {embeddings_file}")
    return embedding_data

def perform_comparative_visualization(embedding_data):
    """Create comparative visualizations"""
    print("Creating comparative visualizations...")
    
    embeddings = embedding_data['embeddings']
    data_types = embedding_data['data_types']
    batches = embedding_data['batches']
    
    # Dimensionality reduction
    pca = PCA(n_components=2, random_state=42)
    pca_embeddings = pca.fit_transform(embeddings)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_embeddings = tsne.fit_transform(embeddings)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Color mapping
    batch1_colors = {'batch1_real': 'blue', 'batch1_rewrite': 'orange', 
                     'batch1_rewrite_strong': 'green', 'batch1_rewrite_weak': 'red'}
    batch2_colors = {'batch2_real': 'navy', 'batch2_rewrite': 'darkorange', 
                     'batch2_rewrite_strong': 'darkgreen', 'batch2_rewrite_weak': 'darkred'}
    all_colors = {**batch1_colors, **batch2_colors}
    
    # 1. PCA - Batch1 only
    ax1 = axes[0, 0]
    batch1_mask = batches == 'batch1'
    for data_type, color in batch1_colors.items():
        type_mask = (data_types == data_type) & batch1_mask
        if np.any(type_mask):
            ax1.scatter(pca_embeddings[type_mask, 0], pca_embeddings[type_mask, 1],
                       label=data_type.replace('batch1_', ''), alpha=0.6, s=20, c=color)
    ax1.set_title('PCA: Batch1 Only')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. PCA - Batch2 only
    ax2 = axes[0, 1]
    batch2_mask = batches == 'batch2'
    for data_type, color in batch2_colors.items():
        type_mask = (data_types == data_type) & batch2_mask
        if np.any(type_mask):
            ax2.scatter(pca_embeddings[type_mask, 0], pca_embeddings[type_mask, 1],
                       label=data_type.replace('batch2_', ''), alpha=0.6, s=20, c=color)
    ax2.set_title('PCA: Batch2 Only')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. PCA - Combined comparison
    ax3 = axes[0, 2]
    for data_type, color in all_colors.items():
        type_mask = data_types == data_type
        if np.any(type_mask):
            marker = 'o' if 'batch1' in data_type else '^'
            ax3.scatter(pca_embeddings[type_mask, 0], pca_embeddings[type_mask, 1],
                       label=data_type, alpha=0.6, s=20, c=color, marker=marker)
    ax3.set_title('PCA: Batch1 vs Batch2 Comparison')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 4. t-SNE - Batch1 only
    ax4 = axes[1, 0]
    for data_type, color in batch1_colors.items():
        type_mask = (data_types == data_type) & batch1_mask
        if np.any(type_mask):
            ax4.scatter(tsne_embeddings[type_mask, 0], tsne_embeddings[type_mask, 1],
                       label=data_type.replace('batch1_', ''), alpha=0.6, s=20, c=color)
    ax4.set_title('t-SNE: Batch1 Only')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. t-SNE - Batch2 only
    ax5 = axes[1, 1]
    for data_type, color in batch2_colors.items():
        type_mask = (data_types == data_type) & batch2_mask
        if np.any(type_mask):
            ax5.scatter(tsne_embeddings[type_mask, 0], tsne_embeddings[type_mask, 1],
                       label=data_type.replace('batch2_', ''), alpha=0.6, s=20, c=color)
    ax5.set_title('t-SNE: Batch2 Only')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. t-SNE - Combined comparison
    ax6 = axes[1, 2]
    for data_type, color in all_colors.items():
        type_mask = data_types == data_type
        if np.any(type_mask):
            marker = 'o' if 'batch1' in data_type else '^'
            ax6.scatter(tsne_embeddings[type_mask, 0], tsne_embeddings[type_mask, 1],
                       label=data_type, alpha=0.6, s=20, c=color, marker=marker)
    ax6.set_title('t-SNE: Batch1 vs Batch2 Comparison')
    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = ANALYSIS_DIR / 'batch_comparative_analysis.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Comparative visualization saved to {plot_file}")
    plt.show()

def run_comparative_ml_experiments():
    """Run ML experiments comparing batch1 vs batch2"""
    print("Running comparative ML experiments...")
    
    # Load fixed test set from batch1
    test_data_file = BATCH1_DIR / 'ml_results/fixed_test_set.csv'
    if not test_data_file.exists():
        print(f"Error: Fixed test set not found: {test_data_file}")
        return None
    
    test_data = pd.read_csv(test_data_file)
    print(f"Using fixed test set: {len(test_data)} samples")
    
    # Define experiments for both batches
    experiments = {
        'batch1_real_only': {
            'batch': 'batch1',
            'malicious_files': [BATCH1_DIR / 'seed_samples/malicious_seeds_1k.csv'],
            'data_sources': ['batch1_real']
        },
        'batch1_rewrite': {
            'batch': 'batch1',
            'malicious_files': [
                BATCH1_DIR / 'seed_samples/malicious_seeds_1k.csv',
                BATCH1_DIR / 'synthetic/malicious_rewrite_1k.csv'
            ],
            'data_sources': ['batch1_real', 'batch1_rewrite']
        },
        'batch2_real_only': {
            'batch': 'batch2',
            'malicious_files': [BATCH2_DIR / 'enhanced_seeds/malicious_enhanced_seeds.csv'],
            'data_sources': ['batch2_real']
        },
        'batch2_rewrite': {
            'batch': 'batch2',
            'malicious_files': [
                BATCH2_DIR / 'enhanced_seeds/malicious_enhanced_seeds.csv',
                BATCH2_DIR / 'synthetic/malicious_rewrite_enhanced.csv'
            ],
            'data_sources': ['batch2_real', 'batch2_rewrite']
        }
    }
    
    # Load benign training data (same for all experiments)
    benign_data = pd.read_csv(BATCH1_DIR / 'seed_samples/benign_seeds_1k.csv')
    
    results = {}
    
    for exp_name, exp_config in experiments.items():
        print(f"\n--- Running experiment: {exp_name} ---")
        
        # Load malicious training data
        malicious_datasets = []
        for file_path in exp_config['malicious_files']:
            if file_path.exists():
                df = pd.read_csv(file_path)
                malicious_datasets.append(df)
            else:
                print(f"Warning: {file_path} not found")
        
        if not malicious_datasets:
            print(f"No data found for {exp_name}")
            continue
        
        malicious_data = pd.concat(malicious_datasets, ignore_index=True)
        
        # Combine training data
        training_data = pd.concat([malicious_data[['subject', 'body', 'label']], 
                                 benign_data[['subject', 'body', 'label']]], ignore_index=True)
        
        # Prepare features
        train_text = training_data['subject'].fillna('') + ' ' + training_data['body'].fillna('')
        test_text = test_data['subject'].fillna('') + ' ' + test_data['body'].fillna('')
        
        vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8, 
                                   stop_words='english', ngram_range=(1, 2))
        
        X_train = vectorizer.fit_transform(train_text)
        X_test = vectorizer.transform(test_text)
        y_train = training_data['label'].values
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
                'f1_macro': report['macro avg']['f1-score'],
                'f1_weighted': report['weighted avg']['f1-score'],
                'training_size': len(training_data)
            }
            
            print(f"  {model_name}: Accuracy = {accuracy:.4f}")
        
        results[exp_name] = exp_results
    
    return results

def analyze_coverage_improvement(embedding_data):
    """Analyze how batch2 improved coverage of real data space"""
    print("Analyzing coverage improvement...")
    
    embeddings = embedding_data['embeddings']
    data_types = embedding_data['data_types']
    
    # Separate batch1 and batch2 real data
    batch1_real_mask = data_types == 'batch1_real'
    batch2_real_mask = data_types == 'batch2_real'
    
    # Combine all synthetic data from both batches
    synthetic_masks = [
        data_types == 'batch1_rewrite',
        data_types == 'batch1_rewrite_strong', 
        data_types == 'batch1_rewrite_weak',
        data_types == 'batch2_rewrite',
        data_types == 'batch2_rewrite_strong',
        data_types == 'batch2_rewrite_weak'
    ]
    
    all_synthetic_mask = np.zeros(len(embeddings), dtype=bool)
    for mask in synthetic_masks:
        all_synthetic_mask |= mask
    
    if not np.any(all_synthetic_mask):
        print("No synthetic data found for coverage analysis")
        return {}
    
    from sklearn.neighbors import NearestNeighbors
    
    # Get synthetic embeddings
    synthetic_embeddings = embeddings[all_synthetic_mask]
    
    # Analyze coverage for both batches
    coverage_analysis = {}
    
    for batch_name, real_mask in [('batch1', batch1_real_mask), ('batch2', batch2_real_mask)]:
        if not np.any(real_mask):
            continue
            
        real_embeddings = embeddings[real_mask]
        
        # Find nearest synthetic neighbor for each real data point
        nbrs = NearestNeighbors(n_neighbors=1, metric='cosine')
        nbrs.fit(synthetic_embeddings)
        distances, _ = nbrs.kneighbors(real_embeddings)
        distances = distances.flatten()
        
        coverage_analysis[batch_name] = {
            'mean_distance_to_synthetic': float(np.mean(distances)),
            'median_distance_to_synthetic': float(np.median(distances)),
            'max_distance_to_synthetic': float(np.max(distances)),
            'std_distance_to_synthetic': float(np.std(distances)),
            'real_samples_count': int(np.sum(real_mask))
        }
    
    return coverage_analysis

def save_comprehensive_results(ml_results, coverage_analysis, embedding_data):
    """Save all comparative analysis results"""
    print("Saving comprehensive comparative results...")
    
    comprehensive_results = {
        'analysis_type': 'batch1_vs_batch2_comprehensive',
        'timestamp': pd.Timestamp.now().isoformat(),
        'embedding_info': {
            'model_name': embedding_data['model_name'],
            'total_samples': embedding_data['data_shape'][0],
            'embedding_dimension': embedding_data['data_shape'][1],
            'batch1_samples': embedding_data['batch1_count'],
            'batch2_samples': embedding_data['batch2_count']
        },
        'ml_comparative_results': ml_results,
        'coverage_analysis': coverage_analysis
    }
    
    # Save full results
    with open(ANALYSIS_DIR / 'comprehensive_comparative_results.json', 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    # Create summary comparison
    if ml_results:
        summary = {}
        for exp_name, exp_data in ml_results.items():
            summary[exp_name] = {}
            for model_name, model_data in exp_data.items():
                summary[exp_name][model_name] = {
                    'accuracy': model_data['accuracy'],
                    'f1_macro': model_data['f1_macro']
                }
        
        with open(ANALYSIS_DIR / 'ml_comparison_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
    
    print(f"Comprehensive results saved to: {ANALYSIS_DIR}")

def print_comparative_summary(ml_results, coverage_analysis):
    """Print summary of comparative analysis"""
    print("\n=== COMPARATIVE ANALYSIS SUMMARY ===")
    
    if ml_results:
        print("\nML Performance Comparison:")
        print(f"{'Experiment':<20} {'Model':<12} {'Accuracy':<10} {'F1-Macro':<10}")
        print("-" * 55)
        
        for exp_name, exp_data in ml_results.items():
            for i, (model_name, model_data) in enumerate(exp_data.items()):
                exp_display = exp_name if i == 0 else ""
                print(f"{exp_display:<20} {model_name:<12} {model_data['accuracy']:<10.4f} {model_data['f1_macro']:<10.4f}")
    
    if coverage_analysis:
        print("\nCoverage Analysis:")
        for batch_name, analysis in coverage_analysis.items():
            print(f"{batch_name}:")
            print(f"  Mean distance to synthetic: {analysis['mean_distance_to_synthetic']:.4f}")
            print(f"  Median distance to synthetic: {analysis['median_distance_to_synthetic']:.4f}")
            print(f"  Real samples: {analysis['real_samples_count']}")

def main():
    print("=== BATCH2: COMPREHENSIVE COMPARATIVE ANALYSIS ===")
    
    # Load batch data
    batch1_data = load_batch_data('batch1')
    batch2_data = load_batch_data('batch2')
    
    if batch1_data is None or batch2_data is None:
        print("Error: Could not load batch data")
        return
    
    # Generate comparative embeddings
    embedding_data = generate_comparative_embeddings(batch1_data, batch2_data)
    
    # Create comparative visualizations
    perform_comparative_visualization(embedding_data)
    
    # Run ML experiments
    ml_results = run_comparative_ml_experiments()
    
    # Analyze coverage improvement
    coverage_analysis = analyze_coverage_improvement(embedding_data)
    
    # Save comprehensive results
    save_comprehensive_results(ml_results, coverage_analysis, embedding_data)
    
    # Print summary
    print_comparative_summary(ml_results, coverage_analysis)
    
    print(f"\nComparative analysis complete. Results saved to: {ANALYSIS_DIR}")

if __name__ == "__main__":
    main()