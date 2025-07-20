# src/cyberdata/scripts/batch2_embedding_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
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
BATCH2_DIR = PROJECT_ROOT / 'data/batch2'
ENHANCED_SEEDS_DIR = BATCH2_DIR / 'enhanced_seeds'
SYNTHETIC_DIR = BATCH2_DIR / 'synthetic'
EMBEDDINGS_DIR = BATCH2_DIR / 'embeddings'

# Create directories
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
MODEL_NAME = 'all-MiniLM-L6-v2'
plt.style.use('default')

def load_all_batch2_data():
    """Load real enhanced seeds and all synthetic variants"""
    print("Loading batch2 data...")
    
    # Load enhanced real malicious seeds
    enhanced_seeds = pd.read_csv(ENHANCED_SEEDS_DIR / 'malicious_enhanced_seeds.csv')
    enhanced_seeds['data_type'] = 'real'
    
    # Load synthetic variants
    datasets = [enhanced_seeds]
    
    for variant in ['rewrite', 'rewrite_strong', 'rewrite_weak']:
        file_path = SYNTHETIC_DIR / f'malicious_{variant}_enhanced.csv'
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['data_type'] = variant
            datasets.append(df)
        else:
            print(f"Warning: {file_path} not found")
    
    combined_data = pd.concat(datasets, ignore_index=True)
    print(f"Loaded {len(combined_data)} total batch2 samples")
    print("Data type distribution:")
    print(combined_data['data_type'].value_counts())
    
    return combined_data

def generate_embeddings(combined_data):
    """Generate embeddings for all batch2 data"""
    print("Generating batch2 embeddings...")
    
    # Check if embeddings already exist
    embeddings_file = EMBEDDINGS_DIR / 'batch2_embeddings.pkl'
    
    if embeddings_file.exists():
        print("Loading existing batch2 embeddings...")
        with open(embeddings_file, 'rb') as f:
            embedding_data = pickle.load(f)
        return embedding_data
    
    # Create text for embedding
    combined_text = combined_data['subject'].fillna('') + ' ' + combined_data['body'].fillna('')
    
    # Generate embeddings
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(combined_text.tolist(), show_progress_bar=True)
    
    # Save embedding data
    embedding_data = {
        'embeddings': embeddings,
        'data_types': combined_data['data_type'].values,
        'data_ids': combined_data['data_id'].values,
        'model_name': MODEL_NAME,
        'data_shape': embeddings.shape
    }
    
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embedding_data, f)
    
    print(f"Batch2 embeddings saved to {embeddings_file}")
    return embedding_data

def perform_dimensionality_reduction(embeddings):
    """Perform PCA and t-SNE"""
    print("Performing dimensionality reduction...")
    
    # PCA
    pca = PCA(n_components=2, random_state=42)
    pca_embeddings = pca.fit_transform(embeddings)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_embeddings = tsne.fit_transform(embeddings)
    
    return {
        'pca': {
            'embeddings': pca_embeddings,
            'explained_variance': pca.explained_variance_ratio_
        },
        'tsne': {
            'embeddings': tsne_embeddings
        }
    }

def identify_uncovered_real_data(embeddings, data_types, method='nearest_neighbor', k_neighbors=5):
    """Identify real data points that are poorly covered by synthetic data"""
    print(f"Identifying uncovered real data using {method} method...")
    
    from sklearn.neighbors import NearestNeighbors
    
    # Get masks for each data type
    real_mask = data_types == 'real'
    synthetic_masks = {
        'rewrite': data_types == 'rewrite',
        'rewrite_strong': data_types == 'rewrite_strong', 
        'rewrite_weak': data_types == 'rewrite_weak'
    }
    
    real_embeddings = embeddings[real_mask]
    real_indices = np.where(real_mask)[0]
    
    uncovered_info = {}
    
    if method == 'nearest_neighbor':
        # For each synthetic type, find real data points far from synthetic data
        all_synthetic_mask = np.zeros(len(embeddings), dtype=bool)
        for syn_mask in synthetic_masks.values():
            all_synthetic_mask |= syn_mask
        
        if np.sum(all_synthetic_mask) == 0:
            print("No synthetic data found")
            return {}
            
        synthetic_embeddings = embeddings[all_synthetic_mask]
        
        # Find nearest synthetic neighbor for each real data point
        nbrs = NearestNeighbors(n_neighbors=1, metric='cosine')
        nbrs.fit(synthetic_embeddings)
        
        distances, _ = nbrs.kneighbors(real_embeddings)
        distances = distances.flatten()
        
        # Find real data points that are far from any synthetic data
        threshold = np.percentile(distances, 80)  # Top 20% most distant
        outlier_mask = distances > threshold
        
        outlier_indices = real_indices[outlier_mask]
        outlier_distances = distances[outlier_mask]
        
        uncovered_info['real'] = {
            'indices': outlier_indices.tolist(),
            'distances_to_synthetic': outlier_distances.tolist(),
            'threshold': float(threshold),
            'count': len(outlier_indices),
            'method': 'nearest_neighbor_to_synthetic'
        }
        
    # Also identify isolated real clusters using DBSCAN on real data only
    if len(real_embeddings) > 10:
        dbscan = DBSCAN(eps=0.15, min_samples=3, metric='cosine')
        cluster_labels = dbscan.fit_predict(real_embeddings)
        
        # Find isolated points (noise in DBSCAN)
        isolated_mask = cluster_labels == -1
        isolated_indices = real_indices[isolated_mask]
        
        uncovered_info['real_isolated'] = {
            'indices': isolated_indices.tolist(),
            'count': len(isolated_indices),
            'method': 'dbscan_isolated'
        }
    
    return uncovered_info

def create_visualizations(reduction_results, data_types, uncovered_info):
    """Create visualization plots"""
    print("Creating batch2 visualizations...")
    
    data_type_list = ['real', 'rewrite', 'rewrite_strong', 'rewrite_weak']
    colors = ['blue', 'orange', 'green', 'red']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # PCA plot
    ax1 = axes[0, 0]
    pca_embeddings = reduction_results['pca']['embeddings']
    for i, data_type in enumerate(data_type_list):
        if data_type in data_types:
            type_mask = data_types == data_type
            ax1.scatter(pca_embeddings[type_mask, 0], pca_embeddings[type_mask, 1],
                       label=data_type, alpha=0.6, s=20, c=colors[i])
    
    ax1.set_xlabel(f'PC1 ({reduction_results["pca"]["explained_variance"][0]:.1%})')
    ax1.set_ylabel(f'PC2 ({reduction_results["pca"]["explained_variance"][1]:.1%})')
    ax1.set_title('PCA: Batch2 Data Type Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # t-SNE plot
    ax2 = axes[0, 1]
    tsne_embeddings = reduction_results['tsne']['embeddings']
    for i, data_type in enumerate(data_type_list):
        if data_type in data_types:
            type_mask = data_types == data_type
            ax2.scatter(tsne_embeddings[type_mask, 0], tsne_embeddings[type_mask, 1],
                       label=data_type, alpha=0.6, s=20, c=colors[i])
    
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.set_title('t-SNE: Batch2 Data Type Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # t-SNE with uncovered real data highlighted
    ax3 = axes[1, 0]
    
    # Plot synthetic data first
    for i, data_type in enumerate(data_type_list[1:], 1):  # Skip 'real'
        if data_type in data_types:
            type_mask = data_types == data_type
            ax3.scatter(tsne_embeddings[type_mask, 0], tsne_embeddings[type_mask, 1],
                       label=data_type, alpha=0.6, s=20, c=colors[i])
    
    # Plot real data
    real_mask = data_types == 'real'
    normal_real_mask = np.ones(len(tsne_embeddings), dtype=bool)
    
    # Highlight uncovered real data
    if 'real' in uncovered_info:
        uncovered_indices = uncovered_info['real']['indices']
        normal_real_mask[uncovered_indices] = False
        
        # Normal real data
        combined_mask = real_mask & normal_real_mask
        ax3.scatter(tsne_embeddings[combined_mask, 0], tsne_embeddings[combined_mask, 1],
                   label='real (covered)', alpha=0.6, s=20, c='lightblue')
        
        # Uncovered real data
        uncovered_mask = np.zeros(len(tsne_embeddings), dtype=bool)
        uncovered_mask[uncovered_indices] = True
        combined_uncovered_mask = real_mask & uncovered_mask
        
        if np.any(combined_uncovered_mask):
            ax3.scatter(tsne_embeddings[combined_uncovered_mask, 0], tsne_embeddings[combined_uncovered_mask, 1],
                       label='real (uncovered)', alpha=0.9, s=40, c='darkblue', 
                       marker='x', linewidths=2)
    else:
        ax3.scatter(tsne_embeddings[real_mask, 0], tsne_embeddings[real_mask, 1],
                   label='real', alpha=0.6, s=20, c='blue')
    
    ax3.set_xlabel('t-SNE 1')
    ax3.set_ylabel('t-SNE 2')
    ax3.set_title('t-SNE: Batch2 Uncovered Real Data Highlighted')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Coverage analysis bar plot
    ax4 = axes[1, 1]
    
    coverage_data = []
    labels = []
    
    if 'real' in uncovered_info:
        total_real = np.sum(data_types == 'real')
        uncovered_count = uncovered_info['real']['count']
        covered_count = total_real - uncovered_count
        
        coverage_data = [covered_count, uncovered_count]
        labels = ['Covered by\nSynthetic', 'Uncovered by\nSynthetic']
        colors_bar = ['lightgreen', 'red']
        
        bars = ax4.bar(labels, coverage_data, color=colors_bar)
        ax4.set_title('Batch2 Real Data Coverage by Synthetic Data')
        ax4.set_ylabel('Number of Real Samples')
        
        # Add percentage labels
        for bar, count in zip(bars, coverage_data):
            percentage = count / total_real * 100
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{count}\n({percentage:.1f}%)', ha='center', va='bottom')
    else:
        ax4.text(0.5, 0.5, 'No coverage analysis\navailable', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Coverage Analysis')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = EMBEDDINGS_DIR / 'batch2_uncovered_analysis.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Batch2 visualization saved to {plot_file}")
    plt.show()

def main():
    print("=== BATCH 2: EMBEDDING ANALYSIS ===")
    
    # Load data
    combined_data = load_all_batch2_data()
    
    # Generate embeddings
    embedding_data = generate_embeddings(combined_data)
    
    # Perform dimensionality reduction
    reduction_results = perform_dimensionality_reduction(embedding_data['embeddings'])
    
    # Identify uncovered real data
    uncovered_info = identify_uncovered_real_data(
        embedding_data['embeddings'], 
        embedding_data['data_types'],
        method='nearest_neighbor'
    )
    
    # Create visualizations
    create_visualizations(reduction_results, embedding_data['data_types'], uncovered_info)
    
    # Save analysis results
    analysis_results = {
        'embedding_shape': embedding_data['data_shape'],
        'model_name': embedding_data['model_name'],
        'pca_explained_variance': reduction_results['pca']['explained_variance'].tolist(),
        'uncovered_analysis': uncovered_info,
        'data_type_counts': {
            dt: int(np.sum(embedding_data['data_types'] == dt)) 
            for dt in np.unique(embedding_data['data_types'])
        }
    }
    
    with open(EMBEDDINGS_DIR / 'batch2_uncovered_analysis.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\nBatch2 analysis complete:")
    print(f"- Total samples analyzed: {len(embedding_data['embeddings'])}")
    if 'real' in uncovered_info:
        print(f"- Uncovered real data: {uncovered_info['real']['count']} samples")
        total_real = int(np.sum(embedding_data['data_types'] == 'real'))
        coverage_rate = (total_real - uncovered_info['real']['count']) / total_real * 100
        print(f"- Synthetic coverage rate: {coverage_rate:.1f}%")
    print(f"- Results saved to: {EMBEDDINGS_DIR}")

if __name__ == "__main__":
    main()