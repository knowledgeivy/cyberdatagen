# src/cyberdata/scripts/batch1_embedding_analysis.py

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
BATCH1_DIR = PROJECT_ROOT / 'data/batch1'
SEED_SAMPLES_DIR = BATCH1_DIR / 'seed_samples'
SYNTHETIC_DIR = BATCH1_DIR / 'synthetic'
EMBEDDINGS_DIR = BATCH1_DIR / 'embeddings'

# Create directories
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
MODEL_NAME = 'all-MiniLM-L6-v2'
plt.style.use('default')

def load_all_data():
    """Load real seeds and all synthetic variants"""
    print("Loading data...")
    
    # Load real malicious seeds
    real_seeds = pd.read_csv(SEED_SAMPLES_DIR / 'malicious_seeds_1k.csv')
    real_seeds['data_type'] = 'real'
    
    # Load synthetic variants
    datasets = [real_seeds]
    
    for variant in ['rewrite', 'rewrite_strong', 'rewrite_weak']:
        file_path = SYNTHETIC_DIR / f'malicious_{variant}_1k.csv'
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['data_type'] = variant
            datasets.append(df)
        else:
            print(f"Warning: {file_path} not found")
    
    combined_data = pd.concat(datasets, ignore_index=True)
    print(f"Loaded {len(combined_data)} total samples")
    print("Data type distribution:")
    print(combined_data['data_type'].value_counts())
    
    return combined_data

def generate_embeddings(combined_data):
    """Generate embeddings for all data"""
    print("Generating embeddings...")
    
    # Check if embeddings already exist
    embeddings_file = EMBEDDINGS_DIR / 'batch1_embeddings.pkl'
    
    if embeddings_file.exists():
        print("Loading existing embeddings...")
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
    
    print(f"Embeddings saved to {embeddings_file}")
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

def identify_outliers(embeddings, data_types, method='distance', threshold_percentile=95):
    """Identify outliers using distance or clustering methods"""
    print(f"Identifying outliers using {method} method...")
    
    outlier_info = {}
    
    if method == 'distance':
        # For each data type, find samples far from centroid
        for data_type in np.unique(data_types):
            type_mask = data_types == data_type
            type_embeddings = embeddings[type_mask]
            
            if len(type_embeddings) == 0:
                continue
                
            # Calculate centroid
            centroid = np.mean(type_embeddings, axis=0)
            
            # Calculate distances to centroid
            distances = cosine_distances(type_embeddings, centroid.reshape(1, -1)).flatten()
            
            # Find outliers (samples in top percentile of distances)
            threshold = np.percentile(distances, threshold_percentile)
            outlier_mask = distances > threshold
            
            outlier_indices = np.where(type_mask)[0][outlier_mask]
            
            outlier_info[data_type] = {
                'indices': outlier_indices.tolist(),
                'distances': distances[outlier_mask].tolist(),
                'threshold': float(threshold),
                'count': len(outlier_indices)
            }
    
    elif method == 'dbscan':
        # Use DBSCAN to find noise points as outliers
        dbscan = DBSCAN(eps=0.1, min_samples=5, metric='cosine')
        cluster_labels = dbscan.fit_predict(embeddings)
        
        outlier_indices = np.where(cluster_labels == -1)[0]
        
        # Group by data type
        for data_type in np.unique(data_types):
            type_mask = data_types == data_type
            type_outliers = outlier_indices[np.isin(outlier_indices, np.where(type_mask)[0])]
            
            outlier_info[data_type] = {
                'indices': type_outliers.tolist(),
                'count': len(type_outliers),
                'method': 'dbscan'
            }
    
    return outlier_info

def create_visualizations(reduction_results, data_types, outlier_info):
    """Create visualization plots"""
    print("Creating visualizations...")
    
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
    ax1.set_title('PCA: Data Type Distribution')
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
    ax2.set_title('t-SNE: Data Type Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # t-SNE with outliers highlighted
    ax3 = axes[1, 0]
    for i, data_type in enumerate(data_type_list):
        if data_type in data_types:
            type_mask = data_types == data_type
            # Normal points
            normal_mask = np.ones(len(tsne_embeddings), dtype=bool)
            if data_type in outlier_info:
                outlier_indices = outlier_info[data_type]['indices']
                normal_mask[outlier_indices] = False
            
            combined_mask = type_mask & normal_mask
            ax3.scatter(tsne_embeddings[combined_mask, 0], tsne_embeddings[combined_mask, 1],
                       label=f'{data_type} (normal)', alpha=0.6, s=20, c=colors[i])
            
            # Outliers
            if data_type in outlier_info:
                outlier_indices = outlier_info[data_type]['indices']
                outlier_mask = np.zeros(len(tsne_embeddings), dtype=bool)
                outlier_mask[outlier_indices] = True
                combined_outlier_mask = type_mask & outlier_mask
                
                if np.any(combined_outlier_mask):
                    ax3.scatter(tsne_embeddings[combined_outlier_mask, 0], tsne_embeddings[combined_outlier_mask, 1],
                               label=f'{data_type} (outlier)', alpha=0.8, s=40, c=colors[i], 
                               marker='x', linewidths=2)
    
    ax3.set_xlabel('t-SNE 1')
    ax3.set_ylabel('t-SNE 2')
    ax3.set_title('t-SNE: Outliers Highlighted')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Outlier counts bar plot
    ax4 = axes[1, 1]
    outlier_counts = [outlier_info.get(dt, {'count': 0})['count'] for dt in data_type_list]
    bars = ax4.bar(data_type_list, outlier_counts, color=colors)
    ax4.set_title('Outlier Counts by Data Type')
    ax4.set_ylabel('Number of Outliers')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, outlier_counts):
        if count > 0:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = EMBEDDINGS_DIR / 'batch1_embedding_analysis.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {plot_file}")
    plt.show()

def main():
    print("=== BATCH 1: EMBEDDING ANALYSIS ===")
    
    # Load data
    combined_data = load_all_data()
    
    # Generate embeddings
    embedding_data = generate_embeddings(combined_data)
    
    # Perform dimensionality reduction
    reduction_results = perform_dimensionality_reduction(embedding_data['embeddings'])
    
    # Identify outliers
    outlier_info = identify_outliers(
        embedding_data['embeddings'], 
        embedding_data['data_types'],
        method='distance',
        threshold_percentile=90
    )
    
    # Create visualizations
    create_visualizations(reduction_results, embedding_data['data_types'], outlier_info)
    
    # Save analysis results
    analysis_results = {
        'embedding_shape': embedding_data['data_shape'],
        'model_name': embedding_data['model_name'],
        'pca_explained_variance': reduction_results['pca']['explained_variance'].tolist(),
        'outlier_analysis': outlier_info,
        'data_type_counts': {
            dt: int(np.sum(embedding_data['data_types'] == dt)) 
            for dt in np.unique(embedding_data['data_types'])
        }
    }
    
    with open(EMBEDDINGS_DIR / 'batch1_analysis_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\nAnalysis complete:")
    print(f"- Total samples analyzed: {len(embedding_data['embeddings'])}")
    print(f"- Outliers found:")
    for data_type, info in outlier_info.items():
        print(f"  {data_type}: {info['count']} outliers")
    print(f"- Results saved to: {EMBEDDINGS_DIR}")

if __name__ == "__main__":
    main()