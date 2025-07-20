# src/cyberdata/scripts/batch2_prepare_enhanced_data.py

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_distances
import pickle

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
BASE_SAMPLES_DIR = BATCH1_DIR / 'base_samples'
EMBEDDINGS_DIR = BATCH1_DIR / 'embeddings'
ENHANCED_SEEDS_DIR = BATCH2_DIR / 'enhanced_seeds'

# Create directories
ENHANCED_SEEDS_DIR.mkdir(parents=True, exist_ok=True)

def load_uncovered_analysis():
    """Load uncovered real data analysis from batch1"""
    print("Loading uncovered analysis from batch1...")
    
    analysis_file = EMBEDDINGS_DIR / 'batch1_uncovered_analysis.json'
    if not analysis_file.exists():
        print(f"Error: Analysis file not found: {analysis_file}")
        print("Please run batch1_embedding_analysis.py first")
        return None
    
    with open(analysis_file, 'r') as f:
        analysis = json.load(f)
    
    return analysis

def load_embeddings_data():
    """Load embedding data from batch1"""
    print("Loading embedding data...")
    
    embeddings_file = EMBEDDINGS_DIR / 'batch1_embeddings.pkl'
    if not embeddings_file.exists():
        print(f"Error: Embeddings file not found: {embeddings_file}")
        return None
    
    with open(embeddings_file, 'rb') as f:
        embedding_data = pickle.load(f)
    
    return embedding_data

def get_uncovered_real_data_ids(analysis):
    """Extract data IDs of uncovered real data"""
    print("Extracting uncovered real data IDs...")
    
    uncovered_indices = []
    
    # Get uncovered indices from analysis
    if 'uncovered_analysis' in analysis and 'real' in analysis['uncovered_analysis']:
        uncovered_indices = analysis['uncovered_analysis']['real']['indices']
        print(f"Found {len(uncovered_indices)} uncovered real data points")
    else:
        print("No uncovered real data found in analysis")
        return []
    
    return uncovered_indices

def find_similar_samples(uncovered_indices, embedding_data, base_malicious_data, target_count=1000):
    """Find similar samples to uncovered data points"""
    print(f"Finding similar samples to expand uncovered data to {target_count} samples...")
    
    embeddings = embedding_data['embeddings']
    data_types = embedding_data['data_types']
    data_ids = embedding_data['data_ids']
    
    # Get real data mask and embeddings
    real_mask = data_types == 'real'
    real_embeddings = embeddings[real_mask]
    real_data_ids = data_ids[real_mask]
    
    # Get uncovered embeddings
    uncovered_embeddings = embeddings[uncovered_indices]
    uncovered_data_ids = data_ids[uncovered_indices]
    
    print(f"Uncovered samples: {len(uncovered_data_ids)}")
    
    # If we have enough uncovered samples, just return them
    if len(uncovered_data_ids) >= target_count:
        selected_ids = uncovered_data_ids[:target_count].tolist()
        print(f"Using first {target_count} uncovered samples")
        return selected_ids
    
    # Need to find similar samples from the 5K base data
    additional_needed = target_count - len(uncovered_data_ids)
    print(f"Need {additional_needed} additional similar samples")
    
    # Map data_ids to indices in base_malicious_data
    base_id_to_idx = {row['data_id']: idx for idx, row in base_malicious_data.iterrows()}
    
    # Get embeddings for all base malicious data
    base_embeddings = []
    base_data_ids_available = []
    for data_id in base_malicious_data['data_id']:
        if data_id in dict(zip(real_data_ids, range(len(real_data_ids)))):
            idx_in_real = list(real_data_ids).index(data_id)
            base_embeddings.append(real_embeddings[idx_in_real])
            base_data_ids_available.append(data_id)
    
    base_embeddings = np.array(base_embeddings)
    
    # Calculate distances from uncovered samples to all base samples
    if len(uncovered_embeddings) > 0 and len(base_embeddings) > 0:
        # For each base sample, find distance to nearest uncovered sample
        distances_matrix = cosine_distances(base_embeddings, uncovered_embeddings)
        min_distances = np.min(distances_matrix, axis=1)
        
        # Exclude already selected uncovered samples
        available_mask = np.ones(len(base_data_ids_available), dtype=bool)
        for uncovered_id in uncovered_data_ids:
            if uncovered_id in base_data_ids_available:
                idx = base_data_ids_available.index(uncovered_id)
                available_mask[idx] = False
        
        # Get available samples and their distances
        available_distances = min_distances[available_mask]
        available_ids = [base_data_ids_available[i] for i, mask in enumerate(available_mask) if mask]
        
        # Sort by distance (closest to uncovered samples first)
        sorted_indices = np.argsort(available_distances)
        additional_ids = [available_ids[i] for i in sorted_indices[:additional_needed]]
        
        print(f"Found {len(additional_ids)} additional similar samples")
    else:
        additional_ids = []
    
    # Combine uncovered and additional similar samples
    selected_ids = uncovered_data_ids.tolist() + additional_ids
    
    print(f"Total enhanced seed samples: {len(selected_ids)}")
    return selected_ids

def create_enhanced_seed_samples(selected_ids, base_malicious_data, base_benign_data):
    """Create enhanced seed sample files"""
    print("Creating enhanced seed sample files...")
    
    # Create enhanced malicious seeds
    enhanced_malicious = base_malicious_data[base_malicious_data['data_id'].isin(selected_ids)].copy()
    enhanced_malicious = enhanced_malicious.reset_index(drop=True)
    
    # Use same benign seeds as batch1 for consistency
    batch1_benign_seeds = pd.read_csv(BATCH1_DIR / 'seed_samples/benign_seeds_1k.csv')
    
    # Save enhanced seed samples
    enhanced_malicious.to_csv(ENHANCED_SEEDS_DIR / 'malicious_enhanced_seeds.csv', index=False)
    batch1_benign_seeds.to_csv(ENHANCED_SEEDS_DIR / 'benign_seeds_1k.csv', index=False)
    
    print(f"Enhanced malicious seeds: {len(enhanced_malicious)}")
    print(f"Benign seeds (reused): {len(batch1_benign_seeds)}")
    
    return enhanced_malicious, batch1_benign_seeds

def save_enhancement_metadata(analysis, selected_ids, enhanced_malicious):
    """Save metadata about the enhancement process"""
    print("Saving enhancement metadata...")
    
    # Create mapping of enhancement reasons
    uncovered_indices = []
    if 'uncovered_analysis' in analysis and 'real' in analysis['uncovered_analysis']:
        uncovered_indices = analysis['uncovered_analysis']['real']['indices']
    
    # Load embedding data to get data_ids
    embedding_data = load_embeddings_data()
    uncovered_data_ids = []
    if embedding_data:
        data_ids = embedding_data['data_ids']
        uncovered_data_ids = [data_ids[i] for i in uncovered_indices]
    
    # Categorize selected samples
    sample_categories = {}
    for data_id in selected_ids:
        if data_id in uncovered_data_ids:
            sample_categories[data_id] = 'uncovered_by_synthetic'
        else:
            sample_categories[data_id] = 'similar_to_uncovered'
    
    metadata = {
        'experiment': 'batch2_enhanced_seeds',
        'enhancement_timestamp': pd.Timestamp.now().isoformat(),
        'batch1_analysis_used': True,
        'total_selected': len(selected_ids),
        'uncovered_count': len([cat for cat in sample_categories.values() if cat == 'uncovered_by_synthetic']),
        'similar_count': len([cat for cat in sample_categories.values() if cat == 'similar_to_uncovered']),
        'selection_strategy': 'uncovered_plus_similar',
        'sample_categories': sample_categories,
        'enhancement_source': {
            'batch1_uncovered_analysis': 'batch1_uncovered_analysis.json',
            'base_malicious_data': 'malicious_5k.csv',
            'similarity_metric': 'cosine_distance'
        }
    }
    
    with open(BATCH2_DIR / 'batch2_enhancement_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Enhancement metadata saved to {BATCH2_DIR}")

def main():
    print("=== BATCH 2: PREPARE ENHANCED DATA ===")
    
    # Load uncovered analysis from batch1
    analysis = load_uncovered_analysis()
    if analysis is None:
        return
    
    # Load embedding data
    embedding_data = load_embeddings_data()
    if embedding_data is None:
        return
    
    # Load base data
    print("Loading base data...")
    base_malicious_data = pd.read_csv(BASE_SAMPLES_DIR / 'malicious_5k.csv')
    base_benign_data = pd.read_csv(BASE_SAMPLES_DIR / 'benign_5k.csv')
    
    # Get uncovered real data IDs
    uncovered_indices = get_uncovered_real_data_ids(analysis)
    
    if len(uncovered_indices) == 0:
        print("No uncovered real data found. Using random sampling as fallback.")
        # Fallback: random sampling
        np.random.seed(42)
        selected_ids = base_malicious_data.sample(n=1000, random_state=42)['data_id'].tolist()
    else:
        # Find similar samples to expand the uncovered data
        selected_ids = find_similar_samples(uncovered_indices, embedding_data, base_malicious_data)
    
    # Create enhanced seed samples
    enhanced_malicious, enhanced_benign = create_enhanced_seed_samples(
        selected_ids, base_malicious_data, base_benign_data
    )
    
    # Save metadata
    save_enhancement_metadata(analysis, selected_ids, enhanced_malicious)
    
    print(f"\nEnhanced seed preparation complete:")
    print(f"- Enhanced malicious seeds: {len(enhanced_malicious)}")
    print(f"- Benign seeds: {len(enhanced_benign)}")
    print(f"- Output directory: {ENHANCED_SEEDS_DIR}")
    
    # Show sample enhancement reasons
    if len(uncovered_indices) > 0:
        uncovered_count = len([sid for sid in selected_ids 
                              if sid in [embedding_data['data_ids'][i] for i in uncovered_indices]])
        similar_count = len(selected_ids) - uncovered_count
        print(f"- Uncovered samples: {uncovered_count}")
        print(f"- Similar samples: {similar_count}")

if __name__ == "__main__":
    main()