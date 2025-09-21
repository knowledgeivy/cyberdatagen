#!/usr/bin/env python3
"""
Batch 6: Comprehensive Embedding Metrics Analysis
================================================

Calculate comprehensive embedding-based metrics to quantify the quality and
effectiveness of synthetic data generation strategies.

Metrics included:
1. Centroid Distance - Overall distributional shift
2. Mean Cosine Similarity - Semantic preservation
3. Co-clustering Rate - Structural similarity
4. Semantic Drift Score - Deviation from original space
5. Intra-strategy Diversity - Internal variation
6. Distribution Coverage - Real data distribution coverage
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import pickle
from sklearn.metrics import pairwise_distances, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, cdist
from scipy.stats import entropy, wasserstein_distance
import warnings
warnings.filterwarnings('ignore')

# Setup project root path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

def setup_logger(name: str) -> logging.Logger:
    """Setup logger for the analysis"""
    log_dir = PROJECT_ROOT / "data" / "batch6" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler
    file_handler = logging.FileHandler(
        log_dir / f"batch6_embedding_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

class ComprehensiveEmbeddingMetrics:
    """Calculate comprehensive metrics for embedding quality assessment"""

    def __init__(self):
        self.logger = setup_logger("batch6_embedding_metrics")
        self.setup_paths()

    def setup_paths(self) -> None:
        """Setup directory paths"""
        self.batch6_dir = PROJECT_ROOT / "data" / "batch6"
        self.input_dir = self.batch6_dir / "phase5a_processed_data"
        self.output_dir = self.batch6_dir / "comprehensive_metrics"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")

    def load_data(self) -> Tuple[np.ndarray, pd.DataFrame, Dict]:
        """Load processed embeddings, metadata, and clustering results"""
        try:
            self.logger.info("Loading processed data...")

            # Load embeddings
            embeddings = np.load(self.input_dir / "embeddings.npy")

            # Load metadata
            metadata = pd.read_csv(self.input_dir / "metadata.csv.gz", compression='gzip')

            # Load clustering results
            with open(self.input_dir / "clustering_results.json", 'r') as f:
                clustering_results = json.load(f)

            self.logger.info(f"✅ Loaded embeddings: {embeddings.shape}")
            self.logger.info(f"✅ Loaded metadata: {len(metadata)} records")
            self.logger.info(f"Data sources: {metadata['data_source'].value_counts().to_dict()}")

            return embeddings, metadata, clustering_results

        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise

    def prepare_data_groups(self, embeddings: np.ndarray, metadata: pd.DataFrame) -> Dict[str, Dict]:
        """Prepare data groups for analysis"""
        try:
            self.logger.info("Preparing data groups...")

            data_groups = {}

            # Real malicious data (seeds)
            real_mask = metadata['data_source'] == 'seeds'
            data_groups['real'] = {
                'embeddings': embeddings[real_mask],
                'metadata': metadata[real_mask],
                'indices': np.where(real_mask)[0]
            }

            # Synthetic data by strategy
            synthetic_mask = metadata['data_source'] == 'batch6_synthetic'
            synthetic_meta = metadata[synthetic_mask]

            if 'prompt_variant' in synthetic_meta.columns:
                for strategy in ['original', 'strong', 'weak']:
                    strategy_mask = synthetic_meta['prompt_variant'] == strategy
                    if strategy_mask.any():
                        full_indices = np.where(synthetic_mask)[0][strategy_mask]
                        data_groups[f'synthetic_{strategy}'] = {
                            'embeddings': embeddings[full_indices],
                            'metadata': metadata.loc[full_indices],
                            'indices': full_indices
                        }
            else:
                # Fallback: treat all synthetic as one group
                data_groups['synthetic_all'] = {
                    'embeddings': embeddings[synthetic_mask],
                    'metadata': synthetic_meta,
                    'indices': np.where(synthetic_mask)[0]
                }

            # Log group sizes
            for group_name, group_data in data_groups.items():
                self.logger.info(f"{group_name}: {len(group_data['embeddings'])} samples")

            return data_groups

        except Exception as e:
            self.logger.error(f"Failed to prepare data groups: {str(e)}")
            raise

    def calculate_centroid_distances(self, data_groups: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate distances between centroids"""
        try:
            self.logger.info("Calculating centroid distances...")

            # Calculate centroids
            centroids = {}
            for group_name, group_data in data_groups.items():
                centroids[group_name] = np.mean(group_data['embeddings'], axis=0)

            # Calculate distances from real centroid
            real_centroid = centroids['real']
            distances = {}

            for group_name, centroid in centroids.items():
                if group_name != 'real':
                    distance = np.linalg.norm(centroid - real_centroid)
                    distances[group_name] = distance

            self.logger.info("✅ Centroid distances calculated")
            return distances

        except Exception as e:
            self.logger.error(f"Failed centroid distance calculation: {str(e)}")
            return {}

    def calculate_cosine_similarities(self, data_groups: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate mean cosine similarities to real data"""
        try:
            self.logger.info("Calculating cosine similarities...")

            real_embeddings = data_groups['real']['embeddings']
            similarities = {}

            for group_name, group_data in data_groups.items():
                if group_name != 'real':
                    group_embeddings = group_data['embeddings']

                    # Calculate pairwise cosine similarities
                    cos_sim_matrix = cosine_similarity(group_embeddings, real_embeddings)

                    # Take mean of maximum similarities (each synthetic to closest real)
                    max_similarities = np.max(cos_sim_matrix, axis=1)
                    mean_similarity = np.mean(max_similarities)

                    similarities[group_name] = mean_similarity

            self.logger.info("✅ Cosine similarities calculated")
            return similarities

        except Exception as e:
            self.logger.error(f"Failed cosine similarity calculation: {str(e)}")
            return {}

    def calculate_co_clustering_rates(self, data_groups: Dict[str, Dict],
                                    clustering_results: Dict) -> Dict[str, float]:
        """Calculate co-clustering rates with real data"""
        try:
            self.logger.info("Calculating co-clustering rates...")

            # Use optimal K from clustering results
            optimal_k = clustering_results['optimal_k']
            if optimal_k is None:
                self.logger.warning("No optimal K found, using K=19")
                optimal_k = 19

            cluster_labels = np.array(clustering_results['kmeans'][str(optimal_k)]['labels'])

            real_indices = data_groups['real']['indices']
            real_clusters = cluster_labels[real_indices]

            co_clustering_rates = {}

            for group_name, group_data in data_groups.items():
                if group_name != 'real':
                    group_indices = group_data['indices']
                    group_clusters = cluster_labels[group_indices]

                    # Calculate co-clustering rate
                    co_clustered = 0
                    total_pairs = 0

                    for i, synthetic_cluster in enumerate(group_clusters):
                        for real_cluster in real_clusters:
                            total_pairs += 1
                            if synthetic_cluster == real_cluster:
                                co_clustered += 1

                    co_clustering_rate = co_clustered / total_pairs if total_pairs > 0 else 0
                    co_clustering_rates[group_name] = co_clustering_rate

            self.logger.info("✅ Co-clustering rates calculated")
            return co_clustering_rates

        except Exception as e:
            self.logger.error(f"Failed co-clustering calculation: {str(e)}")
            return {}

    def calculate_semantic_drift(self, data_groups: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate semantic drift from original distribution"""
        try:
            self.logger.info("Calculating semantic drift scores...")

            real_embeddings = data_groups['real']['embeddings']
            real_centroid = np.mean(real_embeddings, axis=0)

            # Calculate real data spread (standard deviation from centroid)
            real_distances = np.linalg.norm(real_embeddings - real_centroid, axis=1)
            real_std = np.std(real_distances)

            drift_scores = {}

            for group_name, group_data in data_groups.items():
                if group_name != 'real':
                    group_embeddings = group_data['embeddings']

                    # Calculate distances from real centroid
                    distances_to_real_centroid = np.linalg.norm(
                        group_embeddings - real_centroid, axis=1
                    )

                    # Semantic drift = normalized deviation from real distribution
                    mean_distance = np.mean(distances_to_real_centroid)
                    drift_score = mean_distance / real_std if real_std > 0 else 0

                    drift_scores[group_name] = drift_score

            self.logger.info("✅ Semantic drift scores calculated")
            return drift_scores

        except Exception as e:
            self.logger.error(f"Failed semantic drift calculation: {str(e)}")
            return {}

    def calculate_intra_strategy_diversity(self, data_groups: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate diversity within each strategy"""
        try:
            self.logger.info("Calculating intra-strategy diversity...")

            diversity_scores = {}

            for group_name, group_data in data_groups.items():
                group_embeddings = group_data['embeddings']

                if len(group_embeddings) > 1:
                    # Calculate pairwise distances within group
                    pairwise_dists = pdist(group_embeddings, metric='euclidean')
                    mean_diversity = np.mean(pairwise_dists)
                    diversity_scores[group_name] = mean_diversity
                else:
                    diversity_scores[group_name] = 0.0

            self.logger.info("✅ Intra-strategy diversity calculated")
            return diversity_scores

        except Exception as e:
            self.logger.error(f"Failed diversity calculation: {str(e)}")
            return {}

    def calculate_distribution_coverage(self, data_groups: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate how well synthetic data covers real data distribution"""
        try:
            self.logger.info("Calculating distribution coverage...")

            real_embeddings = data_groups['real']['embeddings']
            coverage_scores = {}

            for group_name, group_data in data_groups.items():
                if group_name != 'real':
                    group_embeddings = group_data['embeddings']

                    # For each real sample, find closest synthetic sample
                    distances = cdist(real_embeddings, group_embeddings, metric='euclidean')
                    min_distances = np.min(distances, axis=1)

                    # Coverage = 1 / (1 + mean_minimum_distance)
                    # Higher coverage when synthetic samples are close to real samples
                    mean_min_distance = np.mean(min_distances)
                    coverage = 1 / (1 + mean_min_distance)

                    coverage_scores[group_name] = coverage

            self.logger.info("✅ Distribution coverage calculated")
            return coverage_scores

        except Exception as e:
            self.logger.error(f"Failed coverage calculation: {str(e)}")
            return {}

    def compile_comprehensive_metrics(self, centroid_distances: Dict, cosine_similarities: Dict,
                                    co_clustering_rates: Dict, semantic_drift: Dict,
                                    diversity_scores: Dict, coverage_scores: Dict) -> pd.DataFrame:
        """Compile all metrics into a comprehensive table"""
        try:
            self.logger.info("Compiling comprehensive metrics table...")

            # Get all synthetic strategy names
            strategy_names = set()
            for metric_dict in [centroid_distances, cosine_similarities, co_clustering_rates,
                              semantic_drift, diversity_scores, coverage_scores]:
                strategy_names.update(metric_dict.keys())

            # Create comprehensive metrics table
            metrics_data = []

            for strategy in sorted(strategy_names):
                row = {
                    'Strategy': strategy.replace('synthetic_', '').title(),
                    'Centroid_Distance': centroid_distances.get(strategy, np.nan),
                    'Mean_Cosine_Similarity': cosine_similarities.get(strategy, np.nan),
                    'Co_Clustering_Rate': co_clustering_rates.get(strategy, np.nan),
                    'Semantic_Drift_Score': semantic_drift.get(strategy, np.nan),
                    'Intra_Strategy_Diversity': diversity_scores.get(strategy, np.nan),
                    'Distribution_Coverage': coverage_scores.get(strategy, np.nan)
                }
                metrics_data.append(row)

            # Add real data reference row for comparison
            real_row = {
                'Strategy': 'Real (Reference)',
                'Centroid_Distance': 0.0,  # Distance to itself
                'Mean_Cosine_Similarity': 1.0,  # Perfect similarity to itself
                'Co_Clustering_Rate': 1.0,  # Perfect co-clustering with itself
                'Semantic_Drift_Score': 1.0,  # Baseline drift score
                'Intra_Strategy_Diversity': diversity_scores.get('real', np.nan),
                'Distribution_Coverage': 1.0  # Perfect coverage of itself
            }
            metrics_data.append(real_row)

            metrics_df = pd.DataFrame(metrics_data)

            self.logger.info("✅ Comprehensive metrics table compiled")
            return metrics_df

        except Exception as e:
            self.logger.error(f"Failed metrics compilation: {str(e)}")
            raise

    def save_results(self, metrics_df: pd.DataFrame, individual_metrics: Dict) -> None:
        """Save comprehensive metrics results"""
        try:
            self.logger.info("Saving comprehensive metrics results...")

            # Save comprehensive metrics table
            metrics_df.to_csv(self.output_dir / "comprehensive_embedding_metrics.csv", index=False)

            # Save individual metric dictionaries
            with open(self.output_dir / "individual_metrics.json", 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                json_compatible_metrics = {}
                for metric_name, metric_dict in individual_metrics.items():
                    json_compatible_metrics[metric_name] = {
                        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for k, v in metric_dict.items()
                    }
                json.dump(json_compatible_metrics, f, indent=2)

            # Create summary report
            summary_report = {
                'analysis_completion': {
                    'status': 'completed',
                    'completion_date': datetime.now().isoformat(),
                    'metrics_calculated': len(individual_metrics),
                    'strategies_analyzed': len(metrics_df) - 1  # Exclude real reference
                },
                'metric_definitions': {
                    'Centroid_Distance': 'Euclidean distance between strategy centroid and real data centroid (lower is better)',
                    'Mean_Cosine_Similarity': 'Average maximum cosine similarity to real samples (higher is better)',
                    'Co_Clustering_Rate': 'Rate of co-occurrence in same clusters as real data (higher is better)',
                    'Semantic_Drift_Score': 'Normalized deviation from real distribution (lower is better)',
                    'Intra_Strategy_Diversity': 'Average pairwise distance within strategy (higher indicates more diversity)',
                    'Distribution_Coverage': 'How well synthetic data covers real data distribution (higher is better)'
                },
                'key_findings': {
                    'best_overall_strategy': None,  # To be filled based on analysis
                    'metric_correlations': None     # To be filled based on analysis
                }
            }

            # Identify best performing strategy (excluding real reference)
            if len(metrics_df) > 1:
                synthetic_metrics = metrics_df[metrics_df['Strategy'] != 'Real (Reference)'].copy()

                # Calculate composite score (normalize metrics and combine)
                # Note: Some metrics are "higher is better", others are "lower is better"
                normalized_metrics = synthetic_metrics.copy()

                # Normalize "higher is better" metrics (0-1 scale)
                for col in ['Mean_Cosine_Similarity', 'Co_Clustering_Rate', 'Distribution_Coverage']:
                    if col in normalized_metrics.columns:
                        max_val = normalized_metrics[col].max()
                        min_val = normalized_metrics[col].min()
                        if max_val > min_val:
                            normalized_metrics[col] = (normalized_metrics[col] - min_val) / (max_val - min_val)

                # Normalize "lower is better" metrics (invert scale)
                for col in ['Centroid_Distance', 'Semantic_Drift_Score']:
                    if col in normalized_metrics.columns:
                        max_val = normalized_metrics[col].max()
                        min_val = normalized_metrics[col].min()
                        if max_val > min_val:
                            normalized_metrics[col] = 1 - (normalized_metrics[col] - min_val) / (max_val - min_val)

                # Calculate composite score
                score_columns = ['Mean_Cosine_Similarity', 'Co_Clustering_Rate', 'Distribution_Coverage',
                               'Centroid_Distance', 'Semantic_Drift_Score']
                available_columns = [col for col in score_columns if col in normalized_metrics.columns]

                if available_columns:
                    normalized_metrics['Composite_Score'] = normalized_metrics[available_columns].mean(axis=1)
                    best_strategy_idx = normalized_metrics['Composite_Score'].idxmax()
                    best_strategy = synthetic_metrics.loc[best_strategy_idx, 'Strategy']
                    summary_report['key_findings']['best_overall_strategy'] = best_strategy

            with open(self.output_dir / "comprehensive_metrics_summary.json", 'w') as f:
                json.dump(summary_report, f, indent=2)

            self.logger.info("✅ All results saved successfully")
            self.logger.info(f"Output directory: {self.output_dir}")

        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            raise

    def run_comprehensive_analysis(self) -> None:
        """Execute complete comprehensive embedding metrics analysis"""
        try:
            self.logger.info("🚀 Starting Comprehensive Embedding Metrics Analysis")
            start_time = datetime.now()

            # Step 1: Load data
            embeddings, metadata, clustering_results = self.load_data()

            # Step 2: Prepare data groups
            data_groups = self.prepare_data_groups(embeddings, metadata)

            # Step 3: Calculate individual metrics
            self.logger.info("Calculating individual metrics...")

            centroid_distances = self.calculate_centroid_distances(data_groups)
            cosine_similarities = self.calculate_cosine_similarities(data_groups)
            co_clustering_rates = self.calculate_co_clustering_rates(data_groups, clustering_results)
            semantic_drift = self.calculate_semantic_drift(data_groups)
            diversity_scores = self.calculate_intra_strategy_diversity(data_groups)
            coverage_scores = self.calculate_distribution_coverage(data_groups)

            # Step 4: Compile comprehensive metrics
            metrics_df = self.compile_comprehensive_metrics(
                centroid_distances, cosine_similarities, co_clustering_rates,
                semantic_drift, diversity_scores, coverage_scores
            )

            # Step 5: Save results
            individual_metrics = {
                'centroid_distances': centroid_distances,
                'cosine_similarities': cosine_similarities,
                'co_clustering_rates': co_clustering_rates,
                'semantic_drift': semantic_drift,
                'diversity_scores': diversity_scores,
                'coverage_scores': coverage_scores
            }

            self.save_results(metrics_df, individual_metrics)

            # Analysis completion
            end_time = datetime.now()
            runtime = (end_time - start_time).total_seconds() / 60

            self.logger.info(f"✅ Comprehensive Embedding Metrics Analysis completed!")
            self.logger.info(f"Runtime: {runtime:.2f} minutes")
            self.logger.info(f"Analyzed {len(embeddings)} samples across {len(data_groups)} groups")
            self.logger.info(f"Generated comprehensive metrics for {len(metrics_df)-1} synthetic strategies")

            # Display summary results
            self.logger.info("\n" + "="*60)
            self.logger.info("COMPREHENSIVE METRICS SUMMARY")
            self.logger.info("="*60)
            print(metrics_df.round(4))

        except Exception as e:
            self.logger.error(f"❌ Comprehensive analysis failed: {str(e)}")
            raise

def main():
    """Main function to run comprehensive embedding metrics analysis"""
    import argparse

    parser = argparse.ArgumentParser(description='Comprehensive Embedding Metrics Analysis')
    parser.add_argument('--clean', action='store_true', help='Clean previous results before running')
    args = parser.parse_args()

    logger = setup_logger("batch6_embedding_metrics")

    try:
        # Handle clean option
        if args.clean:
            logger.info("🧹 Cleaning previous results...")

            batch6_dir = PROJECT_ROOT / "data" / "batch6"
            clean_dir = batch6_dir / "comprehensive_metrics"

            if clean_dir.exists():
                import shutil
                shutil.rmtree(clean_dir)
                logger.info(f"   Removed: {clean_dir}")

            logger.info("✅ Cleanup completed!")

        # Run comprehensive analysis
        analyzer = ComprehensiveEmbeddingMetrics()
        analyzer.run_comprehensive_analysis()

    except Exception as e:
        logger.error(f"Analysis execution failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()