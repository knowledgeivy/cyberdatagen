#!/usr/bin/env python3
"""
Extract and save the TF-IDF vectorizer from GPT experiments
This ensures SMOTE uses the exact same feature space as GPT/Claude
"""

import sys
import os
import pickle
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from loguru import logger

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.config_manager import load_config


def extract_and_save_vectorizer(config_path: str, group_id: int = 0):
    """
    Extract TF-IDF vectorizer from GPT experiment data and save it

    Args:
        config_path: Path to GPT experiment config
        group_id: Which group to use for fitting (default: 0)
    """
    logger.info("Loading GPT experiment configuration")
    config = load_config(config_path)

    # Load training data from group 0 (same data GPT uses)
    datasets_dir = config.datasets_path
    strategy = 'within_group'  # Any strategy works, we just need the training data
    prompt = 'original'  # Any prompt works
    synthetic_ratio = 0  # Use 0% synthetic (100% real data)
    trial = 0

    # Construct dataset path (matching actual file naming convention)
    dataset_file = os.path.join(
        datasets_dir,
        strategy,
        f"train_{prompt}_r{synthetic_ratio}_g{group_id}_t{trial}.csv.gz"
    )

    logger.info(f"Loading training data from: {dataset_file}")

    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Training data not found: {dataset_file}")

    train_data = pd.read_csv(dataset_file)
    logger.info(f"Loaded {len(train_data)} training samples")

    # Prepare texts (same as GPT's BaseClassifier._prepare_features)
    texts = (train_data['subject'].fillna('') + ' ' + train_data['body'].fillna('')).tolist()
    logger.info(f"Prepared {len(texts)} text samples")

    # Create TF-IDF vectorizer with EXACT same parameters as GPT
    # (from classifiers.py line 66-73)
    tfidf_config = config.feature_extraction.get('tfidf', {})

    vectorizer = TfidfVectorizer(
        max_features=tfidf_config.get('max_features', 10000),
        ngram_range=tuple(tfidf_config.get('ngram_range', [1, 2])),
        stop_words=tfidf_config.get('stop_words', 'english'),
        lowercase=tfidf_config.get('lowercase', True),
        min_df=2,
        max_df=0.95
    )

    logger.info("Fitting TF-IDF vectorizer (same as GPT does)")
    vectorizer.fit(texts)

    vocab_size = len(vectorizer.vocabulary_)
    logger.info(f"Vectorizer fitted: vocabulary size = {vocab_size}")

    # Save vectorizer
    output_dir = os.path.join(config.output.get('results_path', './output/results/'), 'shared_vectorizers')
    os.makedirs(output_dir, exist_ok=True)

    vectorizer_file = os.path.join(output_dir, f'{config.name}_group{group_id}_vectorizer.pkl')

    with open(vectorizer_file, 'wb') as f:
        pickle.dump(vectorizer, f)

    logger.success(f"Vectorizer saved to: {vectorizer_file}")
    logger.info(f"Vocabulary size: {vocab_size}")

    # Save metadata
    metadata = {
        'config_name': config.name,
        'group_id': group_id,
        'vocabulary_size': vocab_size,
        'vectorizer_params': {
            'max_features': vectorizer.max_features,
            'ngram_range': vectorizer.ngram_range,
            'stop_words': vectorizer.stop_words,
            'lowercase': vectorizer.lowercase,
            'min_df': vectorizer.min_df,
            'max_df': vectorizer.max_df
        }
    }

    metadata_file = vectorizer_file.replace('.pkl', '_metadata.json')
    import json
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.success(f"Metadata saved to: {metadata_file}")

    return vectorizer_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Extract GPT vectorizer')
    parser.add_argument('--config', type=str, required=True, help='Path to GPT config file')
    parser.add_argument('--group_id', type=int, default=0, help='Group ID to use (default: 0)')

    args = parser.parse_args()

    vectorizer_file = extract_and_save_vectorizer(args.config, args.group_id)
    print(f"Vectorizer saved to: {vectorizer_file}")
