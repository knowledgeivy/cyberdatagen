"""
SMOTE-based synthetic data generator for imbalanced spam email datasets
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE, ADASYN
from loguru import logger


class SMOTEGenerator:
    """
    SMOTE/ADASYN-based synthetic data generator

    Unlike LLM-based generators that produce text, SMOTE operates in feature space:
    Text → TF-IDF Features → SMOTE → Synthetic Features → Classification

    This is the standard baseline for imbalanced spam detection.
    """

    def __init__(
        self,
        method: str = 'smote',
        vectorizer_params: Optional[Dict[str, Any]] = None,
        smote_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42
    ):
        """
        Initialize SMOTE generator

        Args:
            method: 'smote' or 'adasyn'
            vectorizer_params: TfidfVectorizer parameters
            smote_params: SMOTE/ADASYN parameters
            random_state: Random seed for reproducibility
        """
        self.method = method.lower()
        self.random_state = random_state

        # Default vectorizer parameters (consistent with classification pipeline)
        default_vectorizer_params = {
            'max_features': 10000,
            'ngram_range': (1, 2),
            'min_df': 2,
            'max_df': 0.95,
            'sublinear_tf': True
        }
        self.vectorizer_params = {**default_vectorizer_params, **(vectorizer_params or {})}

        # Default SMOTE parameters
        default_smote_params = {
            'sampling_strategy': 'auto',
            'k_neighbors': 5,
            'random_state': random_state
        }
        self.smote_params = {**default_smote_params, **(smote_params or {})}

        # Initialize components
        self.vectorizer = TfidfVectorizer(**self.vectorizer_params)
        self.sampler = self._init_sampler()

        # Track fitted state
        self.is_fitted = False

        logger.info(f"Initialized {method.upper()} generator with params: {self.smote_params}")

    def _init_sampler(self):
        """Initialize SMOTE or ADASYN sampler"""
        if self.method == 'smote':
            return SMOTE(**self.smote_params)
        elif self.method == 'adasyn':
            return ADASYN(**self.smote_params)
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'smote' or 'adasyn'")

    def generate_synthetic_samples(
        self,
        texts: pd.Series,
        labels: pd.Series,
        target_count: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic samples using SMOTE/ADASYN

        Args:
            texts: Original spam email texts
            labels: Original labels (1 for spam)
            target_count: Desired number of synthetic spam samples

        Returns:
            Tuple of (synthetic_features, synthetic_labels)

        Note: Returns features, not text (SMOTE operates in feature space)
        """
        logger.info(f"Generating {target_count} synthetic samples using {self.method.upper()}")

        # Extract features
        if not self.is_fitted:
            X_features = self.vectorizer.fit_transform(texts)
            self.is_fitted = True
        else:
            X_features = self.vectorizer.transform(texts)

        logger.info(f"Extracted features: {X_features.shape}")

        # Calculate required sampling strategy
        # SMOTE expects ratio of minority to majority after resampling
        n_spam = (labels == 1).sum()
        n_ham = (labels == 0).sum()

        desired_spam_count = n_spam + target_count
        sampling_ratio = desired_spam_count / n_ham if n_ham > 0 else 1.0

        logger.info(f"Original distribution: {n_spam} spam, {n_ham} ham")
        logger.info(f"Target: {desired_spam_count} spam (sampling_ratio: {sampling_ratio:.3f})")

        # Apply SMOTE/ADASYN
        self.sampler.sampling_strategy = sampling_ratio
        X_resampled, y_resampled = self.sampler.fit_resample(X_features, labels)

        # Extract only the synthetic samples (new samples added by SMOTE)
        n_original = X_features.shape[0]
        X_synthetic = X_resampled[n_original:]
        y_synthetic = y_resampled[n_original:]

        # Filter to get only spam synthetic samples
        spam_mask = y_synthetic == 1
        X_synthetic_spam = X_synthetic[spam_mask]
        y_synthetic_spam = y_synthetic[spam_mask]

        # Limit to target count
        if X_synthetic_spam.shape[0] > target_count:
            indices = np.random.choice(X_synthetic_spam.shape[0], target_count, replace=False)
            X_synthetic_spam = X_synthetic_spam[indices]
            y_synthetic_spam = y_synthetic_spam[indices]

        logger.info(f"Generated {X_synthetic_spam.shape[0]} synthetic spam samples")
        logger.success(f"{self.method.upper()} generation completed")

        return X_synthetic_spam, y_synthetic_spam

    def build_training_dataset(
        self,
        real_texts: pd.Series,
        real_labels: pd.Series,
        synthetic_ratio: float,
        strategy: str = 'within_group'
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Build training dataset with specified synthetic ratio

        Args:
            real_texts: Real spam email texts (all classes)
            real_labels: Real labels (0=ham, 1=spam)
            synthetic_ratio: Percentage of synthetic spam vs real spam (0-100)
            strategy: 'within_group' or 'cross_group' (for consistency)

        Returns:
            Tuple of (X_train_features, y_train, metadata)
        """
        logger.info(f"Building training dataset with {synthetic_ratio}% synthetic data")

        # Extract real features for ALL data (ham + spam)
        X_real = self.vectorizer.fit_transform(real_texts)
        self.is_fitted = True
        y_real = real_labels.values

        # Handle 0% synthetic case
        if synthetic_ratio == 0:
            metadata = {
                'n_real': X_real.shape[0],
                'n_real_spam': (y_real == 1).sum(),
                'n_real_ham': (y_real == 0).sum(),
                'n_synthetic': 0,
                'synthetic_ratio': 0,
                'method': self.method,
                'strategy': strategy
            }
            return X_real, y_real, metadata

        # Calculate number of synthetic spam samples needed
        n_real_spam = (y_real == 1).sum()

        # Handle 100% case specially (all synthetic, no real)
        # In SMOTE context, 100% means "maximum synthetic" = 10x real spam
        if synthetic_ratio >= 100:
            n_synthetic_spam = n_real_spam * 10
            logger.info(f"100% synthetic ratio: generating {n_synthetic_spam} samples (10x real spam)")
        else:
            n_synthetic_spam = int(n_real_spam * synthetic_ratio / (100 - synthetic_ratio))

        if n_synthetic_spam == 0:
            metadata = {
                'n_real': X_real.shape[0],
                'n_real_spam': n_real_spam,
                'n_real_ham': (y_real == 0).sum(),
                'n_synthetic': 0,
                'synthetic_ratio': 0,
                'method': self.method,
                'strategy': strategy
            }
            return X_real, y_real, metadata

        # Generate synthetic spam samples using SMOTE on entire dataset
        # Create temporary imbalanced dataset for SMOTE
        # Target: add n_synthetic_spam spam samples
        target_spam_count = n_real_spam + n_synthetic_spam
        n_ham = (y_real == 0).sum()

        if n_ham == 0:
            logger.warning("No ham samples found, cannot apply SMOTE")
            return X_real, y_real, metadata

        sampling_ratio = target_spam_count / n_ham

        logger.info(f"Applying SMOTE: target {target_spam_count} spam vs {n_ham} ham (ratio={sampling_ratio:.3f})")

        # Apply SMOTE
        # Use dictionary format when ratio > 1.0 (minority becomes majority)
        if sampling_ratio > 1.0:
            # Specify exact target counts for each class
            self.sampler.sampling_strategy = {1: target_spam_count}
            logger.info(f"Using dictionary sampling_strategy for high ratio: {{1: {target_spam_count}}}")
        else:
            self.sampler.sampling_strategy = sampling_ratio

        X_resampled, y_resampled = self.sampler.fit_resample(X_real, y_real)

        # Extract only the synthetic samples (new samples added by SMOTE)
        n_original = X_real.shape[0]
        X_synthetic = X_resampled[n_original:]
        y_synthetic = y_resampled[n_original:]

        logger.info(f"Generated {len(y_synthetic)} synthetic samples")

        # Use the resampled dataset (real + synthetic)
        X_train = X_resampled
        y_train = y_resampled

        # Shuffle
        indices = np.random.permutation(X_train.shape[0])
        X_train = X_train[indices]
        y_train = y_train[indices]

        n_synthetic_actual = len(y_synthetic)
        n_real_ham = (y_real == 0).sum()
        n_real_spam = (y_real == 1).sum()

        metadata = {
            'n_real': X_real.shape[0],
            'n_real_spam': n_real_spam,
            'n_real_ham': n_real_ham,
            'n_synthetic': n_synthetic_actual,
            'n_synthetic_spam': (y_synthetic == 1).sum(),
            'synthetic_ratio': synthetic_ratio,
            'actual_ratio': n_synthetic_actual / X_train.shape[0] * 100,
            'method': self.method,
            'strategy': strategy
        }

        logger.info(f"Training dataset: {X_train.shape[0]} samples "
                   f"({n_real_ham} real ham + {n_real_spam} real spam + {n_synthetic_actual} synthetic)")

        return X_train, y_train, metadata

    def get_vectorizer(self) -> TfidfVectorizer:
        """Get the fitted vectorizer for transforming test data"""
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call generate_synthetic_samples first.")
        return self.vectorizer

    def get_metadata(self) -> Dict[str, Any]:
        """Get generator metadata"""
        return {
            'method': self.method,
            'vectorizer_params': self.vectorizer_params,
            'smote_params': self.smote_params,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted
        }
