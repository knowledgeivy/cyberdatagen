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

        # Default vectorizer parameters (MUST match classifiers.py exactly!)
        # classifiers.py line 65-72: max_features, ngram_range, stop_words, lowercase, min_df, max_df
        # CRITICAL: NO sublinear_tf parameter (defaults to False)
        default_vectorizer_params = {
            'max_features': 10000,
            'ngram_range': (1, 2),
            'min_df': 2,
            'max_df': 0.95
        }
        self.vectorizer_params = {**default_vectorizer_params, **(vectorizer_params or {})}

        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(**self.vectorizer_params)
        self.is_fitted = False

        # Default SMOTE parameters
        default_smote_params = {
            'sampling_strategy': 'auto',
            'k_neighbors': 5,
            'random_state': random_state
        }
        self.smote_params = {**default_smote_params, **(smote_params or {})}

        # Initialize sampler
        self.sampler = self._init_sampler()

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

        # Limit to target count (use random_state for reproducibility)
        if X_synthetic_spam.shape[0] > target_count:
            rng = np.random.RandomState(self.random_state)
            indices = rng.choice(X_synthetic_spam.shape[0], target_count, replace=False)
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

        IMPORTANT: This method ensures spam:ham ratio remains CONSTANT across all synthetic_ratios,
        matching the behavior of GPT/Claude experiments.

        Logic (matching GPT/Claude):
        - Total spam count = FIXED (e.g., 100)
        - synthetic_ratio controls REPLACEMENT of real spam with synthetic spam
        - Training set size remains CONSTANT (except for special cases)

        Example (100 real spam + 900 ham):
        - 0%:   100 real spam + 0 synthetic + 900 ham = 1000 samples
        - 10%:  90 real spam + 10 synthetic + 900 ham = 1000 samples
        - 50%:  50 real spam + 50 synthetic + 900 ham = 1000 samples
        - 100%: 0 real spam + 100 synthetic + 900 ham = 1000 samples

        Args:
            real_texts: Real spam email texts (all classes)
            real_labels: Real labels (0=ham, 1=spam)
            synthetic_ratio: Percentage of synthetic spam vs total spam (0-100)
            strategy: 'within_group' or 'cross_group' (for consistency)

        Returns:
            Tuple of (X_train_features, y_train, metadata)
        """
        logger.info(f"Building training dataset with {synthetic_ratio}% synthetic data")

        # Extract real features for ALL data (ham + spam)
        X_real = self.vectorizer.fit_transform(real_texts)
        self.is_fitted = True
        y_real = real_labels.values

        # Separate spam and ham
        spam_mask = y_real == 1
        ham_mask = y_real == 0

        X_real_spam = X_real[spam_mask]
        y_real_spam = y_real[spam_mask]
        X_real_ham = X_real[ham_mask]
        y_real_ham = y_real[ham_mask]

        n_total_real_spam = X_real_spam.shape[0]
        n_total_ham = X_real_ham.shape[0]

        logger.info(f"Real data: {n_total_real_spam} spam, {n_total_ham} ham")

        # CRITICAL: Keep total spam count FIXED (same as GPT/Claude logic)
        total_spam_needed = n_total_real_spam

        # Calculate how many synthetic vs real spam we need
        synthetic_count = int(total_spam_needed * synthetic_ratio / 100)
        real_count = total_spam_needed - synthetic_count

        logger.info(f"Target composition: {real_count} real spam + {synthetic_count} synthetic spam = {total_spam_needed} total spam")

        # Handle 0% synthetic case (all real)
        if synthetic_ratio == 0 or synthetic_count == 0:
            X_train = X_real
            y_train = y_real

            # NO shuffle - must match GPT/Claude which doesn't shuffle

            metadata = {
                'n_real_spam': n_total_real_spam,
                'n_real_ham': n_total_ham,
                'n_synthetic_spam': 0,
                'n_total_spam': n_total_real_spam,
                'n_total_samples': X_train.shape[0],
                'synthetic_ratio': 0,
                'spam_ratio': n_total_real_spam / X_train.shape[0],
                'method': self.method,
                'strategy': strategy
            }
            logger.info(f"Training dataset (0% synthetic): {X_train.shape[0]} samples, spam ratio: {metadata['spam_ratio']:.3f}")
            return X_train, y_train, metadata

        # Handle 100% synthetic case (all synthetic, no real spam)
        if synthetic_ratio >= 100:
            real_count = 0
            synthetic_count = total_spam_needed
            logger.info(f"100% synthetic: generating {synthetic_count} synthetic spam samples")

        # Step 1: Sample real spam
        if real_count > 0:
            if real_count <= n_total_real_spam:
                rng = np.random.RandomState(self.random_state)
                real_spam_indices = rng.choice(n_total_real_spam, real_count, replace=False)
                X_selected_real_spam = X_real_spam[real_spam_indices]
                y_selected_real_spam = y_real_spam[real_spam_indices]
            else:
                logger.warning(f"Requested {real_count} real spam but only {n_total_real_spam} available")
                X_selected_real_spam = X_real_spam
                y_selected_real_spam = y_real_spam
                real_count = n_total_real_spam
        else:
            X_selected_real_spam = None
            y_selected_real_spam = None

        # Step 2: Generate synthetic spam using SMOTE
        if synthetic_count > 0:
            # We need to generate enough synthetic samples
            # SMOTE generates by oversampling, so we'll generate more than needed and sample

            # To generate synthetic_count samples, we use SMOTE to oversample spam
            # Calculate target: we want to add synthetic_count spam samples to the dataset
            target_spam_total = n_total_real_spam + synthetic_count

            if n_total_ham == 0:
                logger.error("No ham samples available, cannot apply SMOTE")
                raise ValueError("SMOTE requires ham samples")

            sampling_ratio = target_spam_total / n_total_ham

            logger.info(f"Applying SMOTE to generate synthetic spam pool: target {target_spam_total} spam vs {n_total_ham} ham (ratio={sampling_ratio:.3f})")

            # Apply SMOTE
            if sampling_ratio > 1.0:
                self.sampler.sampling_strategy = {1: target_spam_total}
                logger.info(f"Using dictionary sampling_strategy: {{1: {target_spam_total}}}")
            else:
                self.sampler.sampling_strategy = sampling_ratio

            X_resampled, y_resampled = self.sampler.fit_resample(X_real, y_real)

            # Extract only the NEW synthetic samples (those added by SMOTE)
            n_original = X_real.shape[0]
            X_synthetic_all = X_resampled[n_original:]
            y_synthetic_all = y_resampled[n_original:]

            # Filter to get only spam synthetic samples
            synthetic_spam_mask = y_synthetic_all == 1
            X_synthetic_spam_pool = X_synthetic_all[synthetic_spam_mask]
            y_synthetic_spam_pool = y_synthetic_all[synthetic_spam_mask]

            n_synthetic_pool = X_synthetic_spam_pool.shape[0]
            logger.info(f"Generated synthetic spam pool: {n_synthetic_pool} samples")

            # Sample exactly synthetic_count from the pool
            if synthetic_count <= n_synthetic_pool:
                rng = np.random.RandomState(self.random_state)
                synthetic_indices = rng.choice(n_synthetic_pool, synthetic_count, replace=False)
                X_synthetic_spam = X_synthetic_spam_pool[synthetic_indices]
                y_synthetic_spam = y_synthetic_spam_pool[synthetic_indices]
            else:
                logger.warning(f"Requested {synthetic_count} synthetic but only generated {n_synthetic_pool}, using all")
                X_synthetic_spam = X_synthetic_spam_pool
                y_synthetic_spam = y_synthetic_spam_pool
                synthetic_count = n_synthetic_pool
        else:
            X_synthetic_spam = None
            y_synthetic_spam = None

        # Step 3: Combine all parts (real spam + synthetic spam + all ham)
        X_parts = []
        y_parts = []

        if X_selected_real_spam is not None:
            X_parts.append(X_selected_real_spam)
            y_parts.append(y_selected_real_spam)

        if X_synthetic_spam is not None:
            X_parts.append(X_synthetic_spam)
            y_parts.append(y_synthetic_spam)

        # Always include ALL ham
        X_parts.append(X_real_ham)
        y_parts.append(y_real_ham)

        # Stack vertically
        from scipy.sparse import vstack
        X_train = vstack(X_parts)
        y_train = np.concatenate(y_parts)

        # Shuffle
        rng = np.random.RandomState(self.random_state)
        indices = rng.permutation(X_train.shape[0])
        X_train = X_train[indices]
        y_train = y_train[indices]

        # Metadata
        actual_spam_count = (y_train == 1).sum()
        actual_ham_count = (y_train == 0).sum()

        metadata = {
            'n_real_spam': real_count,
            'n_real_ham': n_total_ham,
            'n_synthetic_spam': synthetic_count,
            'n_total_spam': actual_spam_count,
            'n_total_samples': X_train.shape[0],
            'synthetic_ratio': synthetic_ratio,
            'actual_synthetic_ratio': (synthetic_count / actual_spam_count * 100) if actual_spam_count > 0 else 0,
            'spam_ratio': actual_spam_count / X_train.shape[0],
            'method': self.method,
            'strategy': strategy
        }

        logger.info(f"Training dataset: {X_train.shape[0]} samples "
                   f"({real_count} real spam + {synthetic_count} synthetic spam + {actual_ham_count} ham)")
        logger.info(f"Spam ratio: {metadata['spam_ratio']:.3f} (should be constant across ratios)")

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
