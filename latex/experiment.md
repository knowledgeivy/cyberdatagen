# Experimental Setup

Our experimental setup is designed to be a reproducible and rigorous evaluation of our synthetic data generation methodology. The experiment is conducted as a 7-phase pipeline, with each phase building upon the results of the previous one. All scripts and data are containerized to ensure reproducibility.

## 1. Datasets

We use a combination of real and synthetic data in our experiments:

-   **Real Data:** We use the `batch4_fresh` dataset, which contains a large number of real-world malicious and benign samples. This dataset is used for seed selection and as a baseline for model training.
-   **Synthetic Data:** We generate a total of **6,000** synthetic malicious samples using our methodology. These samples are generated from a seed set of **2,000** real malicious samples, with **3** synthetic variants (original, strong, and weak) generated for each seed.

## 2. Experimental Pipeline

The experiment is executed as a 7-phase pipeline, with each phase corresponding to a script in the `src/cyberdata/scripts/` directory.

-   **Phase 1: Seed Preparation:** We select 2,000 seed samples from the `batch4_fresh` dataset using our core/edge stratification methodology. The output of this phase is a high-quality, diverse set of seed samples.
-   **Phase 2: Synthetic Data Generation:** We generate 6,000 synthetic samples from the 2,000 seed samples using our multi-prompt generation strategy. The output of this phase is a large and diverse synthetic dataset.
-   **Phase 3: ID Annotation:** We assign unique, traceable IDs to all synthetic samples. This allows us to track the lineage of each synthetic sample back to its original seed and prompt strategy.
-   **Phase 4: Embedding:** We generate 384-dimensional embeddings for all our data (real and synthetic) using the `all-MiniLM-L6-v2` model. The output of this phase is a unified embedding space containing all our data.
-   **Phase 5: Visualization and Analysis:** We perform a comprehensive analysis of the unified embedding space, including dimensionality reduction, clustering, and distance analysis. The output of this phase is a set of visualizations and a detailed analysis of the synthetic data's quality.
-   **Phase 6: Dataset Construction:** We construct 16 pure datasets for model training. These datasets contain varying proportions of malicious data and different types of malicious data (real vs. synthetic).
-   **Phase 7: Model Training and Evaluation:** We train three machine learning models (RandomForest, SVM, and DeepLearning) on each of the 16 pure datasets and evaluate their performance on a fixed, independent test set. The output of this phase is a detailed performance report for each model and dataset combination.

## 3. Evaluation Metrics

We use a range of metrics to evaluate the quality of our synthetic data and the performance of our machine learning models:

-   **Embedding Space Metrics:**
    -   **PCA and t-SNE Variance Explained:** To measure the amount of information captured by our dimensionality reduction techniques.
    -   **Silhouette Score:** To measure the quality of our clustering results. The best silhouette score achieved was **0.16** with **19** clusters.
    -   **Distance to Centroid:** To measure the distance of data points from the class centroid, which is used for our core/edge stratification.
-   **Classification Metrics:**
    -   **Accuracy:** The proportion of correctly classified samples.
    -   **Precision:** The proportion of true positives among all positive predictions.
    -   **Recall:** The proportion of true positives that were correctly identified.
    -   **F1-Score:** The harmonic mean of precision and recall.
