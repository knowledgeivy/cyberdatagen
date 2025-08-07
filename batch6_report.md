# Batch 6 Experiment Report

**Author:** Claude
**Date:** 2025-08-01

## 1. Research Goal

The primary goal of the `batch6` experiment is to investigate the effectiveness of high-quality synthetic data in training machine learning models for malicious content detection. The experiment aims to determine if models trained on sophisticated synthetic data can achieve performance comparable to or better than models trained on real data. This involves not just generating the data, but also understanding *how* the generation process influences the data's characteristics and, ultimately, the performance of the models trained on it.

## 2. The Experiment: A 7-Phase Approach

The `batch6` experiment is a comprehensive, multi-phase pipeline designed to generate, analyze, and evaluate high-quality synthetic data. The experiment is divided into seven distinct phases:

### Phase 1: Enhanced Seed Preparation with Core/Edge Layer Stratification

-   **Objective:** To select a diverse and high-quality set of seed samples to be used as the basis for synthetic data generation.
-   **Procedure:**
    -   A 2-layer stratification method was used to select 2,000 seed samples from a larger dataset (`batch4_fresh`).
    -   The stratification focused on "core" (typical, high-density) and "edge" (outlier, low-density) samples to ensure the seeds represent the full spectrum of malicious data.
-   **Rationale and Interpretation:**
    -   **Why Core/Edge?** Real-world data is not uniform. Some data points are very typical of their class ("core"), while others are unusual or novel ("edge"). By explicitly sampling from these two layers, we can test the LLM's ability to generate both common and rare examples of malicious content.
    -   **Interpretive Power:** This strategy allows us to answer questions like: "Does the LLM generate high-quality edge cases?" or "Does the model's performance depend on the diversity of the synthetic data?" This is a significant improvement over random sampling, as it provides a framework for understanding the LLM's generative capabilities.

### Phase 2: High-Quality Synthetic Data Generation

-   **Objective:** To generate a large volume of high-quality synthetic data from the seed samples.
-   **Procedure:**
    -   6,000 synthetic samples were generated from the 2,000 seed samples.
    -   A sophisticated "real_data_rewriter.py" methodology was employed, using three different prompt strategies ("original", "strong", and "weak") to create diverse synthetic data.
-   **Rationale:** This phase was designed to be an improvement over previous batches by using structured YAML prompts and a more robust toolchain, leading to higher-quality synthetic data.

### Phase 3: Synthetic Data ID Annotation

-   **Objective:** To ensure complete traceability of the synthetic data.
-   **Procedure:**
    -   Unique, traceable IDs were assigned to all 6,000 synthetic samples.
-   **Rationale:** This allows every synthetic sample to be traced back to its original seed, the layer it came from, and the prompt strategy used to generate it, which is crucial for detailed analysis.

### Phase 4: Unified Embedding Space Construction

-   **Objective:** To represent all data in a unified, high-dimensional space for analysis and comparison.
-   **Procedure:**
    -   A unified embedding space was created for all data (seeds, synthetic data, background data, etc.) using the `all-MiniLM-L6-v2` model.
-   **Rationale and Interpretation:**
    -   **Why Unified Embeddings?** The embedding space is a mathematical representation of the data where similar items are located close to each other. By placing all our data (real, synthetic, core, edge) in the *same* space, we can directly compare them.
    -   **Interpretive Power:** This allows us to answer critical questions about the LLM's generation process:
        -   **Fidelity:** Does the synthetic data cluster closely with the real seed data it was generated from? (Phase 5 Visualization)
        -   **Diversity:** Do the different prompt strategies ("original", "strong", "weak") produce synthetic data in different regions of the space?
        -   **Coverage:** Does the synthetic data fill the gaps in the real data distribution, or does it just reproduce the existing data?
    -   The unified embedding space is the bridge between the LLM's text output and the final classification performance. It allows us to *see* the quality of the synthetic data before we even train a model.

### Phase 5 & 5a/5b: Enhanced Visualization and Analysis

-   **Objective:** To visually analyze the quality and distribution of the synthetic data in the embedding space.
-   **Procedure:**
    -   A comprehensive visualization of the embedding space was performed.
    -   The distribution of synthetic data was compared to the original seeds, the effects of different prompt strategies were analyzed, and the quality of the data clusters was checked.
-   **Rationale:** This phase provides a qualitative assessment of the synthetic data, allowing us to "see" how well it mimics the real data and to diagnose any issues with the generation process.

### Phase 6: Pure Dataset Construction

-   **Objective:** To create a set of controlled datasets for training and evaluating machine learning models.
-   **Procedure:**
    -   16 "pure" datasets were constructed with varying proportions of malicious data (5%, 10%, 15%, 20%) and different types of malicious data (real only, or purely synthetic from each of the three prompt strategies).
-   **Rationale:** These carefully designed datasets allow us to test the effectiveness of the synthetic data in a controlled and systematic manner, isolating the impact of the synthetic data on model performance.

### Phase 7a & 7b: Model Training and Performance Evaluation

-   **Objective:** To train and evaluate machine learning models on the pure datasets.
-   **Procedure:**
    -   Three different machine learning models (RandomForest, SVM, and a Deep Learning model) were trained on each of the 16 pure datasets.
    -   The performance of the models was evaluated on a fixed, independent test set.
-   **Rationale:** This is the ultimate test of our research. The results from this phase, when combined with the insights from the embedding space analysis (Phase 4 & 5), allow us to draw a direct line from the LLM's generation process to the final classification performance.

## 3. Research Problem Solved

The `batch6` experiment successfully addresses the following research problem:

-   **How to generate high-quality, diverse synthetic data that can effectively augment or even replace real data for training robust machine learning classifiers in the cybersecurity domain (specifically, for malicious content detection).**

The experiment implements a sophisticated, end-to-end pipeline that not only generates the data but also rigorously evaluates its quality and impact on model performance. The use of stratified sampling, multiple prompt strategies, and detailed analysis in the embedding space are all strong indicators of a well-designed and successful experiment.

## 4. Implementation for Other Synthetic Data Generation

The methodology and tools developed in the `batch6` experiment can be applied to other synthetic data generation tasks. The key steps to replicate this process are:

1.  **Curate a high-quality seed dataset:** Start with a representative set of real-world data. The use of a stratification strategy like core/edge is highly recommended to ensure diversity.
2.  **Implement a sophisticated data generation strategy:** Use a powerful language model (like GPT-4.1-mini) with well-designed prompts to generate synthetic data. The use of multiple prompt strategies is recommended to ensure diversity.
3.  **Ensure traceability:** Assign unique IDs to all synthetic data to allow for detailed analysis and debugging.
4.  **Analyze the synthetic data in a unified embedding space:** This is a crucial step to qualitatively assess the quality of the synthetic data and to understand the LLM's generation process.
5.  **Rigorously evaluate the synthetic data:** Train and evaluate machine learning models on datasets containing the synthetic data and compare their performance to models trained on real data.

The `batch6` experiment provides a robust and well-documented framework for generating and evaluating high-quality synthetic data. The scripts in `src/cyberdata/scripts/` can be adapted and reused for other synthetic data generation tasks.
