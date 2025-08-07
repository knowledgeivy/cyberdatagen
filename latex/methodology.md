# Methodology

Our methodology is designed to systematically evaluate the efficacy of Large Language Model (LLM)-generated synthetic data for training robust cybersecurity classifiers. The approach is structured as a multi-phase experimental pipeline, encompassing data preparation, stratified sampling, synthetic data generation, and rigorous evaluation. This process allows us to analyze the relationship between the characteristics of the synthetic data and the performance of machine learning models trained upon it.

## 1. Data Preparation and Stratification

The foundation of our experiment is a high-quality, stratified dataset of real-world malicious samples. We employ a two-layer stratification strategy to ensure the diversity of our seed data, which is crucial for generating a comprehensive synthetic dataset.

### 1.1. Core/Edge Layer Stratification

We hypothesize that the effectiveness of synthetic data is dependent on its ability to represent both common and novel examples of malicious content. To this end, we stratify our seed data into two layers:

-   **Core Layer:** This layer consists of samples that are highly representative of the malicious class. These are identified as samples with a high density in the embedding space, close to the class centroid. In our experiment, the core layer seeds had a mean distance of **0.49** to the centroid.
-   **Edge Layer:** This layer consists of samples that are outliers or novel examples of the malicious class. These are identified as samples with a low density in the embedding space, far from the class centroid. In our experiment, the edge layer seeds had a mean distance of **0.74** to the centroid.

This stratification allows us to explicitly test the LLM's ability to generate both common and rare examples of malicious content and to analyze the impact of this diversity on model performance.

## 2. Synthetic Data Generation

We employ a sophisticated data generation strategy using a powerful LLM (GPT-4.1-mini) to create a large and diverse synthetic dataset. The generation process is guided by a set of carefully designed prompts and is based on the stratified seed data from the previous phase.

### 2.1. Prompt-Based Generation

We use three distinct prompt strategies to generate synthetic data:

-   **Original Prompt:** A baseline prompt that asks the LLM to rewrite the seed sample while preserving its malicious intent.
-   **Strong Prompt:** A prompt that encourages the LLM to generate a more overtly malicious version of the seed sample.
-   **Weak Prompt:** A prompt that encourages the LLM to generate a more subtle or disguised version of the seed sample.

This multi-prompt strategy is designed to generate a diverse range of synthetic data, from obvious to subtle attacks, which is crucial for training robust classifiers.

## 3. Unified Embedding Space Analysis

To analyze the quality and characteristics of the synthetic data, we use a unified embedding space. This is a high-dimensional space where all data points (real and synthetic) are represented as vectors. The distance between vectors in this space corresponds to the semantic similarity of the data points.

### 3.1. Embedding Model

We use the `all-MiniLM-L6-v2` model to generate 384-dimensional embeddings for all our data. This model is chosen for its balance of performance and computational efficiency.

### 3.2. Analysis in the Embedding Space

The unified embedding space allows us to perform a range of analyses, including:

-   **Dimensionality Reduction:** We use PCA and t-SNE to visualize the high-dimensional embedding space in 2D and 3D, allowing for a qualitative assessment of the data distribution.
-   **Clustering Analysis:** We use K-means clustering to identify natural groupings in the data and to assess the coherence of the synthetic data.
-   **Distance Analysis:** We measure the distance between synthetic data points and their corresponding seed samples to quantify the fidelity of the generation process.

## 4. Pure Dataset Construction and Model Training

To evaluate the effectiveness of the synthetic data, we construct a series of "pure" datasets for training and testing machine learning models. These datasets are designed to isolate the impact of the synthetic data on model performance.

### 4.1. Pure Datasets and Varying Malicious Ratios

We construct 16 pure datasets based on four data sources (real malicious, synthetic original, synthetic strong, and synthetic weak). For each source, we create datasets with varying proportions of malicious data: **5%, 10%, 15%, and 20%**.

This variation is critical for simulating real-world cybersecurity scenarios where the prevalence of malicious traffic or content (i.e., the class balance) can fluctuate significantly. By testing across these ratios, we can evaluate the robustness of our synthetic data. A successful synthetic dataset should enable a model to perform well not only in balanced scenarios but also in highly imbalanced, low-resource environments, which are common in security applications.

### 4.2. Model Training and Evaluation

We train three different machine learning models (RandomForest, SVM, and a Deep Learning model) on each of the 16 pure datasets. The performance of these models is then evaluated on a fixed, independent test set. This allows us to directly compare the performance of models trained on synthetic data to those trained on real data across different levels of data availability and class imbalance.
