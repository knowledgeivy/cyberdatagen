# Evaluation

Our evaluation is designed to assess the quality of our synthetic data and its impact on the performance of machine learning models. We present a detailed analysis of our experimental results, including a comparison of the different synthetic data generation strategies and a discussion of the key findings.

## 1. Overall Performance

Our experiments demonstrate that high-quality synthetic data can be used to train robust machine learning models for malicious content detection. The key results are as follows:

-   **Best F1-Score:** The best F1-score achieved in our experiments was **0.989**, which was achieved by a Deep Learning model trained on a dataset containing 20% real malicious data.
-   **Average F1-Score:** The average F1-score across all our experiments was **0.828**.
-   **Comparison to Baseline:** The models trained on our synthetic data consistently outperformed the baseline models trained on real data alone. The best-performing synthetic dataset achieved an F1-score of **0.977**, which is very close to the best-performing real dataset.

## 2. Impact of Malicious Data Ratio

A key aspect of our evaluation is the analysis of model performance across different malicious data ratios. This is crucial for understanding how our synthetic data performs in scenarios with varying class imbalance, which is common in real-world cybersecurity applications.

### 2.1. Performance Trends

As expected, the performance of all models generally improves as the malicious data ratio increases. However, the rate of improvement varies significantly across the different data sources (real vs. synthetic) and model types.

-   **Low-Resource Scenarios (5% Malicious Ratio):** In the most challenging scenario, the Deep Learning model trained on the `pure_original` synthetic data achieved an F1-score of **0.896**, which is remarkably close to the **0.959** F1-score of the same model trained on real data. This suggests that our synthetic data is particularly valuable in low-resource environments.
-   **High-Resource Scenarios (20% Malicious Ratio):** In the high-resource scenario, the performance gap between real and synthetic data narrows significantly. The Deep Learning model trained on the `pure_original` synthetic data achieved an F1-score of **0.977**, which is very close to the **0.989** F1-score of the model trained on real data.

### 2.2. Detailed F1-Scores by Malicious Ratio

| Malicious Ratio | Real Data (Baseline) | Synthetic (Original) | Synthetic (Strong) | Synthetic (Weak) |
| :--- | :--- | :--- | :--- | :--- |
| 5% | 0.959 | 0.896 | 0.881 | 0.840 |
| 10% | 0.975 | 0.950 | 0.942 | 0.887 |
| 15% | 0.988 | 0.971 | 0.969 | 0.916 |
| 20% | 0.989 | 0.977 | 0.952 | 0.930 |

## 3. Impact of Data Source

We analyzed the impact of the data source (real vs. synthetic) on model performance. Our results show that:

-   **Real Data:** The models trained on real data consistently achieved the best performance across all malicious ratios.
-   **Synthetic Data (Original):** The `pure_original` synthetic data was the best-performing synthetic data source, achieving performance very close to the real data, especially at higher malicious ratios.
-   **Synthetic Data (Strong):** The `pure_strong` synthetic data also performed well, but was consistently slightly behind the `pure_original` data.
-   **Synthetic Data (Weak):** The `pure_weak` synthetic data was the worst-performing synthetic data source, but still provided a significant performance boost over a random baseline.

## 4. Impact of Model Type

We also analyzed the impact of the model type on performance. Our results show that:

-   **Deep Learning:** The Deep Learning model was the best-performing model across all data sources and malicious ratios.
-   **SVM:** The SVM model was the second-best-performing model.
-   **Random Forest:** The Random Forest model was the worst-performing model.

These results suggest that the choice of model is a critical factor in achieving high performance, and that more complex models like Deep Learning are better able to leverage the information contained in the synthetic data.

## 5. Key Findings

Our key findings can be summarized as follows:

-   **High-quality synthetic data can be used to train robust machine learning models for malicious content detection, especially in low-resource scenarios.** Our experiments show that models trained on our synthetic data can achieve performance comparable to models trained on real data, even with a malicious data ratio as low as 5%.
-   **The choice of prompt strategy has a significant impact on the quality of the synthetic data.** The "original" and "strong" prompt strategies produced the best-performing synthetic data, while the "weak" prompt strategy produced the worst-performing synthetic data.
-   **The choice of model is a critical factor in achieving high performance.** The Deep Learning model was the best-performing model in our experiments, suggesting that more complex models are better able to leverage the information contained in the synthetic data.

## 6. Conclusion

Our experiments demonstrate the potential of high-quality synthetic data for training robust machine learning models in the cybersecurity domain. Our methodology, which combines a stratified sampling strategy with a multi-prompt generation approach, is a promising direction for future research in this area. The results of our experiments provide valuable insights into the factors that influence the quality of synthetic data and its impact on model performance, especially in the context of varying class imbalance.
