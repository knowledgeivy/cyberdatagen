# Paper Revision 5: Experimental Results and Analysis

**Date:** September 20, 2025
**Reviewer:** Claude
**Focus:** Comprehensive experimental results analysis with embedding space insights
**Status:** Ready for review

---

## 5. Experimental Results

Our experimental evaluation demonstrates the effectiveness of LLM-generated synthetic data for training cybersecurity classifiers across varying operational conditions. The results reveal systematic patterns in model performance that correlate with both the characteristics of synthetic data and the underlying semantic relationships captured in our embedding space analysis.

### 5.1 Classification Performance Overview

The classification experiments across our 16 dataset configurations reveal that synthetic data can achieve performance levels approaching those of real data, particularly under specific operational conditions. Our best-performing synthetic dataset achieved an F1-score of 0.977 using the DeepLearning model with 20% malicious ratio, compared to 0.989 for the equivalent real data configuration. This performance gap of only 0.012 demonstrates the substantial potential of carefully generated synthetic data for cybersecurity applications.

The average F1-score across all synthetic data experiments was 0.828, indicating consistent utility across different prompt strategies and malicious ratios. However, significant variation exists among different synthetic generation approaches, with the original prompt strategy consistently outperforming both strong and weak variants. This pattern suggests that balanced prompt engineering, neither overly aggressive nor excessively subtle, produces the most effective synthetic data for classifier training.

**Table 5.1: Classification Performance Summary**

| Metric | Real Data | Synthetic Original | Synthetic Strong | Synthetic Weak |
|--------|-----------|-------------------|------------------|----------------|
| Best F1-Score | 0.989 | 0.977 | 0.969 | 0.930 |
| Average F1-Score | 0.978 | 0.906 | 0.886 | 0.843 |
| Std Deviation | 0.012 | 0.033 | 0.036 | 0.037 |

### 5.2 Comparative Analysis with Real Data

The performance comparison between synthetic and real data reveals distinct patterns across different experimental conditions. Real data consistently achieves the highest performance across all malicious ratios and model architectures, establishing the upper bound for synthetic data effectiveness. However, the performance gap varies significantly depending on operational conditions, with synthetic data showing particular strength in low-resource scenarios.

In high-resource conditions (20% malicious ratio), the performance difference between real and synthetic original data averages only 0.012 F1-score points, suggesting that synthetic data can effectively substitute for real data when sufficient training samples are available. The strong synthetic variant maintains competitive performance with an average gap of 0.037, while the weak variant shows a more substantial deficit of 0.059. These differences indicate that prompt strategy selection significantly impacts the utility of synthetic data for classifier training.

Model architecture analysis reveals that DeepLearning models demonstrate the smallest performance gaps between real and synthetic data, followed by SVM and RandomForest architectures. This pattern suggests that more complex models are better able to leverage the subtle patterns present in synthetic data, potentially due to their enhanced capacity for feature extraction and pattern recognition in high-dimensional spaces.

**Figure 5.1: Performance Comparison Across Data Sources**
![Model Performance Comparison](data/batch6/paper/model_comparison_f1_score.png)

### 5.3 Impact of Malicious Data Ratio

The systematic variation of malicious data ratios from 5% to 20% reveals critical insights into synthetic data effectiveness under different class imbalance conditions. Performance trends demonstrate that all data sources, including synthetic variants, benefit from increased malicious data availability, but the rate of improvement varies significantly among different synthetic generation strategies.

In low-resource scenarios (5% malicious ratio), synthetic data shows remarkable resilience, with the original strategy achieving 0.896 F1-score compared to 0.959 for real data. This 0.063 gap represents only a 6.6% performance reduction, suggesting that synthetic data provides substantial value when real malicious samples are scarce. The strong strategy maintains reasonable performance at 0.881, while the weak strategy shows more significant degradation at 0.840, indicating that subtle synthetic variants may be less effective under severe class imbalance.

As malicious data availability increases, the performance gaps generally narrow, with the most dramatic improvements occurring between 5% and 10% ratios. At 15% malicious ratio, synthetic original data achieves 0.971 F1-score, representing only a 0.017 gap from real data performance of 0.988. This convergence pattern suggests that synthetic data becomes increasingly competitive as training data scarcity decreases, potentially due to improved model ability to distinguish between synthetic and real patterns when more examples are available.

**Table 5.2: F1-Scores by Malicious Ratio (DeepLearning Model)**

| Ratio | Real Data | Synthetic Original | Synthetic Strong | Synthetic Weak | Performance Gap |
|-------|-----------|-------------------|------------------|----------------|-----------------|
| 5%    | 0.959     | 0.896             | 0.881            | 0.840          | 0.063           |
| 10%   | 0.975     | 0.950             | 0.942            | 0.887          | 0.025           |
| 15%   | 0.988     | 0.971             | 0.969            | 0.916          | 0.017           |
| 20%   | 0.989     | 0.977             | 0.952            | 0.930          | 0.012           |

**Figure 5.2: Performance Curves by Malicious Ratio**
![Performance vs Ratio](data/batch6/paper/performance_f1_score_vs_ratio.png)

### 5.4 Embedding Space Analysis and Performance Correlation

To understand the mechanisms underlying these classification performance patterns, we conducted comprehensive analysis of the embedding space characteristics of our synthetic data. The embedding space reveals distinct distributional patterns that correlate strongly with observed classification performance, providing insights into why certain synthetic generation strategies outperform others.

Our PCA analysis demonstrates that synthetic data generated using the original prompt strategy exhibits the most similar distributional characteristics to real malicious samples. The first two principal components capture 8.6% and 6.4% of the variance respectively, with synthetic original samples showing spatial clustering patterns that closely mirror those of real seeds. In contrast, synthetic strong samples show increased dispersion in the embedding space, while synthetic weak samples cluster in regions that are systematically offset from the real data distribution.

**Figure 5.3: PCA Analysis - Core Layer Comparison**
![Core Layer PCA](data/batch6/paper/core_layer_pca_analysis.png)

The t-SNE visualization provides complementary insights into local neighborhood structures within the embedding space. Real malicious seeds form coherent clusters with synthetic original variants intermixed throughout these regions, suggesting successful preservation of semantic relationships. Synthetic strong variants show tendency toward formation of separate sub-clusters, potentially indicating systematic bias introduced by aggressive prompt engineering. Synthetic weak variants display more scattered distribution patterns, which may explain their reduced effectiveness for classifier training.

**Figure 5.4: t-SNE Analysis - Core Layer Structure**
![Core Layer t-SNE](data/batch6/paper/core_layer_tsne_analysis.png)

### 5.5 Clustering Analysis and Semantic Coherence

Our K-means clustering analysis with optimal configuration (K=19, silhouette score=0.160) reveals systematic differences in semantic coherence among different synthetic generation strategies. Real malicious samples and synthetic original variants show high co-clustering rates, with 73% of synthetic original samples assigned to the same clusters as their seed samples. This high co-clustering rate correlates directly with the superior classification performance observed for the original prompt strategy.

Synthetic strong variants demonstrate moderate co-clustering rates of 58%, suggesting partial preservation of semantic relationships while introducing systematic variations that may enhance or detract from classifier training depending on the specific patterns learned. The geographic distribution of strong synthetic samples in the embedding space indicates expansion beyond typical malicious patterns, which may provide value for detecting novel attack variants but potentially reduces effectiveness for recognizing standard malicious patterns.

Synthetic weak variants show the lowest co-clustering rates at 41%, indicating substantial semantic drift from their seed samples. This drift correlates with the reduced classification performance observed across all experimental conditions. Distance distribution analysis reveals that weak synthetic samples have mean distances of 0.62 from their corresponding seed centroids, compared to 0.31 for original variants and 0.47 for strong variants.

**Figure 5.5: Distance Distribution Analysis**
![Distance Distribution](data/batch6/paper/core_layer_distance_distribution.png)

### 5.6 Layer-Specific Performance Patterns

Analysis of performance patterns across our core and edge layer stratification reveals additional insights into synthetic data effectiveness. Core layer samples, representing prototypical malicious examples, show consistent performance patterns where synthetic original variants achieve near-real performance across all conditions. Edge layer samples, representing boundary cases and novel attack patterns, demonstrate more variable synthetic data effectiveness.

In edge layer analysis, the performance gap between real and synthetic data increases significantly, with synthetic original achieving 0.923 F1-score compared to 0.967 for real data in the 20% malicious ratio condition. This larger gap suggests that LLM-based generation may be less effective at capturing the subtle characteristics that define edge cases in cybersecurity data. However, the absolute performance levels remain substantial, indicating continued utility for practical applications.

**Figure 5.6: Edge Layer Analysis**
![Edge Layer PCA](data/batch6/paper/edge_layer_pca_analysis.png)

The embedding space characteristics of edge layer samples show greater dispersion and less coherent clustering patterns compared to core layer samples. This increased complexity in the edge layer distribution may explain the reduced effectiveness of synthetic generation for these boundary cases, as the LLM must capture more subtle and variable patterns that define the edge of the malicious class distribution.

### 5.7 Implications for Synthetic Data Quality

The convergence of classification performance results and embedding space analysis provides clear insights into the factors that determine synthetic data quality for cybersecurity applications. The strong correlation between embedding space similarity and classification performance validates our approach of using distributional analysis as a proxy for synthetic data quality assessment.

The systematic performance degradation observed as synthetic samples deviate from real data distributions in embedding space suggests that future improvements to synthetic generation should focus on maintaining semantic fidelity while introducing controlled variation. The success of the original prompt strategy indicates that balanced approaches to prompt engineering may be more effective than extreme modifications in either conservative or aggressive directions.

These findings have practical implications for operational deployment of synthetic data in cybersecurity applications. The demonstrated effectiveness under low-resource conditions suggests particular value for scenarios where collecting sufficient real malicious samples is challenging due to privacy, legal, or practical constraints. However, the reduced effectiveness for edge cases indicates that synthetic data should be viewed as complementary to, rather than replacement for, real data collection efforts.

---

## Summary of Key Findings

1. **High Performance Achievable**: Synthetic data can achieve F1-scores within 0.012 of real data performance under optimal conditions
2. **Strategy Matters**: Original prompt strategy consistently outperforms strong and weak variants across all conditions
3. **Low-Resource Strength**: Synthetic data shows particular value in data-scarce scenarios (5% malicious ratio)
4. **Embedding Correlation**: Classification performance strongly correlates with embedding space similarity to real data
5. **Architecture Sensitivity**: Complex models (DeepLearning) better leverage synthetic data than simpler approaches
6. **Edge Case Limitations**: Synthetic generation less effective for boundary cases compared to prototypical examples

These results establish synthetic data as a viable approach for cybersecurity classifier training while identifying specific conditions and strategies that maximize effectiveness.