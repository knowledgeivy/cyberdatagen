# Co-author Feedback Analysis and Action Items

**Date**: 2025-10-17
**Status**: Analysis and Planning
**Purpose**: Review co-author suggestions and plan paper revisions

---

## 反馈要点分析 (Feedback Analysis)

### 1. Real-to-Synthetic Detection Experiment 新实验方向

**Co-author建议**:
- Use real spam data to detect synthetic data (reverse detection)
- Mix real spam at different ratios (0%, 50%, 100%) → test on LLM generated synthetic
- Implementation factor consideration

**分析**:
- ✅ **有道理**: 这是一个重要的实验补充,评估"真实数据训练的分类器能否检测出合成垃圾邮件"
- 🔴 **需要明确**: 这个实验方向与当前的"cross-model reverse detection"不同:
  - **Current reverse detection**: Train on Model A synthetic → Test on Model B synthetic (跨模型检测)
  - **Proposed new experiment**: Train on real spam → Test on synthetic spam (真实→合成检测)

**关键问题需要商议**:
1. **实验目的**: 这个实验想回答什么问题?
   - 是否合成数据可以被真实数据训练的分类器检测出来? (detectability)
   - 还是评估合成数据的"真实度"? (authenticity)

2. **实验设计细节**:
   - Training set: Real ham (900) + Real spam (0%, 50%, 100% of 100 spam)
   - Test set: Real ham (900) + Synthetic spam (100% LLM generated)
   - 需要测试哪些LLM? GPT and Claude both?
   - 需要测试哪些prompt? Original, Strong, Weak?
   - 需要测试哪些mixing strategy? Within-group, Cross-group?

3. **计算量评估**:
   - 如果测试 2 models × 3 prompts × 2 strategies × 3 ratios × 20 groups × 2 classifiers = 1,440 experiments
   - 这与当前的cross-model reverse detection规模相同

**建议行动**:
- [ ] 与co-author明确实验目的和研究问题
- [ ] 确定实验参数组合(是否需要所有组合)
- [ ] 评估是否有足够时间和计算资源
- [ ] 如果做,需要更新paper的experiment design和evaluation sections

**优先级**: HIGH (需要尽快确认是否要做这个实验)

---

### 2. Why RF is Better than SVM 模型性能差异解释

**Co-author建议**:
- Need to explain why Random Forest consistently outperforms SVM
- Need to address hyperparameter tuning question

**分析**:
- ✅ **非常有道理**: 这是reviewer会问的关键问题
- 当前paper中RF在几乎所有配置下都优于SVM,但缺乏深入解释

**需要在paper中补充的内容**:

1. **Hyperparameter Tuning Justification** (方法论部分):
   - 说明SVM和RF都使用了什么hyperparameters
   - 是否进行了grid search或其他调优
   - 为什么选择这些参数值

2. **Architectural Differences Analysis** (讨论部分):
   - **SVM特点**:
     - Margin-based linear separator in high-dimensional space
     - Sensitive to feature scale and distribution
     - RBF kernel assumes smooth decision boundaries
     - More prone to overfitting on noisy or overlapping data

   - **Random Forest特点**:
     - Ensemble of decision trees with bootstrap aggregating
     - Naturally handles non-linear patterns and feature interactions
     - Robust to noise through voting mechanism
     - Less sensitive to hyperparameters

3. **Performance Pattern Analysis**:
   - RF maintains stability across synthetic ratios (0.77-0.88)
   - SVM shows dramatic degradation (0.63 → 0.34 at 100% synthetic)
   - **Hypothesis**: Synthetic spam introduces feature-space overlap and distribution shift that violates SVM's margin assumptions but is absorbed by RF's ensemble averaging

**建议行动**:
- [ ] 检查当前paper的methodology section是否明确说明了hyperparameter selection
- [ ] 在Discussion section添加"Classifier Architecture and Robustness"小节
- [ ] 添加statistical analysis解释为什么RF-SVM差异显著
- [ ] 如果SVM没有grid search,需要在limitation中说明

**优先级**: HIGH (必须在paper中解决)

---

### 3. Weak Prompt is More Important 弱提示词的重要性

**Co-author建议**:
- Weak prompt actually is more important

**分析**:
- 🤔 **需要验证**: 这个观点需要数据支持
- 需要检查evaluation results看weak prompt是否有特殊表现

**需要分析的数据**:
1. Forward detection results: Weak vs Original vs Strong
   - 哪个prompt在哪个synthetic ratio下表现最好?
   - Weak prompt是否更realistic因此更难检测?
   - Weak prompt是否在某些metric上特别好?

2. Reverse detection results: Weak vs Original vs Strong
   - Weak prompt的cross-model generalization如何?

3. SMOTE comparison: Weak prompt与SMOTE的差距

**可能的"重要性"含义**:
- **Scenario 1**: Weak prompt生成的synthetic spam更真实,因此训练效果更好
- **Scenario 2**: Weak prompt更难被检测出是synthetic,表示质量更高
- **Scenario 3**: Weak prompt在real-world deployment中更practical

**建议行动**:
- [ ] 重新检查所有evaluation results,对比三种prompt的表现
- [ ] 计算weak vs strong vs original的平均performance和variance
- [ ] 在paper的Discussion中添加"Prompt Strategy Impact"分析
- [ ] 如果weak确实更好,需要在conclusion中强调

**优先级**: MEDIUM (需要先验证数据)

---

### 4. SMOTE Justification SMOTE作为baseline的合理性

**Co-author建议**:
- SMOTE is over-sampling, why use SMOTE for synthetic generation
- Need literature support

**分析**:
- ✅ **合理concern**: SMOTE确实传统上是用于class imbalance,不是synthetic data generation
- ⚠️ **需要clarify**: 我们使用SMOTE是作为baseline comparison,不是作为主要方法

**需要在paper中clarify的内容**:

1. **SMOTE Purpose Clarification** (Introduction/Methodology):
   ```
   我们使用SMOTE不是因为class imbalance问题(我们的10% spam ratio是realistic的),
   而是作为feature-space augmentation的baseline,对比LLM-based text-space generation
   ```

2. **Literature Support for SMOTE in Text Classification**:
   - 需要找到使用SMOTE做text augmentation的相关文献
   - 说明SMOTE在TF-IDF feature space上的interpolation是合理的baseline
   - 强调SMOTE vs LLM的对比是"feature-space vs text-space"的对比

3. **SMOTE Limitations as Synthetic Generation** (Discussion):
   - SMOTE generates feature vectors, not interpretable text
   - SMOTE interpolation may create unrealistic feature combinations
   - LLM generation is semantically coherent but feature-space effectiveness varies

**建议行动**:
- [ ] 搜索literature: SMOTE + text classification + augmentation
- [ ] 在methodology section明确说明SMOTE是baseline而非主要方法
- [ ] 在introduction中clarify为什么需要这个baseline
- [ ] 在discussion中讨论SMOTE的局限性

**优先级**: HIGH (reviewer肯定会问)

**Literature search keywords**:
- "SMOTE text classification"
- "SMOTE data augmentation NLP"
- "feature space augmentation text"
- "oversampling vs data generation"

---

### 5. Model Selection Cost Efficiency 模型选择的成本考虑

**Co-author建议**:
- Why use GPT-4.1-mini and Claude-3.5-Haiku instead of expensive models
- Cost efficiency consideration

**分析**:
- ✅ **非常practical**: 这是实际应用中的重要考虑
- 需要在paper中explicitly说明

**需要在paper中添加的内容**:

1. **Model Selection Justification** (Methodology section):
   ```latex
   We employ GPT-4.1-mini and Claude-3.5-Haiku—the cost-efficient variants
   of their respective model families—to evaluate synthetic spam generation
   under realistic resource constraints. These models offer:

   - Cost efficiency: ~10-20× cheaper than flagship models (GPT-4, Claude-3.5-Sonnet)
   - Sufficient capability: Maintain strong text generation quality
   - Practical relevance: More likely to be used by actual spammers due to lower cost
   - Scalability: Enable large-scale experiments (12,000 API calls) within budget
   ```

2. **Cost-Quality Trade-off Discussion** (Discussion or Conclusion):
   - Spammers would use cheaper models for cost reasons
   - Our findings on cheaper models are more relevant to real-world spam detection
   - Future work could compare with more expensive models

3. **Computational Cost Reporting** (可以加到appendix):
   - Total API calls: 12,000
   - Estimated cost: $XX for GPT-4.1-mini, $YY for Claude-3.5-Haiku
   - Comparison with flagship model costs

**建议行动**:
- [ ] 在methodology section添加model selection justification
- [ ] 计算并报告实际API costs (如果可以的话)
- [ ] 在discussion中提到cost-efficiency作为practical consideration
- [ ] 在future work中提到可以测试更贵的模型

**优先级**: MEDIUM (加分项,但不是critical)

---

### 6. Hyperparameter Details 超参数详细说明

**Co-author建议**:
- Hyperparameters of classifiers, especially SVM grid search or explanations

**分析**:
- ✅ **必须clarify**: Reviewers会质疑classifier performance comparison的fairness
- 需要transparent报告所有hyperparameters和tuning process

**需要检查的内容**:

1. **Current Hyperparameters** (从experiment.tex line 112):
   ```
   SVM: RBF kernel, C=1.0, balanced class weights
   RF: 100 trees, max_depth=10, balanced class weights
   TF-IDF: max 10,000 features, unigram-bigram (1-2), English stop words
   ```

2. **需要clarify的问题**:
   - 这些hyperparameters是如何选择的?
   - 是否进行了grid search或cross-validation?
   - 是否在所有实验中使用相同的hyperparameters?
   - 是否针对different synthetic ratios调整过hyperparameters?

**需要在paper中添加的内容**:

1. **Methodology Section Enhancement**:
   ```latex
   Hyperparameters were selected through preliminary grid search on a
   held-out validation set (details in Appendix~\ref{appendix:hyperparam_tuning}).
   We employ fixed hyperparameters across all experiments to ensure fair
   comparison across synthetic data configurations, eliminating hyperparameter
   optimization as a confounding factor.
   ```

2. **Appendix: Hyperparameter Tuning** (new section):
   - Grid search ranges for SVM: C ∈ {0.1, 1.0, 10.0}, kernel ∈ {linear, rbf}
   - Grid search ranges for RF: n_estimators ∈ {50, 100, 200}, max_depth ∈ {5, 10, 20}
   - Cross-validation results on validation set
   - Justification for final hyperparameter choices

3. **Limitation Discussion**:
   ```
   We use fixed hyperparameters for fair comparison, but adaptive hyperparameter
   tuning for different synthetic ratios might further improve performance.
   Future work could explore adaptive tuning strategies.
   ```

**建议行动**:
- [ ] 检查code确认使用的hyperparameters
- [ ] 如果做过grid search,把结果整理到appendix
- [ ] 如果没做grid search,在methodology中说明使用default/literature-recommended values
- [ ] 在limitation中提到固定hyperparameter的trade-off

**优先级**: HIGH (必须clarify)

---

### 7. Dataset Selection Justification 数据集选择理由

**Co-author建议**:
- Why use CEAS08 data
- Availability, popularity

**分析**:
- ✅ **需要explicit说明**: Dataset choice总是需要justification
- CEAS08确实有availability和popularity优势

**需要在paper中添加的内容**:

1. **Dataset Description Section Enhancement** (Methodology):
   ```latex
   We select the CEAS-08 spam email dataset~\cite{champa2024curated} for the
   following reasons:

   \begin{itemize}
       \item \textbf{Public Availability}: Widely accessible for research reproducibility
       \item \textbf{Established Benchmark}: Extensively used in prior spam detection
             research, enabling comparison with existing work
       \item \textbf{Substantial Size}: 39,154 emails provide sufficient data for
             statistical replication (20 groups × 1,000 samples)
       \item \textbf{Realistic Distribution}: 55.8\% spam ratio reflects historical
             email spam prevalence
       \item \textbf{Diverse Spam Patterns}: Captures multiple spam tactics and
             strategies from real-world campaigns
   \end{itemize}
   ```

2. **Limitation Discussion** (Discussion section):
   ```latex
   \textbf{Dataset Limitations}: The CEAS-08 dataset, collected in 2008, may not
   fully represent modern spam tactics. However, it remains valuable for:
   (1) controlled experimental evaluation of synthetic data generation principles,
   (2) reproducibility and comparison with prior work, and
   (3) methodological insights that generalize beyond specific spam patterns.
   Future work should validate findings on contemporary spam datasets.
   ```

**建议行动**:
- [ ] 在methodology section添加explicit dataset justification
- [ ] 在limitation section提到dataset age作为limitation
- [ ] 在future work中提到需要在newer datasets上验证
- [ ] 找到使用CEAS08的相关文献并cite

**优先级**: MEDIUM (可以快速添加)

---

### 8. Sample Size and Statistical Power 样本量和统计效力

**Co-author建议**:
- Experiment size is small (N=1000 for 20 groups)
- Might not be accurate, that's why we use statistics
- Sample size is small, data is old

**分析**:
- ✅ **honest concern**: N=1000 per group确实相对较小
- ⚠️ **但有mitigation**: 20 groups replication提供statistical power
- 需要在paper中explicitly address这个limitation并explain mitigation

**需要在paper中clarify的内容**:

1. **Sample Size Justification** (Methodology):
   ```latex
   Each group contains N=1,000 samples (100 spam, 900 ham) with 10\% spam ratio.
   While individual group size is moderate, our replicated design with R=20
   independent groups provides statistical power exceeding 80\% for detecting
   medium effect sizes (Cohen's d ≥ 0.5) at significance level α=0.05
   (power analysis in Section~\ref{sec:methodology}).
   ```

2. **Statistical Rigor Explanation** (Methodology):
   ```latex
   To address the moderate per-group sample size, we employ rigorous statistical
   methodology:

   - \textbf{Replication}: 20 independent groups enable robust mean estimation
   - \textbf{Confidence Intervals}: Report 95\% CI to quantify estimation uncertainty
   - \textbf{Hypothesis Testing}: Paired t-tests with FDR correction (α=0.05)
   - \textbf{Effect Sizes}: Cohen's d to assess practical significance
   - \textbf{Cross-validation}: Consistent train-test split across all experiments
   ```

3. **Limitation and Threat to Validity** (Discussion):
   ```latex
   \textbf{Sample Size Limitations}: Our per-group sample size (N=1,000) is
   moderate compared to production spam filtering datasets. However, this is
   mitigated by: (1) systematic replication across 20 groups providing statistical
   power, (2) conservative statistical testing with FDR correction, and
   (3) focus on relative performance comparisons rather than absolute benchmarks.
   Larger-scale validation would strengthen confidence in findings.

   \textbf{Dataset Currency}: The CEAS-08 dataset from 2008 may not capture
   modern spam tactics. However, our focus is on methodological insights about
   synthetic data generation rather than absolute spam detection performance.
   The principles discovered—synthetic data degradation patterns, cross-model
   effects, architectural robustness—are likely to generalize to contemporary
   spam, though validation on newer datasets is recommended.
   ```

**建议行动**:
- [ ] 在methodology section强调replication带来的statistical power
- [ ] 在discussion section添加dedicated "Limitations and Threats to Validity"小节
- [ ] 计算并报告actual statistical power (可能已经在methodology中有了)
- [ ] 在conclusion/future work中提到需要larger-scale validation

**优先级**: HIGH (必须address这个concern)

---

### 9. Embeddings Visualization 嵌入空间可视化

**Co-author建议**:
- Add embeddings graphs
- Previous experiment plot of embeddings found overlapped issue

**分析**:
- ✅ **非常valuable**: 可视化可以直观展示synthetic vs real的feature space difference
- 🤔 **需要确认**: Previous experiment是指哪个?需要找到这个图

**需要做的事情**:

1. **找到之前的embeddings visualization**:
   - 检查是否有之前的实验产生过embeddings plot
   - 确认是t-SNE, UMAP, 还是PCA
   - 确认展示的是什么: TF-IDF embeddings? LLM embeddings?

2. **如果需要重新生成embeddings plot**:
   - 使用TF-IDF features (与classifier使用的相同)
   - 降维方法: t-SNE或UMAP (2D visualization)
   - 可视化内容:
     - Real spam vs Real ham
     - Real spam vs Synthetic spam (GPT, Claude, different prompts)
     - Real spam vs SMOTE synthetic
   - 展示"overlapped issue": synthetic spam与real spam/ham的feature space overlap

3. **Paper中的位置**:
   - 可以放在Evaluation section的"Synthetic Data Quality Analysis"小节
   - 或者放在Discussion section的"Why Synthetic Data Degrades Performance"分析中

**可能的figure caption**:
```latex
\begin{figure}[h]
\centering
\includegraphics[width=\columnwidth]{figures/embeddings_tsne.png}
\caption{t-SNE visualization of TF-IDF feature space showing real spam (blue),
         real ham (green), GPT synthetic spam (red), and Claude synthetic spam
         (orange). Synthetic spam exhibits substantial overlap with real ham
         region, explaining classifier confusion and performance degradation.}
\label{fig:embeddings_overlap}
\end{figure}
```

**建议行动**:
- [ ] 搜索之前的实验结果,找到embeddings plot
- [ ] 如果没有,生成new embeddings visualization
- [ ] 选择representative samples (real vs synthetic, different prompts)
- [ ] 添加到paper的evaluation或discussion section
- [ ] 在text中refer这个figure来解释performance degradation

**优先级**: MEDIUM-HIGH (很有说服力的visualization)

**需要确认**: 这个"previous experiment"是指什么?需要user clarify

---

### 10. Synthetic Examples 合成样本示例

**Co-author建议**:
- Add synthetic LLM generated examples
- Show original, strong, weak prompts

**分析**:
- ✅ **非常important**: Examples让paper更concrete和readable
- 应该展示三种prompt strategy的实际output差异

**需要在paper中添加的内容**:

1. **Examples Section or Table**:
   可以放在:
   - Methodology section的"Prompt Engineering Strategies"之后
   - Appendix (如果examples太长)

2. **建议的table format**:
   ```latex
   \begin{table*}[t]
   \centering
   \caption{Example Synthetic Spam Generated by Different Prompt Strategies}
   \label{tab:synthetic_examples}
   \begin{tabular}{p{0.15\textwidth}p{0.8\textwidth}}
   \toprule
   \textbf{Prompt} & \textbf{Generated Spam Example} \\
   \midrule
   \textbf{Original} &
   Subject: Amazing Deal on Luxury Watches! \\
   Dear Valued Customer, Don't miss out on our exclusive 70\% off sale on
   authentic luxury watches. Limited time offer! Click here to claim your
   discount: [malicious-link]. Act now before stocks run out! \\
   \midrule
   \textbf{Strong} &
   Subject: !!!WIN FREE iPHONE NOW!!! \\
   CONGRATULATIONS!!! You have been selected as our LUCKY WINNER!!! Claim
   your FREE iPhone 15 Pro Max NOW!!! NO PURCHASE NECESSARY!!! Click HERE
   immediately: [malicious-link]. 100\% FREE GUARANTEE!!! \\
   \midrule
   \textbf{Weak} &
   Subject: Quick question about your account \\
   Hello, We noticed some unusual activity on your account and wanted to
   verify your information. Please review your recent transactions at your
   earliest convenience. If you have any questions, feel free to contact us. \\
   \bottomrule
   \end{tabular}
   \end{table*}
   ```

3. **Examples的选择标准**:
   - Representative: 展示typical characteristics of each prompt
   - Comparable: 最好是same topic或similar length
   - Illustrative: 清楚展示三种prompt的difference
   - 需要anonymize: 移除actual malicious links

4. **配合text说明**:
   ```latex
   Table~\ref{tab:synthetic_examples} illustrates the distinct characteristics
   of each prompt strategy:

   - \textbf{Original prompt} generates realistic spam with moderate manipulation
     tactics, balancing deception and believability.

   - \textbf{Strong prompt} produces highly aggressive spam with excessive
     capitalization, multiple exclamation marks, and urgent calls-to-action,
     making it easily detectable by classifiers.

   - \textbf{Weak prompt} creates subtle spam that resembles legitimate emails,
     using professional language and minimal urgency, which may be harder to
     detect but also less effective as actual spam.
   ```

**建议行动**:
- [ ] 从generated synthetic data中挑选representative examples
- [ ] 确保examples是anonymized和safe to publish
- [ ] 为每个prompt strategy选择1-2个好的examples
- [ ] 创建table并添加到methodology或appendix
- [ ] 在text中refer这个table并解释differences

**优先级**: HIGH (make paper much more readable)

---

## 优先级总结 (Priority Summary)

### 🔴 HIGH Priority - Must Address Before Submission

1. **[Point 1] Real-to-Synthetic Detection Experiment** - 需要与co-author确认是否要做
2. **[Point 2] SVM vs RF Explanation** - 必须在paper中clarify
3. **[Point 4] SMOTE Justification** - reviewer肯定会问
4. **[Point 6] Hyperparameter Details** - 必须transparent
5. **[Point 8] Sample Size Limitations** - 必须address
6. **[Point 10] Synthetic Examples** - 提高paper readability

### 🟡 MEDIUM Priority - Should Add for Stronger Paper

7. **[Point 3] Weak Prompt Importance** - 需要先验证数据
8. **[Point 5] Cost Efficiency** - 加分项
9. **[Point 7] Dataset Justification** - 容易添加
10. **[Point 9] Embeddings Visualization** - 很有说服力

---

## 建议的Next Steps

### Immediate Actions (本周完成):

1. **与co-author确认**:
   - [ ] Point 1: 是否要做real-to-synthetic detection实验?
   - [ ] Point 9: Previous embeddings plot在哪里?
   - [ ] Point 3: Weak prompt "more important"的具体含义?

2. **数据验证和分析**:
   - [ ] 重新检查weak vs strong vs original的performance data
   - [ ] 验证hyperparameter settings in code
   - [ ] 计算statistical power (如果还没有)

3. **Literature search**:
   - [ ] SMOTE在text classification中的应用文献
   - [ ] CEAS08 dataset的相关引用文献

### Paper Revisions (下周完成):

1. **Methodology Section**:
   - [ ] Add model selection justification (cost efficiency)
   - [ ] Add hyperparameter selection explanation
   - [ ] Add dataset selection justification
   - [ ] Add sample size and statistical power explanation
   - [ ] Clarify SMOTE purpose (baseline comparison)

2. **Evaluation Section**:
   - [ ] Add synthetic examples table
   - [ ] Add embeddings visualization (if available)
   - [ ] Add weak prompt performance analysis

3. **Discussion Section**:
   - [ ] Add "Classifier Architecture and Robustness" subsection
   - [ ] Add "Limitations and Threats to Validity" subsection
   - [ ] Add SMOTE limitations discussion

4. **Appendix**:
   - [ ] Add hyperparameter tuning details
   - [ ] Add longer synthetic examples (if needed)
   - [ ] Add computational cost breakdown

---

## 需要与Co-author进一步讨论的问题

1. **Real-to-Synthetic Detection Experiment**:
   - 实验的具体研究问题是什么?
   - 需要测试哪些配置? (full factorial or subset?)
   - 时间和计算资源是否充足?
   - 是否会delay paper submission?

2. **Weak Prompt Importance**:
   - "More important"的具体含义?
   - 是based on什么数据或observation?
   - 希望在paper中如何强调?

3. **Paper Revision Timeline**:
   - 目标submission deadline?
   - 哪些revisions是blocking vs nice-to-have?
   - 是否需要re-run任何experiments?

---

## 个人评估总结

**总体评价**: Co-author的反馈非常valuable和practical,大部分建议都很合理且必要。

**最critical的issues**:
1. SVM vs RF explanation (必须clarify)
2. Hyperparameter transparency (必须clarify)
3. Sample size limitations (必须address)
4. SMOTE justification (reviewer会问)

**最有价值的additions**:
1. Synthetic examples (大幅提高readability)
2. Embeddings visualization (直观展示问题)
3. Real-to-synthetic detection (如果时间允许,是重要补充)

**建议优先处理**: Points 2, 4, 6, 8, 10 (都是high priority且相对容易完成)

**需要further discussion**: Points 1, 3, 9 (需要co-author clarify具体要求)
