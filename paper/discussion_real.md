# Synthetic Spam Email Data Generation: A Multi-Sample Evaluation Framework

## 1. 研究背景与动机

网络安全领域中的恶意数据（如spam邮件）通常呈现严重的数据不平衡问题，在真实场景中spam与legitimate邮件的比例往往为1:9或更低。这种不平衡性给机器学习模型的训练带来了以下挑战：

- **数据获取成本高**: 收集足够的labeled spam数据需要大量人力和时间成本
- **模型性能评估困难**: 在极不平衡数据上，准确率等指标容易产生误导
- **泛化能力不足**: 有限的spam样本可能导致模型过拟合

因此，我们提出使用大语言模型（LLM）生成synthetic spam email data来缓解这些问题。

## 2. 研究问题

**核心研究问题**: LLM生成的synthetic spam email data是否能够有效提升机器学习模型在不平衡数据集上的性能？

**具体研究子问题**:
1. 在什么比例的synthetic spam data能够达到最优性能？
2. 不同prompt策略对synthetic data质量的影响如何？
3. 不同LLM引擎在spam email生成任务上的表现差异？
4. synthetic data的加入是否会导致model performance的显著衰减？衰减的临界值是什么？

## 3. 实验设计框架

### 3.1 数据配置策略

**固定比例设计**:
- Spam : Non-spam = 1:9 (模拟真实网络安全场景)
- Testing data: 完全使用真实数据，保持不变
- Training data: 动态调整synthetic spam data比例

**Synthetic Spam Data Ratio调节**:
- 比例范围: 0%, 10%, 20%, 30%, ..., 100%
- 0% = 完全真实spam数据（baseline）
- 100% = 完全synthetic spam数据

### 3.2 Multi-Sample Evaluation Protocol

**重复抽样设计**:
- 分组数量: R = 20组（确保统计显著性）
- 每组样本量: N = 9000
- 每组构成: 1/10 × N spam + 9/10 × N non-spam
- 抽样方式: 不重复随机抽样（stratified sampling）

**评估指标**:
- 分类性能: Accuracy, Precision, Recall, F1-score
- 统计分析: Mean, Standard Deviation, t-test, p-value
- 分布比较: Optimal Transport, K-L Divergence

### 3.3 优化框架

定义优化函数: **O(L, %, M)**

其中:
- **L**: Prompt strategies (原始重写, 强化重写, 弱化重写)
- **%**: Synthetic spam data ratio (0% - 100%)
- **M**: LLM engines (GPT-4.1-mini, Claude, Gemini Pro 2.5)

目标: 找到使分类性能最优的参数组合 (L*, %*, M*)

## 4. 第一阶段实验设计（Pilot Study）

### 4.1 实验参数

**数据集**: CEAS-08 (106,412 samples)
**LLM引擎**: GPT-4.1-mini
**Prompt策略**:
1. **Original**: "请重写以下spam邮件，保持原有的恶意意图和内容结构"
2. **Strong**: "请生成一封具有强烈spam特征的邮件，包含明显的营销词汇和紧迫感"
3. **Weak**: "请生成一封subtle spam邮件，使其看起来更像legitimate邮件但仍保持spam本质"

**分类器**: SVM, Random Forest, Deep Learning
**重复次数**: R = 20
**样本量**: N = 9000

### 4.2 实验流程

```
For each synthetic_ratio in [0%, 10%, 20%, ..., 100%]:
    For each prompt_strategy in [Original, Strong, Weak]:
        For trial in range(20):  # R = 20
            # 1. 数据生成
            generate_synthetic_spam_data(prompt_strategy, synthetic_ratio)

            # 2. 数据集构建
            training_set = combine_data(real_non_spam, mixed_spam_data)
            testing_set = real_data_only

            # 3. 模型训练与评估
            for classifier in [SVM, RandomForest, DeepLearning]:
                model = train(training_set)
                metrics = evaluate(model, testing_set)
                record_results(metrics, synthetic_ratio, prompt_strategy, trial)
```

### 4.3 预期输出

**统计分析结果**:
- 每个synthetic ratio下的performance metrics分布
- 不同prompt策略的效果对比
- Performance衰减曲线和临界值识别

**可视化输出**:
- Performance metrics vs Synthetic ratio曲线图
- 不同prompt策略的box plot比较
- 统计显著性热力图

## 5. 扩展实验计划

### 5.1 第二阶段: 多LLM对比

在pilot study验证可行性后，扩展到：
- **Claude Sonnet**: 对比生成质量和多样性
- **Gemini Pro 2.5**: 评估不同架构的生成能力

### 5.2 第三阶段: 多数据集验证

扩展到其他数据集：
- TREC-07, Assassin, Enron
- 验证跨数据集的泛化能力

### 5.3 第四阶段: 深度分析

- **语义分析**: 使用embedding分析synthetic与real data的语义相似性
- **对抗性评估**: 评估synthetic data是否容易被检测
- **Error analysis**: 深入分析performance衰减的具体原因

## 6. 学术贡献与意义

### 6.1 理论贡献

1. **首次系统性研究**: LLM在不平衡网络安全数据生成中的应用效果
2. **统计严谨性**: 提供了robust的multi-sample evaluation framework
3. **优化理论**: 建立了prompt-ratio-model的三维优化空间

### 6.2 实践价值

1. **成本降低**: 减少expensive manual labeling的需求
2. **性能提升**: 在data-scarce场景下提升模型性能
3. **可复制性**: 提供了standardized的评估协议

### 6.3 网络安全影响

1. **实时响应**: 快速生成训练数据应对新型spam攻击
2. **隐私保护**: 避免使用真实用户邮件数据
3. **模型鲁棒性**: 提升模型对spam变种的检测能力

## 7. 技术实现路径

### 7.1 数据处理pipeline
```
Raw Email Data → Preprocessing → Stratified Sampling →
LLM Generation → Data Mixing → Model Training → Evaluation
```

### 7.2 关键技术挑战
- **Prompt Engineering**: 设计有效的generation prompts
- **Quality Control**: 确保synthetic data的质量和多样性
- **Statistical Analysis**: 处理多重比较和多重测试问题

### 7.3 评估标准
- **内在评估**: Perplexity, BLEU score与真实数据的相似性
- **外在评估**: 下游classification task的performance
- **人工评估**: Expert review的quality assessment

## 8. 预期挑战与解决方案

### 8.1 潜在挑战
1. **计算资源需求**: 大规模LLM调用的成本控制
2. **数据质量控制**: 确保synthetic data的authenticity
3. **统计功效**: 样本量是否足够检测significant difference

### 8.2 解决方案
1. **分阶段实施**: 先验证可行性再扩大规模
2. **质量监控**: 建立automated quality check机制
3. **Power Analysis**: 预先计算所需的统计功效

## 9. 时间计划

- **Week 1-2**: Pilot study implementation
- **Week 3-4**: 初步结果分析和调优
- **Week 5-8**: 多LLM扩展实验
- **Week 9-12**: 深度分析和论文撰写

这个研究框架将为网络安全领域的synthetic data generation提供重要的理论基础和实践指导。

## 10. 技术细节讨论与解决方案

### 10.1 Spam Ratio设计合理性分析

**问题**: Spam:Non-spam = 1:9 是否合理？

**分析与建议**:

**合理性支撑**:
- **现实匹配**: 真实网络环境中spam邮件比例通常在5-15%之间，1:9比例（10%）处于合理范围
- **业界标准**: 多数cybersecurity研究采用类似的不平衡比例
- **挑战性**: 足够不平衡以体现real-world困难，但不至于过于极端

**潜在挑战与解决方案**:
```python
# 需要处理class imbalance的策略
1. Class weighting:
   - SVM: class_weight='balanced'
   - Random Forest: class_weight='balanced_subsample'
   - Deep Learning: 使用weighted loss function

2. Evaluation metrics调整:
   - 重点关注Precision, Recall, F1-score而非Accuracy
   - 使用AUC-ROC和AUC-PR作为主要评估指标
   - 计算balanced accuracy

3. Sampling策略:
   - 在training时可考虑SMOTE或其他oversampling技术
   - 但要确保testing set保持真实分布
```

**最终建议**: 1:9比例合理，但需要配套的class imbalance处理策略。

### 10.2 分组抽样与泛化能力验证

**问题**: 组内生成 vs 跨组混合的对比实验设计是否严谨？

**详细实验设计**:

```python
# 两种synthetic data生成策略对比
Strategy A: 组内生成 (Within-group generation)
For each group i in range(20):
    real_spam_group_i = real_spam[group_i_indices]
    synthetic_spam_group_i = LLM_generate(real_spam_group_i, prompt)
    training_set_i = combine(synthetic_spam_group_i, real_non_spam_group_i)

Strategy B: 跨组混合 (Cross-group mixing)
For each group i in range(20):
    # 从其他组随机抽取real spam作为生成source
    other_groups = [j for j in range(20) if j != i]
    source_spam = random_sample_from(real_spam[other_groups])
    synthetic_spam_group_i = LLM_generate(source_spam, prompt)
    training_set_i = combine(synthetic_spam_group_i, real_non_spam_group_i)
```

**假设验证逻辑**:
- **H1**: 如果Strategy A >> Strategy B，说明LLM生成具有局部优化特性，泛化能力有限
- **H2**: 如果Strategy A ≈ Strategy B，说明LLM具有良好的泛化能力
- **H3**: 如果两者都显著低于real data baseline，说明synthetic data整体质量不足

**技术实现要点**:
```python
# 确保数据完整性的unique ID系统
def assign_unique_ids(datasets):
    for source_name, data in datasets.items():
        data['unique_id'] = f"{source_name}_{range(len(data))}"
        data['source'] = source_name
    return datasets

# 严格的分组策略，避免data leakage
def stratified_group_split(data, n_groups=20, spam_ratio=0.1):
    spam_data = data[data['label'] == 1]
    non_spam_data = data[data['label'] == 0]

    # 确保每组包含足够的spam样本进行生成
    spam_per_group = len(spam_data) // n_groups
    non_spam_per_group = int(spam_per_group * 9)  # 1:9 ratio

    groups = []
    for i in range(n_groups):
        group_spam = spam_data[i*spam_per_group:(i+1)*spam_per_group]
        group_non_spam = non_spam_data[i*non_spam_per_group:(i+1)*non_spam_per_group]
        groups.append({
            'spam': group_spam,
            'non_spam': group_non_spam,
            'group_id': i
        })
    return groups
```

**统计严谨性**: 这个设计很严谨，能够有效验证LLM生成的泛化能力。

### 10.3 超参数优化与资源配置

**问题**: R、N、synthetic ratio增量、数据源充足性是否合适？

**参数分析与建议**:

**1. 分组数量 R = 20**
```python
# Power analysis计算所需样本量
import scipy.stats as stats

def calculate_required_groups(effect_size=0.3, alpha=0.05, power=0.8):
    """
    effect_size: Cohen's d (0.3 = medium effect)
    alpha: Type I error rate
    power: Statistical power (1 - Type II error rate)
    """
    # 使用t-test的power analysis
    required_n = stats.ttest_power(effect_size, power, alpha, alternative='two-sided')
    return math.ceil(required_n)

# 建议: R = 30 更保险，但R = 20在medium effect size下也可接受
```

**2. 样本量 N = 9000**
```python
# 分析每组构成
N = 9000
spam_samples = 900    # 10% of N
non_spam_samples = 8100  # 90% of N

# 对于Deep Learning的考虑
# - 900个spam样本对于简单网络足够
# - 但对于复杂模型可能不足
# - 建议从N=9000开始，根据结果调整到N=15000

# 验证数据充足性
CEAS08_total = 106412
required_total = 20 * 9000  # 180,000
# CEAS-08足够支撑这个实验规模
```

**3. Synthetic ratio增量**
```python
# 当前设计: [0%, 10%, 20%, ..., 100%] 共11个点
# 建议优化:
phase1_ratios = [0%, 25%, 50%, 75%, 100%]  # 粗略扫描
phase2_ratios = [0%, 10%, 20%, ..., 100%]  # 细致分析
phase3_ratios = [0%, 5%, 10%, 15%, ..., 100%]  # 关键区域精细化

# 特别关注critical transition points
```

**4. 数据源充足性**
```python
# 验证现有数据源是否支撑实验
datasets_info = {
    'CEAS-08': {'total': 106412, 'spam': 70432, 'non_spam': 35980},
    'TREC-07': {'total': 134136, 'spam': 87775, 'non_spam': 46361},
    'Assassin': {'total': 111887, 'spam': 88808, 'non_spam': 23079},
    'Enron': {'total': 72122, 'spam': 13856, 'non_spam': 58266}
}

# 实验需求 (单个数据源)
experiment_requirement = {
    'total_needed': 20 * 9000,  # 180,000
    'spam_needed': 20 * 900,    # 18,000
    'non_spam_needed': 20 * 8100  # 162,000
}

# 所有数据源都能支撑单独实验
# CEAS-08和TREC-07最适合作为主要测试集
```

### 10.4 统计检验与多重比较处理

**问题**: p-value是否指t-test？需要其他统计指标吗？

**统计检验策略**:

```python
# 1. 基础统计检验
def statistical_analysis(results_dict):
    """
    results_dict: {synthetic_ratio: [metrics_list_across_trials]}
    """

    # Paired t-test: 比较不同synthetic ratio与baseline (0%)
    baseline_results = results_dict[0]  # 0% synthetic (all real)

    statistical_results = {}
    for ratio in [10, 20, 30, ..., 100]:
        ratio_results = results_dict[ratio]

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(baseline_results, ratio_results)

        # Effect size (Cohen's d)
        cohen_d = (np.mean(baseline_results) - np.mean(ratio_results)) / \
                  np.sqrt(((np.var(baseline_results) + np.var(ratio_results)) / 2))

        # Non-parametric alternative (Wilcoxon signed-rank test)
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(baseline_results, ratio_results)

        statistical_results[ratio] = {
            't_test_p': p_value,
            'cohens_d': cohen_d,
            'wilcoxon_p': wilcoxon_p,
            'mean_diff': np.mean(baseline_results) - np.mean(ratio_results)
        }

    return statistical_results

# 2. 多重比较校正
def multiple_comparison_correction(p_values, method='fdr_bh'):
    """
    method: 'bonferroni', 'fdr_bh' (Benjamini-Hochberg), 'fdr_by'
    """
    from statsmodels.stats.multitest import multipletests

    rejected, p_adjusted, alpha_sidak, alpha_bonf = multipletests(
        p_values, alpha=0.05, method=method
    )

    return {
        'rejected': rejected,
        'p_adjusted': p_adjusted,
        'method': method
    }

# 3. ANOVA用于多组比较
def anova_analysis(results_by_prompt):
    """
    比较不同prompt策略的效果
    results_by_prompt: {'original': [results], 'strong': [results], 'weak': [results]}
    """
    from scipy.stats import f_oneway

    f_stat, p_value = f_oneway(*results_by_prompt.values())

    # Post-hoc analysis (Tukey HSD)
    from scipy.stats import tukey_hsd
    tukey_result = tukey_hsd(*results_by_prompt.values())

    return {
        'anova_f': f_stat,
        'anova_p': p_value,
        'tukey_hsd': tukey_result
    }
```

**关键统计指标**:
- **t-test p-value**: 主要显著性检验
- **Cohen's d**: 效应量大小
- **Wilcoxon test**: 非参数替代检验
- **FDR校正**: 控制False Discovery Rate
- **ANOVA**: 多组间比较
- **Confidence Intervals**: 提供效应量范围

### 10.5 Embedding Analysis集成策略

**问题**: 之前的embedding analysis (edge vs core)是否还有用？

**集成方案**:

```python
# 将embedding analysis作为synthetic data质量评估的补充维度
def comprehensive_evaluation_framework():

    # 1. Performance-based evaluation (主要评估)
    performance_metrics = {
        'accuracy', 'precision', 'recall', 'f1_score',
        'auc_roc', 'auc_pr'
    }

    # 2. Embedding-based evaluation (质量评估)
    embedding_metrics = {
        'edge_vs_core_similarity',
        'real_vs_synthetic_distance',
        'intra_cluster_coherence',
        'inter_cluster_separation'
    }

    # 3. Distribution-based evaluation (分布评估)
    distribution_metrics = {
        'kl_divergence',
        'optimal_transport_distance',
        'maximum_mean_discrepancy'
    }

    return {
        'performance': performance_metrics,
        'embedding': embedding_metrics,
        'distribution': distribution_metrics
    }

# Edge vs Core分析的新用途
def edge_core_analysis_for_synthetic_quality(real_data, synthetic_data):
    """
    使用edge vs core分析来评估synthetic data的质量
    """

    # 1. 计算real data的edge/core structure
    real_embeddings = get_embeddings(real_data)
    real_edge_core = compute_edge_core_structure(real_embeddings)

    # 2. 计算synthetic data的edge/core structure
    synthetic_embeddings = get_embeddings(synthetic_data)
    synthetic_edge_core = compute_edge_core_structure(synthetic_embeddings)

    # 3. 比较结构相似性
    structure_similarity = compare_edge_core_structures(
        real_edge_core, synthetic_edge_core
    )

    # 4. 预测performance
    # 如果synthetic保持了类似的edge/core结构，性能应该更好
    predicted_performance = predict_performance_from_structure(structure_similarity)

    return {
        'structure_similarity': structure_similarity,
        'predicted_performance': predicted_performance,
        'quality_score': compute_quality_score(structure_similarity)
    }
```

**学术价值**: Edge vs core分析可以提供performance metrics之外的解释性洞察，帮助理解为什么某些synthetic data效果更好。

### 10.6 其他技术建议与改进

**1. Cross-validation策略**
```python
# 时间序列式cross-validation (避免data leakage)
def temporal_cross_validation(data, n_splits=5):
    """
    确保训练集的时间早于测试集，模拟真实部署场景
    """
    # 按时间戳排序（如果有的话）
    # 或者按照数据收集顺序进行分割
    pass

# Stratified cross-validation (确保每折中class分布一致)
from sklearn.model_selection import StratifiedKFold
def stratified_evaluation(X, y, model, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
    return scores
```

**2. 实验可重现性**
```python
# 确保实验可重现的配置
RANDOM_SEEDS = {
    'data_split': 42,
    'model_training': 123,
    'synthetic_generation': 456,
    'cross_validation': 789
}

def set_all_seeds(seed_dict):
    np.random.seed(seed_dict['data_split'])
    random.seed(seed_dict['data_split'])
    # 如果使用torch
    torch.manual_seed(seed_dict['model_training'])
    torch.cuda.manual_seed_all(seed_dict['model_training'])
```

**3. 计算资源优化**
```python
# 分阶段实验策略，控制成本
phase_1_config = {
    'datasets': ['CEAS-08'],
    'llm_engines': ['GPT-4.1-mini'],
    'prompt_strategies': ['original'],
    'synthetic_ratios': [0, 25, 50, 75, 100],
    'n_groups': 10,
    'sample_size': 6000
}

phase_2_config = {
    'datasets': ['CEAS-08'],
    'llm_engines': ['GPT-4.1-mini'],
    'prompt_strategies': ['original', 'strong', 'weak'],
    'synthetic_ratios': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'n_groups': 20,
    'sample_size': 9000
}

# 只有在phase_1结果promising的情况下才进行phase_2
```

**4. 质量控制机制**
```python
def quality_control_pipeline(synthetic_data, real_data):
    """
    多层次的质量控制
    """

    # 1. 基础质量检查
    basic_checks = {
        'length_similarity': check_length_distribution(synthetic_data, real_data),
        'vocabulary_overlap': check_vocabulary_overlap(synthetic_data, real_data),
        'language_quality': check_language_quality(synthetic_data)
    }

    # 2. 语义质量检查
    semantic_checks = {
        'embedding_similarity': check_embedding_similarity(synthetic_data, real_data),
        'topic_consistency': check_topic_consistency(synthetic_data, real_data)
    }

    # 3. 任务相关质量检查
    task_checks = {
        'spam_characteristics': check_spam_characteristics(synthetic_data),
        'classifier_confusion': check_classifier_confusion(synthetic_data, real_data)
    }

    return {
        'basic': basic_checks,
        'semantic': semantic_checks,
        'task_specific': task_checks,
        'overall_quality_score': compute_overall_quality(basic_checks, semantic_checks, task_checks)
    }
```

## 10.7 总结与建议

基于以上技术细节分析，提出以下优化建议：

1. **保持1:9的spam ratio**，但增加class imbalance处理策略
2. **实施组内生成vs跨组混合的对比实验**，这个设计很有学术价值
3. **调整超参数**: R=30, N=12000, 在关键区域增加ratio检测点
4. **完善统计检验**: 增加多重比较校正和非参数检验
5. **集成embedding analysis**: 作为性能评估的补充解释维度
6. **分阶段实施**: 先pilot study验证，再全面展开

这个实验设计具有很强的学术严谨性和实践价值，建议按照优化后的方案实施。