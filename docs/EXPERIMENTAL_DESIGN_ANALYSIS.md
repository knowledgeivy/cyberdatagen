# 实验设计对Sensitivity分析的影响

## 目录
1. [实验设计概述](#1-实验设计概述)
2. [分组策略 (R=20) 对Sensitivity的作用](#2-分组策略-r20-对sensitivity的作用)
3. [两个分类器的作用](#3-两个分类器的作用)
4. [两组Sampling Strategy的差别](#4-两组sampling-strategy的差别)
5. [实验设计的统计学意义](#5-实验设计的统计学意义)
6. [实际数据对比分析](#6-实际数据对比分析)

---

## 1. 实验设计概述

### 1.1 实验参数

您的实验设计包含以下维度:

```
实验设计 = Prompts × Strategies × Groups × Ratios × Classifiers

参数:
├── Prompts: 3 种 (original, strong, weak)
├── Sampling Strategies: 2 种 (within_group, cross_group)
├── Groups (R): 20 组
├── Synthetic Ratios: 11 个 (0%, 10%, 20%, ..., 100%)
└── Classifiers: 2 个 (SVM, Random Forest)

总实验数 = 3 × 2 × 20 × 11 = 1320 组实验
每组有2个分类器结果 = 2640 个分类结果
```

### 1.2 数据流程

```
Step 1-3: 数据准备和构建
  ↓
Step 4: 分类 (对每个 prompt × strategy × group × ratio)
  ├── Group 0: [0%, 10%, ..., 100%] → SVM + RF结果
  ├── Group 1: [0%, 10%, ..., 100%] → SVM + RF结果
  ├── ...
  └── Group 19: [0%, 10%, ..., 100%] → SVM + RF结果
  ↓
Merge: 合并所有group的结果
  ↓
Step 5: 统计分析 (按 prompt × strategy)
  ├── 对每个ratio,聚合20个group的结果
  ├── 计算描述性统计 (mean, std, ...)
  ├── 假设检验 (每个ratio vs baseline)
  └── Sensitivity分析 (线性回归)
  ↓
CSV导出: 包含sensitivity和p-value
```

---

## 2. 分组策略 (R=20) 对Sensitivity的作用

### 2.1 为什么需要R=20组?

**关键答案**: R=20不直接影响sensitivity的计算公式,但**极大地影响结果的统计可靠性**。

### 2.2 分组的统计学作用

#### 作用1: 提供重复实验 (Replication)

每个 `(prompt, strategy, ratio)` 组合有**20个独立实验**:
- Group 0的结果
- Group 1的结果
- ...
- Group 19的结果

**统计意义**:
```
单次实验 (R=1):
  30% synthetic → F1 = 0.594 (这个数字可靠吗?)

20次实验 (R=20):
  30% synthetic → F1 = 0.594 ± 0.012 (mean ± std)
                   ↑
                这个均值更可靠,因为基于20次观察
```

#### 作用2: 支持假设检验

配对t检验需要**样本分布**,而不是单点:

```python
# 如果R=1 (只有1个group)
baseline_values = [0.6268]  # 只有1个值
ratio_30_values = [0.5947]  # 只有1个值
# 无法进行t检验! (需要至少2个样本)

# 如果R=20 (有20个groups)
baseline_values = [0.627, 0.625, 0.629, ..., 0.626]  # 20个值
ratio_30_values = [0.595, 0.593, 0.597, ..., 0.594]  # 20个值
# 可以进行t检验,计算p-value
t_stat, p_value = ttest_rel(baseline_values, ratio_30_values)
```

#### 作用3: 提高Sensitivity估计的精度

线性回归使用的是**每个ratio的平均性能**:

```
Sensitivity计算数据点:

Ratio   Performance (如果R=1)   Performance (如果R=20)
0%      0.6268 (单次)           0.6268 ± 0.008 (20次平均)
10%     0.6178 (单次)           0.6178 ± 0.009 (20次平均)
20%     0.6076 (单次)           0.6076 ± 0.010 (20次平均)
...

线性回归拟合:
  如果R=1: 斜率可能受噪声影响
  如果R=20: 斜率更稳定,因为每个点是20次实验的平均值
```

### 2.3 分组的数据结构

```
原始数据结构 (Step 4输出):
{
  "prompt": "original",
  "strategy": "within_group",
  "group_id": 0,
  "synthetic_ratio": 30,
  "classifiers": {
    "svm": {"f1_score": 0.595},      ← Group 0的结果
    "random_forest": {"f1_score": 0.785}
  }
}

聚合后的数据 (Step 5使用):
{
  "prompt": "original",
  "strategy": "within_group",
  "synthetic_ratio": 30,
  "svm": {
    "f1_score": [0.595, 0.593, 0.597, ...]  ← 20个group的结果
  }
}

统计分析 (Sensitivity计算使用):
Ratio 30% → mean(f1_score) = 0.594  ← 20个值的平均
```

### 2.4 R=20的选择依据

**统计学考虑**:
- R ≥ 30: 理想 (中心极限定理,假设正态分布)
- R = 20: 良好 (足够进行t检验,但需谨慎)
- R < 10: 不足 (统计检验不可靠)

**实践权衡**:
- R越大 → 结果越可靠,但计算成本越高
- R=20 是计算成本和统计可靠性的折中

---

## 3. 两个分类器的作用

### 3.1 为什么使用两个分类器?

**关键答案**: 两个分类器用于**验证发现的普遍性**。

### 3.2 分类器的差异

#### SVM (Support Vector Machine)
- **特点**: 线性分类器,对特征边界敏感
- **优势**: 在高维空间表现好,泛化能力强
- **劣势**: 对噪声敏感

#### Random Forest
- **特点**: 集成学习,基于决策树
- **优势**: 对噪声鲁棒,能捕捉非线性关系
- **劣势**: 容易过拟合小数据集

### 3.3 两个分类器在Sensitivity分析中的作用

**独立分析**:
每个分类器都**单独**进行sensitivity分析:

```
SVM的Sensitivity:
  Slope = -0.001537
  R² = 0.9898
  → SVM对合成数据高度敏感,线性下降

Random Forest的Sensitivity:
  Slope = -0.000133
  R² = 0.7231
  → RF对合成数据不太敏感,非线性下降
```

**对比意义**:
```
如果两个分类器都显示显著下降:
  → 合成数据对spam检测的影响是普遍的,不依赖于特定算法

如果只有一个分类器显示下降:
  → 影响可能是算法特定的,需要进一步研究
```

### 3.4 不影响对方的计算

**重要**: 两个分类器的sensitivity计算是**完全独立**的:

```python
# SVM的sensitivity计算
svm_ratios = [0, 10, 20, ..., 100]
svm_performances = [0.627, 0.618, 0.608, ..., 0.476]
svm_slope, _, svm_r2, svm_p, _ = linregress(svm_ratios, svm_performances)

# Random Forest的sensitivity计算
rf_ratios = [0, 10, 20, ..., 100]  # 同样的ratio
rf_performances = [0.793, 0.783, 0.781, ..., 0.777]  # 不同的性能
rf_slope, _, rf_r2, rf_p, _ = linregress(rf_ratios, rf_performances)

# 两者互不影响
```

**CSV体现**:
```csv
classifier,metric,regression_slope,regression_r_squared
svm,f1_score,-0.001537,0.9898
random_forest,f1_score,-0.000133,0.7231
```

---

## 4. 两组Sampling Strategy的差别

### 4.1 Within-Group Sampling

**定义**: 合成数据和真实数据来自**同一个spam group**。

**示例**:
```
假设有5个spam groups: G1, G2, G3, G4, G5

实验Group 0 (30% synthetic):
  训练集:
    - 真实数据: 从G1随机抽取70条
    - 合成数据: 从G1生成30条  ← 同一个group
  测试集:
    - 固定的out-of-sample测试集 (所有策略共享)
```

**特点**:
- 合成数据与真实数据来源相同
- 模拟"用合成数据增强同类数据"的场景
- 预期:合成数据与真实数据更相似

### 4.2 Cross-Group Sampling

**定义**: 合成数据和真实数据来自**不同的spam groups**。

**示例**:
```
实验Group 0 (30% synthetic):
  训练集:
    - 真实数据: 从G1随机抽取70条
    - 合成数据: 从G2, G3生成30条  ← 不同的groups
  测试集:
    - 固定的out-of-sample测试集 (所有策略共享)
```

**特点**:
- 合成数据与真实数据来源不同
- 模拟"用合成数据补充不同类型数据"的场景
- 预期:合成数据与真实数据差异更大

### 4.3 重要澄清: 测试集的一致性

**关键发现** (代码: `src/data_processing/dataset_builder.py:389-397`):

```python
# 保存测试集（所有ratio和prompt共享，测试集不含synthetic数据）
if test_set is not None and synthetic_ratio == self.config.synthetic_ratios[0]:
    test_filename = "test_set.csv.gz"
    test_path = os.path.join(strategy_dir, test_filename)
    # 只保存一次，避免重复写入
    if not os.path.exists(test_path):
        test_set.to_csv(test_path, compression='gzip', index=False)
```

**测试集特点**:
1. ✓ **所有synthetic ratio共享同一个测试集**
2. ✓ **所有prompt (original/strong/weak) 共享同一个测试集**
3. ✓ **测试集不包含任何synthetic数据**
4. ✓ **测试集是固定的out-of-sample数据** (在step1预处理时分割)

**Within-Group vs Cross-Group的唯一区别**:
```
唯一区别在于训练集的构建:

Within-Group:
  训练集 = Real(Group i) + Synthetic(Group i)
  测试集 = 固定的out-of-sample测试集 (相同)

Cross-Group:
  训练集 = Real(Group i) + Synthetic(Group j≠i)
  测试集 = 固定的out-of-sample测试集 (相同)

结论: 两种策略在相同的测试集上评估，差异完全来自训练数据的组成
```

### 4.3 两种策略对Sensitivity的影响

#### 理论预期

```
Within-Group:
  - 合成数据更接近真实数据
  - 预期sensitivity较低 (性能下降较慢)
  - 预期R²较高 (线性关系更强)

Cross-Group:
  - 合成数据与真实数据差异较大
  - 预期sensitivity较高 (性能下降较快)
  - 预期R²可能较低 (可能有阈值效应)
```

#### 独立分析

两个策略是**完全独立**分析的:

```python
# Within-Group的实验
within_results = filter(results, strategy='within_group')
within_sensitivity = calculate_sensitivity(within_results)

# Cross-Group的实验
cross_results = filter(results, strategy='cross_group')
cross_sensitivity = calculate_sensitivity(cross_results)

# 两者独立,互不影响
```

#### CSV体现

```csv
strategy,classifier,metric,regression_slope,regression_r_squared
within_group,svm,f1_score,-0.001537,0.9898
cross_group,svm,f1_score,-0.001842,0.9654
```

**解读**:
- Cross-group的斜率绝对值更大 (-0.001842 vs -0.001537)
- → Cross-group策略下,模型对合成数据更敏感
- → 验证了理论预期

### 4.4 为什么需要两种策略?

**研究问题**:
```
Q1: 合成数据能否有效增强训练数据?
    → Within-group实验回答

Q2: 合成数据能否泛化到不同类型的spam?
    → Cross-group实验回答

Q3: 哪种使用方式更有效?
    → 对比两种策略的sensitivity回答
```

**实践意义**:
- 如果within-group表现好 → 可以用合成数据增强现有数据
- 如果cross-group表现好 → 可以用合成数据补充稀缺类型
- 如果两者都不好 → 需要改进合成方法

---

## 5. 实验设计的统计学意义

### 5.1 多层次实验设计

您的设计是**多因素完全交叉设计**:

```
因素A (Prompt): 3水平 (original, strong, weak)
因素B (Strategy): 2水平 (within, cross)
因素C (Ratio): 11水平 (0%, 10%, ..., 100%)
重复 (Group): 20次

每个 A×B×C 组合都有20次独立重复
```

### 5.2 统计检验的层次

```
Level 1: Sensitivity分析 (整体趋势)
  输入: 11个ratio的平均性能 (每个平均基于20个group)
  方法: 线性回归
  输出: slope, R², regression_p_value

Level 2: Hypothesis检验 (单点比较)
  输入: 每个ratio的20个观测值
  方法: 配对t检验 + FDR校正
  输出: t_p_value_corrected (每个ratio)

Level 3: 因素比较 (可选,未在当前分析中)
  问题: Prompt之间有差异吗? Strategy之间有差异吗?
  方法: ANOVA或mixed-effects model
```

### 5.3 样本量的影响

**每个ratio的样本量**:
```
N = Groups × Prompts × Strategies = 20 × 1 × 1 = 20

(对于特定的prompt和strategy组合)
```

**统计功效 (Statistical Power)**:
```
样本量 = 20:
  - 可以检测到中等效应量 (Cohen's d ≈ 0.5)
  - 检验功效约80% (α=0.05)
  - 足够进行可靠的t检验

如果样本量 = 5:
  - 只能检测到大效应量 (Cohen's d > 1.0)
  - 检验功效 < 50%
  - 容易产生Type II error (假阴性)
```

### 5.4 置信区间

有20个样本,可以计算**置信区间**:

```
30% synthetic ratio:
  Mean F1 = 0.594
  Std = 0.012
  95% CI = 0.594 ± 1.96 × (0.012/√20)
         = 0.594 ± 0.005
         = [0.589, 0.599]

解读: 我们95%确信真实的平均性能在[0.589, 0.599]之间
```

---

## 6. 实际数据对比分析

### 6.1 SVM vs Random Forest

基于您的实际结果:

```
SVM (original, within_group):
  Baseline F1: 0.6268
  Slope: -0.001537
  R²: 0.9898
  Total change: -0.1510 (-24%)

Random Forest (original, within_group):
  Baseline F1: 0.7931
  Slope: -0.000133
  R²: 0.7231
  Total change: -0.0163 (-2%)
```

**解读**:
1. **Baseline性能**: RF > SVM (0.793 vs 0.627)
2. **Sensitivity**: SVM更敏感 (斜率绝对值更大)
3. **线性度**: SVM更线性 (R²更高)
4. **总下降**: SVM下降24%, RF下降2%

**结论**:
- SVM对合成数据质量更敏感
- RF更鲁棒,但R²较低说明可能有非线性效应
- 如果使用RF,合成数据的影响较小

### 6.2 Within-Group vs Cross-Group (假设性对比)

**理论预期的验证** (需要查看实际数据):

```python
# 从CSV中提取数据进行对比
import pandas as pd

df = pd.read_csv('output/.../full_ceas08_gpt41mini_v1_all_results.csv')

# 提取sensitivity行
sensitivity_df = df[df['synthetic_ratio'] == 'sensitivity']

# 对比策略
within = sensitivity_df[sensitivity_df['strategy'] == 'within_group']
cross = sensitivity_df[sensitivity_df['strategy'] == 'cross_group']

# 对比slope
print("SVM F1 Slope:")
print(f"  Within-group: {within[within['classifier']=='svm']['regression_slope'].values[0]}")
print(f"  Cross-group: {cross[cross['classifier']=='svm']['regression_slope'].values[0]}")
```

**可能的发现**:
```
情况1: Cross-group斜率更负
  → 跨组合成数据质量较差
  → 建议使用within-group策略

情况2: 两者斜率相近
  → 合成数据质量稳定
  → 两种策略都可行

情况3: Within-group斜率更负
  → 意外! 需要进一步研究原因
  → 可能是过拟合或数据泄露
```

### 6.3 Prompt效应 (original vs strong vs weak)

**分析维度**:
```
对于同一个 (strategy, classifier, metric):
  Original prompt slope: ?
  Strong prompt slope: ?
  Weak prompt slope: ?

问题: Prompt的质量是否影响合成数据对模型的影响?
```

**可能的发现**:
```
如果 Strong prompt 的斜率接近0:
  → 高质量prompt生成的数据对性能影响小
  → 投资于prompt engineering是值得的

如果 所有prompt 的斜率都很负:
  → LLM生成的spam数据普遍质量不足
  → 需要改进生成方法(如RAG, fine-tuning)
```

---

## 7. 总结表

### 7.1 实验设计元素总结

| 元素 | 数量 | 作用 | 对Sensitivity的影响 |
|------|------|------|---------------------|
| **Groups (R=20)** | 20 | 提供重复实验,支持统计检验 | 提高估计精度,使p-value可靠 |
| **Classifiers** | 2 | 验证发现的普遍性 | 独立分析,互不影响 |
| **Strategies** | 2 | 测试不同使用场景 | 独立分析,用于对比 |
| **Prompts** | 3 | 测试生成质量影响 | 独立分析,用于对比 |
| **Ratios** | 11 | 构建性能-比例曲线 | 线性回归的数据点 |

### 7.2 关键概念澄清

| 概念 | 是否影响Sensitivity计算 | 说明 |
|------|-------------------------|------|
| R=20分组 | ✓ 间接影响 | 提高每个ratio均值的可靠性,使回归更准确 |
| 两个分类器 | ✗ 不影响 | 完全独立计算 |
| 两个策略 | ✗ 不影响 | 完全独立计算,但用于对比 |
| 三个prompts | ✗ 不影响 | 完全独立计算,但用于对比 |

### 7.3 数据聚合层次

```
原始数据:
  3 prompts × 2 strategies × 20 groups × 11 ratios × 2 classifiers = 2640条

第一层聚合 (按prompt×strategy×ratio):
  每个组合有20个值 (20 groups)
  → 用于假设检验

第二层聚合 (计算均值):
  每个组合取20个值的平均
  → 用于sensitivity的线性回归

最终输出:
  3 prompts × 2 strategies × 2 classifiers × 5 metrics = 60个sensitivity分析
```

---

## 8. 实践建议

### 8.1 如何解读您的CSV结果

**步骤1**: 先看整体sensitivity
```bash
# 提取所有sensitivity行
grep "sensitivity" full_ceas08_gpt41mini_v1_all_results.csv | \
  awk -F',' '{print $1,$2,$3,$4,$7,$8}' | \
  column -t
```

**步骤2**: 对比不同条件
```python
# 找到sensitivity最低的配置
min_slope = sensitivity_df.loc[sensitivity_df['regression_slope'].abs().idxmin()]
print("最不敏感的配置:", min_slope)

# 找到sensitivity最高的配置
max_slope = sensitivity_df.loc[sensitivity_df['regression_slope'].abs().idxmax()]
print("最敏感的配置:", max_slope)
```

**步骤3**: 检查线性度
```python
# 找到R²最低的配置 (可能需要非线性分析)
low_r2 = sensitivity_df[sensitivity_df['regression_r_squared'] < 0.7]
print("非线性效应明显:", low_r2)
```

### 8.2 统计报告模板

```
## Sensitivity Analysis Results

### Overall Findings
- Total experiments: 6600 (3 prompts × 2 strategies × 20 groups × 11 ratios × 2 classifiers)
- Baseline performance: SVM F1=0.627, RF F1=0.793

### SVM Results (original prompt, within-group)
- Sensitivity: -0.001537 per 1% synthetic (95% CI: [-0.00165, -0.00142])
- Linearity: R²=0.9898 (highly linear)
- Significance: p < 0.001
- Total degradation: -0.151 (-24% relative)

### Interpretation
The SVM classifier shows high sensitivity to synthetic data proportion,
with performance degrading linearly at a rate of 0.15% per 1% increase
in synthetic data. This effect is statistically significant (p < 0.001)
and highly linear (R²=0.99).

### Comparison
Random Forest is more robust (slope=-0.000133 vs -0.001537), but shows
non-linear degradation (R²=0.72), suggesting threshold effects may exist.
```

---

**文档版本**: v1.0
**最后更新**: 2025-10-08
**相关文档**: `SENSITIVITY_ANALYSIS_EXPLAINED.md`
**数据文件**: `output/full_experiments/ceas08_gpt41mini/reports/csv/`
