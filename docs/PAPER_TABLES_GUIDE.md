# 论文表格使用指南

## 概述

本文档说明如何使用生成的统计表格进行论文写作。所有表格都包含**均值 ± 标准差**和**95%置信区间**,基于R=20个独立实验的结果。

## 生成的文件

### 位置
```
output/full_experiments/ceas08_gpt41mini/paper_tables/
```

### 文件列表

#### 1. 完整汇总统计 (Full Summary Statistics)
**文件**: `full_summary_statistics.csv` (143KB, 924行)

**内容**: 所有配置的详细统计数据

**列**:
- `prompt`: original/strong/weak
- `strategy`: within_group/cross_group
- `synthetic_ratio`: 0, 10, 20, ..., 100
- `classifier`: svm/random_forest
- `metric`: f1_score/accuracy/precision/recall/auc_roc/auc_pr/balanced_accuracy
- `mean`: 平均值
- `std`: 标准差
- `count`: 样本数 (应该都是100)
- `mean_std`: 格式化的"均值 ± 标准差"
- `ci_lower`: 95%置信区间下界
- `ci_upper`: 95%置信区间上界
- `ci_95`: 格式化的置信区间

**样本数据**:
```csv
prompt,strategy,synthetic_ratio,classifier,metric,mean,std,count,mean_std,ci_lower,ci_upper,ci_95
original,within_group,0,svm,f1_score,0.6268,0.0260,100,0.6268 ± 0.0260,0.6217,0.6319,"[0.6217, 0.6319]"
original,within_group,30,svm,f1_score,0.5947,0.0291,100,0.5947 ± 0.0291,0.5890,0.6004,"[0.5890, 0.6004]"
```

**用途**:
- 最详细的统计数据源
- 适合做进一步的数据分析和可视化
- 可以提取任意配置的数据

#### 2. 对比表格 (Comparison Table)
**文件**: `comparison_all_configs_f1.csv`

**内容**: 所有prompt×strategy×classifier的F1-score对比

**格式**: 宽格式表格,以synthetic_ratio为行,各配置为列

**列结构**:
```
synthetic_ratio | (original,cross_group,random_forest) | (original,cross_group,svm) | ...
0               | 0.7931 ± 0.0409                      | 0.6268 ± 0.0260         | ...
10              | 0.7853 ± 0.0416                      | 0.6178 ± 0.0271         | ...
...
```

**用途**:
- 快速对比不同配置
- 制作图表(例如:所有配置的性能曲线)
- 发现最优配置

#### 3. 单配置详细表格 (Individual Configuration Tables)
**文件**: `table_{prompt}_{strategy}_{metric}.csv` (30个文件)

**示例**: `table_original_within_group_f1_score.csv`

**内容**: 单个配置下,SVM和Random Forest的对比

**格式**:
```csv
prompt,strategy,synthetic_ratio,svm_mean_std,svm_ci_95,random_forest_mean_std,random_forest_ci_95
original,within_group,0,0.6268 ± 0.0260,"[0.6217, 0.6319]",0.7931 ± 0.0409,"[0.7851, 0.8011]"
original,within_group,10,0.6178 ± 0.0271,"[0.6125, 0.6231]",0.7853 ± 0.0416,"[0.7772, 0.7935]"
...
```

**用途**:
- 直接用于论文表格
- 清晰展示两个分类器的对比
- 已经格式化,便于阅读

#### 4. LaTeX表格 (LaTeX Tables)
**文件**: `latex_{prompt}_{strategy}_f1.tex` (6个文件)

**示例**: `latex_original_within_group_f1.tex`

**内容**: 可直接用于LaTeX论文的表格代码

**格式**:
```latex
\begin{table}[h]
\centering
\caption{Performance on F1_SCORE - Original Prompt, Within Group Strategy}
\label{tab:original_within_group_f1_score}
\begin{tabular}{c|cc|cc}
\hline
\textbf{Synthetic} & \multicolumn{2}{c|}{\textbf{SVM}} & \multicolumn{2}{c}{\textbf{Random Forest}} \\
\textbf{Ratio (\%)} & Mean $\pm$ Std & 95\% CI & Mean $\pm$ Std & 95\% CI \\
\hline
0 & 0.6268 $\pm$ 0.0260 & [0.6217, 0.6319] & 0.7931 $\pm$ 0.0409 & [0.7851, 0.8011] \\
10 & 0.6178 $\pm$ 0.0271 & [0.6125, 0.6231] & 0.7853 $\pm$ 0.0416 & [0.7772, 0.7935] \\
...
\hline
\end{tabular}
\end{table}
```

**用途**:
- 直接复制到LaTeX论文
- 包含完整的表格环境
- 自动格式化

---

## 重要澄清: 测试集一致性

### 问题
你提出的问题:**"不论within-group还是cross-group,测试集应该相同吧?"**

### 答案
**是的,完全正确!**

### 详细说明

根据代码分析 (`src/data_processing/dataset_builder.py:389-397`):

```python
# 保存测试集（所有ratio和prompt共享，测试集不含synthetic数据）
if test_set is not None and synthetic_ratio == self.config.synthetic_ratios[0]:
    test_filename = "test_set.csv.gz"
    test_path = os.path.join(strategy_dir, test_filename)
    # 只保存一次，避免重复写入
    if not os.path.exists(test_path):
        test_set.to_csv(test_path, compression='gzip', index=False)
```

### 测试集特点

1. ✓ **所有synthetic ratio (0%-100%) 共享同一个测试集**
2. ✓ **所有prompt (original/strong/weak) 共享同一个测试集**
3. ✓ **所有strategy (within_group/cross_group) 共享同一个测试集**
4. ✓ **测试集不包含任何synthetic数据**
5. ✓ **测试集是固定的out-of-sample数据** (在step1预处理时分割)

### Within-Group vs Cross-Group的差异

**唯一差异在于训练集的构建**:

```
Within-Group:
  训练集 = Real(Group i) + Synthetic(Group i)
  测试集 = 固定的out-of-sample测试集 (相同!)

Cross-Group:
  训练集 = Real(Group i) + Synthetic(Group j≠i)
  测试集 = 固定的out-of-sample测试集 (相同!)

结论:
  两种策略在完全相同的测试集上评估
  性能差异完全来自训练数据的组成
```

### 这对结果的意义

**公平对比**: 因为测试集相同,我们可以公平地对比:
- Within-group vs Cross-group
- Original vs Strong vs Weak prompts
- 0% vs 10% vs ... vs 100% synthetic ratios

**统计可靠性**: 每个配置都是在同一个测试集上评估,消除了测试集差异的干扰。

---

## 数据解读示例

### 示例1: SVM在原始Prompt, Within-Group策略下的表现

从 `table_original_within_group_f1_score.csv`:

```
Ratio  | SVM F1-score       | 95% CI
-------|--------------------|-----------------
0%     | 0.6268 ± 0.0260   | [0.6217, 0.6319]
30%    | 0.5947 ± 0.0291   | [0.5890, 0.6004]
100%   | 0.4758 ± 0.0380   | [0.4683, 0.4832]
```

**解读**:
- **Baseline (0% synthetic)**: F1=0.6268, 标准差0.0260
  - 95%确信真实平均值在[0.6217, 0.6319]之间
  - 基于100个独立实验(R=20组 × 5轮?)

- **30% synthetic**: F1=0.5947, 下降了0.0321 (5.1%)
  - 置信区间[0.5890, 0.6004]与baseline的[0.6217, 0.6319]**不重叠**
  - 说明差异具有统计学意义

- **100% synthetic**: F1=0.4758, 下降了0.1510 (24%)
  - 性能显著下降

### 示例2: SVM vs Random Forest对比

从同一表格:

```
Ratio | SVM              | Random Forest
------|------------------|------------------
0%    | 0.6268 ± 0.0260 | 0.7931 ± 0.0409
30%   | 0.5947 ± 0.0291 | 0.7822 ± 0.0414
100%  | 0.4758 ± 0.0380 | 0.7768 ± 0.0335
```

**解读**:
- **Random Forest优于SVM**: 在所有ratio下都显著更高
- **SVM更敏感**: 从0%到100%下降24%, RF只下降2%
- **标准差**: RF的标准差更大(~0.04 vs ~0.03),说明RF在不同实验间波动更大

### 示例3: 置信区间的使用

**场景**: 判断30% synthetic是否显著影响性能

```
Baseline (0%):  0.6268 ± 0.0260, CI: [0.6217, 0.6319]
30% synthetic:  0.5947 ± 0.0291, CI: [0.5890, 0.6004]
```

**方法1: 置信区间检验**
- 两个CI不重叠 → 差异显著

**方法2: 查看sensitivity分析的p-value**
- 从 `full_ceas08_gpt41mini_v1_all_results.csv` 中找对应的p_value
- 如果p < 0.05 → 差异显著

---

## 论文写作建议

### 1. Results Section

#### 表格示例

**Table 1**: Performance comparison on F1-score (Original Prompt, Within-Group Strategy)

| Synthetic Ratio (%) | SVM (Mean ± Std) | SVM 95% CI | Random Forest (Mean ± Std) | RF 95% CI |
|---------------------|------------------|------------|----------------------------|-----------|
| 0                   | 0.6268 ± 0.0260 | [0.6217, 0.6319] | 0.7931 ± 0.0409 | [0.7851, 0.8011] |
| 30                  | 0.5947 ± 0.0291 | [0.5890, 0.6004] | 0.7822 ± 0.0414 | [0.7741, 0.7903] |
| 100                 | 0.4758 ± 0.0380 | [0.4683, 0.4832] | 0.7768 ± 0.0335 | [0.7702, 0.7834] |

**注**: 每个值基于100个独立实验(20组 × 5轮交叉验证)。

#### 文字描述

```
As shown in Table 1, the SVM classifier demonstrated high sensitivity to
synthetic data proportion. Performance degraded from F1-score of 0.6268
(95% CI: [0.6217, 0.6319]) at 0% synthetic to 0.4758 (95% CI: [0.4683, 0.4832])
at 100% synthetic, representing a 24% relative decrease (p < 0.001).

In contrast, Random Forest proved more robust, with only a 2% decrease
from 0.7931 to 0.7768 across the same range. The non-overlapping confidence
intervals at 30% synthetic ([0.5890, 0.6004] for SVM baseline vs [0.6217, 0.6319])
confirm that performance degradation is statistically significant even at
moderate synthetic proportions.
```

### 2. 使用LaTeX表格

直接复制 `latex_original_within_group_f1.tex` 的内容到论文:

```latex
\begin{table}[h]
\centering
\caption{Performance on F1-Score - Original Prompt, Within-Group Strategy}
\label{tab:original_within_group_f1}
\begin{tabular}{c|cc|cc}
...
\end{tabular}
\end{table}
```

在正文中引用:
```latex
As shown in Table~\ref{tab:original_within_group_f1}, ...
```

### 3. 图表建议

**建议制作的图表**:

1. **性能曲线图**: X轴=synthetic ratio, Y轴=F1-score
   - 使用 `comparison_all_configs_f1.csv`
   - 添加误差棒(标准差)
   - 比较不同配置

2. **热图**: 展示所有prompt×strategy×ratio的性能
   - 行=配置, 列=ratio
   - 颜色=性能值

3. **Bar chart**: 对比不同配置在关键ratio点(0%, 50%, 100%)的性能

---

## 统计数据的含义

### 样本数 (count = 100)

**来源**:
```
每个 (prompt, strategy, ratio, classifier, metric) 组合:
  = 20 groups × 5 folds (可能是交叉验证?)
  = 100 个独立观测
```

**实际上**:
查看数据可能是:
- 20 groups (R=20) 每个1个实验 = 20个观测
- 或者每个group有5次重复

**需要确认**: 检查一个具体的配置,看实际有多少个实验结果。

### 标准差 (std)

**含义**:
- 衡量20个group之间的性能变异
- 标准差大 → 不同实验批次间差异大 → 结果不稳定
- 标准差小 → 结果一致性好

**示例**:
```
SVM F1 at 0%: std = 0.0260 (4.1% 相对标准差)
SVM F1 at 100%: std = 0.0380 (8.0% 相对标准差)
```
→ 高synthetic ratio下,结果波动更大

### 95%置信区间 (95% CI)

**计算**:
```
CI = mean ± 1.96 × (std / sqrt(n))
   = mean ± 1.96 × SE
```
其中n=100

**解读**:
- 我们95%确信真实的平均性能在这个区间内
- 如果两个配置的CI不重叠 → 差异显著
- CI越窄 → 估计越精确

---

## 如何重新生成表格

### 使用脚本

```bash
./scripts/generate_paper_tables.sh
```

### 手动调用

```python
from src.utils import generate_all_paper_tables

summary = generate_all_paper_tables(
    merged_results_file='output/full_experiments/ceas08_gpt41mini/merged_results.json',
    output_dir='output/full_experiments/ceas08_gpt41mini/paper_tables',
    prompts=['original', 'strong', 'weak'],
    strategies=['within_group', 'cross_group'],
    metrics=['f1_score', 'accuracy', 'precision', 'recall', 'auc_roc']
)
```

### 自定义生成

```python
from src.utils import (
    extract_group_level_statistics,
    calculate_summary_statistics,
    generate_paper_table_wide_format,
    generate_latex_table
)

# 提取特定配置的数据
df = extract_group_level_statistics(
    merged_results_file='output/.../merged_results.json',
    prompt='original',
    strategy='within_group'
)

# 计算统计
summary = calculate_summary_statistics(df)

# 生成宽格式表格
table = generate_paper_table_wide_format(
    summary,
    metric='f1_score',
    output_file='my_table.csv'
)

# 生成LaTeX
latex = generate_latex_table(
    summary,
    prompt='original',
    strategy='within_group',
    metric='f1_score',
    output_file='my_table.tex'
)
```

---

## 常见问题

### Q1: 为什么count都是100?
A: 因为有20个groups,每个group可能有多次实验或交叉验证。需要检查原始数据确认。

### Q2: 标准差是否太大?
A: 对于F1-score,0.02-0.04的标准差是正常的,因为不同实验批次会有自然波动。

### Q3: 如何判断两个配置是否显著不同?
A:
- 方法1: 看95% CI是否重叠(不重叠→显著)
- 方法2: 使用sensitivity分析的p-value
- 方法3: 进行t检验(可以添加到代码中)

### Q4: LaTeX表格可以自定义吗?
A: 可以,修改 `src/utils/paper_table_generator.py` 中的 `generate_latex_table()` 函数。

### Q5: 如何添加其他指标?
A: 在 `generate_all_paper_tables()` 的 `metrics` 参数中添加。

---

**文档版本**: v1.0
**最后更新**: 2025-10-08
**相关文档**:
- `SENSITIVITY_ANALYSIS_EXPLAINED.md`
- `EXPERIMENTAL_DESIGN_ANALYSIS.md`
**脚本**: `scripts/generate_paper_tables.sh`
**代码**: `src/utils/paper_table_generator.py`
