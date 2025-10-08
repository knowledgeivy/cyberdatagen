# Sensitivity Analysis - 统计方法详解

## 目录
1. [Sensitivity Score 计算方法](#1-sensitivity-score-计算方法)
2. [Hypothesis Test P-value 计算方法](#2-hypothesis-test-p-value-计算方法)
3. [两类P值的区别](#3-两类p值的区别)
4. [CSV数据结构说明](#4-csv数据结构说明)
5. [实际例子解读](#5-实际例子解读)

---

## 1. Sensitivity Score 计算方法

### 1.1 定义
**Sensitivity score** 衡量模型性能对合成数据比例变化的敏感程度。

### 1.2 计算方法
使用**线性回归 (Linear Regression)** 来拟合以下关系:

```
Performance = β₀ + β₁ × Synthetic_Ratio + ε
```

**数据点:**
- **X轴**: synthetic ratio (0%, 10%, 20%, ..., 100%)
- **Y轴**: 对应的模型性能 (f1_score, accuracy, 等)

**实现:**
使用 `scipy.stats.linregress()` 函数,返回:
1. **slope (β₁)**: `regression_slope` - **这就是主要的 sensitivity score**
2. **intercept (β₀)**: `regression_intercept`
3. **r_value**: Pearson相关系数
4. **p_value**: `regression_p_value`
5. **std_err**: 标准误差

### 1.3 核心指标解释

#### regression_slope (斜率)
- **含义**: 每增加 1% 合成数据,性能的平均变化量
- **示例**: `-0.001537` 表示每增加1%合成数据,F1-score下降0.001537
- **解读**:
  - 绝对值越大 → 模型对合成数据越敏感
  - 负值 → 性能下降
  - 正值 → 性能提升

#### regression_r_squared (R²决定系数)
- **计算**: R² = r_value²
- **含义**: 线性模型解释的方差比例
- **范围**: 0 到 1
- **解读**:
  - R² = 0.99 → 99%的性能变化可以用线性关系解释
  - R² 越接近1 → 性能衰减越接近线性
  - R² < 0.7 → 性能衰减可能是非线性的

#### regression_p_value
- **含义**: 检验斜率是否显著不为0
- **假设**:
  - H₀: β₁ = 0 (合成数据比例对性能无影响)
  - H₁: β₁ ≠ 0 (合成数据比例对性能有显著影响)
- **解读**:
  - p < 0.05 → 拒绝H₀,说明合成数据比例对性能有显著影响
- **计算方法**: t-检验, t = slope / std_err

#### total_change (总变化)
- **计算**: Performance(100%) - Performance(0%)
- **含义**: 从0%到100%合成数据的总性能变化
- **示例**: `-0.1510` 表示F1-score总共下降了0.1510

---

## 2. Hypothesis Test P-value 计算方法

### 2.1 定义
检验每个synthetic ratio下的性能是否与baseline (0% synthetic)有**显著差异**。

### 2.2 计算方法
使用 **Paired t-test (配对t检验)**:

**假设:**
- H₀: μ_baseline = μ_ratio (没有显著差异)
- H₁: μ_baseline ≠ μ_ratio (有显著差异)

**步骤:**

1. **准备配对数据**:
   - `baseline_values`: 0% synthetic ratio的所有实验性能 (例如: 100个实验)
   - `ratio_values`: 某个ratio (如30%)的所有实验性能 (例如: 100个实验)

2. **计算t统计量**:
   ```
   t = (mean(baseline) - mean(ratio)) / SE(差值)
   其中 SE = std(差值) / sqrt(n)
   ```

3. **使用 scipy.stats.ttest_rel() 计算p值**

4. **多重比较校正 (FDR-BH)**:
   - 因为我们做了多次比较 (10个ratio × 2个分类器 × 7个指标 ≈ 140次)
   - 使用 **Benjamini-Hochberg FDR** 方法校正p值
   - 校正后的p值: `t_p_value_corrected`
   - 这防止了假阳性 (Type I error) 的累积

### 2.3 辅助检验

#### Wilcoxon Signed-Rank Test (非参数检验)
- 不假设数据服从正态分布
- 对异常值更稳健
- 使用 `scipy.stats.wilcoxon()`

#### Effect Size (效应量) - Cohen's d
**计算**:
```
d = (mean_baseline - mean_ratio) / pooled_std
```

**解释**:
- |d| < 0.2: 小效应
- |d| ≥ 0.2: 中等效应
- |d| ≥ 0.5: 大效应
- |d| ≥ 0.8: 非常大效应

---

## 3. 两类P值的区别

CSV中有两个p值,它们检验的是**不同的问题**:

### 3.1 regression_p_value (回归p值)
- **检验对象**: 线性趋势(斜率)是否存在
- **问题**: "性能是否随合成数据比例线性变化?"
- **方法**: 线性回归的t检验
- **用途**: 验证sensitivity分析的有效性
- **数据**: 使用所有11个ratio点 (0%, 10%, ..., 100%)

### 3.2 p_value (t_p_value_corrected)
- **检验对象**: 每个ratio与baseline的差异
- **问题**: "30% synthetic下的性能是否显著低于0% synthetic?"
- **方法**: 配对t检验 + FDR校正
- **用途**: 确定哪些ratio下性能显著下降
- **数据**: 使用该ratio的所有实验样本 (例如: 100个)

### 3.3 关键区别示意

```
Regression P-value:
  看整体趋势       Y (Performance)
                    |
                    |   •  <-- 这些点是否呈线性?
                    |  •
                    | •
                    |•__________ X (Ratio)

Hypothesis Test P-value:
  看单点差异       Distribution
                    |  Baseline
                    |  /‾‾‾\        30% ratio
                    | |     |       /‾‾‾\
                    | |     |      |     |
                    |_|_____|______|_____|___ Performance
                       ↑           ↑
                    这两组有显著差异吗?
```

---

## 4. CSV数据结构说明

### 4.1 行结构
每个 `classifier × metric` 组合有 **12 行**:

#### 第1行: synthetic_ratio = "sensitivity"
这是**汇总行**,包含回归分析结果:
- `performance` = baseline (0% synthetic时的性能)
- `regression_slope` = 斜率 (主要的sensitivity指标)
- `regression_r_squared` = R²
- `regression_p_value` = 回归p值
- `total_change` = 总变化
- 其他列为空

#### 第2-12行: synthetic_ratio = 10, 20, ..., 100
每行对应一个ratio,包含该ratio的详细信息:
- `performance` = 该ratio下的平均性能
- `absolute_degradation` = baseline - performance
- `relative_degradation` = (baseline - performance) / baseline
- `p_value` = 该ratio与baseline比较的校正后p值
- regression相关列为空

### 4.2 列说明

| 列名 | 含义 | 出现位置 |
|------|------|----------|
| `prompt` | Prompt类型 (original/strong/weak) | 所有行 |
| `strategy` | 采样策略 (within_group/cross_group) | 所有行 |
| `classifier` | 分类器 (svm/random_forest) | 所有行 |
| `metric` | 评估指标 (f1_score/accuracy/等) | 所有行 |
| `synthetic_ratio` | 合成数据比例或"sensitivity" | 所有行 |
| `performance` | 性能值 | 所有行 |
| `regression_slope` | 回归斜率 | sensitivity行 |
| `regression_r_squared` | R²决定系数 | sensitivity行 |
| `regression_p_value` | 回归p值 | sensitivity行 |
| `total_change` | 总变化量 | sensitivity行 |
| `absolute_degradation` | 绝对性能下降 | ratio行 |
| `relative_degradation` | 相对性能下降 | ratio行 |
| `p_value` | 假设检验p值(FDR校正) | ratio行 |

---

## 5. 实际例子解读

### 5.1 Sensitivity行示例

```csv
original,within_group,svm,f1_score,sensitivity,0.6268,-0.001537,0.9898,2.8e-10,-0.1510
```

**解读:**
- ✓ **Baseline F1 = 0.6268** (0% synthetic时的性能)
- ✓ **每1% synthetic → F1下降0.001537**
- ✓ **R² = 0.9898** → 98.98%方差可用线性解释 (高度线性!)
- ✓ **p = 2.8e-10** → 线性趋势高度显著
- ✓ **0% → 100%总下降 0.1510** (约24%相对下降)

**结论**: SVM在within_group策略下对合成数据非常敏感,性能呈高度线性下降。

### 5.2 Ratio行示例

```csv
original,within_group,svm,f1_score,30,0.5947,,,,0.0321,0.0512,1.96e-26
```

**解读:**
- ✓ **30% synthetic时F1 = 0.5947**
- ✓ **绝对下降 = 0.0321** (从0.6268降到0.5947)
- ✓ **相对下降 = 5.12%** (下降了约5%的性能)
- ✓ **p = 1.96e-26** → 与baseline差异极其显著

**结论**: 当使用30%合成数据时,性能显著下降,差异有统计学意义。

### 5.3 对比不同R²值

#### 高R² (线性衰减)
```csv
original,within_group,svm,f1_score,sensitivity,0.6268,-0.001537,0.9898,2.8e-10,-0.1510
```
- R² = 0.9898 → 性能衰减非常线性
- 可以用斜率准确预测任何ratio下的性能

#### 低R² (非线性衰减)
```csv
original,within_group,random_forest,f1_score,sensitivity,0.7931,-0.000133,0.7231,0.000911,-0.0163
```
- R² = 0.7231 → 性能衰减不太线性
- 可能存在阈值效应或非线性模式
- 需要进一步分析性能曲线的形状

---

## 6. 代码实现参考

### 6.1 Sensitivity计算 (src/analysis/statistical_analysis.py:431-476)

```python
def _calculate_sensitivity(self, baseline: float, degradation_curve: List[Dict]) -> Dict[str, float]:
    # 准备数据
    ratios = [0] + [point['ratio'] for point in degradation_curve]
    performances = [baseline] + [point['performance'] for point in degradation_curve]

    # 线性回归
    slope, intercept, r_value, p_value, std_err = stats.linregress(ratios, performances)

    # 计算总变化
    max_ratio = max(ratios)
    total_change = performances[-1] - baseline

    return {
        'regression_slope': float(slope),
        'regression_r_squared': float(r_value ** 2),
        'regression_p_value': float(p_value),
        'total_change_0_to_max': float(total_change),
        ...
    }
```

### 6.2 Hypothesis Test计算 (src/analysis/statistical_analysis.py:177-263)

```python
def perform_hypothesis_tests(self, organized_results: Dict) -> Dict:
    # 获取baseline数据
    baseline_data = organized_results[0]

    for ratio, ratio_data in organized_results.items():
        if ratio == 0:
            continue

        # 配对t检验
        baseline_values = np.array(baseline_data[classifier][metric])
        ratio_values = np.array(ratio_data[classifier][metric])

        t_stat, t_p_value = ttest_rel(baseline_values, ratio_values)

        # 计算Cohen's d
        cohens_d = (np.mean(baseline_values) - np.mean(ratio_values)) / pooled_std

    # FDR校正
    corrected_tests = self._correct_multiple_comparisons(hypothesis_tests)
    return corrected_tests
```

---

## 7. 关键要点总结

1. **Sensitivity Score (regression_slope)**:
   - 衡量性能对合成数据的敏感度
   - 斜率绝对值越大 → 越敏感
   - 用于整体趋势分析

2. **R² (regression_r_squared)**:
   - 衡量线性拟合的好坏
   - 接近1 → 性能线性下降
   - 远离1 → 性能可能非线性下降

3. **Regression P-value**:
   - 检验整体趋势是否存在
   - p < 0.05 → 有显著的线性趋势

4. **Hypothesis Test P-value**:
   - 检验每个ratio与baseline的差异
   - p < 0.05 → 该ratio下性能显著不同
   - 使用FDR校正防止假阳性

5. **数据量的重要性**:
   - 每个ratio有多个实验样本 (例如100个)
   - 样本量越大,统计检验越可靠
   - 配对设计减少了个体差异的影响

---

**文档版本**: v1.0
**最后更新**: 2025-10-08
**相关文件**:
- 实现: `src/analysis/statistical_analysis.py`
- CSV导出: `src/utils/csv_exporter.py`
- 结果示例: `output/full_experiments/ceas08_gpt41mini/reports/csv/`
