# 批次5实验设计方案 - 纯合成数据有效性验证
*基于批次4问题修正的科学实验重设计*

## 🎯 实验背景与动机

### 批次4发现的关键问题
在批次4的Phase 3数据集构建中(`batch4_phase3_dataset_construction.py:111`)，存在**数据混合污染**问题：
```python
# 问题代码：真实数据和合成数据混合训练
malicious_pool = pd.concat([real_malicious, synthetic_malicious], ignore_index=True)
```

**科学问题**：这种混合训练无法准确评估LLM合成数据的独立有效性，实验结论存在偏差。

### 批次5核心改进目标
1. **数据独立性**：严格分离真实数据和合成数据的训练过程
2. **可视化增强**：统一embedding空间的综合可视化分析  
3. **对比科学性**：建立Baseline vs Pure Synthetic的严格对比实验

---

## 📊 实验设计框架

### 核心研究问题
1. LLM合成数据能否**独立**达到与真实数据相当的检测性能？
2. 不同prompt策略(rewrite/strong/weak)的合成数据质量差异如何？
3. 不同分层(core/inner/outer/edge)合成数据在embedding空间的分布特征如何？
4. 在不同malicious ratio下，纯合成数据的性能表现规律如何？

### 实验假设
- **H1**: 纯合成数据的检测性能显著低于真实数据baseline
- **H2**: rewrite_strong > rewrite > rewrite_weak 的性能排序
- **H3**: core层合成数据性能最优，edge层性能最差
- **H4**: 低malicious ratio下，合成数据性能差距更明显

---

## 🏗️ 数据资源复用分析

### 可直接复用的批次4资源

#### ✅ 完全复用数据
```
data/batch4_fresh/
├── raw_train_set.csv           # 原始训练集(104万样本)
├── raw_test_set.csv            # 固定测试集(26万样本) 
├── train_malicious.csv         # 训练集恶意样本(52万)
├── train_benign.csv           # 训练集良性样本(52万)
└── synthetic/                  # 1200个合成样本
    ├── core_synthetic.csv      # 300个core层合成样本
    ├── inner_synthetic.csv     # 300个inner层合成样本  
    ├── outer_synthetic.csv     # 300个outer层合成样本
    └── edge_synthetic.csv      # 300个edge层合成样本
```

#### ✅ 部分复用代码模块
```
src/cyberdata/scripts/
├── batch4_phase0_data_preprocessing.py          # ✅ 数据预处理逻辑
├── batch4_phase1_embedding_stratification_clean.py  # ✅ embedding分层算法
├── batch4_phase2_synthetic_generation.py       # ✅ 合成数据生成逻辑
├── batch4_phase4_embedding_visualization.py    # 🔄 需要修改可视化逻辑
└── batch4_phase5_model_training.py            # 🔄 需要重新设计训练逻辑
```

### ❌ 需要重新设计的部分
- **Phase 3**: 数据集构建逻辑(避免数据混合)
- **Phase 2**: 统一embedding空间可视化
- **Phase 4**: 纯合成数据独立训练评估

---

## 🔬 批次5实验流程重设计

### Phase 0: 数据基础准备 ✅ (复用批次4)
```
输入: 批次4已生成的完整数据
操作: 验证数据完整性和ID追踪体系
输出: 确认可用的52万真实恶意样本 + 1200个分层合成样本
```

### Phase 1: 统一Embedding空间构建 🆕
```
目标: 为所有数据建立统一的embedding表示空间

输入数据:
- 真实恶意样本 (52万，用于采样)
- 1200个分层合成样本 (core/inner/outer/edge × 300)
- 固定测试集 (26万样本)
- 真实良性样本 (52万，用于构建训练集)

操作:
1. 对所有数据生成384维embedding向量
2. 保持ID追踪体系完整性
3. 计算分层距离和质心信息
4. 为后续可视化准备数据结构

输出:
- data/batch5/unified_embeddings/
  ├── all_embeddings.npy           # 统一embedding矩阵
  ├── embedding_metadata.csv      # ID和标签对应关系
  ├── layer_centroids.json        # 各层质心信息
  └── distance_statistics.json    # 距离分布统计
```

### Phase 2: 增强可视化分析 🆕
```
目标: 全面可视化不同数据源在embedding空间的分布

分析维度:
1. 数据源对比: Real vs Synthetic (按层)
2. Prompt策略对比: rewrite vs rewrite_strong vs rewrite_weak  
3. 分层效果对比: core vs inner vs outer vs edge
4. 聚类特征分析: 自动识别cluster数量和边界

可视化类型:
1. 2D静态图: PCA/t-SNE降维可视化
2. 3D交互图: 支持旋转和缩放的立体展示
3. 聚类分析图: DBSCAN/K-means聚类结果
4. 距离分布图: 各层到质心的距离分布对比

输出:
- data/batch5/visualizations/
  ├── static_plots/              # 静态图表
  ├── interactive_plots/         # 交互式图表
  ├── cluster_analysis/          # 聚类分析结果
  └── distribution_analysis/     # 分布统计图表
```

### Phase 3: 纯净数据集重构 🆕
```
目标: 构建严格分离的纯净训练数据集

数据集配置:
1. baseline_real: 100%真实恶意数据 + 固定良性数据
2. pure_rewrite: 100%基础重写合成数据 + 固定良性数据  
3. pure_strong: 100%强化合成数据 + 固定良性数据
4. pure_weak: 100%弱化合成数据 + 固定良性数据

比例配置: 5%, 10%, 15%, 20%恶意数据比例
数据集大小: 每个10,000样本 (与批次4保持一致)

关键改进:
- ❌ 绝不混合真实和合成数据
- ✅ 每个配置使用单一数据源
- ✅ 保持良性数据池固定不变
- ✅ 使用相同的独立测试集

数据充足性验证:
- 最大需求: 20%比例需要2000个恶意样本
- 可用合成样本: 
  - rewrite: 400个 (core100+inner100+outer100+edge100)
  - rewrite_strong: 400个
  - rewrite_weak: 400个
- 结论: 合成数据充足，需要采样策略平衡各层

输出:
- data/batch5/pure_datasets/
  ├── baseline_real_5pct/        # 纯真实数据基线
  ├── baseline_real_10pct/
  ├── baseline_real_15pct/
  ├── baseline_real_20pct/
  ├── pure_rewrite_5pct/         # 纯基础重写合成数据
  ├── pure_rewrite_10pct/
  ├── pure_rewrite_15pct/
  ├── pure_rewrite_20pct/
  ├── pure_strong_5pct/          # 纯强化合成数据  
  ├── pure_strong_10pct/
  ├── pure_strong_15pct/
  ├── pure_strong_20pct/
  ├── pure_weak_5pct/            # 纯弱化合成数据
  ├── pure_weak_10pct/
  ├── pure_weak_15pct/
  └── pure_weak_20pct/
  
总计: 16个纯净配置 (4组 × 4比例)
```

### 可选扩展: 分层对比实验 (如数据充足)
```
目标: 细粒度分析不同分层合成数据的性能

额外配置 (如果数据充足):
- pure_core_5pct/10pct/15pct/20pct
- pure_inner_5pct/10pct/15pct/20pct  
- pure_outer_5pct/10pct/15pct/20pct
- pure_edge_5pct/10pct/15pct/20pct

注意: 每层仅300个样本，最高支持15%比例(1500样本)
```

### Phase 4: 模型训练与评估 🆕
```
目标: 独立评估不同纯净数据源的检测性能

训练配置:
- 模型: RandomForest + SVM (与批次4保持一致)
- 评估指标: Accuracy, Precision, Recall, F1-Score
- 测试集: 固定的独立测试集(与批次4相同)

实验矩阵:
- 16个数据集配置 × 2个模型 = 32个实验点
- 如果包含分层实验: +64个实验点

性能对比分析:
1. Baseline Real vs Pure Synthetic 主要对比
2. 不同Prompt策略性能排序
3. 不同Malicious Ratio下的性能曲线
4. 模型敏感性分析 (RF vs SVM)

输出:
- data/batch5/results/
  ├── performance_matrix.csv     # 32×指标的性能矩阵
  ├── comparison_curves.png      # 4条性能对比曲线
  ├── statistical_tests.json    # 统计显著性检验
  └── comprehensive_report.md    # 综合分析报告
```

---

## 📋 分步实施计划

### 第一阶段: 基础设施准备 (预计1-2天)
1. **环境验证**
   - [ ] 确认批次4数据完整性
   - [ ] 验证代码依赖和路径配置
   - [ ] 建立批次5项目结构

2. **代码框架搭建**
   - [ ] 复制并修改批次4相关脚本
   - [ ] 创建`batch5_phase1_unified_embedding.py`
   - [ ] 创建`batch5_phase2_enhanced_visualization.py`
   - [ ] 创建`batch5_phase3_pure_dataset_construction.py`
   - [ ] 创建`batch5_phase5_independent_evaluation.py`

### 第二阶段: 核心实验实施 (预计2-3天)
1. **Phase 1执行**: 统一embedding空间构建
2. **Phase 2执行**: 增强可视化分析  
3. **Phase 3执行**: 纯净数据集重构

### 第三阶段: 模型训练评估 (预计1-2天)
1. **Phase 4执行**: 32个实验点完整训练
2. **结果分析**: 性能对比和统计检验
3. **报告生成**: 综合分析和可视化报告

### 第四阶段: 分析总结 (预计1天)
1. **科学发现总结**: 验证/拒绝实验假设
2. **方法论评估**: 纯合成数据方法的优劣
3. **实践指导**: 为后续研究提供建议

---

## 🔍 预期科学发现

### 核心假设验证

#### H1: 纯合成数据性能差距
**预期**: Pure Synthetic < Baseline Real (显著差异)
**量化指标**: F1-Score差异 > 5%
**统计检验**: 配对t检验，p < 0.05

#### H2: Prompt策略性能排序  
**预期**: pure_strong ≥ pure_rewrite ≥ pure_weak
**验证方法**: 性能曲线斜率和AUC对比

#### H3: 分层效果确认
**预期**: 在embedding可视化中观察到层次聚类模式
**定量分析**: 层间距离 > 层内距离，聚类质量指标

#### H4: 比例敏感性
**预期**: 低比例(5%)下性能差距更大
**分析重点**: 性能曲线在低比例区间的斜率变化

### 方法论贡献
1. **纯合成数据基准**: 建立LLM合成数据的独立性能基线
2. **可视化方法**: 统一embedding空间的多维度分析框架
3. **评估标准**: 合成数据质量的定量评估指标体系

---

## ⚠️ 潜在挑战与应对

### 数据充足性挑战
**问题**: 每个prompt变体仅400个样本，高比例配置可能数据不足
**应对**: 
- 优先实施5%-15%比例配置
- 20%比例配置采用重采样策略
- 分层实验作为可选扩展

### 计算资源需求
**embedding计算**: ~80万样本 × 384维，预计2-3小时
**模型训练**: 32个实验点，预计4-6小时
**可视化生成**: 交互式图表，预计1-2小时

### 实验对照严谨性
**关键控制变量**:
- 固定测试集(绝不变化)
- 固定良性数据池
- 固定随机种子(2025)
- 统一的模型超参数

---

## 📊 成功标准

### 技术指标
- [ ] 32个实验点100%成功完成
- [ ] 统一embedding空间构建完成
- [ ] 多维度可视化分析完成
- [ ] 统计显著性检验完成

### 科学标准  
- [ ] 4个核心假设得到验证或拒绝
- [ ] 纯合成数据性能基线建立
- [ ] 方法论框架完整输出
- [ ] 实践指导建议明确

### 可重现性标准
- [ ] 完整的代码和数据追踪
- [ ] 详细的实验参数记录  
- [ ] 清晰的复现步骤文档
- [ ] 开放的数据和结果共享

---

## 🎯 批次5 vs 批次4 核心差异总结

| 维度 | 批次4 | 批次5 |
|------|-------|-------|
| **核心问题** | 数据混合污染 | 数据独立性验证 |
| **训练数据** | 真实+合成混合 | 严格分离的纯净数据 |
| **对比方案** | 混合vs真实 | 纯合成vs纯真实 |
| **实验数量** | 20配置×2模型=40点 | 16配置×2模型=32点 |
| **可视化** | 分层可视化 | 统一空间多维可视化 |
| **科学价值** | 实用性验证 | 理论机制探索 |

---

*创建日期: 2025-07-30*  
*基于批次4问题修正的科学实验重设计*