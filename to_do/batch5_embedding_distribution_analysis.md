# Batch 5: Malicious Data Embedding Distribution Analysis

## 实验目标
基于batch4的分类结果发现，synthetic data虽然性能略低于real data但可以达到comparable results。现在要深入分析malicious数据的embedding分布情况，探究synthetic data与real malicious data在特征空间中的分布差异。

## 实验设计

### 1. 数据准备
- **Real malicious data**: CEAS-08训练集中的恶意邮件
- **Synthetic rewrite**: 使用原始rewrite prompt生成的合成数据
- **Synthetic rewrite strong**: 使用强化rewrite prompt生成的合成数据  
- **Synthetic rewrite weak**: 使用弱化rewrite prompt生成的合成数据
- **Benign data**: 保持一致的良性邮件数据（用于对比参考）

### 2. 数据标注策略
为每个数据样本添加唯一标识：
- `real_malicious`: 真实恶意数据
- `synthetic_rewrite`: 合成数据（原始prompt）
- `synthetic_rewrite_strong`: 合成数据（强化prompt）
- `synthetic_rewrite_weak`: 合成数据（弱化prompt）
- `real_benign`: 真实良性数据（参考基线）

### 3. Embedding生成
- **模型选择**: all-MiniLM-L6-v2 (与之前实验保持一致)
- **文本预处理**: subject + body拼接
- **统一embedding空间**: 将所有数据放在同一个embedding空间中确保可比性

### 4. 降维分析
#### 4.1 参数调优
- **t-SNE**: 
  - perplexity: [5, 10, 30, 50, 100]
  - learning_rate: [10, 50, 200, 1000]
  - n_iter: [1000, 3000, 5000]
- **UMAP**:
  - n_neighbors: [5, 15, 50, 100]
  - min_dist: [0.01, 0.1, 0.5]
  - metric: ['cosine', 'euclidean']

#### 4.2 最优参数选择
- 使用silhouette score、Davies-Bouldin index等指标
- 视觉评估聚类分离度
- 选择最能区分不同数据类型的参数组合

### 5. 聚类分析
#### 5.1 聚类数量确定
- Elbow method (K-means inertia)
- Silhouette analysis
- Gap statistic
- 预期聚类数: 4-6个（对应不同的数据类型及可能的子类）

#### 5.2 聚类质量评估
- **Silhouette Score**: 评估聚类内聚性和分离性
- **Adjusted Rand Index**: 与真实标签的一致性
- **Davies-Bouldin Index**: 聚类紧密度和分离度

### 6. 相似度分析
#### 6.1 类别间距离计算
- 计算各synthetic类别与real malicious的中心距离
- 计算类别内variance和类别间separation
- 使用cosine similarity和euclidean distance

#### 6.2 分布重叠分析
- 计算各类别在embedding空间中的重叠程度
- 识别哪种synthetic方法最接近real malicious分布
- 量化synthetic data的"authentic度"

### 7. 可视化设计
#### 7.1 静态可视化
- 2D散点图，不同颜色表示不同数据类型
- 添加聚类中心标记
- 密度等高线图显示分布密度

#### 7.2 交互式可视化
- **Plotly interactive scatter plot**:
  - Hover显示原始邮件内容（subject + body前100字符）
  - 点击高亮同类别数据点
  - 缩放和平移功能
- **Dashboard界面**:
  - 参数调节滑块（perplexity, n_neighbors等）
  - 实时更新可视化结果
  - 聚类指标实时显示

### 8. 预期结果分析
#### 8.1 假设验证
- **H1**: Real malicious data形成相对紧密的聚类
- **H2**: 不同synthetic方法产生不同的分布模式
- **H3**: Strong prompt的synthetic data更接近real malicious分布
- **H4**: 存在部分synthetic data偏离主要malicious分布的异常点

#### 8.2 质量评估指标
- **分布相似度**: 与real malicious的KL散度
- **聚类纯度**: 同类别数据的聚集程度
- **边界清晰度**: 不同类别间的分离程度

### 9. 实验输出
#### 9.1 量化结果
- 各参数组合下的聚类质量指标表格
- 类别间相似度矩阵
- 最优参数配置报告

#### 9.2 可视化结果
- 最优参数下的2D embedding分布图
- 交互式HTML可视化文件
- 聚类分析结果图表

#### 9.3 分析报告
- Synthetic data质量评估结论
- 不同rewrite策略的效果对比
- 对后续synthetic data生成的改进建议

## 数据存储结构
```
data/batch5/
├── labeled_data/                    # 标注后的数据
│   ├── real_malicious_labeled.csv.gz
│   ├── synthetic_rewrite_labeled.csv.gz
│   ├── synthetic_strong_labeled.csv.gz
│   ├── synthetic_weak_labeled.csv.gz
│   ├── real_benign_labeled.csv.gz
│   └── combined_labeled_data.csv.gz  # 合并后的所有数据
├── embeddings/                      # embedding结果
│   ├── combined_embeddings.npy      # 统一embedding矩阵
│   ├── embedding_metadata.json     # embedding相关信息
│   └── text_data.csv.gz           # 对应的文本和标签
├── analysis_results/               # 分析结果
│   ├── parameter_optimization/     # 参数调优结果
│   ├── clustering_results/        # 聚类分析结果
│   └── similarity_analysis/       # 相似度分析结果
└── visualizations/                # 可视化结果
    ├── static_plots/              # 静态图表
    ├── interactive_plots/         # 交互式图表
    └── dashboard/                 # Dashboard文件
```

## 技术实现路径 - 阶段化脚本

### Phase 1: 数据整合与标注 (`scripts/batch5_phase1_data_preparation.py`)
1. 加载所有原始数据集
2. 为每个数据样本添加类别标签
3. 合并所有数据到统一格式
4. 保存标注后的数据到 `data/batch5/labeled_data/`

### Phase 2: Embedding生成 (`scripts/batch5_phase2_embedding_generation.py`)
1. 加载合并后的标注数据
2. 使用all-MiniLM-L6-v2生成统一embedding
3. 保存embedding到 `data/batch5/embeddings/`
4. 保存embedding元数据和对应文本信息

### Phase 3: 降维参数调优 (`scripts/batch5_phase3_dimensionality_reduction.py`)
1. 加载embedding数据
2. 实现t-SNE和UMAP参数网格搜索
3. 计算各种聚类质量指标
4. 选择最优参数组合
5. 保存参数调优结果到 `data/batch5/analysis_results/parameter_optimization/`

### Phase 4: 聚类分析 (`scripts/batch5_phase4_clustering_analysis.py`)
1. 使用最优参数进行降维
2. 确定最优聚类数量
3. 执行聚类算法
4. 评估聚类质量
5. 保存聚类结果到 `data/batch5/analysis_results/clustering_results/`

### Phase 5: 相似度分析 (`scripts/batch5_phase5_similarity_analysis.py`)
1. 计算类别间距离矩阵
2. 分析分布重叠情况
3. 量化synthetic data与real malicious的相似度
4. 保存相似度分析结果到 `data/batch5/analysis_results/similarity_analysis/`

### Phase 6: 可视化实现 (`scripts/batch5_phase6_visualization.py`)
1. 创建静态分布图并保存到 `data/batch5/visualizations/static_plots/`
2. 实现交互式可视化并保存到 `data/batch5/visualizations/interactive_plots/`
3. 构建分析dashboard并保存到 `data/batch5/visualizations/dashboard/`

## 预期价值
1. **验证synthetic data质量**: 通过分布分析验证synthetic data是否保持了malicious特征
2. **优化生成策略**: 识别哪种rewrite方法效果最好
3. **指导后续研究**: 为改进synthetic data生成提供数据支持
4. **可视化洞察**: 提供直观的数据分布理解

## 成功标准
1. 成功生成所有数据的统一embedding
2. 找到最优的降维参数配置
3. 清晰区分不同数据类型的分布
4. 量化synthetic data与real data的相似度
5. 创建可交互的可视化分析工具

---
*创建日期: 2025-07-30*  
*实验类型: Embedding Distribution Analysis*  
*数据集: CEAS-08 + Synthetic Rewrite Variants*