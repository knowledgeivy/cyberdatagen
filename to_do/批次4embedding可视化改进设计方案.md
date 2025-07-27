# 批次4 Embedding可视化改进设计方案
*创建日期: 2025-07-27*  
*基于用户反馈的embedding可视化优化设计*

---

## 🎯 问题识别

### 当前batch4可视化的主要缺陷

1. **分离的embedding空间**: 每个配置独立进行PCA/t-SNE降维，无法直接对比
2. **缺失多数据集对比**: 无法在同一图中对比不同生成方式(real, core, inner, outer, edge)
3. **缺失交互式功能**: 只有静态PNG图片，不如batch3的interactive HTML
4. **维度信息丢失**: 无法同时展示距离层、malicious ratio、prompt类型等多维信息

### 与batch3对比差异
- **batch3**: 多数据集统一空间 + interactive plot + 动态探索
- **batch4**: 20个独立静态图 + 分离的embedding空间

---

## 📊 改进方案设计

### **阶段1: 立即修复 - 统一Embedding空间**

#### 1.1 数据源整合
```
输入数据源:
├── 真实数据 (real_only组)
│   ├── train_malicious.csv (17,463样本) 
│   └── train_benign.csv (1,027,102样本)
└── 合成数据 (4个距离层)
    ├── core_synthetic.csv (300样本, ID: synth_core_{seed_id}_{prompt_type})
    ├── inner_synthetic.csv (300样本, ID: synth_inner_{seed_id}_{prompt_type})  
    ├── outer_synthetic.csv (300样本, ID: synth_outer_{seed_id}_{prompt_type})
    └── edge_synthetic.csv (300样本, ID: synth_edge_{seed_id}_{prompt_type})

总计: ~1.05M样本 (真实) + 1,200样本 (合成)
```

#### 1.2 统一Embedding流程
```
Step 1: 数据合并和标注
- 合并所有文本数据到统一DataFrame
- 添加数据源标签: ['real_malicious', 'real_benign', 'core_synthetic', 'inner_synthetic', 'outer_synthetic', 'edge_synthetic']
- 添加prompt类型: ['real', 'rewrite', 'rewrite_strong', 'rewrite_weak']
- 保持原始ID追踪

Step 2: 统一Embedding生成
- 使用相同的all-MiniLM-L6-v2模型
- 批量处理所有样本到384维向量
- 保存为统一的embedding矩阵: unified_embeddings.npy

Step 3: 统一降维
- 对完整embedding矩阵进行PCA (n_components=50)
- 再进行t-SNE (n_components=2, perplexity=30) 
- 确保所有样本在相同的2D空间中
```

#### 1.3 采样策略 (解决大数据量问题)
```
为了可视化效率，按比例采样:
- real_benign: 5,000样本 (从1M+中采样)
- real_malicious: 5,000样本 (从17K中采样)  
- core_synthetic: 300样本 (全部)
- inner_synthetic: 300样本 (全部)
- outer_synthetic: 300样本 (全部)
- edge_synthetic: 300样本 (全部)

总计: 11,200样本用于可视化
```

---

### **阶段2: 补充Interactive功能**

#### 2.1 复用batch3框架结构
```
基于batch3_visualize_results.py的设计模式:
├── 数据准备层: 统一的DataFrame格式
├── 可视化引擎: Plotly图表生成
├── 交互功能: 动态过滤和hover信息
└── 输出管理: HTML文件和静态图片
```

#### 2.2 适配batch4数据结构
```
DataFrame Schema:
columns = [
    'text',                    # 原始文本
    'embedding_2d_x',          # t-SNE X坐标
    'embedding_2d_y',          # t-SNE Y坐标  
    'data_source',             # 数据来源: real/core/inner/outer/edge
    'label',                   # 0=benign, 1=malicious
    'prompt_type',             # real/rewrite/rewrite_strong/rewrite_weak
    'original_id',             # 原始样本ID
    'distance_to_centroid',    # 到恶意样本质心的距离
    'malicious_ratio_group'    # 5%/10%/15%/20% (用于分组)
]
```

---

### **阶段3: Interactive多维探索 (方案B详细设计)**

#### 3.1 Plotly 3D Scatter设计

##### 3.1.1 3D坐标轴定义
```
X轴: t-SNE Component 1 (embedding降维后的第一维)
Y轴: t-SNE Component 2 (embedding降维后的第二维)  
Z轴: Distance to Malicious Centroid (到恶意样本质心的余弦距离)

颜色编码: 数据来源层级
- 蓝色: real_malicious
- 绿色: real_benign  
- 红色: core_synthetic
- 橙色: inner_synthetic
- 紫色: outer_synthetic
- 黄色: edge_synthetic

点大小编码: Malicious Ratio (5%小点 → 20%大点)
```

##### 3.1.2 数据源映射
```
距离层 (Z轴): 
- real_malicious: 原始计算的距离值 [0.0, 1.0]
- real_benign: 计算到恶意质心的距离 [0.2, 0.9] 
- core_synthetic: [0.0, 0.25] (第一四分位数)
- inner_synthetic: [0.25, 0.50] (第二四分位数)
- outer_synthetic: [0.50, 0.75] (第三四分位数)  
- edge_synthetic: [0.75, 1.0] (第四四分位数)

数据来源: data/batch4_fresh/
├── train_malicious.csv (with distance_to_centroid)
├── train_benign.csv (需要计算distance_to_centroid)
├── synthetic/core_synthetic.csv (inherits seed distance)
├── synthetic/inner_synthetic.csv  
├── synthetic/outer_synthetic.csv
└── synthetic/edge_synthetic.csv
```

##### 3.1.3 Malicious Ratio维度
```
当前batch4设计中的ratio是数据集构建参数，而非embedding特征
建议映射方案:

方案A - 虚拟Ratio (推荐):
基于样本在embedding空间的密度分组:
- 5% group: 核心区域高密度样本
- 10% group: 中等密度样本  
- 15% group: 较低密度样本
- 20% group: 边缘稀疏样本

方案B - 实际Ratio:
使用batch4构建的20个数据集，分别可视化
需要生成4个interactive图表对应4个比例
```

#### 3.2 动态过滤功能

##### 3.2.1 图层控制
```
Plotly Dropdown/Checkbox控件:

数据层过滤:
☑ Real Malicious (blue)
☑ Real Benign (green)  
☑ Core Synthetic (red)
☑ Inner Synthetic (orange)
☑ Outer Synthetic (purple)
☑ Edge Synthetic (yellow)

Prompt类型过滤:
☑ Real Data
☑ Rewrite Basic
☑ Rewrite Strong  
☑ Rewrite Weak

比例组过滤 (如果使用方案A):
☑ 5% Dense Core
☑ 10% Medium Dense
☑ 15% Lower Dense  
☑ 20% Sparse Edge
```

##### 3.2.2 距离范围滑块
```
Distance Range Slider:
- Min: 0.0 (最接近质心)
- Max: 1.0 (最远离质心)  
- Default: [0.0, 1.0] (显示全部)
- Step: 0.05

实时过滤Z轴数据点
```

#### 3.3 Hover信息设计

##### 3.3.1 详细信息显示
```
Hover Template:
┌─────────────────────────────────────┐
│ Sample ID: {original_id}            │
│ Source: {data_source}               │
│ Label: {label_name}                 │
│ Prompt: {prompt_type}               │
│ ─────────────────────────────────   │
│ Position: ({x:.3f}, {y:.3f})       │
│ Distance: {distance_to_centroid:.3f}│
│ Ratio Group: {malicious_ratio_group}│
│ ─────────────────────────────────   │
│ Text Preview: {text[:100]}...       │
└─────────────────────────────────────┘
```

##### 3.3.2 数据源标识
```
样本ID格式识别:
- train_real_mal_{index}: 真实恶意样本
- train_real_ben_{index}: 真实良性样本  
- synth_core_{seed_id}_rewrite: Core层基础重写
- synth_core_{seed_id}_rewrite_strong: Core层强化重写
- synth_core_{seed_id}_rewrite_weak: Core层弱化重写
- (inner/outer/edge同理)

距离值来源:
- 真实样本: 来自phase1计算的centroid距离
- 合成样本: 继承自seed样本的距离值
```

---

## 🔧 技术实现规格

### 数据流设计
```
Input Sources:
├── data/batch4_fresh/train_malicious.csv (17,463 samples)
├── data/batch4_fresh/train_benign.csv (1,027,102 samples)  
├── data/batch4_fresh/synthetic/core_synthetic.csv (300 samples)
├── data/batch4_fresh/synthetic/inner_synthetic.csv (300 samples)
├── data/batch4_fresh/synthetic/outer_synthetic.csv (300 samples)
├── data/batch4_fresh/synthetic/edge_synthetic.csv (300 samples)
└── data/batch4_fresh/malicious_centroid.json (质心坐标)

Processing Pipeline:
1. load_and_merge_all_data() → unified_df (11,200 samples after sampling)
2. generate_unified_embeddings() → embeddings_384d.npy  
3. unified_dimensionality_reduction() → embeddings_2d.npy
4. add_distance_and_metadata() → final_visualization_df.csv
5. create_interactive_3d_plot() → batch4_enhanced_interactive.html
6. create_comparative_analysis() → static_comparisons.png

Output Structure:
├── data/batch4_fresh/enhanced_visualizations/
│   ├── unified_embedding_data.csv (完整数据表)
│   ├── batch4_interactive_3d.html (主要交互图)
│   ├── layer_comparison_matrix.png (层间对比)  
│   ├── distance_distribution.png (距离分布)
│   └── prompt_effectiveness_analysis.png (prompt效果)
```

### 性能优化考虑
```
内存管理:
- 分批处理大型embedding生成 (batch_size=1000)
- 采样策略减少可视化数据量
- 压缩保存中间结果 (.npz格式)

计算效率:  
- 复用已有embedding结果 (如果存在)
- 并行计算距离矩阵
- 缓存t-SNE结果以避免重复计算

交互性能:
- Plotly WebGL渲染加速
- 分层数据加载 (按需显示)
- 简化hover信息计算
```

---

## 📈 预期成果

### 可视化输出
1. **统一3D交互图**: 展示所有数据层在相同空间的分布
2. **动态探索界面**: 支持实时过滤和多维分析  
3. **详细hover信息**: 完整的样本元数据展示
4. **层间对比图**: 静态图表补充定量分析

### 科学价值
1. **空间分布验证**: 验证不同距离层的embedding分布假设
2. **生成质量评估**: 通过embedding相似性评估LLM生成效果
3. **边缘效应可视化**: 直观展示edge层样本的分布特征
4. **prompt策略比较**: 对比三种prompt的效果差异

### 技术优势
1. **统一embedding空间**: 确保所有比较的科学性
2. **多维信息融合**: 距离、来源、prompt类型一图展示
3. **交互式探索**: 支持假设验证和深度分析
4. **可扩展设计**: 便于后续batch实验复用

---

## 🚀 实施计划

### Phase 1: 数据统一 (预计2小时)
- [ ] 数据源整合和清洗
- [ ] 统一embedding生成  
- [ ] 统一降维处理
- [ ] 元数据标注和ID追踪

### Phase 2: 基础可视化 (预计1小时)  
- [ ] 统一2D scatter plot
- [ ] 多层对比静态图
- [ ] 基础交互功能

### Phase 3: 3D交互增强 (预计2小时)
- [ ] 3D scatter plot实现
- [ ] 动态过滤控件  
- [ ] 详细hover信息
- [ ] 性能优化

### Phase 4: 分析报告 (预计1小时)
- [ ] 生成综合分析报告
- [ ] 创建使用说明文档
- [ ] 验证结果科学性

**总预计时间**: 6小时  
**关键里程碑**: 统一embedding空间 + 交互3D可视化

---

*设计完成，等待用户确认后开始实施* ✅