# 批次6实验设计方案 - 改进合成数据生成方法论
*基于批次5结果分析的合成数据质量优化重设计*

## 🚀 实验背景与问题分析

### 批次5关键问题发现
**性能差距显著**: 批次5实验结果显示合成数据检测性能严重不足
- 合成数据F1 Score: ~0.6 
- 真实数据F1 Score: ~0.8
- **性能差距**: 25%+ 的显著差异

### 根因分析: 合成数据生成方法缺陷

#### 批次5的问题方法 (❌ 低质量)
```python
# batch4_phase2_synthetic_generation.py
'rewrite_strong': {
    'system': "You are an expert in advanced phishing email generation...",
    'user': "Create a sophisticated variant of this phishing email..."
}
```
**问题**:
1. **硬编码prompts**: 简单字符串，缺乏结构化指导
2. **输出格式不统一**: 无统一约束，解析容易出错  
3. **指令模糊**: "sophisticated", "harder to detect"等抽象描述
4. **缺乏质量控制**: 无validation机制确保生成质量

#### 成功的方法 (✅ 高质量)
```yaml
# config/prompts/rewrite_generation_strong.yaml
prompts:
  rewrite_phishing:
    system:
      template: |
        You are an expert at rewriting phishing email content...
        Make the rewritten data strong to reflect representative phishing email characteristics
    user:
      template: |
        Rewrite this phishing email but keep the content very similar...
        Return only CSV format data with subject and body fields.
```
**优势**:
1. **结构化prompts**: YAML配置，专业模板设计
2. **统一CSV输出**: 标准格式，可靠解析
3. **具体指导**: "representative phishing email characteristics"明确目标
4. **成熟工具链**: `real_data_rewriter.py`经过验证的生成流程

---

## 🎯 批次6核心改进策略

### 设计理念
1. **保持科学框架**: 沿用批次5的纯净数据验证设计
2. **升级生成方法**: 采用配置化、高质量的合成数据生成
3. **整合最佳实践**: 融合批次4的ID体系 + 批次5的实验设计

### 技术路线图
```
原始数据 → [批次5方法] → 种子标注 → [real_data_rewriter方法] → 合成数据生成 
    ↓
合成数据 → [批次4方法] → 合成数据ID标注 → [批次5方法] → 可视化+分类
```

---

## 📊 批次6实验流程设计

### Phase 0: 数据基础准备 🔄
```
目标: 沿用批次4/5的基础数据，确保实验可比性

输入数据复用:
- data/batch4_fresh/raw_train_set.csv (~104万样本)
- data/batch4_fresh/train_malicious.csv (~52万恶意样本)  
- data/batch4_fresh/train_benign.csv (~52万良性样本)
- data/batch4_fresh/raw_test_set.csv (~26万测试样本)

验证要求:
✅ 确认数据完整性和可访问性
✅ 验证与批次5的数据一致性
✅ 建立批次6项目结构
```

### Phase 1: 改进的种子样本准备 🆕  
```
目标: 使用批次5的方法为原始数据建立unique ID追踪

核心改进:
- 保持批次5的数据源分类修复 (background vs seeds)
- 确保1:1种子到合成数据的对应关系

操作流程:
1. 提取真实恶意种子样本 (2000个，扩大规模)
2. 建立完整的ID追踪体系
   - real_malicious_background: 15,000样本
   - real_malicious_seeds: 2000样本 (关键种子)
3. 为种子样本分配stratified layers (仅core/edge两层)

输出:
- data/batch6/seed_preparation/
  ├── real_malicious_seeds_with_layers.csv    # 2000个带层标注的种子
  ├── seed_layer_distribution.json           # 层分布统计  
  └── seed_id_mapping.json                  # ID映射关系

预期种子分布 (简化为2层):
- core层种子: 1000个 (距离最近)
- edge层种子: 1000个 (距离最远)
```

### Phase 2: 高质量合成数据生成 🆕 核心改进
```
目标: 使用real_data_rewriter.py方法生成高质量合成数据

关键改进点:
❌ 避免硬编码prompt (批次5问题)
✅ 使用YAML配置的专业prompt模板
✅ 统一CSV输出格式  
✅ 并行生成提高效率

生成配置:
- 输入: 2000个分层种子样本 (core: 1000个, edge: 1000个)
- 方法: src/cyberdata/process/real_data_rewriter.py
- Prompt配置:
  * rewrite_generation.yaml (基础重写)
  * rewrite_generation_strong.yaml (强化变体)  
  * rewrite_generation_weak.yaml (弱化变体)
- 并行处理: 100 workers (优化并发，充分利用API限制)
- 模型: GPT-4.1-mini (temperature=0.8)

生成流程:
1. 分层处理种子数据 (简化为2层)
   foreach layer in [core, edge]:
     foreach prompt_type in [original, strong, weak]:
       rewrite_malicious_data_parallel(layer_seeds, prompt_config)

2. 质量验证
   - CSV格式验证
   - 内容完整性检查
   - 恶意标签一致性确认

输出:
- data/batch6/synthetic_generation/
  ├── core_original_synthetic.csv.gz         # 1000个core基础重写
  ├── core_strong_synthetic.csv.gz           # 1000个core强化重写  
  ├── core_weak_synthetic.csv.gz             # 1000个core弱化重写
  ├── edge_original_synthetic.csv.gz         # 1000个edge基础重写
  ├── edge_strong_synthetic.csv.gz           # 1000个edge强化重写
  └── edge_weak_synthetic.csv.gz             # 1000个edge弱化重写

生成统计:
- 总API调用: 6,000次 (2000种子 × 3个prompt)
- 预计耗时: 1-1.5小时 (100并发，充分利用30,000 RPM限制)
- 预计成本: $15-25
- 生成样本: 6,000个高质量合成样本
```

### Phase 3: 合成数据ID标注 🔄
```
目标: 使用批次4的成熟ID标注方法为合成数据建立追踪体系

ID标注格式 (沿用批次4标准):
- 格式: synth_{layer}_{seed_id}_{prompt_type}
- 示例: 
  * synth_core_001_original
  * synth_inner_099_strong  
  * synth_edge_050_weak

操作流程:
1. 读取各个合成数据文件
2. 提取原始种子ID信息
3. 分配标准化的合成数据ID
4. 建立完整的种子→合成数据映射关系

输出:
- data/batch6/synthetic_with_ids/
  ├── all_synthetic_data.csv                 # 6,000个带ID的合成样本
  ├── synthetic_id_mapping.json              # 完整ID映射
  └── generation_metadata.json               # 生成过程元数据

验证要求:
✅ 每个种子样本对应3个合成变体 (2000×3=6,000)
✅ ID格式统一且唯一
✅ 与原始种子的可追踪对应关系
```

### Phase 4: 统一Embedding空间构建 🔄  
```
目标: 复用批次5的成功方法，为所有数据建立统一embedding

输入数据汇总:
- 真实恶意背景样本: 15,000个
- 真实恶意种子样本: 2,000个  
- 高质量合成样本: 6,000个 (改进生成)
- 真实良性样本: 13,837个
- 固定测试集: 7,826个

操作 (复用批次5 Phase 1):
1. ✅ 对所有数据生成384维embedding向量
2. ✅ 保持ID追踪体系完整性  
3. ✅ 计算分层距离和质心信息
4. ✅ 应用数据源分类修复

预期改进:
🎯 更高质量的合成数据应在embedding空间呈现更好的分布特征
🎯 种子样本与合成数据的距离关系更加合理

输出:
- data/batch6/unified_embeddings/
  ├── all_embeddings.npy                     # 统一embedding矩阵
  ├── embedding_metadata.csv.gz              # ID和标签对应关系
  ├── layer_centroids.json                   # 各层质心信息
  └── distance_statistics.json               # 距离分布统计
```

### Phase 5: 增强可视化分析 🔄
```
目标: 复用批次5的可视化框架，验证合成数据质量改进

可视化维度 (简化层次):
1. ✅ 数据源对比: 背景数据 vs 种子样本 vs 合成数据
2. ✅ Prompt策略对比: original vs strong vs weak
3. ✅ 分层效果对比: core vs edge (简化为2层)  
4. ✅ 聚类特征分析: K-means 和 DBSCAN

关键改进预期:
🎯 合成数据在embedding空间的分布更加合理
🎯 种子与合成数据的聚类关系更加清晰
🎯 不同prompt策略的差异更加明显

输出 (格式与批次5一致):
- data/batch6/visualizations/
  ├── static_plots/                          # 2D静态可视化
  ├── interactive_plots/                     # 3D交互可视化
  ├── cluster_analysis/                      # 聚类质量分析
  └── distribution_analysis/                 # 距离分布分析
```

### Phase 6: 纯净数据集重构 🔄
```
目标: 复用批次5的纯净数据集设计，验证改进的合成数据性能

数据集配置 (与批次5保持一致):
1. baseline_real: 100%真实恶意数据 + 固定良性数据
2. pure_original: 100%基础重写合成数据 + 固定良性数据
3. pure_strong: 100%强化合成数据 + 固定良性数据  
4. pure_weak: 100%弱化合成数据 + 固定良性数据

比例配置: 5%, 10%, 15%, 20%恶意数据比例
数据集大小: 每个10,000样本

关键改进预期:
🎯 改进的合成数据应显著缩小与真实数据的性能差距
🎯 F1 Score差距从25%减少到15%以内

输出:
- data/batch6/pure_datasets/
  ├── baseline_real_5pct/                   # 纯真实数据基线
  ├── pure_original_5pct/                   # 纯基础重写合成数据
  ├── pure_strong_5pct/                     # 纯强化合成数据
  └── pure_weak_5pct/                       # 纯弱化合成数据
  (每个比例4个配置 × 4个比例 = 16个数据集)
```

### Phase 7: 模型训练与性能验证 🔄
```
目标: 复用批次5的评估框架，量化合成数据质量改进

训练配置 (与批次5保持一致):
- 模型: RandomForest + SVM  
- 评估指标: Accuracy, Precision, Recall, F1-Score
- 测试集: 固定的独立测试集

实验矩阵: 16个数据集配置 × 2个模型 = 32个实验点

成功标准:
🎯 **主要目标**: Pure Synthetic F1 Score > 0.75 (从0.6提升)
🎯 **理想目标**: Pure Synthetic接近Baseline Real (F1差距<10%)
🎯 **方法验证**: Strong > Original > Weak 的性能排序更加明显

输出:
- data/batch6/results/
  ├── performance_matrix.csv                # 32×指标的性能矩阵
  ├── batch5_vs_batch6_comparison.csv       # 直接性能对比
  ├── improvement_analysis.png              # 改进效果可视化
  └── comprehensive_report.md               # 综合分析报告
```

---

## 🔧 关键技术改进总结

### 1. 合成数据生成质量提升
| 方面 | 批次5方法 | 批次6改进 |
|------|-----------|-----------|
| **Prompt设计** | 硬编码字符串 | YAML专业模板 |
| **输出格式** | 自由文本 | 强制JSON结构 |
| **指导精度** | 抽象描述 | 具体特征指导 |  
| **工具成熟度** | 实验性代码 | 验证的工具链 |
| **质量控制** | 基础验证 | 多层验证机制 |

### 2. 实验设计连续性
```
批次4: ID体系设计 → 批次5: 纯净数据验证 → 批次6: 高质量生成
  ↓                    ↓                        ↓
成熟的追踪方法      科学的实验框架           改进的数据源
```

### 3. 预期性能改进量化
- **当前差距**: 真实数据F1(0.8) - 合成数据F1(0.6) = 0.2 (25%差距)
- **目标改进**: 合成数据F1提升至0.75+ (≤15%差距)  
- **理想结果**: 合成数据F1接近0.8 (≤10%差距)

---

## 📋 实施计划与资源估算

### 第一阶段: 数据准备与生成 (预计1.5-2天)
1. **Phase 1**: 种子样本准备 (0.5天)
2. **Phase 2**: 高质量合成数据生成 (0.5天，100并发加速)  
3. **Phase 3**: 合成数据ID标注 (0.5天)

### 第二阶段: 分析与可视化 (预计1-2天)  
1. **Phase 4**: 统一embedding构建 (0.5天)
2. **Phase 5**: 增强可视化分析 (0.5-1天)

### 第三阶段: 验证与评估 (预计1-2天)
1. **Phase 6**: 纯净数据集重构 (0.5天)  
2. **Phase 7**: 模型训练与性能验证 (0.5-1天)

### 资源需求估算
- **API调用**: 6,000次 GPT-4.1-mini调用
- **预计成本**: $15-25  
- **计算资源**: 中等GPU需求 (embedding生成)
- **存储需求**: ~25GB (数据+模型+结果)

---

## ⚠️ 风险管控与应急预案

### 潜在挑战
1. **API限制**: OpenAI API调用限制可能影响生成速度
   - **应对**: 实施请求间隔控制，准备API key轮换
   
2. **生成质量控制**: CSV解析失败或内容质量问题
   - **应对**: 多层验证，自动重试机制
   
3. **性能改进不及预期**: 合成数据质量仍然不足
   - **应对**: 准备prompt模板微调，考虑temperature参数调整

### 回退策略
如果批次6方法未能达到预期改进:
1. **渐进优化**: 保留批次6基础设施，仅调整prompt策略
2. **混合方法**: 探索真实+合成数据的最优混合比例
3. **方法论总结**: 为未来实验提供详细的失败分析和改进建议

---

## 🎯 预期科学贡献

### 方法论验证
1. **配置化生成**: 验证YAML配置vs硬编码的质量差异
2. **输出格式影响**: 量化CSV结构化输出对生成质量的影响
3. **工具链成熟度**: 证明成熟工具链vs临时代码的优势

### 实践指导
1. **最佳实践**: 为LLM合成数据生成建立标准化流程
2. **质量基准**: 建立合成数据质量的量化评估标准  
3. **方法复用**: 为其他cybersecurity数据集提供可复用的生成框架

### 理论探索
1. **质量-性能关系**: 建立合成数据生成质量与下游任务性能的定量关系
2. **prompt工程**: 验证精细化prompt设计对生成效果的影响程度
3. **评估方法论**: 完善合成数据有效性的多维度评估体系

---

## 📊 成功标准与验收条件

### 技术指标
- [ ] 32个实验点100%成功完成
- [ ] 合成数据F1 Score > 0.75 (相比批次5的0.6有显著提升)
- [ ] Strong > Original > Weak 的性能排序清晰可见
- [ ] 统一embedding空间成功构建

### 科学标准
- [ ] 生成方法改进的有效性得到定量验证
- [ ] 批次5 vs 批次6的对比分析完成  
- [ ] 方法论框架的可复用性得到验证
- [ ] 实践指导建议明确具体

### 可重现性标准  
- [ ] 完整的配置文件和代码追踪
- [ ] 详细的生成过程日志记录
- [ ] 清晰的复现步骤文档
- [ ] 开放的数据和结果共享

---

## 🔄 批次间方法论演进总结

| 维度 | 批次4 | 批次5 | 批次6 |  
|------|-------|-------|-------|
| **核心问题** | 分层效果验证 | 数据独立性验证 | 合成数据质量优化 |
| **生成方法** | 硬编码prompts | 硬编码prompts | YAML配置prompts |
| **输出格式** | 自由文本 | 自由文本 | CSV结构化输出 |
| **工具链** | 临时脚本 | 临时脚本 | real_data_rewriter.py |
| **ID体系** | 成熟完整 ✓ | 改进优化 ✓ | 继承最佳实践 ✓ |
| **实验设计** | 混合数据对比 | 纯净数据对比 ✓ | 纯净数据对比 ✓ |
| **预期F1** | 混合结果0.75+ | 合成数据0.6 | 合成数据0.75+ |

**进化逻辑**: 批次4建立基础 → 批次5发现问题 → 批次6解决问题

---

*创建日期: 2025-07-30*  
*基于批次5性能分析的合成数据生成方法论改进设计*