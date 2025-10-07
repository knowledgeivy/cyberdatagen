# 🎯 正式实验最终配置总结

## ✅ 配置完成确认

**实验名称**: `full_ceas08_gpt41mini_v1`  
**配置文件**: `config/full_ceas08_gpt41mini_v1.yaml`  
**状态**: ✅ 准备就绪

---

## 📊 核心参数

| 参数 | 值 | 说明 |
|------|-----|------|
| **数据集** | CEAS-08 | SevenPhishingEmailDataset (39,154样本) |
| **模型** | GPT-4.1-mini | OpenAI API |
| **Groups (R)** | 20 | 数据分组数 |
| **Sample size (N)** | 1,000 | 每组样本数 |
| **Trials** | 5 | 每个配置重复次数 |
| **Spam ratio** | 0.1 (1:9) | 不平衡比例 |
| **Synthetic ratios** | [0, 10, 20, ..., 100] | 11个测试点 |
| **Strategies** | 4 | within/cross/real_fixed/full_random |

---

## 🔬 统计设计

- **比较样本量**: 100 (20 groups × 5 trials)
- **统计power**: 84.4% (检测 Cohen's d ≥ 0.3)
- **显著性检验**: Paired t-test with FDR correction (α=0.05)
- **效应量分析**: Cohen's d, KL divergence

**关键洞察**: 
- 统计power主要由 `R × trials` 决定，而非 N
- 100个比较样本足以检测中等效应
- 每组100个spam样本足够训练高质量分类器

---

## 💰 成本与规模

| 指标 | 数值 |
|------|------|
| **API调用** (3 prompts) | 6,000 |
| **估算成本** | ~$2-5 |
| **总配置数** | 2,640 |
| **总训练次数** | 13,200 |
| **预计时间** | 1-2天 |

**成本节省**: 相比原计划N=9000节省90%成本 ($2 vs $19)

---

## 📂 目录结构

### Pilot实验 (已完成)
```
data/pilot_experiments/ceas08_gpt41mini/
  ├── processed/
  ├── synthetic/
  └── datasets/

output/pilot_experiments/ceas08_gpt41mini/
  ├── results/
  ├── plots/
  └── reports/
```

### 正式实验 (准备就绪)
```
data/full_experiments/ceas08_gpt41mini/
  ├── processed/  ✅
  ├── synthetic/  ✅
  └── datasets/   ✅

output/full_experiments/ceas08_gpt41mini/
  ├── results/    ✅
  ├── plots/      ✅
  └── reports/    ✅
```

---

## 🔧 已完成的工作

1. ✅ **Pilot实验整理**
   - 所有文件移至 `pilot_experiments/` 目录
   - 配置文件路径已更新

2. ✅ **数据源更新**
   - 从旧batch数据 (2,000样本) 升级到 SevenPhishingEmailDataset (39,154样本)
   - `preprocessor.py` 已更新支持新数据源
   - 向后兼容旧格式

3. ✅ **参数优化**
   - N从9000优化至1000（成本降低90%，统计power保持84%）
   - 基于统计分析的科学决策

4. ✅ **代码验证**
   - 配置文件加载测试通过
   - 数据加载功能测试通过
   - SevenPhishingEmailDataset兼容性确认

5. ✅ **可视化优化**
   - 600 DPI学术期刊风格
   - 线型+标记组合（黑白打印友好）
   - 95% CI置信区间
   - 标题间距优化

---

## 🚀 执行命令

### 方式1: 一键执行（推荐）
```bash
tmux new -s full_experiment
./run_full_experiment.sh
```

### 方式2: 分步执行

**Step 1: 数据预处理**
```bash
poetry run python scripts/step1_data_preprocessing.py \
  --config config/full_ceas08_gpt41mini_v1.yaml
```

**Step 2: LLM生成 (3个prompts)**
```bash
for prompt in original strong weak; do
  poetry run python scripts/step2_llm_generation.py \
    --config config/full_ceas08_gpt41mini_v1.yaml \
    --prompt $prompt
done
```

**Step 3: 数据集构建 (4个strategies)**
```bash
for strategy in within_group cross_group real_fixed_random_synthetic full_random; do
  poetry run python scripts/step3_dataset_construction.py \
    --config config/full_ceas08_gpt41mini_v1.yaml \
    --prompt original \
    --strategy $strategy
done
```

**Step 4: 分类实验 (顺序执行避免冲突)**
```bash
for strategy in within_group cross_group real_fixed_random_synthetic full_random; do
  poetry run python scripts/step4_classification.py \
    --config config/full_ceas08_gpt41mini_v1.yaml \
    --strategy $strategy \
    --resume
done
```

**Step 5: 统计分析**
```bash
poetry run python scripts/step5_enhanced_statistical_analysis.py \
  --config config/full_ceas08_gpt41mini_v1.yaml
```

**Step 6: 可视化**
```bash
poetry run python scripts/step6_enhanced_visualization.py \
  --config config/full_ceas08_gpt41mini_v1.yaml \
  --results output/full_experiments/ceas08_gpt41mini/results/full_ceas08_gpt41mini_v1_classification_results.json
```

---

## ⚠️ 开始前检查清单

- [ ] OpenAI API key已配置 (`echo $OPENAI_API_KEY`)
- [ ] API余额充足 (建议 >$10)
- [ ] 网络连接稳定
- [ ] 磁盘空间充足 (>5GB)
- [ ] 内存充足 (建议32GB+)
- [ ] 已创建tmux会话（长时间运行）
- [ ] 已阅读 `PRE_EXPERIMENT_CHECKLIST.md`

---

## 📚 关键文件

- **配置**: `config/full_ceas08_gpt41mini_v1.yaml`
- **执行脚本**: `run_full_experiment.sh`
- **检查清单**: `PRE_EXPERIMENT_CHECKLIST.md`
- **实验设计**: `README_experiment.md`
- **Pilot结果**: `output/pilot_experiments/ceas08_gpt41mini/plots/`

---

## 🎓 技术亮点

1. **统计严谨**: 84.4% power, FDR校正, 效应量分析
2. **成本优化**: $2预算，90%成本节省
3. **可重复性**: 随机种子控制，5次trials
4. **学术规范**: 高分辨率PDF图表，置信区间可视化
5. **代码质量**: 向后兼容，错误处理，完整日志

---

**准备完成！确认API和网络后即可开始实验 🚀**

---

*最后更新: 2025-10-03*
*实验设计者: Based on README_experiment.md*
