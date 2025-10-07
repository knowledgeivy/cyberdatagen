# Pre-Experiment Checklist & Code Review

**实验名称**: `full_ceas08_gpt41mini_v1`
**配置文件**: `config/full_ceas08_gpt41mini_v1.yaml`
**检查日期**: 2025-10-03

---

## ✅ 代码设计检查

### 1. **文件命名规范** ✅
所有输出文件统一使用 `{experiment.name}` 前缀：

- **日志文件**: `logs/full_experiments/ceas08_gpt41mini/{experiment.name}_{step}.log`
- **数据文件**:
  - Processed: `data/full_experiments/ceas08_gpt41mini/processed/ceas08_processed.csv.gz`
  - Synthetic: `data/full_experiments/ceas08_gpt41mini/synthetic/ceas08_synthetic_{prompt}_{model}.csv.gz`
  - Datasets: `data/full_experiments/ceas08_gpt41mini/datasets/{strategy}/train_r{ratio}_g{group}_t{trial}.csv.gz`
- **结果文件**: `output/full_experiments/ceas08_gpt41mini/results/{experiment.name}_classification_results.json`
- **可视化**: `output/full_experiments/ceas08_gpt41mini/plots/{experiment.name}_{plot_type}.png`
- **报告**: `output/full_experiments/ceas08_gpt41mini/reports/{experiment.name}_{report_type}.csv`

### 2. **路径隔离检查** ✅
不同实验的数据完全隔离，避免混淆：

```
data/full_experiments/
├── ceas08_gpt41mini/          # 本次实验
├── ceas08_claude/             # 未来实验（不同模型）
└── trec07_gpt41mini/          # 未来实验（不同数据集）

output/full_experiments/
├── ceas08_gpt41mini/          # 本次实验结果
└── ...                        # 其他实验结果
```

### 3. **关键参数对比 README** ✅

| 参数 | README设计值 | Pilot配置 | Full配置 | ✅ |
|------|-------------|----------|---------|---|
| n_groups (R) | 20 | 3 | **20** | ✅ |
| sample_size_per_group (N) | 9000 | 200 | **9000** | ✅ |
| spam_ratio | 1:9 | 0.1 | **0.1** | ✅ |
| synthetic_ratios | 平滑 | [0,25,50,75,100] | **[0,10,20,...,100]** | ✅ |
| trials_per_config | 5 | 5 | **5** | ✅ |
| strategies | 4种 | 4种 | **4种** | ✅ |

### 4. **数据采样Bug修复验证** ✅
已修复的关键Bug（dataset_builder.py）：
- ✅ Line 195: `random_state=self.random_state` (之前用的是 self.config.random_seed)
- ✅ Line 210: `random_state=self.random_state`
- ✅ Line 272: `random_state=self.random_state`

这确保了每个trial使用不同的随机种子，避免4个策略生成完全相同的数据。

### 5. **可视化风格检查** ✅
学术期刊风格配置：
- ✅ 600 DPI高分辨率
- ✅ Times New Roman字体
- ✅ 线型+标记组合（黑白打印友好）
- ✅ 95% CI置信区间阴影
- ✅ 合理的标题间距

---

## 📊 实验规模估算

### 配置复杂度
```
总配置数 = 4 strategies × 11 ratios × 3 prompts × 20 groups × 3 classifiers
         = 7,920 configurations

总训练次数 = 7,920 × 5 trials
          = 39,600 individual training runs
```

### API调用估算
```
LLM调用次数 = 20 groups × 9,000 samples × 0.1 spam_ratio × 3 prompts
           = 20 × 900 × 3
           = 54,000 API calls
```

### 成本估算（GPT-4.1-mini）
- **估计输入tokens**: ~54,000 calls × 300 tokens/call = 16.2M tokens
- **估计输出tokens**: ~54,000 calls × 500 tokens/call = 27M tokens
- **预估成本**: $200-$300 USD

### 时间估算
- **Step 1 (预处理)**: ~30分钟
- **Step 2 (LLM生成)**: ~8-12小时（取决于API速率）
- **Step 3 (数据集构建)**: ~2-3小时
- **Step 4 (分类实验)**: ~24-36小时（39,600次训练）
- **Step 5-6 (分析可视化)**: ~30分钟
- **总计**: 2-3天

### 存储估算
- **原始数据**: ~50MB
- **合成数据**: ~150MB (3 prompts)
- **训练集**: ~2GB (4 strategies × 11 ratios × 20 groups × 5 trials)
- **结果文件**: ~500MB
- **总计**: ~3GB

---

## 🔧 正式实验前准备

### 1. API配置检查
```bash
# 检查API key
echo $OPENAI_API_KEY

# 测试API连接（可选）
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY" | jq '.data[] | select(.id | contains("gpt-4"))'
```

### 2. 系统资源检查
```bash
# 检查磁盘空间（需要至少5GB）
df -h .

# 检查内存（建议32GB+）
free -h

# 检查CPU核心数
nproc
```

### 3. 依赖环境检查
```bash
# 激活环境
poetry shell

# 验证关键包
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
```

### 4. 配置文件验证
```bash
# 验证YAML语法
python -c "import yaml; yaml.safe_load(open('config/full_ceas08_gpt41mini_v1.yaml'))"

# 加载配置测试
poetry run python -c "
from src.config.config_manager import load_config
config = load_config('config/full_ceas08_gpt41mini_v1.yaml')
print(f'实验名称: {config.name}')
print(f'Groups: {config.data.n_groups}')
print(f'Sample size: {config.data.sample_size_per_group}')
print(f'Synthetic ratios: {config.data.synthetic_ratios}')
"
```

---

## 📝 执行步骤（准备好后执行）

### Step 1: 数据预处理
```bash
poetry run python scripts/step1_data_preprocessing.py \
  --config config/full_ceas08_gpt41mini_v1.yaml
```

### Step 2: LLM生成（分3次执行）
```bash
# Original prompt
poetry run python scripts/step2_llm_generation.py \
  --config config/full_ceas08_gpt41mini_v1.yaml \
  --prompt original

# Strong prompt
poetry run python scripts/step2_llm_generation.py \
  --config config/full_ceas08_gpt41mini_v1.yaml \
  --prompt strong

# Weak prompt
poetry run python scripts/step2_llm_generation.py \
  --config config/full_ceas08_gpt41mini_v1.yaml \
  --prompt weak
```

### Step 3: 数据集构建（4个策略）
```bash
for strategy in within_group cross_group real_fixed_random_synthetic full_random; do
  poetry run python scripts/step3_dataset_construction.py \
    --config config/full_ceas08_gpt41mini_v1.yaml \
    --prompt original \
    --strategy $strategy
done
```

### Step 4: 分类实验（4个策略，顺序执行避免冲突）
```bash
for strategy in within_group cross_group real_fixed_random_synthetic full_random; do
  poetry run python scripts/step4_classification.py \
    --config config/full_ceas08_gpt41mini_v1.yaml \
    --strategy $strategy \
    --resume
done
```

### Step 5: 统计分析
```bash
poetry run python scripts/step5_enhanced_statistical_analysis.py \
  --config config/full_ceas08_gpt41mini_v1.yaml
```

### Step 6: 可视化
```bash
poetry run python scripts/step6_enhanced_visualization.py \
  --config config/full_ceas08_gpt41mini_v1.yaml \
  --results output/full_experiments/ceas08_gpt41mini/results/full_ceas08_gpt41mini_v1_classification_results.json
```

---

## ⚠️ 风险提示与建议

### 1. API稳定性
- ✅ 已配置: batch_size=5（降低并发）
- ✅ 已配置: max_retries=5（增加重试）
- ✅ 已配置: retry_delay=2.0（延长等待）
- ✅ 已配置: save_checkpoint_every=50（频繁保存）

### 2. 长时间运行
- 建议使用 `tmux` 或 `screen` 会话
- 监控日志: `tail -f logs/full_experiments/ceas08_gpt41mini/*.log`
- 定期检查进度和错误

### 3. 成本控制
- 先确认OpenAI账户余额充足（建议$500+）
- 可先用1个prompt测试完整流程
- 监控API使用量: https://platform.openai.com/usage

### 4. 数据备份
```bash
# Step 2完成后立即备份synthetic数据
tar -czf synthetic_backup_$(date +%Y%m%d).tar.gz \
  data/full_experiments/ceas08_gpt41mini/synthetic/

# Step 4完成后备份结果
tar -czf results_backup_$(date +%Y%m%d).tar.gz \
  output/full_experiments/ceas08_gpt41mini/results/
```

---

## ✅ 最终检查清单

在开始正式实验前，请确认：

- [ ] OpenAI API key已配置且余额充足（>$500）
- [ ] 网络连接稳定
- [ ] 磁盘空间充足（>5GB）
- [ ] 内存充足（建议32GB+）
- [ ] 已阅读并理解所有执行步骤
- [ ] 已创建tmux/screen会话（长时间运行）
- [ ] 已设置日志监控
- [ ] 已准备数据备份方案
- [ ] 配置文件已验证无误
- [ ] 理解预估时间（2-3天）和成本（$200-300）

---

## 📚 参考文档

- **实验设计**: `README_experiment.md`
- **理论基础**: `paper/discussion_real.md`
- **Pilot结果**: `output/plots/pilot_ceas08_v1_*.png`
- **配置文件**: `config/full_ceas08_gpt41mini_v1.yaml`

---

**准备好后，请告知即可开始实验！** 🚀
