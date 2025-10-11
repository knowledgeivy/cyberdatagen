# SMOTE实验逻辑修复总结

## 🔴 发现的严重问题

### 原始错误逻辑（已修复）

**SMOTE实验的原始设计存在严重缺陷：不断增加spam总数，改变了spam:ham ratio**

```python
# 错误的逻辑 (旧代码)
n_synthetic_spam = int(n_real_spam * synthetic_ratio / (100 - synthetic_ratio))
training_set = all_real_data + synthetic_data  # 不断增加！

# 结果：
# 0%:   100 spam + 900 ham = 1000 samples (spam ratio = 10%)
# 10%:  111 spam + 900 ham = 1011 samples (spam ratio = 11%) ❌
# 50%:  200 spam + 900 ham = 1100 samples (spam ratio = 18%) ❌
# 100%: 1100 spam + 900 ham = 2000 samples (spam ratio = 55%) ❌
```

**这导致SMOTE测试的是错误的问题：**
- GPT/Claude实验：测试"用synthetic替换real能否保持性能"
- SMOTE错误实验：测试"增加训练数据能否提升性能"（完全不同的研究问题！）

---

## ✅ 正确的逻辑（已修复）

### GPT/Claude实验的逻辑（参考标准）

```python
# GPT/Claude实验 (dataset_builder.py line 184-186)
total_spam_needed = len(real_spam_pool)  # 固定 = 100
synthetic_count = int(total_spam_needed * synthetic_ratio / 100)
real_count = total_spam_needed - synthetic_count

# 结果：
# 0%:   100 real + 0 synthetic = 100 spam + 900 ham = 1000 samples ✓
# 10%:  90 real + 10 synthetic = 100 spam + 900 ham = 1000 samples ✓
# 50%:  50 real + 50 synthetic = 100 spam + 900 ham = 1000 samples ✓
# 100%: 0 real + 100 synthetic = 100 spam + 900 ham = 1000 samples ✓
```

### 修复后的SMOTE逻辑（现在一致）

```python
# 修复后的SMOTE (smote_generator.py line 199-206)
total_spam_needed = n_total_real_spam  # 固定 = 100
synthetic_count = int(total_spam_needed * synthetic_ratio / 100)
real_count = total_spam_needed - synthetic_count

# 步骤：
# 1. 从real spam中采样 real_count 个
# 2. SMOTE生成synthetic pool，从中采样 synthetic_count 个
# 3. 合并：real_count spam + synthetic_count spam + all ham
# 4. 训练集大小固定 = 1000
```

---

## 📊 修复验证

### 测试结果（运行 `scripts/test_smote_logic.py`）

```
Testing SMOTE dataset construction:
--------------------------------------------------------------------------------
Ratio      Train Size   Spam Count   Ham Count    Spam Ratio
--------------------------------------------------------------------------------
0%         1000         100          900          0.1000
10%        1000         100          900          0.1000
20%        1000         100          900          0.1000
50%        1000         100          900          0.1000
100%       1000         100          900          0.1000
--------------------------------------------------------------------------------

Verification Checks:
✓ Check 1: Training set size is constant (1000 samples)
✓ Check 2: Spam count is constant (100 samples)
✓ Check 3: Spam ratio is constant (~0.1000)
✓ Check 4: Synthetic ratio progression
  0% synthetic: 100 real + 0 synthetic = 100 total spam
  10% synthetic: 90 real + 10 synthetic = 100 total spam
  20% synthetic: 80 real + 20 synthetic = 100 total spam
  50% synthetic: 50 real + 50 synthetic = 100 total spam
  100% synthetic: 0 real + 100 synthetic = 100 total spam

✓ ALL CHECKS PASSED!
```

---

## 📝 修改的文件

### 1. `/src/traditional_methods/smote_generator.py`

**修改方法：`build_training_dataset()`**

**关键改动：**
- Line 199-206: 固定spam总数，计算real/synthetic分配
- Line 239-254: 从real spam中采样指定数量
- Line 256-307: SMOTE生成synthetic pool并采样指定数量
- Line 309-334: 合并所有部分并返回

**核心逻辑：**
```python
# 固定spam总数
total_spam_needed = n_total_real_spam  # 100

# 按比例分配
synthetic_count = int(total_spam_needed * synthetic_ratio / 100)
real_count = total_spam_needed - synthetic_count

# 采样并组合
selected_real = sample(real_spam, real_count)
generated_synthetic = smote_generate_and_sample(synthetic_count)
training_set = selected_real + generated_synthetic + all_ham
```

### 2. 清理的文件

- `output/full_experiments/ceas08_smote/results/*.json` - 所有错误的实验结果已删除

---

## 🔍 审核要点

### 请重点审核以下代码段：

1. **Line 199-206** - spam总数是否固定？
   ```python
   total_spam_needed = n_total_real_spam  # ← 应该固定为100
   synthetic_count = int(total_spam_needed * synthetic_ratio / 100)
   real_count = total_spam_needed - synthetic_count
   ```

2. **Line 240-253** - real spam采样是否正确？
   ```python
   if real_count > 0:
       real_spam_indices = rng.choice(n_total_real_spam, real_count, replace=False)
       X_selected_real_spam = X_real_spam[real_spam_indices]
   ```

3. **Line 295-304** - synthetic采样是否正确？
   ```python
   if synthetic_count <= n_synthetic_pool:
       synthetic_indices = rng.choice(n_synthetic_pool, synthetic_count, replace=False)
       X_synthetic_spam = X_synthetic_spam_pool[synthetic_indices]
   ```

4. **Line 337-338** - 最终spam总数是否恒定？
   ```python
   actual_spam_count = (y_train == 1).sum()  # ← 应该始终 = 100
   ```

5. **Line 353-355** - 日志信息是否清晰？
   ```python
   logger.info(f"Training dataset: {X_train.shape[0]} samples "
              f"({real_count} real spam + {synthetic_count} synthetic spam + {actual_ham_count} ham)")
   logger.info(f"Spam ratio: {metadata['spam_ratio']:.3f} (should be constant across ratios)")
   ```

---

## 🚀 下一步

### 审核完成后需要：

1. ✅ **确认修复正确** - 运行测试脚本验证
   ```bash
   python scripts/test_smote_logic.py
   ```

2. ⏳ **重新运行所有SMOTE实验** （暂停，等待审核）
   ```bash
   # 40个实验任务 (20 groups × 2 strategies)
   # 预计时间：约15分钟（并发运行）
   ./scripts/run_all_smote_experiments_parallel.sh
   ```

3. ⏳ **更新论文表格** - 使用正确的baseline数据

---

## 📌 关键结论

### 修复前后对比

| 指标 | 修复前（错误） | 修复后（正确） | GPT/Claude |
|------|--------------|--------------|------------|
| Training size (0%) | 1000 | 1000 | 1000 |
| Training size (50%) | 1100 ❌ | 1000 ✓ | 1000 ✓ |
| Training size (100%) | 2000 ❌ | 1000 ✓ | 1000 ✓ |
| Spam count (0%) | 100 | 100 | 100 |
| Spam count (50%) | 200 ❌ | 100 ✓ | 100 ✓ |
| Spam count (100%) | 1100 ❌ | 100 ✓ | 100 ✓ |
| Spam ratio | 变化 ❌ | 固定 ✓ | 固定 ✓ |

### 修复后的实验完整性

✅ **现在三个实验测试的是同一个研究问题：**
- "用synthetic spam替换real spam，能否保持分类性能？"
- 控制变量：训练集大小、spam:ham ratio
- 自变量：synthetic:real ratio（0% → 100%）

---

## 📄 附录：完整的代码diff

查看完整修改：
```bash
git diff src/traditional_methods/smote_generator.py
```

或查看当前版本：
```bash
cat src/traditional_methods/smote_generator.py | grep -A 220 "def build_training_dataset"
```
