# 代码重组完成总结

## ✅ 完成的工作

### 1. scripts/目录清理
- **只保留6个文件**：step1-6.py
- **删除**：所有utils、enhanced、improved、combined等额外文件
- **合并**：step6_visualization.py 和 step6_combined_visualization.py → step6_visualization.py（含single和combined模式）

### 2. src/utils/扩展
新增功能模块：
- `batch_processor.py`: 批量运行分析和可视化
- `csv_exporter.py`: 导出详细CSV（含sensitivity和p-value）
- 更新 `__init__.py` 导出所有工具函数

### 3. Sensitivity指标集成
- 在 `src/analysis/statistical_analysis.py` 添加 `_calculate_sensitivity()` 方法
- 自动计算9个sensitivity指标：
  - `regression_slope`: 每1% synthetic的性能变化
  - `regression_r_squared`: 线性拟合度
  - `regression_p_value`: 统计显著性
  - `total_change_0_to_max`: 总变化量
  - 等等...

### 4. CSV导出功能
创建 `src/utils/csv_exporter.py`，支持：
- 导出每个ratio（0-100%）的详细信息
- 包含performance、sensitivity指标、p-value
- 自动合并所有prompt×strategy到一个CSV
- 输出位置：`output/.../reports/csv/`

### 5. Step6可视化合并
`step6_visualization.py` 现在支持两种模式：
- **single模式**：单个分析的图表（curves, heatmap, etc.）
- **combined模式**：多prompt对比图（3行×5列，两个classifier用不同颜色）

### 6. 配置更新
- step4默认strategies只有within_group和cross_group
- 所有输出在output/，日志在logs/
- 批处理通过Python调用，无.sh文件

## 📁 最终结构

```
cyberdata/
├── scripts/                    # 6个核心步骤
│   ├── step1_data_preprocessing.py
│   ├── step2_llm_generation.py
│   ├── step3_dataset_construction.py
│   ├── step4_classification.py      # 只用within_group, cross_group
│   ├── step5_statistical_analysis.py # 自动计算sensitivity
│   └── step6_visualization.py        # single + combined模式
│
├── src/utils/                  # 所有辅助工具
│   ├── result_merger.py        # 合并结果
│   ├── sensitivity_display.py  # 显示sensitivity
│   ├── batch_processor.py      # 批处理
│   └── csv_exporter.py         # CSV导出（新增）
│
├── output/                     # 所有输出
│   └── .../reports/csv/        # CSV格式结果（新增）
│
├── logs/                       # 所有日志
│
├── USAGE.md                    # 使用说明
└── example_workflow.py         # 完整工作流示例
```

## 🔧 使用方式

### 典型工作流

```python
from pathlib import Path
from src.utils import (
    merge_classification_results,
    run_batch_analysis,
    run_batch_visualization,
    export_all_analyses_to_csv,
    create_sensitivity_table
)

# 1. 合并结果
merge_classification_results(...)

# 2. 批量分析（自动包含sensitivity）
run_batch_analysis(...)

# 3. 批量可视化（single + combined）
run_batch_visualization(...)

# 4. 导出CSV
export_all_analyses_to_csv(...)

# 5. 查看sensitivity表
create_sensitivity_table(...)
```

### Step6两种模式

```bash
# Single模式
poetry run python scripts/step6_visualization.py \
  --config config/xxx.yaml \
  --mode single \
  --analysis_file output/.../analysis.json \
  --output_dir output/.../plots/

# Combined模式
poetry run python scripts/step6_visualization.py \
  --config config/xxx.yaml \
  --mode combined \
  --base_dir output/.../ceas08_gpt41mini \
  --strategies within_group cross_group
```

## 📊 Sensitivity结果位置

### 1. JSON格式
位置：`output/.../reports/*_statistical_analysis.json`

在`performance_degradation[classifier][metric]['sensitivity']`字段

### 2. CSV格式（新增）
位置：`output/.../reports/csv/`

包含：
- 每个ratio的performance
- Sensitivity指标（slope, R², p-value）
- Hypothesis test p-value
- Degradation信息

使用：
```python
from src.utils import export_all_analyses_to_csv
from pathlib import Path

export_all_analyses_to_csv(
    base_dir=Path('output/full_experiments/ceas08_gpt41mini'),
    experiment_name='full_ceas08_gpt41mini_v1',
    prompts=['original', 'strong', 'weak'],
    strategies=['within_group', 'cross_group']
)
```

## 📝 重要变更

1. ✅ scripts/只有step1-6，无其他文件
2. ✅ 所有辅助功能在src/utils/
3. ✅ step6合并为一个文件（两种模式）
4. ✅ 新增CSV导出功能
5. ✅ Sensitivity自动计算并集成
6. ✅ 默认只用within_group和cross_group
7. ✅ 所有输出在output/，日志在logs/

