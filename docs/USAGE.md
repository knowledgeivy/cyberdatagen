# 使用说明

## 目录结构

```
cyberdata/
├── scripts/                    # 核心实验步骤（Step 1-6，共6个文件）
│   ├── step1_data_preprocessing.py
│   ├── step2_llm_generation.py
│   ├── step3_dataset_construction.py
│   ├── step4_classification.py
│   ├── step5_statistical_analysis.py
│   └── step6_visualization.py  # 包含single和combined模式
├── src/utils/                  # 辅助工具
│   ├── result_merger.py        # 合并结果
│   ├── sensitivity_display.py  # 显示sensitivity
│   ├── batch_processor.py      # 批处理
│   └── csv_exporter.py         # CSV导出
├── output/                     # 所有输出数据
│   └── .../reports/csv/        # CSV格式的详细结果
└── logs/                       # 所有日志文件
```

## 核心实验步骤（scripts/）

### Step 1-4: 数据处理和分类

```bash
# Step 1: 数据预处理
poetry run python scripts/step1_data_preprocessing.py --config config/xxx.yaml

# Step 2: LLM生成合成数据
poetry run python scripts/step2_llm_generation.py --config config/xxx.yaml

# Step 3: 构建数据集
poetry run python scripts/step3_dataset_construction.py --config config/xxx.yaml

# Step 4: 分类实验（默认只用within_group和cross_group）
poetry run python scripts/step4_classification.py \
    --config config/xxx.yaml \
    --strategy within_group \
    --prompt original
```

### Step 5: 统计分析（含sensitivity指标）

单个分析：
```bash
poetry run python scripts/step5_statistical_analysis.py \
    --config config/xxx.yaml \
    --results_file output/.../results.json \
    --output_file output/.../analysis.json \
    --filter_prompt original \
    --filter_strategy within_group
```

批量分析（所有prompt×strategy组合）：
```python
from src.utils import run_batch_analysis

run_batch_analysis(
    config_file='config/xxx.yaml',
    results_file='output/.../results.json',
    reports_dir='output/.../reports',
    experiment_name='xxx'
)
```

### Step 6: 可视化

**Single模式**（单个分析）：
```bash
poetry run python scripts/step6_visualization.py \
    --config config/xxx.yaml \
    --mode single \
    --analysis_file output/.../analysis.json \
    --output_dir output/.../plots/
```

**Combined模式**（多prompt对比）：
```bash
poetry run python scripts/step6_visualization.py \
    --config config/xxx.yaml \
    --mode combined \
    --experiment_name xxx \
    --base_dir output/.../ceas08_gpt41mini \
    --strategies within_group cross_group
```

批量可视化（Python）：
```python
from src.utils import run_batch_visualization

run_batch_visualization(
    config_file='config/xxx.yaml',
    base_dir='output/.../ceas08_gpt41mini',
    experiment_name='xxx'
)
```

## 辅助工具（src/utils/）

### 1. CSV导出

导出详细的sensitivity和p-value数据到CSV：

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

输出位置：`output/.../reports/csv/`

CSV包含列：
- `synthetic_ratio`: 0%, 10%, ..., 100%
- `performance`: 该ratio下的性能
- `regression_slope`: Sensitivity斜率
- `regression_r_squared`: R²值
- `regression_p_value`: Regression p值
- `p_value`: Hypothesis test p值（每个ratio与baseline比较）
- `absolute_degradation`: 绝对性能下降
- `relative_degradation`: 相对性能下降

### 2. 合并结果

```python
from src.utils import merge_classification_results

merge_classification_results(
    output_dir='output/.../results',
    experiment_name='xxx',
    strategies=['within_group', 'cross_group'],
    total_groups=20
)
```

### 2. 显示sensitivity

```python
from src.utils import display_sensitivity_analysis, create_sensitivity_table
from pathlib import Path

# 显示单个文件
display_sensitivity_analysis(
    Path('output/.../analysis.json'),
    show_all_metrics=False
)

# 显示对比表
create_sensitivity_table(
    Path('output/.../ceas08_gpt41mini'),
    prompts=['original', 'strong', 'weak'],
    strategies=['within_group', 'cross_group']
)
```

## 典型工作流

```python
from pathlib import Path
from src.utils import (
    merge_classification_results,
    run_batch_analysis,
    run_batch_visualization,
    create_sensitivity_table
)

# 1. 合并结果
merge_classification_results(
    output_dir='output/full_experiments/ceas08_gpt41mini/results',
    experiment_name='full_ceas08_gpt41mini_v1',
    strategies=['within_group', 'cross_group']
)

# 2. 批量分析
run_batch_analysis(
    config_file='config/full_ceas08_gpt41mini_v1.yaml',
    results_file='output/full_experiments/ceas08_gpt41mini/results/full_ceas08_gpt41mini_v1_classification_results.json',
    reports_dir='output/full_experiments/ceas08_gpt41mini/reports',
    experiment_name='full_ceas08_gpt41mini_v1'
)

# 3. 批量可视化
run_batch_visualization(
    config_file='config/full_ceas08_gpt41mini_v1.yaml',
    base_dir='output/full_experiments/ceas08_gpt41mini',
    experiment_name='full_ceas08_gpt41mini_v1'
)

# 4. 查看sensitivity
create_sensitivity_table(
    Path('output/full_experiments/ceas08_gpt41mini'),
    prompts=['original', 'strong', 'weak'],
    strategies=['within_group', 'cross_group']
)
```

## Sensitivity指标说明

每个分析报告自动包含sensitivity指标：
- `regression_slope`: 每1% synthetic的性能变化
- `regression_r_squared`: 线性拟合度（0-1）
- `regression_p_value`: 统计显著性
- `total_change_0_to_max`: 总变化量

解读：
- 负斜率：性能随synthetic比例增加而下降
- 绝对值大：对synthetic data更敏感
- R²接近1：线性关系强

## 默认设置

- **Sampling strategies**: 只使用 `within_group` 和 `cross_group`
- **Prompts**: `original`, `strong`, `weak`
- **输出目录**: `output/`
- **日志目录**: `logs/`
