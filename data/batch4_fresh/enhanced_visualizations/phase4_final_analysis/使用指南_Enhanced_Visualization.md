# 批次4增强版可视化使用指南

*生成日期: 2025-07-27 13:24:51*

## 🎯 快速开始

### 主要可视化文件
```
📂 enhanced_visualizations/
├── 📊 2D交互可视化
│   ├── batch4_interactive_2d.html          # 主要2D图表
│   └── batch4_distance_analysis.html       # 距离分析
├── 🌐 3D交互可视化
│   ├── batch4_interactive_dashboard.html   # ⭐ 推荐主图
│   ├── batch4_main_3d_interactive.html     # 标准3D图
│   ├── batch4_multi_view_3d.html          # 多视图对比
│   └── batch4_distance_slider.html         # 距离滑块
└── 📈 静态图表
    ├── unified_2d_scatter_static.png       # 统一散点图
    └── layer_comparison_matrix.png         # 层间对比
```

### 推荐使用流程
1. **初次探索**: 打开 `batch4_interactive_dashboard.html`
2. **深入分析**: 使用 `batch4_multi_view_3d.html`
3. **具体验证**: 查看 `batch4_distance_analysis.html`
4. **演示展示**: 使用 `batch4_main_3d_interactive.html`

---

## 🔍 功能详解

### 3D交互仪表板 (推荐)
**文件**: `batch4_interactive_dashboard.html`

#### 坐标系统
- **X轴**: t-SNE Component 1 (embedding第一维)
- **Y轴**: t-SNE Component 2 (embedding第二维)  
- **Z轴**: Distance to Centroid (到恶意质心距离)

#### 颜色编码
- 🔵 **蓝色**: real_malicious (真实恶意)
- 🟢 **绿色**: real_benign (真实良性)
- 🔴 **红色**: core_synthetic (核心层合成)
- 🟠 **橙色**: inner_synthetic (内层合成)
- 🟣 **紫色**: outer_synthetic (外层合成)
- 🟡 **黄色**: edge_synthetic (边缘层合成)

#### 交互操作
1. **3D导航**:
   - 左键拖拽: 旋转视角
   - 滚轮: 缩放
   - 右键拖拽: 平移视角

2. **动态过滤** (左上角下拉菜单):
   - All Layers: 显示所有数据
   - Real Data Only: 仅显示真实数据
   - Synthetic Data Only: 仅显示合成数据
   - Core Layer Only: 仅显示核心层
   - Edge Layer Only: 仅显示边缘层
   - Core vs Edge: 对比核心和边缘层

3. **详细信息**: 鼠标悬停查看样本详情

### 多视图对比
**文件**: `batch4_multi_view_3d.html`

#### 四个子图功能
1. **左上**: 全数据默认视图
2. **右上**: Core vs Edge层对比
3. **左下**: 真实vs合成恶意对比
4. **右下**: Prompt类型对比

#### 使用场景
- 科学论文插图
- 多角度数据分析
- 假设验证对比

### 2D分析图表
**文件**: `batch4_interactive_2d.html`

#### 四个分析面板
1. **PCA可视化**: 主成分分析结果
2. **t-SNE可视化**: 非线性降维结果
3. **距离分布**: 各层距离直方图
4. **Prompt分析**: 生成策略效果

---

## 🔬 科学分析指南

### 层间距离验证
观察Z轴(距离)的层次关系:
```
期望趋势: Core < Inner < Outer < Edge
实际结果: 0.538 < 0.551 < 0.596 < 0.660 ✅
```

### 数据质量评估
1. **聚类清晰度**: 观察同色点聚集程度
2. **层间分离**: 不同层在3D空间的分布
3. **异常检测**: 远离主体的离群点

### Prompt效果对比
在多视图的右下角子图中:
- **rewrite**: 基础重写效果
- **rewrite_strong**: 强化效果(应更分散)
- **rewrite_weak**: 弱化效果(应更集中)

---

## 🛠️ 故障排除

### 常见问题

#### 1. 页面加载缓慢
**原因**: 11,200个数据点的3D渲染
**解决**: 
- 使用现代浏览器(Chrome/Firefox推荐)
- 确保显卡驱动更新
- 关闭其他占用内存的程序

#### 2. 交互响应慢
**原因**: 硬件性能限制
**解决**:
- 使用过滤功能减少显示点数
- 降低浏览器缩放比例
- 关闭后台应用程序

#### 3. hover信息不显示
**原因**: JavaScript被禁用或版本过旧
**解决**:
- 启用JavaScript
- 更新浏览器到最新版本
- 检查浏览器插件冲突

#### 4. 3D图表显示异常
**原因**: WebGL支持问题
**解决**:
- 在浏览器地址栏输入 `chrome://flags/` (Chrome)
- 启用硬件加速
- 更新显卡驱动

### 性能优化建议

#### 大数据量处理
- 使用过滤功能聚焦特定层
- 优先使用2D可视化做初步分析
- 保存关键发现的截图

#### 演示优化
- 预先加载页面避免现场等待
- 准备好过滤器设置
- 使用全屏模式提高视觉效果

---

## 📊 数据结构说明

### 主数据表字段
```
unified_embedding_data.csv (11,200行)
├── text: 原始文本内容
├── label: 0=良性, 1=恶意
├── original_id: 样本唯一标识
├── data_source: 数据来源层级
├── prompt_type: 生成策略类型
├── distance_to_centroid: 到质心距离
├── pca_x, pca_y: PCA坐标
├── tsne_x, tsne_y: t-SNE坐标
├── density_group: 密度分组
└── ...其他元数据字段
```

### 关键指标含义
- **Distance to Centroid**: 0-1范围，越小越接近恶意样本中心
- **Density Group**: 基于距离的虚拟密度分组
- **Data Source**: 6种类型(真实malicious/benign + 4层synthetic)
- **Prompt Type**: 4种类型(real + 3种rewrite策略)

---

## 🎨 自定义与扩展

### 颜色自定义
如需修改颜色方案，编辑源代码中的 `color_config`:
```python
color_config = {
    'real_malicious': '#1f77b4',    # 改为您需要的颜色
    'real_benign': '#2ca02c',
    'core_synthetic': '#d62728',
    # ... 其他配置
}
```

### 过滤条件扩展
可在 `create_interactive_dashboard` 函数中添加新的过滤按钮:
```python
buttons.append(
    dict(
        label="Custom Filter",
        method="update",
        args=[{"visible": custom_visible_list}]
    )
)
```

### 新指标集成
1. 在Phase 1数据准备中添加新字段
2. 在3D可视化中映射到颜色/大小/形状
3. 在hover信息中显示新指标

---

## 📈 最佳实践

### 探索性分析流程
1. **宏观概览**: 使用"All Layers"模式观察整体分布
2. **层间对比**: 切换到"Core vs Edge"模式验证假设
3. **细节调查**: 使用hover功能检查异常样本
4. **多角度验证**: 切换到多视图模式确认发现

### 科学研究应用
1. **假设验证**: 使用统计功能验证层间差异
2. **质量评估**: 通过距离分布评估生成质量
3. **方法对比**: 使用prompt类型分析优化策略
4. **结果展示**: 选择最清晰的视角制作论文图表

### 演示展示技巧
1. **预设角度**: 旋转到最佳观察角度后开始讲解
2. **逐步展示**: 使用过滤器逐步引导观众注意力
3. **互动环节**: 让观众自己操作hover功能
4. **关键总结**: 突出显示核心科学发现

---

*本指南涵盖了批次4增强版可视化的所有主要功能和使用方法*  
*如有其他问题，请参考技术文档或联系开发团队*
