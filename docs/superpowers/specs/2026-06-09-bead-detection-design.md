# Bead Detection System Design

**Date**: 2026-06-09
**Status**: Approved

## Overview

从"均匀网格切割"方案改为"豆子检测模型"方案。利用拼豆形状高度统一（圆形、大小一致）的特性，训练 YOLO 目标检测模型直接定位每颗豆子，再通过 LAB 颜色匹配识别颜色。

## Background

原有的网格切割方案（透视矫正 → 29×29 均匀网格 → 采样颜色）在验证中发现问题：
- 手持拍摄存在镜头畸变（桶形/枕形）
- 仅做透视矫正无法消除光学畸变
- 中心区域对齐较好，边缘区域偏移 1/4~1/2 个格子宽度
- 相机标定方案要求固定手机/镜头，不适用随意拍摄场景

选择直接检测豆子本身，因为拼豆形状具有高度相似性，适合目标检测。

## Requirements

- **定位**：检测每颗豆子的位置 (x, y, w, h)
- **颜色识别**：识别每颗豆子的颜色
- **计数**：统计各色豆子的数量
- **场景**：手持随意拍摄，角度/光线/背景不固定
- **标注方式**：预标注 + 人工修正

## Architecture

```
拍照
 ↓
[Step 1] 板子检测（现有 YOLO beadboard 模型）
 ↓ 输出：板子区域 mask
[Step 2] 透视矫正（现有 PerspectiveCorrector）
 ↓ 输出：矫正后的板子正面图
[Step 3] 豆子检测（新训练 YOLOv8n，1 类：bead）
 ↓ 输出：每颗豆子的 (x, y, w, h) + 置信度
[Step 4] 排除干扰（板内过滤 + 尺寸过滤 + 置信度阈值）
 ↓ 输出：干净的豆子列表
[Step 5] 颜色识别（现有 ColorMatcher，LAB 匹配）
 ↓ 输出：每颗豆子 → 颜色名
[Step 6] 结果输出（位置 + 颜色 + 计数统计）
```

Step 1-2 保持现有代码不变，Step 3-6 是新增的，替换原来的均匀网格切割。

## Component Details

### 1. Board Detection (Step 1-2, existing)

保持现有的板子检测 + 透视矫正流程。先定位板子可以：
- 大幅缩小豆子检测的搜索空间
- 排除桌面、手、杂物等干扰
- 透视矫正后豆子形状更规则，利于检测

训练数据需求：50-100 张照片，混合空板和有豆子的板子，不同角度/光线/背景。

### 2. Bead Detection Model (Step 3)

**模型**：YOLOv8n（Nano 检测版，非分割版）

**类别**：1 个类 `bead`

**为什么用检测框而非分割 mask**：
- 豆子是圆形，边界框足以定位
- 框标注比 mask 标注快很多
- 框中心点就是豆子位置，足够做颜色采样

**输入**：透视矫正后的板子正面图（600×600）

**输出**：N 个检测框，每个 `[x_center, y_center, width, height, confidence]`

**为什么颜色不交给 YOLO 分 20 类**：
- 标注成本：只需标"是豆子"，不用标颜色
- 光线影响：LAB 空间比 RGB 更抗光照变化
- 扩展性：新增颜色只需改 colors.json，不用重训模型
- 稳定性：数学计算比模型推理可预测

### 3. Post-processing (Step 4)

检测后需要过滤干扰：

| 过滤规则 | 说明 |
|---------|------|
| 板子区域内 | 只保留在板子 mask 内的检测框 |
| 尺寸过滤 | 豆子大小一致，过滤明显过大/过小的框 |
| 置信度阈值 | 过滤低置信度检测 |
| 密度检查 | 拼板区域应密集均匀，孤立框可能是误检 |

### 4. Color Recognition (Step 5)

复用现有 `ColorMatcher`：

1. 从检测框中取中心 40% 区域（避开边缘，减少背景干扰）
2. 计算该区域的 RGB 中值
3. `ColorMatcher.match()` 在 LAB 空间匹配最近色板颜色
4. `ColorMatcher.is_bead()` 判断是豆子还是空板

现有 20 色色板 (`data/colors.json`)，后续可扩展——加颜色不用动模型。

### 5. Result Output (Step 6)

输出格式：
- 每颗豆子：`(row, col, color_name, rgb, confidence)`
- 统计：各色豆子数量
- 可视化：在原图上标注检测框和颜色名

## Annotation Tool Design

### Pre-labeling Pipeline

```
原始照片
  → 板子检测模型（Phase 1 训练好的）
  → 透视矫正
  → HoughCircles 自动检测圆形
  → 生成初始 YOLO 框标注
  → 用户在工具里修正/确认
  → 导出 YOLO 格式训练数据
```

选择 HoughCircles 预标注的原因：
- 拼豆是圆形 + 大小一致 + 排列规则
- 在矫正后的图上效果应该不错
- 预计可覆盖 80% 的豆子，用户只需修正剩余部分

### Tool Features

Gradio 标注工具（类似现有 label_ui.py）：

1. 加载图片 → 自动运行 HoughCircles 预标注
2. 显示预标注结果 → 绿框标出检测到的豆子
3. 用户操作：
   - 点击误检的框 → 删除
   - 点击漏检的位置 → 添加框
   - 调整框大小
4. 确认保存 → 导出 YOLO 格式标注文件

## Dataset Plan

### Board Detection Dataset (Phase 1)

| 项目 | 规格 |
|------|------|
| 图片数量 | 50-100 张 |
| 标注类型 | 分割 mask |
| 类别 | 1 类（beadboard） |
| 划分 | 80/20 train/valid |
| 拍摄建议 | 空板 10-15 张 + 有豆子的板子 20-30 张，不同角度/光线/背景 |

### Bead Detection Dataset (Phase 2)

| 项目 | 规格 |
|------|------|
| 图片数量 | 30-50 张（矫正后的板子图） |
| 标注类型 | 检测框 |
| 类别 | 1 类（bead） |
| 划分 | 80/20 train/valid |

标注格式（YOLO detection）：
```
0 0.523 0.341 0.032 0.032
# class x_center y_center width height
```

目录结构：
```
training/bead_dataset/
├── data.yaml
├── images/
│   ├── train/
│   └── valid/
└── labels/
    ├── train/
    └── valid/
```

## Implementation Phases

```
Phase 1：板子检测
  ├── 收集 50-100 张照片（空板 + 有豆子，多场景）
  ├── 用现有 label_ui.py 标注板子轮廓
  ├── 训练 YOLOv8n-seg（1 类：beadboard）
  └── 测试验证，bad case 加回训练集

Phase 2：豆子检测
  ├── 用训练好的板子模型做透视矫正
  ├── 开发豆子预标注工具（HoughCircles + Gradio）
  ├── 预标注 + 人工修正，产出 30-50 张标注数据
  ├── 训练 YOLOv8n（1 类：bead）
  └── 测试验证

Phase 3：整合流水线
  ├── 板子检测 → 透视矫正 → 豆子检测 → 颜色识别
  └── 对比蓝图输出差异报告
```
