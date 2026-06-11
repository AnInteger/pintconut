---
title: Training Pipeline
tags:
  - technical
  - training
  - yolo
date: 2026-06-11
status: active
---

# 训练流水线

Pintconut 有两套独立的训练流程，分别训练两个不同粒度的模型：

| 训练目标 | 模型 | 用途 |
|----------|------|------|
| 拼板检测 | YOLOv8-seg（分割） | 从照片中定位拼板区域 |
| 豆子检测 | YOLOv8n（检测） | 从校正后图像中定位单个豆子 |

> [!tip] 两阶段关系
> 先训练拼板检测模型（必须），再训练豆子检测模型（可选增强）。CLI 流水线中两个模型串联使用。

## 拼板检测训练

### 方式一：Gradio Web UI 标注（推荐）

使用 [[labeling-tool-guide|标注工具]] 完成：

1. 启动 `python src/label_ui.py`
2. 上传拼板照片
3. FastSAM 自动分割，人工确认正确的拼板区域
4. 导出 YOLO 分割数据集

输出到 `training/dataset/`。

### 方式二：终端半自动标注

```bash
python training/semi_auto_label.py \
  --input training/photos \
  --output training/dataset
```

操作方式：
- 程序对每张照片运行 FastSAM 分割
- 显示最多 10 个候选区域（按面积排序）
- 输入编号 `0-9` 选择，`s` 跳过，`q` 退出
- 候选预览图保存到 `training/label_previews/`

### 训练模型

```bash
python training/train.py \
  --data training/dataset \
  --model yolov8n-seg.pt \
  --epochs 100 \
  --batch 16 \
  --device cpu
```

#### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | （必填） | 数据集目录路径 |
| `--model` | `yolov8n-seg.pt` | 基础模型权重 |
| `--epochs` | `100` | 训练轮数 |
| `--imgsz` | `640` | 输入图片尺寸 |
| `--batch` | `16` | 批次大小 |
| `--device` | `cpu` | 训练设备（`cpu` 或 `0,1,2...`） |
| `--project` | `training/runs` | 输出目录 |
| `--name` | `beadboard-v1` | 实验名称 |
| — | `single_cls=True` | 固定单类别 |
| — | `patience=50` | 早停耐心值 |

训练完成后最佳权重保存在 `training/runs/beadboard-v1/weights/best.pt`。

### 验证模型

```bash
python training/validate.py \
  --model training/runs/beadboard-v1/weights/best.pt \
  --image training/test_images/ \
  --output training/validation_results
```

对每张测试图片运行推理并保存可视化结果到输出目录。

## 豆子检测训练

### 预标注（HoughCircles）

使用 `src/bead_annotate_ui.py` Gradio 界面：

```bash
python src/bead_annotate_ui.py    # http://localhost:7860
```

操作流程：
1. 上传拼板照片（或从 `training/bead_photos/` 加载）
2. 调整 HoughCircles 参数后点击「🔍 预标注」
3. 查看检测到的珠子（绿色框）
4. 点击「💾 导出数据」保存 YOLO 检测格式标签

#### HoughCircles 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dp` | 1.2 | 分辨率比例，越大越慢但越准确 |
| `min_dist` | 10 | 珠子中心最小距离（像素） |
| `param1` | 100 | Canny 边缘检测高阈值 |
| `param2` | 30 | 累加器阈值，越小检测越多 |
| `min_radius` | 3 | 珠子最小半径 |
| `max_radius` | 30 | 珠子最大半径 |

输出到 `training/bead_dataset/`。

### 训练模型

```bash
python training/bead_train.py \
  --data training/bead_dataset \
  --model yolov8n.pt \
  --epochs 150 \
  --batch 16 \
  --device cpu
```

参数与拼板训练类似，区别：
- 基础模型：`yolov8n.pt`（检测而非分割）
- 默认 epochs：150
- 实验名称：`bead-v1`
- 类别名：`bead`

### 验证模型

```bash
python training/bead_validate.py \
  --model training/runs/bead-v1/weights/best.pt \
  --image training/test_images/ \
  --output training/validation_results \
  --conf 0.5
```

`--conf` 参数控制置信度阈值（默认 0.5）。

## 数据集格式

### 拼板检测：YOLO Segmentation

```
# class x1 y1 x2 y2 x3 y3 ... (归一化坐标)
0 0.123456 0.234567 0.345678 0.456789 ...
```

- `class`：固定为 `0`（beadboard）
- 坐标为多边形顶点的归一化坐标（0.0~1.0）

### 豆子检测：YOLO Detection

```
# class x_center y_center width height (归一化坐标)
0 0.500000 0.500000 0.030000 0.030000
```

- `class`：固定为 `0`（bead）
- 坐标为边界框中心点 + 宽高（归一化）

### 目录结构

拼板数据集：

```
training/dataset/
├── data.yaml              # 自动生成
├── images/
│   ├── train/             # 训练图片
│   └── valid/             # 验证图片（需手动从 train 划分）
└── labels/
    ├── train/             # .txt 分割标签
    └── valid/             # .txt 验证标签
```

豆子数据集：

```
training/bead_dataset/
├── data.yaml              # 自动生成
├── images/
│   ├── train/
│   └── valid/
└── labels/
    ├── train/             # .txt 检测标签
    └── valid/
```

> [!warning] 验证集
> 导出工具只生成 `train/` 目录。需要手动将部分数据移动到 `valid/` 目录（建议 80/20 划分）。`data.yaml` 配置文件会自动指向这两个目录。

## 文件说明

```
training/
├── semi_auto_label.py     # 终端版半自动标注（FastSAM）
├── train.py               # 拼板模型训练（YOLOv8-seg）
├── validate.py            # 拼板模型验证
├── bead_train.py          # 豆子模型训练（YOLOv8n）
├── bead_validate.py       # 豆子模型验证
├── collect_helper.py      # 照片批量重命名工具
├── dataset/               # 拼板训练数据（gitignored）
├── bead_dataset/          # 豆子训练数据（gitignored）
├── photos/                # 原始照片（gitignored）
├── runs/                  # 训练输出（gitignored）
└── test_images/           # 验证用测试图片
```

## 关联文档

- [[product-overview]] — 产品功能概述
- [[labeling-tool-guide]] — Gradio 标注工具操作手册
- [[architecture]] — 系统架构和模块设计
