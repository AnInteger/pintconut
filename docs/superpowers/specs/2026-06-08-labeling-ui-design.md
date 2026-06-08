# Pintconut 标注工具 UI 设计文档

> 日期: 2026-06-08
> 状态: 已确认

## 1. 概述

为拼豆差异检测器的训练数据标注流程提供一个 Web UI 工具。采用 Gradio 框架，以向导式流程引导用户完成从上传照片到导出数据集的完整标注工作流。

## 2. 用户流程

```
① 上传照片 → ② 自动分割 → ③ 逐张确认 → ④ 导出数据集
```

### 步骤 ① 上传照片

- 用户通过 Gradio File 组件上传多张拼板照片
- 支持拖拽上传和文件选择
- 上传后保存到 `training/photos/` 目录
- 显示已上传图片数量

### 步骤 ② 自动分割

- 用户点击"开始分割"按钮
- 后端使用 FastSAM 对所有照片进行批量分割
- 显示处理进度（已完成/总数）
- 为每张照片生成候选区域列表
- 完成后自动进入步骤③

### 步骤 ③ 逐张确认

- 左侧展示当前照片，候选区域用彩色边框标注
- 右侧展示候选区域列表（编号、面积、矩形度）
- 用户点击选择正确的拼板区域
- 操作按钮：确认选择 / 跳过此图
- 显示进度：已标注/总数
- 确认后自动加载下一张

### 步骤 ④ 导出数据集

- 显示标注统计：已标注数、跳过数
- 点击"导出"按钮生成 YOLO 格式数据集
- 自动生成目录结构和 data.yaml
- 输出到 training/dataset/ 目录
- 提供"开始训练"按钮，可直接启动训练

## 3. 技术方案

### 框架

- **Gradio** — Python Web UI 框架
- 使用 `gr.Tabs` 或自定义状态管理实现向导步骤切换
- 使用 `gr.Gallery` 或 `gr.Image` 展示图片
- 使用 `gr.Button` 实现交互操作

### 复用逻辑

从现有模块中提取核心逻辑：

| 功能 | 复用来源 | 提取为 |
|------|---------|--------|
| FastSAM 分割 | `training/semi_auto_label.py` | 共享函数 |
| 候选区域筛选 | `training/semi_auto_label.py` | `is_rectangular()` |
| YOLO 标签生成 | `training/semi_auto_label.py` | `_save_label()` |
| data.yaml 生成 | `training/train.py` | `generate_data_yaml()` |

### 项目结构

```
pintconut/
├── src/
│   ├── label_ui.py           # Gradio Web UI 入口
│   └── label_service.py      # 标注核心逻辑（从 semi_auto_label.py 提取）
├── training/
│   ├── semi_auto_label.py    # [已存在] 终端版标注工具
│   └── ...
```

- `src/label_service.py` — 纯逻辑层：分割、筛选、标签生成，不含 UI 代码
- `src/label_ui.py` — Gradio UI 层：调用 label_service 完成标注工作流

## 4. 依赖

- `gradio>=4.0.0` — Web UI 框架（新增）

## 5. 运行方式

```bash
source .venv/bin/activate && pip install gradio
python src/label_ui.py
```

启动后浏览器自动打开 `http://localhost:7860`
