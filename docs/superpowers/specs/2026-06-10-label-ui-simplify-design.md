# 板子标注 UI 精简设计

> 日期：2026-06-10
> 范围：`src/label_ui.py`
> 前序：替换 `2026-06-10-label-ui-improvements-design.md`

## 背景

Playwright 测试发现两个严重问题：

1. **候选按钮完全不显示** — 10 个候选按钮用 `visible=False` 创建，Gradio 6.x 不渲染到 DOM，`gr.update(visible=True)` 无效。标注流程完全无法工作。
2. **"自动分割"Tab 多余** — 用户需要先切换到 Tab 2 手动点击分割，再到 Tab 3 标注，增加了不必要的操作步骤。

## 改动总览

**从 4-Tab 简化为 2-Tab：**

```
当前 (4-Tab)                    新 (2-Tab)
┌──────────────────┐           ┌──────────────────┐
│ ① 上传照片       │           │ ① 上传照片       │
│ ② 自动分割       │  ──→     │ ② 标注确认       │
│ ③ 标注确认       │           │   (含分割+导出)   │
│ ④ 导出数据集     │           └──────────────────┘
└──────────────────┘
```

## 改动 1：删除 Tab ② "自动分割"和 Tab ④ "导出数据集"

**删除的代码：**
- Tab ② "自动分割" 的整个 UI 块（`gr.Tab("② 自动分割")`）
- Tab ④ "导出数据集" 的整个 UI 块（`gr.Tab("④ 导出数据集")`）
- "下一步 →" 导航按钮（Tab 1 和 Tab 2 之间的 `nav_next_1`、`nav_next_2`）
- `_js_goto_tab()` JS 辅助函数
- `_nav_dummy` 隐藏文本框
- `BUTTON_COLORS` 常量
- `_build_button_css()` 函数
- CSS 传入 `launch()` 的参数

**保留的逻辑：**
- `_handle_segment()` 的分割核心代码 → 改为在"开始标注"按钮回调中调用
- `_segmentation_cache` → 保留，批量分割后缓存所有结果
- `_handle_export()` → 保留，直接在 Tab ② 内的导出按钮调用

**分割时机变更：** 从"用户手动在 Tab 2 点击分割"改为"点击 Tab ② 的'开始标注'按钮后自动批量分割"。

## 改动 2：候选区域选择改为 Radio 单选组

**替换 10 个独立按钮（`visible=False` bug）为 1 个 `gr.Radio` 组件。**

**组件生命周期：**

1. 初始状态：`gr.Radio(choices=[], interactive=False, label="候选区域")`
2. 分割完成后加载图片时：`gr.update(choices=["#0 面积 45.2%", "#1 面积 32.1%", ...], interactive=True)`
3. 用户选择一个选项 → 触发回调，展示绿色轮廓预览
4. 预览时：Radio 变为 `interactive=False`（防止重复选择），显示确认/重新选择按钮
5. 重新选择时：Radio 恢复 `interactive=True`

**颜色对应方式：**
- 图片上已有彩色编号标签（`draw_candidates()` 的白色 pill + 彩色边框 + 黑色编号）
- Radio 选项文字包含编号：`"#0 面积 45.2%"`
- 用户对照编号选择

**删除的代码：**
- 10 个 `cand-btn-{i}` 按钮变量和循环创建逻辑
- `BUTTON_COLORS` / `_build_button_css()`
- `step3_outputs` 中 10 个按钮的位置 → 替换为 1 个 Radio

**候选数量不再限制为 10 个**，由 `segment_image()` 返回数量决定。

## 改动 3：交互流程重设计

### Tab ① 上传照片

- 上传完成后显示"▶ 开始标注"按钮（初始 `interactive=False`，上传后激活）
- 点击后切换到 Tab ② 并开始分割

### Tab ② 标注确认

**点击"开始标注"后的流程：**

1. 批量分割所有图片（带进度条 "正在分割 1/3..."）
2. 展示第 1 张图片：左侧图片（带彩色区域覆盖 + 编号标签），右侧 Radio 选择候选
3. 用户选择候选 → 展示绿色轮廓预览 + "确认，下一张"/"重新选择"按钮
4. 确认 → 保存标注，自动下一张
5. 跳过 → 标记为跳过，自动下一张
6. 全部完成 → 展示"📦 导出数据集"按钮 + 点击后展示数据集报告

**按钮状态控制：**

| 按钮 | 初始 | 选择候选后 | 确认/跳过后 | 全部完成后 |
|------|------|-----------|-------------|-----------|
| Radio | interactive=True | interactive=False | interactive=True (下一张) | — |
| ⏭️ 跳过 | interactive=True | interactive=False | interactive=True | interactive=False |
| ✅ 确认 | interactive=False | interactive=True | interactive=False | interactive=False |
| ↩️ 重新选择 | interactive=False | interactive=True | interactive=False | interactive=False |
| 📦 导出 | interactive=False | interactive=False | interactive=False | interactive=True |

## 不改动的内容

- `src/label_service.py` — 核心分割、绘制、保存逻辑不变
- `draw_candidates()` — 图片上的彩色区域覆盖和编号标签不变
- `draw_result()` — 绿色轮廓预览不变
- `save_label()` / `generate_dataset_yaml()` — 数据导出逻辑不变
- `_segmentation_cache` — 缓存机制不变
- `_annotation_state` / `_pending_selection` — 状态管理不变

## 涉及文件

| 文件 | 改动类型 |
|------|---------|
| `src/label_ui.py` | 大幅重构：删除 2 个 Tab，替换 10 个按钮为 Radio，重写流程 |
| `tests/test_label_service.py` | 不变 |
| `src/label_service.py` | 不变 |
