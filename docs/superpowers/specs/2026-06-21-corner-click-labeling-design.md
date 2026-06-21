# 角点定位标注设计文档

> 日期: 2026-06-21
> 状态: Draft
> 涉及模块: 新增 `src/bead_label_service.py` 的 `generate_grid_boxes` / `preview_box_colors`；重写 `src/bead_annotate_ui.py` 的「位置生成」环节
> 取代: 标注 UI 中失败的 HoughCircles 预标注（`prelabel`）作为位置来源
> 关联: `2026-06-21-bead-detect-training-loop-design.md`（其「评估关卡」已触发：HoughCircles 在真实照片不可靠）

---

## 1. 背景：自动检测在真实照片上全军覆没

标注 UI 原计划用 bead-grid 的 HoughCircles 自动预标注豆子位置。实测在 `training/photos/` 真实照片上**完全失败**：

- **根因（已 visual+数据双重确认）**：`detect_beads` 的半径启发式 `max_r=min(h,w)/20≈58-64` 在真实照片上偏差 8-12 倍——画出的圆直径约等于 8-12 颗豆子，锁定的是板子大结构/背景，不是豆子。→「框比豆子大好多」「一个准的点都没有」。
- 即便修正到正确尺度（豆子半径 ~9px），HoughCircles 也只命中 ~60-70%，且尺度估计脆弱（自相关差 2 倍）、`param2` 在部分照片上一颗都检不到。
- **FastSAM（项目已有的预训练模型）也分不出单颗豆子**：实测只产生大色块 mask（1.png 22 个、0 个豆子尺寸），把整片同色豆子当一个区域。

**结论**：目前没有任何「全自动」方案能在真实照片上可靠解析单颗豆子。要全自动，必须有训练好的检测器；要检测器，必须有标签；标签必须有起点。**这个 bootstrap 躲不掉。**

## 2. 目标

用**角点定位 + 几何网格生成**作为标注 bootstrap：用户点 4 颗角豆 + 选板尺寸 → 单应矩阵直接算出全部 rows×cols 颗豆子位置 → 少量校正 → 导出干净 YOLO 标签 → 训练检测器 → 之后真正全自动。

为什么是 4 颗角豆：对人 trivial（人一眼找到 4 颗），对当前算法难；4 个点 + 规则网格几何 → 生成全部 ~800 颗，比通用目标标注或修 HoughCircles（60% 脏标签）都低成本、且标签近乎完美（脏标签会污染训练）。

## 3. 核心流程

1. 加载照片（复用现有 `h_load`）。
2. **点 4 颗角豆**：canvas 依次点击 左上→右上→右下→左下 四颗**豆子中心**（UI 带 ①②③④ 序号提示 + 实时四边形预览）。
3. **选板尺寸**：下拉 `data/board_sizes.json` 预设（6 个）或手填 rows×cols。
4. **生成网格**：4 角豆 + rows×cols → 单应 H → 全部豆位；每颗框按**局部间距**缩放（透视下边角与中心间距不同，逐点算）。
5. 叠加框（`source="generated"`，新颜色 🔵）+ 可选配色预览。
6. **校正**：增/删/移框（复用现有编辑，稳定 id、非变异 state）——角点准时几乎不用改。
7. **导出 YOLO**（复用 `export_yolo`）。

## 4. 几何（精确，已验证）

4 颗角豆对应网格角点（坐标系 x=col, y=row）：

```
左上豆 (r=0,c=0)         → 网格 (0,0)
右上豆 (r=0,c=cols-1)    → 网格 (cols-1, 0)
右下豆 (r=rows-1,c=cols-1) → 网格 (cols-1, rows-1)
左下豆 (r=rows-1,c=0)    → 网格 (0, rows-1)
```

`H = cv2.getPerspectiveTransform(src_quad, dst_quad)`，其中 src=网格四角、dst=用户点的 4 颗角豆图像坐标。任意豆 `(r,c)` 中心 = `H @ [c, r, 1]`（齐次除以 w）。

**已验证**：带透视的合成 10×10 网格，4 角 → 全部 100 颗，还原误差 **0.0000px**（4 点单应对平面板透视数学上精确）。真实照片误差仅来自点击不精 + 板子微翘，人工校正兜底。

**框尺寸**：每颗框半径 = 该点与相邻生成点（右邻、下邻）距离的均值 × 0.4。透视下逐点不同。

## 5. 代码改动

### 5.1 `src/bead_label_service.py`（新增两个纯函数）

- `generate_grid_boxes(corners: np.ndarray, rows: int, cols: int) -> list[dict]`：4 角豆 + 尺寸 → 全部豆位框（`source="generated"`）。用 `_box_from_xy` 产框（复用，schema 一致），半径按局部间距。纯几何、无模型依赖、可单测。
- `preview_box_colors(image, boxes, color_matcher) -> list[dict]`：每个框中心采样色 → 调色板匹配，返回 `{xy, name, rgb}`（与 `match_cell_colors` 同 schema，但因这条路径无 `GridResult`，单独基于 boxes 实现）。

### 5.2 `src/bead_annotate_ui.py`（改位置生成环节）

- **加「角点定位」模式**：canvas `.select` 收集点击点（最多 4 个），存入 `state["corners"]`；带序号提示 + 实时画四边形；`state` 新增 `corners: list`、`rows`、`cols`。
- **板尺寸输入**：`gr.Dropdown`（board_sizes.json 预设的 `"rows x cols"`）+ 两个 `gr.Number`（手填覆盖）。
- **「生成网格」按钮**：corners 满 4 且尺寸有效 → 调 `generate_grid_boxes` → 设 `state["boxes"]`；否则提示。
- **移除 HoughCircles「预标注」按钮**（实测不可靠，留着误导）及 `prelabel`/`holes_to_boxes`/autofill 相关入口（角点定位已生成满网格，无「洞」概念）。
- **配色预览**：`_draw` 改用 `preview_box_colors(image, boxes, cm)` 替代 `match_cell_colors(result, cm)`（因为无 result）。
- **SRC_BGR** 新增 `"generated"` 颜色（🔵 蓝）。
- 复用：`export_yolo`、框编辑（`_stamp_ids`/`_box_label`/`h_delete`/`h_add_click`）、`_draw` 框绘制。

### 5.3 不动的部分
- `src/bead_grid.py`（HoughCircles / `detect_beads` / `fit`）：**保留**，CLI 运行时仍用（等训练好检测器替换）。只是标注 UI 不再调用它。
- `src/bead_detect.py`、`training/bead_train.py` 等：不动。

## 6. 错误处理

| 情况 | 处理 |
|---|---|
| 点击 < 4 颗就按「生成网格」 | 禁用按钮 / 提示「请点满 4 颗角豆」 |
| 板尺寸为 0 或非法 | 提示「请选/填有效板尺寸」 |
| 4 角共线（退化单应） | `getPerspectiveTransform` 会报错 → 捕获，提示「4 颗角豆不能共线」 |
| 生成的框落在图外 | 正常画/导出（YOLO 容忍归一化值略超界）；多数情况下四角都在板内 |

## 7. 测试

- `tests/test_bead_label_service.py` 新增：
  - `generate_grid_boxes` 精确还原：合成带透视网格，取 4 角豆，断言生成的全部中心 ≈ 真实位置（容差 < 1px）；框半径 ≈ 局部间距×0.4。
  - `generate_grid_boxes` 数量 = rows×cols；全部 `source=="generated"`。
  - `preview_box_colors`：返回每框 `{xy,name,rgb}`、`name` 为 str。
- UI：手动冒烟（点 4 角 → 生成 → 校正 → 导出），子代理无浏览器，由用户执行。

## 8. 不做的事（YAGNI）

- ❌ 板子非平面/翘曲的高阶校正（4 点单应 + 人工校正对 MVP 足够；后续若需要再加内部参考点）。
- ❌ 部分板（截断）标注——MVP 假设整板可见。
- ❌ 保留 HoughCircles 预标注作次要入口——直接移除，避免误导（代码仍在 repo 供 CLI）。
- ❌ 训练流程本身——本 spec 只到「产出标签」，训练是后续计划。
