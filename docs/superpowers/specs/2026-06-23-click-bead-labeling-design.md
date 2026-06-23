# 点豆标注设计文档

> 日期: 2026-06-23
> 状态: Draft
> 涉及模块: 新增 `src/bead_label_service.py` 的 `gradient_magnitude` / `find_bead_radius`；`src/bead_annotate_ui.py` 新增「点豆」模式（取代「加框」）
> 取代: 标注 UI 的「加框」固定半径手框模式
> 关联: `2026-06-21-corner-click-labeling-design.md`（角点网格模式保留为另一场景）；`tests/validation/eval_edge_finder.py` + `tests/validation/gt_NYQC4978.txt`（本设计算法的用户真值验证基准）

---

## 1. 背景：角点网格用「几何估半径」，逐颗「看像素量半径」更准

角点网格模式（2026-06-21 spec）用单应矩阵算豆位 + 局部间距 × 0.4 估框半径——靠**几何**。但同一张照片里，透视会让豆子看起来大小不同，且豆子真实半径未必等于 0.4×间距（变形、部分豆、间距非线性变化时偏差大）。

本设计补一条**靠实际像素量半径**的路径：用户点一颗豆中心 → 径向梯度找边缘 → 每颗框贴合真实豆子。这条路径已用**用户手标真值**验证（见 §4），不是自评。

**与角点网格的关系**：两条路径并存，场景不同——
- 角点网格：规则大板，4 角点快速铺满，框按几何估（够用）。
- 点豆：小/不规则板，或需要每颗都贴合真实边缘时，逐颗点。

## 2. 目标

新增「点豆」标注模式：用户点一颗豆中心 → 算法自动找边缘半径 → 画框；不准就再点一下豆边缘手动覆盖。逐颗点完 → 导出 YOLO。取代固定半径的「加框」模式（点豆是它的超集：自动半径 + 可手盖）。

## 3. 核心流程

1. 加载照片（复用 `h_load`）。
2. 模式选「**点豆**」。
3. **点豆中心**：在豆上点一下 → `find_bead_radius` 算半径 → 画框（🟢 绿=自动）；该颗进入「待覆盖」态。
4. **可选覆盖**：紧接着再点一下**同一颗豆的边缘** → 半径=中心到该点距离，手动覆盖（框转 🔵）；若点在别处（新豆）则上一颗自动确认。
5. **逐颗点完** → 撤销/删除校正（复用现有框编辑）。
6. **导出 YOLO**（复用 `export_yolo`）。

## 4. 算法（已用用户真值验证）

基准：`tests/validation/gt_NYQC4978.txt`（用户在 `NYQC4978.png` 上手标的 12 颗红豆 `cx cy r` 真值）+ `tests/validation/eval_edge_finder.py`（客观评估，用户可自跑）。**本 spec 不自评，引用该基准的数。**

### 4.1 grad_outer（逐颗找边）
从中心向外画同心圆，取径向梯度剖面（每圈 Sobel 梯度均值）；在 `>0.6×峰值` 的局部峰里选**最外**那个 → 候选半径。高光是内圈、真边是外圈，故跳过高光。

**验证结果**：12 颗里 **11 颗 |dR|≤6px**（中位 2px、IoU 中位 0.93）；±4px 点击抖动下 IoU 中位 0.86（鲁棒）。

### 4.2 钳制（兜底 ballooning）
第 12 颗（#11）是「超大高光 + 被 3 颗邻居包围」的病理豆：局部梯度信号 r=4→118 全程平坦偏高（0.5–0.74×峰值），**根本没有干净边缘**，grad_outer ballooning 到 118。

纯局部算法在此无解（梯度法 / 饱和度法 / 覆盖率法 / 0.4×峰 / 0.6×峰 五个局部修复全败）→ 引入**全局先验**：候选半径钳到「**已标豆半径中位数 × [0.8, 1.2]**」。#11 钳到 ~66（118 不再灾难性，IoU 从 0.20 回到可用区间；残余 ~13px 由 §6 红框+手覆盖收尾）。

钳制仅在已标豆 **≥3 颗**时启用（中位数才稳）；不足 3 颗用原始 grad_outer（函数自带 [3,120] 边界）。

## 5. 代码改动

### 5.1 `src/bead_label_service.py`（新增纯函数，算法唯一真源）
- `gradient_magnitude(img) -> np.ndarray`：Sobel 梯度幅值（从 eval 脚本迁入，全项目唯一实现）。
- `find_bead_radius(gmag, cx, cy, prior_radii=None, r_min=3, r_max=120) -> tuple[int, bool]`：返回 `(radius, warn)`。
  - 内部算 grad_outer 候选（`>0.6×峰值` 的最外局部峰）。
  - `prior_radii` 长度 ≥3 时钳到 `[0.8, 1.2]×中位数`。
  - `warn=True` 当（钳制前）候选触底(`≤r_min+2`)或触顶(`≥0.85×r_max`) **且**未钳制（即 prior<3）→ 提示 UI 画红框。
  - 纯逻辑、无模型/无 UI 依赖、可单测。

`tests/validation/eval_edge_finder.py` 改为 import 这两个函数（删掉重复实现，保留基准 harness + 用户真值），保持单一真源。

### 5.2 `src/bead_annotate_ui.py`（新增「点豆」模式，取代「加框」）
- **模式** `["角点", "点豆"]`（删「加框」——点豆是超集）。加载照片时预算一次 `gradient_magnitude` 存入 state（每图一次，避免每点重算）。
- **`h_click` 点豆分支**：
  - 无 pending → 记中心，调 `find_bead_radius(gmag, x, y, prior_radii=[已标框的 radius])` → 加框 `source="auto"`；存 `pending=(cx,cy,r)`。
  - 有 pending 且新点到 pending 中心距离 < pending 半径 → **边缘覆盖**：`r=dist`，框改 `source="manual"`、`warn=False`，清 pending。
  - 有 pending 但点在 pending 框外 → 上一颗确认，新点作新豆中心。
  - `warn=True` → 该框带 `"warn": True` 字段；`_draw` 见 `warn` 画🔴红描边（独立于 source 颜色）。
- 删「加框」的固定半径 Number 输入。
- 复用：`_draw`、`_stamp_ids`、`_box_label`、`h_delete`、`export_yolo`、配色预览。

### 5.3 不动的部分
- 角点网格模式（`generate_grid_boxes` / `preview_box_colors` + 其 UI 分支）：**保留**。
- `src/bead_grid.py` 等：不动。

## 6. 错误处理

| 情况 | 处理 |
|---|---|
| 已标豆 <3 颗（无中位数钳制）且 grad_outer ballooning/触底 | `warn=True` → 画🔴红框，提示「这颗点边缘覆盖」 |
| ballooning 但 ≥3 颗 → 钳制拉回 | 不告警（钳制已兜底），框正常 |
| 用户点边缘覆盖 | 半径=中心到边缘点距离，框转 `source="manual"`(🔵)、`warn=False`，绕开算法 |
| 误点（点空了/点错豆） | 撤销/删除复用现有 |
| 框落图外 | 正常画/导出（YOLO 容忍归一化值略超界） |

## 7. 测试

- `tests/test_bead_label_service.py` 新增：
  - `find_bead_radius` 干净合成豆（实心圆 + 板背景）：返回半径 ≈ 真值（容差 <2px），`warn=False`。
  - 合成带高光豆（中心白斑）：grad_outer 跳过高光，半径 ≈ 真值（容差 <3px）。
  - 钳制：`prior_radii=[55,56,54]`，候选 118 → 返回 ≤1.2×55≈66；候选 3 → ≥0.8×55≈44。
  - `warn` 触发：候选触底/触顶且 prior<3 → `warn=True`；≥3 钳制后 → `warn=False`。
- `tests/test_bead_annotate_ui.py` 新增 `h_click` 点豆分支：点中心加 auto 框；点同框内→manual 覆盖；点框外→新豆。
- UI 冒烟：手动（点豆→覆盖→导出），用户执行。

## 8. 不做的事（YAGNI）

- ❌ 网格中心 + 全自动逐颗找边（方案 A，用户已选 B 纯点豆）。
- ❌ 颜色区域生长 / FastSAM 实例分割（验证已证不可靠或鸡生蛋）。
- ❌ K-近邻局部中位数钳制——v1 用全局中位数 ±20%（测试图半径 52–57 极紧，够）；极端透视若不够再加。
- ❌ 批量扫行 / 拖拽多点——逐颗点击对 MVP 足够，慢了再说。
- ❌ 训练流程——本 spec 只到产出标签。
