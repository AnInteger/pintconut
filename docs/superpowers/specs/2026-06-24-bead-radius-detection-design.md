# 拼豆半径检测算法重设计（方案 B+）

> 日期: 2026-06-24 | 状态: 设计待审 | 关联: `src/bead_label_service.py::find_bead_radius`

## 1. 背景与问题

当前 `find_bead_radius`（bead_label_service.py:157）流程：
1. 灰度 Sobel 梯度幅值
2. 以点击中心做径向剖面（每半径取一圈梯度均值）
3. 取"≥0.6×峰值的最外层局部极大值"（grad_outer）作半径
4. 已标 ≥3 颗时，半径夹到 `[0.8, 1.2]×中位`（clamp）

**两个病根**（用户在真实照片上观察到）：
- **偏小**：塑料高光在豆面内侧形成强梯度环，被误当边缘
- **偏大 / ballooning**：相切豆之间梯度弱或缺失，邻居外圈边缘更强，把半径拉走

**deep-research 关键证据**（见 `journal.jsonl`，wf_83c89ed4-e11）：
- van de Weijer (ICCV'03) / Gevers：DRM（二色反射）模型下 **opponent 色空间梯度对高光不变**，只依赖材质+几何 → 色度梯度能把材质边和高光分开
- 径向剖面取峰在**相切圆场景结构性失效**（多源论坛+论文证实：相切处梯度弱/缺失）
- Taubin/Pratt **圆拟合 + outlier rejection**：从点集估圆的标准鲁棒做法；Kasa 对残缺弧严重偏小

## 2. 目标 / 非目标

**目标**
- 给定点击中心，鲁棒估计单颗豆的**圆心 + 半径**
- 治偏小（高光）+ 偏大（邻居污染）
- 适应透视导致的豆子大小不一（逐豆独立拟合）
- 中心点击偏移可被纠正
- 同色紧挨的死穴有兜底 + 标红人工复核

**非目标（YAGNI）**
- 不引入活动轮廓 / GVF（overkill、参数敏感、有"Mickey Mouse ears"失败）
- 不做全板深度学习分割
- 不改 YOLO 导出格式（仍矩形框，detection 标准）
- 不强制全局统一半径（仅视角正时作辅助验证/兜底）

## 3. 算法设计（方案 B+）

### 3.1 数据流

```
photo(BGR)
  → chroma_gradient            → a*b* 梯度幅值图（h_load 预计算）
点击 (cx, cy)
  → extract_edge_points        → N 个候选边缘点 [(x, y, 响应)]
  → robust_circle_fit          → (cx', cy', r, inliers, fit_ok)   圆心+半径一起拟合
  → adaptive_prior(已标半径集) → (全局中位, use_global)
  → 融合：拟合可信→用拟合；偏差大/失败→先验兜底 + warn
  → 输出 (cx', cy', r, warn)
```

### 3.2 组件

**(a) `chroma_gradient(img) → gmag_ab`**
- BGR→LAB，取 a*、b* 两通道各做 Sobel，`gmag_ab = magnitude(sobel(a*), sobel(b*))`
- 不用 L*（亮度，高光所在）、不用灰度
- 替代现有 `gradient_magnitude`（旧函数保留，供 eval 对比）

**(b) `extract_edge_points(gmag_ab, cx, cy, r_min=3, r_max=120, n_ang=72) → points`**
- 以 (cx,cy) 为中心，n_ang 个角度（每 5°）各射一条线
- 每条线上 r∈[r_min,r_max] 找 gmag_ab 最大响应点 → (x, y, 响应值)
- 返回 N 个带响应权重的候选边缘点；沿角度批量采样，向量化

**(c) `robust_circle_fit(points, cx0, cy0) → (cx, cy, r, inliers, ok)`**
- **Taubin 代数圆拟合**：圆心+半径同时解；非迭代、100% 收敛、bias 小（手写小矩阵特征值，不引入新依赖）
- **IRLS 风格离群剔除**：全部拟合 → 算各点几何残差 → 丢残差过大者（自适应阈值，如 2×中位残差）→ 再拟合，迭代 2-3 轮
- （可选增强）曲率一致性：真豆缘点曲率 ≈ 1/r，曲率异常点辅助剔除
- 返回**修正后的圆心**（纠正点击偏移）、半径、内点数、是否可信

**(d) `adaptive_prior(radii) → (median, use_global)`**
- 输入已标豆的半径集；离散度 = `std/median`（或 IQR/median）
- `use_global = (离散度 < 阈值，建议 0.1)` → 视角正，启用强先验
- 视角歪 → `use_global=False`，逐豆独立（不假设统一半径）

**(e) `find_bead_radius_v2(gmag_ab, cx, cy, prior_radii=None) → (cx, cy, r, warn)`**【总入口】
- extract → fit → prior 融合
- `warn=True` 当：拟合失败 / 拟合半径 vs 全局先验偏差超阈值 / 内点过少
- 签名向后兼容（返回半径+warn），**增加返回修正圆心**

### 3.3 UI 集成（bead_annotate_ui）
- `h_load`：预计算 `gmag_ab`（替代当前 `gmag`）
- `h_click` 点豆：调 `find_bead_radius_v2`，box 用修正后的 `(cx', cy', 2r)`
- **显示改画圆（`cv2.circle`）替代矩形** —— 直观看半径准不准（顺带消除"方框视觉误差"）
- `warn`：红圈 + 状态提示"该颗拟合不稳，建议点边缘手动覆盖"
- 点豆的"二次点击覆盖边缘半径"逻辑保留（修正个别不准的）

### 3.4 基础设施（路径解耦 + 目录重构）
- **新增 `src/paths.py`**：集中 `BASE_DIR` / `PHOTOS_DIR` / `DATASET_DIR` / `IMAGES_DIR` / `LABELS_DIR` / `BOARD_SIZES_PATH`，代码 `from src.paths import ...`，不再写死
- **目录重构**：
  - `tmp/training/training/{1,2,3}/` → `training/photos/{1,2,3}/`（扁平化多余的 `training/` 层）
  - 删 `training/photos` 旧散落副本 + `training/test_images/`
  - `NYQC4978.JPG` 归入 `photos/1/`（UI 下拉 + gt 验证都能找到）
- `bead_annotate_ui._list_photos`：适配 `photos/{1,2,3}/` 子目录扫描（递归列图）
- `eval_edge_finder.py`：默认路径修正 + 增加 B+ 对比列

## 4. 验证（gt_NYQC4978）

用 `tests/validation/gt_NYQC4978.txt`（12 颗真值）量化：
- 两种输入：精确中心 + ±2/±4px 抖动
- 对比：当前 grad_outer **vs** B+（色度+拟合）
- 指标：
  - `|dR|` 中位 / 最大
  - `IoU`（拟合圆 vs 真值圆）中位 / 最小
  - **圆心偏移修正量**（点击点 → 拟合圆心 → 真值圆心）
  - `off-by>10px` 计数
- **建议成功标准**（待确认）：`|dR|` 中位 < 3px；`off-by>10px = 0/12`；圆心修正后偏移 < 2px

## 5. 测试策略

**单元**（`tests/test_bead_label_service.py`，合成图）：
- `chroma_gradient`：高光环在 a*b* 下响应 ≈ 0
- `extract_edge_points`：干净环 → 点落指定半径
- `robust_circle_fit`：带离群点的点集 → 拟合圆 + 剔除离群
- `adaptive_prior`：均匀半径 → use_global=True；离散 → False
- `find_bead_radius_v2`：高光（不偏小）+ 邻居环（不偏大）+ 中心偏移（被纠正）
- 现有 6 个 `find_bead_radius` 测试适配新签名

**验证脚本**：`eval_edge_finder` 跑 gt，出对比表 + overlay（绿=真值，品红=B+）

## 6. 风险与对策

- **同色紧挨 + 模糊（死穴）**：先验兜底 + warn 标红 + 人工点边缘覆盖。文档明确标注为已知限制。
- **LAB a* 梯度可能跟灰度差不多（弱证据）**：gt 实测；若色度无优势，试 opponent (o1,o2) 或 Hue。
- **IRLS 阈值**：gt 调参，最终值写入文档。
- **依赖**：仅 numpy + cv2；Taubin 拟合手写（小矩阵特征值），不引入新依赖。

## 7. 文件清单

| 动作 | 文件 |
|---|---|
| 新增 | `src/paths.py` |
| 改 | `src/bead_label_service.py`（新增组件 + `find_bead_radius_v2`） |
| 改 | `src/bead_annotate_ui.py`（画圆显示 + 用 v2 + 路径导入） |
| 改 | `tests/test_bead_label_service.py`（适配 + 新测试） |
| 改 | `tests/validation/eval_edge_finder.py`（默认路径 + B+ 对比） |
| 迁移 | `tmp/` → `training/photos/{1,2,3}/`，删旧 `photos` / `test_images` |
