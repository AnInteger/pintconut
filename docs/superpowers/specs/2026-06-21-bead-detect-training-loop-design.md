# 珠子检测训练闭环设计文档

> 日期: 2026-06-21
> 状态: Draft
> 涉及模块: 新增 `src/bead_label_service.py`、`training/bead_split.py`；重写 `src/bead_annotate_ui.py`；小改 `src/bead_grid.py`、`src/cli.py`
> 关联: 在 `2026-06-17-bead-grid-detection-design.md`（bead-grid 算法）之上，建立「标注 → 训练 → 接入」闭环

---

## 1. 背景与目标

bead-grid 算法（`BeadGridFitter`）已在 `feat/bead-grid-detection` 分支**实现**，但**尚未在真实照片上端到端验证过 CLI**——需要先通过标注 UI 观察算法在真实照片上的实际输出，才能判断其效果与瓶颈。该算法的**第一步「检测豆子」目前用纯几何的 HoughCircles**，从调参经验看（最新提交 `test(bead_grid): real-photo regression set + tuned detection params`）在真实随手拍上偏脆弱：光照/角度一变就需要重新调参。**但是否真要训练检测器替换它，须等 UI 评估确认「检测」确实是真实照片上的主要失败点后再定**（见第 3 节阶段 1 的评估关卡）。

**目标（分两层）**：
- **近期（确定要做的）**：把 bead-grid 算法接进标注 UI，使其成为**在真实照片上评估算法效果的主要界面**，同时顺带产出训练数据。
- **远期（评估确认后才做）**：若 UI 评估证明「检测」是真实照片上的主要瓶颈，则训练一个单类别（珠子）YOLOv8n 检测器，**只替换 bead-grid 算法里的 HoughCircles 这一步**，让检测稳健可泛化；检测稳了，下游的网格拟合、LAB 采色、图纸差异比对自动跟着稳。

**关键认知（决定范围）**：
- **位置/行列**：由网格拟合算出，不是模型。
- **颜色**：LAB 空间比对 221 色板，**确定性、不需要训练**。
- → 模型唯一的活是「把每颗豆子框出来」。训练范围因此非常小：一个模型、一个任务、一个类别。

**现状盘点**：
- 真实照片充足：`training/photos/` + `training/test_images/` 共约 30~40 张。
- 标注从零开始：`training/bead_dataset/` 为空（0 图 0 标签、无 data.yaml），无 `bead-best.pt`，仅有 `FastSAM-s.pt`。
- 现有 `src/bead_annotate_ui.py` 只能「调 HoughCircles 参数 + 导出」，**不能编辑/校正框**——产出不了干净训练数据。
- `training/bead_train.py` / `bead_validate.py` / `src/bead_detect.py`（`BeadDetector`，YOLOv8n）已就绪。

## 2. 核心架构发现：检测是干净的「接缝」

`bead_grid.py` 中检测豆子是独立的模块级函数 `detect_beads()`（第 87 行），`BeadGridFitter.fit()` 在第 335 行调它。**fit() 之后的全部逻辑（`estimate_grid_axes` / `label_beads` / `build_cells` / 采色）都只吃 `list[Bead]`（豆子中心坐标 + 颜色 + 半径），完全不关心豆子是怎么检出来的。**

→ 把「检测」做成可切换，改动极小：`fit()` 增加可选 `detector` 参数，下游一行不改。这是整个闭环能低成本成立的地基，也让「HoughCircles 启动 → 学过的检测器替换」成为一次平滑演进而非重写。

## 3. 范围与分阶段

本 spec 覆盖完整闭环，但实施按依赖顺序分阶段。**阶段 1（标注 UI）是当前唯一卡点、主工作量，且身兼「评估算法」与「产训练数据」两职**：

| 阶段 | 内容 | 产物 |
|---|---|---|
| **1. 标注 UI（双重目的）** | 重写标注 UI：bead-grid 预标注 + 网格辅助补漏 + 框编辑 + 导出；同时作为**观察算法在真实照片上效果的界面**（见 4.3 评估预览） | 算法效果评估 + 可产出干净 YOLO 标签 |
| **— 评估关卡 —** | 用 UI 在 5~10 张真实照片上观察算法输出，判断「检测」是否为主要失败点、HoughCircles 是否够用 | **决定是否进入阶段 2**（若 HoughCircles 已够好则跳过训练） |
| **2. 划分 + 训练（条件性）** | 仅当评估关卡确认检测是瓶颈：train/valid 划分 → `bead_train.py` 训练 → `bead_validate.py` 验证 | `models/bead-best.pt` |
| **3. 接缝接入** | `fit()` 加 `detector` 参数；`cli.py` 优先用训练好的检测器 | 真实照片上的位置+颜色+差异 |
| **4. 主动学习迭代** | 失败照片回流重标重训（手工） | 检测质量收敛 |

## 4. 详细设计

### 4.1 检测可切换（`src/bead_grid.py` 小改）

- `BeadGridFitter.fit(self, image, board_size=None, detector=None) -> GridResult`
  - `detector=None`：走现有 `detect_beads(image)`（HoughCircles）。
  - `detector` 给定（一个有 `.detect(img)->list[box]` 接口的对象，即 `BeadDetector`）：调用它取 box，再用新 helper `_beads_from_boxes(image, boxes)` 转成 `list[Bead]`（中心→`xy`，box 半宽→`radius`，`_sample_color` 取色）。
- 新增 `_beads_from_boxes(image, boxes) -> list[Bead]`：纯函数，box→Bead。
- `detect_beads()` 保持不变（既是 HoughCircles 路径，也是标注 UI 的预标注引擎）。
- **不引入对 `bead_detect.py` 的硬依赖**：`fit()` 只依赖传入对象的 `.detect()` 接口，模型加载仍在 `BeadDetector` 内部懒加载。

### 4.2 标注服务层（`src/bead_label_service.py`，新）

沿用项目 `label_ui.py`/`label_service.py`「薄 UI + 服务逻辑分离」模式，把可单测的纯逻辑放这里。核心：

- `prelabel(image: np.ndarray) -> PrelabelResult`
  - 调 `BeadGridFitter.fit(image)`；失败（`GridFitError`，豆子 < `MIN_BEADS`=20）则降级为 `detect_beads(image)`（仅检测框、无网格辅助）。
  - 返回：检测到的框（绿）、网格映射 `grid_map`、尺寸统计（中位半径）、以及计算出的**漏检洞** `holes`。
- **漏检洞计算**：`holes` = 落在网格有效区域内（排除截断边缘 `truncation.clipped_edges`）但无检测豆子的格点 `(r,c)`，其预测图像坐标 = `grid_map.to_xy(r,c)`、预测尺寸 = 中位半径。这些即检测器漏检的难样本。
- `apply_autofill(result) -> list[box]`：把 `holes` 转成候选框（黄），供用户核对。
- `export_yolo(image, boxes, name, split_dir)`：写图 + YOLO 检测标签（复用 `bead_prelabel.save_yolo_boxes`）。
- 标注会话状态（当前框列表、网格、当前图）由 UI 层持有；服务层保持无状态、纯函数为主，便于单测。

**框按来源区分**，方便扫一眼定位需核对处：🟢 绿=检出、🟡 黄=网格补的(推断)、🔵 蓝=手动加。

### 4.3 标注 UI（`src/bead_annotate_ui.py`，重写）

薄 Gradio 界面，调 `bead_label_service`。沿用现有 UI 的「左输入/控制、右结果」布局。能力：

1. 加载/上传照片（复用 `training/bead_photos/` 或直接读 `training/photos/`）。
2. **预标注**按钮 → 显示带框 + 网格叠加层的图。
3. **网格补漏**按钮 → 把洞补成黄色候选框。
4. **框编辑**：增加 / 删除 /（可选）移动或缩放框。
5. **导出**按钮 → 写入 `bead_dataset/images/train` + `labels/train`，并刷新「已标注 N/总数」统计。
6. **加载下一张未标注**（沿用现有逻辑）。

**评估预览（UI 的第一职责）**：由于算法尚未在真实照片上验证，UI 要让用户能直观判断算法效果。预标注视图除框 + 网格外，**叠加每颗豆子/每个格子的 LAB 匹配色**（`fit()` 已产出 `CellInfo.color`，零额外成本）；可选再加「对照图纸看差异」入口（传 blueprint → 复用 `DiffComparator` 标不匹配格），此项非阶段 1 必需、可后置。

**⚠️ 主要 UX 风险**：Gradio 原生框编辑交互较弱。实施阶段需先 spike 交互模型（候选：沿用 `label_ui.py` 的点选模式——点图删框、点空位+尺寸输入加框；或评估 `gr.ImageEditor` / 第三方标注组件）。网格叠加层必须清晰可见，因为「网格补漏」的可信度完全依赖网格是否正确。

### 4.4 数据流与 train/valid 划分

- 标注导出 → `training/bead_dataset/images/train/<name>.jpg` + `labels/train/<name>.txt`（YOLO 检测格式，单类别 `0`）。
- **训练坑（已纳入设计）**：`bead_train.py` 要求 `images/valid` + `labels/valid`，但 UI 只写 `train/`。
- 新增 `training/bead_split.py`：把 `train/` 按 80/20 划分出 `valid/`，**按文件名 hash 决定归属，保证多次运行结果稳定可重复**（移动文件，可重跑）。训练前运行一次。

```
照片 → prelabel → 人工校正 → export(train/) → bead_split(train/→train/+valid/) → bead_train.py
```

### 4.5 训练与接入

- 训练：`python training/bead_train.py --data training/bead_dataset --device 0 ...` → `training/runs/bead-vN/weights/best.pt`；复制为 `models/bead-best.pt`。
- 验证：`python training/bead_validate.py --model <best.pt>` 看 mAP，再用几张未见过照片肉眼复核。
- 接入：`cli.py` 中若 `models/bead-best.pt` 存在则构造 `BeadDetector` 传入 `fit(image, detector=...)`，否则 fallback HoughCircles（打日志）；可选 `--detector hough|yolo` 覆盖。
- 端到端验证：跑完整 CLI（photo + blueprint）→ 真实照片上的「位置+颜色+差异」，与 HoughCircles 基线对比。

## 5. 错误处理

| 情况 | 处理 |
|---|---|
| fit() 检出豆子 < 20（`GridFitError`） | UI 降级：仅显示检测框、关闭网格辅助，提示「豆子太少，无法拟合网格」 |
| 训练好的检测器模型缺失 | cli 自动 fallback 到 HoughCircles，日志告警；不崩溃 |
| 网格补漏基于错误网格补错框 | 叠加层可见 + 黄色区分 + 用户须核对；网格歪时先调检测/删坏点重拟合再补 |
| 导出时框为空 | 警告并跳过该图（不写空标签误导训练） |
| `bead_split.py` 重复运行 | hash 决定归属，幂等：已在 valid 的不重复迁移 |

## 6. 测试策略

- `tests/test_bead_label_service.py`（新）：用合成网格/豆子 fixture（复用 `0758248` 引入的合成夹具）断言 prelabel 的检测框 + 漏检洞计算正确；`export_yolo` 产出合法 YOLO 行；`apply_autofill` 正确把洞转框。**无需模型文件**（走 HoughCircles / 合成路径）。
- `tests/test_bead_grid.py`（扩展）断言接缝：传入 fake detector（返回固定 box）调 `fit(image, detector=fake)`，结果应与直接喂同样 beads 一致——验证检测可切换不破坏下游。
- `tests/test_bead_split.py`（新）：划分比例 ≈ 80/20、按 hash 幂等可重复。
- 集成：阶段 3 完成后手工跑 CLI 对照 HoughCircles 基线（无自动化集成测试，YAGNI）。

## 7. 不做的事（YAGNI）

- ❌ 训练颜色模型——颜色保持 LAB 比对。
- ❌ 训练网格映射模型 / 端到端模型——本阶段只做检测器；若后续检测器 + 几何仍不够再评估。
- ❌ 自动重训闭环——手工回流重标重训即可。
- ❌ 改动/退役 `label_ui.py`（FastSAM 板子分割标注）——与本次无关，保持原样。
- ❌ BeadDetector 的置信度/尺寸过滤逻辑改动——仅在 bead-grid 内部复用其 `.detect()`。

## 8. 验收标准

- [ ] 标注 UI 可对真实照片预标注、网格补漏、人工校正、导出 YOLO 标签。
- [ ] 标注 UI 可叠加显示每豆/每格匹配色，作为算法效果评估界面（支撑「评估关卡」决策）。
- [ ] 导出数据经 `bead_split.py` 划分后，`bead_train.py` 可正常训练出 `bead-best.pt`。
- [ ] `BeadGridFitter.fit(image, detector=...)` 可切换检测来源，单测验证下游不变。
- [ ] `cli.py` 优先用训练好的检测器，在真实照片上产出「位置+颜色+差异」，质量优于 HoughCircles 基线。
