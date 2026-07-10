# 标注工具性能 + 预填闭环 + 数据可信化（方案 2 落地）

> 日期: 2026-07-10 | 状态: 设计待审
> 关联: `src/bead_annotate_mpl.py`, `src/bead_label_service.py`, `training/bead_train.py`, `training/board30_train.log`
> 前序: `docs/research/2026-06-25-small-object-detection-for-bead-counting.md`, `docs/superpowers/specs/2026-07-01-yolo-training-pipeline-design.md`

## 1. 背景

### 现状（基于代码 + 训练日志核实）
- **生产检测仍走 HoughCircles**：`cli.py` 调 `BeadGridFitter().fit(...)` 未传 `detector` → 走 `bead_grid.py:88` 的 `detect_beads()`（HoughCircles）。YOLOv8n 模型只用在 `bead_tune.py` 和 `bead_annotate_mpl.py` 的 `f` 键 SAHI 预填里。
- **训练没学进去**：`board30_train.log` 全程 val `P=R=mAP50=mAP50-95=0`，且 `cls_loss` 多次飙到 35-87（正常 ~3.7）→ 模型没学到 + 训练不稳定。
- **标注痛点**：一张 432-640 框的图要标 ~1 小时；工具卡顿（放大/拖动/两点标注都慢）。
- **已验证无效、不再尝试**（见 CLAUDE.md 项目宪法）：几何网格拟合当检测器、四角网格批量生成、梯度自动定半径、HoughCircles 预标+补洞——在本项目照片（光照/反光/透视）上都不灵。
- **数据路径失效**：`bead_dataset/data.yaml` 绝对路径指向旧路径 `/home/sunxing/code/pintconut/...`，与当前仓库 `claude_workspace/pintconut` 不符。
- **死代码**：`bead_label_service.py` 中除 `export_yolo` 外全是死代码（梯度定半径/网格生成/补洞/prelabel 等），且 `tests/test_bead_label_service.py` 给它们写了测试。

### 核心判断（精度天花板 = 数据）
mAP=0 的根在**数据质与量**；数据量被**标注速度**锁死；标注速度被**工具卡顿**锁死。
所以最高 ROI 杠杆链：**修工具性能 → 标得快 → 产可信数据 → 诊断/修 mAP=0 → 再谈 YOLO11/调参/接线**。
YOLO11 升级、训练调参、生产接 YOLO 都是拿到可用 baseline 之后的后续动作（本期不做）。

## 2. 目标 / 非目标

**目标**
1. 修标注工具卡顿，让"1h/600 颗"显著下降。
2. 把 `f` 键预填改成**非破坏式合并 + 置信度着色 + 快捷接受/清理**，形成主动学习闭环。
3. 用改进后的工具产出一小批**可信 GT**（train/val 分），验证 mAP 能否从 0 起来。
4. 评测用标准 mAP50（在可信 GT 上），不引入自定义计数指标。

**非目标（YAGNI）**
- 不做任何基于自动检测的标注辅助（梯度/网格/补洞——已验证无效）。
- 不盲目扩数据到 1500 张（先修质，指标说不够再补多样性）。
- 本期不换 YOLO11、不接生产 CLI、不做密度图计数（后续阶段）。

## 3. 设计

### 3.1 性能（核心；分叉 P1/P2/P3 — 待 review 确认）

诊断（已核实，三个原因叠加）：
- 每图 432-640 框 → `redraw()`（`bead_annotate_mpl.py:71-88`）每次全量 `remove()`/`add` ~1300-1900 个 artist（圆+点+文字），每次点击/删除/撤销都重建一遍；
- TkAgg 后端 = 纯软件渲染（最慢主流后端）；
- 大图整张 `imshow`，每次拖动/缩放都整张重渲染（matplotlib 不是图像浏览器）。

| 方案 | 做法 | 成本 | 预期 |
|---|---|---|---|
| **P1（推荐先做）** | 换 QtAgg 后端 + 显示图降采样到 ~1600px（点击坐标按比例还原全分辨率，标签按原图尺寸归一化）+ 增量重绘（只加新框/删指定框，不全量重建） | 小（几小时） | "好很多"，但 matplotlib 上限有限 |
| **P2（P1 不够再上）** | 换 PyQt/PySide `QGraphicsView` 原生 viewer（GPU 加速 pan/zoom），框用 `QGraphicsItem` | 大（UI 层基本重写） | 体验质变，根治 |
| **P3（备选）** | 用 labelme / CVAT 等现成工具，SAHI 预填通过 CVAT SDK / labelme 插件重接 | 中（对接） | 不维护 viewer，但预填+颜色匹配要重接 |

> ⚠️ **P1 唯一易踩的坑**：降采样后所有点击坐标必须按 `display_scale` 还原到原图坐标；`save_yolo_boxes` 归一化时 `w/h` 必须用**原图全分辨率**，不是显示分辨率。保存前抽样校验 1-2 个框坐标。

### 3.2 预填闭环（B1 + B2）
- **B1 非破坏式合并**：`prefill()` 不再 `boxes.clear()` 全部手工框；改为预测框与已有手工框做 **NMS 合并**，手工框权威保留，预测框以 `source="auto"` 作为"建议"加入。
- **B2 置信度着色 + 快捷键**：`auto` 框按 conf 上色（绿 ≥0.5 / 黄 0.2-0.5 / 红 <0.2）；快捷键：接受全部高置信、删全部低置信、逐个跳转不确定的。
- **闭环**：v0 模型（哪怕 mediocre）→ 预填 → 人工改错 → 重训 v1 → 预填更准。前提是先有 v0（见 3.3 bootstrap）。

### 3.3 bootstrap + 诊断 mAP=0
- 用修好性能的工具，**手标 5-8 张多样、可信**的图（train/val 分开；**val 即可信评测集**）。
- 修 `data.yaml` 失效绝对路径（改相对路径）。
- GPU 重训：`imgsz 1280` / `epochs 300` / `patience` / `cos-lr` / `single_cls` / `close_mosaic` 开。
- **双线诊断 mAP=0**：① 数据线——部分标注/框不一致（用可信重标验证）；② 训练线——`cls_loss` 飙到 87 的发散（查 lr / 0 实例 batch / 标签格式）。

### 3.4 评测
- 可信 GT 上跑标准 **mAP50**（headline）+ **mAP50-95**（参考）。
- **不引入计数误差等自定义指标**（避免 hack）。
- 看 train vs val 差距判过拟合。

## 4. 文件清单

| 动作 | 文件 |
|---|---|
| 改 | `src/bead_annotate_mpl.py`（性能 P1/P2/P3 + B1/B2） |
| 改 | `training/bead_dataset/data.yaml`（修相对路径） |
| 改 | `training/bead_train.py`（imgsz/epochs 默认值或文档化推荐参数） |
| 删 | `src/bead_label_service.py` 死代码 + `tests/test_bead_label_service.py` 对应用例（保留 `export_yolo`） |

## 5. 风险与对策

- **P1 降采样坐标还原出错 → 标签全偏**：保存前抽样校验 1-2 个框；保留"原图全分辨率"开关。
- **v0 模型仍很差 → 预填垃圾，闭环转不起来**：bootstrap 先纯手标几张可信图，别指望预填。
- **mAP=0 是训练发散不是数据 → 重标也没用**：3.3 双线诊断，先看 `cls_loss` 曲线。
- **多样性不足（IMG_6093 类）→ 重标救不了**：干净重训后看这类图，失败则补几张同类型（工具已快）。

## 6. 后续阶段（本期不做）

- 生产 CLI 接 YOLO（`BeadDetector` 注入 `BeadGridFitter.fit(detector=...)`，`bead_grid.py:348` 已留参数）。
- YOLO11 vs YOLOv8 A/B（Ultralytics API 一行可换：`YOLO("yolo11n.pt")`，等 baseline 稳）。
- 生产推理 SAHI / P2 小目标策略。
