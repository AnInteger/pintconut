# 豆子检测驱动设计文档

> 日期: 2026-06-17
> 状态: Draft（修订：简化为"检测 + 共线分组编号"，砍掉 DLT+RANSAC 单应重拟合）
> 涉及模块: 新增 `src/bead_grid.py`；删除 `src/edge_refiner.py` 与 `src/detect.py` 的 `BoardDetector`；修改 `src/cli.py`、`src/grid.py`、`src/compare.py`
> 取代: `2026-06-11-board-edge-refinement-design.md`（该方案废弃）

---

## 1. 背景与问题

当前板子检测管线（YOLOv8-seg 粗 mask → `edge_refiner.py` 启发式精化）实测存在三个核心问题：

1. **自训 YOLOv8-seg 泛化差**：经常根本检不到板子，拍摄条件（光照/角度/背景）一变就崩。
2. **`edge_refiner` 启发式不可靠**：6 层过滤 + ~30 个魔法参数，常给出离谱结果。
3. **本质错配**：用像素级分割去拟合一个几何上的直边四边形，模型表达力与物体几何先验不匹配。

**约束**（已与用户确认）：

- 拍摄条件：**必须适应任意手机随手拍**（含明显倾斜/透视，如 `IMG_6121`）。
- 自动化：**必须全自动**，不接受"算法先猜 + 人工确认"。
- 技术方向：**纯经典 CV**，不再依赖自训分割模型。

**真实照片观察**（`training/test_images/1.png`、`2.png`、`photos/IMG_6121.JPG` 等，经视觉确认）：

- 豆子为**双圆环/同心圆结构**，且**所有颜色的豆子都有深色（黑/深灰）边缘环**——这是**颜色无关的强特征**，221 种豆身颜色不影响边缘环检测。→ 豆子检测可靠，不需要抗噪机制。
- 豆子清晰可分成独立圆形个体，无粘连/模糊/过曝。
- 俯拍照（多数）排列规则、间距均匀；但确有**强透视照**（`IMG_6121`：梯形、近大远小、边缘间距≠中心）。

## 2. 核心思路：豆子驱动，板子消解

**砍掉"检测板子"这个独立阶段。** 板子的几何信息（方向、边界、截断）都从**豆子**里推导。板子尺寸能数出来就自动得到，截断时才需要用户补。

```
旧：检测板子 → 提角点 → 透视校正 → 切网格 → 比色
新：检测豆子 → 共线分组成行列 → 编号(row,col) → 构建 rows×cols 格子 → 比色
```

关键认识：

- **豆子检测不难**（深色环颜色无关）——所以不需要 RANSAC 那套抗噪。
- 我们要的不是"亚像素精确的单应矩阵 H"，而是**"每颗豆子分对 (row, col)"**。编号对了，LAB 容差内的颜色比对就能跑。
- **透视由"共线分组"天然吸收**：真实同一行的豆子，在任何透视下都共线（汇聚到灭点）；按共线分行分列，透视不再是"要拟合的难题"。
- `edge_refiner.py` 那 900 行整体删除；`PerspectiveCorrector` 降为显示用（主流程不再 warp 出校正图）。

### 2.1 为什么"只检测豆子"还不够半步

检测豆子给出**位置**；共线分组给出**编号 (row, col)**。逐格比对图纸必须知道每颗豆子是第几行第几列。这两步合起来才是完整的"映射进网格"——但都比单应重拟合简单得多，是流水线的两段，不是可选项。

## 3. 详细设计

### 3.1 新模块 `src/bead_grid.py`

核心类 `BeadGridFitter`，单一入口 `fit(image, board_size=None) -> GridResult`。

#### 3.1.1 数据结构

```python
@dataclass
class GridConfidence:
    bead_count: int            # 检出的豆子数
    grid_fill_ratio: float     # 已编号格子 / (rows×cols)
    labeling_residual: float   # 豆子落到整数格点的残差（越低越规整）
    perspective_tier: bool     # 是否触发了透视分级（强透视照）
    level: str                 # "高"/"中"/"低"


@dataclass
class TruncationInfo:
    """板子被画面截断（仅当 board_size 给定且检出范围 < 尺寸时才有意义）。"""
    is_truncated: bool
    clipped_edges: list[str]   # ["top"/"bottom"/"left"/"right"] 子集


@dataclass
class GridResult:
    rows: int
    cols: int
    cells: list[CellInfo]          # rows×cols，每格 (row,col,color,is_visible,is_edge,confidence)
    outline: np.ndarray | None     # 4×2，板子外轮廓点（显示用，可 None）
    confidence: GridConfidence
    truncation: TruncationInfo
```

> `CellInfo` 复用 `src/grid.py` 现有定义。`board_size` 不再是必填：能从检出网格数出来就自动，截断时才用用户给的尺寸补全。

#### 3.1.2 `fit()` 管线

```python
class BeadGridFitter:
    def fit(self, image, board_size=None):
        # ① 检测豆子（双圆环深色环特征）→ 中心点 + 颜色
        beads = self._detect_beads(image)
        if len(beads) < MIN_BEADS:
            raise GridFitError("检出豆子太少，无法分组")

        # ② 估网格方向 + 间距（近邻向量投票）
        d1, d2, spacing = self._estimate_grid_axes(beads)

        # ③ 行列编号（核心，见 3.2）：默认仿射；强透视分级到射影
        labels, perspective_tier = self._label_beads(beads, d1, d2, spacing)

        # ④ 定绝对偏移 + 板子尺寸（自动数行列；截断用 board_size）；空格插值
        rows, cols, abs_labels = self._resolve_dims_and_offset(labels, board_size, image.shape)

        # ⑤ 构建 rows×cols 格子（有豆取豆色，空格取插值/板底色）
        cells = self._build_cells(image, beads, abs_labels, rows, cols)
        confidence = self._evaluate_confidence(beads, cells, rows, cols, perspective_tier)
        return GridResult(rows, cols, cells, outline, confidence, truncation)
```

### 3.2 第 ③ 步：共线分组编号（核心）

目标：给每颗豆子一个整数 (row, col)，**任何透视下都分对**。分两级，先简后繁：

**默认级 —— 仿射分配（覆盖多数俯拍照）**

```python
def _label_beads(self, beads, d1, d2, spacing):
    origin = beads[左上极端点]
    for bead in beads:
        bead.row = round(dot(bead.xy - origin, d1) / spacing)
        bead.col = round(dot(bead.xy - origin, d2) / spacing)
    # 评估：多少豆子的小数残差大（落不进整数格）
    if 残差超阈值的豆子比例 < PERSPECTIVE_TRIGGER_RATIO:
        return labels, perspective_tier=False        # 仿射够用
    return self._label_projective(beads), perspective_tier=True
```

**透视级 —— 射影分配（强透视照，如 `IMG_6121`）**

仅当仿射残差大时触发。从仿射结果里取 4 个行列极端豆子作为板子四角，单次求单应：

```python
def _label_projective(self, beads):
    corners = 取行列极端的 4 颗豆子的图像坐标
    # 四角 ↔ (0,0),(rows,0),(rows,cols),(0,cols)
    H = cv2.findHomography(格点四角, 图像四角)[0]   # 单次 DLT，无 RANSAC/ICP
    for bead in beads:
        (r, c) = H⁻¹ · bead.xy; bead.row,c = round(r), round(c)
    return labels
```

> **没有 RANSAC 迭代、没有 ICP**。因为豆子检测干净（深色环颜色无关），不需要抗噪采样；4 个角豆子确定单应后，所有豆子的编号一次算清。

为什么两级都能分对：仿射级处理"近俯拍"（间距近似恒定）；射影级处理"倾斜"（4 角定单应，吸收近大远小）。真实同一行的豆子在两种模型下都被正确归到同一行。

### 3.3 各子步骤说明

| 步骤 | 方法 | 输出 |
|------|------|------|
| ① 检测豆子 | 灰度图上找**深色环**圆结构（HoughCircles 调 `bead_prelabel`，或梯度图 blob）；半径用较宽范围，编号阶段会剔除尺寸不合者。每颗同时采样中心色 | 豆子列表 [{xy, color}] |
| ② 估网格轴 | 每颗豆子取 k 近邻向量，角度直方图取两个 ≈90° 的峰 → d1,d2；中位长度 → spacing | d1, d2, spacing |
| ③ 行列编号 | 共线分组：默认仿射投影取整；残差大则 4 角单应射影（见 3.2） | 每豆相对 (row,col) + perspective_tier |
| ④ 定尺寸+偏移 | 数检出网格的行列范围 → rows,cols（板子拍全即得）；若给了 board_size 且检出范围更小 → 截断，按 board_size 补；空格（无豆位置）从相邻豆子插值定位 | rows, cols, 绝对 (row,col) |
| ⑤ 构格 | 有豆格子取豆色；空格取插值位置处板底色；越出画面/截断侧标不可见 | cells: list[CellInfo] |
| 置信度 | 豆子数 + 填充率 + 编号残差 + 是否透视分级 | GridConfidence |

### 3.4 对现有模块的改动

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/bead_grid.py` | **新增** | 核心模块，`BeadGridFitter` + 数据结构（~300–400 行） |
| `src/edge_refiner.py` | **删除** | 900 行启发式整体移除 |
| `src/detect.py` | **删除 `BoardDetector`** | 不留 legacy；板子模型路径不再被引用 |
| `src/cli.py` | 改 | 主流程改走 `BeadGridFitter.fit()`；`--board-size` 改为**可选**（能数出来就自动）；移除 `--legacy` 与板子 `--model` |
| `src/grid.py` | 简化 | `PerspectiveCorrector` 降为显示用；`GridExtractor.extract` 保留供显示，主流程改用 `GridResult.cells` |
| `src/compare.py` | 微调 | 输入吃 `GridResult.cells`；`compare_with_confidence`/`annotate_with_confidence` 沿用 |
| `src/bead_prelabel.py` | 复用 | `hough_circles_to_boxes()` 作为豆子检测基础（深色环适合 HoughCircles） |
| `src/blueprint.py` | 不动 | 图纸规范、好分割，`parse_blueprint` 已就绪 |
| `data/colors.json` | 不动 | 移除对板子颜色参考字段的引用（若有） |

### 3.5 错误处理与降级

原则：**宁可报"不确定"，也不输出错误编号/角点**——这正是旧管线最大毛病（edge_refiner 硬猜畸形四边形）。

| 失败场景 | 处理 |
|------|------|
| 检出豆子太少（< `MIN_BEADS`）| 报 `GridFitError`，提示"豆子太少/对焦不准/离近些拍" |
| 背景也有规则圆点阵列 | ② 的方向直方图取最强峰；若两组接近 → 报"检测到多个网格，请靠近板子拍" |
| 强透视但 4 角豆子缺失（角是空的）| 射影级降级回仿射 + 降置信度，提示"板角豆子缺失，结果可能偏差" |
| `board_size` 给错 | 数出的行列范围与给定严重不符 → 置信度暴跌，提示"尺寸可能有误" |
| 板子被截断 | 用用户提供的 board_size 外推缺失侧 → 标 truncation，该侧格子差异标低置信度；若未提供 board_size 则只比对可见部分 |
| 板子露出太少（2+ 相邻边截断）| 绝对偏移有歧义 → 明确报"板子露出太少，无法对齐图纸"，不硬猜 |

### 3.6 二期：peg 孔通道（本 spec 不实现）

peg 孔位是永远存在的规则网格，二期在第 ① 步叠加 peg 孔检测，提升"早期稀疏板"鲁棒性。一期先用真实回归集验证豆子通道是否足够。

## 4. 测试策略

- **合成网格（主力）**：程序生成已知网格 + 渲染双圆环豆子图 → 验证每颗豆子的 (row,col) 编号正确。系统覆盖：旋转角、**透视强度**（含 `IMG_6121` 级梯形）、截断每条边、加随机外点（背景）、缺失部分豆子（在拼状态）。
- **步级单测**：② 方向/间距估计、③ 仿射编号 + 射影分级、④ 尺寸/偏移求解，各自独立测。
- **透视分级测试**：构造仿射能搞定的弱透视 vs 必须射影的强透视，验证分级触发正确、两种都编号无误。
- **真实回归集**：用户提供的 **13 张**当前方案崩掉的照片（`/mnt/c/Users/AnInteger/Downloads/training/training/1/`，仓库内 `training/photos/` 有镜像但被 gitignore）。含俯拍、强透视（`IMG_6121`）、截断等多种条件。**最关键验收：新方案必须在旧方案失败的照片上工作。**
  - **Fixture 策略**：全 13 张体积大且含个人照片，不整批入库；从中挑代表性子集（俯拍/强透视/截断/稀疏各 1–2 张）复制进 `tests/fixtures/board_regression/`（提交，供 CI）；完整 13 张通过环境变量/路径在本地跑全量回归。
- **降级测试**：豆子太少、多网格、错误尺寸、角缺失 → 优雅报告而非崩溃。
- **不依赖模型文件**：纯经典 CV，单测全靠合成图，CI 友好。

## 5. 文件变更清单

| 文件 | 变更 |
|------|------|
| `src/bead_grid.py` | 新增（核心，~300–400 行） |
| `src/edge_refiner.py` | 删除 |
| `src/detect.py` | 删除 `BoardDetector` |
| `src/cli.py` | 改走新管线，`--board-size` 改可选，清理旧参数 |
| `src/grid.py` | `PerspectiveCorrector` 降级为显示用 |
| `src/compare.py` | 输入接口微调 |
| `tests/test_bead_grid.py` | 新增 |
| `tests/test_labeling.py` | 新增（③ 共线编号 + 透视分级专项） |
| `tests/test_grid_regression.py` | 新增（真实回归集，照片存 fixture） |

## 6. 依赖与实现

- **依赖**：无新增（numpy / opencv / scipy 均已在）。射影分级用 `cv2.findHomography`（4 点单次 DLT）。
- **实现分支**：`feat/bead-grid-detection`（实现阶段创建）。
- **废弃**：`2026-06-11-board-edge-refinement-design.md` 及 `edge_refiner.py`。
