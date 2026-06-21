# 珠子标注 UI（阶段 1）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重写珠子标注 UI，用 bead-grid 算法预标注 + 网格辅助补漏 + 框编辑 + 评估预览，产出 YOLO 训练标签；并为未来「训练检测器替换 HoughCircles」预留 `fit(detector=...)` 接缝。

**Architecture:** 服务/界面分离——纯逻辑放新模块 `src/bead_label_service.py`（可单测），`src/bead_annotate_ui.py` 只做薄 Gradio 接线。`src/bead_grid.py` 小幅扩展：`CellInfo` 加 `has_bead`、`GridResult` 加 `median_radius`、`fit()` 加可选 `detector` 参数（接缝）。真实照片输入读 `training/photos/`，标签导出到 `training/bead_dataset/{images,labels}/train/`。

**Tech Stack:** Python 3.10+, OpenCV (BGR), NumPy, Gradio, pytest（用 `tests/_synth.py` 合成夹具，无需模型文件）。

**Spec:** `docs/superpowers/specs/2026-06-21-bead-detect-training-loop-design.md`（本计划覆盖阶段 1 全部 + 阶段 3 的接缝预留；阶段 2 训练、`bead_split.py`、`cli.py` 接模型均不在本计划，待「评估关卡」决策后另立计划。）

---

## 文件结构

| 文件 | 责任 | 动作 |
|---|---|---|
| `src/grid.py` | `CellInfo` 数据结构 | 改：加 `has_bead` 字段 |
| `src/bead_grid.py` | bead-grid 拟合 | 改：`build_cells` 设 `has_bead`；`GridResult` 加 `median_radius`；`fit()` 加 `detector` 参数 + `_beads_from_boxes` |
| `src/bead_label_service.py` | 标注纯逻辑（预标注/补漏/导出/配色预览） | 新建 |
| `src/bead_annotate_ui.py` | Gradio 薄界面 | 重写 |
| `tests/test_bead_grid.py` | bead_grid 单测 | 扩展 |
| `tests/test_bead_label_service.py` | 服务层单测 | 新建 |

---

## Task 1: bead_grid 暴露 `has_bead` 与 `median_radius`

让标注服务能从 `GridResult` 推导「哪些格有豆子（→检测框）」和「中位半径（→框尺寸/补漏尺寸）」。

**Files:**
- Modify: `src/grid.py`（`CellInfo` 数据类）
- Modify: `src/bead_grid.py`（`build_cells`、`GridResult`、`fit`）
- Test: `tests/test_bead_grid.py`

- [ ] **Step 1: 写失败测试**（追加到 `tests/test_bead_grid.py` 末尾）

```python
def test_cellinfo_has_bead_default_false():
    c = CellInfo(row=0, col=0, color=np.zeros(3, dtype=np.uint8),
                 is_visible=True, is_edge=False, confidence=1.0)
    assert c.has_bead is False


def test_fit_sets_median_radius_and_has_bead():
    from src.bead_grid import BeadGridFitter
    centers = synth_grid_centers(5, 5, spacing=30.0, origin=(50, 50))
    img = render_beads(centers, img_size=(250, 250), bead_radius=10)
    res = BeadGridFitter().fit(img)
    assert res.median_radius > 0
    filled = [c for c in res.cells if c.has_bead]
    assert len(filled) >= 20
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_bead_grid.py::test_cellinfo_has_bead_default_false tests/test_bead_grid.py::test_fit_sets_median_radius_and_has_bead -v`
Expected: FAIL（`CellInfo` 无 `has_bead` 属性 / `GridResult` 无 `median_radius`）

- [ ] **Step 3: 给 `CellInfo` 加 `has_bead`**（`src/grid.py`，在 `image_xy` 字段后加）

```python
@dataclass
class CellInfo:
    """Metadata for a single grid cell."""
    row: int
    col: int
    color: np.ndarray          # RGB uint8
    is_visible: bool           # Cell centre is inside the visible region
    is_edge: bool              # Cell is near the truncation boundary
    confidence: float          # Comparison confidence for this cell [0, 1]
    image_xy: tuple[float, float] | None = None   # cell centre in original image
    has_bead: bool = False     # True if a detected bead landed in this cell
```

- [ ] **Step 4: `build_cells` 设置 `has_bead`**（`src/bead_grid.py`，改最后一处 `cells.append`）

把 `build_cells` 内三处 `cells.append(CellInfo(...))` 改为带 `has_bead`。有豆子的分支设 `True`，两处空格分支设 `False`：

```python
            if (r, c) in cell_bead:
                bead = cell_bead[(r, c)]
                color = bead.color
                is_visible = inside
                image_xy = (float(bead.xy[0]), float(bead.xy[1]))
            elif inside:
                color = _sample_color(image, xy)
                is_visible = True
                image_xy = (float(xy[0]), float(xy[1]))
            else:
                color = np.array([0, 0, 0], dtype=np.uint8)
                is_visible = False
                image_xy = None
            is_edge = _is_edge_cell(r, c, rows, cols, truncation)
            conf = 0.5 if is_edge else (1.0 if is_visible else 0.0)
            cells.append(CellInfo(row=r, col=c, color=color, is_visible=is_visible,
                                  is_edge=is_edge, confidence=conf, image_xy=image_xy,
                                  has_bead=((r, c) in cell_bead)))
```

- [ ] **Step 5: `GridResult` 加 `median_radius`，`fit()` 计算它**（`src/bead_grid.py`）

`GridResult` 末尾加字段：

```python
@dataclass
class GridResult:
    rows: int
    cols: int
    cells: list  # list[CellInfo]
    outline: np.ndarray | None
    confidence: GridConfidence
    truncation: TruncationInfo
    median_radius: float = 0.0
```

`fit()` 在 `beads` 取得后、返回前加计算，并传入 `GridResult`。把 `fit` 第一段改为：

```python
    def fit(self, image: np.ndarray, board_size: tuple[int, int] | None = None,
            detector=None) -> GridResult:
        if detector is not None:
            beads = _beads_from_boxes(image, detector.detect(image))
        else:
            beads = detect_beads(image)
        if len(beads) < MIN_BEADS:
            raise GridFitError(f"检出豆子太少({len(beads)})，无法分组")
        median_radius = float(np.median([b.radius for b in beads]))

        d_row, d_col, spacing = estimate_grid_axes(beads)
```

返回处补 `median_radius=median_radius`：

```python
        return GridResult(rows=rows, cols=cols, cells=cells, outline=outline,
                          confidence=confidence, truncation=trunc,
                          median_radius=median_radius)
```

（注：`detector` 参数与 `_beads_from_boxes` 在 Task 2 实现；本步先加占位 `if detector is not None` 分支会因 `_beads_from_boxes` 未定义而报错——所以**本步只加 `median_radius`，不要加 `detector` 分支**。`fit` 签名本步保持 `fit(self, image, board_size=None)`，Task 2 再加 `detector`。修正后的本步 `fit` 头部：）

```python
    def fit(self, image: np.ndarray, board_size: tuple[int, int] | None = None) -> GridResult:
        beads = detect_beads(image)
        if len(beads) < MIN_BEADS:
            raise GridFitError(f"检出豆子太少({len(beads)})，无法分组")
        median_radius = float(np.median([b.radius for b in beads]))

        d_row, d_col, spacing = estimate_grid_axes(beads)
        labels, persp = label_beads(beads, d_row, d_col, spacing)
        rows, cols, abs_labels, trunc = resolve_dims_and_offset(labels, board_size, image.shape)
        grid_map = fit_grid_map(beads, abs_labels, persp, d_row, d_col, spacing)
        cells = build_cells(image, beads, abs_labels, grid_map, rows, cols, trunc)
        confidence = evaluate_confidence(beads, cells, rows, cols, persp, grid_map)

        outline = np.array([
            grid_map.to_xy(0, 0),
            grid_map.to_xy(0, cols),
            grid_map.to_xy(rows, cols),
            grid_map.to_xy(rows, 0),
        ], dtype=np.float32)

        return GridResult(rows=rows, cols=cols, cells=cells, outline=outline,
                          confidence=confidence, truncation=trunc,
                          median_radius=median_radius)
```

- [ ] **Step 6: 跑测试确认通过**

Run: `python -m pytest tests/test_bead_grid.py -v`
Expected: PASS（含新两条；原有不回归）

- [ ] **Step 7: 提交**

```bash
git add src/grid.py src/bead_grid.py tests/test_bead_grid.py
git commit -m "feat(bead_grid): expose has_bead + median_radius for labeling"
```

---

## Task 2: `fit()` 检测可切换接缝（为训练后的检测器预留）

让 `BeadGridFitter.fit()` 接受可选 `detector`（任何带 `.detect(img)->list[box]` 的对象），下游全部逻辑不变。这是「未来用训练好的 BeadDetector 替换 HoughCircles」的接缝。

**Files:**
- Modify: `src/bead_grid.py`（`fit` 签名 + 新模块函数 `_beads_from_boxes`）
- Test: `tests/test_bead_grid.py`

- [ ] **Step 1: 写失败测试**（追加到 `tests/test_bead_grid.py` 末尾）

```python
def test_fit_uses_injected_detector():
    from src.bead_grid import BeadGridFitter
    centers = synth_grid_centers(5, 5, spacing=30.0, origin=(50, 50))
    img = render_beads(centers, img_size=(250, 250), bead_radius=10)

    class FakeDetector:
        def detect(self, image):
            return [{"xyxy": [int(x) - 10, int(y) - 10, int(x) + 10, int(y) + 10],
                     "cx": int(x), "cy": int(y), "width": 20, "height": 20, "conf": 1.0}
                    for x, y in centers]

    res = BeadGridFitter().fit(img, detector=FakeDetector())
    assert res.rows == 5 and res.cols == 5
    assert sum(1 for c in res.cells if c.has_bead) == 25
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_bead_grid.py::test_fit_uses_injected_detector -v`
Expected: FAIL（`fit()` 不接受 `detector` 参数）

- [ ] **Step 3: 加 `_beads_from_boxes` 模块函数**（`src/bead_grid.py`，放在 `detect_beads` 之后）

```python
def _beads_from_boxes(image: np.ndarray, boxes: list[dict]) -> list[Bead]:
    """Convert detection boxes (cx/cy/width/height) into Bead objects (center + color + radius)."""
    beads: list[Bead] = []
    for b in boxes:
        cx, cy = int(b["cx"]), int(b["cy"])
        r = float(max(b["width"], b["height"]) / 2.0)
        color = _sample_color(image, (cx, cy), half=max(1, int(r / 3)))
        beads.append(Bead(xy=np.array([cx, cy], dtype=np.float64),
                          color=color, radius=r))
    return beads
```

- [ ] **Step 4: `fit()` 加 `detector` 参数**（`src/bead_grid.py`，改 `fit` 头部）

```python
    def fit(self, image: np.ndarray, board_size: tuple[int, int] | None = None,
            detector=None) -> GridResult:
        if detector is not None:
            beads = _beads_from_boxes(image, detector.detect(image))
        else:
            beads = detect_beads(image)
        if len(beads) < MIN_BEADS:
            raise GridFitError(f"检出豆子太少({len(beads)})，无法分组")
        median_radius = float(np.median([b.radius for b in beads]))
```

（其余 `fit` 函数体不变。）

- [ ] **Step 5: 跑测试确认通过 + 不回归**

Run: `python -m pytest tests/test_bead_grid.py tests/test_grid_regression.py -v`
Expected: PASS（新测试过；回归不破）

- [ ] **Step 6: 提交**

```bash
git add src/bead_grid.py tests/test_bead_grid.py
git commit -m "feat(bead_grid): swappable detection seam via fit(detector=)"
```

---

## Task 3: 服务层——预标注 + 漏检洞计算

纯逻辑：照片 → `BeadGridFitter.fit()` → 检测框 + 中位半径 + 网格漏检洞；fit 失败（豆子太少）降级为仅检测框。

**Files:**
- Create: `src/bead_label_service.py`
- Test: `tests/test_bead_label_service.py`

- [ ] **Step 1: 写失败测试**（创建 `tests/test_bead_label_service.py`）

```python
import numpy as np
from tests._synth import synth_grid_centers, render_beads
from src.bead_label_service import prelabel


def test_prelabel_full_grid():
    centers = synth_grid_centers(5, 5, spacing=30.0, origin=(50, 50))
    img = render_beads(centers, img_size=(250, 250), bead_radius=10)
    r = prelabel(img)
    assert r.fit_ok is True
    assert r.median_radius > 0
    assert len(r.boxes) >= 20
    assert all(b["source"] == "detect" for b in r.boxes)


def test_prelabel_finds_interior_hole():
    # 5x5 网格，抠掉正中 (r=2,c=2)→图像坐标 (110,110) 的那颗，制造一个内部洞
    centers = synth_grid_centers(5, 5, spacing=30.0, origin=(50, 50))
    centers = np.array([c for c in centers
                        if not (abs(c[0] - 110.0) < 1 and abs(c[1] - 110.0) < 1)])
    img = render_beads(centers, img_size=(250, 250), bead_radius=10)
    r = prelabel(img)
    assert r.fit_ok is True
    hole_xy = np.array([h["xy"] for h in r.holes])
    assert len(hole_xy) >= 1
    # 正中那个洞应在 (110,110) 附近
    assert np.min(np.linalg.norm(hole_xy - np.array([110.0, 110.0]), axis=1)) < 8.0


def test_prelabel_degraded_when_too_few_beads():
    centers = synth_grid_centers(3, 3, spacing=30.0, origin=(40, 40))  # 9 < MIN_BEADS(20)
    img = render_beads(centers, img_size=(200, 200), bead_radius=10)
    r = prelabel(img)
    assert r.fit_ok is False
    assert r.holes == []
    assert len(r.boxes) >= 5
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_bead_label_service.py -v`
Expected: FAIL（`ModuleNotFoundError: src.bead_label_service`）

- [ ] **Step 3: 实现 `src/bead_label_service.py`（预标注部分）**

```python
"""Bead annotation service — pure logic: prelabel, hole fill, export, color preview.

UI-agnostic so it can be unit-tested without Gradio or model files.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .bead_grid import BeadGridFitter, GridFitError, GridResult, detect_beads
from .bead_prelabel import save_yolo_boxes


@dataclass
class PrelabelResult:
    boxes: list[dict]           # detected boxes (source="detect")
    median_radius: float
    holes: list[dict]           # predicted bead positions for empty interior cells
    result: GridResult | None   # full grid result (None when fit failed)
    fit_ok: bool


def _box_from_xy(xy, radius: float, source: str) -> dict:
    cx, cy = int(round(xy[0])), int(round(xy[1]))
    r = int(round(radius))
    return {"xyxy": [cx - r, cy - r, cx + r, cy + r],
            "cx": cx, "cy": cy, "width": 2 * r, "height": 2 * r, "source": source}


def _compute_holes(result: GridResult) -> list[dict]:
    """Interior, visible, non-edge cells with no bead → predicted fill positions."""
    holes: list[dict] = []
    for c in result.cells:
        if c.has_bead or c.is_edge or not c.is_visible or c.image_xy is None:
            continue
        holes.append({"row": c.row, "col": c.col, "xy": c.image_xy,
                      "radius": result.median_radius})
    return holes


def prelabel(image: np.ndarray) -> PrelabelResult:
    """Run bead-grid fit; return detected boxes + median radius + holes.

    Degrades gracefully: if too few beads to fit a grid, returns detected boxes only.
    """
    try:
        result = BeadGridFitter().fit(image)
        boxes = [_box_from_xy(c.image_xy, result.median_radius, "detect")
                 for c in result.cells if c.has_bead]
        holes = _compute_holes(result)
        return PrelabelResult(boxes=boxes, median_radius=result.median_radius,
                              holes=holes, result=result, fit_ok=True)
    except GridFitError:
        beads = detect_beads(image)
        boxes = [_box_from_xy(b.xy, b.radius, "detect") for b in beads]
        med = float(np.median([b.radius for b in beads])) if beads else 0.0
        return PrelabelResult(boxes=boxes, median_radius=med, holes=[],
                              result=None, fit_ok=False)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_bead_label_service.py -v`
Expected: PASS（三条全过）

- [ ] **Step 5: 提交**

```bash
git add src/bead_label_service.py tests/test_bead_label_service.py
git commit -m "feat(bead_label_service): prelabel + interior hole detection"
```

---

## Task 4: 服务层——网格补漏 + YOLO 导出

把洞转成「autofill」框；把任意框集合导出为 YOLO 检测标签（单类别 `0`）。

**Files:**
- Modify: `src/bead_label_service.py`
- Test: `tests/test_bead_label_service.py`

- [ ] **Step 1: 写失败测试**（追加到 `tests/test_bead_label_service.py`）

```python
import os
from src.bead_label_service import holes_to_boxes, export_yolo


def test_holes_to_boxes_marks_autofill():
    holes = [{"row": 2, "col": 2, "xy": (110.0, 110.0), "radius": 10.0}]
    boxes = holes_to_boxes(holes)
    assert len(boxes) == 1
    b = boxes[0]
    assert b["source"] == "autofill"
    assert b["cx"] == 110 and b["cy"] == 110
    assert b["width"] == 20 and b["height"] == 20


def test_export_yolo_writes_valid_labels(tmp_path):
    img = render_beads(synth_grid_centers(5, 5, spacing=30.0, origin=(50, 50)),
                       img_size=(250, 250), bead_radius=10)
    boxes = [{"xyxy": [10, 10, 30, 30], "cx": 20, "cy": 20,
              "width": 20, "height": 20, "source": "detect"}]
    images_dir = tmp_path / "images" / "train"
    labels_dir = tmp_path / "labels" / "train"
    img_path, lbl_path, n = export_yolo(img, boxes, "shot1",
                                        str(images_dir), str(labels_dir))
    assert n == 1
    assert os.path.exists(img_path) and os.path.exists(lbl_path)
    parts = open(lbl_path).read().strip().split()
    assert parts[0] == "0"          # 单类别 bead
    assert len(parts) == 5          # class cx cy w h
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_bead_label_service.py::test_holes_to_boxes_marks_autofill tests/test_bead_label_service.py::test_export_yolo_writes_valid_labels -v`
Expected: FAIL（`ImportError: cannot import name 'holes_to_boxes'`）

- [ ] **Step 3: 实现**（追加到 `src/bead_label_service.py`）

```python
def holes_to_boxes(holes: list[dict]) -> list[dict]:
    """Convert predicted hole positions into autofill boxes (source='autofill')."""
    return [_box_from_xy(h["xy"], h["radius"], "autofill") for h in holes]


def export_yolo(image: np.ndarray, boxes: list[dict], name: str,
                images_dir: str, labels_dir: str) -> tuple[str, str, int]:
    """Write image (jpg) + YOLO detection labels (single class 0). Returns (img_path, label_path, n)."""
    import os
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    h, w = image.shape[:2]
    img_path = os.path.join(images_dir, name + ".jpg")
    cv2.imwrite(img_path, image)
    label_path = os.path.join(labels_dir, name + ".txt")
    save_yolo_boxes(boxes, w, h, label_path)
    return img_path, label_path, len(boxes)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_bead_label_service.py -v`
Expected: PASS（五条全过）

- [ ] **Step 5: 提交**

```bash
git add src/bead_label_service.py tests/test_bead_label_service.py
git commit -m "feat(bead_label_service): autofill + YOLO export"
```

---

## Task 5: 服务层——评估预览（每格匹配色）

为 spec 的「评估关卡」提供数据：把每个可见格的采样色经 `ColorMatcher` 匹配到色板，返回 `{位置, 色名, 色板RGB}` 供 UI 叠加显示，让用户直观判断算法效果。

**Files:**
- Modify: `src/bead_label_service.py`
- Test: `tests/test_bead_label_service.py`

- [ ] **Step 1: 写失败测试**（追加到 `tests/test_bead_label_service.py`）

```python
from src.bead_label_service import prelabel, match_cell_colors
from src.color import ColorMatcher
from src.bead_grid import BeadGridFitter


def test_match_cell_colors_returns_palette_entries():
    centers = synth_grid_centers(5, 5, spacing=30.0, origin=(50, 50))
    img = render_beads(centers, img_size=(250, 250), bead_radius=10,
                       body_color=(0, 0, 255))   # BGR 红
    res = BeadGridFitter().fit(img)
    matched = match_cell_colors(res, ColorMatcher())
    assert len(matched) > 0
    m = matched[0]
    assert {"row", "col", "xy", "name", "rgb"} <= set(m.keys())
    assert isinstance(m["name"], str)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_bead_label_service.py::test_match_cell_colors_returns_palette_entries -v`
Expected: FAIL（`ImportError: cannot import name 'match_cell_colors'`）

- [ ] **Step 3: 实现**（追加到 `src/bead_label_service.py`）

```python
def match_cell_colors(result: GridResult, color_matcher) -> list[dict]:
    """Map each visible cell's sampled color to a palette entry (for eval-preview overlay)."""
    out: list[dict] = []
    for c in result.cells:
        if not c.is_visible or c.image_xy is None:
            continue
        m = color_matcher.match(c.color.tolist())
        out.append({"row": c.row, "col": c.col, "xy": c.image_xy,
                    "name": m["name"], "rgb": m["rgb"]})
    return out
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_bead_label_service.py -v`
Expected: PASS（六条全过）

- [ ] **Step 5: 提交**

```bash
git add src/bead_label_service.py tests/test_bead_label_service.py
git commit -m "feat(bead_label_service): palette color preview for evaluation"
```

---

## Task 6: 重写标注 UI（Gradio）

把服务层接到薄 Gradio 界面：加载照片 → 预标注（框+网格+可选配色叠加）→ 网格补漏 → 删框（勾选）/ 加框（点击）→ 导出 YOLO。框按来源上色：🟢检出 / 🟡网格补 / 🔵手动。

**Files:**
- Rewrite: `src/bead_annotate_ui.py`

- [ ] **Step 1: 整体重写 `src/bead_annotate_ui.py`**

```python
"""Gradio UI for bead annotation — bead-grid prelabel + grid-assisted fill + box edit + export.

Thin UI over src.bead_label_service. Reads photos from training/photos/, exports YOLO
detection labels to training/bead_dataset/{images,labels}/train/.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import gradio as gr
import numpy as np

from src.bead_label_service import prelabel, holes_to_boxes, export_yolo, match_cell_colors
from src.color import ColorMatcher

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHOTOS_DIR = os.path.join(BASE_DIR, "training", "photos")
DATASET_DIR = os.path.join(BASE_DIR, "training", "bead_dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images", "train")
LABELS_DIR = os.path.join(DATASET_DIR, "labels", "train")

_cm = None


def _color_matcher():
    global _cm
    if _cm is None:
        _cm = ColorMatcher()
    return _cm


# box source -> BGR draw color
SRC_BGR = {"detect": (0, 255, 0), "autofill": (0, 255, 255), "manual": (255, 128, 0)}


def _list_photos():
    if not os.path.isdir(PHOTOS_DIR):
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(f for f in os.listdir(PHOTOS_DIR) if os.path.splitext(f)[1].lower() in exts)


def _stats():
    n = len(os.listdir(LABELS_DIR)) if os.path.isdir(LABELS_DIR) else 0
    return f"数据集: {n} 张已标注 / {len(_list_photos())} 张照片"


def _box_label(i, b):
    return f"#{i} {b['source']} @({b['cx']},{b['cy']})"


def _choices(state):
    return [_box_label(i, b) for i, b in enumerate(state["boxes"])]


def _draw(state, show_color):
    img = state.get("img_bgr")
    if img is None:
        return None
    disp = img.copy()
    if show_color and state.get("result") is not None:
        for cc in match_cell_colors(state["result"], _color_matcher()):
            x, y = cc["xy"]
            bgr = [int(v) for v in reversed(cc["rgb"])]
            cv2.circle(disp, (int(x), int(y)), 5, bgr, -1)
    res = state.get("result")
    if res is not None and getattr(res, "outline", None) is not None:
        cv2.polylines(disp, [res.outline.astype(np.int32)], True, (0, 0, 255), 2)
    for b in state["boxes"]:
        x1, y1, x2, y2 = b["xyxy"]
        cv2.rectangle(disp, (x1, y1), (x2, y2), SRC_BGR.get(b["source"], (255, 255, 255)), 2)
    return cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)


def _new_state(img_bgr=None, name=""):
    return {"img_bgr": img_bgr, "boxes": [], "holes": [], "result": None, "name": name}


def h_load(name, state):
    path = os.path.join(PHOTOS_DIR, name or "")
    img = cv2.imread(path) if name else None
    if img is None:
        return None, _new_state(), f"❌ 读不到 {name}", gr.update(), _stats()
    state = _new_state(img, os.path.splitext(name)[0])
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), state, f"✅ 已加载 {name}", gr.update(choices=_choices(state)), _stats()


def h_prelabel(state, show_color):
    img = state.get("img_bgr")
    if img is None:
        return None, state, "❌ 请先加载照片", gr.update(), _stats()
    r = prelabel(img)
    state["boxes"] = r.boxes
    state["holes"] = r.holes
    state["result"] = r.result
    if r.fit_ok:
        msg = f"✅ 检出 {len(r.boxes)} 颗；可补漏 {len(r.holes)} 处"
    else:
        msg = f"⚠️ 检出 {len(r.boxes)} 颗，豆子太少无法拟合网格（仅显示框）"
    return _draw(state, show_color), state, msg, gr.update(choices=_choices(state)), _stats()


def h_autofill(state, show_color):
    if not state.get("holes"):
        return _draw(state, show_color), state, "⚠️ 无可补漏（先预标注）", gr.update(), _stats()
    state["boxes"].extend(holes_to_boxes(state["holes"]))
    state["holes"] = []
    return _draw(state, show_color), state, f"✅ 已补漏，共 {len(state['boxes'])} 个框", gr.update(choices=_choices(state)), _stats()


def h_delete(state, sel, show_color):
    sel = sel or []
    keep = [b for i, b in enumerate(state["boxes"]) if _box_label(i, b) not in sel]
    state["boxes"] = keep
    return _draw(state, show_color), state, f"剩余 {len(keep)} 个框", gr.update(choices=_choices(state)), _stats()


def h_add_click(evt, state, radius, show_color):
    img = state.get("img_bgr")
    if img is None or evt is None:
        return _draw(state, show_color), state, gr.update()
    x, y = evt.index
    cx, cy = int(round(x)), int(round(y))
    r = int(radius)
    state["boxes"].append({"xyxy": [cx - r, cy - r, cx + r, cy + r],
                           "cx": cx, "cy": cy, "width": 2 * r, "height": 2 * r, "source": "manual"})
    return _draw(state, show_color), state, gr.update(choices=_choices(state))


def h_export(state, name_override):
    img = state.get("img_bgr")
    if img is None:
        return "❌ 未加载照片", _stats()
    if not state["boxes"]:
        return "❌ 无框可导出（先预标注/补漏/加框）", _stats()
    name = (name_override or state.get("name") or "").strip()
    if not name:
        return "❌ 缺导出名", _stats()
    img_path, lbl_path, n = export_yolo(img, state["boxes"], name, IMAGES_DIR, LABELS_DIR)
    return f"✅ 导出 {n} 框 → {lbl_path}", _stats()


def build_ui():
    with gr.Blocks(title="Pintconut 珠子标注") as app:
        gr.Markdown("# 🟡 Pintconut 珠子标注 (bead-grid)\n"
                    "加载 → 预标注 → (可选)网格补漏 → 校正 → 导出。  "
                    "框色：🟢检出 / 🟡网格补 / 🔵手动。勾选「叠加匹配色」可评估算法效果。")
        state = gr.State(_new_state())

        with gr.Row():
            with gr.Column(scale=1):
                photo = gr.Dropdown(choices=_list_photos(), label="照片 (training/photos/)")
                load_btn = gr.Button("📥 加载")
                show_color = gr.Checkbox(value=True, label="叠加匹配色(评估)")
                prelabel_btn = gr.Button("🔍 预标注", variant="primary")
                autofill_btn = gr.Button("🟡 网格补漏")
                box_list = gr.CheckboxGroup(choices=[], label="框列表 (勾选→删除)")
                delete_btn = gr.Button("🗑️ 删除选中")
                radius = gr.Number(value=10, label="点击加框半径(px)")
                name_tb = gr.Textbox(label="导出名 (留空用照片名)")
                export_btn = gr.Button("💾 导出 YOLO", variant="secondary")
            with gr.Column(scale=1):
                canvas = gr.Image(label="标注画布 (点击=加🔵框)", type="numpy")
                status = gr.Textbox(label="状态", interactive=False)
                stats = gr.Textbox(label="统计", interactive=False)

        # 先声明所有组件，再统一接线
        load_btn.click(h_load, [photo, state], [canvas, state, status, box_list, stats])
        prelabel_btn.click(h_prelabel, [state, show_color], [canvas, state, status, box_list, stats])
        autofill_btn.click(h_autofill, [state, show_color], [canvas, state, status, box_list, stats])
        delete_btn.click(h_delete, [state, box_list, show_color], [canvas, state, status, box_list, stats])
        canvas.select(h_add_click, [state, radius, show_color], [canvas, state, box_list])
        export_btn.click(h_export, [state, name_tb], [status, stats])
        app.load(fn=lambda: gr.update(choices=_list_photos()), outputs=photo)
        app.load(fn=_stats, outputs=stats)
    return app


if __name__ == "__main__":
    build_ui().launch(share=False)
```

- [ ] **Step 2: 冒烟测试——服务层全绿先确认**

Run: `python -m pytest tests/test_bead_label_service.py tests/test_bead_grid.py -v`
Expected: PASS（UI 不影响服务层测试）

- [ ] **Step 3: 冒烟测试——启动 UI 手动走一遍**

Run: `python src/bead_annotate_ui.py`
然后在浏览器 http://localhost:7860 手动验证（每条都要亲测，这是 UI 的「测试」）：

1. **加载**：下拉选一张 `training/photos/` 里的照片 → 点「📥 加载」→ 画布显示原图，状态「✅ 已加载」。
2. **预标注**：点「🔍 预标注」→ 画布出现🟢绿框 + 红色网格轮廓；若勾选「叠加匹配色」，每颗豆子上叠加色板色点；状态显示检出数与可补漏数。
3. **评估观察**（spec 的评估关卡）：看绿框是否对准每颗豆子、网格轮廓是否贴合板子、配色点是否合理——判断算法效果。**这是判断「要不要训练检测器」的依据。**
4. **网格补漏**：点「🟡 网格补漏」→ 漏检处出现🟡黄框，框列表增加。
5. **删框**：在「框列表」勾选 1~2 个误检 → 点「🗑️ 删除选中」→ 对应框消失。
6. **加框**：在画布上点一个空位 → 该处出现🔵蓝框（半径=左侧数值）。
7. **导出**：填导出名 → 点「💾 导出 YOLO」→ 状态显示路径；检查 `training/bead_dataset/images/train/<名>.jpg` 与 `labels/train/<名>.txt` 已生成，`.txt` 每行 `0 cx cy w h`。
8. **降级路径**：加载一张豆子很少的照片预标注 → 状态应显示「⚠️ ...豆子太少无法拟合网格」，仍显示检测框、不崩。

Expected: 1~8 全部行为符合描述。**第 3 步是本阶段的核心产出——据此决定是否进入训练阶段。**

- [ ] **Step 4: 提交**

```bash
git add src/bead_annotate_ui.py
git commit -m "feat(bead_annotate_ui): bead-grid prelabel + grid-assisted editing + eval preview"
```

---

## Self-Review

**1. Spec 覆盖**
- 4.1 检测可切换 → Task 2 ✓
- 4.2 服务层（prelabel/补漏/export/配色）→ Task 3/4/5 ✓
- 4.3 标注 UI（预标注/补漏/框编辑/导出/评估预览/降级）→ Task 6 ✓
- 4.4 数据流（导出到 train/）→ Task 4/6 ✓；train/valid 划分（`bead_split.py`）→ **不在本计划**（属阶段 2，待评估关卡后）
- 4.5 训练/接入 → **不在本计划**（接缝已在 Task 2 预留；`cli.py` 接模型、`bead_train` 属阶段 2/3）
- 第 3 节「评估关卡」→ Task 6 Step 3.3 是该关卡的执行点 ✓
- 第 5 节错误处理：fit 降级（Task 3/6 ✓）、空框拒导出（Task 6 h_export ✓）

**2. 占位符扫描**：无 TBD/TODO；每个代码步均含完整代码与命令。

**3. 类型一致性**：`PrelabelResult.{boxes,holes,result,fit_ok}`、`_box_from_xy` 产出的 box schema（`xyxy/cx/cy/width/height/source`）、`holes` schema（`row/col/xy/radius`）在 Task 3/4/5/6 间一致；`CellInfo.has_bead`、`GridResult.median_radius` 在 Task 1 定义后被 Task 2/3/5 使用，命名统一。`fit(detector=)` 在 Task 2 定义、Task 6 不直接用（Phase 1 走 HoughCircles）。

**范围声明**：本计划 = 阶段 1（标注 UI）+ 阶段 3 接缝预留。训练、划分、`cli.py` 接模型是独立后续计划，依赖 Task 6 Step 3.3「评估关卡」的决策结果。
