# 豆子网格驱动检测 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用"检测豆子 → 共线分组编号 → 逐格比色"的经典 CV 管线，替换掉不可靠的 YOLOv8-seg 板子分割 + `edge_refiner` 启发式精化。

**Architecture:** 不再"检测板子"。直接检测每颗豆子（双圆环深色边缘，颜色无关）得到中心点云；按共线关系分成行/列并编号 (row, col)（默认仿射，强透视时用全点单次 DLT 单应，无 RANSAC/ICP）；构建 rows×cols 格子网格与图纸逐格比色。板子尺寸能数出来就自动，截断时才需用户给。

**Tech Stack:** Python 3.10+, OpenCV (`cv2.findHomography`, HoughCircles), NumPy, SciPy (`cKDTree`), pytest。复用 `src/bead_prelabel.hough_circles_to_boxes` 与 `src/compare.py`。

**Spec:** `docs/superpowers/specs/2026-06-17-bead-grid-detection-design.md`

**执行前：**
- 创建分支 `feat/bead-grid-detection`（`git checkout -b feat/bead-grid-detection`）。
- 所有 commit message 末尾加 `Co-Authored-By: Claude <noreply@anthropic.com>`。
- 常量门槛值（`MIN_BEADS`、`AFFINE_OK_THRESHOLD` 等）是初始值，Task 15 会对照回归集调整。

**关键不变量（贯穿所有任务）：** `fit_grid_map()` 在偏移归一化**之后**、用**绝对** abs_labels 拟合映射，保证 `grid_map.to_xy(r, c)` 对任意绝对格点 (r, c) 都给出正确图像坐标。`build_cells` 只用绝对标签 + 这个绝对映射。

**File Structure:**

| 文件 | 职责 |
|------|------|
| `src/bead_grid.py`（新增） | `BeadGridFitter` + 数据结构 + 各步纯函数（检测/估轴/编号/定维/映射/构格/置信） |
| `src/bead_prelabel.py`（复用） | `hough_circles_to_boxes`——豆子圆检测 |
| `src/grid.py`（改） | `CellInfo` 增 `image_xy` 字段；`PerspectiveCorrector` 不再被主流程调用（保留） |
| `src/compare.py`（改） | `DiffResult` 增 `image_xy`；`annotate_with_confidence` 在原图按位置画标记 |
| `src/cli.py`（改） | 走 `BeadGridFitter.fit`；`--board-size` 改可选；删 `--legacy` 与板子 `--model` |
| `src/edge_refiner.py`（删） | 整文件删除 |
| `src/detect.py`（删） | 删除整个文件（仅含 `BoardDetector`） |
| `tests/_synth.py`（新增） | 合成网格/豆子图测试助手 |
| `tests/test_bead_grid.py`（新增） | 各步单测 + fit 集成 |
| `tests/test_labeling.py`（新增） | 编号与透视分级专项 |
| `tests/test_grid_regression.py`（新增） | 真实照片回归（子集入库） |
| `tests/fixtures/board_regression/`（新增） | 回归照片子集 |

---

## Task 1: 数据结构与异常

**Files:**
- Create: `src/bead_grid.py`
- Modify: `src/grid.py`（`CellInfo` 增字段）
- Test: `tests/test_bead_grid.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_bead_grid.py
import numpy as np
from src.bead_grid import (
    Bead, GridConfidence, TruncationInfo, GridResult, GridFitError,
    AffineMap, ProjectiveMap,
)
from src.grid import CellInfo


def test_bead_dataclass():
    b = Bead(xy=np.array([1.0, 2.0]), color=np.array([10, 20, 30], dtype=np.uint8), radius=5.0)
    assert b.xy.shape == (2,)
    assert b.color.dtype == np.uint8

def test_cellinfo_has_image_xy():
    c = CellInfo(row=0, col=0, color=np.zeros(3, dtype=np.uint8),
                 is_visible=True, is_edge=False, confidence=1.0)
    assert c.image_xy is None  # 新字段，默认 None

def test_affine_map_to_xy():
    m = AffineMap(origin=np.array([10.0, 20.0]),
                  d_row=np.array([0.0, 1.0]), d_col=np.array([1.0, 0.0]), spacing=5.0)
    # to_xy(r, c) = origin + r*spacing*d_row + c*spacing*d_col
    np.testing.assert_allclose(m.to_xy(2, 3), np.array([10.0 + 3 * 5.0, 20.0 + 2 * 5.0]))

def test_projective_map_to_xy_identity():
    m = ProjectiveMap(H=np.eye(3, dtype=np.float64))
    np.testing.assert_allclose(m.to_xy(2, 3), np.array([2.0, 3.0]))

def test_gridresult_constructs():
    gr = GridResult(rows=2, cols=2, cells=[], outline=None,
                    confidence=GridConfidence(0, 0.0, 0.0, False, "高"),
                    truncation=TruncationInfo(False, []))
    assert gr.rows == 2

def test_gridfiterror_is_exception():
    assert issubclass(GridFitError, Exception)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_bead_grid.py -v`
Expected: FAIL — `cannot import name 'Bead' from 'src.bead_grid'`

- [ ] **Step 3: 写最小实现**

```python
# src/bead_grid.py
"""Bead-grid-driven board detection (replaces YOLOv8-seg + edge_refiner)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class GridFitError(Exception):
    """Bead grid fitting failed."""


@dataclass
class Bead:
    xy: np.ndarray              # (2,) float64 image coords
    color: np.ndarray           # (3,) uint8 RGB
    radius: float


@dataclass
class AffineMap:
    """Absolute lattice (row, col) -> image (x, y): origin + two oriented unit axes."""
    origin: np.ndarray
    d_row: np.ndarray           # unit, points downward
    d_col: np.ndarray           # unit, points rightward
    spacing: float

    def to_xy(self, r: int, c: int) -> np.ndarray:
        return self.origin + r * self.spacing * self.d_row + c * self.spacing * self.d_col


@dataclass
class ProjectiveMap:
    """Absolute lattice (row, col) -> image (x, y) via homography H."""
    H: np.ndarray               # 3x3, maps (row, col, 1) -> (x, y, 1)

    def to_xy(self, r: int, c: int) -> np.ndarray:
        v = self.H @ np.array([r, c, 1.0], dtype=np.float64)
        return v[:2] / v[2]


@dataclass
class GridConfidence:
    bead_count: int
    grid_fill_ratio: float
    labeling_residual: float
    perspective_tier: bool
    level: str


@dataclass
class TruncationInfo:
    is_truncated: bool
    clipped_edges: list[str]


@dataclass
class GridResult:
    rows: int
    cols: int
    cells: list  # list[CellInfo]
    outline: np.ndarray | None
    confidence: GridConfidence
    truncation: TruncationInfo
```

Modify `src/grid.py` `CellInfo` — add optional `image_xy`:

```python
@dataclass
class CellInfo:
    """Metadata for a single grid cell."""
    row: int
    col: int
    color: np.ndarray          # RGB uint8
    is_visible: bool
    is_edge: bool
    confidence: float
    image_xy: tuple[float, float] | None = None   # NEW: cell centre in original image
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_bead_grid.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: 提交**

```bash
git add src/bead_grid.py src/grid.py tests/test_bead_grid.py
git commit -m "feat(bead_grid): add data structures for bead-grid detection"
```

---

## Task 2: 合成测试助手

**Files:**
- Create: `tests/_synth.py`
- Test: `tests/test_bead_grid.py`（追加）

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_bead_grid.py`：

```python
from tests._synth import synth_grid_centers, apply_homography, render_beads, make_beads


def test_synth_grid_shape_and_spacing():
    pts = synth_grid_centers(rows=3, cols=4, spacing=20.0, origin=(10.0, 10.0))
    assert pts.shape == (12, 2)
    np.testing.assert_allclose(pts[1] - pts[0], [20.0, 0.0], atol=1e-9)   # 相邻列
    np.testing.assert_allclose(pts[4] - pts[0], [0.0, 20.0], atol=1e-9)   # 相邻行

def test_synth_grid_rotation():
    pts = synth_grid_centers(2, 2, spacing=10.0, origin=(0.0, 0.0), angle=np.pi / 2)
    np.testing.assert_allclose(pts[1], [0.0, 10.0], atol=1e-9)

def test_apply_homography_identity():
    pts = synth_grid_centers(2, 2)
    np.testing.assert_allclose(apply_homography(pts, np.eye(3)), pts, atol=1e-9)

def test_render_beads_shape():
    img = render_beads(synth_grid_centers(3, 3), img_size=(200, 200))
    assert img.shape == (200, 200, 3)

def test_make_beads():
    beads = make_beads(synth_grid_centers(2, 2))
    assert len(beads) == 4
    assert beads[0].xy.shape == (2,)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_bead_grid.py -v`
Expected: FAIL — `No module named 'tests._synth'`

- [ ] **Step 3: 写实现**

```python
# tests/_synth.py
"""Synthetic fixtures for bead-grid tests."""
import numpy as np
import cv2


def synth_grid_centers(rows, cols, spacing=20.0, origin=(50.0, 50.0), angle=0.0):
    """Affine grid of (x, y) bead centres. local = (c*spacing, r*spacing), rotated by angle."""
    o = np.array(origin, dtype=np.float64)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    pts = []
    for r in range(rows):
        for c in range(cols):
            local = np.array([c * spacing, r * spacing])
            pts.append(o + R @ local)
    return np.array(pts, dtype=np.float64)


def apply_homography(pts, H):
    pts = np.asarray(pts, dtype=np.float64)
    ones = np.ones((len(pts), 1))
    ph = np.hstack([pts, ones])
    out = (H @ ph.T).T
    return out[:, :2] / out[:, 2:3]


def render_beads(centers, img_size=(600, 600), bead_radius=8, ring_thickness=2,
                 body_color=(60, 60, 180), base_color=(235, 235, 235)):
    """Render double-ring beads (dark ring + coloured body) on a light board base."""
    img = np.full((img_size[0], img_size[1], 3), base_color, dtype=np.uint8)
    for (x, y) in centers:
        cx, cy = int(round(x)), int(round(y))
        cv2.circle(img, (cx, cy), bead_radius, (40, 40, 40), ring_thickness)              # dark ring
        inner = max(1, bead_radius - ring_thickness)
        cv2.circle(img, (cx, cy), inner, body_color, -1)                                  # body
    return img


def make_beads(centers, color=None, radius=8.0):
    from src.bead_grid import Bead
    col = np.array(color if color is not None else [60, 60, 180], dtype=np.uint8)
    return [Bead(xy=np.array(c, dtype=np.float64), color=col.copy(), radius=radius)
            for c in centers]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_bead_grid.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: 提交**

```bash
git add tests/_synth.py tests/test_bead_grid.py
git commit -m "test(bead_grid): add synthetic grid/bead fixtures"
```

---

## Task 3: 豆子检测 `detect_beads`

**Files:**
- Modify: `src/bead_grid.py`
- Test: `tests/test_bead_grid.py`（追加）

- [ ] **Step 1: 写失败测试**

```python
from src.bead_grid import detect_beads


def test_detect_beads_finds_grid():
    centers = synth_grid_centers(5, 5, spacing=30.0, origin=(40, 40))
    img = render_beads(centers, img_size=(250, 250), bead_radius=10)
    beads = detect_beads(img)
    assert len(beads) >= 20
    detected = np.array([b.xy for b in beads])
    close = sum(1 for t in centers
                if np.min(np.linalg.norm(detected - t, axis=1)) < 6.0)
    assert close >= int(0.8 * len(centers))

def test_detect_beads_empty_image():
    img = np.full((100, 100, 3), 235, dtype=np.uint8)
    assert detect_beads(img) == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_bead_grid.py -k detect_beads -v`
Expected: FAIL — `cannot import name 'detect_beads'`

- [ ] **Step 3: 写实现**

追加到 `src/bead_grid.py`：

```python
import cv2

from .bead_prelabel import hough_circles_to_boxes
from .grid import CellInfo


MIN_BEADS = 20


def _sample_color(image, xy, half=3):
    h, w = image.shape[:2]
    x, y = int(round(xy[0])), int(round(xy[1]))
    y1, y2 = max(0, y - half), min(h, y + half + 1)
    x1, x2 = max(0, x - half), min(w, x + half + 1)
    region = image[y1:y2, x1:x2]
    if region.size == 0:
        return np.array([0, 0, 0], dtype=np.uint8)
    bgr = np.median(region.reshape(-1, 3), axis=0)
    return cv2.cvtColor(bgr.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2RGB)[0, 0]


def detect_beads(image: np.ndarray) -> list[Bead]:
    """Detect beads via HoughCircles (dark ring) + median-size filter + colour sample."""
    h, w = image.shape[:2]
    max_r = max(4, int(min(h, w) / 20))
    min_r = max(2, max_r // 6)
    boxes = hough_circles_to_boxes(
        image, min_radius=min_r, max_radius=max_r,
        min_dist=max(6, max_r), param2=30,
    )
    if not boxes:
        return []
    radii = np.array([b["width"] / 2.0 for b in boxes])
    med = float(np.median(radii))
    beads: list[Bead] = []
    for b, r in zip(boxes, radii):
        if not (med * 0.5 <= r <= med * 2.0):
            continue
        cx, cy = int(b["cx"]), int(b["cy"])
        color = _sample_color(image, (cx, cy), half=max(1, int(r / 3)))
        beads.append(Bead(xy=np.array([cx, cy], dtype=np.float64),
                          color=color, radius=float(r)))
    return beads
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_bead_grid.py -k detect_beads -v`
Expected: PASS (2 tests)

- [ ] **Step 5: 提交**

```bash
git add src/bead_grid.py tests/test_bead_grid.py
git commit -m "feat(bead_grid): detect beads via HoughCircles + size filter"
```

---

## Task 4: 网格方向/间距估计 `estimate_grid_axes`

**Files:**
- Modify: `src/bead_grid.py`
- Test: `tests/test_bead_grid.py`（追加）

- [ ] **Step 1: 写失败测试**

```python
from src.bead_grid import estimate_grid_axes


def test_axes_upright_grid():
    centers = synth_grid_centers(6, 6, spacing=20.0, origin=(30, 30))
    beads = make_beads(centers)
    d_row, d_col, spacing = estimate_grid_axes(beads)
    assert abs(spacing - 20.0) < 2.0
    assert d_row[1] > 0.9        # d_row 指向下 ≈ (0,1)
    assert d_col[0] > 0.9        # d_col 指向右 ≈ (1,0)

def test_axes_rotated_grid():
    centers = synth_grid_centers(6, 6, spacing=20.0, origin=(200, 30), angle=np.radians(30))
    beads = make_beads(centers)
    d_row, d_col, spacing = estimate_grid_axes(beads)
    assert abs(spacing - 20.0) < 2.0
    assert 0.4 < d_row[1] < 0.7  # d_row 方向角 ~30° (sin30≈0.5)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_bead_grid.py -k axes -v`
Expected: FAIL — `cannot import name 'estimate_grid_axes'`

- [ ] **Step 3: 写实现**

追加到 `src/bead_grid.py`：

```python
def estimate_grid_axes(beads: list[Bead]) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate oriented row/col unit axes + spacing from bead centres (vector voting)."""
    from scipy.spatial import cKDTree

    pts = np.array([b.xy for b in beads], dtype=np.float64)
    n = len(pts)
    tree = cKDTree(pts)
    k = min(9, n)
    _, idx = tree.query(pts, k=k)
    vecs = np.vstack([pts[j] - pts[i] for i in range(n) for j in idx[i, 1:]])
    lengths = np.linalg.norm(vecs, axis=1)
    med = float(np.median(lengths))
    near_mask = lengths <= med * 1.8
    near = vecs[near_mask]
    near_len = lengths[near_mask]

    # dominant orientations (undirected, folded to [0, pi))
    ang = np.mod(np.arctan2(near[:, 1], near[:, 0]), np.pi)
    hist, edges = np.histogram(ang, bins=180, range=(0.0, np.pi))
    i1 = int(np.argmax(hist))
    a1 = edges[i1]
    suppressed = hist.copy()
    for w in range(-20, 21):
        suppressed[(i1 + w) % 180] = 0
    i2 = int(np.argmax(suppressed))
    a2 = edges[i2]

    v1 = np.array([np.cos(a1), np.sin(a1)])
    v2 = np.array([np.cos(a2), np.sin(a2)])

    # spacing: median length of vectors aligned with either axis
    align1 = np.abs(near @ v1)
    align2 = np.abs(near @ v2)
    aligned = near_len[(align1 > 0.9 * near_len) | (align2 > 0.9 * near_len)]
    spacing = float(np.median(aligned)) if len(aligned) else med

    # orient: row axis = more vertical (points down), col axis = more horizontal (points right)
    if abs(v1[0]) > abs(v1[1]):
        horiz, vert = v1, v2
    else:
        horiz, vert = v2, v1
    if horiz[0] < 0:
        horiz = -horiz
    if vert[1] < 0:
        vert = -vert
    return vert, horiz, spacing   # (d_row, d_col, spacing)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_bead_grid.py -k axes -v`
Expected: PASS (2 tests)

- [ ] **Step 5: 提交**

```bash
git add src/bead_grid.py tests/test_bead_grid.py
git commit -m "feat(bead_grid): estimate oriented grid axes + spacing"
```

---

## Task 5: 仿射编号 `label_affine`

**Files:**
- Modify: `src/bead_grid.py`
- Test: `tests/test_labeling.py`（新建）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_labeling.py
import numpy as np
from src.bead_grid import label_affine
from tests._synth import synth_grid_centers, make_beads, apply_homography


def test_label_affine_clean_grid():
    centers = synth_grid_centers(5, 5, spacing=20.0, origin=(40, 40))
    beads = make_beads(centers)
    d_row = np.array([0.0, 1.0]); d_col = np.array([1.0, 0.0]); spacing = 20.0
    labels, frac = label_affine(beads, d_row, d_col, spacing)
    assert frac < 0.05                                   # 干净仿射网格残差极小
    assert len(set(labels)) == 25                        # 5x5 唯一
    rs = [r for r, _ in labels]; cs = [c for _, c in labels]
    assert min(rs) == 0 and max(rs) == 4
    assert min(cs) == 0 and max(cs) == 4

def test_label_affine_perspective_has_high_residual():
    H = np.array([[1.2, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0008, 0.0, 1.0]])
    centers = apply_homography(synth_grid_centers(8, 8, spacing=25.0, origin=(60, 60)), H)
    beads = make_beads(centers)
    d_row = np.array([0.0, 1.0]); d_col = np.array([1.0, 0.0]); spacing = 25.0
    _, frac = label_affine(beads, d_row, d_col, spacing)
    assert frac > 0.3                                     # 透视下仿射残差偏大
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_labeling.py -v`
Expected: FAIL — `cannot import name 'label_affine'`

- [ ] **Step 3: 写实现**

追加到 `src/bead_grid.py`：

```python
AFFINE_OK_THRESHOLD = 0.3


def label_affine(beads, d_row, d_col, spacing):
    """Assign (row, col) by orthogonal projection + rounding. Returns (labels, residual)."""
    pts = np.array([b.xy for b in beads], dtype=np.float64)
    s = pts @ (d_row + d_col)
    origin = pts[int(np.argmin(s))]
    a = (pts - origin) @ d_row / spacing     # row coordinate
    b = (pts - origin) @ d_col / spacing     # col coordinate
    labels = list(zip(np.round(a).astype(int), np.round(b).astype(int)))
    frac = float(np.mean(np.abs(a - np.round(a)) + np.abs(b - np.round(b))))
    return labels, frac
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_labeling.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: 提交**

```bash
git add src/bead_grid.py tests/test_labeling.py
git commit -m "feat(bead_grid): affine bead labeling with residual"
```

---

## Task 6: 射影编号 `label_projective`

**Files:**
- Modify: `src/bead_grid.py`
- Test: `tests/test_labeling.py`（追加）

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_labeling.py`：

```python
from src.bead_grid import label_projective


def test_label_projective_recovers_perspective_grid():
    H = np.array([[1.2, 0.05, 0.0], [0.03, 1.0, 0.0], [0.0008, 0.0001, 1.0]])
    true_centers = synth_grid_centers(8, 8, spacing=25.0, origin=(80, 80))
    centers = apply_homography(true_centers, H)
    beads = make_beads(centers)
    from src.bead_grid import label_affine
    d_row = np.array([0.0, 1.0]); d_col = np.array([1.0, 0.0]); spacing = 25.0
    aff_labels, _ = label_affine(beads, d_row, d_col, spacing)
    labels, res_px = label_projective(beads, aff_labels)
    assert res_px < spacing * 0.3                         # 重投影残差很小
    assert len(set(labels)) == 64                         # 8x8 唯一
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_labeling.py -k projective -v`
Expected: FAIL — `cannot import name 'label_projective'`

- [ ] **Step 3: 写实现**

追加到 `src/bead_grid.py`：

```python
def label_projective(beads, affine_labels):
    """Refit labels via a single over-constrained homography (lattice -> image), no RANSAC.
    Returns (labels, reprojection_residual_px)."""
    pts = np.array([b.xy for b in beads], dtype=np.float32)
    lat = np.array(affine_labels, dtype=np.float32)            # (N, 2) = (row, col)
    H, _ = cv2.findHomography(lat, pts)                        # lattice -> image, least squares
    if H is None:
        return affine_labels, float("inf")
    Hinv = np.linalg.inv(H)
    ones = np.ones((len(pts), 1), dtype=np.float64)
    img_h = np.hstack([pts.astype(np.float64), ones])
    lat2 = (Hinv @ img_h.T).T
    lat2 = lat2[:, :2] / lat2[:, 2:3]
    labels = list(zip(np.round(lat2[:, 0]).astype(int), np.round(lat2[:, 1]).astype(int)))
    # reprojection residual in pixels
    lat_h = np.hstack([lat.astype(np.float64), ones])
    pred = (H @ lat_h.T).T
    pred = pred[:, :2] / pred[:, 2:3]
    res_px = float(np.mean(np.linalg.norm(pred - pts.astype(np.float64), axis=1)))
    return labels, res_px
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_labeling.py -k projective -v`
Expected: PASS (1 test)

- [ ] **Step 5: 提交**

```bash
git add src/bead_grid.py tests/test_labeling.py
git commit -m "feat(bead_grid): projective relabeling via single homography"
```

---

## Task 7: 编号编排 `label_beads`

**Files:**
- Modify: `src/bead_grid.py`
- Test: `tests/test_labeling.py`（追加）

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_labeling.py`：

```python
from src.bead_grid import label_beads


def test_label_beads_affine_for_upright():
    centers = synth_grid_centers(6, 6, spacing=20.0, origin=(40, 40))
    beads = make_beads(centers)
    d_row = np.array([0.0, 1.0]); d_col = np.array([1.0, 0.0]); spacing = 20.0
    labels, persp = label_beads(beads, d_row, d_col, spacing)
    assert persp is False
    assert len(set(labels)) == 36

def test_label_beads_projective_for_perspective():
    H = np.array([[1.2, 0.05, 0.0], [0.03, 1.0, 0.0], [0.0008, 0.0001, 1.0]])
    centers = apply_homography(synth_grid_centers(8, 8, spacing=25.0, origin=(80, 80)), H)
    beads = make_beads(centers)
    d_row = np.array([0.0, 1.0]); d_col = np.array([1.0, 0.0]); spacing = 25.0
    labels, persp = label_beads(beads, d_row, d_col, spacing)
    assert persp is True
    assert len(set(labels)) == 64
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_labeling.py -k label_beads -v`
Expected: FAIL — `cannot import name 'label_beads'`

- [ ] **Step 3: 写实现**

追加到 `src/bead_grid.py`：

```python
def label_beads(beads, d_row, d_col, spacing):
    """Two-tier labeling: affine by default, escalate to projective when residual is high.
    Returns (labels, perspective_tier)."""
    labels, frac = label_affine(beads, d_row, d_col, spacing)
    if frac < AFFINE_OK_THRESHOLD:
        return labels, False
    labels2, res_px = label_projective(beads, labels)
    if res_px < spacing * 0.3:
        return labels2, True
    return labels, False
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_labeling.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: 提交**

```bash
git add src/bead_grid.py tests/test_labeling.py
git commit -m "feat(bead_grid): two-tier labeling (affine + projective escalation)"
```

---

## Task 8: 尺寸/偏移/截断 `resolve_dims_and_offset`

**Files:**
- Modify: `src/bead_grid.py`
- Test: `tests/test_labeling.py`（追加）

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_labeling.py`：

```python
from src.bead_grid import resolve_dims_and_offset


def test_resolve_full_board_auto_size():
    labels = [(r, c) for r in range(5) for c in range(7)]
    rows, cols, abs_labels, trunc = resolve_dims_and_offset(labels, None, (400, 400))
    assert (rows, cols) == (5, 7)
    assert trunc.is_truncated is False
    assert min(r for r, _ in abs_labels) == 0

def test_resolve_truncated_with_board_size():
    labels = [(r, c) for r in range(5) for c in range(5)]      # 只看到 5x5，板子其实是 5x8
    rows, cols, abs_labels, trunc = resolve_dims_and_offset(labels, (5, 8), (400, 400))
    assert (rows, cols) == (5, 8)
    assert trunc.is_truncated is True
    assert "left/right" in trunc.clipped_edges

def test_resolve_offset_normalizes_topleft():
    labels = [(r + 3, c + 4) for r in range(2) for c in range(2)]
    rows, cols, abs_labels, trunc = resolve_dims_and_offset(labels, None, (400, 400))
    assert (rows, cols) == (2, 2)
    assert abs_labels[0] == (0, 0)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_labeling.py -k resolve -v`
Expected: FAIL — `cannot import name 'resolve_dims_and_offset'`

- [ ] **Step 3: 写实现**

追加到 `src/bead_grid.py`：

```python
def resolve_dims_and_offset(labels, board_size, img_shape):
    """Resolve rows/cols + absolute (0-based) offset + truncation. Assumes top-left visible (MVP)."""
    arr = np.array(labels)
    a_min, a_max = int(arr[:, 0].min()), int(arr[:, 0].max())
    b_min, b_max = int(arr[:, 1].min()), int(arr[:, 1].max())
    det_rows = a_max - a_min + 1
    det_cols = b_max - b_min + 1

    if board_size is not None:
        rows, cols = board_size
    else:
        rows, cols = det_rows, det_cols

    abs_labels = [(a - a_min, b - b_min) for (a, b) in labels]

    clipped: list[str] = []
    if board_size is not None:
        if det_rows < rows:
            clipped.append("top/bottom")
        if det_cols < cols:
            clipped.append("left/right")
    return rows, cols, abs_labels, TruncationInfo(is_truncated=bool(clipped), clipped_edges=clipped)
```

> **Note:** 顶/底、左/右的精确区分需对照图像边缘邻近度（spec §3.5）。MVP 仅标轴向；Task 15 对照回归集细化。顶/左截断（偏移歧义）的严重情况由 `evaluate_confidence` 低置信度覆盖。

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_labeling.py -k resolve -v`
Expected: PASS (3 tests)

- [ ] **Step 5: 提交**

```bash
git add src/bead_grid.py tests/test_labeling.py
git commit -m "feat(bead_grid): resolve board dims, offset, truncation"
```

---

## Task 9: 绝对映射 `fit_grid_map` + 构格 `build_cells`

**Files:**
- Modify: `src/bead_grid.py`
- Test: `tests/test_bead_grid.py`（追加）

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_bead_grid.py`：

```python
from src.bead_grid import fit_grid_map, build_cells, AffineMap, TruncationInfo


def test_fit_grid_map_affine_roundtrip():
    centers = synth_grid_centers(4, 4, spacing=20.0, origin=(40, 40))
    beads = make_beads(centers)
    abs_labels = [(r, c) for r in range(4) for c in range(4)]
    gmap = fit_grid_map(beads, abs_labels, False,
                        np.array([0.0, 1.0]), np.array([1.0, 0.0]), 20.0)
    for bead, (r, c) in zip(beads, abs_labels):
        np.testing.assert_allclose(gmap.to_xy(r, c), bead.xy, atol=2.0)

def test_build_cells_filled_and_empty():
    centers = synth_grid_centers(3, 3, spacing=20.0, origin=(40, 40))
    beads = make_beads(centers)
    abs_labels = [(r, c) for r in range(3) for c in range(3)]
    gmap = AffineMap(origin=np.array([40.0, 40.0]),
                     d_row=np.array([0.0, 1.0]), d_col=np.array([1.0, 0.0]), spacing=20.0)
    cells = build_cells(np.full((200, 200, 3), 235, dtype=np.uint8),
                        beads, abs_labels, gmap, 3, 3, TruncationInfo(False, []))
    assert len(cells) == 9
    assert all(c.image_xy is not None for c in cells)
    assert len([c for c in cells if c.is_visible]) == 9
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_bead_grid.py -k "fit_grid_map or build_cells" -v`
Expected: FAIL — `cannot import name 'fit_grid_map'`

- [ ] **Step 3: 写实现**

追加到 `src/bead_grid.py`：

```python
def fit_grid_map(beads, abs_labels, perspective_tier, d_row, d_col, spacing):
    """Fit an ABSOLUTE (row, col) -> image map consistent with abs_labels.
    Must be called AFTER offset normalization so to_xy(r, c) is valid for any cell."""
    pts = np.array([b.xy for b in beads], dtype=np.float64)
    labels_arr = np.array(abs_labels, dtype=np.float64)
    if perspective_tier:
        H, _ = cv2.findHomography(labels_arr.astype(np.float32), pts.astype(np.float32))
        return ProjectiveMap(H=H)
    # affine: robust origin = median over beads of (xy - r*s*d_row - c*s*d_col)
    origins = pts - labels_arr[:, 0:1] * spacing * d_row - labels_arr[:, 1:2] * spacing * d_col
    origin = np.median(origins, axis=0)
    return AffineMap(origin=origin, d_row=d_row, d_col=d_col, spacing=spacing)


def _is_edge_cell(r, c, rows, cols, truncation, margin=1):
    return r < margin or r >= rows - margin or c < margin or c >= cols - margin


def build_cells(image, beads, abs_labels, grid_map, rows, cols, truncation):
    """Build rows x cols CellInfo grid. Filled cells take bead colour; empty cells sample board base."""
    h, w = image.shape[:2]
    cell_bead = {}
    for bead, (r, c) in zip(beads, abs_labels):
        if 0 <= r < rows and 0 <= c < cols:
            cell_bead[(r, c)] = bead
    cells = []
    for r in range(rows):
        for c in range(cols):
            xy = grid_map.to_xy(r, c)
            inside = (0 <= xy[0] < w and 0 <= xy[1] < h)
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
                                  is_edge=is_edge, confidence=conf, image_xy=image_xy))
    return cells
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_bead_grid.py -k "fit_grid_map or build_cells" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: 提交**

```bash
git add src/bead_grid.py tests/test_bead_grid.py
git commit -m "feat(bead_grid): absolute grid map + cell grid construction"
```

---

## Task 10: 置信度 `evaluate_confidence`

**Files:**
- Modify: `src/bead_grid.py`
- Test: `tests/test_bead_grid.py`（追加）

- [ ] **Step 1: 写失败测试**

```python
from src.bead_grid import evaluate_confidence


def test_confidence_high_for_full_grid():
    centers = synth_grid_centers(5, 5, spacing=20.0, origin=(40, 40))
    beads = make_beads(centers)
    abs_labels = [(r, c) for r in range(5) for c in range(5)]
    gmap = AffineMap(origin=np.array([40.0, 40.0]),
                     d_row=np.array([0.0, 1.0]), d_col=np.array([1.0, 0.0]), spacing=20.0)
    cells = build_cells(np.full((200, 200, 3), 235, dtype=np.uint8),
                        beads, abs_labels, gmap, 5, 5, TruncationInfo(False, []))
    conf = evaluate_confidence(beads, cells, 5, 5, False, gmap)
    assert conf.bead_count == 25
    assert 0.0 < conf.grid_fill_ratio <= 1.0
    assert conf.level in ("高", "中", "低")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_bead_grid.py -k confidence -v`
Expected: FAIL — `cannot import name 'evaluate_confidence'`

- [ ] **Step 3: 写实现**

追加到 `src/bead_grid.py`：

```python
def evaluate_confidence(beads, cells, rows, cols, perspective_tier, grid_map):
    """Confidence from fill ratio. (Reprojection residual wired into this in Task 15.)"""
    fill = len(beads) / max(1, rows * cols)
    if fill > 0.5:
        level = "高"
    elif fill > 0.2:
        level = "中"
    else:
        level = "低"
    return GridConfidence(bead_count=len(beads), grid_fill_ratio=float(fill),
                          labeling_residual=0.0, perspective_tier=perspective_tier, level=level)
```

> **Note:** `grid_map` 参数预留给 Task 15：届时把 `label_beads` 透出的真实重投影残差接进来重定阈值。MVP 用填充率。

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_bead_grid.py -k confidence -v`
Expected: PASS (1 test)

- [ ] **Step 5: 提交**

```bash
git add src/bead_grid.py tests/test_bead_grid.py
git commit -m "feat(bead_grid): confidence from grid fill ratio"
```

---

## Task 11: 编排 `BeadGridFitter.fit`

**Files:**
- Modify: `src/bead_grid.py`
- Test: `tests/test_bead_grid.py`（追加）

- [ ] **Step 1: 写失败测试**

```python
from src.bead_grid import BeadGridFitter


def test_fit_end_to_end_upright():
    centers = synth_grid_centers(6, 7, spacing=25.0, origin=(50, 50))
    img = render_beads(centers, img_size=(300, 250), bead_radius=9)
    result = BeadGridFitter().fit(img)
    assert result.rows == 6 and result.cols == 7
    assert len(result.cells) == 42
    assert result.outline is not None and result.outline.shape == (4, 2)
    assert result.confidence.bead_count >= 30

def test_fit_raises_on_too_few_beads():
    import pytest
    from src.bead_grid import GridFitError
    img = np.full((200, 200, 3), 235, dtype=np.uint8)  # no beads
    with pytest.raises(GridFitError):
        BeadGridFitter().fit(img)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_bead_grid.py -k fit -v`
Expected: FAIL — `BeadGridFitter` 不存在 / `fit` 未实现

- [ ] **Step 3: 写实现**

追加到 `src/bead_grid.py`：

```python
class BeadGridFitter:
    """Detects the bead grid directly from beads (no board model)."""

    def fit(self, image: np.ndarray, board_size: tuple[int, int] | None = None) -> GridResult:
        beads = detect_beads(image)
        if len(beads) < MIN_BEADS:
            raise GridFitError(f"检出豆子太少({len(beads)})，无法分组")

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
                          confidence=confidence, truncation=trunc)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_bead_grid.py tests/test_labeling.py -v`
Expected: PASS (all)

- [ ] **Step 5: 提交**

```bash
git add src/bead_grid.py tests/test_bead_grid.py
git commit -m "feat(bead_grid): orchestrate full fit() pipeline"
```

---

## Task 12: 在原图按位置标注差异（compare.py）

**Files:**
- Modify: `src/compare.py:16-25`（`DiffResult` 增字段）、`src/compare.py:82-141`（填充 + 按位置画）
- Test: `tests/test_compare.py`（追加；若无则新建）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_compare.py（若已有文件则追加）
import numpy as np
from src.compare import DiffComparator, DiffResult
from src.grid import CellInfo


def _cell(r, c, rgb, xy):
    return CellInfo(row=r, col=c, color=np.array(rgb, dtype=np.uint8),
                    is_visible=True, is_edge=False, confidence=1.0, image_xy=xy)

def test_compare_carries_image_xy():
    cells = [_cell(0, 0, [255, 0, 0], (30.0, 40.0))]
    bp = np.zeros((1, 1, 3), dtype=np.uint8)  # blueprint black
    diffs = DiffComparator(color_tolerance=10.0).compare_with_confidence(cells, bp)
    assert len(diffs) == 1
    assert diffs[0].image_xy == (30.0, 40.0)

def test_annotate_draws_at_image_xy():
    photo = np.full((100, 100, 3), 200, dtype=np.uint8)
    diffs = [DiffResult(row=0, col=0, type="color_mismatch",
                        photo_color=[255, 0, 0], blueprint_color=[0, 0, 0],
                        cell_confidence=1.0, is_reliable=True, image_xy=(50.0, 60.0))]
    out = DiffComparator().annotate_with_confidence(photo, diffs, rows=4, cols=4)
    region = out[55:65, 45:55]
    assert np.any(np.all(region == (0, 0, 255), axis=-1))   # (50,60) 附近有红色标注
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_compare.py -k "image_xy or annotate_draws" -v`
Expected: FAIL — `DiffResult` 无 `image_xy`

- [ ] **Step 3: 写实现**

修改 `src/compare.py`。

`DiffResult` 增字段：
```python
@dataclass
class DiffResult:
    """A single mismatch with per-cell confidence."""
    row: int
    col: int
    type: str
    photo_color: list[int]
    blueprint_color: list[int]
    cell_confidence: float
    is_reliable: bool
    image_xy: tuple[float, float] | None = None   # NEW
```

`compare_with_confidence` 的 `diffs.append(...)` 增一行 `image_xy=cell.image_xy,`：
```python
                diffs.append(DiffResult(
                    row=r,
                    col=c,
                    type="color_mismatch",
                    photo_color=cell.color.tolist(),
                    blueprint_color=blueprint_grid[r, c].tolist(),
                    cell_confidence=cell.confidence,
                    is_reliable=cell.confidence >= 0.8,
                    image_xy=cell.image_xy,          # NEW
                ))
```

`annotate_with_confidence` 改为按 `image_xy` 画（有则用，无则回退均分网格）：
```python
    def annotate_with_confidence(
        self,
        photo: np.ndarray,
        diffs: list[DiffResult],
        rows: int,
        cols: int,
    ) -> np.ndarray:
        """Annotate at each diff's real image position (red=reliable, orange=unreliable)."""
        result = photo.copy()
        h, w = result.shape[:2]
        cell_h = h / rows
        cell_w = w / cols
        for diff in diffs:
            if diff.image_xy is not None:
                cx, cy = int(diff.image_xy[0]), int(diff.image_xy[1])
                half_w = max(4, int(cell_w / 2))
                half_h = max(4, int(cell_h / 2))
                x1, y1 = cx - half_w, cy - half_h
                x2, y2 = cx + half_w, cy + half_h
            else:
                r, c = diff.row, diff.col
                x1 = int(c * cell_w); y1 = int(r * cell_h)
                x2 = int((c + 1) * cell_w); y2 = int((r + 1) * cell_h)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            colour = (0, 0, 255) if diff.is_reliable else (0, 165, 255)
            cv2.rectangle(result, (x1, y1), (x2, y2), colour, 2)
            size = min(int(cell_h), int(cell_w)) // 6
            cv2.line(result, (cx - size, cy - size), (cx + size, cy + size), colour, 2)
            cv2.line(result, (cx - size, cy + size), (cx + size, cy - size), colour, 2)
        return result
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_compare.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/compare.py tests/test_compare.py
git commit -m "feat(compare): annotate diffs at real image positions"
```

---

## Task 13: CLI 集成

**Files:**
- Modify: `src/cli.py`
- Test: 手动冒烟（步骤内）

- [ ] **Step 1: 修改 CLI 走新管线**

修改 `src/cli.py`：

1. 替换 imports——去掉 `from src.detect import BoardDetector` 与 `from src.edge_refiner import RefinerConfig, DetectionError`，新增：
```python
from src.bead_grid import BeadGridFitter, GridFitError
```
（保留 `from src.grid import PerspectiveCorrector, GridExtractor` 不再必需，可删。）

2. `main()` 参数：`--board-size` 已是 optional，保持；删 `--model`（板子）参数；删 `--legacy`；删 `--bead-model`（新流程不用豆子模型）。

3. 删除 `_run_refined` 与 `_run_legacy`，新增 `_run_bead_grid`：
```python
def _run_bead_grid(args, photo, board_size):
    fitter = BeadGridFitter()
    try:
        result = fitter.fit(photo, board_size=board_size)
    except GridFitError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)

    conf = result.confidence
    print(f"   ✅ 拟合豆子网格: {result.rows}×{result.cols}")
    print(f"   豆子数={conf.bead_count}  填充率={conf.grid_fill_ratio:.2f}  "
          f"透视分级={conf.perspective_tier}  置信度={conf.level}")
    if result.truncation.is_truncated:
        print(f"   截断: {', '.join(result.truncation.clipped_edges)}")

    annotated = photo.copy()

    if args.blueprint:
        print("📊 对比图纸...")
        from src.blueprint import parse_blueprint
        blueprint_img = cv2.imread(args.blueprint)
        bp_grid = parse_blueprint(blueprint_img, result.rows, result.cols)
        comparator = DiffComparator(color_tolerance=args.color_tolerance)
        diffs = comparator.compare_with_confidence(result.cells, bp_grid)
        reliable = [d for d in diffs if d.is_reliable]
        unreliable = [d for d in diffs if not d.is_reliable]
        print(f"   可靠差异 {len(reliable)} 处；边缘/低置信 {len(unreliable)} 处")
        annotated = comparator.annotate_with_confidence(photo, diffs, result.rows, result.cols)
    else:
        cv2.polylines(annotated, [result.outline.astype(np.int32)], True, (0, 255, 0), 3)

    print(f"\n💾 保存到 {args.output}...")
    cv2.imwrite(args.output, annotated)
    print("   ✅ 已保存")
```

4. `main()` 中板子尺寸解析 + 调用：
```python
    board_size = None
    if args.board_size:
        rows, cols = parse_board_size(args.board_size)
        board_size = (rows, cols)
        print(f"📐 拼板: {rows}×{cols}（提供，用于截断补全）")
    _run_bead_grid(args, photo, board_size)
```

5. 删掉对 `args.model` / `args.bead_model` 存在性的检查，以及 `BoardDetector` 相关打印。

- [ ] **Step 2: 跑现有测试确认没破坏**

Run: `python -m pytest tests/ -v -k "not regression and not label_ui"`
Expected: PASS（除需要服务器的 UI 集成测试外）

- [ ] **Step 3: 冒烟测试（俯拍照）**

Run:
```bash
python -m src.cli --photo "training/photos/IMG_6097.PNG" --output /tmp/out_grid.jpg
```
Expected: 打印 `✅ 拟合豆子网格`、豆子数、置信度；生成 `/tmp/out_grid.jpg`。打开确认绿框贴合板子外轮廓。

- [ ] **Step 4: 冒烟测试（强透视照）**

Run:
```bash
python -m src.cli --photo "training/photos/IMG_6121.JPG" --output /tmp/out_persp.jpg
```
Expected: 成功拟合，`透视分级=True`，绿框呈梯形贴合板子。

- [ ] **Step 5: 提交**

```bash
git add src/cli.py
git commit -m "feat(cli): wire bead-grid pipeline, make --board-size optional"
```

---

## Task 14: 删除旧板子检测 + 更新文档

**Files:**
- Delete: `src/edge_refiner.py`, `src/detect.py`
- Possibly delete: `tests/test_detect.py`（若仅测 `BoardDetector`）
- Modify: `CLAUDE.md`、`docs/obs/technical/architecture.md`
- Test: 全量回归

- [ ] **Step 1: 删旧文件**

```bash
git rm src/edge_refiner.py src/detect.py
```

- [ ] **Step 2: 清理残留引用**

```bash
grep -rn "edge_refiner\|BoardDetector\|from src.detect\|from .detect\|RefinerConfig\|DetectionError" src/ tests/
```
修复命中处（删除该 import / 调用）。若 `tests/test_detect.py` 存在且仅测 `BoardDetector`，删除：`git rm tests/test_detect.py`。注意 `src/bead_grid.py` 用的是自己的 `GridFitError`，不受影响。

- [ ] **Step 3: 更新 CLAUDE.md**

把 `src/` 模块表里的 `detect.py`/`edge_refiner.py` 两行替换为：
```
├── bead_grid.py        # BeadGridFitter — 豆子检测 + 网格编号（取代板子检测）
```
并把 "Two-stage detection" 设计模式说明改为：板子定位已由豆子网格直接推导（不再有独立板子模型）；Common Commands 里涉及 `--model models/beadboard-best.pt` 的 CLI 示例改为不传 `--model`。

- [ ] **Step 4: 更新 `docs/obs/technical/architecture.md`**

把 BoardDetector / EdgeRefiner 相关段落改为 BeadGridFitter 描述（检测豆子→共线分组编号→逐格比色），数据流 mermaid 图相应更新（`BoardDetector` 节点改为 `BeadGridFitter`，去掉 mask 中间产物）。

- [ ] **Step 5: 跑全量测试**

Run: `python -m pytest tests/ -v -k "not label_ui_integration"`
Expected: PASS（无对已删模块的引用残留）

- [ ] **Step 6: 提交**

```bash
git add -A
git commit -m "refactor: remove BoardDetector + edge_refiner, update docs"
```

---

## Task 15: 真实照片回归 + 调参

**Files:**
- Create: `tests/fixtures/board_regression/`（子集）
- Create: `tests/test_grid_regression.py`
- Modify: `src/bead_grid.py`（调参 + 接入真实残差）

- [ ] **Step 1: 选回归子集入库**

从 `training/photos/` 挑代表 3 张复制到 `tests/fixtures/board_regression/`：俯拍 `IMG_6097.PNG`→`upright.png`、强透视 `IMG_6121.JPG`→`perspective.jpg`、背景杂 `IJYJ0881.jpg`→`clutter.jpg`。

```bash
mkdir -p tests/fixtures/board_regression
cp training/photos/IMG_6097.PNG tests/fixtures/board_regression/upright.png
cp training/photos/IMG_6121.JPG tests/fixtures/board_regression/perspective.jpg
cp training/photos/IJYJ0881.jpg tests/fixtures/board_regression/clutter.jpg
```

- [ ] **Step 2: 写回归测试**

```python
# tests/test_grid_regression.py
from pathlib import Path
import cv2
import pytest
from src.bead_grid import BeadGridFitter, GridFitError

FIX = Path(__file__).parent / "fixtures" / "board_regression"
PHOTOS = ["upright.png", "perspective.jpg", "clutter.jpg"]


@pytest.mark.parametrize("name", PHOTOS)
def test_fit_does_not_crash_on_real_photo(name):
    img = cv2.imread(str(FIX / name))
    if img is None:
        pytest.skip(f"无法读取 {name}")
    try:
        result = BeadGridFitter().fit(img)
    except GridFitError:
        pytest.fail(f"真实照片 {name} 拟合失败——这正是要修的回归")
    assert result.rows >= 5 and result.cols >= 5
    assert result.confidence.bead_count >= 50


def test_perspective_photo_uses_projective_tier():
    img = cv2.imread(str(FIX / "perspective.jpg"))
    if img is None:
        pytest.skip("无法读取 perspective.jpg")
    result = BeadGridFitter().fit(img)
    assert result.confidence.perspective_tier is True
```

- [ ] **Step 3: 跑回归，记录失败项**

Run: `python -m pytest tests/test_grid_regression.py -v`
Expected: 多半有失败（参数未对真实数据校准）。逐个记录失败现象（豆子数过少？分级没触发？rows/cols 异常？）。

- [ ] **Step 4: 对照真实照片调参**

针对失败现象调整 `src/bead_grid.py` 初始值，每改一个参数重跑：
- 豆子漏检多 → 放宽 `detect_beads` 的 `param2`（更敏感）、调整 `min_r/max_r` 比例。
- 强透视未触发分级 → 调低 `AFFINE_OK_THRESHOLD` 或 `label_beads` 里 `spacing * 0.3` 判据。
- 置信度不准 → 在 `evaluate_confidence` 接入 `label_beads` 透传的真实 `res_px`（改 `label_beads` 返回 `(labels, persp, res_px)`，`fit` 把 `res_px` 传给 `evaluate_confidence` 写入 `labeling_residual`，按残差+填充率重定 level 阈值）。

```bash
python -m pytest tests/test_bead_grid.py tests/test_labeling.py tests/test_grid_regression.py -v
```

> 肉眼核对：`python -m src.cli --photo <照片> --output /tmp/check.jpg`，看绿框是否贴合板子外轮廓、差异标注是否落在错位豆子上。

- [ ] **Step 5: 全量回归通过后提交**

Run: `python -m pytest tests/ -v -k "not label_ui_integration"`
Expected: PASS

```bash
git add tests/fixtures/board_regression tests/test_grid_regression.py src/bead_grid.py
git commit -m "test(bead_grid): real-photo regression set + tuned detection params"
```

---

## 完成标准

- `src/edge_refiner.py`、`src/detect.py` 已删除；全仓库无残留引用。
- `python -m pytest tests/ -v` 全绿（UI 集成测试除外）。
- 13 张回归照片（`training/photos/` 全量）经 CLI 跑通，绿框贴合板子外轮廓，倾斜照走透视分级。
- CLI `--board-size` 可选；无图纸时画板子外轮廓，有图纸时在原图按位置标注差异。

## 已知限制（MVP 不处理，后续迭代）

- 多块板/背景也含规则圆点阵列：靠主方向胜出，但两组强度接近时不显式报警（spec §3.5 该项为降级而非硬失败）。
- 顶/左截断（偏移歧义）与板子严重露出不足：以低置信度提示，不硬解偏移。
- 板子被旋转 90°/180° 拍摄：当前假定正立拍摄；如需支持，需对照图纸尝试 4/8 种朝向取最佳色匹配（二期）。
- peg 孔通道：二期（spec §3.6）。
