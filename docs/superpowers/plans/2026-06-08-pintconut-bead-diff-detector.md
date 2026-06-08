# Pintconut 拼豆差异检测器 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个命令行工具，通过 YOLOv8-seg 定位拼板、提取网格、比较颜色差异，输出标注了拼错位置的图片。

**Architecture:** 三阶段流水线——模型推理定位拼板区域，OpenCV 做透视校正和网格提取，NumPy 在 LAB 色彩空间做颜色匹配。训练流程使用 FastSAM 半自动标注生成数据集，再 fine-tune YOLOv8n-seg。

**Tech Stack:** Python 3.10+, ultralytics (YOLOv8-seg + FastSAM), OpenCV, NumPy, Pillow

---

## 文件结构总览

```
pintconut/
├── src/
│   ├── __init__.py              # 包初始化
│   ├── detect.py                # 阶段①：拼板定位
│   ├── grid.py                  # 阶段②：透视校正 + 网格提取
│   ├── color.py                 # 221色盘颜色匹配 + 有豆判断
│   ├── blueprint.py             # 图纸解析
│   ├── compare.py               # 阶段③：对比 + 差异标注
│   └── cli.py                   # 命令行入口
├── training/
│   ├── collect_helper.py        # 照片批量重命名
│   ├── semi_auto_label.py       # 半自动标注工具
│   ├── train.py                 # 训练脚本
│   ├── validate.py              # 验证脚本
│   └── auto_label_test.py       # [已存在] 自动标注测试
├── tests/
│   ├── __init__.py
│   ├── test_color.py            # 颜色匹配测试
│   ├── test_grid.py             # 网格提取测试
│   ├── test_blueprint.py        # 图纸解析测试
│   ├── test_detect.py           # 拼板定位测试
│   └── test_compare.py          # 对比标注测试
├── data/
│   ├── colors.json              # 221色盘定义
│   └── board_sizes.json         # 拼板尺寸规格
├── models/                      # 训练好的模型权重
└── requirements.txt
```

---

### Task 1: 项目初始化 + requirements.txt

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: 创建 requirements.txt**

```txt
ultralytics>=8.4.0
opencv-python>=4.6.0
numpy>=1.23.0
Pillow>=9.0.0
```

- [ ] **Step 2: 创建包初始化文件**

`src/__init__.py`:
```python
```

`tests/__init__.py`:
```python
```

- [ ] **Step 3: 安装依赖到 venv**

Run: `source .venv/bin/activate && pip install -r requirements.txt`
Expected: 所有依赖安装成功

- [ ] **Step 4: Commit**

```bash
git add requirements.txt src/__init__.py tests/__init__.py
git commit -m "chore: init project with requirements and package structure"
```

---

### Task 2: 221 色盘数据 + 颜色匹配模块

**Files:**
- Create: `data/colors.json`
- Create: `data/board_sizes.json`
- Create: `src/color.py`
- Create: `tests/test_color.py`

这是整个系统的数据基础，其他模块依赖它来做颜色判断。

- [ ] **Step 1: 创建 221 色盘数据文件**

`data/colors.json` — 包含拼豆常见 221 色的 RGB 值。格式：
```json
[
  {"id": 1, "name": "White", "rgb": [255, 255, 255]},
  {"id": 2, "name": "Ivory", "rgb": [255, 255, 240]},
  {"id": 3, "name": "Yellow", "rgb": [255, 255, 0]}
]
```

注意：完整的 221 色数据需要从拼豆厂商色卡获取。初始版本先放入 20 种常见颜色用于开发测试，后续再补全。使用以下 20 色：

```json
[
  {"id": 1, "name": "White", "rgb": [255, 255, 255]},
  {"id": 2, "name": "Ivory", "rgb": [255, 255, 240]},
  {"id": 3, "name": "Yellow", "rgb": [255, 255, 0]},
  {"id": 4, "name": "Orange", "rgb": [255, 165, 0]},
  {"id": 5, "name": "Red", "rgb": [255, 0, 0]},
  {"id": 6, "name": "Pink", "rgb": [255, 192, 203]},
  {"id": 7, "name": "Hot Pink", "rgb": [255, 105, 180]},
  {"id": 8, "name": "Purple", "rgb": [128, 0, 128]},
  {"id": 9, "name": "Blue", "rgb": [0, 0, 255]},
  {"id": 10, "name": "Light Blue", "rgb": [173, 216, 230]},
  {"id": 11, "name": "Cyan", "rgb": [0, 255, 255]},
  {"id": 12, "name": "Green", "rgb": [0, 128, 0]},
  {"id": 13, "name": "Light Green", "rgb": [144, 238, 144]},
  {"id": 14, "name": "Brown", "rgb": [139, 69, 19]},
  {"id": 15, "name": "Tan", "rgb": [210, 180, 140]},
  {"id": 16, "name": "Peach", "rgb": [255, 218, 185]},
  {"id": 17, "name": "Black", "rgb": [0, 0, 0]},
  {"id": 18, "name": "Dark Gray", "rgb": [64, 64, 64]},
  {"id": 19, "name": "Gray", "rgb": [128, 128, 128]},
  {"id": 20, "name": "Light Gray", "rgb": [192, 192, 192]}
]
```

- [ ] **Step 2: 创建拼板尺寸数据文件**

`data/board_sizes.json`:
```json
[
  {"name": "Square Mini", "rows": 14, "cols": 14},
  {"name": "Square Small", "rows": 21, "cols": 21},
  {"name": "Square Medium", "rows": 29, "cols": 29},
  {"name": "Square Large", "rows": 40, "cols": 40},
  {"name": "Rectangle Small", "rows": 21, "cols": 29},
  {"name": "Rectangle Large", "rows": 29, "cols": 58}
]
```

- [ ] **Step 3: 写颜色匹配模块的测试**

`tests/test_color.py`:
```python
"""Tests for color matching module."""
import json
import os
import numpy as np

from src.color import ColorMatcher, load_color_palette


def test_load_color_palette():
    """Test loading color palette from JSON file."""
    palette = load_color_palette("data/colors.json")
    assert len(palette) == 20
    assert palette[0]["name"] == "White"
    assert palette[0]["rgb"] == [255, 255, 255]


def test_match_exact_color():
    """Test matching an exact color from the palette."""
    matcher = ColorMatcher("data/colors.json")
    # Pure red should match Red
    result = matcher.match([255, 0, 0])
    assert result["name"] == "Red"


def test_match_close_color():
    """Test matching a color close to a palette entry."""
    matcher = ColorMatcher("data/colors.json")
    # Slightly off-red should still match Red
    result = matcher.match([250, 10, 5])
    assert result["name"] == "Red"


def test_match_in_lab_space():
    """Test that matching works in LAB color space (perceptually uniform)."""
    matcher = ColorMatcher("data/colors.json")
    # Two shades that are close in LAB but far in RGB
    result = matcher.match([240, 240, 240])
    assert result["name"] == "White"


def test_is_bead_present():
    """Test detecting whether a bead is present at a grid cell."""
    matcher = ColorMatcher("data/colors.json")
    # A bright colored pixel should be a bead
    assert matcher.is_bead([255, 0, 0]) is True
    # White (close to board color) should not be a bead
    assert matcher.is_bead([250, 250, 250]) is False
    # Very light gray (close to board) should not be a bead
    assert matcher.is_bead([245, 245, 245]) is False


def test_is_bead_dark_color():
    """Test that dark colored beads are detected (not confused with empty)."""
    matcher = ColorMatcher("data/colors.json")
    # Black is a valid bead color, not an empty cell
    assert matcher.is_bead([0, 0, 0]) is True
    # Dark gray bead
    assert matcher.is_bead([64, 64, 64]) is True
```

- [ ] **Step 4: 运行测试确认失败**

Run: `source .venv/bin/activate && python -m pytest tests/test_color.py -v`
Expected: FAIL（模块不存在）

- [ ] **Step 5: 实现颜色匹配模块**

`src/color.py`:
```python
"""
Color matching module for bead detection.

Loads the 221-color palette and provides:
- Color matching (find nearest palette color in LAB space)
- Bead presence detection (is a pixel a bead or empty board?)
"""
import json
import os

import cv2
import numpy as np


def load_color_palette(path: str = "data/colors.json") -> list[dict]:
    """Load color palette from JSON file.

    Args:
        path: Path to colors.json file.

    Returns:
        List of color dicts with 'id', 'name', 'rgb' keys.
    """
    with open(path, "r") as f:
        return json.load(f)


class ColorMatcher:
    """Matches pixel colors to the bead palette in LAB color space."""

    # Board base color in LAB (white / light gray typical pegboard color)
    BOARD_LAB = np.array([255.0, 128.0, 128.0])  # Will be converted from RGB

    def __init__(self, palette_path: str = "data/colors.json"):
        palette = load_color_palette(palette_path)
        self.palette = palette
        # Pre-compute LAB values for all palette colors
        rgb_array = np.array([c["rgb"] for c in palette], dtype=np.uint8).reshape(-1, 1, 3)
        lab_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2Lab)
        self.palette_lab = lab_array.reshape(-1, 3).astype(np.float64)

        # Board color in LAB (white-ish pegboard)
        board_rgb = np.array([[[250, 250, 250]]], dtype=np.uint8)
        board_lab = cv2.cvtColor(board_rgb, cv2.COLOR_RGB2Lab)
        self.board_lab = board_lab.flatten().astype(np.float64)

    def match(self, rgb: list[int]) -> dict:
        """Find the nearest palette color for an RGB value.

        Uses LAB color space for perceptually accurate matching.

        Args:
            rgb: [R, G, B] color values (0-255).

        Returns:
            The matching palette entry dict with 'id', 'name', 'rgb'.
        """
        rgb_arr = np.array(rgb, dtype=np.uint8).reshape(1, 1, 3)
        lab = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2Lab).flatten().astype(np.float64)

        # Euclidean distance in LAB space
        distances = np.sqrt(np.sum((self.palette_lab - lab) ** 2, axis=1))
        nearest_idx = np.argmin(distances)
        return self.palette[nearest_idx]

    def is_bead(self, rgb: list[int], threshold: float = 30.0) -> bool:
        """Check if a color represents a bead (vs empty board cell).

        A pixel is considered a bead if its LAB distance from the board color
        exceeds the threshold. This distinguishes colored beads from the
        white/transparent pegboard base.

        Args:
            rgb: [R, G, B] color values (0-255).
            threshold: LAB distance threshold. Pixels closer to board color
                       than this are considered empty.

        Returns:
            True if the color is likely a bead, False if empty board.
        """
        rgb_arr = np.array(rgb, dtype=np.uint8).reshape(1, 1, 3)
        lab = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2Lab).flatten().astype(np.float64)

        # For very dark colors (black beads), check brightness first
        brightness = rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114
        if brightness < 100:
            # Dark color: check if it's far from board white
            # LAB L channel: board is high L, dark beads are low L
            return True  # Dark colors on a white board are always beads

        distance = np.sqrt(np.sum((lab - self.board_lab) ** 2))
        return distance > threshold

    def match_grid(self, rgb_grid: np.ndarray) -> np.ndarray:
        """Match an entire grid of RGB colors to palette entries.

        Args:
            rgb_grid: 3D array of shape (rows, cols, 3) with RGB values.

        Returns:
            2D array of shape (rows, cols) with palette entry dicts.
        """
        rows, cols = rgb_grid.shape[:2]
        result = np.empty((rows, cols), dtype=object)
        for r in range(rows):
            for c in range(cols):
                result[r, c] = self.match(rgb_grid[r, c].tolist())
        return result
```

- [ ] **Step 6: 运行测试确认通过**

Run: `source .venv/bin/activate && python -m pytest tests/test_color.py -v`
Expected: 全部 PASS

- [ ] **Step 7: Commit**

```bash
git add data/colors.json data/board_sizes.json src/color.py tests/test_color.py
git commit -m "feat: add 221-color palette and LAB color matching module"
```

---

### Task 3: 图纸解析模块

**Files:**
- Create: `src/blueprint.py`
- Create: `tests/test_blueprint.py`

图纸是带颜色的网格图片，需要提取出每格的目标颜色矩阵。

- [ ] **Step 1: 创建测试用图纸图片**

创建一个简单的 5×5 测试图纸用于测试。用 Python 脚本生成：

```python
# scripts/generate_test_blueprint.py (一次性运行)
import numpy as np
from PIL import Image

# 5x5 grid, each cell is 20x20 pixels, with grid lines
cell_size = 20
grid_size = 5
margin = 10
total = margin * 2 + cell_size * grid_size

img = np.ones((total, total, 3), dtype=np.uint8) * 255

colors = [
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 0],
    [0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 0, 0], [0, 255, 0],
    [0, 0, 255], [255, 255, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
    [255, 0, 0], [0, 0, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0],
    [0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 0, 0], [0, 255, 0],
]

for r in range(grid_size):
    for c in range(grid_size):
        y1 = margin + r * cell_size
        x1 = margin + c * cell_size
        img[y1:y1+cell_size, x1:x1+cell_size] = colors[r * grid_size + c]
        # Draw grid lines
        img[y1, x1:x1+cell_size] = [200, 200, 200]
        img[y1:y1+cell_size, x1] = [200, 200, 200]

Image.fromarray(img).save("tests/fixtures/test_blueprint_5x5.png")
```

Run: `mkdir -p tests/fixtures && source .venv/bin/activate && python -c "上面的脚本内容"`

- [ ] **Step 2: 写图纸解析的测试**

`tests/test_blueprint.py`:
```python
"""Tests for blueprint parsing module."""
import numpy as np

from src.blueprint import parse_blueprint


def test_parse_blueprint_returns_grid():
    """Test that parse_blueprint returns a 2D color grid."""
    result = parse_blueprint("tests/fixtures/test_blueprint_5x5.png", rows=5, cols=5)
    assert result.shape == (5, 5, 3)
    assert result.dtype == np.uint8


def test_parse_blueprint_detects_colors():
    """Test that parse_blueprint correctly extracts cell colors."""
    result = parse_blueprint("tests/fixtures/test_blueprint_5x5.png", rows=5, cols=5)
    # Top-left cell is red
    top_left = result[0, 0]
    assert abs(int(top_left[0]) - 255) < 30  # R channel near 255
    assert abs(int(top_left[1]) - 0) < 30    # G channel near 0
    assert abs(int(top_left[2]) - 0) < 30    # B channel near 0


def test_parse_blueprint_handles_margin():
    """Test that parse_blueprint ignores non-grid margins."""
    result = parse_blueprint("tests/fixtures/test_blueprint_5x5.png", rows=5, cols=5)
    # Should still return 5x5 grid despite 10px margin
    assert result.shape == (5, 5, 3)
```

- [ ] **Step 3: 运行测试确认失败**

Run: `source .venv/bin/activate && python -m pytest tests/test_blueprint.py -v`
Expected: FAIL

- [ ] **Step 4: 实现图纸解析模块**

`src/blueprint.py`:
```python
"""
Blueprint parsing module.

Extracts a color grid from a blueprint image (PNG/JPG).
Blueprints are grid images where each cell represents one bead position.
"""
import cv2
import numpy as np


def parse_blueprint(
    image_path: str,
    rows: int,
    cols: int,
    margin_fraction: float = 0.02,
) -> np.ndarray:
    """Parse a blueprint image into a rows×cols×3 RGB color grid.

    The function:
    1. Loads the image
    2. Detects the grid region (trims margins)
    3. Divides into rows×cols cells
    4. Samples the median color from the center of each cell

    Args:
        image_path: Path to the blueprint image file.
        rows: Number of bead rows in the grid.
        cols: Number of bead columns in the grid.
        margin_fraction: Fraction of image dimension to trim as margin (0-0.5).

    Returns:
        numpy array of shape (rows, cols, 3) with RGB uint8 values.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Convert BGR (OpenCV default) to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # Trim margins
    margin_y = int(h * margin_fraction)
    margin_x = int(w * margin_fraction)
    img_cropped = img_rgb[margin_y:h - margin_y, margin_x:w - margin_x]

    # If the image has a visible border with text/markings, try to detect
    # the actual grid region by looking for the grid structure.
    # For now, we assume the grid fills the cropped region.
    grid_h, grid_w = img_cropped.shape[:2]
    cell_h = grid_h / rows
    cell_w = grid_w / cols

    result = np.zeros((rows, cols, 3), dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            # Sample center region of each cell (inner 60% to avoid edges)
            y_start = int(r * cell_h + cell_h * 0.2)
            y_end = int(r * cell_h + cell_h * 0.8)
            x_start = int(c * cell_w + cell_w * 0.2)
            x_end = int(c * cell_w + cell_w * 0.8)

            cell = img_cropped[y_start:y_end, x_start:x_end]
            # Use median for robustness against grid lines
            median_color = np.median(cell.reshape(-1, 3), axis=0)
            result[r, c] = median_color.astype(np.uint8)

    return result
```

- [ ] **Step 5: 运行测试确认通过**

Run: `source .venv/bin/activate && python -m pytest tests/test_blueprint.py -v`
Expected: 全部 PASS

- [ ] **Step 6: Commit**

```bash
git add src/blueprint.py tests/test_blueprint.py tests/fixtures/
git commit -m "feat: add blueprint parsing module"
```

---

### Task 4: 网格提取模块（透视校正 + 网格采样）

**Files:**
- Create: `src/grid.py`
- Create: `tests/test_grid.py`

这是阶段②的核心：将照片中的拼板区域校正为标准矩形，然后按尺寸等分为网格。

- [ ] **Step 1: 写网格提取的测试**

`tests/test_grid.py`:
```python
"""Tests for grid extraction module."""
import numpy as np

from src.grid import (
    PerspectiveCorrector,
    GridExtractor,
)


def _make_test_image_with_board():
    """Create a synthetic test image with a known 'board' region."""
    # 300x300 white image with a 200x200 colored region in center
    img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    img[50:250, 50:250] = [200, 200, 200]  # gray board
    return img


def test_perspective_correction_returns_rectangle():
    """Test that perspective correction produces a rectangular output."""
    img = _make_test_image_with_board()
    corners = np.array([
        [50, 50],   # top-left
        [250, 50],  # top-right
        [250, 250], # bottom-right
        [50, 250],  # bottom-left
    ], dtype=np.float32)

    corrector = PerspectiveCorrector()
    result = corrector.correct(img, corners, output_size=(200, 200))
    assert result.shape == (200, 200, 3)


def test_perspective_correction_handles_skew():
    """Test that perspective correction handles skewed corners."""
    img = _make_test_image_with_board()
    # Skewed trapezoid corners
    corners = np.array([
        [70, 40],   # top-left (shifted right)
        [240, 60],  # top-right (shifted left)
        [260, 260], # bottom-right
        [40, 240],  # bottom-left
    ], dtype=np.float32)

    corrector = PerspectiveCorrector()
    result = corrector.correct(img, corners, output_size=(200, 200))
    assert result.shape == (200, 200, 3)
    # Output should be a proper rectangle
    assert result.dtype == np.uint8


def test_grid_extractor_returns_color_matrix():
    """Test that GridExtractor returns a rows×cols×3 color matrix."""
    # Create a corrected board image (200x200)
    board_img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Fill with alternating colors to simulate beads
    for r in range(5):
        for c in range(5):
            y1 = r * 40
            x1 = c * 40
            color = [255, 0, 0] if (r + c) % 2 == 0 else [0, 255, 0]
            board_img[y1:y1+40, x1:x1+40] = color

    extractor = GridExtractor()
    result = extractor.extract(board_img, rows=5, cols=5)
    assert result.shape == (5, 5, 3)
    assert result.dtype == np.uint8


def test_grid_extractor_samples_center():
    """Test that GridExtractor samples from cell center, not edges."""
    # Create a cell where edges are black but center is red
    board_img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Black border around each cell, red center
    board_img[10:90, 10:90] = [255, 0, 0]

    extractor = GridExtractor()
    result = extractor.extract(board_img, rows=1, cols=1)
    # Should sample red center, not black edge
    assert result[0, 0, 0] > 200  # R channel should be high
```

- [ ] **Step 2: 运行测试确认失败**

Run: `source .venv/bin/activate && python -m pytest tests/test_grid.py -v`
Expected: FAIL

- [ ] **Step 3: 实现网格提取模块**

`src/grid.py`:
```python
"""
Grid extraction module.

Handles perspective correction of the detected board region
and extraction of per-cell color values.
"""
import cv2
import numpy as np


class PerspectiveCorrector:
    """Corrects perspective distortion of the bead board in photos."""

    def correct(
        self,
        image: np.ndarray,
        corners: np.ndarray,
        output_size: tuple[int, int] = (400, 400),
    ) -> np.ndarray:
        """Apply perspective transform to correct board skew.

        Args:
            image: Input image (H×W×3 BGR or RGB array).
            corners: 4×2 array of corner points in order:
                     [top-left, top-right, bottom-right, bottom-left].
            output_size: (width, height) of the output corrected image.

        Returns:
            Perspective-corrected image of size output_size.
        """
        # Destination points: a perfect rectangle
        dst = np.array([
            [0, 0],
            [output_size[0], 0],
            [output_size[0], output_size[1]],
            [0, output_size[1]],
        ], dtype=np.float32)

        # Ensure corners are float32
        src = corners.astype(np.float32)

        # Compute perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src, dst)

        # Apply the transform
        corrected = cv2.warpPerspective(
            image, matrix, output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return corrected


class GridExtractor:
    """Extracts a color grid from a perspective-corrected board image."""

    def extract(
        self,
        board_image: np.ndarray,
        rows: int,
        cols: int,
        sample_fraction: float = 0.4,
    ) -> np.ndarray:
        """Extract per-cell colors from a corrected board image.

        Args:
            board_image: Perspective-corrected board image (square or rectangular).
            rows: Number of bead rows.
            cols: Number of bead columns.
            sample_fraction: Fraction of cell to sample from center (0-1).
                             0.4 means use the inner 40% of each cell.

        Returns:
            numpy array of shape (rows, cols, 3) with RGB uint8 values.
        """
        h, w = board_image.shape[:2]
        cell_h = h / rows
        cell_w = w / cols

        result = np.zeros((rows, cols, 3), dtype=np.uint8)

        for r in range(rows):
            for c in range(cols):
                # Center region of the cell
                margin_y = cell_h * (1 - sample_fraction) / 2
                margin_x = cell_w * (1 - sample_fraction) / 2

                y_start = int(r * cell_h + margin_y)
                y_end = int((r + 1) * cell_h - margin_y)
                x_start = int(c * cell_w + margin_x)
                x_end = int((c + 1) * cell_w - margin_x)

                # Clamp to image bounds
                y_start = max(0, y_start)
                y_end = min(h, y_end)
                x_start = max(0, x_start)
                x_end = min(w, x_end)

                if y_end <= y_start or x_end <= x_start:
                    result[r, c] = [0, 0, 0]
                    continue

                cell = board_image[y_start:y_end, x_start:x_end]
                # Use median for robustness
                median_color = np.median(cell.reshape(-1, 3), axis=0)
                result[r, c] = median_color.astype(np.uint8)

        return result
```

- [ ] **Step 4: 运行测试确认通过**

Run: `source .venv/bin/activate && python -m pytest tests/test_grid.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/grid.py tests/test_grid.py
git commit -m "feat: add perspective correction and grid extraction module"
```

---

### Task 5: 拼板定位模块（YOLOv8-seg 推理）

**Files:**
- Create: `src/detect.py`
- Create: `tests/test_detect.py`

这是阶段①：用训练好的模型从照片中定位拼板区域并提取四角点。

- [ ] **Step 1: 写拼板定位的测试**

`tests/test_detect.py`:
```python
"""Tests for bead board detection module."""
import numpy as np

from src.detect import BoardDetector


def test_board_detector_init():
    """Test that BoardDetector can be initialized (model path may not exist yet)."""
    # This test verifies the class structure, not actual inference
    # Real inference tests need a trained model
    detector = BoardDetector.__new__(BoardDetector)
    assert hasattr(detector, "model_path")
    assert hasattr(detector, "model")


def test_extract_corners_from_mask():
    """Test extracting 4 corner points from a rectangular mask."""
    detector = BoardDetector.__new__(BoardDetector)

    # Create a simple rectangular mask (100x100, white rectangle from 20,20 to 80,80)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255

    corners = detector.extract_corners(mask)
    assert corners is not None
    assert corners.shape == (4, 2)

    # Corners should be approximately at the rectangle corners
    # Top-left should be near (20, 20)
    tl = corners[0]
    assert abs(tl[0] - 20) < 5 or abs(tl[1] - 20) < 5


def test_extract_corners_returns_none_for_empty_mask():
    """Test that extract_corners returns None for empty mask."""
    detector = BoardDetector.__new__(BoardDetector)
    mask = np.zeros((100, 100), dtype=np.uint8)
    result = detector.extract_corners(mask)
    assert result is None
```

- [ ] **Step 2: 运行测试确认失败**

Run: `source .venv/bin/activate && python -m pytest tests/test_detect.py -v`
Expected: FAIL

- [ ] **Step 3: 实现拼板定位模块**

`src/detect.py`:
```python
"""
Bead board detection module.

Uses a fine-tuned YOLOv8-seg model to locate the bead board
in a photo and extract its four corner points.
"""
import numpy as np
import cv2


class BoardDetector:
    """Detects the bead board in a photo using YOLOv8-seg."""

    def __init__(self, model_path: str = "models/beadboard-best.pt"):
        """Initialize the detector with a trained model.

        Args:
            model_path: Path to the trained YOLOv8-seg weights file.
        """
        self.model_path = model_path
        self.model = None

    def _load_model(self):
        """Lazy-load the YOLO model (avoids import overhead until needed)."""
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)

    def detect(self, image: np.ndarray) -> np.ndarray | None:
        """Detect the bead board in an image and return its mask.

        Args:
            image: Input image (H×W×3 BGR array).

        Returns:
            Binary mask of the board region (H×W uint8), or None if not found.
        """
        self._load_model()
        results = self.model(image, verbose=False)

        if not results or results[0].masks is None:
            return None

        # Take the largest mask (most likely the board)
        masks = results[0].masks.data.cpu().numpy()
        h, w = image.shape[:2]

        best_mask = None
        best_area = 0
        for mask in masks:
            mask_resized = cv2.resize(mask, (w, h))
            mask_binary = (mask_resized > 0.5).astype(np.uint8)
            area = np.sum(mask_binary)
            if area > best_area:
                best_area = area
                best_mask = mask_binary

        return best_mask

    def extract_corners(self, mask: np.ndarray) -> np.ndarray | None:
        """Extract 4 corner points from a binary board mask.

        Uses convex hull + Douglas-Peucker to approximate the mask
        as a quadrilateral and extract its corners.

        Args:
            mask: Binary mask (H×W uint8, 255 = board region).

        Returns:
            4×2 array of corner points: [top-left, top-right, bottom-right, bottom-left],
            or None if no valid quadrilateral found.
        """
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        # Get the largest contour
        contour = max(contours, key=cv2.contourArea)

        # Approximate as polygon (targeting 4 corners)
        perimeter = cv2.arcLength(contour, True)

        # Try different epsilon values to get a quadrilateral
        for epsilon_factor in [0.01, 0.02, 0.03, 0.05, 0.1]:
            epsilon = epsilon_factor * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                corners = approx.reshape(4, 2).astype(np.float32)
                return self._order_corners(corners)

        # Fallback: use convex hull and find 4 extreme points
        hull = cv2.convexHull(contour)
        if len(hull) < 4:
            return None

        # Find the 4 extreme points (top-left, top-right, bottom-right, bottom-left)
        corners = self._find_extreme_corners(hull.reshape(-1, 2))
        return self._order_corners(corners.astype(np.float32))

    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners as: top-left, top-right, bottom-right, bottom-left.

        Args:
            corners: 4×2 array of corner points (unordered).

        Returns:
            4×2 array of corner points in standard order.
        """
        # Sum of coordinates: top-left has smallest sum, bottom-right has largest
        sums = corners[:, 0] + corners[:, 1]
        # Difference: top-right has smallest diff, bottom-left has largest
        diffs = corners[:, 1] - corners[:, 0]

        ordered = np.array([
            corners[np.argmin(sums)],   # top-left
            corners[np.argmin(diffs)],  # top-right
            corners[np.argmax(sums)],   # bottom-right
            corners[np.argmax(diffs)],  # bottom-left
        ], dtype=np.float32)

        return ordered

    def _find_extreme_corners(self, points: np.ndarray) -> np.ndarray:
        """Find the 4 extreme corners from a set of hull points.

        Args:
            points: N×2 array of hull points.

        Returns:
            4×2 array of extreme corner points.
        """
        # Use the same ordering logic on hull points
        sums = points[:, 0] + points[:, 1]
        diffs = points[:, 1] - points[:, 0]

        return np.array([
            points[np.argmin(sums)],
            points[np.argmin(diffs)],
            points[np.argmax(sums)],
            points[np.argmax(diffs)],
        ])
```

- [ ] **Step 4: 运行测试确认通过**

Run: `source .venv/bin/activate && python -m pytest tests/test_detect.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/detect.py tests/test_detect.py
git commit -m "feat: add board detection module with corner extraction"
```

---

### Task 6: 差异对比 + 标注模块

**Files:**
- Create: `src/compare.py`
- Create: `tests/test_compare.py`

这是阶段③：比较照片网格和图纸网格，标注差异。

- [ ] **Step 1: 写差异对比的测试**

`tests/test_compare.py`:
```python
"""Tests for diff comparison and annotation module."""
import numpy as np

from src.compare import DiffComparator


def test_compare_identical_grids():
    """Test that identical grids produce no differences."""
    grid = np.zeros((3, 3, 3), dtype=np.uint8)
    grid[0, 0] = [255, 0, 0]

    comparator = DiffComparator()
    diffs = comparator.compare(grid, grid)
    assert len(diffs) == 0


def test_compare_detects_color_mismatch():
    """Test that a color difference is detected."""
    photo_grid = np.full((3, 3, 3), [255, 0, 0], dtype=np.uint8)
    blueprint_grid = np.full((3, 3, 3), [0, 255, 0], dtype=np.uint8)

    comparator = DiffComparator()
    diffs = comparator.compare(photo_grid, blueprint_grid)
    assert len(diffs) == 9  # All 9 cells differ


def test_compare_detects_single_mismatch():
    """Test that only mismatched cells are reported."""
    photo_grid = np.full((3, 3, 3), [255, 0, 0], dtype=np.uint8)
    blueprint_grid = np.full((3, 3, 3), [255, 0, 0], dtype=np.uint8)
    # One cell different
    blueprint_grid[1, 1] = [0, 255, 0]

    comparator = DiffComparator()
    diffs = comparator.compare(photo_grid, blueprint_grid)
    assert len(diffs) == 1
    assert diffs[0]["row"] == 1
    assert diffs[0]["col"] == 1


def test_compare_ignores_similar_colors():
    """Test that colors within tolerance are treated as matching."""
    photo_grid = np.full((3, 3, 3), [250, 5, 5], dtype=np.uint8)
    blueprint_grid = np.full((3, 3, 3), [255, 0, 0], dtype=np.uint8)

    comparator = DiffComparator(color_tolerance=20.0)
    diffs = comparator.compare(photo_grid, blueprint_grid)
    assert len(diffs) == 0  # Within tolerance


def test_annotate_creates_output_image():
    """Test that annotate produces an output image with red markers."""
    photo = np.full((200, 200, 3), [200, 200, 200], dtype=np.uint8)
    diffs = [
        {"row": 1, "col": 1, "type": "color_mismatch"},
        {"row": 2, "col": 2, "type": "color_mismatch"},
    ]

    comparator = DiffComparator()
    result = comparator.annotate(photo, diffs, rows=5, cols=5)

    assert result.shape == photo.shape
    assert result.dtype == np.uint8
    # The result should differ from the input (annotation drawn)
    assert not np.array_equal(result, photo)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `source .venv/bin/activate && python -m pytest tests/test_compare.py -v`
Expected: FAIL

- [ ] **Step 3: 实现差异对比 + 标注模块**

`src/compare.py`:
```python
"""
Diff comparison and annotation module.

Compares the photo grid against the blueprint grid,
finds mismatches, and annotates the original photo.
"""
import cv2
import numpy as np


class DiffComparator:
    """Compares photo and blueprint grids and annotates differences."""

    def __init__(self, color_tolerance: float = 30.0):
        """Initialize the comparator.

        Args:
            color_tolerance: Maximum LAB distance to consider colors matching.
                             Colors within this distance are treated as identical.
        """
        self.color_tolerance = color_tolerance

    def compare(
        self,
        photo_grid: np.ndarray,
        blueprint_grid: np.ndarray,
    ) -> list[dict]:
        """Compare photo grid against blueprint grid.

        Args:
            photo_grid: (rows, cols, 3) RGB array from photo extraction.
            blueprint_grid: (rows, cols, 3) RGB array from blueprint parsing.

        Returns:
            List of difference dicts, each with:
            - row, col: position in the grid
            - type: "color_mismatch" | "missing_bead" | "extra_bead"
            - photo_color: [R, G, B] from photo
            - blueprint_color: [R, G, B] from blueprint
        """
        if photo_grid.shape != blueprint_grid.shape:
            raise ValueError(
                f"Grid shape mismatch: photo {photo_grid.shape} "
                f"vs blueprint {blueprint_grid.shape}"
            )

        rows, cols = photo_grid.shape[:2]
        diffs = []

        for r in range(rows):
            for c in range(cols):
                photo_rgb = photo_grid[r, c].astype(np.float32)
                bp_rgb = blueprint_grid[r, c].astype(np.float32)

                # Convert to LAB for perceptual comparison
                photo_lab = self._rgb_to_lab(photo_rgb)
                bp_lab = self._rgb_to_lab(bp_rgb)

                distance = np.sqrt(np.sum((photo_lab - bp_lab) ** 2))

                if distance > self.color_tolerance:
                    # Determine diff type
                    diff_type = "color_mismatch"
                    diffs.append({
                        "row": r,
                        "col": c,
                        "type": diff_type,
                        "photo_color": photo_grid[r, c].tolist(),
                        "blueprint_color": blueprint_grid[r, c].tolist(),
                    })

        return diffs

    def annotate(
        self,
        photo: np.ndarray,
        diffs: list[dict],
        rows: int,
        cols: int,
    ) -> np.ndarray:
        """Annotate the original photo with diff markers.

        Draws red rectangles around mismatched bead positions.

        Args:
            photo: Original photo image (H×W×3 BGR array).
            diffs: List of difference dicts from compare().
            rows: Number of bead rows on the board.
            cols: Number of bead columns on the board.

        Returns:
            Annotated image (same shape as input).
        """
        result = photo.copy()
        h, w = result.shape[:2]
        cell_h = h / rows
        cell_w = w / cols

        for diff in diffs:
            r, c = diff["row"], diff["col"]
            x1 = int(c * cell_w)
            y1 = int(r * cell_h)
            x2 = int((c + 1) * cell_w)
            y2 = int((r + 1) * cell_h)

            # Draw red rectangle
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Add a small "X" marker in the center
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            size = min(cell_h, cell_w) // 6
            cv2.line(result, (cx - size, cy - size), (cx + size, cy + size), (0, 0, 255), 2)
            cv2.line(result, (cx - size, cy + size), (cx + size, cy - size), (0, 0, 255), 2)

        return result

    @staticmethod
    def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
        """Convert a single RGB color to LAB.

        Args:
            rgb: [R, G, B] float32 array.

        Returns:
            [L, a, b] float64 array.
        """
        pixel = np.array([[rgb.astype(np.uint8)]], dtype=np.uint8)
        lab = cv2.cvtColor(pixel, cv2.COLOR_RGB2Lab)
        return lab.flatten().astype(np.float64)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `source .venv/bin/activate && python -m pytest tests/test_compare.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/compare.py tests/test_compare.py
git commit -m "feat: add diff comparison and annotation module"
```

---

### Task 7: CLI 命令行入口

**Files:**
- Create: `src/cli.py`

将所有模块串起来，提供命令行接口。

- [ ] **Step 1: 实现 CLI 模块**

`src/cli.py`:
```python
"""
Command-line interface for Pintconut bead diff detector.

Usage:
    python -m src.cli \\
        --photo photo.jpg \\
        --blueprint blueprint.png \\
        --board-size 29x29 \\
        --model models/beadboard-best.pt \\
        --output result.jpg
"""
import argparse
import os
import sys

import cv2
import numpy as np

from src.color import ColorMatcher
from src.blueprint import parse_blueprint
from src.grid import PerspectiveCorrector, GridExtractor
from src.detect import BoardDetector
from src.compare import DiffComparator


def parse_board_size(size_str: str) -> tuple[int, int]:
    """Parse board size string like '29x29' into (rows, cols)."""
    parts = size_str.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid board size format: {size_str}. Expected 'ROWSxCOLS' like '29x29'.")
    return int(parts[0]), int(parts[1])


def main():
    parser = argparse.ArgumentParser(
        description="Pintconut - 拼豆差异检测器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m src.cli --photo IMG_001.jpg --blueprint pattern.png --board-size 29x29
  python -m src.cli --photo IMG_001.jpg --blueprint pattern.png --board-size 29x29 --model models/best.pt
        """,
    )
    parser.add_argument("--photo", required=True, help="拼豆照片路径")
    parser.add_argument("--blueprint", required=True, help="图纸图片路径")
    parser.add_argument("--board-size", required=True, help="拼板尺寸，格式为 ROWSxCOLS（如 29x29）")
    parser.add_argument("--model", default="models/beadboard-best.pt", help="模型权重路径")
    parser.add_argument("--output", default="annotated_result.jpg", help="输出图片路径")
    parser.add_argument("--color-tolerance", type=float, default=30.0, help="颜色匹配容差（LAB距离）")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.photo):
        print(f"❌ 照片文件不存在: {args.photo}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.blueprint):
        print(f"❌ 图纸文件不存在: {args.blueprint}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.model):
        print(f"❌ 模型文件不存在: {args.model}", file=sys.stderr)
        print(f"   请先训练模型，参见 training/train.py", file=sys.stderr)
        sys.exit(1)

    rows, cols = parse_board_size(args.board_size)

    print(f"📷 照片: {args.photo}")
    print(f"📋 图纸: {args.blueprint}")
    print(f"📐 拼板: {rows}×{cols}")
    print()

    # Step 1: Parse blueprint
    print("📋 解析图纸...")
    blueprint_grid = parse_blueprint(args.blueprint, rows=rows, cols=cols)
    print(f"   图纸网格: {blueprint_grid.shape}")

    # Step 2: Detect board in photo
    print("🔍 检测拼板位置...")
    photo = cv2.imread(args.photo)
    if photo is None:
        print(f"❌ 无法读取照片: {args.photo}", file=sys.stderr)
        sys.exit(1)

    detector = BoardDetector(model_path=args.model)
    mask = detector.detect(photo)

    if mask is None:
        print("❌ 未能在照片中检测到拼板", file=sys.stderr)
        print("   请确保照片中包含拼板，且模型已正确训练", file=sys.stderr)
        sys.exit(1)

    print("   ✅ 检测到拼板区域")

    # Step 3: Extract corners and correct perspective
    print("📐 透视校正...")
    corners = detector.extract_corners(mask)
    if corners is None:
        print("❌ 无法提取拼板角点", file=sys.stderr)
        sys.exit(1)

    corrector = PerspectiveCorrector()
    # Use a square output if rows == cols, otherwise rectangle
    output_w = cols * 20
    output_h = rows * 20
    corrected = corrector.correct(photo, corners, output_size=(output_w, output_h))
    print(f"   校正后尺寸: {corrected.shape[1]}×{corrected.shape[0]}")

    # Step 4: Extract grid
    print("🔲 提取网格...")
    extractor = GridExtractor()
    photo_grid = extractor.extract(corrected, rows=rows, cols=cols)
    print(f"   照片网格: {photo_grid.shape}")

    # Step 5: Compare
    print("🔎 对比差异...")
    comparator = DiffComparator(color_tolerance=args.color_tolerance)
    diffs = comparator.compare(photo_grid, blueprint_grid)

    if not diffs:
        print("\n🎉 完美！没有发现差异！")
    else:
        print(f"\n⚠️  发现 {len(diffs)} 处差异：")
        for d in diffs:
            print(f"   行{d['row']+1} 列{d['col']+1}: "
                  f"实际={[d['photo_color'][i] for i in range(3)]} "
                  f"应为 {[d['blueprint_color'][i] for i in range(3)]}")

    # Step 6: Annotate and save
    print(f"\n💾 保存标注图到 {args.output}...")
    # Annotate on the perspective-corrected image
    annotated = comparator.annotate(corrected, diffs, rows=rows, cols=cols)
    cv2.imwrite(args.output, annotated)
    print(f"   ✅ 已保存")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 测试 CLI help 输出**

Run: `source .venv/bin/activate && python -m src.cli --help`
Expected: 显示帮助信息，无报错

- [ ] **Step 3: Commit**

```bash
git add src/cli.py
git commit -m "feat: add CLI entry point connecting all modules"
```

---

### Task 8: 半自动标注工具

**Files:**
- Create: `training/semi_auto_label.py`

基于 Task 5 中验证过的 FastSAM，构建交互式半自动标注工具。

- [ ] **Step 1: 实现半自动标注工具**

`training/semi_auto_label.py`:
```python
"""
Semi-automatic labeling tool for bead board dataset.

Uses FastSAM to generate candidate regions, then prompts the user
to select the correct board region by number.

Usage:
    python training/semi_auto_label.py --input training/photos/ --output training/dataset/
"""
import argparse
import os
import sys

import cv2
import numpy as np


# Color palette for drawing candidate regions (up to 10 colors)
COLORS = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue (BGR)
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 255),  # Orange-ish
    (0, 128, 255),  # Light orange
    (255, 128, 0),  # Azure
    (128, 255, 0),  # Chartreuse
]


def is_rectangular(mask, threshold=0.80):
    """Check if a mask region is approximately rectangular."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return False, 0
    contour = max(contours, key=cv2.contourArea)
    mask_area = cv2.contourArea(contour)
    if mask_area < 1000:
        return False, 0
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    ratio = mask_area / hull_area if hull_area > 0 else 0
    return ratio > threshold, ratio


def process_image(model, image_path, output_images_dir, output_labels_dir):
    """Process a single image: segment, show candidates, get user selection.

    Returns:
        True if labeled successfully, False if skipped.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"  ❌ Cannot read: {image_path}")
        return False

    h, w = image.shape[:2]
    img_area = h * w

    # Run FastSAM
    results = model(image, device="cpu", retina_masks=True, imgsz=640, conf=0.4, iou=0.9)

    if results[0].masks is None:
        print("  ⚠️  No regions detected")
        return False

    masks = results[0].masks.data.cpu().numpy()

    # Filter candidates
    candidates = []
    for i, mask in enumerate(masks):
        mask_resized = cv2.resize(mask, (w, h))
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        area = np.sum(mask_binary)
        area_ratio = area / img_area

        is_rect, rect_ratio = is_rectangular(mask_binary)

        # Accept regions that are reasonably sized and somewhat rectangular
        if 0.05 < area_ratio < 0.95 and rect_ratio > 0.5:
            candidates.append({
                "index": i,
                "mask": mask_binary,
                "area_ratio": area_ratio,
                "rect_ratio": rect_ratio,
            })

    # Sort by area (largest first)
    candidates.sort(key=lambda c: c["area"], reverse=True)

    # Take top 10 candidates
    candidates = candidates[:10]

    if not candidates:
        print("  ⚠️  No suitable candidates found")
        return False

    # Draw candidates on image
    display = image.copy()
    for ci, cand in enumerate(candidates):
        color = COLORS[ci % len(COLORS)]
        # Draw contour
        contours, _ = cv2.findContours(
            cand["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(display, contours, -1, color, 2)

        # Draw label number at centroid
        M = cv2.moments(contours[0])
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 100, 100 + ci * 50

        # Draw number with background
        label = f"{ci}: {cand['area_ratio']:.0%} rect={cand['rect_ratio']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(display, (cx - 2, cy - th - 4), (cx + tw + 2, cy + 4), (0, 0, 0), -1)
        cv2.putText(display, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show image
    # Scale to fit screen if too large
    screen_h = 900
    if display.shape[0] > screen_h:
        scale = screen_h / display.shape[0]
        display = cv2.resize(display, None, fx=scale, fy=scale)

    cv2.imshow("Select board region", display)
    print(f"  候选区域已显示。请输入正确的区域编号 (0-{len(candidates)-1})，或 s 跳过，q 退出：")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            print("  退出")
            return False
        elif key == ord('s'):
            cv2.destroyAllWindows()
            print("  跳过")
            return False
        elif ord('0') <= key <= ord('9'):
            idx = key - ord('0')
            if idx < len(candidates):
                cv2.destroyAllWindows()
                selected = candidates[idx]
                print(f"  ✅ 选中区域 {idx}: 面积={selected['area_ratio']:.0%}")
                # Save label
                _save_label(selected["mask"], image_path, output_images_dir, output_labels_dir, w, h)
                return True
            else:
                print(f"  区域 {idx} 不存在，请重新选择")

    cv2.destroyAllWindows()
    return False


def _save_label(mask, image_path, output_images_dir, output_labels_dir, img_w, img_h):
    """Save the selected mask as a YOLO segmentation label."""
    basename = os.path.splitext(os.path.basename(image_path))[0]

    # Copy image to dataset
    src_img = cv2.imread(image_path)
    dst_img_path = os.path.join(output_images_dir, f"{basename}.jpg")
    cv2.imwrite(dst_img_path, src_img)

    # Convert mask to polygon (normalized coordinates)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return

    # Use the largest contour
    contour = max(contours, key=cv2.contourArea)

    # Simplify polygon to reduce points
    epsilon = 0.005 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Normalize coordinates
    points = approx.reshape(-1, 2).astype(np.float64)
    points[:, 0] /= img_w
    points[:, 1] /= img_h

    # Write YOLO label format: class x1 y1 x2 y2 ...
    label_path = os.path.join(output_labels_dir, f"{basename}.txt")
    with open(label_path, "w") as f:
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
        f.write(f"0 {coords}\n")


def main():
    parser = argparse.ArgumentParser(description="Semi-automatic bead board labeling tool")
    parser.add_argument("--input", default="training/photos", help="Input photos directory")
    parser.add_argument("--output", default="training/dataset", help="Output dataset directory")
    args = parser.parse_args()

    input_dir = args.input
    train_images = os.path.join(args.output, "images", "train")
    train_labels = os.path.join(args.output, "labels", "train")
    os.makedirs(train_images, exist_ok=True)
    os.makedirs(train_labels, exist_ok=True)

    print("=" * 60)
    print("Pintconut 半自动标注工具")
    print("=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {args.output}")
    print("操作说明: 0-9 选择区域, s 跳过, q 退出")
    print()

    # Load FastSAM
    print("📦 加载 FastSAM...")
    from ultralytics import FastSAM
    model = FastSAM("FastSAM-s.pt")
    print("✅ 模型加载完成\n")

    # Process images
    images = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if not images:
        print(f"❌ 输入目录中没有图片: {input_dir}")
        sys.exit(1)

    print(f"找到 {len(images)} 张图片\n")

    success = 0
    skipped = 0
    for img_file in images:
        img_path = os.path.join(input_dir, img_file)
        print(f"📷 {img_file}")
        if process_image(model, img_path, train_images, train_labels):
            success += 1
        else:
            skipped += 1
        print()

    print("=" * 60)
    print(f"📊 完成: {success} 张标注, {skipped} 张跳过")
    print(f"📁 标注数据: {train_images}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 测试工具能否启动**

Run: `source .venv/bin/activate && python training/semi_auto_label.py --help`
Expected: 显示帮助信息，无报错

- [ ] **Step 3: Commit**

```bash
git add training/semi_auto_label.py
git commit -m "feat: add semi-automatic labeling tool using FastSAM"
```

---

### Task 9: 训练 + 验证脚本

**Files:**
- Create: `training/train.py`
- Create: `training/validate.py`

- [ ] **Step 1: 创建 data.yaml 模板生成逻辑**

`training/train.py`:
```python
"""
Training script for bead board YOLOv8-seg model.

Usage:
    python training/train.py --data training/dataset/data.yaml
    python training/train.py --data training/dataset/data.yaml --epochs 200
"""
import argparse
import os
import sys


def generate_data_yaml(dataset_dir: str) -> str:
    """Generate data.yaml for the dataset if it doesn't exist."""
    yaml_path = os.path.join(dataset_dir, "data.yaml")

    if os.path.exists(yaml_path):
        return yaml_path

    yaml_content = f"""# Auto-generated dataset config
path: {os.path.abspath(dataset_dir)}
train: images/train
val: images/valid

names:
  0: beadboard
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"📝 生成 data.yaml: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8-seg for bead board detection")
    parser.add_argument("--data", default="training/dataset/data.yaml", help="Dataset config path")
    parser.add_argument("--model", default="yolov8n-seg.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="cpu", help="Device: cpu, 0, 0,1")
    parser.add_argument("--project", default="training/runs", help="Output directory")
    parser.add_argument("--name", default="beadboard-v1", help="Run name")
    args = parser.parse_args()

    # Ensure data.yaml exists
    dataset_dir = os.path.dirname(args.data)
    if not os.path.exists(args.data):
        args.data = generate_data_yaml(dataset_dir)

    # Validate dataset
    for split in ["train", "valid"]:
        split_dir = os.path.join(dataset_dir, "images", split)
        if not os.path.exists(split_dir):
            print(f"❌ 数据集目录不存在: {split_dir}", file=sys.stderr)
            print(f"   请先使用 semi_auto_label.py 标注数据", file=sys.stderr)
            sys.exit(1)
        n_images = len([f for f in os.listdir(split_dir) if f.endswith(('.jpg', '.png'))])
        print(f"📁 {split}: {n_images} 张图片")

    print(f"\n🏋️ 开始训练...")
    print(f"   模型: {args.model}")
    print(f"   数据: {args.data}")
    print(f"   Epochs: {args.epochs}")
    print(f"   设备: {args.device}")
    print()

    from ultralytics import YOLO
    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        single_cls=True,
        patience=50,
        exist_ok=True,
    )

    best_weights = os.path.join(args.project, args.name, "weights", "best.pt")
    print(f"\n✅ 训练完成！最佳权重: {best_weights}")
    print(f"   将此文件复制到 models/beadboard-best.pt 即可使用")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 创建验证脚本**

`training/validate.py`:
```python
"""
Validation script for trained bead board detection model.

Usage:
    python training/validate.py --model training/runs/beadboard-v1/weights/best.pt --image test.jpg
    python training/validate.py --model training/runs/beadboard-v1/weights/best.pt --image training/test_images/
"""
import argparse
import os
import sys

import cv2
import numpy as np


def validate_single(model, image_path, output_dir=None):
    """Validate model on a single image and display results."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"  ❌ Cannot read: {image_path}")
        return False

    results = model(image, verbose=False)

    if results[0].masks is None:
        print(f"  ⚠️  未检测到拼板: {os.path.basename(image_path)}")
        if output_dir:
            cv2.imwrite(os.path.join(output_dir, f"fail_{os.path.basename(image_path)}"), image)
        return False

    # Draw mask on image
    masks = results[0].masks.data.cpu().numpy()
    h, w = image.shape[:2]
    annotated = image.copy()

    for mask in masks:
        mask_resized = cv2.resize(mask, (w, h))
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        overlay = annotated.copy()
        overlay[mask_binary > 0] = [0, 255, 0]
        annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)

    print(f"  ✅ 检测到拼板: {os.path.basename(image_path)}")

    if output_dir:
        out_path = os.path.join(output_dir, f"det_{os.path.basename(image_path)}")
        cv2.imwrite(out_path, annotated)
        print(f"     保存到: {out_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Validate trained bead board model")
    parser.add_argument("--model", required=True, help="Path to trained model weights")
    parser.add_argument("--image", required=True, help="Test image or directory")
    parser.add_argument("--output", default="training/validation_results", help="Output directory")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"❌ 模型文件不存在: {args.model}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print("📦 加载模型...")
    from ultralytics import YOLO
    model = YOLO(args.model)
    print("✅ 模型加载完成\n")

    # Collect test images
    if os.path.isdir(args.image):
        images = sorted([
            os.path.join(args.image, f)
            for f in os.listdir(args.image)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
    else:
        images = [args.image]

    print(f"🔍 验证 {len(images)} 张图片\n")

    success = 0
    for img_path in images:
        if validate_single(model, img_path, args.output):
            success += 1

    print(f"\n{'='*60}")
    print(f"📊 结果: {success}/{len(images)} 张成功检测")
    print(f"📁 输出: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 测试两个脚本能启动**

Run: `source .venv/bin/activate && python training/train.py --help && python training/validate.py --help`
Expected: 两个都显示帮助信息

- [ ] **Step 4: Commit**

```bash
git add training/train.py training/validate.py
git commit -m "feat: add training and validation scripts for YOLOv8-seg"
```

---

### Task 10: 照片批量重命名辅助工具

**Files:**
- Create: `training/collect_helper.py`

- [ ] **Step 1: 实现重命名工具**

`training/collect_helper.py`:
```python
"""
Photo collection helper - batch rename photos for training.

Usage:
    python training/collect_helper.py --input ~/Downloads/ --output training/photos/
"""
import argparse
import os
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(description="Batch rename photos for training dataset")
    parser.add_argument("--input", required=True, help="Source directory with raw photos")
    parser.add_argument("--output", default="training/photos", help="Output directory")
    parser.add_argument("--prefix", default="board", help="Filename prefix")
    parser.add_argument("--copy", action="store_true", default=True, help="Copy files (default)")
    parser.add_argument("--move", action="store_true", help="Move files instead of copy")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ 目录不存在: {args.input}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # Collect all image files
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    files = sorted([
        f for f in os.listdir(args.input)
        if os.path.splitext(f)[1].lower() in extensions
    ])

    if not files:
        print(f"❌ 目录中没有图片: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"找到 {len(files)} 张图片")

    # Find the next available number in output
    existing = [
        f for f in os.listdir(args.output)
        if f.startswith(args.prefix)
    ]
    start_idx = len(existing) + 1

    action = shutil.move if args.move else shutil.copy2
    action_name = "移动" if args.move else "复制"

    for i, filename in enumerate(files):
        ext = os.path.splitext(filename)[1]
        new_name = f"{args.prefix}_{start_idx + i:04d}{ext}"
        src = os.path.join(args.input, filename)
        dst = os.path.join(args.output, new_name)
        action(src, dst)
        print(f"  {action_name}: {filename} → {new_name}")

    print(f"\n✅ 完成: {len(files)} 张图片已{action_name}到 {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 测试**

Run: `source .venv/bin/activate && python training/collect_helper.py --help`
Expected: 显示帮助信息

- [ ] **Step 3: Commit**

```bash
git add training/collect_helper.py
git commit -m "feat: add photo collection helper for batch renaming"
```

---

### Task 11: 集成测试 + .gitignore

**Files:**
- Create: `.gitignore`
- Create: `tests/test_integration.py`

确保所有模块能正确串联工作。

- [ ] **Step 1: 创建 .gitignore**

```
.venv/
__pycache__/
*.pyc
*.pt
!FastSAM-s.pt
training/dataset/
training/photos/
training/runs/
training/auto_label_results/
training/validation_results/
models/
*.egg-info/
dist/
build/
.pytest_cache/
annotated_result.jpg
tests/fixtures/
```

- [ ] **Step 2: 运行全部单元测试**

Run: `source .venv/bin/activate && python -m pytest tests/ -v`
Expected: 所有测试 PASS

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: add .gitignore"
```

---

## 实现顺序总结

```
Task 1  项目初始化 + 依赖
  ↓
Task 2  色盘数据 + 颜色匹配（其他模块的基础依赖）
  ↓
Task 3  图纸解析
  ↓
Task 4  网格提取（透视校正 + 采样）
  ↓
Task 5  拼板定位（YOLOv8-seg 推理 + 角点提取）
  ↓
Task 6  差异对比 + 标注
  ↓
Task 7  CLI 入口（串联所有模块）
  ↓
Task 8  半自动标注工具（训练数据准备）
  ↓
Task 9  训练 + 验证脚本
  ↓
Task 10 照片批量重命名工具
  ↓
Task 11 集成测试 + .gitignore
```

**Task 1-7 是推理管线**（用于检测差异），**Task 8-10 是训练管线**（用于准备数据和训练模型）。

完成 Task 1-7 后，推理管线代码就位。但要实际运行，还需要训练好的模型（通过 Task 8-9 产出）。
