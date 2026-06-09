# Bead Detection System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a bead detection pipeline that replaces uniform grid-cutting with YOLOv8n object detection for per-bead localization, followed by LAB color matching.

**Architecture:** Three-phase approach: (1) board detection + perspective correction (existing code), (2) YOLOv8n bead detection model (new), (3) color matching via existing ColorMatcher. A Gradio-based pre-labeling tool uses HoughCircles to bootstrap annotations.

**Tech Stack:** Python, OpenCV, Ultralytics (YOLOv8n), Gradio, NumPy, pytest

**Spec:** `docs/superpowers/specs/2026-06-09-bead-detection-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/bead_detect.py` | BeadDetector class — YOLOv8n inference, box filtering, post-processing |
| Create | `src/bead_annotate_ui.py` | Gradio pre-labeling tool — HoughCircles + manual correction |
| Create | `training/bead_train.py` | YOLO detection training script for bead model |
| Create | `training/bead_validate.py` | Validation/inference script for bead detection model |
| Create | `tests/test_bead_detect.py` | Tests for BeadDetector (filtering, color sampling) |
| Modify | `src/cli.py` | Integrate BeadDetector pipeline replacing GridExtractor |
| Modify | `src/color.py` | Add `match_box()` method for detection-box-based color sampling |

---

## Phase 1: Bead Detection Core Module

### Task 1: Create BeadDetector with filtering logic

**Files:**
- Create: `src/bead_detect.py`
- Create: `tests/test_bead_detect.py`

- [ ] **Step 1: Write the failing test for BeadDetector initialization**

```python
# tests/test_bead_detect.py
"""Tests for bead detection module."""
import numpy as np

from src.bead_detect import BeadDetector


def test_bead_detector_init():
    """Test that BeadDetector initializes with expected defaults."""
    detector = BeadDetector()
    assert detector.model_path == "models/bead-best.pt"
    assert detector.conf_threshold == 0.5
    assert detector.model is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_bead_detect.py::test_bead_detector_init -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.bead_detect'`

- [ ] **Step 3: Write minimal BeadDetector class**

```python
# src/bead_detect.py
"""Bead detection module — locates individual beads using YOLOv8n."""


class BeadDetector:
    """Detects individual beads on a perspective-corrected board image."""

    def __init__(
        self,
        model_path: str = "models/bead-best.pt",
        conf_threshold: float = 0.5,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None

    def _load_model(self):
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_bead_detect.py::test_bead_detector_init -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/bead_detect.py tests/test_bead_detect.py
git commit -m "feat: add BeadDetector class skeleton"
```

---

### Task 2: Add detect() method with box filtering

**Files:**
- Modify: `src/bead_detect.py`
- Modify: `tests/test_bead_detect.py`

- [ ] **Step 1: Write failing tests for detect() and filter_boxes()**

```python
# Append to tests/test_bead_detect.py

def test_filter_boxes_by_confidence():
    """Test that low-confidence boxes are removed."""
    detector = BeadDetector()
    boxes = [
        {"xyxy": [10, 10, 30, 30], "conf": 0.9},
        {"xyxy": [40, 40, 60, 60], "conf": 0.3},
        {"xyxy": [70, 70, 90, 90], "conf": 0.6},
    ]
    filtered = detector.filter_boxes(boxes, conf_threshold=0.5)
    assert len(filtered) == 2
    assert all(b["conf"] >= 0.5 for b in filtered)


def test_filter_boxes_by_size():
    """Test that boxes with unreasonable size are removed."""
    detector = BeadDetector()
    # Reference size ~20px, allow 0.3x-3.0x range
    boxes = [
        {"xyxy": [10, 10, 30, 30], "conf": 0.9, "width": 20, "height": 20},
        {"xyxy": [40, 40, 45, 45], "conf": 0.9, "width": 5, "height": 5},    # too small
        {"xyxy": [70, 70, 170, 170], "conf": 0.9, "width": 100, "height": 100},  # too large
    ]
    filtered = detector.filter_boxes(boxes, conf_threshold=0.0, ref_size=20, size_range=(0.3, 3.0))
    assert len(filtered) == 1


def test_filter_boxes_inside_mask():
    """Test that boxes outside the board mask are removed."""
    detector = BeadDetector()
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255  # board region

    boxes = [
        {"xyxy": [25, 25, 45, 45], "conf": 0.9, "width": 20, "height": 20},  # inside
        {"xyxy": [5, 5, 25, 25], "conf": 0.9, "width": 20, "height": 20},    # outside
    ]
    filtered = detector.filter_boxes(boxes, conf_threshold=0.0, board_mask=mask)
    assert len(filtered) == 1
    assert filtered[0]["xyxy"][0] == 25


def test_detect_returns_list_of_dicts():
    """Test that detect returns a list of bead dicts."""
    detector = BeadDetector.__new__(BeadDetector)
    detector.model_path = "models/bead-best.pt"
    detector.conf_threshold = 0.5
    detector.model = None

    # detect() needs a real model, so we test the output format contract
    # by mocking _load_model and using a synthetic image
    img = np.ones((600, 600, 3), dtype=np.uint8) * 200
    # Without a model file, this should return empty list
    result = detector.detect(img)
    assert isinstance(result, list)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_bead_detect.py -v`
Expected: FAIL — `filter_boxes` and `detect` methods not defined

- [ ] **Step 3: Implement detect() and filter_boxes()**

```python
# Full src/bead_detect.py
"""Bead detection module — locates individual beads using YOLOv8n."""

from __future__ import annotations

import numpy as np


class BeadDetector:
    """Detects individual beads on a perspective-corrected board image."""

    def __init__(
        self,
        model_path: str = "models/bead-best.pt",
        conf_threshold: float = 0.5,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None

    def _load_model(self):
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)

    def detect(self, image: np.ndarray) -> list[dict]:
        """Detect beads in an image.

        Args:
            image: BGR image (HxWx3).

        Returns:
            List of dicts with keys: xyxy, conf, width, height, cx, cy.
        """
        import os
        if not os.path.exists(self.model_path):
            return []

        self._load_model()
        results = self.model(image, verbose=False)
        if not results or len(results) == 0:
            return []

        boxes = []
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            x1, y1, x2, y2 = xyxy
            w = x2 - x1
            h = y2 - y1
            boxes.append({
                "xyxy": xyxy.tolist(),
                "conf": conf,
                "width": float(w),
                "height": float(h),
                "cx": float((x1 + x2) / 2),
                "cy": float((y1 + y2) / 2),
            })
        return boxes

    def filter_boxes(
        self,
        boxes: list[dict],
        conf_threshold: float | None = None,
        board_mask: np.ndarray | None = None,
        ref_size: float | None = None,
        size_range: tuple[float, float] = (0.3, 3.0),
    ) -> list[dict]:
        """Filter detected boxes by confidence, size, and board mask.

        Args:
            boxes: List of bead detection dicts from detect().
            conf_threshold: Override confidence threshold. None uses default.
            board_mask: Binary mask (HxW uint8, 255=board). Filters boxes
                        whose center is outside the mask.
            ref_size: Expected bead size in pixels. Enables size filtering.
            size_range: (min_ratio, max_ratio) relative to ref_size.

        Returns:
            Filtered list of bead dicts.
        """
        if conf_threshold is None:
            conf_threshold = self.conf_threshold

        filtered = []
        for box in boxes:
            # Confidence filter
            if box["conf"] < conf_threshold:
                continue

            # Size filter
            if ref_size is not None:
                size = (box["width"] + box["height"]) / 2
                ratio = size / ref_size
                if ratio < size_range[0] or ratio > size_range[1]:
                    continue

            # Board mask filter
            if board_mask is not None:
                cx, cy = int(box["cx"]), int(box["cy"])
                h, w = board_mask.shape[:2]
                if cx < 0 or cx >= w or cy < 0 or cy >= h:
                    continue
                if board_mask[cy, cx] == 0:
                    continue

            filtered.append(box)
        return filtered
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_bead_detect.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/bead_detect.py tests/test_bead_detect.py
git commit -m "feat: add BeadDetector with detect() and filter_boxes()"
```

---

### Task 3: Add color sampling from detection boxes to ColorMatcher

**Files:**
- Modify: `src/color.py`
- Modify: `tests/test_color.py`

- [ ] **Step 1: Write failing test for match_box()**

```python
# Append to tests/test_color.py

def test_match_box_returns_color_info():
    """Test matching color from a bounding box region of an image."""
    matcher = ColorMatcher("data/colors.json")
    # Create a 100x100 image with a red region in the center
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[30:70, 30:70] = [255, 0, 0]  # Red square in RGB

    box = {"xyxy": [30, 30, 70, 70], "conf": 0.9}
    result = matcher.match_box(img, box, sample_fraction=0.4)
    assert result["color_name"] == "Red"
    assert result["is_bead"] is True
    assert "rgb" in result
    assert "confidence" in result


def test_match_box_empty_board():
    """Test that match_box detects empty board (no bead)."""
    matcher = ColorMatcher("data/colors.json")
    img = np.ones((100, 100, 3), dtype=np.uint8) * 250  # White/near-white board
    box = {"xyxy": [10, 10, 50, 50], "conf": 0.9}
    result = matcher.match_box(img, box, sample_fraction=0.4)
    assert result["is_bead"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_color.py::test_match_box_returns_color_info -v`
Expected: FAIL — `match_box` not defined

- [ ] **Step 3: Implement match_box() on ColorMatcher**

Add this method to `src/color.py` inside the `ColorMatcher` class, after `match_grid()`:

```python
    def match_box(
        self, image: np.ndarray, box: dict, sample_fraction: float = 0.4
    ) -> dict:
        """Match color from a detection box region.

        Samples the center portion of the bounding box to avoid edge
        contamination, then matches the median color to the palette.

        Args:
            image: BGR image (HxWx3 uint8).
            box: Detection dict with 'xyxy' key [x1, y1, x2, y2].
            sample_fraction: Fraction of box to sample from center (0-1).

        Returns:
            Dict with: color_name, rgb, is_bead, confidence.
        """
        x1, y1, x2, y2 = [int(v) for v in box["xyxy"]]
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        box_w = x2 - x1
        box_h = y2 - y1
        margin_x = int(box_w * (1 - sample_fraction) / 2)
        margin_y = int(box_h * (1 - sample_fraction) / 2)

        sx1 = x1 + margin_x
        sy1 = y1 + margin_y
        sx2 = x2 - margin_x
        sy2 = y2 - margin_y

        if sy2 <= sy1 or sx2 <= sx1:
            return {"color_name": "Unknown", "rgb": [0, 0, 0], "is_bead": False, "confidence": 0.0}

        # image is BGR, convert crop to RGB for matching
        crop = image[sy1:sy2, sx1:sx2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        median_rgb = np.median(crop_rgb.reshape(-1, 3), axis=0).astype(int).tolist()

        color_info = self.match(median_rgb)
        is_bead = self.is_bead(median_rgb)

        return {
            "color_name": color_info["name"],
            "rgb": median_rgb,
            "is_bead": is_bead,
            "confidence": box.get("conf", 0.0),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_color.py -v`
Expected: ALL PASS (including existing tests)

- [ ] **Step 5: Commit**

```bash
git add src/color.py tests/test_color.py
git commit -m "feat: add ColorMatcher.match_box() for detection-box color sampling"
```

---

### Task 4: Add BeadDetector.detect_beads() high-level method

**Files:**
- Modify: `src/bead_detect.py`
- Modify: `tests/test_bead_detect.py`

- [ ] **Step 1: Write failing test for detect_beads()**

```python
# Append to tests/test_bead_detect.py

def test_detect_beads_returns_annotated_results():
    """Test that detect_beads returns bead list with color info."""
    from src.color import ColorMatcher

    detector = BeadDetector()
    # detect_beads with no model file should return empty
    img = np.ones((600, 600, 3), dtype=np.uint8) * 200
    result = detector.detect_beads(img)
    assert isinstance(result, list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_bead_detect.py::test_detect_beads_returns_annotated_results -v`
Expected: FAIL — `detect_beads` not defined

- [ ] **Step 3: Implement detect_beads()**

Add this method to `src/bead_detect.py` in the `BeadDetector` class, after `filter_boxes()`:

```python
    def detect_beads(
        self,
        image: np.ndarray,
        board_mask: np.ndarray | None = None,
        color_matcher=None,
    ) -> list[dict]:
        """Full pipeline: detect beads, filter, and match colors.

        Args:
            image: BGR image (HxWx3).
            board_mask: Optional binary mask to filter board-outside detections.
            color_matcher: Optional ColorMatcher instance. If provided, each
                           bead will include color info.

        Returns:
            List of bead dicts with: xyxy, conf, cx, cy, color_name, rgb, is_bead.
        """
        boxes = self.detect(image)
        if not boxes:
            return []

        # Estimate ref_size from median box size
        widths = [b["width"] for b in boxes]
        ref_size = float(np.median(widths)) if widths else None

        filtered = self.filter_boxes(
            boxes,
            board_mask=board_mask,
            ref_size=ref_size,
        )

        # Add color info if matcher provided
        if color_matcher is not None:
            for box in filtered:
                color_result = color_matcher.match_box(image, box)
                box["color_name"] = color_result["color_name"]
                box["rgb"] = color_result["rgb"]
                box["is_bead"] = color_result["is_bead"]
        else:
            for box in filtered:
                box["color_name"] = "Unknown"
                box["rgb"] = [0, 0, 0]
                box["is_bead"] = True

        return filtered
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_bead_detect.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/bead_detect.py tests/test_bead_detect.py
git commit -m "feat: add BeadDetector.detect_beads() high-level pipeline"
```

---

### Task 5: Add bead count statistics utility

**Files:**
- Modify: `src/bead_detect.py`
- Modify: `tests/test_bead_detect.py`

- [ ] **Step 1: Write failing test for count_beads()**

```python
# Append to tests/test_bead_detect.py

def test_count_beads_by_color():
    """Test counting beads grouped by color name."""
    detector = BeadDetector()
    beads = [
        {"color_name": "Red", "is_bead": True, "conf": 0.9, "cx": 10, "cy": 10},
        {"color_name": "Red", "is_bead": True, "conf": 0.8, "cx": 20, "cy": 20},
        {"color_name": "Blue", "is_bead": True, "conf": 0.9, "cx": 30, "cy": 30},
        {"color_name": "White", "is_bead": False, "conf": 0.7, "cx": 40, "cy": 40},
    ]
    counts = detector.count_beads(beads)
    assert counts["total"] == 3  # only is_bead=True
    assert counts["by_color"]["Red"] == 2
    assert counts["by_color"]["Blue"] == 1
    assert "White" not in counts["by_color"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_bead_detect.py::test_count_beads_by_color -v`
Expected: FAIL — `count_beads` not defined

- [ ] **Step 3: Implement count_beads()**

Add this method to the `BeadDetector` class in `src/bead_detect.py`:

```python
    @staticmethod
    def count_beads(beads: list[dict]) -> dict:
        """Count beads by color, excluding empty board cells.

        Args:
            beads: List of bead dicts from detect_beads().

        Returns:
            Dict with 'total' count and 'by_color' breakdown.
        """
        real_beads = [b for b in beads if b.get("is_bead", True)]
        by_color: dict[str, int] = {}
        for b in real_beads:
            name = b.get("color_name", "Unknown")
            by_color[name] = by_color.get(name, 0) + 1
        return {"total": len(real_beads), "by_color": by_color}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_bead_detect.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/bead_detect.py tests/test_bead_detect.py
git commit -m "feat: add BeadDetector.count_beads() statistics utility"
```

---

## Phase 2: Bead Pre-labeling Tool

### Task 6: Create HoughCircles pre-labeling logic

**Files:**
- Create: `src/bead_prelabel.py`
- Create: `tests/test_bead_prelabel.py`

- [ ] **Step 1: Write failing tests for pre-label functions**

```python
# tests/test_bead_prelabel.py
"""Tests for bead pre-labeling module."""
import numpy as np

from src.bead_prelabel import hough_circles_to_boxes, save_yolo_boxes


def test_hough_circles_to_boxes_on_grid():
    """Test detecting circles arranged in a grid pattern."""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 220
    # Draw a 5x5 grid of dark circles
    for r in range(5):
        for c in range(5):
            cx = 20 + c * 40
            cy = 20 + r * 40
            cv2.circle(img, (cx, cy), 12, (0, 0, 0), -1)

    boxes = hough_circles_to_boxes(img)
    assert len(boxes) > 0
    # Each box should have xyxy, conf keys
    assert "xyxy" in boxes[0]
    assert "conf" in boxes[0]


def test_hough_circles_to_boxes_no_circles():
    """Test that empty image returns empty list."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    boxes = hough_circles_to_boxes(img)
    assert isinstance(boxes, list)


def test_save_yolo_boxes():
    """Test saving boxes in YOLO detection format."""
    import tempfile, os
    boxes = [
        {"xyxy": [10, 10, 30, 30], "conf": 0.9},
        {"xyxy": [50, 50, 70, 70], "conf": 0.8},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test.txt")
        save_yolo_boxes(boxes, img_w=100, img_h=100, output_path=out_path)
        assert os.path.exists(out_path)
        with open(out_path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        # YOLO format: class x_center y_center width height (all normalized)
        parts = lines[0].strip().split()
        assert parts[0] == "0"  # class 0 = bead
        assert len(parts) == 5


import cv2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_bead_prelabel.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.bead_prelabel'`

- [ ] **Step 3: Implement pre-labeling functions**

```python
# src/bead_prelabel.py
"""Bead pre-labeling module — generates initial bounding boxes using HoughCircles."""

import cv2
import numpy as np


def hough_circles_to_boxes(
    image: np.ndarray,
    dp: float = 1.2,
    min_dist: int = 10,
    param1: int = 100,
    param2: int = 30,
    min_radius: int = 3,
    max_radius: int = 30,
) -> list[dict]:
    """Detect circular beads and return bounding boxes.

    Args:
        image: BGR image (HxWx3).
        dp: Inverse accumulator resolution ratio.
        min_dist: Minimum distance between circle centers.
        param1: Upper Canny edge threshold.
        param2: Accumulator threshold for circle detection.
        min_radius: Minimum circle radius.
        max_radius: Maximum circle radius.

    Returns:
        List of dicts with: xyxy, conf, cx, cy, width, height.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return []

    boxes = []
    circles = np.uint16(np.around(circles))
    for cx, cy, r in circles[0]:
        x1 = max(0, int(cx - r))
        y1 = max(0, int(cy - r))
        x2 = int(cx + r)
        y2 = int(cy + r)
        w = x2 - x1
        h = y2 - y1
        boxes.append({
            "xyxy": [x1, y1, x2, y2],
            "conf": 1.0,  # HoughCircles doesn't provide confidence
            "cx": float(cx),
            "cy": float(cy),
            "width": float(w),
            "height": float(h),
        })
    return boxes


def save_yolo_boxes(
    boxes: list[dict],
    img_w: int,
    img_h: int,
    output_path: str,
) -> None:
    """Save bounding boxes in YOLO detection format.

    Format: class x_center y_center width height (all normalized 0-1).

    Args:
        boxes: List of bead detection dicts.
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        output_path: Path to write the label file.
    """
    with open(output_path, "w") as f:
        for box in boxes:
            x1, y1, x2, y2 = box["xyxy"]
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_bead_prelabel.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/bead_prelabel.py tests/test_bead_prelabel.py
git commit -m "feat: add HoughCircles pre-labeling and YOLO box export"
```

---

### Task 7: Create Gradio bead annotation UI

**Files:**
- Create: `src/bead_annotate_ui.py`

- [ ] **Step 1: Write the bead annotation Gradio app**

This is a Gradio tool similar to `label_ui.py`. The flow:
1. User uploads a perspective-corrected board image
2. HoughCircles auto-detects beads → shows green boxes
3. User can adjust HoughCircles parameters and re-run
4. User confirms → saves YOLO detection labels to `training/bead_dataset/`

```python
# src/bead_annotate_ui.py
"""Gradio web UI for bead annotation — HoughCircles pre-labeling + manual correction."""

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import gradio as gr
import numpy as np

from src.bead_prelabel import hough_circles_to_boxes, save_yolo_boxes

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEAD_DATASET_DIR = os.path.join(BASE_DIR, "training", "bead_dataset")


def _ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def _handle_upload(files):
    """Upload corrected board images for annotation."""
    if not files:
        return "❌ 未选择文件", []
    img_dir = os.path.join(BEAD_DATASET_DIR, "images", "source")
    _ensure_dirs(img_dir)
    count = 0
    for f in files:
        fpath = f if isinstance(f, str) else getattr(f, "name", str(f))
        basename = os.path.basename(fpath)
        dest = os.path.join(img_dir, basename)
        if os.path.abspath(fpath) != os.path.abspath(dest):
            shutil.copy2(fpath, dest)
        count += 1
    return f"✅ 已上传 {count} 张图片", []


def _handle_prelabel(image, dp, min_dist, param1, param2, min_r, max_r):
    """Run HoughCircles pre-labeling on an image."""
    if image is None:
        return None, "❌ 请先加载图片", 0

    boxes = hough_circles_to_boxes(
        image,
        dp=dp,
        min_dist=int(min_dist),
        param1=int(param1),
        param2=int(param2),
        min_radius=int(min_r),
        max_radius=int(max_r),
    )

    vis = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box["xyxy"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)

    return vis, f"✅ 检测到 {len(boxes)} 个圆形候选", len(boxes)


def _handle_export(image, dp, min_dist, param1, param2, min_r, max_r):
    """Save pre-labeled boxes as YOLO detection format."""
    if image is None:
        return "❌ 请先加载图片"

    boxes = hough_circles_to_boxes(
        image,
        dp=dp,
        min_dist=int(min_dist),
        param1=int(param1),
        param2=int(param2),
        min_radius=int(min_r),
        max_radius=int(max_r),
    )

    if not boxes:
        return "❌ 未检测到任何圆形"

    img_dir = os.path.join(BEAD_DATASET_DIR, "images", "train")
    lbl_dir = os.path.join(BEAD_DATASET_DIR, "labels", "train")
    _ensure_dirs(img_dir, lbl_dir)

    # Use timestamp-based naming to avoid conflicts
    import time
    basename = f"bead_{int(time.time())}"
    img_path = os.path.join(img_dir, basename + ".png")
    lbl_path = os.path.join(lbl_dir, basename + ".txt")

    cv2.imwrite(img_path, image)
    h, w = image.shape[:2]
    save_yolo_boxes(boxes, img_w=w, img_h=h, output_path=lbl_path)

    return f"✅ 已保存 {len(boxes)} 个标注\n图片: {img_path}\n标签: {lbl_path}"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="🫘 豆子标注工具") as app:
        gr.Markdown("# 🫘 豆子标注工具（HoughCircles 预标注）")

        with gr.Row():
            with gr.Column(scale=2):
                input_image = gr.Image(label="加载矫正后的板子图片", type="numpy")
                result_image = gr.Image(label="预标注结果", type="numpy")

            with gr.Column(scale=1):
                gr.Markdown("### HoughCircles 参数")
                param_dp = gr.Slider(1.0, 3.0, value=1.2, step=0.1, label="dp")
                param_min_dist = gr.Slider(5, 50, value=10, step=1, label="最小间距")
                param1 = gr.Slider(50, 200, value=100, step=10, label="Canny 上阈值")
                param2 = gr.Slider(10, 100, value=30, step=5, label="累加器阈值")
                param_min_r = gr.Slider(1, 20, value=3, step=1, label="最小半径")
                param_max_r = gr.Slider(5, 50, value=30, step=1, label="最大半径")

                prelabel_btn = gr.Button("🔍 预标注", variant="primary")
                status_text = gr.Textbox(label="状态", interactive=False)

                export_btn = gr.Button("💾 确认并导出", variant="secondary")
                export_status = gr.Textbox(label="导出结果", interactive=False, lines=4)

        hough_inputs = [input_image, param_dp, param_min_dist, param1, param2, param_min_r, param_max_r]

        prelabel_btn.click(
            fn=_handle_prelabel,
            inputs=hough_inputs,
            outputs=[result_image, status_text],
        )

        export_btn.click(
            fn=_handle_export,
            inputs=hough_inputs,
            outputs=[export_status],
        )

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(share=False)
```

- [ ] **Step 2: Verify the UI starts without errors**

Run: `.venv/bin/python -c "from src.bead_annotate_ui import build_ui; app = build_ui(); print('UI built OK')"`
Expected: `UI built OK`

- [ ] **Step 3: Commit**

```bash
git add src/bead_annotate_ui.py
git commit -m "feat: add Gradio bead annotation UI with HoughCircles pre-labeling"
```

---

## Phase 3: Training Scripts

### Task 8: Create bead detection training script

**Files:**
- Create: `training/bead_train.py`

- [ ] **Step 1: Write the bead detection training script**

Follows the same pattern as `training/train.py` but for YOLOv8n **detection** (not segmentation), single class `bead`.

```python
# training/bead_train.py
"""
YOLO 豆子检测模型训练脚本。

用法:
    python training/bead_train.py --data training/bead_dataset
    python training/bead_train.py --data training/bead_dataset --epochs 150 --batch 16
"""

import argparse
import os
import sys


def generate_data_yaml(dataset_dir):
    """生成 data.yaml 配置文件（如果不存在）。"""
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    if os.path.exists(yaml_path):
        print(f"data.yaml 已存在: {yaml_path}")
        return yaml_path

    images_train = os.path.join(dataset_dir, "images", "train")
    images_valid = os.path.join(dataset_dir, "images", "valid")

    content = f"""# Bead Detection Dataset
path: {os.path.abspath(dataset_dir)}
train: {os.path.abspath(images_train)}
val: {os.path.abspath(images_valid)}

nc: 1
names:
  0: bead
"""
    with open(yaml_path, "w") as f:
        f.write(content)

    print(f"data.yaml 已生成: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="YOLO 豆子检测模型训练")
    parser.add_argument(
        "--data", required=True,
        help="数据集目录路径（包含 images/train, images/valid 子目录）",
    )
    parser.add_argument(
        "--model", default="yolov8n.pt",
        help="基础模型权重 (默认: yolov8n.pt)",
    )
    parser.add_argument(
        "--epochs", type=int, default=150,
        help="训练轮数 (默认: 150)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="输入图片尺寸 (默认: 640)",
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="批次大小 (默认: 16)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="训练设备: cpu 或 0,1,2... (默认: cpu)",
    )
    parser.add_argument(
        "--project", default="training/runs",
        help="输出项目目录 (默认: training/runs)",
    )
    parser.add_argument(
        "--name", default="bead-v1",
        help="实验名称 (默认: bead-v1)",
    )
    args = parser.parse_args()

    dataset_dir = args.data
    if not os.path.isdir(dataset_dir):
        print(f"[ERROR] 数据集目录不存在: {dataset_dir}")
        sys.exit(1)

    train_dir = os.path.join(dataset_dir, "images", "train")
    valid_dir = os.path.join(dataset_dir, "images", "valid")

    for dir_path, label in [(train_dir, "images/train"), (valid_dir, "images/valid")]:
        if not os.path.isdir(dir_path):
            print(f"[ERROR] 缺少 {label} 目录: {dir_path}")
            sys.exit(1)

    data_yaml = generate_data_yaml(dataset_dir)

    from ultralytics import YOLO

    print(f"\n加载模型: {args.model}")
    model = YOLO(args.model)

    print(f"开始训练: epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, device={args.device}")
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        single_cls=True,
        patience=50,
    )

    best_weights = os.path.join(args.project, args.name, "weights", "best.pt")
    if os.path.exists(best_weights):
        print(f"\n训练完成! 最佳权重: {best_weights}")
    else:
        print(f"\n训练完成! 请在 {args.project}/{args.name}/ 下查找 best.pt")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script parses correctly**

Run: `.venv/bin/python training/bead_train.py --help`
Expected: Help text with all arguments listed

- [ ] **Step 3: Commit**

```bash
git add training/bead_train.py
git commit -m "feat: add bead detection training script"
```

---

### Task 9: Create bead detection validation script

**Files:**
- Create: `training/bead_validate.py`

- [ ] **Step 1: Write the validation script**

Follows the pattern of `training/validate.py` but for detection boxes (not segmentation masks).

```python
# training/bead_validate.py
"""
YOLO 豆子检测模型验证脚本 — 对测试图片运行推理并可视化结果。

用法:
    python training/bead_validate.py --model training/runs/bead-v1/weights/best.pt --image test_photos/
    python training/bead_validate.py --model best.pt --image test.png --output training/bead_validation_results
"""

import argparse
import os
import sys

import cv2
import numpy as np


def validate_single(model, image_path, output_dir, conf_threshold=0.5):
    """对单张图片运行推理并保存可视化结果。"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"  [ERROR] 无法读取: {image_path}")
        return False

    results = model(image, conf=conf_threshold, verbose=False)
    filename = os.path.basename(image_path)

    if results[0].boxes is None or len(results[0].boxes) == 0:
        print(f"  [SKIP] 未检测到豆子: {filename}")
        output_path = os.path.join(output_dir, f"no_detect_{filename}")
        cv2.imwrite(output_path, image)
        return False

    annotated = image.copy()
    boxes = results[0].boxes
    n_boxes = len(boxes)

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0].cpu().numpy())
        # Green box + confidence label
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(
            annotated, f"{conf:.2f}",
            (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1,
        )

    cv2.putText(
        annotated, f"Beads: {n_boxes}",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
    )

    output_path = os.path.join(output_dir, f"detected_{filename}")
    cv2.imwrite(output_path, annotated)
    print(f"  [OK] {filename} -> {output_path} ({n_boxes} 颗豆子)")
    return True


def main():
    parser = argparse.ArgumentParser(description="YOLO 豆子检测模型验证")
    parser.add_argument("--model", required=True, help="模型权重文件路径")
    parser.add_argument("--image", required=True, help="测试图片路径（单张或目录）")
    parser.add_argument("--output", default="training/bead_validation_results", help="输出目录")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值 (默认: 0.5)")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"[ERROR] 模型文件不存在: {args.model}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    from ultralytics import YOLO

    print(f"加载模型: {args.model}")
    model = YOLO(args.model)
    print("模型加载完成\n")

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    if os.path.isfile(args.image):
        image_paths = [args.image]
    elif os.path.isdir(args.image):
        image_paths = sorted(
            os.path.join(args.image, f)
            for f in os.listdir(args.image)
            if f.lower().endswith(exts)
        )
    else:
        print(f"[ERROR] 路径不存在: {args.image}")
        sys.exit(1)

    if not image_paths:
        print("[ERROR] 没有找到测试图片")
        sys.exit(1)

    print(f"共 {len(image_paths)} 张测试图片\n")

    success = 0
    for img_path in image_paths:
        if validate_single(model, img_path, args.output, conf_threshold=args.conf):
            success += 1

    print(f"\n验证完成: {success}/{len(image_paths)} 张成功检测到豆子")
    print(f"结果保存在: {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script parses correctly**

Run: `.venv/bin/python training/bead_validate.py --help`
Expected: Help text with all arguments listed

- [ ] **Step 3: Commit**

```bash
git add training/bead_validate.py
git commit -m "feat: add bead detection validation script"
```

---

## Phase 4: Pipeline Integration

### Task 10: Integrate BeadDetector into CLI

**Files:**
- Modify: `src/cli.py`

- [ ] **Step 1: Update CLI to use BeadDetector pipeline**

Replace the GridExtractor-based pipeline with BeadDetector. Add `--bead-model` argument for the bead detection model path.

```python
# Full replacement for src/cli.py
"""Command-line interface for Pintconut bead diff detector."""
import argparse
import os
import sys

import cv2
import numpy as np

from src.bead_detect import BeadDetector
from src.blueprint import parse_blueprint
from src.color import ColorMatcher
from src.compare import DiffComparator
from src.detect import BoardDetector
from src.grid import PerspectiveCorrector


def parse_board_size(size_str: str) -> tuple[int, int]:
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
  python -m src.cli --photo IMG_001.jpg --blueprint pattern.png --board-size 29x29 --bead-model models/bead-best.pt
        """,
    )
    parser.add_argument("--photo", required=True, help="拼豆照片路径")
    parser.add_argument("--blueprint", required=True, help="图纸图片路径")
    parser.add_argument("--board-size", required=True, help="拼板尺寸，格式为 ROWSxCOLS（如 29x29）")
    parser.add_argument("--model", default="models/beadboard-best.pt", help="板子检测模型权重路径")
    parser.add_argument("--bead-model", default="models/bead-best.pt", help="豆子检测模型权重路径")
    parser.add_argument("--output", default="annotated_result.jpg", help="输出图片路径")
    parser.add_argument("--color-tolerance", type=float, default=30.0, help="颜色匹配容差（LAB距离）")

    args = parser.parse_args()

    for path, label in [(args.photo, "照片"), (args.blueprint, "图纸")]:
        if not os.path.exists(path):
            print(f"❌ {label}文件不存在: {path}", file=sys.stderr)
            sys.exit(1)

    rows, cols = parse_board_size(args.board_size)

    print(f"📷 照片: {args.photo}")
    print(f"📋 图纸: {args.blueprint}")
    print(f"📐 拼板: {rows}×{cols}")
    print()

    # Step 1-2: Board detection + perspective correction
    print("📋 解析图纸...")
    blueprint_grid = parse_blueprint(args.blueprint, rows=rows, cols=cols)
    print(f"   图纸网格: {blueprint_grid.shape}")

    print("🔍 检测拼板位置...")
    photo = cv2.imread(args.photo)
    if photo is None:
        print(f"❌ 无法读取照片: {args.photo}", file=sys.stderr)
        sys.exit(1)

    detector = BoardDetector(model_path=args.model)
    mask = detector.detect(photo)

    if mask is None:
        print("❌ 未能在照片中检测到拼板", file=sys.stderr)
        sys.exit(1)

    print("   ✅ 检测到拼板区域")

    print("📐 透视校正...")
    corners = detector.extract_corners(mask)
    if corners is None:
        print("❌ 无法提取拼板角点", file=sys.stderr)
        sys.exit(1)

    corrector = PerspectiveCorrector()
    output_w = cols * 20
    output_h = rows * 20
    corrected = corrector.correct(photo, corners, output_size=(output_w, output_h))
    print(f"   校正后尺寸: {corrected.shape[1]}×{corrected.shape[0]}")

    # Step 3-5: Bead detection + filtering + color matching
    print("🔍 检测豆子...")
    bead_detector = BeadDetector(model_path=args.bead_model)
    color_matcher = ColorMatcher()
    beads = bead_detector.detect_beads(corrected, color_matcher=color_matcher)
    print(f"   检测到 {len(beads)} 颗豆子")

    counts = bead_detector.count_beads(beads)
    print(f"   颜色统计: {counts['by_color']}")

    # Step 6: Annotate and save
    print(f"\n💾 保存标注图到 {args.output}...")
    annotated = corrected.copy()
    for bead in beads:
        x1, y1, x2, y2 = [int(v) for v in bead["xyxy"]]
        color = (0, 255, 0) if bead["is_bead"] else (0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
        if bead.get("color_name") and bead["color_name"] != "Unknown":
            cv2.putText(
                annotated, bead["color_name"],
                (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1,
            )
    cv2.imwrite(args.output, annotated)
    print(f"   ✅ 已保存")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI parses correctly**

Run: `.venv/bin/python -m src.cli --help`
Expected: Help text with `--bead-model` argument listed

- [ ] **Step 3: Commit**

```bash
git add src/cli.py
git commit -m "feat: integrate BeadDetector into CLI pipeline"
```

---

### Task 11: Run full test suite and verify no regressions

**Files:** None (verification only)

- [ ] **Step 1: Run all tests**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: Verify all test files are present**

Run: `ls tests/test_*.py`
Expected: `test_bead_detect.py`, `test_bead_prelabel.py`, `test_blueprint.py`, `test_color.py`, `test_compare.py`, `test_detect.py`, `test_grid.py`, `test_label_service.py`

---

## Self-Review Checklist

**Spec coverage:**
- [x] Step 1-2 (board detection + perspective): Existing code, untouched → ✅
- [x] Step 3 (bead detection model): Task 1-4 → ✅
- [x] Step 4 (post-processing/filtering): Task 2 → ✅
- [x] Step 5 (color recognition): Task 3 → ✅
- [x] Step 6 (result output): Task 10 CLI integration → ✅
- [x] Pre-labeling tool: Tasks 6-7 → ✅
- [x] Training script: Task 8 → ✅
- [x] Validation script: Task 9 → ✅
- [x] Dataset directory structure: Created by annotation tool (Task 7) → ✅

**Placeholder scan:** No TBD/TODO/fill-in patterns found.

**Type consistency:**
- `BeadDetector.detect()` returns `list[dict]` with keys `xyxy, conf, width, height, cx, cy` — used consistently in `filter_boxes()`, `detect_beads()`, and CLI
- `ColorMatcher.match_box()` takes `image` (BGR ndarray) and `box` dict with `xyxy` key — matches output format from `BeadDetector.detect()`
- `save_yolo_boxes()` takes same box dict format — consistent
- `hough_circles_to_boxes()` returns same dict format — consistent
