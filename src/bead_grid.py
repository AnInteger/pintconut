"""Bead-grid-driven board detection (replaces YOLOv8-seg + edge_refiner)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import cv2

from .bead_prelabel import hough_circles_to_boxes
from .grid import CellInfo


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
