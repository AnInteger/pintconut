"""Bead-grid-driven board detection (replaces YOLOv8-seg + edge_refiner)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import cv2

from .bead_prelabel import hough_circles_to_boxes


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
