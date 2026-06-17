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


def label_projective(beads, affine_labels):
    """Refit labels via a single over-constrained homography (lattice -> image), no RANSAC.
    Returns (labels, reprojection_residual_px)."""
    pts = np.array([b.xy for b in beads], dtype=np.float64)
    lat = np.array(affine_labels, dtype=np.float64)             # (N, 2) = (row, col)

    # Deduplicate: keep the first bead at each lattice point
    unique_labels: dict[tuple[int, int], int] = {}
    for i, lab in enumerate(affine_labels):
        key = (int(lab[0]), int(lab[1]))
        if key not in unique_labels:
            unique_labels[key] = i
    idx = sorted(unique_labels.values())
    lat_u = lat[idx].astype(np.float32)
    pts_u = pts[idx].astype(np.float32)

    H, _ = cv2.findHomography(lat_u, pts_u)                     # lattice -> image, least squares
    if H is None:
        return affine_labels, float("inf")
    Hinv = np.linalg.inv(H)

    # Inverse-project each bead to lattice coords
    ones = np.ones((len(pts), 1), dtype=np.float64)
    img_h = np.hstack([pts, ones])
    lat2 = (Hinv @ img_h.T).T
    lat2 = lat2[:, :2] / lat2[:, 2:3]

    # Greedy nearest-lattice-point assignment (one pass, sorted by confidence)
    rounded = np.round(lat2).astype(int)
    used: set[tuple[int, int]] = set()
    order = np.argsort(np.linalg.norm(lat2 - rounded, axis=1))  # most confident first
    labels = [None] * len(beads)
    for i in order:
        r0, c0 = int(rounded[i, 0]), int(rounded[i, 1])
        if (r0, c0) not in used:
            used.add((r0, c0))
            labels[i] = (r0, c0)
        else:
            # Search nearby integer lattice points for the closest free one
            best_dist = float("inf")
            best_label = None
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    cand = (r0 + dr, c0 + dc)
                    if cand not in used:
                        d = np.linalg.norm(lat2[i] - np.array(cand, dtype=np.float64))
                        if d < best_dist:
                            best_dist = d
                            best_label = cand
            if best_label is not None:
                used.add(best_label)
                labels[i] = best_label

    # Second pass: assign any remaining beads to nearest free lattice point
    # (handles cases where homography bias shifts beads outside the ±3 window)
    r_min = min(r for r, _ in used) - 2
    r_max = max(r for r, _ in used) + 2
    c_min = min(c for _, c in used) - 2
    c_max = max(c for _, c in used) + 2
    candidates = {(r, c) for r in range(r_min, r_max + 1) for c in range(c_min, c_max + 1)}
    for i in order:
        if labels[i] is not None:
            continue
        free = candidates - used
        if not free:
            break
        best_dist = float("inf")
        best_label = None
        for cand in free:
            d = np.linalg.norm(lat2[i] - np.array(cand, dtype=np.float64))
            if d < best_dist:
                best_dist = d
                best_label = cand
        if best_label is not None:
            used.add(best_label)
            labels[i] = best_label

    # Reprojection residual in pixels (over deduplicated set)
    ones_u = np.ones((len(idx), 1), dtype=np.float64)
    lat_uh = np.hstack([lat_u.astype(np.float64), ones_u])
    pred = (H @ lat_uh.T).T
    pred = pred[:, :2] / pred[:, 2:3]
    res_px = float(np.mean(np.linalg.norm(pred - pts_u.astype(np.float64), axis=1)))
    return labels, res_px
