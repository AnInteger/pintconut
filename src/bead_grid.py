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
    median_radius: float = 0.0


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
    """Detect beads via HoughCircles (dark ring) + median-size filter + colour sample.

    Radius range derived from image size (assuming board fills most of frame).
    min_dist set to ~1.2x max radius to enforce minimum bead spacing.
    Median-size filter (0.6x–1.5x) rejects outlier detections.
    param2=25 (slightly more sensitive than default 30) for better recall on photos.
    """
    h, w = image.shape[:2]
    max_r = max(4, int(min(h, w) / 20))
    min_r = max(2, max_r // 3)
    min_dist = max(8, int(max_r * 1.2))
    boxes = hough_circles_to_boxes(
        image, min_radius=min_r, max_radius=max_r,
        min_dist=min_dist, param2=25,
    )
    if not boxes:
        return []
    radii = np.array([b["width"] / 2.0 for b in boxes])
    med = float(np.median(radii))
    beads: list[Bead] = []
    for b, r in zip(boxes, radii):
        if not (med * 0.6 <= r <= med * 1.5):
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
    Dedups colliding affine correspondences before fitting, then assigns each bead a unique
    lattice cell (greedy by reprojection confidence, nearest-free-cell fallback).
    Returns (labels, reprojection_residual_px)."""
    pts = np.array([b.xy for b in beads], dtype=np.float64)
    lat = np.array(affine_labels, dtype=np.float64)

    # Dedup: one bead per lattice cell (first encountered)
    seen: dict[tuple[int, int], int] = {}
    for i, (r, c) in enumerate(affine_labels):
        seen.setdefault((int(r), int(c)), i)
    idx = sorted(seen.values())
    lat_u, pts_u = lat[idx].astype(np.float32), pts[idx].astype(np.float32)

    H, _ = cv2.findHomography(lat_u, pts_u)
    if H is None:
        return list(affine_labels), float("inf")
    Hinv = np.linalg.inv(H)
    ones = np.ones((len(pts), 1), dtype=np.float64)
    lat2 = (Hinv @ np.hstack([pts, ones]).T).T
    lat2 = lat2[:, :2] / lat2[:, 2:3]
    rounded = np.round(lat2).astype(int)

    # Greedy by confidence: take rounded cell if free, else nearest free cell in bbox.
    rs = np.arange(int(rounded[:, 0].min()) - 2, int(rounded[:, 0].max()) + 3)
    cs = np.arange(int(rounded[:, 1].min()) - 2, int(rounded[:, 1].max()) + 3)
    free = {(int(r), int(c)) for r in rs for c in cs}
    order = np.argsort(np.linalg.norm(lat2 - rounded, axis=1))
    labels: list = [None] * len(beads)
    for i in order:
        cand = (int(rounded[i, 0]), int(rounded[i, 1]))
        if cand not in free:
            if free:
                cand = min(free, key=lambda lc: np.linalg.norm(lat2[i] - np.array(lc)))
            # else: free exhausted — keep the duplicate rounded cell rather than crash
        free.discard(cand)
        labels[i] = cand

    lat_uh = np.hstack([lat_u.astype(np.float64), np.ones((len(idx), 1))])
    pred = (H @ lat_uh.T).T
    pred = pred[:, :2] / pred[:, 2:3]
    res_px = float(np.mean(np.linalg.norm(pred - pts_u, axis=1)))
    return labels, res_px


def label_beads(beads, d_row, d_col, spacing):
    """Two-tier labeling: affine by default, escalate to projective when residual is high.
    Returns (labels, perspective_tier).

    Perspective upgrade threshold set to spacing * 0.4 (pixels) — more forgiving than
    the original 0.3x to correctly flag tilted boards in real photos where the affine fit
    is only mildly inaccurate.
    """
    labels, frac = label_affine(beads, d_row, d_col, spacing)
    if frac < AFFINE_OK_THRESHOLD:
        return labels, False
    labels2, res_px = label_projective(beads, labels)
    if res_px < spacing * 0.4:
        return labels2, True
    return labels, False


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
                                  is_edge=is_edge, confidence=conf, image_xy=image_xy,
                                  has_bead=((r, c) in cell_bead)))
    return cells


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


class BeadGridFitter:
    """Detects the bead grid directly from beads (no board model)."""

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
