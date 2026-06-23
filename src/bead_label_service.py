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


def generate_grid_boxes(corners, rows: int, cols: int) -> list[dict]:
    """4 corner-bead image positions (TL,TR,BR,BL) + dims -> all bead boxes via homography.

    Each box is sized to its local spacing (mean distance to existing neighbors) * 0.4,
    so box size adapts under perspective. Raises cv2.error if the 4 corners are degenerate.
    """
    corners = np.asarray(corners, dtype=np.float32)
    src = np.array([[0, 0], [cols - 1, 0], [cols - 1, rows - 1], [0, rows - 1]],
                   dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, corners)  # maps grid (c,r) -> image (x,y)
    centers = {}
    for r in range(rows):
        for c in range(cols):
            v = H @ np.array([c, r, 1.0])
            centers[(r, c)] = v[:2] / v[2]
    boxes = []
    for r in range(rows):
        for c in range(cols):
            xy = centers[(r, c)]
            dists = []
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nb = centers.get((r + dr, c + dc))
                if nb is not None:
                    dists.append(float(np.linalg.norm(nb - xy)))
            spacing = float(np.mean(dists)) if dists else 10.0
            boxes.append(_box_from_xy(xy, spacing * 0.4, "generated"))
    return boxes


def preview_box_colors(image, boxes, color_matcher) -> list[dict]:
    """Sample each box center's color and match to palette (for eval-preview overlay)."""
    from .bead_grid import _sample_color
    out = []
    for b in boxes:
        color = _sample_color(image, (b["cx"], b["cy"]))
        m = color_matcher.match(color.tolist())
        out.append({"xy": (float(b["cx"]), float(b["cy"])),
                    "name": m["name"], "rgb": m["rgb"]})
    return out


def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude of a BGR image (single-channel)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _ring_profile(sample: np.ndarray, cx, cy, r_min=3, r_max=120, n_ang=120):
    """Mean of `sample` along each circle of radius r (radial profile)."""
    H, W = sample.shape
    ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    cos, sin = np.cos(ang), np.sin(ang)
    prof = []
    for r in range(r_min, r_max + 1):
        xs = np.round(cx + r * cos).astype(int)
        ys = np.round(cy + r * sin).astype(int)
        ok = (xs >= 0) & (ys >= 0) & (xs < W) & (ys < H)
        prof.append(float(sample[ys[ok], xs[ok]].mean()) if ok.sum() > n_ang * 0.6 else 0.0)
    return np.array(prof), list(range(r_min, r_max + 1))


def find_bead_radius(gmag: np.ndarray, cx, cy, prior_radii=None,
                     r_min=3, r_max=120) -> tuple[int, bool]:
    """Edge radius from a center click. Returns (radius, warn).

    Algorithm (validated on user ground truth gt_NYQC4978.txt, 11/12 within 6px):
      1. grad_outer: outermost radial-gradient ring above 0.6*peak (skips highlights).
      2. clamp to [0.8, 1.2]*median(prior_radii) once >=3 priors exist (kills ballooning).
      3. warn=True when the pre-clamp candidate hit floor/ceiling AND no clamp ran.
    """
    prof, rs = _ring_profile(gmag, cx, cy, r_min, r_max)
    peak = float(prof.max()) if prof.size else 0.0
    if peak <= 0:
        return r_min, True
    thr = 0.6 * peak
    outer = r_min
    for k in range(1, len(prof) - 1):
        if prof[k] > thr and prof[k] >= prof[k - 1] and prof[k] >= prof[k + 1]:
            outer = rs[k]          # keep updating -> outermost local max above thr
    candidate = outer if outer > r_min else rs[int(np.argmax(prof))]

    hit_bound = candidate <= r_min + 2 or candidate >= 0.85 * r_max

    radii = list(prior_radii or [])
    if len(radii) >= 3:
        med = float(np.median(radii))
        candidate = float(np.clip(candidate, 0.8 * med, 1.2 * med))
        return int(round(candidate)), False

    return int(round(candidate)), bool(hit_bound)
