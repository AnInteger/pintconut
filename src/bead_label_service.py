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
