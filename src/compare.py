"""Diff comparison and annotation module."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .grid import CellInfo


# ---------------------------------------------------------------------------
# Confidence-aware diff result
# ---------------------------------------------------------------------------

@dataclass
class DiffResult:
    """A single mismatch with per-cell confidence."""
    row: int
    col: int
    type: str                       # "color_mismatch"
    photo_color: list[int]
    blueprint_color: list[int]
    cell_confidence: float          # [0, 1]
    is_reliable: bool               # True when confidence ≥ 0.8
    image_xy: tuple[float, float] | None = None   # cell centre in original image


class DiffComparator:
    """Compares photo and blueprint grids and annotates differences."""

    def __init__(self, color_tolerance: float = 30.0):
        self.color_tolerance = color_tolerance

    # -- legacy interface (unchanged) ----------------------------------------

    def compare(self, photo_grid: np.ndarray, blueprint_grid: np.ndarray) -> list[dict]:
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
                photo_lab = self._rgb_to_lab(photo_rgb)
                bp_lab = self._rgb_to_lab(bp_rgb)
                distance = np.sqrt(np.sum((photo_lab - bp_lab) ** 2))
                if distance > self.color_tolerance:
                    diffs.append({
                        "row": r,
                        "col": c,
                        "type": "color_mismatch",
                        "photo_color": photo_grid[r, c].tolist(),
                        "blueprint_color": blueprint_grid[r, c].tolist(),
                    })
        return diffs

    def annotate(self, photo: np.ndarray, diffs: list[dict], rows: int, cols: int) -> np.ndarray:
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
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            size = min(int(cell_h), int(cell_w)) // 6
            cv2.line(result, (cx - size, cy - size), (cx + size, cy + size), (0, 0, 255), 2)
            cv2.line(result, (cx - size, cy + size), (cx + size, cy - size), (0, 0, 255), 2)
        return result

    # -- confidence-aware interface ------------------------------------------

    def compare_with_confidence(
        self,
        photo_cells: list[CellInfo],
        blueprint_grid: np.ndarray,
    ) -> list[DiffResult]:
        """Compare cells with per-cell confidence, skipping invisible cells."""
        diffs: list[DiffResult] = []
        for cell in photo_cells:
            if not cell.is_visible:
                continue

            r, c = cell.row, cell.col
            if r >= blueprint_grid.shape[0] or c >= blueprint_grid.shape[1]:
                continue

            photo_lab = self._rgb_to_lab(cell.color.astype(np.float32))
            bp_lab = self._rgb_to_lab(blueprint_grid[r, c].astype(np.float32))
            dist = float(np.sqrt(np.sum((photo_lab - bp_lab) ** 2)))
            if dist > self.color_tolerance:
                diffs.append(DiffResult(
                    row=r,
                    col=c,
                    type="color_mismatch",
                    photo_color=cell.color.tolist(),
                    blueprint_color=blueprint_grid[r, c].tolist(),
                    cell_confidence=cell.confidence,
                    is_reliable=cell.confidence >= 0.8,
                    image_xy=cell.image_xy,
                ))
        return diffs

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

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
        pixel = np.array([[rgb.astype(np.uint8)]], dtype=np.uint8)
        lab = cv2.cvtColor(pixel, cv2.COLOR_RGB2Lab)
        return lab.flatten().astype(np.float64)
