"""Diff comparison and annotation module."""
import cv2
import numpy as np


class DiffComparator:
    """Compares photo and blueprint grids and annotates differences."""

    def __init__(self, color_tolerance: float = 30.0):
        self.color_tolerance = color_tolerance

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

    @staticmethod
    def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
        pixel = np.array([[rgb.astype(np.uint8)]], dtype=np.uint8)
        lab = cv2.cvtColor(pixel, cv2.COLOR_RGB2Lab)
        return lab.flatten().astype(np.float64)
