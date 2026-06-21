"""
Grid extraction module.

Handles perspective correction of the detected board region
and extraction of per-cell color values.
"""
from __future__ import annotations

from dataclasses import dataclass

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
            image: Input image (HxWx3 BGR or RGB array).
            corners: 4x2 array of corner points in order:
                     [top-left, top-right, bottom-right, bottom-left].
            output_size: (width, height) of the output corrected image.

        Returns:
            Perspective-corrected image of size output_size.
        """
        dst = np.array([
            [0, 0],
            [output_size[0], 0],
            [output_size[0], output_size[1]],
            [0, output_size[1]],
        ], dtype=np.float32)

        src = corners.astype(np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst)
        corrected = cv2.warpPerspective(
            image, matrix, output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return corrected

    def correct_with_matrix(
        self,
        image: np.ndarray,
        corners: np.ndarray,
        output_size: tuple[int, int] = (400, 400),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perspective correction that also returns the transform matrix.

        The matrix can be used to warp the visibility mask into corrected
        coordinates via ``cv2.warpPerspective``.

        Args:
            image: Input image (HxWx3 BGR or RGB array).
            corners: 4x2 array of corner points in order:
                     [top-left, top-right, bottom-right, bottom-left].
            output_size: (width, height) of the output corrected image.

        Returns:
            (corrected_image, perspective_transform_matrix)
        """
        dst = np.array([
            [0, 0],
            [output_size[0], 0],
            [output_size[0], output_size[1]],
            [0, output_size[1]],
        ], dtype=np.float32)

        src = corners.astype(np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst)
        corrected = cv2.warpPerspective(
            image, matrix, output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return corrected, matrix


# ---------------------------------------------------------------------------
# Cell-level metadata for visibility-aware extraction
# ---------------------------------------------------------------------------

@dataclass
class CellInfo:
    """Metadata for a single grid cell."""
    row: int
    col: int
    color: np.ndarray          # RGB uint8
    is_visible: bool           # Cell centre is inside the visible region
    is_edge: bool              # Cell is near the truncation boundary
    confidence: float          # Comparison confidence for this cell [0, 1]
    image_xy: tuple[float, float] | None = None   # cell centre in original image
    has_bead: bool = False     # True if a detected bead landed in this cell


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
            board_image: Perspective-corrected board image.
            rows: Number of bead rows.
            cols: Number of bead columns.
            sample_fraction: Fraction of cell to sample from center (0-1).

        Returns:
            numpy array of shape (rows, cols, 3) with RGB uint8 values.
        """
        h, w = board_image.shape[:2]
        cell_h = h / rows
        cell_w = w / cols

        result = np.zeros((rows, cols, 3), dtype=np.uint8)

        for r in range(rows):
            for c in range(cols):
                margin_y = cell_h * (1 - sample_fraction) / 2
                margin_x = cell_w * (1 - sample_fraction) / 2

                y_start = int(r * cell_h + margin_y)
                y_end = int((r + 1) * cell_h - margin_y)
                x_start = int(c * cell_w + margin_x)
                x_end = int((c + 1) * cell_w - margin_x)

                y_start = max(0, y_start)
                y_end = min(h, y_end)
                x_start = max(0, x_start)
                x_end = min(w, x_end)

                if y_end <= y_start or x_end <= x_start:
                    result[r, c] = [0, 0, 0]
                    continue

                cell = board_image[y_start:y_end, x_start:x_end]
                median_color = np.median(cell.reshape(-1, 3), axis=0)
                result[r, c] = median_color.astype(np.uint8)

        return result

    def extract_with_visibility(
        self,
        board_image: np.ndarray,
        rows: int,
        cols: int,
        visibility_mask: np.ndarray,
        edge_margin: int = 2,
    ) -> list[CellInfo]:
        """Visibility-aware grid extraction.

        Parameters
        ----------
        board_image : H×W×3 BGR
            Perspective-corrected board image.
        rows, cols : int
            Grid dimensions.
        visibility_mask : H×W bool
            **Must already be in corrected-image coordinates**
            (apply ``cv2.warpPerspective`` to the original mask using the
            same transform matrix as the image).
        edge_margin : int
            Number of grid-cells from an invisible neighbour to mark as
            "edge" (low confidence).

        Returns
        -------
        list[CellInfo]
        """
        h, w = board_image.shape[:2]
        assert visibility_mask.shape[:2] == (h, w), (
            f"visibility_mask shape {visibility_mask.shape[:2]} != "
            f"board_image shape {(h, w)} — did you forget to warpPerspective?"
        )

        cell_h = h / rows
        cell_w = w / cols
        cells: list[CellInfo] = []

        for r in range(rows):
            for c in range(cols):
                cy = int((r + 0.5) * cell_h)
                cx = int((c + 0.5) * cell_w)

                is_visible = (0 <= cy < h and 0 <= cx < w and visibility_mask[cy, cx])
                is_edge = self._is_near_boundary(
                    r, c, rows, cols, visibility_mask, cell_h, cell_w, edge_margin,
                )

                if is_visible:
                    my = cell_h * 0.3
                    mx = cell_w * 0.3
                    y1 = max(0, int(r * cell_h + my))
                    y2 = min(h, int((r + 1) * cell_h - my))
                    x1 = max(0, int(c * cell_w + mx))
                    x2 = min(w, int((c + 1) * cell_w - mx))
                    if y2 > y1 and x2 > x1:
                        region = board_image[y1:y2, x1:x2]
                        color = np.median(region.reshape(-1, 3), axis=0).astype(np.uint8)
                    else:
                        color = np.array([0, 0, 0], dtype=np.uint8)
                else:
                    color = np.array([0, 0, 0], dtype=np.uint8)

                confidence = 0.5 if is_edge else (1.0 if is_visible else 0.0)
                cells.append(CellInfo(r, c, color, is_visible, is_edge, confidence))

        return cells

    @staticmethod
    def _is_near_boundary(
        r: int, c: int,
        rows: int, cols: int,
        vis_mask: np.ndarray,
        cell_h: float, cell_w: float,
        margin: int,
    ) -> bool:
        h, w = vis_mask.shape[:2]
        for dr in range(-margin, margin + 1):
            for dc in range(-margin, margin + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    cy = int((nr + 0.5) * cell_h)
                    cx = int((nc + 0.5) * cell_w)
                    if 0 <= cy < h and 0 <= cx < w and not vis_mask[cy, cx]:
                        return True
        return False
