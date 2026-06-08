"""
Grid extraction module.

Handles perspective correction of the detected board region
and extraction of per-cell color values.
"""
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
