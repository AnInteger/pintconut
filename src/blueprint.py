"""
Blueprint parsing module.

Extracts a color grid from a blueprint image (PNG/JPG).
"""
import cv2
import numpy as np


def parse_blueprint(
    image_path: str,
    rows: int,
    cols: int,
    margin_fraction: float = 0.10,
) -> np.ndarray:
    """Parse a blueprint image into a rows*cols*3 RGB color grid.

    Args:
        image_path: Path to the blueprint image file.
        rows: Number of bead rows in the grid.
        cols: Number of bead columns in the grid.
        margin_fraction: Fraction of image dimension to trim as margin (0-0.5).

    Returns:
        numpy array of shape (rows, cols, 3) with RGB uint8 values.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    margin_y = int(h * margin_fraction)
    margin_x = int(w * margin_fraction)
    img_cropped = img_rgb[margin_y:h - margin_y, margin_x:w - margin_x]

    grid_h, grid_w = img_cropped.shape[:2]
    cell_h = grid_h / rows
    cell_w = grid_w / cols

    result = np.zeros((rows, cols, 3), dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            y_start = int(r * cell_h + cell_h * 0.2)
            y_end = int(r * cell_h + cell_h * 0.8)
            x_start = int(c * cell_w + cell_w * 0.2)
            x_end = int(c * cell_w + cell_w * 0.8)

            cell = img_cropped[y_start:y_end, x_start:x_end]
            median_color = np.median(cell.reshape(-1, 3), axis=0)
            result[r, c] = median_color.astype(np.uint8)

    return result
