"""
Color matching module for bead detection.

Loads the 221-color palette and provides:
- Color matching (find nearest palette color in LAB space)
- Bead presence detection (is a pixel a bead or empty board?)
"""
import json
import os

import cv2
import numpy as np


def load_color_palette(path: str = "data/colors.json") -> list[dict]:
    """Load color palette from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


class ColorMatcher:
    """Matches pixel colors to the bead palette in LAB color space."""

    def __init__(self, palette_path: str = "data/colors.json"):
        palette = load_color_palette(palette_path)
        self.palette = palette
        rgb_array = np.array([c["rgb"] for c in palette], dtype=np.uint8).reshape(-1, 1, 3)
        lab_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2Lab)
        self.palette_lab = lab_array.reshape(-1, 3).astype(np.float64)

        board_rgb = np.array([[[250, 250, 250]]], dtype=np.uint8)
        board_lab = cv2.cvtColor(board_rgb, cv2.COLOR_RGB2Lab)
        self.board_lab = board_lab.flatten().astype(np.float64)

    def match(self, rgb: list[int]) -> dict:
        """Find the nearest palette color for an RGB value in LAB space."""
        rgb_arr = np.array(rgb, dtype=np.uint8).reshape(1, 1, 3)
        lab = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2Lab).flatten().astype(np.float64)
        distances = np.sqrt(np.sum((self.palette_lab - lab) ** 2, axis=1))
        nearest_idx = np.argmin(distances)
        return self.palette[nearest_idx]

    def is_bead(self, rgb: list[int], threshold: float = 30.0) -> bool:
        """Check if a color represents a bead (vs empty board cell)."""
        rgb_arr = np.array(rgb, dtype=np.uint8).reshape(1, 1, 3)
        lab = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2Lab).flatten().astype(np.float64)
        brightness = rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114
        if brightness < 100:
            return True
        distance = np.sqrt(np.sum((lab - self.board_lab) ** 2))
        return bool(distance > threshold)

    def match_grid(self, rgb_grid: np.ndarray) -> np.ndarray:
        """Match an entire grid of RGB colors to palette entries."""
        rows, cols = rgb_grid.shape[:2]
        result = np.empty((rows, cols), dtype=object)
        for r in range(rows):
            for c in range(cols):
                result[r, c] = self.match(rgb_grid[r, c].tolist())
        return result
