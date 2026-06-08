"""Tests for diff comparison and annotation module."""
import numpy as np
from src.compare import DiffComparator


def test_compare_identical_grids():
    grid = np.zeros((3, 3, 3), dtype=np.uint8)
    grid[0, 0] = [255, 0, 0]
    comparator = DiffComparator()
    diffs = comparator.compare(grid, grid)
    assert len(diffs) == 0


def test_compare_detects_color_mismatch():
    photo_grid = np.full((3, 3, 3), [255, 0, 0], dtype=np.uint8)
    blueprint_grid = np.full((3, 3, 3), [0, 255, 0], dtype=np.uint8)
    comparator = DiffComparator()
    diffs = comparator.compare(photo_grid, blueprint_grid)
    assert len(diffs) == 9


def test_compare_detects_single_mismatch():
    photo_grid = np.full((3, 3, 3), [255, 0, 0], dtype=np.uint8)
    blueprint_grid = np.full((3, 3, 3), [255, 0, 0], dtype=np.uint8)
    blueprint_grid[1, 1] = [0, 255, 0]
    comparator = DiffComparator()
    diffs = comparator.compare(photo_grid, blueprint_grid)
    assert len(diffs) == 1
    assert diffs[0]["row"] == 1
    assert diffs[0]["col"] == 1


def test_compare_ignores_similar_colors():
    photo_grid = np.full((3, 3, 3), [250, 5, 5], dtype=np.uint8)
    blueprint_grid = np.full((3, 3, 3), [255, 0, 0], dtype=np.uint8)
    comparator = DiffComparator(color_tolerance=20.0)
    diffs = comparator.compare(photo_grid, blueprint_grid)
    assert len(diffs) == 0


def test_annotate_creates_output_image():
    photo = np.full((200, 200, 3), [200, 200, 200], dtype=np.uint8)
    diffs = [
        {"row": 1, "col": 1, "type": "color_mismatch"},
        {"row": 2, "col": 2, "type": "color_mismatch"},
    ]
    comparator = DiffComparator()
    result = comparator.annotate(photo, diffs, rows=5, cols=5)
    assert result.shape == photo.shape
    assert result.dtype == np.uint8
    assert not np.array_equal(result, photo)
