"""Tests for diff comparison and annotation module."""
import numpy as np
from src.compare import DiffComparator, DiffResult
from src.grid import CellInfo


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


def _cell(r, c, rgb, xy):
    return CellInfo(row=r, col=c, color=np.array(rgb, dtype=np.uint8),
                    is_visible=True, is_edge=False, confidence=1.0, image_xy=xy)


def test_compare_carries_image_xy():
    cells = [_cell(0, 0, [255, 0, 0], (30.0, 40.0))]
    bp = np.zeros((1, 1, 3), dtype=np.uint8)  # blueprint black
    diffs = DiffComparator(color_tolerance=10.0).compare_with_confidence(cells, bp)
    assert len(diffs) == 1
    assert diffs[0].image_xy == (30.0, 40.0)


def test_annotate_draws_at_image_xy():
    photo = np.full((100, 100, 3), 200, dtype=np.uint8)
    diffs = [DiffResult(row=0, col=0, type="color_mismatch",
                        photo_color=[255, 0, 0], blueprint_color=[0, 0, 0],
                        cell_confidence=1.0, is_reliable=True, image_xy=(50.0, 60.0))]
    out = DiffComparator().annotate_with_confidence(photo, diffs, rows=4, cols=4)
    region = out[55:65, 45:55]
    assert np.any(np.all(region == (0, 0, 255), axis=-1))   # red mark near (50,60)
