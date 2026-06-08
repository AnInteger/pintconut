"""Tests for grid extraction module."""
import numpy as np

from src.grid import (
    PerspectiveCorrector,
    GridExtractor,
)


def _make_test_image_with_board():
    """Create a synthetic test image with a known 'board' region."""
    img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    img[50:250, 50:250] = [200, 200, 200]
    return img


def test_perspective_correction_returns_rectangle():
    """Test that perspective correction produces a rectangular output."""
    img = _make_test_image_with_board()
    corners = np.array([
        [50, 50],
        [250, 50],
        [250, 250],
        [50, 250],
    ], dtype=np.float32)

    corrector = PerspectiveCorrector()
    result = corrector.correct(img, corners, output_size=(200, 200))
    assert result.shape == (200, 200, 3)


def test_perspective_correction_handles_skew():
    """Test that perspective correction handles skewed corners."""
    img = _make_test_image_with_board()
    corners = np.array([
        [70, 40],
        [240, 60],
        [260, 260],
        [40, 240],
    ], dtype=np.float32)

    corrector = PerspectiveCorrector()
    result = corrector.correct(img, corners, output_size=(200, 200))
    assert result.shape == (200, 200, 3)
    assert result.dtype == np.uint8


def test_grid_extractor_returns_color_matrix():
    """Test that GridExtractor returns a rows*cols*3 color matrix."""
    board_img = np.zeros((200, 200, 3), dtype=np.uint8)
    for r in range(5):
        for c in range(5):
            y1 = r * 40
            x1 = c * 40
            color = [255, 0, 0] if (r + c) % 2 == 0 else [0, 255, 0]
            board_img[y1:y1+40, x1:x1+40] = color

    extractor = GridExtractor()
    result = extractor.extract(board_img, rows=5, cols=5)
    assert result.shape == (5, 5, 3)
    assert result.dtype == np.uint8


def test_grid_extractor_samples_center():
    """Test that GridExtractor samples from cell center, not edges."""
    board_img = np.zeros((100, 100, 3), dtype=np.uint8)
    board_img[10:90, 10:90] = [255, 0, 0]

    extractor = GridExtractor()
    result = extractor.extract(board_img, rows=1, cols=1)
    assert result[0, 0, 0] > 200
