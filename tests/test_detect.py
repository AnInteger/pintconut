"""Tests for bead board detection module."""
import numpy as np

from src.detect import BoardDetector


def test_board_detector_init():
    """Test that BoardDetector has expected attributes."""
    detector = BoardDetector.__new__(BoardDetector)
    assert hasattr(detector, "model_path")
    assert hasattr(detector, "model")


def test_extract_corners_from_mask():
    """Test extracting 4 corner points from a rectangular mask."""
    detector = BoardDetector.__new__(BoardDetector)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255

    corners = detector.extract_corners(mask)
    assert corners is not None
    assert corners.shape == (4, 2)


def test_extract_corners_returns_none_for_empty_mask():
    """Test that extract_corners returns None for empty mask."""
    detector = BoardDetector.__new__(BoardDetector)
    mask = np.zeros((100, 100), dtype=np.uint8)
    result = detector.extract_corners(mask)
    assert result is None
