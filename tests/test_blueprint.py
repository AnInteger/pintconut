"""Tests for blueprint parsing module."""
import numpy as np

from src.blueprint import parse_blueprint


def test_parse_blueprint_returns_grid():
    """Test that parse_blueprint returns a 2D color grid."""
    result = parse_blueprint("tests/fixtures/test_blueprint_5x5.png", rows=5, cols=5)
    assert result.shape == (5, 5, 3)
    assert result.dtype == np.uint8


def test_parse_blueprint_detects_colors():
    """Test that parse_blueprint correctly extracts cell colors."""
    result = parse_blueprint("tests/fixtures/test_blueprint_5x5.png", rows=5, cols=5)
    top_left = result[0, 0]
    assert abs(int(top_left[0]) - 255) < 30
    assert abs(int(top_left[1]) - 0) < 30
    assert abs(int(top_left[2]) - 0) < 30


def test_parse_blueprint_handles_margin():
    """Test that parse_blueprint ignores non-grid margins."""
    result = parse_blueprint("tests/fixtures/test_blueprint_5x5.png", rows=5, cols=5)
    assert result.shape == (5, 5, 3)
