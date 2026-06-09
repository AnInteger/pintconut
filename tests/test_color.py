"""Tests for color matching module."""
import json
import os
import numpy as np

from src.color import ColorMatcher, load_color_palette


def test_load_color_palette():
    """Test loading color palette from JSON file."""
    palette = load_color_palette("data/colors.json")
    assert len(palette) == 20
    assert palette[0]["name"] == "White"
    assert palette[0]["rgb"] == [255, 255, 255]


def test_match_exact_color():
    """Test matching an exact color from the palette."""
    matcher = ColorMatcher("data/colors.json")
    result = matcher.match([255, 0, 0])
    assert result["name"] == "Red"


def test_match_close_color():
    """Test matching a color close to a palette entry."""
    matcher = ColorMatcher("data/colors.json")
    result = matcher.match([250, 10, 5])
    assert result["name"] == "Red"


def test_match_in_lab_space():
    """Test that matching works in LAB color space."""
    matcher = ColorMatcher("data/colors.json")
    result = matcher.match([240, 240, 240])
    assert result["name"] == "White"


def test_is_bead_present():
    """Test detecting whether a bead is present at a grid cell."""
    matcher = ColorMatcher("data/colors.json")
    assert matcher.is_bead([255, 0, 0]) is True
    assert matcher.is_bead([250, 250, 250]) is False
    assert matcher.is_bead([245, 245, 245]) is False


def test_is_bead_dark_color():
    """Test that dark colored beads are detected."""
    matcher = ColorMatcher("data/colors.json")
    assert matcher.is_bead([0, 0, 0]) is True
    assert matcher.is_bead([64, 64, 64]) is True

def test_match_box_returns_color_info():
    matcher = ColorMatcher("data/colors.json")
    # Create a 100x100 BGR image with a red region
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[30:70, 30:70] = [0, 0, 255]  # Red in BGR
    box = {"xyxy": [30, 30, 70, 70], "conf": 0.9}
    result = matcher.match_box(img, box, sample_fraction=0.4)
    assert result["color_name"] == "Red"
    assert result["is_bead"] is True
    assert "rgb" in result
    assert "confidence" in result

def test_match_box_empty_board():
    matcher = ColorMatcher("data/colors.json")
    img = np.ones((100, 100, 3), dtype=np.uint8) * 250  # Near-white in BGR
    box = {"xyxy": [10, 10, 50, 50], "conf": 0.9}
    result = matcher.match_box(img, box, sample_fraction=0.4)
    assert result["is_bead"] is False
