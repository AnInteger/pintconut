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
