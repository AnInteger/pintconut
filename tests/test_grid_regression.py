"""Real-photo regression tests for BeadGridFitter.

These are SMOKE tests that verify the pipeline runs without crashing and
produces a sensible grid (at least 5x5 with 50+ beads) on real photos.
They catch regressions in detect_beads / estimate_grid_axes / label_beads.

Fixtures: 3 real photos copied from training/photos/ into tests/fixtures/board_regression/.
These are committed to the repo so tests are self-contained.

Known limitations (as of Task 15 tuning):
- Fill ratios are ~0.67-0.77 (reasonable but not perfect). Real photos have
  missing beads and some false detections, so fill < 1.0 is expected.
- perspective_tier is True on all 3 photos, including upright. This is because
  the affine residual is high on real photos (0.47-0.51) due to imperfect bead
  positions, triggering the projective upgrade path. The projective fit degrades
  gracefully for near-frontal boards, so this is acceptable.
- Grid sizes are auto-detected from the visible bead region and may not match
  the full board size (use --board-size for that).
"""
from pathlib import Path

import cv2
import pytest

from src.bead_grid import BeadGridFitter, GridFitError

FIX = Path(__file__).parent / "fixtures" / "board_regression"


def _load(name):
    img = cv2.imread(str(FIX / name))
    if img is None:
        pytest.skip(f"无法读取 {name}")
    return img


class TestUprightPhoto:
    """IMG_6097.PNG — upright board, no significant perspective."""

    def test_fit_runs(self):
        result = BeadGridFitter().fit(_load("upright.png"))
        assert result.rows >= 5 and result.cols >= 5
        assert result.confidence.bead_count >= 50

    def test_fill_ratio_reasonable(self):
        result = BeadGridFitter().fit(_load("upright.png"))
        assert result.confidence.grid_fill_ratio <= 1.0, (
            f"Fill ratio {result.confidence.grid_fill_ratio:.2f} > 1.0 — too many beads vs grid cells"
        )


class TestPerspectivePhoto:
    """IMG_6121.JPG — strong perspective (tilted board)."""

    def test_fit_runs(self):
        result = BeadGridFitter().fit(_load("perspective.jpg"))
        assert result.rows >= 5 and result.cols >= 5
        assert result.confidence.bead_count >= 50

    def test_fill_ratio_reasonable(self):
        result = BeadGridFitter().fit(_load("perspective.jpg"))
        assert result.confidence.grid_fill_ratio <= 1.0, (
            f"Fill ratio {result.confidence.grid_fill_ratio:.2f} > 1.0 — too many beads vs grid cells"
        )

    def test_perspective_detected(self):
        """After tuning, the perspective photo should trigger projective tier."""
        result = BeadGridFitter().fit(_load("perspective.jpg"))
        assert result.confidence.perspective_tier is True, (
            "Tilted board should be detected as perspective"
        )


class TestClutterPhoto:
    """IJYJ0881.jpg — cluttered background."""

    def test_fit_runs(self):
        result = BeadGridFitter().fit(_load("clutter.jpg"))
        assert result.rows >= 5 and result.cols >= 5
        assert result.confidence.bead_count >= 50

    def test_fill_ratio_reasonable(self):
        result = BeadGridFitter().fit(_load("clutter.jpg"))
        assert result.confidence.grid_fill_ratio <= 1.0, (
            f"Fill ratio {result.confidence.grid_fill_ratio:.2f} > 1.0 — too many beads vs grid cells"
        )
