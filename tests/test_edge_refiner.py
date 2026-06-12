"""Tests for src.edge_refiner — core refinement pipeline."""
from __future__ import annotations

import numpy as np
import cv2
import pytest

from src.edge_refiner import (
    EdgeRefiner,
    RefinerConfig,
    EdgeResult,
    EdgeQuality,
    BoardConfidence,
    ClipInfo,
    DetectionError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rect_mask(h: int, w: int, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Binary mask with a filled rectangle."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def _white_board_image(h: int, w: int, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """BGR image: white rectangle on dark background."""
    img = np.full((h, w, 3), 40, dtype=np.uint8)
    img[y1:y2, x1:x2] = (230, 220, 210)  # light warm grey (BGR)
    return img


def _grid_texture_image(size: int = 200, period: int = 14) -> np.ndarray:
    """Synthetic image with a regular dot-grid pattern (simulates bead board)."""
    img = np.full((size, size, 3), 220, dtype=np.uint8)
    for y in range(period // 2, size, period):
        for x in range(period // 2, size, period):
            cv2.circle(img, (x, y), period // 4, (80, 80, 80), -1)
    return img


# ---------------------------------------------------------------------------
# Truncation detection
# ---------------------------------------------------------------------------

class TestTruncationDetection:
    def test_full_board_no_clipping(self):
        """Board fully inside frame → all 4 edges visible."""
        mask = _rect_mask(400, 600, 50, 50, 550, 350)
        refiner = EdgeRefiner()
        clips = refiner._detect_truncation(mask, 400, 600)
        assert len(clips) == 4
        assert all(not c.is_clipped for c in clips)

    def test_one_side_clipped(self):
        """Board touching bottom edge → bottom segment detected as clipped."""
        mask = _rect_mask(400, 600, 50, 50, 550, 400)  # bottom flush with frame
        refiner = EdgeRefiner(RefinerConfig(clip_boundary_threshold=5))
        clips = refiner._detect_truncation(mask, 400, 600)
        # At least one segment should be clipped (the one near bottom)
        assert any(c.is_clipped for c in clips)

    def test_two_adjacent_sides_clipped(self):
        """Board touching right and bottom edges."""
        mask = _rect_mask(400, 600, 50, 50, 599, 399)
        refiner = EdgeRefiner()
        clips = refiner._detect_truncation(mask, 400, 600)
        clipped_count = sum(1 for c in clips if c.is_clipped)
        assert clipped_count >= 1  # at least the segments near the edges

    def test_empty_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        refiner = EdgeRefiner()
        clips = refiner._detect_truncation(mask, 100, 100)
        assert len(clips) == 4
        assert all(not c.is_clipped for c in clips)


# ---------------------------------------------------------------------------
# ROI computation
# ---------------------------------------------------------------------------

class TestROI:
    def test_roi_is_ring(self):
        mask = _rect_mask(200, 200, 30, 30, 170, 170)
        refiner = EdgeRefiner()
        roi = refiner._compute_roi(mask, 200, 200)
        # Centre should be excluded
        assert roi[100, 100] == 0
        # Near edge should be included
        assert roi[31, 100] == 1 or roi[32, 100] == 1

    def test_boundary_exclusion(self):
        mask = _rect_mask(200, 200, 0, 0, 200, 200)  # full frame
        refiner = EdgeRefiner()
        roi = refiner._compute_roi(mask, 200, 200)
        # Top few rows should be zeroed
        assert roi[0, 100] == 0
        assert roi[1, 100] == 0


# ---------------------------------------------------------------------------
# Colour pre-filter
# ---------------------------------------------------------------------------

class TestColourPreFilter:
    def test_white_passes(self):
        mask = _rect_mask(200, 200, 30, 30, 170, 170)
        lab = np.full((200, 200, 3), 0, dtype=np.uint8)
        lab[:, :] = [220, 128, 135]  # L=220, a≈0, b≈7 → white/off-white in LAB
        refiner = EdgeRefiner()
        score, passed = refiner._color_prefilter(mask, lab)
        assert passed
        assert score > 0.3

    def test_dark_fails(self):
        mask = _rect_mask(200, 200, 30, 30, 170, 170)
        lab = np.full((200, 200, 3), 0, dtype=np.uint8)
        lab[:, :] = [30, 128, 128]  # very dark
        refiner = EdgeRefiner()
        score, passed = refiner._color_prefilter(mask, lab)
        assert not passed

    def test_small_mask_passes_gracefully(self):
        mask = _rect_mask(200, 200, 95, 95, 96, 96)  # tiny 1x1 rect
        lab = np.full((200, 200, 3), 0, dtype=np.uint8)
        refiner = EdgeRefiner()
        _, passed = refiner._color_prefilter(mask, lab)
        assert passed  # should not block on insufficient samples


# ---------------------------------------------------------------------------
# Hough line detection
# ---------------------------------------------------------------------------

class TestHoughLines:
    def test_detects_horizontal_line(self):
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.line(img, (20, 100), (180, 100), 255, 2)
        roi = np.ones((200, 200), dtype=np.uint8)
        refiner = EdgeRefiner()
        lines = refiner._detect_hough_lines(roi, img)
        assert len(lines) > 0
        # At least one line should be roughly horizontal
        horizontal = [l for l in lines if abs(l[1]) < 0.2 or abs(l[1] - np.pi) < 0.2]
        assert len(horizontal) > 0

    def test_no_lines_on_blank(self):
        img = np.zeros((200, 200), dtype=np.uint8)
        roi = np.ones((200, 200), dtype=np.uint8)
        refiner = EdgeRefiner()
        lines = refiner._detect_hough_lines(roi, img)
        assert len(lines) == 0


# ---------------------------------------------------------------------------
# Corner computation
# ---------------------------------------------------------------------------

class TestCornerComputation:
    def test_order_corners_tl_tr_br_bl(self):
        corners = np.array([
            [300, 100],  # TR
            [100, 300],  # BL
            [100, 100],  # TL
            [300, 300],  # BR
        ], dtype=np.float32)
        ordered = EdgeRefiner._order_corners(corners)
        assert ordered[0][0] < ordered[1][0]  # TL.x < TR.x
        assert ordered[0][1] < ordered[3][1]  # TL.y < BL.y
        assert ordered[2][0] > ordered[3][0]  # BR.x > BL.x

    def test_line_intersection(self):
        # Horizontal line y=100, vertical line x=200
        l1 = (100.0, np.pi / 2, 300.0)   # rho=100, theta=90° → y=100
        l2 = (200.0, 0.0, 300.0)          # rho=200, theta=0°  → x=200
        pt = EdgeRefiner._line_intersection(l1, l2)
        assert pt is not None
        assert abs(pt[0] - 200) < 1
        assert abs(pt[1] - 100) < 1

    def test_parallel_lines_no_intersection(self):
        l1 = (100.0, np.pi / 2, 300.0)
        l2 = (200.0, np.pi / 2, 300.0)
        assert EdgeRefiner._line_intersection(l1, l2) is None


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_high_confidence_total(self):
        c = BoardConfidence(q_object=0.95, q_detection=0.90)
        assert abs(c.total - 0.855) < 0.01
        assert c.level == "高"

    def test_medium_confidence(self):
        c = BoardConfidence(q_object=0.7, q_detection=0.8)
        assert c.level == "中"

    def test_low_confidence(self):
        c = BoardConfidence(q_object=0.3, q_detection=0.4)
        assert c.level == "低"


# ---------------------------------------------------------------------------
# Visibility mask
# ---------------------------------------------------------------------------

class TestVisibilityMask:
    def test_mask_and_quad_intersection(self):
        h, w = 200, 200
        mask = _rect_mask(h, w, 40, 40, 160, 160)
        corners = np.array([[40, 40], [160, 40], [160, 160], [40, 160]], dtype=np.float32)
        vis = EdgeRefiner._build_visibility_mask(corners, mask, h, w)
        # Centre should be visible
        assert vis[100, 100]
        # Corner (outside mask) should not
        assert not vis[10, 10]


# ---------------------------------------------------------------------------
# Full pipeline integration (synthetic data)
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def _make_synthetic_board(self) -> tuple[np.ndarray, np.ndarray]:
        """Create a synthetic board image + mask for end-to-end testing."""
        h, w = 400, 600
        mask = _rect_mask(h, w, 80, 60, 520, 340)

        img = np.full((h, w, 3), 30, dtype=np.uint8)
        # Board region: light grey with grid dots
        for y in range(60, 340):
            for x in range(80, 520):
                img[y, x] = (215, 210, 205)
        return img, mask

    def test_refine_returns_detection(self):
        img, mask = self._make_synthetic_board()
        refiner = EdgeRefiner(RefinerConfig(
            hough_threshold=30,
            hough_min_line_length=20,
        ))
        result = refiner.refine(mask, img, board_size=(29, 29))
        assert result.corners.shape == (4, 2)
        assert result.visibility_mask.shape == (400, 600)
        assert 0 <= result.confidence.total <= 1

    def test_refine_handles_truncated_board(self):
        h, w = 300, 400
        mask = _rect_mask(h, w, 40, 40, 399, 260)  # right edge at frame boundary
        img = np.full((h, w, 3), 30, dtype=np.uint8)
        img[40:260, 40:399] = (215, 210, 205)

        refiner = EdgeRefiner(RefinerConfig(
            hough_threshold=30,
            hough_min_line_length=20,
        ))
        result = refiner.refine(mask, img, board_size=(21, 29))
        assert result.corners.shape == (4, 2)


# ---------------------------------------------------------------------------
# Geometry validation
# ---------------------------------------------------------------------------

class TestGeometryValidation:
    def test_valid_rectangle_passes(self):
        refiner = EdgeRefiner()
        edges = [
            EdgeResult(0, (200, np.pi / 2, 400), EdgeQuality(0.9, 0.8, 0.9, 0.8, 0.7, 0.6), False, None),
            EdgeResult(1, (400, 0, 300), EdgeQuality(0.9, 0.8, 0.9, 0.8, 0.7, 0.6), False, None),
            EdgeResult(2, (100, np.pi / 2, 400), EdgeQuality(0.9, 0.8, 0.9, 0.8, 0.7, 0.6), False, None),
            EdgeResult(3, (100, 0, 300), EdgeQuality(0.9, 0.8, 0.9, 0.8, 0.7, 0.6), False, None),
        ]
        assert refiner._validate_geometry(edges, (29, 29))

    def test_single_edge_always_passes(self):
        refiner = EdgeRefiner()
        edges = [
            EdgeResult(0, (200, np.pi / 2, 400), EdgeQuality(0.9, 0.8, 0.9, 0.8, 0.7, 0.6), False, None),
        ]
        assert refiner._validate_geometry(edges, (29, 29))
