"""Tests for bead pre-labeling module."""
import cv2
import numpy as np
import os
import tempfile
from src.bead_prelabel import hough_circles_to_boxes, save_yolo_boxes


def test_hough_circles_to_boxes_on_grid():
    """Test detecting circles arranged in a grid pattern."""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 220
    for r in range(5):
        for c in range(5):
            cx = 20 + c * 40
            cy = 20 + r * 40
            cv2.circle(img, (cx, cy), 12, (0, 0, 0), -1)
    boxes = hough_circles_to_boxes(img)
    assert len(boxes) > 0
    assert "xyxy" in boxes[0]
    assert "conf" in boxes[0]


def test_hough_circles_to_boxes_no_circles():
    """Test that empty image returns empty list."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    boxes = hough_circles_to_boxes(img)
    assert isinstance(boxes, list)


def test_save_yolo_boxes():
    """Test saving boxes in YOLO detection format."""
    boxes = [
        {"xyxy": [10, 10, 30, 30], "conf": 0.9},
        {"xyxy": [50, 50, 70, 70], "conf": 0.8},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test.txt")
        save_yolo_boxes(boxes, img_w=100, img_h=100, output_path=out_path)
        assert os.path.exists(out_path)
        with open(out_path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        parts = lines[0].strip().split()
        assert parts[0] == "0"
        assert len(parts) == 5
