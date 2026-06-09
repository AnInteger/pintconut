"""Tests for bead detection module."""
import numpy as np
from src.bead_detect import BeadDetector

def test_bead_detector_init():
    """Test that BeadDetector initializes with expected defaults."""
    detector = BeadDetector()
    assert detector.model_path == "models/bead-best.pt"
    assert detector.conf_threshold == 0.5
    assert detector.model is None

def test_filter_boxes_by_confidence():
    detector = BeadDetector()
    boxes = [
        {"xyxy": [10, 10, 30, 30], "conf": 0.9, "width": 20, "height": 20, "cx": 20, "cy": 20},
        {"xyxy": [40, 40, 60, 60], "conf": 0.3, "width": 20, "height": 20, "cx": 50, "cy": 50},
        {"xyxy": [70, 70, 90, 90], "conf": 0.6, "width": 20, "height": 20, "cx": 80, "cy": 80},
    ]
    filtered = detector.filter_boxes(boxes, conf_threshold=0.5)
    assert len(filtered) == 2
    assert all(b["conf"] >= 0.5 for b in filtered)

def test_filter_boxes_by_size():
    detector = BeadDetector()
    boxes = [
        {"xyxy": [10, 10, 30, 30], "conf": 0.9, "width": 20, "height": 20},
        {"xyxy": [40, 40, 45, 45], "conf": 0.9, "width": 5, "height": 5},
        {"xyxy": [70, 70, 170, 170], "conf": 0.9, "width": 100, "height": 100},
    ]
    filtered = detector.filter_boxes(boxes, conf_threshold=0.0, ref_size=20, size_range=(0.3, 3.0))
    assert len(filtered) == 1

def test_filter_boxes_inside_mask():
    detector = BeadDetector()
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    boxes = [
        {"xyxy": [25, 25, 45, 45], "conf": 0.9, "width": 20, "height": 20, "cx": 35, "cy": 35},
        {"xyxy": [5, 5, 25, 25], "conf": 0.9, "width": 20, "height": 20, "cx": 15, "cy": 15},
    ]
    filtered = detector.filter_boxes(boxes, conf_threshold=0.0, board_mask=mask)
    assert len(filtered) == 1
    assert filtered[0]["xyxy"][0] == 25

def test_detect_returns_list_of_dicts():
    detector = BeadDetector.__new__(BeadDetector)
    detector.model_path = "models/bead-best.pt"
    detector.conf_threshold = 0.5
    detector.model = None
    img = np.ones((600, 600, 3), dtype=np.uint8) * 200
    result = detector.detect(img)
    assert isinstance(result, list)

def test_detect_beads_returns_annotated_results():
    from src.color import ColorMatcher
    detector = BeadDetector()
    img = np.ones((600, 600, 3), dtype=np.uint8) * 200
    result = detector.detect_beads(img)
    assert isinstance(result, list)

def test_count_beads_by_color():
    detector = BeadDetector()
    beads = [
        {"color_name": "Red", "is_bead": True, "conf": 0.9, "cx": 10, "cy": 10},
        {"color_name": "Red", "is_bead": True, "conf": 0.8, "cx": 20, "cy": 20},
        {"color_name": "Blue", "is_bead": True, "conf": 0.9, "cx": 30, "cy": 30},
        {"color_name": "White", "is_bead": False, "conf": 0.7, "cx": 40, "cy": 40},
    ]
    counts = detector.count_beads(beads)
    assert counts["total"] == 3
    assert counts["by_color"]["Red"] == 2
    assert counts["by_color"]["Blue"] == 1
    assert "White" not in counts["by_color"]
