"""Tests for bead detection module."""
import numpy as np
from src.bead_detect import BeadDetector

def test_bead_detector_init():
    """Test that BeadDetector initializes with expected defaults."""
    detector = BeadDetector()
    assert detector.model_path == "models/bead-best.pt"
    assert detector.conf_threshold == 0.5
    assert detector.model is None
