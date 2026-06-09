"""Bead detection module — locates individual beads using YOLOv8n."""
import os
from pathlib import Path

import numpy as np


class BeadDetector:
    """Detects individual beads on a perspective-corrected board image."""
    def __init__(self, model_path: str = "models/bead-best.pt", conf_threshold: float = 0.5):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None

    def _load_model(self):
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)

    def detect(self, img: np.ndarray) -> list[dict]:
        """Run YOLO detection on image and return list of box dictionaries.

        Args:
            img: Input image (BGR format, from OpenCV)

        Returns:
            List of dicts with keys: xyxy, conf, width, height, cx, cy
            Returns empty list if model file doesn't exist.
        """
        if not os.path.exists(self.model_path):
            return []

        self._load_model()

        results = self.model(img, conf=self.conf_threshold, verbose=False)
        boxes = []

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = xyxy
                    width = x2 - x1
                    height = y2 - y1
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    boxes.append({
                        "xyxy": xyxy,
                        "conf": conf,
                        "width": width,
                        "height": height,
                        "cx": cx,
                        "cy": cy,
                    })

        return boxes

    def filter_boxes(
        self,
        boxes: list[dict],
        conf_threshold: float | None = None,
        ref_size: float | None = None,
        size_range: tuple[float, float] = (0.3, 3.0),
        board_mask: np.ndarray | None = None,
    ) -> list[dict]:
        """Filter detected boxes by confidence, size, and mask containment.

        Args:
            boxes: List of box dictionaries from detect()
            conf_threshold: Minimum confidence (uses self.conf_threshold if None)
            ref_size: Reference size for median-based filtering (uses median if None)
            size_range: (min_ratio, max_ratio) relative to ref_size
            board_mask: Binary mask where boxes should be contained (255=inside)

        Returns:
            Filtered list of box dictionaries
        """
        if conf_threshold is None:
            conf_threshold = self.conf_threshold

        # Filter by confidence
        filtered = [b for b in boxes if b["conf"] >= conf_threshold]

        # Filter by size if ref_size provided or can be estimated
        if ref_size is None and filtered:
            # Use median width as reference size
            ref_size = np.median([b["width"] for b in filtered])

        if ref_size is not None and ref_size > 0:
            min_size = ref_size * size_range[0]
            max_size = ref_size * size_range[1]
            filtered = [b for b in filtered if min_size <= b["width"] <= max_size]

        # Filter by mask containment
        if board_mask is not None and filtered:
            filtered_masked = []
            for b in filtered:
                cx, cy = b["cx"], b["cy"]
                # Check if center point is inside the mask
                if 0 <= cy < board_mask.shape[0] and 0 <= cx < board_mask.shape[1]:
                    if board_mask[cy, cx] > 0:
                        filtered_masked.append(b)
            filtered = filtered_masked

        return filtered
