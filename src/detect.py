"""
Bead board detection module.

Uses a fine-tuned YOLOv8-seg model to locate the bead board
in a photo and extract its four corner points.
"""
import numpy as np
import cv2


class BoardDetector:
    """Detects the bead board in a photo using YOLOv8-seg."""

    model_path = "models/beadboard-best.pt"
    model = None

    def __init__(self, model_path: str = "models/beadboard-best.pt"):
        self.model_path = model_path
        self.model = None

    def _load_model(self):
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)

    def detect(self, image: np.ndarray) -> np.ndarray | None:
        """Detect the bead board in an image and return its mask.

        Args:
            image: Input image (HxWx3 BGR array).

        Returns:
            Binary mask of the board region (HxW uint8), or None if not found.
        """
        self._load_model()
        results = self.model(image, verbose=False)

        if not results or results[0].masks is None:
            return None

        masks = results[0].masks.data.cpu().numpy()
        h, w = image.shape[:2]

        best_mask = None
        best_area = 0
        for mask in masks:
            mask_resized = cv2.resize(mask, (w, h))
            mask_binary = (mask_resized > 0.5).astype(np.uint8)
            area = np.sum(mask_binary)
            if area > best_area:
                best_area = area
                best_mask = mask_binary

        return best_mask

    def extract_corners(self, mask: np.ndarray) -> np.ndarray | None:
        """Extract 4 corner points from a binary board mask.

        Args:
            mask: Binary mask (HxW uint8, 255 = board region).

        Returns:
            4x2 array of corner points: [top-left, top-right, bottom-right, bottom-left],
            or None if no valid quadrilateral found.
        """
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)

        for epsilon_factor in [0.01, 0.02, 0.03, 0.05, 0.1]:
            epsilon = epsilon_factor * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                corners = approx.reshape(4, 2).astype(np.float32)
                return self._order_corners(corners)

        hull = cv2.convexHull(contour)
        if len(hull) < 4:
            return None

        corners = self._find_extreme_corners(hull.reshape(-1, 2))
        return self._order_corners(corners.astype(np.float32))

    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners as: top-left, top-right, bottom-right, bottom-left."""
        sums = corners[:, 0] + corners[:, 1]
        diffs = corners[:, 1] - corners[:, 0]

        ordered = np.array([
            corners[np.argmin(sums)],
            corners[np.argmin(diffs)],
            corners[np.argmax(sums)],
            corners[np.argmax(diffs)],
        ], dtype=np.float32)

        return ordered

    def _find_extreme_corners(self, points: np.ndarray) -> np.ndarray:
        """Find the 4 extreme corners from a set of hull points."""
        sums = points[:, 0] + points[:, 1]
        diffs = points[:, 1] - points[:, 0]

        return np.array([
            points[np.argmin(sums)],
            points[np.argmin(diffs)],
            points[np.argmax(sums)],
            points[np.argmax(diffs)],
        ])
