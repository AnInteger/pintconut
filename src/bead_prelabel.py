"""Bead pre-labeling module — uses HoughCircles to auto-detect beads."""
import os

import numpy as np


def hough_circles_to_boxes(
    image,
    dp=1.2,
    min_dist=10,
    param1=100,
    param2=30,
    min_radius=3,
    max_radius=30,
):
    """Detect circular beads using HoughCircles and convert to YOLO boxes.

    Args:
        image: Input image (BGR format from OpenCV)
        dp: Inverse ratio of accumulator resolution to image resolution
        min_dist: Minimum distance between detected circle centers
        param1: Higher threshold for Canny edge detection
        param2: Accumulator threshold for circle detection (lower = more circles)
        min_radius: Minimum circle radius
        max_radius: Maximum circle radius

    Returns:
        List of dicts with keys: xyxy, conf, cx, cy, width, height
        Returns empty list if no circles found
    """
    import cv2

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Run HoughCircles
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return []

    boxes = []
    circles = np.round(circles[0, :]).astype("int")

    for (x, y, r) in circles:
        x1 = x - r
        y1 = y - r
        x2 = x + r
        y2 = y + r
        width = x2 - x1
        height = y2 - y1

        boxes.append({
            "xyxy": [x1, y1, x2, y2],
            "conf": 1.0,  # HoughCircles doesn't provide confidence
            "cx": x,
            "cy": y,
            "width": width,
            "height": height,
        })

    return boxes


def save_yolo_boxes(boxes, img_w, img_h, output_path):
    """Save detected boxes in YOLO detection format.

    Format: class_id x_center y_center width height (all normalized 0-1)

    Args:
        boxes: List of box dictionaries with xyxy coordinates
        img_w: Image width in pixels
        img_h: Image height in pixels
        output_path: Path to save the label file

    Returns:
        Number of boxes written
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        for box in boxes:
            x1, y1, x2, y2 = box["xyxy"]

            # Calculate center and dimensions (normalized)
            x_center = (x1 + x2) / 2.0 / img_w
            y_center = (y1 + y2) / 2.0 / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h

            # Class ID is 0 for beads
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    return len(boxes)
