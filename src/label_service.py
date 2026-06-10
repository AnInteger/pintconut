"""Label service - core logic for bead board labeling."""
import os
import shutil
import cv2
import numpy as np

# BGR colors for candidate overlay — 10 distinct color families.
# Each from a different hue/saturation group for max perceptual distinction.
# For >10 regions, colors cycle (COLORS[i % len]) but numbers stay unique.
COLORS = [
    (0, 0, 255),       # 0: Red
    (255, 0, 0),       # 1: Blue
    (0, 255, 255),     # 2: Yellow
    (0, 255, 0),       # 3: Green
    (255, 0, 255),     # 4: Magenta
    (0, 128, 255),     # 5: Orange
    (255, 255, 0),     # 6: Cyan
    (200, 0, 200),     # 7: Purple
    (180, 0, 255),     # 8: Pink
    (128, 128, 128),   # 9: Gray
]


def is_rectangular(mask: np.ndarray, threshold: float = 0.80) -> tuple[bool, float]:
    """Check whether a mask is approximately rectangular.

    Compares the mask contour area against its convex hull area.
    A ratio close to 1.0 indicates a convex (and likely rectangular) shape.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, 0.0
    contour = max(contours, key=cv2.contourArea)
    mask_area = cv2.contourArea(contour)
    if mask_area < 100:
        return False, 0.0
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return False, 0.0
    ratio = mask_area / hull_area
    return ratio > threshold, ratio


def segment_image(image: np.ndarray, model=None) -> list[dict]:
    """Run FastSAM segmentation on an image and return rectangular candidates.

    Returns up to 10 candidate regions sorted by area (descending).
    Each candidate dict contains: mask, area, area_ratio, rect_ratio.
    """
    if model is None:
        from ultralytics import FastSAM
        model = FastSAM("FastSAM-s.pt")
    h, w = image.shape[:2]
    img_area = h * w
    results = model(image, device="cpu", retina_masks=True, imgsz=640, conf=0.4, iou=0.9)
    if results[0].masks is None:
        return []
    masks = results[0].masks.data.cpu().numpy()
    candidates = []
    for m in masks:
        mask_resized = cv2.resize(m, (w, h))
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        area = int(np.sum(mask_binary))
        area_ratio = area / img_area
        if area_ratio < 0.05 or area_ratio > 0.95:
            continue
        rect_ok, rect_ratio = is_rectangular(mask_binary, threshold=0.5)
        if not rect_ok:
            continue
        candidates.append({"mask": mask_binary, "area": area, "area_ratio": area_ratio, "rect_ratio": rect_ratio})
    candidates.sort(key=lambda c: c["area"], reverse=True)
    return candidates[:10]


def draw_candidates(image: np.ndarray, candidates: list[dict]) -> np.ndarray:
    """Draw candidate region overlays on a copy of the image.

    Each candidate is rendered as a semi-transparent colored mask with
    a contour outline and a numeric label.
    """
    display = image.copy()
    for i, cand in enumerate(candidates):
        color = COLORS[i % len(COLORS)]
        mask = cand["mask"]
        overlay = display.copy()
        overlay[mask > 0] = color
        cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(display, contours, -1, color, 2)
        if contours:
            M = cv2.moments(contours[0])
            if M["m00"] > 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            else:
                cx, cy = 30, 30 + i * 40
        else:
            cx, cy = 30, 30 + i * 40
        # Draw label with white background pill for readability on any color
        label = str(i)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        pad = 6
        # White filled rectangle behind text
        cv2.rectangle(display,
                      (cx - tw // 2 - pad, cy - th // 2 - pad),
                      (cx + tw // 2 + pad, cy + th // 2 + pad + baseline),
                      (255, 255, 255), -1)
        # Colored border around the pill matching the region color
        cv2.rectangle(display,
                      (cx - tw // 2 - pad, cy - th // 2 - pad),
                      (cx + tw // 2 + pad, cy + th // 2 + pad + baseline),
                      color, 2)
        # Text in the region color
        cv2.putText(display, label, (cx - tw // 2, cy + th // 2),
                    font, font_scale, color, thickness, cv2.LINE_AA)
    return display


def save_label(mask, image_path, output_images_dir, output_labels_dir, img_w, img_h) -> str:
    """Save a mask as a YOLO segmentation label and copy the source image.

    Generates normalized polygon coordinates in YOLO format (class x1 y1 x2 y2 ...).
    Returns the path to the created label file, or empty string if no valid contour.
    """
    basename = os.path.splitext(os.path.basename(image_path))[0]
    ext = os.path.splitext(image_path)[1]
    dest_image = os.path.join(output_images_dir, basename + ext)
    if not os.path.exists(dest_image):
        shutil.copy2(image_path, dest_image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ""
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.005 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)
    label_path = os.path.join(output_labels_dir, basename + ".txt")
    with open(label_path, "w") as f:
        coords = []
        for point in contour:
            coords.append(f"{point[0][0] / img_w:.6f}")
            coords.append(f"{point[0][1] / img_h:.6f}")
        f.write("0 " + " ".join(coords) + "\n")
    return label_path


def generate_dataset_yaml(dataset_dir: str) -> str:
    """Generate a YOLO-format data.yaml config file for the dataset.

    If the file already exists, returns its path without overwriting.
    """
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    if os.path.exists(yaml_path):
        return yaml_path
    yaml_content = f"""# Auto-generated dataset config
path: {os.path.abspath(dataset_dir)}
train: images/train
val: images/valid

names:
  0: beadboard
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    return yaml_path


def draw_result(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Draw annotation result on a copy of the image.

    Renders the mask contour as a green outline with a label, suitable for
    showing the user what they selected before confirming.
    """
    display = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return display
    # Draw green contour
    cv2.drawContours(display, contours, -1, (0, 255, 0), 2)
    # Place label above the largest contour
    contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(contour)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        top_y = contour[:, :, 1].min()
        text_y = max(top_y - 10, 20)
        cv2.putText(display, "board", (cx - 25, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return display
