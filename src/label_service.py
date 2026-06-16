"""Label service - core logic for bead board labeling."""
import os
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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


# Segmentation parameter presets. The label UI can switch between these per
# photo: 「标准」 is fast and accurate for well-framed boards (clear margin),
# while 「宽松」 lowers conf/iou and raises imgsz to merge the tiny shards that
# FastSAM produces on dense, full-frame bead photos — recovering the whole
# board as one candidate (verified: IMG_6121 goes 0 -> 40% with loose params).
SEG_PRESETS = {
    "标准": {"conf": 0.4, "iou": 0.9, "imgsz": 640},
    "宽松": {"conf": 0.25, "iou": 0.6, "imgsz": 1024},
}


def segment_image(image: np.ndarray, model=None, conf: float = 0.4,
                  iou: float = 0.9, imgsz: int = 640) -> list[dict]:
    """Run FastSAM segmentation on an image and return rectangular candidates.

    Returns up to 10 candidate regions sorted by area (descending).
    Each candidate dict contains: mask, area, area_ratio, rect_ratio.

    Masks are kept at the model's low resolution (retina_masks=False) for the
    area/rectangularity filter, then only the handful of surviving candidates
    are upscaled to full image resolution. With retina_masks=True every mask is
    materialized at full resolution, which OOMs on large photos with many
    regions (e.g. a 1920x1440 HEIC yielding 100+ masks ≈ >1 GB of float data).
    """
    if model is None:
        from ultralytics import FastSAM
        model = FastSAM("FastSAM-s.pt")
    h, w = image.shape[:2]
    img_area = h * w
    results = model(image, device="cpu", retina_masks=False, imgsz=imgsz, conf=conf, iou=iou)
    if results[0].masks is None:
        return []
    masks = results[0].masks.data.cpu().numpy()
    mh, mw = masks.shape[1], masks.shape[2]
    lo_area = mh * mw
    candidates = []
    for m in masks:
        # Filter cheaply at the model's low resolution first.
        m_bin = (m > 0.5).astype(np.uint8)
        area_ratio = float(m_bin.sum()) / lo_area  # ratio is scale-invariant
        if area_ratio < 0.05 or area_ratio > 0.95:
            continue
        rect_ok, rect_ratio = is_rectangular(m_bin, threshold=0.5)
        if not rect_ok:
            continue
        # Survivor: upscale just this mask to full resolution for display/labels.
        mask_full = cv2.resize(m_bin, (w, h), interpolation=cv2.INTER_NEAREST)
        area = int(mask_full.sum())
        candidates.append({"mask": mask_full, "area": area,
                           "area_ratio": area / img_area, "rect_ratio": rect_ratio})
    candidates.sort(key=lambda c: c["area"], reverse=True)
    return candidates[:10]


def draw_candidates(image: np.ndarray, candidates: list[dict]) -> np.ndarray:
    """Draw candidate region overlays on a copy of the image.

    Each candidate is rendered as a semi-transparent colored mask with a colored
    contour outline. Rather than scattering tiny number labels across a busy
    bead photo (hard to read), all candidates are consolidated into a single
    chart-style legend panel in the top-left corner: each row pairs a color
    swatch — matching the region's fill color — with its index and area, so the
    user can cross-reference a colored region on the image with its legend entry.
    """
    display = image.copy()
    h, w = display.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(1.2, h / 700.0)
    thickness = max(2, int(round(font_scale * 1.2)))
    contour_th = max(6, int(round(font_scale * 3)))

    # Winner-take-all coloring: assign each pixel to the SMALLEST candidate that
    # contains it (candidates are sorted by area descending, so iterating in
    # index order lets the smallest overwrite the larger ones). Each pixel then
    # gets exactly one tint instead of a muddy stack of translucent overlays —
    # so the region colors stay clean and distinct, matching the legend swatches.
    owner = np.zeros((h, w), dtype=np.int32)
    for i, cand in enumerate(candidates):
        owner[cand["mask"] > 0] = i + 1

    disp_f = display.astype(np.float32)
    for i, cand in enumerate(candidates):
        m = owner == (i + 1)
        if not m.any():
            continue
        color = np.array(COLORS[i % len(COLORS)], dtype=np.float32)
        disp_f[m] = disp_f[m] * 0.45 + color * 0.55
    display = disp_f.astype(np.uint8)

    # Colored contour outlines on top, to delineate each region's boundary.
    for i, cand in enumerate(candidates):
        color = COLORS[i % len(COLORS)]
        contours, _ = cv2.findContours(cand["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(display, contours, -1, color, contour_th)

    _draw_legend(display, candidates)
    return display


# ---------------------------------------------------------------------------
# Legend rendering (anti-aliased via Pillow for a clean, professional look)
# ---------------------------------------------------------------------------
_FONT_CANDIDATES = {
    "bold": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ],
    "cjk": [
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ],
}
_font_cache: dict = {}


def _font(kind: str, size: int):
    """Load a TrueType font (cached), falling back to Pillow's default bitmap."""
    key = (kind, size)
    if key not in _font_cache:
        for path in _FONT_CANDIDATES.get(kind, []):
            if os.path.exists(path):
                try:
                    _font_cache[key] = ImageFont.truetype(path, size)
                    break
                except Exception:
                    continue
        else:
            _font_cache[key] = ImageFont.load_default()
    return _font_cache[key]


def _to_rgb(bgr) -> tuple[int, int, int]:
    """Convert a BGR scalar (as used in COLORS) to an RGB tuple."""
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def _draw_legend(display: np.ndarray, candidates: list[dict]) -> None:
    """Draw a polished, chart-style legend in the top-left corner.

    Rendered anti-aliased through Pillow (cv2 Hershey fonts can't anti-alias or
    show CJK). Each row pairs a rounded color swatch — matching the region's fill
    color — with its index and area, on a dark rounded panel with a soft shadow
    and a header. The result reads like a chart legend, so a colored region on
    the image is matched to its entry by color.
    """
    if not candidates:
        return
    h, w = display.shape[:2]

    # Size the legend relative to image height so that after Gradio downscales
    # the photo to ~400-550px for display, the legend text is still ~16px and
    # crisp rather than tiny and blurry.
    font_size = max(28, int(round(h * 0.04)))
    font = _font("bold", font_size)
    header_font = _font("cjk", max(18, int(round(font_size * 0.7))))

    pad = int(font_size * 0.95)            # panel inner padding
    swatch_gap = int(font_size * 0.7)      # gap between swatch and text
    row_gap = int(font_size * 0.55)        # gap between rows
    swatch = int(font_size * 0.95)         # square swatch side
    radius = max(3, swatch // 3)

    # Work on a transparent RGBA layer composited over an RGB copy of the image.
    base = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB)).convert("RGBA")
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    # Measure rows.
    rows = []
    max_tw = 0
    for i, cand in enumerate(candidates):
        rgb = _to_rgb(COLORS[i % len(COLORS)])
        text = f"#{i}    {cand['area_ratio'] * 100:.1f}%"
        bbox = draw.textbbox((0, 0), text, font=font)
        rows.append((rgb, text, bbox[2] - bbox[0], bbox[3] - bbox[1]))
        max_tw = max(max_tw, bbox[2] - bbox[0])

    header = "候选区域"
    hbbox = draw.textbbox((0, 0), header, font=header_font)
    hth = hbbox[3] - hbbox[1]
    header_gap = int(font_size * 0.75)

    content_w = max(max_tw, hbbox[2] - hbbox[0])
    panel_w = pad + swatch + swatch_gap + content_w + pad
    panel_h = pad + hth + header_gap + len(rows) * (swatch + row_gap) - row_gap + pad

    x0, y0 = int(w * 0.025), int(h * 0.025)
    x1, y1 = x0 + panel_w, y0 + panel_h
    pr = radius + 4  # panel corner radius

    # Soft drop shadow, then the dark panel with a subtle light border.
    sh = max(3, font_size // 6)
    draw.rounded_rectangle((x0 + sh, y0 + sh, x1 + sh, y1 + sh),
                           radius=pr, fill=(0, 0, 0, 90))
    draw.rounded_rectangle((x0, y0, x1, y1), radius=pr,
                           fill=(26, 27, 30, 232),
                           outline=(255, 255, 255, 60),
                           width=max(1, font_size // 18))

    # Header + divider line.
    hx = x0 + pad
    hy = y0 + pad
    draw.text((hx, hy), header, font=header_font, fill=(236, 236, 236, 255), anchor="la")
    div_y = hy + hth + header_gap // 2
    draw.line((hx, div_y, x1 - pad, div_y), fill=(255, 255, 255, 45),
              width=max(1, font_size // 22))

    # Rows: rounded color swatch + label text, both vertically centered.
    ry = hy + hth + header_gap
    for rgb, text, _tw, _th in rows:
        cy = ry + swatch // 2
        sx = hx
        sy = ry
        draw.rounded_rectangle((sx, sy, sx + swatch, sy + swatch), radius=radius,
                               fill=rgb + (255,), outline=(255, 255, 255, 200),
                               width=max(1, font_size // 24))
        draw.text((sx + swatch + swatch_gap, cy), text, font=font,
                  fill=(240, 240, 240, 255), anchor="lm")
        ry += swatch + row_gap

    # Composite back onto the BGR image.
    composited = Image.alpha_composite(base, layer).convert("RGB")
    display[:] = cv2.cvtColor(np.array(composited), cv2.COLOR_RGB2BGR)


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
