"""Gradio web UI for bead board annotation — HoughCircles pre-labeling workflow."""

import os
import shutil
import sys

# Ensure project root is on the path so `src.*` imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import gradio as gr
import numpy as np

from src.bead_prelabel import hough_circles_to_boxes, save_yolo_boxes

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEAD_PHOTOS_DIR = os.path.join(BASE_DIR, "training", "bead_photos")
DATASET_DIR = os.path.join(BASE_DIR, "training", "bead_dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images", "train")
LABELS_DIR = os.path.join(DATASET_DIR, "labels", "train")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def _list_bead_photos() -> list[str]:
    """Return sorted list of image file paths in BEAD_PHOTOS_DIR."""
    if not os.path.isdir(BEAD_PHOTOS_DIR):
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(
        os.path.join(BEAD_PHOTOS_DIR, f)
        for f in os.listdir(BEAD_PHOTOS_DIR)
        if os.path.splitext(f)[1].lower() in exts
    )


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------
# Cache for current detection results
_current_boxes = []
_current_image_path = None
_current_image = None

# Dataset statistics
_dataset_stats = {"total": 0, "labeled": 0}


# ---------------------------------------------------------------------------
# UI Handlers
# ---------------------------------------------------------------------------
def _handle_upload(files):
    """Upload bead board photos and return (status_text, gallery_images)."""
    if not files:
        return "❌ 未选择文件", []
    _ensure_dirs(BEAD_PHOTOS_DIR)
    count = 0
    for f in files:
        fpath = f if isinstance(f, str) else getattr(f, "name", str(f))
        basename = os.path.basename(fpath)
        dest = os.path.join(BEAD_PHOTOS_DIR, basename)
        if os.path.abspath(fpath) != os.path.abspath(dest):
            shutil.copy2(fpath, dest)
        count += 1

    previews = []
    for img_path in _list_bead_photos():
        img = cv2.imread(img_path)
        if img is not None:
            previews.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    total = len(_list_bead_photos())
    return f"✅ 已上传 {count} 张照片（共 {total} 张）", previews


def _draw_boxes(image, boxes):
    """Draw detection boxes on image."""
    display = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box["xyxy"]
        cx, cy = box["cx"], box["cy"]

        # Draw green box
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw center point
        cv2.circle(display, (cx, cy), 3, (0, 0, 255), -1)

        # Draw label
        label = f"#{i}"
        cv2.putText(
            display, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
        )

    return cv2.cvtColor(display, cv2.COLOR_BGR2RGB)


def _handle_prelabel(
    image_input,
    dp,
    min_dist,
    param1,
    param2,
    min_radius,
    max_radius,
):
    """Run HoughCircles pre-labeling on input image."""
    global _current_boxes, _current_image, _current_image_path

    if image_input is None:
        return None, "❌ 请先上传图片", "", ""

    # Convert Gradio image to OpenCV format
    if isinstance(image_input, str):
        # File path
        img = cv2.imread(image_input)
        _current_image_path = image_input
    elif isinstance(image_input, np.ndarray):
        # Numpy array (already loaded)
        if len(image_input.shape) == 3 and image_input.shape[2] == 4:
            # RGBA to RGB
            img = cv2.cvtColor(image_input, cv2.COLOR_RGBA2BGR)
        elif len(image_input.shape) == 3:
            # RGB to BGR
            img = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
        else:
            # Grayscale to BGR
            img = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        _current_image_path = None
    else:
        return None, "❌ 不支持的图片格式", "", ""

    if img is None:
        return None, "❌ 无法读取图片", "", ""

    _current_image = img

    # Run HoughCircles detection
    boxes = hough_circles_to_boxes(
        img,
        dp=dp,
        min_dist=min_dist,
        param1=param1,
        param2=param2,
        min_radius=min_radius,
        max_radius=max_radius,
    )

    _current_boxes = boxes

    # Draw boxes on image
    if boxes:
        annotated = _draw_boxes(img, boxes)
        status = f"✅ 检测到 {len(boxes)} 个珠子候选"

        # Show box size distribution
        widths = [b["width"] for b in boxes]
        avg_width = np.mean(widths) if widths else 0
        status += f"\n平均尺寸: {avg_width:.1f}px"
    else:
        annotated = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        status = "⚠️ 未检测到珠子，请调整参数"

    # Update dataset stats
    total_imgs = len(_list_bead_photos())
    labeled_imgs = len([f for f in os.listdir(LABELS_DIR) if f.endswith(".txt")]) if os.path.exists(LABELS_DIR) else 0
    stats = f"数据集统计: {labeled_imgs}/{total_imgs} 已标注"

    return annotated, status, "", stats


def _handle_export():
    """Export current detection results to YOLO format."""
    global _current_boxes, _current_image, _current_image_path

    if _current_image is None:
        return "❌ 请先运行预标注", ""

    if not _current_boxes:
        return "❌ 没有检测结果，请调整参数后重新预标注", ""

    _ensure_dirs(IMAGES_DIR, LABELS_DIR)

    # Generate label filename
    if _current_image_path:
        basename = os.path.splitext(os.path.basename(_current_image_path))[0]
    else:
        import time
        basename = f"bead_{int(time.time())}"

    # Save image
    img_ext = ".jpg"
    img_path = os.path.join(IMAGES_DIR, basename + img_ext)
    cv2.imwrite(img_path, _current_image)

    # Save labels
    label_path = os.path.join(LABELS_DIR, basename + ".txt")
    h, w = _current_image.shape[:2]
    save_yolo_boxes(_current_boxes, w, h, label_path)

    # Update stats
    total_imgs = len(_list_bead_photos())
    labeled_imgs = len([f for f in os.listdir(LABELS_DIR) if f.endswith(".txt")])
    stats = f"数据集统计: {labeled_imgs}/{total_imgs} 已标注"

    msg = (
        f"✅ 导出成功！\n\n"
        f"检测到 {len(_current_boxes)} 个珠子\n"
        f"图片: {img_path}\n"
        f"标签: {label_path}\n\n"
        f"数据集目录: {DATASET_DIR}"
    )

    return msg, stats


def _handle_load_next():
    """Load next unlabeled photo from bead_photos directory."""
    photos = _list_bead_photos()
    if not photos:
        return None, "❌ 没有照片，请先上传照片到 training/bead_photos/ 目录", "", ""

    # Find first unlabeled photo
    for img_path in photos:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(LABELS_DIR, basename + ".txt")
        if not os.path.exists(label_path):
            img = cv2.imread(img_path)
            if img is not None:
                return (
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                    f"✅ 加载: {os.path.basename(img_path)}",
                    "",
                    f"数据集统计: {len(_list_bead_photos())} 张照片"
                )

    return None, "🎉 所有照片都已标注！", "", f"数据集统计: {len(_list_bead_photos())} 张照片 (全部完成)"


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    """Build Gradio interface for bead annotation."""
    with gr.Blocks(title="Pintconut 珠子标注工具") as app:
        gr.Markdown("# 🔴 Pintconut 珠子标注工具")
        gr.Markdown("使用 HoughCircles 自动检测珠子，导出 YOLO 检测格式训练数据")

        with gr.Row():
            # Left column: Input and Controls
            with gr.Column(scale=1):
                gr.Markdown("## 📤 图片输入")
                image_input = gr.Image(label="上传珠板图片", type="pil")

                gr.Markdown("---")
                gr.Markdown("## ⚙️ HoughCircles 参数")

                dp = gr.Slider(
                    label="dp (分辨率比例)",
                    minimum=1.0,
                    maximum=3.0,
                    step=0.1,
                    value=1.2,
                    info="检测器分辨率，越大越慢但越准确"
                )
                min_dist = gr.Slider(
                    label="min_dist (最小距离)",
                    minimum=5,
                    maximum=50,
                    step=1,
                    value=10,
                    info="珠子中心最小距离"
                )
                param1 = gr.Slider(
                    label="param1 (Canny高阈值)",
                    minimum=50,
                    maximum=200,
                    step=10,
                    value=100,
                    info="边缘检测高阈值"
                )
                param2 = gr.Slider(
                    label="param2 (累加器阈值)",
                    minimum=10,
                    maximum=100,
                    step=5,
                    value=30,
                    info="检测阈值，越小检测越多"
                )
                min_radius = gr.Slider(
                    label="min_radius (最小半径)",
                    minimum=1,
                    maximum=20,
                    step=1,
                    value=3,
                    info="珠子最小半径 (像素)"
                )
                max_radius = gr.Slider(
                    label="max_radius (最大半径)",
                    minimum=5,
                    maximum=50,
                    step=1,
                    value=30,
                    info="珠子最大半径 (像素)"
                )

                prelabel_btn = gr.Button("🔍 预标注", variant="primary", size="lg")
                export_btn = gr.Button("💾 导出数据", variant="secondary", size="lg")
                load_next_btn = gr.Button("⏭️ 加载下一张", variant="secondary")

            # Right column: Results and Status
            with gr.Column(scale=1):
                gr.Markdown("## 📊 检测结果")
                result_image = gr.Image(label="检测结果 (绿色框=珠子)", type="numpy")

                gr.Markdown("---")
                status_text = gr.Textbox(label="状态", interactive=False, lines=3)
                export_status = gr.Textbox(label="导出状态", interactive=False, lines=4)
                dataset_stats = gr.Textbox(label="数据集统计", interactive=False, lines=2)

        # Event handlers
        prelabel_btn.click(
            fn=_handle_prelabel,
            inputs=[image_input, dp, min_dist, param1, param2, min_radius, max_radius],
            outputs=[result_image, status_text, export_status, dataset_stats],
        )

        export_btn.click(
            fn=_handle_export,
            inputs=[],
            outputs=[export_status, dataset_stats],
        )

        load_next_btn.click(
            fn=_handle_load_next,
            inputs=[],
            outputs=[image_input, status_text, export_status, dataset_stats],
        )

        # Initialize dataset stats on load
        app.load(
            fn=lambda: f"数据集统计: 0/{len(_list_bead_photos())} 已标注",
            outputs=[dataset_stats],
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = build_ui()
    app.launch(share=False)
