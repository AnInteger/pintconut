"""Gradio web UI for bead board labeling — 4-step wizard flow."""

import os
import shutil
import sys

# Ensure project root is on the path so `src.*` imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import gradio as gr
import numpy as np

from src.label_service import (
    draw_candidates,
    generate_dataset_yaml,
    save_label,
    segment_image,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHOTOS_DIR = os.path.join(BASE_DIR, "training", "photos")
DATASET_DIR = os.path.join(BASE_DIR, "training", "dataset")


# ---------------------------------------------------------------------------
# Lazy-loaded FastSAM model (singleton)
# ---------------------------------------------------------------------------
_model = None


def _get_model():
    global _model
    if _model is None:
        from ultralytics import FastSAM

        _model = FastSAM("FastSAM-s.pt")
    return _model


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _list_photos() -> list[str]:
    """Return sorted list of image file paths in PHOTOS_DIR."""
    if not os.path.isdir(PHOTOS_DIR):
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(
        os.path.join(PHOTOS_DIR, f)
        for f in os.listdir(PHOTOS_DIR)
        if os.path.splitext(f)[1].lower() in exts
    )


def _ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Tab 1 — Upload
# ---------------------------------------------------------------------------
def _handle_upload(files: list[str] | None) -> str:
    if not files:
        return "No files selected."
    _ensure_dirs(PHOTOS_DIR)
    count = 0
    for fpath in files:
        # fpath may be a tempfile path; copy to photos dir
        basename = os.path.basename(fpath)
        dest = os.path.join(PHOTOS_DIR, basename)
        if os.path.abspath(fpath) != os.path.abspath(dest):
            shutil.copy2(fpath, dest)
        count += 1
    total = len(_list_photos())
    return f"Uploaded {count} file(s). Total photos: {total}"


# ---------------------------------------------------------------------------
# Tab 2 — Segment
# ---------------------------------------------------------------------------
def _handle_segment(progress=gr.Progress()):
    photos = _list_photos()
    if not photos:
        return [], "No photos found. Upload photos first."

    model = _get_model()
    annotated: list[np.ndarray] = []
    n = len(photos)

    for idx, img_path in enumerate(photos):
        progress((idx + 1) / n, desc=f"Segmenting {idx + 1}/{n}")
        img = cv2.imread(img_path)
        if img is None:
            continue
        candidates = segment_image(img, model=model)
        vis = draw_candidates(img, candidates)
        vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        annotated.append(vis)

    # Store candidate data in sidecar for Tab 3 use
    _store_segmentation_results(photos, model)

    msg = f"Segmented {len(annotated)} image(s)."
    if len(annotated) < n:
        msg += f" ({n - len(annotated)} could not be read.)"
    return annotated, msg


# Sidecar storage: map image_path -> list of candidate dicts
_segmentation_cache: dict[str, list[dict]] = {}


def _store_segmentation_results(photos: list[str], model) -> None:
    _segmentation_cache.clear()
    for img_path in photos:
        img = cv2.imread(img_path)
        if img is None:
            continue
        candidates = segment_image(img, model=model)
        _segmentation_cache[img_path] = candidates


# ---------------------------------------------------------------------------
# Tab 3 — Confirm / skip
# ---------------------------------------------------------------------------
def _load_image_for_review(index: int):
    """Load the image at *index* and return (rgb_image, candidate_gallery, info_text)."""
    photos = _list_photos()
    if not photos:
        return None, [], "No photos available."
    index = max(0, min(index, len(photos) - 1))
    img_path = photos[index]
    img = cv2.imread(img_path)
    if img is None:
        return None, [], f"Cannot read image: {img_path}"
    candidates = _segmentation_cache.get(img_path, [])
    vis = draw_candidates(img, candidates)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    cand_previews = []
    h, w = img.shape[:2]
    for ci, cand in enumerate(candidates):
        overlay = img.copy()
        color = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)][ci % 5]
        mask = cand["mask"]
        ov = overlay.copy()
        ov[mask > 0] = color
        cv2.addWeighted(ov, 0.45, overlay, 0.55, 0, overlay)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)
        cand_previews.append(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

    info = f"Image {index + 1}/{len(photos)} — {len(candidates)} candidate(s) — {os.path.basename(img_path)}"
    return vis_rgb, cand_previews, info


def _handle_confirm(img_index: int, cand_index: int):
    """Confirm a candidate and save its label."""
    photos = _list_photos()
    if not photos:
        return img_index, None, [], "No photos."
    img_index = max(0, min(img_index, len(photos) - 1))
    img_path = photos[img_index]
    candidates = _segmentation_cache.get(img_path, [])
    if not candidates:
        return img_index, None, [], "No candidates for this image."

    cand_index = max(0, min(cand_index, len(candidates) - 1))
    cand = candidates[cand_index]
    mask = cand["mask"]

    img = cv2.imread(img_path)
    if img is None:
        return img_index, None, [], "Cannot read image."
    h, w = img.shape[:2]

    # Save into training/dataset directories
    img_dir = os.path.join(DATASET_DIR, "images", "train")
    lbl_dir = os.path.join(DATASET_DIR, "labels", "train")
    _ensure_dirs(img_dir, lbl_dir)

    label_path = save_label(mask, img_path, img_dir, lbl_dir, w, h)
    if label_path:
        msg = f"Saved label: {os.path.basename(label_path)} (candidate {cand_index})"
    else:
        msg = f"Failed to save label for candidate {cand_index} (no valid contour)."

    # Move to next image
    next_idx = min(img_index + 1, len(photos) - 1)
    vis, cand_previews, info = _load_image_for_review(next_idx)
    return next_idx, vis, cand_previews, msg + "\n" + info


def _handle_skip(img_index: int):
    """Skip to the next image."""
    photos = _list_photos()
    if not photos:
        return img_index, None, [], "No photos."
    next_idx = min(img_index + 1, len(photos) - 1)
    vis, cand_previews, info = _load_image_for_review(next_idx)
    return next_idx, vis, cand_previews, f"Skipped image {img_index + 1}.\n" + info


def _handle_goto(img_index: int):
    """Jump to the specified image index (0-based)."""
    vis, cand_previews, info = _load_image_for_review(img_index)
    return img_index, vis, cand_previews, info


# ---------------------------------------------------------------------------
# Tab 4 — Export
# ---------------------------------------------------------------------------
def _handle_export():
    _ensure_dirs(
        os.path.join(DATASET_DIR, "images", "train"),
        os.path.join(DATASET_DIR, "images", "valid"),
        os.path.join(DATASET_DIR, "labels", "train"),
        os.path.join(DATASET_DIR, "labels", "valid"),
    )
    yaml_path = generate_dataset_yaml(DATASET_DIR)

    # Count items
    train_imgs = len(
        [
            f
            for f in os.listdir(os.path.join(DATASET_DIR, "images", "train"))
            if os.path.isfile(os.path.join(DATASET_DIR, "images", "train", f))
        ]
    )
    train_lbls = len(
        [
            f
            for f in os.listdir(os.path.join(DATASET_DIR, "labels", "train"))
            if os.path.isfile(os.path.join(DATASET_DIR, "labels", "train", f))
        ]
    )

    report = (
        f"Dataset exported to: {DATASET_DIR}\n"
        f"Config: {yaml_path}\n\n"
        f"Training images: {train_imgs}\n"
        f"Training labels: {train_lbls}\n\n"
        f"Next step — train with:\n"
        f"  python training/train.py\n"
    )
    return report


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Pintconut 拼豆标注工具") as app:
        gr.Markdown("# Pintconut 拼豆标注工具")

        # Shared state
        current_index = gr.State(value=0)

        # ---- Tab 1: Upload ----
        with gr.Tab("1. 上传照片"):
            gr.Markdown("Upload bead board photos to `training/photos/`.")
            file_input = gr.File(label="Select photos", file_count="multiple", file_types=["image"])
            upload_btn = gr.Button("Upload", variant="primary")
            upload_status = gr.Textbox(label="Status", interactive=False)
            upload_btn.click(fn=_handle_upload, inputs=[file_input], outputs=[upload_status])

        # ---- Tab 2: Segment ----
        with gr.Tab("2. 自动分割"):
            gr.Markdown("Run FastSAM segmentation on all uploaded photos.")
            segment_btn = gr.Button("开始分割", variant="primary")
            segment_gallery = gr.Gallery(label="Segmented images", columns=3, height="auto")
            segment_status = gr.Textbox(label="Status", interactive=False)
            segment_btn.click(
                fn=_handle_segment,
                inputs=[],
                outputs=[segment_gallery, segment_status],
            )

        # ---- Tab 3: Confirm ----
        with gr.Tab("3. 逐张确认"):
            gr.Markdown("Review each image and confirm or skip candidates.")
            with gr.Row():
                img_idx_input = gr.Number(label="Image index (0-based)", value=0, precision=0)
                cand_idx_input = gr.Number(label="Candidate index", value=0, precision=0)
            with gr.Row():
                goto_btn = gr.Button("Go to image")
                confirm_btn = gr.Button("Confirm", variant="primary")
                skip_btn = gr.Button("Skip")
            review_image = gr.Image(label="Current image with candidates", type="numpy")
            cand_gallery = gr.Gallery(label="Individual candidates", columns=5, height="auto")
            confirm_status = gr.Textbox(label="Status", interactive=False)

            goto_btn.click(
                fn=_handle_goto,
                inputs=[img_idx_input],
                outputs=[current_index, review_image, cand_gallery, confirm_status],
            )
            confirm_btn.click(
                fn=_handle_confirm,
                inputs=[img_idx_input, cand_idx_input],
                outputs=[current_index, review_image, cand_gallery, confirm_status],
            )
            skip_btn.click(
                fn=_handle_skip,
                inputs=[img_idx_input],
                outputs=[current_index, review_image, cand_gallery, confirm_status],
            )

        # ---- Tab 4: Export ----
        with gr.Tab("4. 导出数据集"):
            gr.Markdown("Generate YOLO-format dataset config and review statistics.")
            export_btn = gr.Button("导出数据集", variant="primary")
            export_status = gr.Textbox(label="Dataset report", interactive=False, lines=10)
            export_btn.click(fn=_handle_export, inputs=[], outputs=[export_status])

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = build_ui()
    app.launch(share=False)
