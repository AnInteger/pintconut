"""Gradio web UI for bead board labeling — 2-tab wizard flow."""

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
    draw_result,
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


def _build_radio_choices(candidates: list[dict]) -> list[str]:
    """Build Radio choice strings from candidate list."""
    choices = []
    for ci, cand in enumerate(candidates):
        area_pct = f"{cand['area_ratio'] * 100:.1f}%"
        choices.append(f"#{ci} 面积 {area_pct}")
    return choices


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------
_segmentation_cache: dict[str, list[dict]] = {}
_annotation_state: dict[str, str | None] = {}
_pending_selection: dict[str, int] = {}


def _init_annotation_state() -> None:
    _annotation_state.clear()
    for img_path in _list_photos():
        _annotation_state[img_path] = None


def _get_annotation_progress() -> tuple[int, int, int]:
    total = len(_annotation_state)
    labeled = sum(1 for v in _annotation_state.values() if v == "labeled")
    skipped = sum(1 for v in _annotation_state.values() if v == "skipped")
    return labeled, skipped, total


def _find_next_unprocessed(current_path: str | None = None) -> int:
    photos = _list_photos()
    start = 0
    if current_path and current_path in photos:
        start = photos.index(current_path) + 1
    for i in range(start, len(photos)):
        if _annotation_state.get(photos[i]) is None:
            return i
    for i in range(0, start):
        if _annotation_state.get(photos[i]) is None:
            return i
    return -1


def _build_thumbnails_html() -> str:
    photos = _list_photos()
    if not photos:
        return ""
    parts = []
    for i, p in enumerate(photos):
        state = _annotation_state.get(p)
        if state == "labeled":
            icon = "✅"
        elif state == "skipped":
            icon = "⏭️"
        else:
            icon = "🟡"
        name = os.path.basename(p)
        parts.append(f"{icon} {name}")
    return "  |  ".join(parts)


# ---------------------------------------------------------------------------
# Output tuple helper
# ---------------------------------------------------------------------------
def _annotate(image, info, progress, index, radio_update, thumbnails,
              confirm=False, reselect=False, skip=True, export=False):
    """Build 10-element output tuple for annotate_outputs."""
    return (image, info, progress, index, radio_update, thumbnails,
            gr.update(interactive=confirm),
            gr.update(interactive=reselect),
            gr.update(interactive=skip),
            gr.update(interactive=export))


# ---------------------------------------------------------------------------
# Tab ① — Upload
# ---------------------------------------------------------------------------
def _handle_upload(files):
    """Upload photos and return (status_text, gallery_images, btn_update)."""
    if not files:
        return "❌ 未选择文件", [], gr.update(interactive=False)
    _ensure_dirs(PHOTOS_DIR)
    count = 0
    for f in files:
        fpath = f if isinstance(f, str) else getattr(f, "name", str(f))
        basename = os.path.basename(fpath)
        dest = os.path.join(PHOTOS_DIR, basename)
        if os.path.abspath(fpath) != os.path.abspath(dest):
            shutil.copy2(fpath, dest)
        count += 1
    previews = []
    for img_path in _list_photos():
        img = cv2.imread(img_path)
        if img is not None:
            previews.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    total = len(_list_photos())
    return (f"✅ 已上传 {count} 张照片（共 {total} 张）\n\n👉 请点击「② 标注确认」标签开始标注",
            previews,
            gr.update(interactive=True))


# ---------------------------------------------------------------------------
# Tab ② — Annotate handlers
# ---------------------------------------------------------------------------
def _handle_start_annotation(progress=gr.Progress()):
    """Segment all photos and load first image for annotation."""
    photos = _list_photos()
    empty_radio = gr.update(choices=[], interactive=False)
    if not photos:
        return _annotate(None, "❌ 没有照片，请先返回上一步上传照片。", "进度：0/0", 0,
                         empty_radio, "")

    model = _get_model()
    _segmentation_cache.clear()
    n = len(photos)

    for idx, img_path in enumerate(photos):
        progress((idx + 1) / n, desc=f"正在分割 {idx + 1}/{n}")
        img = cv2.imread(img_path)
        if img is None:
            continue
        candidates = segment_image(img, model=model)
        _segmentation_cache[img_path] = candidates

    _init_annotation_state()
    return _load_annotate_image(0)


def _load_annotate_image(index: int) -> tuple:
    """Load image for annotation. Returns 10-element tuple for annotate_outputs."""
    photos = _list_photos()
    empty_radio = gr.update(choices=[], interactive=False)

    if not photos:
        return _annotate(None, "没有照片", "进度：0 已标注，0 跳过，共 0 张", 0,
                         empty_radio, "")

    index = max(0, min(index, len(photos) - 1))
    img_path = photos[index]
    img = cv2.imread(img_path)
    if img is None:
        return _annotate(None, f"无法读取: {os.path.basename(img_path)}", "", index,
                         empty_radio, "")

    candidates = _segmentation_cache.get(img_path, [])
    vis = draw_candidates(img, candidates)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    labeled, skipped, total = _get_annotation_progress()
    progress_text = f"进度：{labeled} 已标注，{skipped} 跳过，共 {total} 张"
    info = f"📷 {os.path.basename(img_path)}（第 {index + 1}/{total} 张）— {len(candidates)} 个候选区域"

    radio_update = gr.update(choices=_build_radio_choices(candidates),
                             interactive=True, value=None)
    return _annotate(vis_rgb, info, progress_text, index, radio_update,
                     _build_thumbnails_html(), skip=True)


def _handle_select_candidate(img_index: int, selected: str):
    """Handle Radio selection: show result preview."""
    photos = _list_photos()
    if not photos or img_index >= len(photos) or not selected:
        return _load_annotate_image(img_index if photos else 0)

    img_path = photos[img_index]
    candidates = _segmentation_cache.get(img_path, [])

    # Parse candidate index from selected string "#N 面积 X.X%"
    try:
        cand_index = int(selected.split("#")[1].split(" ")[0])
    except (IndexError, ValueError):
        return _load_annotate_image(img_index)

    if cand_index >= len(candidates):
        return _load_annotate_image(img_index)

    # Store pending selection and show result preview
    _pending_selection[img_path] = cand_index

    cand = candidates[cand_index]
    mask = cand["mask"]
    img = cv2.imread(img_path)
    if img is None:
        return _load_annotate_image(img_index)

    vis = draw_result(img, mask)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    labeled, skipped, total = _get_annotation_progress()
    progress_text = f"进度：{labeled} 已标注，{skipped} 跳过，共 {total} 张"
    info = (f"📷 {os.path.basename(img_path)}（第 {img_index + 1}/{total} 张）\n\n"
            f"✅ 已选择区域 {cand_index}，请确认或重新选择")
    radio_update = gr.update(value=selected, interactive=False)

    return _annotate(vis_rgb, info, progress_text, img_index, radio_update,
                     _build_thumbnails_html(), confirm=True, reselect=True, skip=False)


def _handle_confirm(img_index: int):
    """Confirm pending selection: save label and advance to next image."""
    photos = _list_photos()
    if not photos or img_index >= len(photos):
        return _load_annotate_image(0)

    img_path = photos[img_index]
    cand_index = _pending_selection.pop(img_path, None)
    if cand_index is None:
        return _load_annotate_image(img_index)

    candidates = _segmentation_cache.get(img_path, [])
    if cand_index >= len(candidates):
        return _load_annotate_image(img_index)

    cand = candidates[cand_index]
    mask = cand["mask"]

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    img_dir = os.path.join(DATASET_DIR, "images", "train")
    lbl_dir = os.path.join(DATASET_DIR, "labels", "train")
    _ensure_dirs(img_dir, lbl_dir)
    label_path = save_label(mask, img_path, img_dir, lbl_dir, w, h)

    if label_path:
        _annotation_state[img_path] = "labeled"
    else:
        _annotation_state[img_path] = "skipped"

    next_idx = _find_next_unprocessed(img_path)
    if next_idx < 0:
        labeled, skipped, total = _get_annotation_progress()
        empty_radio = gr.update(choices=[], interactive=False)
        return _annotate(None, "🎉 全部处理完成！",
                         f"进度：{labeled} 已标注，{skipped} 跳过，共 {total} 张",
                         img_index, empty_radio, _build_thumbnails_html(),
                         skip=False, export=True)

    return _load_annotate_image(next_idx)


def _handle_reselect(img_index: int):
    """Cancel pending selection and return to candidate view."""
    photos = _list_photos()
    if not photos or img_index >= len(photos):
        return _load_annotate_image(0)

    img_path = photos[img_index]
    _pending_selection.pop(img_path, None)

    return _load_annotate_image(img_index)


def _handle_skip_photo(img_index: int):
    photos = _list_photos()
    if not photos or img_index >= len(photos):
        return _load_annotate_image(0)

    img_path = photos[img_index]
    _annotation_state[img_path] = "skipped"

    next_idx = _find_next_unprocessed(img_path)
    if next_idx < 0:
        labeled, skipped, total = _get_annotation_progress()
        empty_radio = gr.update(choices=[], interactive=False)
        return _annotate(None, "🎉 全部处理完成！",
                         f"进度：{labeled} 已标注，{skipped} 跳过，共 {total} 张",
                         img_index, empty_radio, _build_thumbnails_html(),
                         skip=False, export=True)

    return _load_annotate_image(next_idx)


def _handle_export():
    _ensure_dirs(
        os.path.join(DATASET_DIR, "images", "train"),
        os.path.join(DATASET_DIR, "images", "valid"),
        os.path.join(DATASET_DIR, "labels", "train"),
        os.path.join(DATASET_DIR, "labels", "valid"),
    )
    yaml_path = generate_dataset_yaml(DATASET_DIR)
    labeled, skipped, total = _get_annotation_progress()
    train_imgs = len([
        f for f in os.listdir(os.path.join(DATASET_DIR, "images", "train"))
        if os.path.isfile(os.path.join(DATASET_DIR, "images", "train", f))
    ])
    train_lbls = len([
        f for f in os.listdir(os.path.join(DATASET_DIR, "labels", "train"))
        if os.path.isfile(os.path.join(DATASET_DIR, "labels", "train", f))
    ])
    report = (
        f"📊 数据集统计\n\n"
        f"照片总数：{total}\n"
        f"已标注：{labeled}\n"
        f"跳过：{skipped}\n\n"
        f"训练图片：{train_imgs}\n"
        f"训练标签：{train_lbls}\n\n"
        f"数据集路径：{DATASET_DIR}\n"
        f"配置文件：{yaml_path}\n\n"
        f"下一步 — 训练模型：\n"
        f"  python training/train.py --data training/dataset/data.yaml"
    )
    return report


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Pintconut 拼豆标注工具") as app:
        gr.Markdown("# 🫘 Pintconut 拼豆标注工具")

        with gr.Tabs():
            # ==== Tab 1: Upload ====
            with gr.Tab("① 上传照片"):
                gr.Markdown("上传拼板照片，用于训练拼板检测模型。")
                file_input = gr.File(label="选择照片（支持多选）",
                                     file_count="multiple", file_types=["image"])
                upload_btn = gr.Button("📤 上传", variant="primary")
                upload_status = gr.Textbox(label="状态", interactive=False, lines=3)
                upload_gallery = gr.Gallery(label="已上传照片", columns=6, height="auto")
                start_annotation_btn = gr.Button("▶ 开始标注", variant="primary",
                                                 interactive=False)

                upload_btn.click(
                    fn=_handle_upload,
                    inputs=[file_input],
                    outputs=[upload_status, upload_gallery, start_annotation_btn],
                )

            # ==== Tab 2: Annotate ====
            with gr.Tab("② 标注确认"):
                gr.Markdown(
                    "FastSAM 自动识别了多个区域，你需要选出**哪个区域是拼板**。\n"
                    "选中的结果将作为训练数据，教会 AI 自动识别拼板位置。"
                )
                annotate_progress = gr.Markdown("进度：0 已标注，0 跳过，共 0 张")
                thumbnail_strip = gr.Markdown("")

                with gr.Row():
                    with gr.Column(scale=1):
                        annotate_info = gr.Markdown("📷 点击下方「开始标注」加载第一张照片")
                        review_image = gr.Image(label="当前照片", type="numpy", height=400)
                    with gr.Column(scale=1):
                        cand_radio = gr.Radio(
                            choices=[], label="候选区域（点击选择拼板所在的区域）",
                            interactive=False,
                        )
                        gr.Markdown("---")
                        skip_btn = gr.Button("⏭️ 跳过这张", variant="secondary")
                        confirm_btn = gr.Button("✅ 确认，下一张", variant="primary",
                                                interactive=False)
                        reselect_btn = gr.Button("↩️ 重新选择", variant="secondary",
                                                 interactive=False)

                annotate_current_idx = gr.State(value=0)
                init_annotate_btn = gr.Button("▶ 开始标注", variant="primary")
                complete_export_btn = gr.Button("📦 导出数据集", variant="primary",
                                                interactive=False)
                export_report = gr.Textbox(label="数据集报告", interactive=False, lines=12)

        # ===== Wiring =====

        # annotate_outputs: 10 elements
        annotate_outputs = [
            review_image, annotate_info, annotate_progress,
            annotate_current_idx,
            cand_radio, thumbnail_strip,
            confirm_btn, reselect_btn, skip_btn, complete_export_btn,
        ]

        # "开始标注" → segment all + load first image
        init_annotate_btn.click(
            fn=_handle_start_annotation,
            inputs=[],
            outputs=annotate_outputs,
        )

        # Radio selection → show preview
        cand_radio.change(
            fn=lambda selected, idx: _handle_select_candidate(int(idx), selected),
            inputs=[cand_radio, annotate_current_idx],
            outputs=annotate_outputs,
        )

        # Confirm → save and next
        confirm_btn.click(
            fn=lambda idx: _handle_confirm(int(idx)),
            inputs=[annotate_current_idx],
            outputs=annotate_outputs,
        )

        # Reselect → back to candidate view
        reselect_btn.click(
            fn=lambda idx: _handle_reselect(int(idx)),
            inputs=[annotate_current_idx],
            outputs=annotate_outputs,
        )

        # Skip → mark skipped and next
        skip_btn.click(
            fn=lambda idx: _handle_skip_photo(int(idx)),
            inputs=[annotate_current_idx],
            outputs=annotate_outputs,
        )

        # Export → generate report
        complete_export_btn.click(
            fn=_handle_export,
            inputs=[],
            outputs=[export_report],
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    app = build_ui()
    app.launch(share=False, server_port=port)
