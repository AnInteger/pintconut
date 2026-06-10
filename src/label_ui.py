"""Gradio web UI for bead board labeling — wizard flow with auto-advancing tabs."""

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

# Emojis matching COLORS order in label_service.py
COLOR_EMOJIS = ["🟢", "🔴", "🔵", "🟡", "🟦", "🟣", "🟩", "🔶", "🟠", "🟤"]


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
# Step 1 — Upload
# ---------------------------------------------------------------------------
def _handle_upload(files):
    """Upload photos and return (status_text, gallery_images)."""
    if not files:
        return "❌ 未选择文件", []
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
    return f"✅ 已上传 {count} 张照片（共 {total} 张）\n\n👉 点击上方「② 自动分割」标签继续", previews


# ---------------------------------------------------------------------------
# Step 2 — Segment
# ---------------------------------------------------------------------------
def _handle_segment(progress=gr.Progress()):
    """Segment all photos, store results, return (gallery_images, status_text)."""
    photos = _list_photos()
    if not photos:
        return [], "❌ 没有照片，请先返回上一步上传照片。"

    model = _get_model()
    annotated: list[np.ndarray] = []
    n = len(photos)

    for idx, img_path in enumerate(photos):
        progress((idx + 1) / n, desc=f"正在分割 {idx + 1}/{n}")
        img = cv2.imread(img_path)
        if img is None:
            continue
        candidates = segment_image(img, model=model)
        vis = draw_candidates(img, candidates)
        vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        annotated.append(vis)

    # Store candidate data for Step 3
    _store_segmentation_results(photos, model)

    msg = f"✅ 已分割 {len(annotated)} 张照片"
    if len(annotated) < n:
        msg += f"（{n - len(annotated)} 张无法读取）"
    msg += "\n\n👉 点击上方「③ 标注确认」标签继续"
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
# Annotation state management
# ---------------------------------------------------------------------------
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


def _step3(image, info, progress, index, btn_updates, thumbnails,
           confirm=False, reselect=False, export=False):
    """Build a flat 18-element tuple for step3_outputs."""
    return (image, info, progress, index, *btn_updates, thumbnails,
            gr.update(visible=confirm),
            gr.update(visible=reselect),
            gr.update(visible=export))


# ---------------------------------------------------------------------------
# Step 3 — Review / Annotate
# ---------------------------------------------------------------------------
def _load_review_image(index: int) -> tuple:
    """Load image for review. Returns 18-element tuple for step3_outputs."""
    photos = _list_photos()
    empty_btns = [gr.update(visible=False, value="") for _ in range(10)]

    if not photos:
        return _step3(None, "没有照片", "进度：0 已标注，0 跳过，共 0 张", 0,
                      empty_btns, "")

    index = max(0, min(index, len(photos) - 1))
    img_path = photos[index]
    img = cv2.imread(img_path)
    if img is None:
        return _step3(None, f"无法读取: {os.path.basename(img_path)}", "", index,
                      empty_btns, "")

    candidates = _segmentation_cache.get(img_path, [])
    vis = draw_candidates(img, candidates)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    labeled, skipped, total = _get_annotation_progress()
    progress_text = f"进度：{labeled} 已标注，{skipped} 跳过，共 {total} 张"
    info = f"📷 {os.path.basename(img_path)}（第 {index + 1}/{total} 张）— {len(candidates)} 个候选区域"

    btn_updates = []
    for ci in range(10):
        if ci < len(candidates):
            area_pct = f"{candidates[ci]['area_ratio'] * 100:.1f}%"
            emoji = COLOR_EMOJIS[ci % len(COLOR_EMOJIS)]
            btn_updates.append(gr.update(value=f"{emoji} #{ci} 面积 {area_pct}", visible=True))
        else:
            btn_updates.append(gr.update(visible=False, value=""))

    return _step3(vis_rgb, info, progress_text, index, btn_updates,
                  _build_thumbnails_html())


def _handle_select_candidate(img_index: int, cand_index: int):
    photos = _list_photos()
    if not photos or img_index >= len(photos):
        return _load_review_image(0)

    img_path = photos[img_index]
    candidates = _segmentation_cache.get(img_path, [])
    if cand_index >= len(candidates):
        return _load_review_image(img_index)

    # Store pending selection and show result preview
    _pending_selection[img_path] = cand_index

    cand = candidates[cand_index]
    mask = cand["mask"]
    img = cv2.imread(img_path)
    if img is None:
        return _load_review_image(img_index)

    vis = draw_result(img, mask)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    labeled, skipped, total = _get_annotation_progress()
    progress_text = f"进度：{labeled} 已标注，{skipped} 跳过，共 {total} 张"
    info = f"📷 {os.path.basename(img_path)}（第 {img_index + 1}/{total} 张）\n\n✅ 已选择区域 {cand_index}，请确认或重新选择"
    empty_btns = [gr.update(visible=False, value="") for _ in range(10)]

    return _step3(vis_rgb, info, progress_text, img_index, empty_btns,
                  _build_thumbnails_html(), confirm=True, reselect=True)


def _handle_confirm(img_index: int):
    """Confirm pending selection: save label and advance to next image."""
    photos = _list_photos()
    if not photos or img_index >= len(photos):
        return _load_review_image(0)

    img_path = photos[img_index]
    cand_index = _pending_selection.pop(img_path, None)
    if cand_index is None:
        return _load_review_image(img_index)

    candidates = _segmentation_cache.get(img_path, [])
    if cand_index >= len(candidates):
        return _load_review_image(img_index)

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
        empty_btns = [gr.update(visible=False, value="") for _ in range(10)]
        return _step3(None, "🎉 全部处理完成！",
                      f"进度：{labeled} 已标注，{skipped} 跳过，共 {total} 张",
                      img_index, empty_btns, _build_thumbnails_html(),
                      export=True)

    return _load_review_image(next_idx)


def _handle_reselect(img_index: int):
    """Cancel pending selection and return to candidate view."""
    photos = _list_photos()
    if not photos or img_index >= len(photos):
        return _load_review_image(0)

    img_path = photos[img_index]
    _pending_selection.pop(img_path, None)

    return _load_review_image(img_index)


def _handle_skip_photo(img_index: int):
    photos = _list_photos()
    if not photos or img_index >= len(photos):
        return _load_review_image(0)

    img_path = photos[img_index]
    _annotation_state[img_path] = "skipped"

    next_idx = _find_next_unprocessed(img_path)
    if next_idx < 0:
        labeled, skipped, total = _get_annotation_progress()
        empty_btns = [gr.update(visible=False, value="") for _ in range(10)]
        return _step3(None, "🎉 全部处理完成！",
                      f"进度：{labeled} 已标注，{skipped} 跳过，共 {total} 张",
                      img_index, empty_btns, _build_thumbnails_html(),
                      export=True)

    return _load_review_image(next_idx)


# ---------------------------------------------------------------------------
# Step 4 — Export
# ---------------------------------------------------------------------------
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
# JS helper to switch Gradio tab by index (0-based)
# ---------------------------------------------------------------------------


def _js_goto_tab(idx: int) -> str:
    """Return JS snippet that clicks the Nth tab (0-based) and returns empty string for dummy output."""
    return f"""
    () => {{
        const tabs = document.querySelectorAll('button[role="tab"]');
        if (tabs[{idx}]) tabs[{idx}].click();
        return [""];
    }}
    """


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Pintconut 拼豆标注工具") as app:
        gr.Markdown("# 🫘 Pintconut 拼豆标注工具")

        # Hidden textbox used as dummy output for JS-only navigation buttons
        _nav_dummy = gr.Textbox(visible=False)

        with gr.Tabs():
            # ==== Tab 1: Upload ====
            with gr.Tab("① 上传照片"):
                gr.Markdown("上传拼板照片，用于训练拼板检测模型。")
                file_input = gr.File(label="选择照片（支持多选）", file_count="multiple", file_types=["image"])
                upload_btn = gr.Button("📤 上传", variant="primary")
                upload_status = gr.Textbox(label="状态", interactive=False, lines=3)
                upload_gallery = gr.Gallery(label="已上传照片", columns=6, height="auto")
                nav_next_1 = gr.Button("下一步 →", variant="primary")

                upload_btn.click(
                    fn=_handle_upload,
                    inputs=[file_input],
                    outputs=[upload_status, upload_gallery],
                )
                nav_next_1.click(
                    fn=lambda: "",
                    js=_js_goto_tab(1),
                    outputs=[_nav_dummy],
                )

            # ==== Tab 2: Segment ====
            with gr.Tab("② 自动分割"):
                gr.Markdown("FastSAM 将自动识别每张照片中的候选区域。")
                segment_btn = gr.Button("🔍 开始分割", variant="primary")
                segment_gallery = gr.Gallery(label="分割结果", columns=3, height="auto")
                segment_status = gr.Textbox(label="状态", interactive=False, lines=3)

                segment_btn.click(
                    fn=_handle_segment,
                    inputs=[],
                    outputs=[segment_gallery, segment_status],
                )

                nav_next_2 = gr.Button("下一步 →", variant="primary")

            # ==== Tab 3: Annotate ====
            with gr.Tab("③ 标注确认"):
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
                        gr.Markdown("**候选区域**（点击选择拼板所在的区域）")
                        cand_buttons = []
                        for ci in range(10):
                            btn = gr.Button(f"候选 {ci}", visible=False)
                            cand_buttons.append(btn)
                        gr.Markdown("---")
                        skip_btn = gr.Button("⏭️ 跳过这张", variant="secondary")
                        confirm_btn = gr.Button("✅ 确认，下一张", variant="primary", visible=False)
                        reselect_btn = gr.Button("↩️ 重新选择", variant="secondary", visible=False)

                annotate_current_idx = gr.State(value=0)

                init_annotate_btn = gr.Button("▶ 开始标注", variant="primary")
                complete_export_btn = gr.Button("📦 导出数据集", variant="primary", visible=False)

            # ==== Tab 4: Export ====
            with gr.Tab("④ 导出数据集"):
                gr.Markdown("生成 YOLO 格式数据集，用于训练模型。")
                export_btn = gr.Button("📦 导出数据集", variant="primary")
                export_status = gr.Textbox(label="数据集报告", interactive=False, lines=12)
                export_btn.click(fn=_handle_export, inputs=[], outputs=[export_status])

        # ===== Wiring =====

        # Step 2 → Step 3: init annotation state + load first image + switch tab
        step3_outputs = [review_image, annotate_info, annotate_progress,
                         annotate_current_idx] + cand_buttons + [thumbnail_strip,
                         confirm_btn, reselect_btn, complete_export_btn]

        def _enter_step3():
            _init_annotation_state()
            return _load_review_image(0)

        nav_next_2.click(
            fn=_enter_step3,
            inputs=[],
            outputs=step3_outputs,
            js=_js_goto_tab(2),
        )

        # "开始标注" button also loads first image (stays on same tab)
        init_annotate_btn.click(
            fn=_enter_step3,
            inputs=[],
            outputs=step3_outputs,
        )

        # Candidate button clicks
        for ci, btn in enumerate(cand_buttons):
            def _make_handler(cand_idx):
                def handler(current_idx):
                    return _handle_select_candidate(int(current_idx), cand_idx)
                return handler
            btn.click(
                fn=_make_handler(ci),
                inputs=[annotate_current_idx],
                outputs=step3_outputs,
            )

        # Skip button
        skip_btn.click(
            fn=lambda idx: _handle_skip_photo(int(idx)),
            inputs=[annotate_current_idx],
            outputs=step3_outputs,
        )

        # Confirm and reselect buttons
        confirm_btn.click(
            fn=lambda idx: _handle_confirm(int(idx)),
            inputs=[annotate_current_idx],
            outputs=step3_outputs,
        )
        reselect_btn.click(
            fn=lambda idx: _handle_reselect(int(idx)),
            inputs=[annotate_current_idx],
            outputs=step3_outputs,
        )

        # Complete -> export button
        complete_export_btn.click(
            fn=lambda: "",
            js=_js_goto_tab(3),
            outputs=[_nav_dummy],
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = build_ui()
    app.launch(share=False)
