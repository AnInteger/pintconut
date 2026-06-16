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
    SEG_PRESETS,
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
    """Return the photos available for annotation.

    This session's uploads take precedence over the persistent training/photos
    directory, so the upload gallery and the annotation flow only reflect what
    was uploaded now — not photos accumulated from earlier sessions or test
    runs (which previously made the gallery show many stale entries).
    """
    if _session_uploads:
        return list(_session_uploads)
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
# Which segmentation preset produced the cached result for each photo, so the
# UI can show it and so re-segment can pick the other preset. 「标准」/「宽松」.
_seg_mode_used: dict[str, str] = {}
# Paths uploaded in the CURRENT session. Takes precedence over the persistent
# photos directory (see _list_photos) so the upload gallery and annotation flow
# only ever reflect what was uploaded now — not photos left over from earlier
# sessions or test runs.
_session_uploads: list[str] = []


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
def _load_image(path: str):
    """Read an image as a BGR ndarray, returning None if unreadable.

    Tries OpenCV first, then falls back to Pillow, which decodes some JPEGs /
    TIFFs / non-ASCII-path files that cv2.imread rejects. Used during upload so
    corrupt or non-image files can be detected and reported instead of silently
    dropped (which previously made the gallery count not match reality).
    """
    img = cv2.imread(path)
    if img is not None:
        return img
    try:
        from PIL import Image

        with Image.open(path) as im:
            rgb = np.array(im.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def _handle_upload(files):
    """Upload photos and return (status_text, gallery_images, btn_update).

    Each file is validated (must be a readable image) before being accepted.
    Unreadable or corrupt files are skipped and reported by name in the status,
    rather than silently counted — so the gallery reflects only valid photos
    and the user knows which file failed (e.g. NYQC4978.JPG that isn't a real
    JPEG).
    """
    if not files:
        return "❌ 未选择文件", [], gr.update(interactive=False)
    _ensure_dirs(PHOTOS_DIR)
    _session_uploads.clear()
    previews = []
    failed = []
    for f in files:
        fpath = f if isinstance(f, str) else getattr(f, "name", str(f))
        basename = os.path.basename(fpath)
        dest = os.path.join(PHOTOS_DIR, basename)
        img = _load_image(fpath)
        if img is None:
            failed.append(basename)
            continue
        if os.path.abspath(fpath) != os.path.abspath(dest):
            shutil.copy2(fpath, dest)
        _session_uploads.append(dest)
        previews.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    ok = len(previews)
    if ok == 0:
        names = "\n".join(f"   - {n}" for n in failed)
        return f"❌ 没有可用的图片，以下 {len(failed)} 个文件无法读取：\n{names}", [], \
            gr.update(interactive=False)

    lines = [f"✅ 已上传 {ok} 张照片"]
    if failed:
        lines.append(f"⚠ 跳过 {len(failed)} 张无法读取的文件：")
        lines.extend(f"   - {n}" for n in failed)
    lines += ["", "👉 请点击下方「▶ 开始标注」按钮开始标注"]
    return "\n".join(lines), previews, gr.update(interactive=True)


# ---------------------------------------------------------------------------
# Tab ② — Annotate handlers
# ---------------------------------------------------------------------------
def _segment_with(img_path: str, mode: str, progress=None) -> list[dict]:
    """Segment a photo with a given preset and cache the result.

    ``mode`` is a key in SEG_PRESETS (「标准」/「宽松」). Overwrites any cached
    result for this photo, so this is what the manual re-segment buttons call
    to force a fresh pass at a different setting.
    """
    img = _load_image(img_path)
    if img is None:
        _segmentation_cache[img_path] = []
        _seg_mode_used[img_path] = mode
        return []
    if progress is not None:
        progress(0.5, desc=f"正在分割（{mode}参数）{os.path.basename(img_path)} …")
    model = _get_model()
    cands = segment_image(img, model=model, **SEG_PRESETS[mode])
    if progress is not None:
        progress(1.0, desc="完成")
    _segmentation_cache[img_path] = cands
    _seg_mode_used[img_path] = mode
    return cands


def _ensure_segmented(img_path: str, progress=None) -> list[dict]:
    """Lazily segment a photo on first view and cache the result.

    Tries 「标准」 first. If that yields 0 candidates — common on dense,
    full-frame bead photos that FastSAM shreds into single-bead shards —
    automatically retries once with 「宽松」 params, which can recover the whole
    board. The user can also manually switch presets via the re-segment buttons.
    """
    if img_path not in _segmentation_cache:
        cands = _segment_with(img_path, "标准", progress=progress)
        if not cands:
            cands = _segment_with(img_path, "宽松", progress=progress)
    return _segmentation_cache[img_path]


def _handle_start_annotation_and_switch():
    """开始标注 段1（立即反馈）：切到「② 标注确认」+ 显示 loading 占位。

    分割一张照片要数秒。若在同一个回调里把分割做完才返回，点击后这几秒图片
    区会一直空白、标签页也不切换——用户会以为"点了没反应 / 没分割"。所以这里
    只做不耗时的准备（清缓存、切 tab、显示「正在分割」提示），真正的分割交给
    紧随其后的 .then() 链（_load_first_annotate）异步完成，用户一点击就能看
    到反馈。

    返回 11 元素 = Tab 切换 + annotate_outputs(10)，供 Tab1/Tab2 两个「开始
    标注」按钮共用。
    """
    photos = _list_photos()
    empty_radio = gr.update(choices=[], interactive=False)
    if not photos:
        return (gr.Tabs(selected="annotate"),) + _annotate(
            None, "❌ 没有照片，请先返回上一步上传照片。", "进度：0/0", 0,
            empty_radio, "")

    _segmentation_cache.clear()
    _init_annotation_state()
    total = len(photos)
    return (gr.Tabs(selected="annotate"),) + _annotate(
        None,
        f"⏳ 正在分割第 1 张照片，请稍候…（共 {total} 张）",
        f"进度：0 已标注，0 跳过，共 {total} 张",
        0, empty_radio, _build_thumbnails_html())


def _load_first_annotate(progress=gr.Progress()):
    """开始标注 段2（实际分割）：分割并加载第一张照片，显示候选区域。

    由 _handle_start_annotation_and_switch 的 .then() 链触发，确保段1 的
    loading 占位先渲染出来后再做耗时的分割。返回 10 元素 = annotate_outputs。
    """
    return _load_annotate_image(0, progress=progress)


def _load_annotate_image(index: int, progress=None) -> tuple:
    """Load image for annotation. Returns 10-element tuple for annotate_outputs."""
    photos = _list_photos()
    empty_radio = gr.update(choices=[], interactive=False)

    if not photos:
        return _annotate(None, "没有照片", "进度：0 已标注，0 跳过，共 0 张", 0,
                         empty_radio, "")

    index = max(0, min(index, len(photos) - 1))
    img_path = photos[index]
    img = _load_image(img_path)
    if img is None:
        return _annotate(None, f"无法读取: {os.path.basename(img_path)}", "", index,
                         empty_radio, "")

    candidates = _ensure_segmented(img_path, progress=progress)
    vis = draw_candidates(img, candidates)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    labeled, skipped, total = _get_annotation_progress()
    progress_text = f"进度：{labeled} 已标注，{skipped} 跳过，共 {total} 张"
    mode = _seg_mode_used.get(img_path, "标准")
    if candidates:
        info = (f"📷 {os.path.basename(img_path)}（第 {index + 1}/{total} 张）— "
                f"{len(candidates)} 个候选区域（{mode}参数）")
    else:
        # 0 candidates: guide the user based on which preset has been tried.
        if mode == "标准":
            tip = "标准参数未切出，可点「🔧 宽松分割」换参数重试，或「⏭️ 跳过这张」"
        else:
            tip = "宽松参数仍未切出（珠子可能过密），可「⏭️ 跳过这张」"
        info = (f"📷 {os.path.basename(img_path)}（第 {index + 1}/{total} 张，{mode}参数）\n\n"
                f"⚠ 未识别出矩形板子区域。{tip}")

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


def _handle_resegment(img_index: int, mode: str, progress=None) -> tuple:
    """Re-segment the current photo with the chosen preset, then refresh view.

    Clears any pending selection, re-runs segmentation at the requested preset
    (overwriting the cache), and reloads the annotate view so the new candidates
    show immediately.
    """
    photos = _list_photos()
    if not photos or img_index >= len(photos):
        return _load_annotate_image(0)

    img_path = photos[img_index]
    _pending_selection.pop(img_path, None)
    _segment_with(img_path, mode, progress=progress)
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

        with gr.Tabs() as tabs:
            # ==== Tab 1: Upload ====
            with gr.Tab("① 上传照片", id="upload"):
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
            with gr.Tab("② 标注确认", id="annotate"):
                gr.Markdown(
                    "FastSAM 自动识别了多个区域，你需要选出**哪个区域是拼板**。\n"
                    "选中的结果将作为训练数据，教会 AI 自动识别拼板位置。"
                )
                annotate_progress = gr.Markdown("进度：0 已标注，0 跳过，共 0 张")
                thumbnail_strip = gr.Markdown("")

                with gr.Row():
                    with gr.Column(scale=1):
                        annotate_info = gr.Markdown("📷 点击下方「开始标注」加载第一张照片")
                        review_image = gr.Image(label="当前照片", type="numpy", height=550)
                        with gr.Row():
                            resegment_std_btn = gr.Button("🔧 标准分割", variant="secondary", size="sm")
                            resegment_loose_btn = gr.Button("🔧 宽松分割", variant="secondary", size="sm")
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

        # "开始标注" → switch to annotate tab + segment all + load first image.
        # Tab1 的 start_annotation_btn 和 Tab2 的 init_annotate_btn 共用同一处理器。
        start_outputs = [tabs] + annotate_outputs
        start_annotation_btn.click(
            fn=_handle_start_annotation_and_switch,
            inputs=[],
            outputs=start_outputs,
        ).then(
            fn=_load_first_annotate,
            inputs=[],
            outputs=annotate_outputs,
        )
        init_annotate_btn.click(
            fn=_handle_start_annotation_and_switch,
            inputs=[],
            outputs=start_outputs,
        ).then(
            fn=_load_first_annotate,
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

        # Re-segment the current photo at a different preset
        resegment_std_btn.click(
            fn=lambda idx: _handle_resegment(int(idx), "标准"),
            inputs=[annotate_current_idx],
            outputs=annotate_outputs,
        )
        resegment_loose_btn.click(
            fn=lambda idx: _handle_resegment(int(idx), "宽松"),
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
    # Load FastSAM SYNCHRONOUSLY before serving. This guarantees the model is
    # warm the instant the server is up, so clicking 「开始标注」 segments in
    # well under a second. The previous background-thread preheat could lose
    # the race with a fast user: if they uploaded and clicked within the
    # ~10-20s load window, the click hit a cold model and the image area sat
    # blank for many seconds with no visible feedback — which read exactly
    # like "no segmentation happened". Paying the one-time load at startup is
    # far less confusing than a frozen UI after the click.
    print("正在加载分割模型（首次约需 10-20 秒，请稍候）…", flush=True)
    _get_model()
    print("模型加载完成，启动服务…", flush=True)
    app = build_ui()
    app.launch(share=False, server_port=port)
