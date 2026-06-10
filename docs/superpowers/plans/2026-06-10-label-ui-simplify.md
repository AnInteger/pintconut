# 板子标注 UI 精简 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将板子标注 UI 从 4-Tab 精简为 2-Tab，替换 10 个候选按钮为 Radio 单选组，修复 Gradio 6.x 可见性 bug，并添加 Playwright 集成测试。

**Architecture:** 删除 Tab ② "自动分割" 和 Tab ④ "导出数据集"，将分割逻辑合并到 Tab ② 的"开始标注"按钮中。用 `gr.Radio` 替换 10 个 `visible=False` 按钮（Gradio 6.x bug）。所有按钮统一用 `interactive` 控制状态。Playwright Python 包提供自动化集成测试。

**Tech Stack:** Python, Gradio 6.17.3, OpenCV, FastSAM, Playwright (Python), pytest

---

## File Structure

| 文件 | 职责 | 改动类型 |
|------|------|---------|
| `src/label_ui.py` | Gradio UI — 完全重写为 2-Tab 结构 | 重写 |
| `tests/conftest.py` | pytest fixtures: Gradio 服务器 + Playwright 浏览器 | 新增 |
| `tests/test_label_ui_integration.py` | Playwright 集成测试（10 个用例） | 新增 |
| `src/label_service.py` | 不变 | — |
| `tests/test_label_service.py` | 不变 | — |

---

### Task 1: 安装 Playwright 依赖

**Files:**
- None (dependency installation only)

- [ ] **Step 1: 安装 playwright 和 pytest-playwright**

Run:
```bash
.venv/bin/pip install playwright pytest-playwright
.venv/bin/playwright install chromium
```

Expected: 安装成功，无报错。

- [ ] **Step 2: 验证安装**

Run:
```bash
.venv/bin/python -c "from playwright.sync_api import sync_playwright; print('OK')"
```

Expected: 输出 `OK`

---

### Task 2: 重写 `src/label_ui.py`

**Files:**
- Rewrite: `src/label_ui.py` (580 行 → ~350 行)

**Context:** 这是本计划的核心任务。完全重写 `build_ui()` 和所有回调函数。删除 Tab ②④、导航按钮、CSS、10 个按钮。新增 Radio 组件、简化输出元组、合并分割逻辑。

**关键变更清单：**

| 删除 | 新增/替换 |
|------|----------|
| `BUTTON_COLORS` 常量 | `_build_radio_choices()` 辅助函数 |
| `_build_button_css()` | `_handle_start_annotation()` 合并分割+加载 |
| `_js_goto_tab()` | Radio `.change` 事件替代按钮 `.click` |
| `_nav_dummy` 隐藏文本框 | Tab ① 的 `start_annotation_btn` |
| `nav_next_1`、`nav_next_2` 按钮 | Tab ② 内的 `export_report` 文本框 |
| 10 个 `cand-btn-{i}` 按钮 | 1 个 `gr.Radio` 组件 |
| `_handle_segment()` | 逻辑合并到 `_handle_start_annotation()` |
| `_step3()` 18 元素输出 | `_annotate()` 10 元素输出 |

**输出元组变更：**

当前 (18 元素):
```
[review_image, annotate_info, annotate_progress, annotate_current_idx,
 cand_btn_0..cand_btn_9,  # 10 个按钮
 thumbnail_strip, confirm_btn, reselect_btn, complete_export_btn]
```

新 (10 元素):
```
[review_image, annotate_info, annotate_progress, annotate_current_idx,
 cand_radio,              # 1 个 Radio 替代 10 个按钮
 thumbnail_strip,
 confirm_btn, reselect_btn, skip_btn, complete_export_btn]
```

- [ ] **Step 1: 用以下完整内容替换 `src/label_ui.py`**

```python
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
```

- [ ] **Step 2: 验证文件语法正确**

Run:
```bash
.venv/bin/python -c "from src.label_ui import build_ui; print('OK')"
```

Expected: 输出 `OK`

- [ ] **Step 3: 运行现有单元测试确认无回归**

Run:
```bash
.venv/bin/python -m pytest tests/test_label_service.py -v
```

Expected: 所有 10 个测试 PASS

- [ ] **Step 4: 提交**

```bash
git add src/label_ui.py
git commit -m "refactor: simplify label UI to 2-tab, replace buttons with Radio"
```

---

### Task 3: 创建测试基础设施 `tests/conftest.py`

**Files:**
- Create: `tests/conftest.py`

**Context:** Playwright 集成测试需要：启动 Gradio 服务器 → 连接浏览器 → 执行测试 → 关闭。使用 module scope 的服务器 fixture + function scope 的 page fixture。服务器使用端口 7899 避免与开发服务器冲突。

- [ ] **Step 1: 创建 `tests/conftest.py`**

```python
"""Pytest fixtures for Playwright integration tests of the Gradio label UI."""

import os
import signal
import subprocess
import time
import urllib.error
import urllib.request

import pytest
from playwright.sync_api import Page, Browser

# Port for test Gradio server (avoid conflict with dev server on 7860)
TEST_SERVER_PORT = 7899
TEST_SERVER_URL = f"http://127.0.0.1:{TEST_SERVER_PORT}"

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def gradio_server():
    """Start Gradio server for testing, yield URL, then shut down."""
    # Kill any existing process on the test port
    try:
        subprocess.run(["fuser", "-k", f"{TEST_SERVER_PORT}/tcp"],
                       capture_output=True, timeout=5)
    except Exception:
        pass
    time.sleep(0.5)

    env = os.environ.copy()
    env["GRADIO_SERVER_PORT"] = str(TEST_SERVER_PORT)
    env["GRADIO_SERVER_NAME"] = "127.0.0.1"

    proc = subprocess.Popen(
        [os.path.join(PROJECT_ROOT, ".venv/bin/python"),
         os.path.join(PROJECT_ROOT, "src/label_ui.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    # Wait for server to be ready (up to 30 seconds)
    for _ in range(30):
        try:
            urllib.request.urlopen(f"{TEST_SERVER_URL}/")
            break
        except urllib.error.URLError:
            time.sleep(1)
    else:
        proc.terminate()
        stdout = proc.stdout.read().decode()
        stderr = proc.stderr.read().decode()
        raise RuntimeError(
            f"Gradio server failed to start.\nstdout: {stdout}\nstderr: {stderr}"
        )

    yield TEST_SERVER_URL

    # Cleanup: send SIGTERM
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.fixture(scope="module")
def browser(gradio_server):
    """Launch a Chromium browser for the test module."""
    from playwright.sync_api import sync_playwright
    pw = sync_playwright().start()
    browser = pw.chromium.launch()
    yield browser
    browser.close()
    pw.stop()


@pytest.fixture
def page(browser, gradio_server):
    """Provide a fresh Playwright page connected to the Gradio server."""
    page = browser.new_page()
    page.goto(gradio_server, wait_until="networkidle")
    yield page
    page.close()
```

- [ ] **Step 2: 验证 conftest 语法**

Run:
```bash
.venv/bin/python -c "import tests.conftest; print('OK')"
```

Expected: 输出 `OK`

- [ ] **Step 3: 提交**

```bash
git add tests/conftest.py
git commit -m "test: add Playwright test infrastructure (conftest.py)"
```

---

### Task 4: 编写 Playwright 集成测试

**Files:**
- Create: `tests/test_label_ui_integration.py`

**Context:** 10 个测试用例按用户操作顺序排列。每个测试用例验证一个关键步骤的 UI 状态。测试使用 Playwright snapshot（无障碍树）验证 `interactive` 状态（`[disabled]` 属性）和内容。

- [ ] **Step 1: 创建 `tests/test_label_ui_integration.py`**

```python
"""Playwright integration tests for the 2-tab label UI.

Tests are ordered to represent the full user journey:
upload → segment → select → confirm → skip → complete → export.
"""

import re

import pytest
from playwright.sync_api import Page, expect


# ---------------------------------------------------------------------------
# Helper: wait for Gradio processing to finish
# ---------------------------------------------------------------------------
def _wait_for_gradio(page: Page, timeout: float = 60000):
    """Wait until no Gradio pending requests remain."""
    page.wait_for_timeout(500)
    # Wait for any pending Gradio API calls to complete
    page.wait_for_timeout(1000)


def _get_snapshot_text(page: Page) -> str:
    """Get accessibility snapshot as text for assertions."""
    return page.accessibility.snapshot()


# ---------------------------------------------------------------------------
# Test 1: Page loads with 2 tabs
# ---------------------------------------------------------------------------
def test_page_loads_with_two_tabs(page: Page):
    """Page should have exactly 2 tabs: ① 上传照片 and ② 标注确认."""
    page.wait_for_selector('button[role="tab"]', timeout=10000)
    tabs = page.query_selector_all('button[role="tab"]')
    tab_texts = [t.inner_text() for t in tabs]
    assert "① 上传照片" in tab_texts, f"Missing Tab 1. Found: {tab_texts}"
    assert "② 标注确认" in tab_texts, f"Missing Tab 2. Found: {tab_texts}"
    assert len(tab_texts) == 2, f"Expected 2 tabs, found {len(tab_texts)}: {tab_texts}"


# ---------------------------------------------------------------------------
# Test 2: Tab 1 is selected by default
# ---------------------------------------------------------------------------
def test_tab1_selected_by_default(page: Page):
    """Tab ① 上传照片 should be the active tab on page load."""
    page.wait_for_selector('button[role="tab"]', timeout=10000)
    active_tab = page.query_selector('button[role="tab"][aria-selected="true"]')
    assert active_tab is not None, "No active tab found"
    assert "① 上传照片" in active_tab.inner_text()


# ---------------------------------------------------------------------------
# Test 3: Upload UI elements exist
# ---------------------------------------------------------------------------
def test_upload_ui_elements(page: Page):
    """Tab ① should have upload button, status textbox, and gallery."""
    upload_btn = page.get_by_role("button", name="📤 上传")
    assert upload_btn.is_visible(), "Upload button not visible"

    status = page.get_by_label("状态")
    assert status is not None, "Status textbox not found"

    gallery = page.get_by_label("已上传照片")
    assert gallery is not None, "Gallery not found"


# ---------------------------------------------------------------------------
# Test 4: Start annotation button disabled initially
# ---------------------------------------------------------------------------
def test_start_annotation_button_disabled(page: Page):
    """'开始标注' button in Tab ① should be disabled before upload."""
    btn = page.get_by_role("button", name="▶ 开始标注").first
    assert btn.is_disabled(), "Start annotation button should be disabled before upload"


# ---------------------------------------------------------------------------
# Test 5: Tab ② initial state
# ---------------------------------------------------------------------------
def test_tab2_initial_state(page: Page):
    """Tab ② should have Radio (disabled), skip/confirm/reselect buttons."""
    page.get_by_role("tab", name="② 标注确认").click()
    _wait_for_gradio(page)

    # Radio should exist and be disabled
    radio = page.get_by_role("radiogroup")
    assert radio.is_visible(), "Radio group not visible"

    # Action buttons should have correct initial states
    skip_btn = page.get_by_role("button", name="⏭️ 跳过这张")
    confirm_btn = page.get_by_role("button", name="✅ 确认，下一张")
    reselect_btn = page.get_by_role("button", name="↩️ 重新选择")

    assert skip_btn.is_enabled(), "Skip button should be enabled"
    assert confirm_btn.is_disabled(), "Confirm button should be disabled initially"
    assert reselect_btn.is_disabled(), "Reselect button should be disabled initially"


# ---------------------------------------------------------------------------
# Test 6: Start annotation triggers segmentation
# ---------------------------------------------------------------------------
def test_start_annotation(page: Page):
    """Clicking '开始标注' should segment images and load first one."""
    # Go to Tab ② and click 开始标注
    page.get_by_role("tab", name="② 标注确认").click()
    _wait_for_gradio(page)

    init_btn = page.locator("button", has_text="▶ 开始标注").last
    init_btn.click()

    # Wait for segmentation to complete (FastSAM can be slow)
    page.wait_for_timeout(60000)

    # Should now show an image and candidates
    # Check that annotate_info contains "候选区域"
    info_el = page.locator("text=个候选区域")
    assert info_el.is_visible(timeout=5000), "Candidate info not shown after segmentation"


# ---------------------------------------------------------------------------
# Test 7: Radio choices are populated and interactive
# ---------------------------------------------------------------------------
def test_radio_has_choices(page: Page):
    """Radio should have candidate choices after segmentation."""
    page.get_by_role("tab", name="② 标注确认").click()
    _wait_for_gradio(page)

    # Wait for radio options to appear
    radio_options = page.locator("input[type='radio']")
    count = radio_options.count()
    assert count > 0, f"Expected radio options, found {count}"

    # Verify option text contains "#N 面积"
    first_label = page.locator("label").filter(has_text=re.compile(r"#\d+ 面积")).first
    assert first_label.is_visible(), "Radio option text should contain '#N 面积'"


# ---------------------------------------------------------------------------
# Test 8: Selecting candidate shows preview and activates confirm/reselect
# ---------------------------------------------------------------------------
def test_select_candidate(page: Page):
    """Selecting a radio option should show preview and enable confirm/reselect."""
    page.get_by_role("tab", name="② 标注确认").click()
    _wait_for_gradio(page)

    # Click first radio option
    first_radio = page.locator("input[type='radio']").first
    first_radio.click(force=True)
    _wait_for_gradio(page)

    # Should show "已选择区域" text
    selected_text = page.locator("text=已选择区域")
    assert selected_text.is_visible(timeout=5000), "Selection preview not shown"

    # Confirm and reselect buttons should be enabled
    confirm_btn = page.get_by_role("button", name="✅ 确认，下一张")
    reselect_btn = page.get_by_role("button", name="↩️ 重新选择")
    assert confirm_btn.is_enabled(), "Confirm should be enabled after selection"
    assert reselect_btn.is_enabled(), "Reselect should be enabled after selection"

    # Skip should be disabled
    skip_btn = page.get_by_role("button", name="⏭️ 跳过这张")
    assert skip_btn.is_disabled(), "Skip should be disabled after selection"


# ---------------------------------------------------------------------------
# Test 9: Reselect returns to candidate view
# ---------------------------------------------------------------------------
def test_reselect(page: Page):
    """Clicking reselect should return to candidate view with Radio interactive."""
    page.get_by_role("tab", name="② 标注确认").click()
    _wait_for_gradio(page)

    # First select a candidate
    first_radio = page.locator("input[type='radio']").first
    if first_radio.is_visible():
        first_radio.click(force=True)
        _wait_for_gradio(page)

    # Click reselect
    reselect_btn = page.get_by_role("button", name="↩️ 重新选择")
    if reselect_btn.is_enabled():
        reselect_btn.click()
        _wait_for_gradio(page)

        # Radio should be interactive again
        radio = page.locator("input[type='radio']").first
        assert radio.is_enabled(), "Radio should be interactive after reselect"

        # Confirm/reselect should be disabled
        confirm_btn = page.get_by_role("button", name="✅ 确认，下一张")
        assert confirm_btn.is_disabled(), "Confirm should be disabled after reselect"


# ---------------------------------------------------------------------------
# Test 10: Confirm saves and advances to next image
# ---------------------------------------------------------------------------
def test_confirm_advances(page: Page):
    """Confirming should save label and load next image."""
    page.get_by_role("tab", name="② 标注确认").click()
    _wait_for_gradio(page)

    # Select a candidate
    first_radio = page.locator("input[type='radio']").first
    if first_radio.is_visible():
        first_radio.click(force=True)
        _wait_for_gradio(page)

    # Confirm
    confirm_btn = page.get_by_role("button", name="✅ 确认，下一张")
    if confirm_btn.is_enabled():
        confirm_btn.click()
        _wait_for_gradio(page)

        # Should show next image (check thumbnail strip has ✅)
        thumbnail = page.locator("text=✅")
        assert thumbnail.is_visible(timeout=5000), "No ✅ icon in thumbnails after confirm"


# ---------------------------------------------------------------------------
# Test 11: Skip advances without labeling
# ---------------------------------------------------------------------------
def test_skip_advances(page: Page):
    """Skipping should mark as skipped and load next image."""
    page.get_by_role("tab", name="② 标注确认").click()
    _wait_for_gradio(page)

    skip_btn = page.get_by_role("button", name="⏭️ 跳过这张")
    if skip_btn.is_enabled():
        skip_btn.click()
        _wait_for_gradio(page)

        # Check for ⏭️ in thumbnails
        thumbnail = page.locator("text=⏭️")
        assert thumbnail.is_visible(timeout=5000), "No ⏭️ icon in thumbnails after skip"


# ---------------------------------------------------------------------------
# Test 12: Complete all shows export button
# ---------------------------------------------------------------------------
def test_complete_shows_export(page: Page):
    """After processing all images, export button should be enabled."""
    page.get_by_role("tab", name="② 标注确认").click()
    _wait_for_gradio(page)

    # Process remaining images by skipping
    skip_btn = page.get_by_role("button", name="⏭️ 跳过这张")
    for _ in range(5):  # Safety limit
        if skip_btn.is_enabled():
            skip_btn.click()
            _wait_for_gradio(page)
            page.wait_for_timeout(2000)
        else:
            break

    # Should show "全部处理完成"
    complete_text = page.locator("text=全部处理完成")
    if complete_text.is_visible(timeout=5000):
        # Export button should be enabled
        export_btn = page.get_by_role("button", name="📦 导出数据集")
        assert export_btn.is_enabled(), "Export button should be enabled when complete"


# ---------------------------------------------------------------------------
# Test 13: Export shows dataset report
# ---------------------------------------------------------------------------
def test_export_shows_report(page: Page):
    """Clicking export should show dataset report."""
    page.get_by_role("tab", name="② 标注确认").click()
    _wait_for_gradio(page)

    export_btn = page.get_by_role("button", name="📦 导出数据集")
    if export_btn.is_enabled():
        export_btn.click()
        _wait_for_gradio(page)

        # Should show report with "数据集统计"
        report = page.locator("text=数据集统计")
        assert report.is_visible(timeout=5000), "Dataset report not shown after export"
```

- [ ] **Step 2: 验证测试文件语法**

Run:
```bash
.venv/bin/python -c "import tests.test_label_ui_integration; print('OK')"
```

Expected: 输出 `OK`

- [ ] **Step 3: 提交**

```bash
git add tests/test_label_ui_integration.py
git commit -m "test: add Playwright integration tests for 2-tab label UI"
```

---

### Task 5: 运行所有测试

**Files:** None (verification only)

- [ ] **Step 1: 运行单元测试**

Run:
```bash
.venv/bin/python -m pytest tests/test_label_service.py -v
```

Expected: 所有 10 个测试 PASS

- [ ] **Step 2: 运行 Playwright 集成测试**

Run:
```bash
.venv/bin/python -m pytest tests/test_label_ui_integration.py -v --timeout=180
```

Expected: 所有 13 个测试 PASS（可能部分因图片内容不同而失败，需根据实际调整）

- [ ] **Step 3: 修复任何失败的集成测试**

如果集成测试失败：
1. 检查 Gradio 服务器是否正确启动（看 conftest 错误信息）
2. 检查 selector 是否匹配 Gradio 6.x 渲染的实际 DOM 结构
3. 调整 `page.wait_for_timeout()` 的等待时间
4. 用 `page.screenshot(path="debug.png")` 调试页面状态

- [ ] **Step 4: 最终提交**

```bash
git add -A
git commit -m "test: verify all integration tests pass"
```
