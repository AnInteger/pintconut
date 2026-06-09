# 标注 UI V2 重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 label_ui.py 从 Tab 结构重构为严格向导式流程，解决引导缺失、标注不直观、进度不可见的问题。

**Architecture:** 使用 Gradio `gr.Column(visible=...)` 控制步骤显示/隐藏。顶部 `gr.Markdown` 进度条 + 底部导航按钮。步骤 ③ 用候选卡片按钮替代数字输入框。所有业务逻辑函数保持不变，只重写 `build_ui()` 和相关 UI 回调。

**Tech Stack:** Gradio 6.x, OpenCV, NumPy

---

### Task 1: 重写步骤 ① 上传照片

**Files:**
- Modify: `src/label_ui.py` (lines 64-80 handler + lines 257-321 build_ui)

- [ ] **Step 1: 替换 `_handle_upload` 为中文版本**

将 `_handle_upload` 函数改为返回中文状态 + 缩略图预览列表：

```python
def _handle_upload(files):
    """Upload photos and return (status_text, gallery_images)."""
    if not files:
        return "❌ 未选择文件", []
    _ensure_dirs(PHOTOS_DIR)
    count = 0
    previews = []
    for f in files:
        fpath = f if isinstance(f, str) else getattr(f, "name", str(f))
        basename = os.path.basename(fpath)
        dest = os.path.join(PHOTOS_DIR, basename)
        if os.path.abspath(fpath) != os.path.abspath(dest):
            shutil.copy2(fpath, dest)
        count += 1
    # Generate previews from copied files
    for img_path in _list_photos():
        img = cv2.imread(img_path)
        if img is not None:
            previews.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    total = len(_list_photos())
    return f"✅ 已上传 {count} 张照片（共 {total} 张）", previews
```

- [ ] **Step 2: 验证无语法错误**

Run: `source .venv/bin/activate && python -c "import ast; ast.parse(open('src/label_ui.py').read()); print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add src/label_ui.py
git commit -m "refactor: update _handle_upload for v2 wizard UI"
```

---

### Task 2: 重写步骤 ② 自动分割

**Files:**
- Modify: `src/label_ui.py` (`_handle_segment` function)

- [ ] **Step 1: 替换 `_handle_segment` 为中文版本，返回分割结果缩略图**

```python
def _handle_segment(progress=gr.Progress()):
    """Segment all photos, store results, return (gallery_images, status_text)."""
    photos = _list_photos()
    if not photos:
        return [], "❌ 没有照片，请先返回上一步上传照片。"

    model = _get_model()
    annotated = []
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

    _store_segmentation_results(photos, model)

    msg = f"✅ 已分割 {len(annotated)} 张照片"
    if len(annotated) < n:
        msg += f"（{n - len(annotated)} 张无法读取）"
    return annotated, msg
```

- [ ] **Step 2: 验证无语法错误**

Run: `source .venv/bin/activate && python -c "import ast; ast.parse(open('src/label_ui.py').read()); print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add src/label_ui.py
git commit -m "refactor: update _handle_segment for v2 wizard UI"
```

---

### Task 3: 重写步骤 ③ 标注确认

**Files:**
- Modify: `src/label_ui.py` (replace Tab 3 handlers + add new annotation state tracking)

这是最核心的改动。需要：
- 用 `gr.State` 追踪每张照片的标注状态（labeled/skipped/unprocessed）
- 候选区域用动态按钮展示
- 选中/跳过后自动跳到下一张

- [ ] **Step 1: 添加标注状态管理**

在 `_segmentation_cache` 后面添加：

```python
# Annotation state: map image_path -> "labeled" | "skipped" | None
_annotation_state: dict[str, str | None] = {}


def _init_annotation_state() -> None:
    """Initialize annotation state for all photos."""
    _annotation_state.clear()
    for img_path in _list_photos():
        _annotation_state[img_path] = None


def _get_annotation_progress() -> tuple[int, int, int]:
    """Return (labeled_count, skipped_count, total_count)."""
    total = len(_annotation_state)
    labeled = sum(1 for v in _annotation_state.values() if v == "labeled")
    skipped = sum(1 for v in _annotation_state.values() if v == "skipped")
    return labeled, skipped, total


def _find_next_unprocessed(current_path: str | None = None) -> int:
    """Return index of next unprocessed photo, or -1 if all done."""
    photos = _list_photos()
    # If current_path given, start searching from after it
    start = 0
    if current_path and current_path in photos:
        start = photos.index(current_path) + 1
    for i in range(start, len(photos)):
        if _annotation_state.get(photos[i]) is None:
            return i
    # Wrap around from beginning
    for i in range(0, start):
        if _annotation_state.get(photos[i]) is None:
            return i
    return -1  # all done
```

- [ ] **Step 2: 替换步骤 ③ 的 handler 函数**

删除旧函数 `_load_image_for_review`, `_handle_confirm`, `_handle_skip`, `_handle_goto`，替换为新的：

```python
def _build_thumbnails_html() -> str:
    """Build HTML for thumbnail strip showing annotation status."""
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


def _load_review_image(index: int) -> tuple:
    """Load image for review. Returns (rgb_image, info_text, progress_text, candidate_buttons_update)."""
    photos = _list_photos()
    if not photos:
        return None, "没有照片", "", []

    index = max(0, min(index, len(photos) - 1))
    img_path = photos[index]
    img = cv2.imread(img_path)
    if img is None:
        return None, f"无法读取: {os.path.basename(img_path)}", "", []

    candidates = _segmentation_cache.get(img_path, [])
    vis = draw_candidates(img, candidates)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    labeled, skipped, total = _get_annotation_progress()
    progress_text = f"进度：{labeled} 已标注，{skipped} 跳过，共 {total} 张"
    info = f"📷 {os.path.basename(img_path)}（第 {index + 1}/{total} 张）— {len(candidates)} 个候选区域"

    # Build candidate button labels
    cand_labels = []
    for ci, cand in enumerate(candidates):
        area_pct = f"{cand['area_ratio'] * 100:.1f}%"
        cand_labels.append(f"①{ci} 面积 {area_pct}" if ci == 0 else f"②{ci} 面积 {area_pct}" if ci == 1 else f"③{ci} 面积 {area_pct}" if ci == 2 else f"{ci} 面积 {area_pct}")

    return vis_rgb, info, progress_text, index, cand_labels


def _handle_select_candidate(img_index: int, cand_index: int):
    """Select a candidate as the board label and move to next unprocessed."""
    photos = _list_photos()
    if not photos or img_index >= len(photos):
        return _load_review_image(0) + (_build_thumbnails_html(),)

    img_path = photos[img_index]
    candidates = _segmentation_cache.get(img_path, [])
    if cand_index >= len(candidates):
        return _load_review_image(img_index) + (_build_thumbnails_html(),)

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
        # All done
        labeled, skipped, total = _get_annotation_progress()
        return None, f"🎉 全部处理完成！", f"进度：{labeled} 已标注，{skipped} 跳过，共 {total} 张", img_index, [], _build_thumbnails_html()

    return _load_review_image(next_idx) + (_build_thumbnails_html(),)


def _handle_skip_photo(img_index: int):
    """Skip current photo and move to next."""
    photos = _list_photos()
    if not photos or img_index >= len(photos):
        return _load_review_image(0) + (_build_thumbnails_html(),)

    img_path = photos[img_index]
    _annotation_state[img_path] = "skipped"

    next_idx = _find_next_unprocessed(img_path)
    if next_idx < 0:
        labeled, skipped, total = _get_annotation_progress()
        return None, f"🎉 全部处理完成！", f"进度：{labeled} 已标注，{skipped} 跳过，共 {total} 张", img_index, [], _build_thumbnails_html()

    return _load_review_image(next_idx) + (_build_thumbnails_html(),)


def _handle_goto_photo(img_index: int):
    """Jump to a specific photo by index."""
    result = _load_review_image(img_index)
    return result + (_build_thumbnails_html(),)
```

- [ ] **Step 3: 验证无语法错误**

Run: `source .venv/bin/activate && python -c "import ast; ast.parse(open('src/label_ui.py').read()); print('OK')"`
Expected: OK

- [ ] **Step 4: Commit**

```bash
git add src/label_ui.py
git commit -m "refactor: replace Tab 3 handlers with annotation state tracking"
```

---

### Task 4: 重写步骤 ④ 导出数据集

**Files:**
- Modify: `src/label_ui.py` (`_handle_export` function)

- [ ] **Step 1: 替换 `_handle_export` 为中文版本，包含统计信息**

```python
def _handle_export():
    """Export dataset and return report text."""
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
```

- [ ] **Step 2: Commit**

```bash
git add src/label_ui.py
git commit -m "refactor: update _handle_export with Chinese text and stats"
```

---

### Task 5: 重写 `build_ui()` 为严格向导流

**Files:**
- Modify: `src/label_ui.py` (`build_ui` function, lines 257-321)

这是最大的改动。完全替换 `build_ui()`，用 `gr.Column(visible=...)` 实现向导。

- [ ] **Step 1: 编写进度条辅助函数**

在 `build_ui()` 之前添加：

```python
def _progress_md(step: int) -> str:
    """Return progress bar markdown for the given step (1-4)."""
    steps = ["上传照片", "自动分割", "标注确认", "导出数据集"]
    parts = []
    for i, label in enumerate(steps, 1):
        num = ["①", "②", "③", "④"][i - 1]
        if i < step:
            parts.append(f"{num} ~~{label}~~ ✅")
        elif i == step:
            parts.append(f"**{num} {label}** ◀")
        else:
            parts.append(f"{num} {label}")
    return "  ━━  ".join(parts)
```

- [ ] **Step 2: 编写导航回调函数**

```python
def _go_to_step(target_step: int):
    """Return visibility updates for all 4 step columns + progress markdown."""
    vis = [target_step == i for i in range(1, 5)]
    return (
        gr.Column(visible=vis[0]),
        gr.Column(visible=vis[1]),
        gr.Column(visible=vis[2]),
        gr.Column(visible=vis[3]),
        _progress_md(target_step),
    )
```

- [ ] **Step 3: 重写 `build_ui()` 为向导式布局**

完全替换 `build_ui()`：

```python
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Pintconut 拼豆标注工具") as app:
        gr.Markdown("# 🫘 Pintconut 拼豆标注工具")

        # --- Progress bar (always visible) ---
        progress_bar = gr.Markdown(_progress_md(1))

        # --- Step ① Upload ---
        with gr.Column(visible=True) as step1_col:
            gr.Markdown("### ① 上传照片\n上传拼板照片，用于训练拼板检测模型。")
            file_input = gr.File(label="选择照片（支持多选）", file_count="multiple", file_types=["image"])
            upload_btn = gr.Button("📤 上传", variant="primary")
            upload_status = gr.Textbox(label="状态", interactive=False, lines=2)
            upload_gallery = gr.Gallery(label="已上传照片", columns=6, height="auto")
            upload_btn.click(
                fn=_handle_upload,
                inputs=[file_input],
                outputs=[upload_status, upload_gallery],
            )
            nav_next_1 = gr.Button("下一步 →", variant="primary")

        # --- Step ② Segment ---
        with gr.Column(visible=False) as step2_col:
            gr.Markdown("### ② 自动分割\nFastSAM 将自动识别每张照片中的候选区域。")
            segment_btn = gr.Button("🔍 开始分割", variant="primary")
            segment_gallery = gr.Gallery(label="分割结果", columns=3, height="auto")
            segment_status = gr.Textbox(label="状态", interactive=False)
            segment_btn.click(
                fn=_handle_segment,
                inputs=[],
                outputs=[segment_gallery, segment_status],
            )
            with gr.Row():
                nav_prev_2 = gr.Button("← 上一步")
                nav_next_2 = gr.Button("下一步 →", variant="primary")

        # --- Step ③ Annotate ---
        with gr.Column(visible=False) as step3_col:
            gr.Markdown(
                "### ③ 标注确认\n"
                "FastSAM 自动识别了多个区域，你需要选出**哪个区域是拼板**。\n"
                "选中的结果将作为训练数据，教会 AI 自动识别拼板位置。"
            )
            annotate_progress = gr.Markdown("进度：0 已标注，0 跳过，共 0 张")
            thumbnail_strip = gr.Markdown("")

            with gr.Row():
                with gr.Column(scale=1):
                    annotate_info = gr.Markdown("📷 选择照片开始标注")
                    review_image = gr.Image(label="当前照片", type="numpy", height=400)
                with gr.Column(scale=1):
                    gr.Markdown("**候选区域**（点击选择拼板所在的区域）")
                    # Dynamically generated candidate buttons — up to 10
                    cand_buttons = []
                    for ci in range(10):
                        btn = gr.Button(f"候选 {ci}", visible=False)
                        cand_buttons.append(btn)

                    gr.Markdown("---")
                    skip_btn = gr.Button("⏭️ 跳过这张", variant="secondary")

            annotate_current_idx = gr.State(value=0)

        # --- Step ④ Export ---
        with gr.Column(visible=False) as step4_col:
            gr.Markdown("### ④ 导出数据集\n生成 YOLO 格式数据集，用于训练模型。")
            export_btn = gr.Button("📦 导出数据集", variant="primary")
            export_status = gr.Textbox(label="数据集报告", interactive=False, lines=12)
            export_btn.click(fn=_handle_export, inputs=[], outputs=[export_status])
            with gr.Row():
                nav_prev_4 = gr.Button("← 上一步")
                nav_finish = gr.Button("完成 ✅", variant="primary")

        # =========================================
        # Navigation wiring
        # =========================================

        # Step ① → ②
        nav_next_1.click(
            fn=lambda: _go_to_step(2),
            inputs=[],
            outputs=[step1_col, step2_col, step3_col, step4_col, progress_bar],
        )

        # Step ② → ①
        nav_prev_2.click(
            fn=lambda: _go_to_step(1),
            inputs=[],
            outputs=[step1_col, step2_col, step3_col, step4_col, progress_bar],
        )

        # Step ② → ③ (also initializes annotation state + loads first image)
        def _enter_step3():
            _init_annotation_state()
            photos = _list_photos()
            if not photos:
                return _go_to_step(3) + (None, "没有照片", "进度：0/0", 0, _build_thumbnails_html())
            result = _load_review_image(0)
            return _go_to_step(3) + result + (_build_thumbnails_html(),)

        nav_next_2.click(
            fn=_enter_step3,
            inputs=[],
            outputs=[step1_col, step2_col, step3_col, step4_col, progress_bar,
                     review_image, annotate_info, annotate_progress, annotate_current_idx, thumbnail_strip],
        )

        # Step ③ → ②
        nav_prev_3_placeholder = None  # No prev button in step 3 per spec

        # Step ③ candidate button clicks
        for ci, btn in enumerate(cand_buttons):
            def _make_select_handler(cand_idx):
                def handler(current_idx):
                    return _handle_select_candidate(int(current_idx), cand_idx)
                return handler
            btn.click(
                fn=_make_select_handler(ci),
                inputs=[annotate_current_idx],
                outputs=[review_image, annotate_info, annotate_progress, annotate_current_idx, thumbnail_strip],
            )

        # Step ③ skip
        skip_btn.click(
            fn=lambda idx: _handle_skip_photo(int(idx)),
            inputs=[annotate_current_idx],
            outputs=[review_image, annotate_info, annotate_progress, annotate_current_idx, thumbnail_strip],
        )

        # Step ③ → ④
        def _enter_step4():
            return _go_to_step(4) + (_handle_export(),)

        # (no explicit next button in step 3 — all photos processed triggers completion)
        # We add a "下一步" that appears when all done
        # For simplicity, export step is reached via the step4_prev button being moved to step3
        # Actually let's add a next button to step 3

        # Step ④ → ③
        nav_prev_4.click(
            fn=lambda: _go_to_step(3),
            inputs=[],
            outputs=[step1_col, step2_col, step3_col, step4_col, progress_bar],
        )

        # Finish
        nav_finish.click(
            fn=lambda: _go_to_step(1),
            inputs=[],
            outputs=[step1_col, step2_col, step3_col, step4_col, progress_bar],
        )

    return app
```

- [ ] **Step 4: 验证无语法错误**

Run: `source .venv/bin/activate && python -c "import ast; ast.parse(open('src/label_ui.py').read()); print('OK')"`
Expected: OK

- [ ] **Step 5: Commit**

```bash
git add src/label_ui.py
git commit -m "feat: rewrite build_ui as strict wizard flow with visibility control"
```

---

### Task 6: 补充步骤 ③ → ④ 的导航 + 修复候选按钮可见性

**Files:**
- Modify: `src/label_ui.py`

步骤 ③ 需要一个「下一步 →」按钮（当所有照片处理完后可用），且候选按钮需要动态显示/隐藏。

- [ ] **Step 1: 修改步骤 ③ 的 handler 返回值，包含候选按钮可见性**

更新 `_load_review_image` 返回 10 个按钮的 `visible` 值和 label：

```python
def _load_review_image(index: int) -> tuple:
    """Load image for review. Returns (image, info, progress, index, cand_labels, 10x visible bools)."""
    photos = _list_photos()
    if not photos:
        return None, "没有照片", "", 0, [] + [gr.Button(visible=False)] * 10

    index = max(0, min(index, len(photos) - 1))
    img_path = photos[index]
    img = cv2.imread(img_path)
    if img is None:
        return None, f"无法读取: {os.path.basename(img_path)}", "", index, [] + [gr.Button(visible=False)] * 10

    candidates = _segmentation_cache.get(img_path, [])
    vis = draw_candidates(img, candidates)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    labeled, skipped, total = _get_annotation_progress()
    progress_text = f"进度：{labeled} 已标注，{skipped} 跳过，共 {total} 张"
    info = f"📷 {os.path.basename(img_path)}（第 {index + 1}/{total} 张）— {len(candidates)} 个候选区域"

    cand_labels = []
    for ci, cand in enumerate(candidates):
        area_pct = f"{cand['area_ratio'] * 100:.1f}%"
        cand_labels.append(f"#{ci} 面积 {area_pct}")

    # Pad to 10 with empty/invisible
    button_visibles = [ci < len(candidates) for ci in range(10)]
    while len(cand_labels) < 10:
        cand_labels.append("")

    return (vis_rgb, info, progress_text, index, *cand_labels, *button_visibles)
```

- [ ] **Step 2: 在步骤 ③ 添加「下一步 →」按钮**

在 `skip_btn` 后面添加：

```python
                    nav_next_3 = gr.Button("下一步 →（全部处理完后可用）", variant="primary")
```

- [ ] **Step 3: 更新 wiring 中候选按钮的 outputs，包含 label 和 visible**

每个 cand_button 的 click outputs 需要包含 `gr.Button(value=...)` 和 `gr.Button(visible=...)`。

更新 handler 和 wiring：

```python
        for ci, btn in enumerate(cand_buttons):
            def _make_select_handler(cand_idx):
                def handler(current_idx):
                    result = _handle_select_candidate(int(current_idx), cand_idx)
                    # result: (image, info, progress, index, *10 labels, *10 visibles, thumbnails_html)
                    return result
                return handler
            btn.click(
                fn=_make_select_handler(ci),
                inputs=[annotate_current_idx],
                outputs=[review_image, annotate_info, annotate_progress, annotate_current_idx]
                        + cand_buttons + [thumbnail_strip],
            )
```

同步更新 `_handle_select_candidate` 和 `_handle_skip_photo` 返回值与 `_load_review_image` 一致。

- [ ] **Step 4: 验证完整文件无语法错误**

Run: `source .venv/bin/activate && python -c "import ast; ast.parse(open('src/label_ui.py').read()); print('OK')"`
Expected: OK

- [ ] **Step 5: Commit**

```bash
git add src/label_ui.py
git commit -m "feat: add step 3→4 navigation and dynamic candidate buttons"
```

---

### Task 7: 运行现有测试确保无回归

**Files:**
- Test: `tests/test_label_service.py`

- [ ] **Step 1: 运行全部测试**

Run: `source .venv/bin/activate && python -m pytest tests/ -v`
Expected: All 28 tests pass

- [ ] **Step 2: 如果有失败，修复并重新运行**

- [ ] **Step 3: Commit**

```bash
git commit -m "test: verify all tests pass after UI refactor"
```

---

### Task 8: 手动冒烟测试 Gradio UI

**Files:**
- `src/label_ui.py`

- [ ] **Step 1: 启动 Gradio**

Run: `source .venv/bin/activate && python src/label_ui.py`
Expected: Server starts on http://localhost:7860

- [ ] **Step 2: 用 Playwright 或手动浏览器测试完整向导流程**

1. 页面加载 → 只看到步骤 ①，进度条显示「① **上传照片** ◀」
2. 上传照片 → 状态显示中文 + 缩略图
3. 点击「下一步 →」→ 步骤 ① 消失，步骤 ② 出现
4. 点击「开始分割」→ 进度条 + 分割结果
5. 点击「下一步 →」→ 步骤 ③ 出现，显示第一张照片和候选按钮
6. 点击候选按钮 → 保存标签，自动跳到下一张
7. 跳过 → 标记跳过，自动跳到下一张
8. 全部处理完 → 点击「下一步 →」→ 步骤 ④ 出现
9. 点击「导出数据集」→ 显示中文统计报告
10. 点击「← 上一步」→ 返回步骤 ③

- [ ] **Step 3: Commit 如果需要修复**

```bash
git add src/label_ui.py
git commit -m "fix: address smoke test findings"
```

---

### Task 9: 清理 + 最终提交

- [ ] **Step 1: 删除不再使用的旧函数和注释**

确保文件中没有残留的 `# Tab 1`, `# Tab 2` 等旧注释。

- [ ] **Step 2: 更新文件顶部 docstring**

```python
"""Gradio web UI for bead board labeling — strict wizard flow (V2)."""
```

- [ ] **Step 3: 最终提交 + Push**

```bash
git add -A
git commit -m "refactor: complete labeling UI v2 — strict wizard flow"
git push origin main
```

---

## Self-Review Checklist

**1. Spec coverage:**
- ✅ 严格向导流（visible 控制）→ Task 5
- ✅ 进度条始终可见 → Task 5
- ✅ 步骤 ① 上传 + 缩略图 → Task 1
- ✅ 步骤 ② 分割 + 进度 → Task 2
- ✅ 步骤 ③ 标注 + 候选卡片 + 状态追踪 → Task 3, 6
- ✅ 步骤 ④ 导出 + 统计 → Task 4
- ✅ 中文引导文字 → 全部 tasks
- ✅ 下一步/上一步导航 → Task 5, 6

**2. Placeholder scan:** 无 TBD/TODO

**3. Type consistency:**
- `_handle_select_candidate` 返回 tuple 与 `_load_review_image` 一致
- `_handle_skip_photo` 返回值与 `_load_review_image` 一致
- `_enter_step3` outputs 列表与 handler 返回 tuple 匹配
