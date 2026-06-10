# 板子标注 UI 改进 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 改进板子标注 UI 的 3 个体验问题：颜色对照、完成引导、结果展示

**Architecture:** 在现有 `label_service.py` 新增 `draw_result()` 渲染函数，在 `label_ui.py` 改进按钮显示、增加完成引导按钮和结果确认交互流

**Tech Stack:** Python, OpenCV, Gradio, pytest

---

## File Structure

| 文件 | 职责 | 改动类型 |
|------|------|---------|
| `src/label_service.py` | 新增 `draw_result(image, mask)` — 在原图上绘制标注结果轮廓 | 修改 |
| `tests/test_label_service.py` | `draw_result()` 的单元测试 | 修改 |
| `src/label_ui.py` | 颜色 emoji 按钮、完成引导、结果确认流 | 修改 |

---

### Task 1: 新增 `draw_result()` 函数

**Files:**
- Modify: `src/label_service.py` — 在文件末尾追加
- Modify: `tests/test_label_service.py` — 追加测试

- [ ] **Step 1: 写失败测试**

在 `tests/test_label_service.py` 末尾追加：

```python
from src.label_service import draw_result


def test_draw_result_returns_same_shape():
    """draw_result should return an image with the same shape as input."""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 128
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:150, 50:150] = 255
    result = draw_result(img, mask)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_draw_result_draws_contour():
    """draw_result should modify the image around the mask boundary."""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 128
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:150, 50:150] = 255
    result = draw_result(img, mask)
    # The pixel at the mask boundary (row 50) should differ from original
    assert not np.array_equal(result[50, 50], img[50, 50])


def test_draw_result_empty_mask():
    """draw_result with an empty mask should return a copy of the image unchanged."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 200
    mask = np.zeros((100, 100), dtype=np.uint8)
    result = draw_result(img, mask)
    # Should still be a valid image, just no contour drawn
    assert result.shape == img.shape
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_label_service.py::test_draw_result_returns_same_shape tests/test_label_service.py::test_draw_result_draws_contour tests/test_label_service.py::test_draw_result_empty_mask -v`
Expected: FAIL — `ImportError: cannot import name 'draw_result'`

- [ ] **Step 3: 实现 `draw_result()`**

在 `src/label_service.py` 末尾追加：

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_label_service.py -v`
Expected: 所有测试 PASS

- [ ] **Step 5: 提交**

```bash
git add src/label_service.py tests/test_label_service.py
git commit -m "feat: add draw_result() for annotation result preview"
```

---

### Task 2: 候选按钮增加颜色 emoji 标识

**Files:**
- Modify: `src/label_ui.py`

**Context:** 当前 `_load_review_image()` 中按钮文字是 `f"#{ci} 面积 {area_pct}"`。需要在按钮文字前加上对应 `COLORS[i]` 的 emoji。

- [ ] **Step 1: 添加 `COLOR_EMOJIS` 常量并修改按钮文字**

在 `src/label_ui.py` 顶部的 import 区域后（约第 27 行后，`DATASET_DIR` 定义之前），添加常量：

```python
# Emojis matching COLORS order in label_service.py
COLOR_EMOJIS = ["🟢", "🔴", "🔵", "🟡", "🟦", "🟣", "🟩", "🔶", "🟠", "🟤"]
```

修改 `_load_review_image()` 中的按钮文字生成（约第 216-218 行），将：

```python
            area_pct = f"{candidates[ci]['area_ratio'] * 100:.1f}%"
            btn_updates.append(gr.update(value=f"#{ci} 面积 {area_pct}", visible=True))
```

改为：

```python
            area_pct = f"{candidates[ci]['area_ratio'] * 100:.1f}%"
            emoji = COLOR_EMOJIS[ci % len(COLOR_EMOJIS)]
            btn_updates.append(gr.update(value=f"{emoji} #{ci} 面积 {area_pct}", visible=True))
```

- [ ] **Step 2: 验证改动**

Run: `.venv/bin/python -c "from src.label_ui import COLOR_EMOJIS; print(COLOR_EMOJIS)"`
Expected: 输出 emoji 列表

- [ ] **Step 3: 提交**

```bash
git add src/label_ui.py
git commit -m "feat: add color emoji labels to candidate buttons"
```

---

### Task 3: 标注后展示结果图 + 确认/重新选择 + 完成引导按钮

**Files:**
- Modify: `src/label_ui.py`

**Context:** 这是最大的改动，包含 3 个子功能：

1. 选择候选区域后展示绿色轮廓结果图
2. 显示"确认，下一张"和"重新选择"按钮
3. 全部完成后显示"📦 导出数据集"引导按钮

**输出数量统一**：当前 `step3_outputs` 有 15 个值。本 Task 新增 3 个输出（`confirm_btn`、`reselect_btn`、`complete_export_btn`），总计 18 个值。所有返回 step3_outputs 的函数必须返回 18 个值。

输出顺序：`(review_image, annotate_info, annotate_progress, annotate_current_idx, *cand_buttons[10], thumbnail_strip, confirm_btn, reselect_btn, complete_export_btn)`

- [ ] **Step 1: 添加 import、状态变量、常量**

**1a.** 修改 import（第 14-19 行），添加 `draw_result`：

```python
from src.label_service import (
    draw_candidates,
    draw_result,
    generate_dataset_yaml,
    save_label,
    segment_image,
)
```

**1b.** 在 `DATASET_DIR` 定义后（约第 27 行后），添加：

```python
# Emojis matching COLORS order in label_service.py
COLOR_EMOJIS = ["🟢", "🔴", "🔵", "🟡", "🟦", "🟣", "🟩", "🔶", "🟠", "🟤"]
```

注意：如果 Task 2 已经添加了 `COLOR_EMOJIS`，跳过此步。

**1c.** 在 `_annotation_state` 定义后（约第 139 行后），添加：

```python
_pending_selection: dict[str, int] = {}
```

- [ ] **Step 2: 定义辅助函数 `_make_step3_output()` 统一输出格式**

为了避免每个函数都手写 18 个值的 tuple，在 `_build_thumbnails_html()` 函数之后（约第 184 行后）添加辅助函数：

```python
def _step3(image, info, progress, index, btn_updates, thumbnails,
           confirm=False, reselect=False, export=False):
    """Build a flat 18-element tuple for step3_outputs.

    btn_updates: list of 10 gr.update values for candidate buttons.
    confirm/reselect/export: booleans controlling visibility of action buttons.
    """
    return (image, info, progress, index, *btn_updates, thumbnails,
            gr.update(visible=confirm),
            gr.update(visible=reselect),
            gr.update(visible=export))
```

- [ ] **Step 3: 重写 `_load_review_image()` 使用 `_step3()`**

将整个 `_load_review_image()` 函数（约第 190-222 行）替换为：

```python
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
```

- [ ] **Step 4: 替换 `_handle_select_candidate()` — 选择后展示预览**

将现有的 `_handle_select_candidate()` 函数（约第 225-259 行）替换为：

```python
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
```

- [ ] **Step 5: 新增 `_handle_confirm()` 函数**

在 `_handle_select_candidate()` 之后添加：

```python
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
```

- [ ] **Step 6: 新增 `_handle_reselect()` 函数**

在 `_handle_confirm()` 之后添加：

```python
def _handle_reselect(img_index: int):
    """Cancel pending selection and return to candidate view."""
    photos = _list_photos()
    if not photos or img_index >= len(photos):
        return _load_review_image(0)

    img_path = photos[img_index]
    _pending_selection.pop(img_path, None)

    return _load_review_image(img_index)
```

- [ ] **Step 7: 重写 `_handle_skip_photo()` 使用 `_step3()`**

将现有的 `_handle_skip_photo()` 函数（约第 262-278 行）替换为：

```python
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
```

- [ ] **Step 8: 在 UI 构建中添加新按钮并更新 wiring**

在 `build_ui()` 的 Tab 3（③ 标注确认）区域：

**8a.** 在 `skip_btn` 定义之后（约第 398 行 `skip_btn = gr.Button("⏭️ 跳过这张", ...)` 后），添加：

```python
                        confirm_btn = gr.Button("✅ 确认，下一张", variant="primary", visible=False)
                        reselect_btn = gr.Button("↩️ 重新选择", variant="secondary", visible=False)
```

**8b.** 在 `init_annotate_btn` 之后（约第 402 行 `init_annotate_btn = gr.Button(...)` 后），添加：

```python
                complete_export_btn = gr.Button("📦 导出数据集", variant="primary", visible=False)
```

**8c.** 更新 `step3_outputs` 列表（约第 414 行）：

```python
        step3_outputs = [review_image, annotate_info, annotate_progress,
                         annotate_current_idx] + cand_buttons + [thumbnail_strip,
                         confirm_btn, reselect_btn, complete_export_btn]
```

**8d.** 在 wiring 区域的 `skip_btn.click` 之后（约第 452 行后），添加：

```python
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
```

- [ ] **Step 9: 运行全部测试确认无破坏**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: 所有测试 PASS

- [ ] **Step 10: 提交**

```bash
git add src/label_ui.py
git commit -m "feat: add result preview, confirm/reselect flow, and export guidance button"
```
