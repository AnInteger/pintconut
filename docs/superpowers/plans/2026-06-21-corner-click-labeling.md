# 角点定位标注 实施 Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把标注 UI 的「位置生成」从失败的 HoughCircles 预标注换成角点定位——点 4 颗角豆 + 板尺寸 → 单应矩阵生成全部豆位 → 校正 → 导出 YOLO。

**Architecture:** 服务层加两个纯函数（`generate_grid_boxes` 几何、`preview_box_colors` 配色）；UI 重写为角点定位模式（canvas 点 4 角 + 板尺寸 + 生成），复用现有导出/编辑/稳定 id。几何已验证：4 点单应对平面板透视精确（合成网格还原误差 0px）。

**Tech Stack:** Python 3.10+, OpenCV (cv2 getPerspectiveTransform), NumPy, Gradio, pytest（`tests/_synth.py` 合成夹具）。

**Spec:** `docs/superpowers/specs/2026-06-21-corner-click-labeling-design.md`

---

## 文件结构

| 文件 | 责任 | 动作 |
|---|---|---|
| `src/bead_label_service.py` | 标注纯逻辑 | 加 `generate_grid_boxes`、`preview_box_colors` |
| `src/bead_annotate_ui.py` | Gradio 薄界面 | 重写为角点定位模式（移除 HoughCircles 预标注） |
| `tests/test_bead_label_service.py` | 服务层单测 | 加 2 个几何/配色测试 |

`src/bead_grid.py`（HoughCircles/detect_beads/fit）**不动**——CLI 运行时仍用，只是标注 UI 不再调用。

---

## Task 1: `generate_grid_boxes`（4 角豆 → 全部豆位框）

**Files:**
- Modify: `src/bead_label_service.py`（追加函数）
- Test: `tests/test_bead_label_service.py`

- [ ] **Step 1: 写失败测试**（追加到 `tests/test_bead_label_service.py` 末尾）

```python
def test_generate_grid_boxes_recovers_perspective_grid():
    from tests._synth import synth_grid_centers, apply_homography
    from src.bead_label_service import generate_grid_boxes
    rows, cols = 10, 10
    centers = synth_grid_centers(rows, cols, spacing=30.0, origin=(40, 40))
    H = np.array([[1.2, 0.1, 10], [0.05, 1.1, 5], [0.0003, 0.0002, 1.0]], dtype=np.float64)
    warped = apply_homography(centers, H)  # ground-truth bead positions under perspective
    # 4 corner beads: TL(0,0), TR(0,cols-1), BR(rows-1,cols-1), BL(rows-1,0)
    corners = np.array([warped[0], warped[cols - 1], warped[rows * cols - 1],
                        warped[(rows - 1) * cols]], dtype=np.float32)
    boxes = generate_grid_boxes(corners, rows, cols)
    assert len(boxes) == rows * cols
    assert all(b["source"] == "generated" for b in boxes)
    # row-major order matches synth_grid_centers → compare centers directly
    gen_xy = np.array([[b["cx"], b["cy"]] for b in boxes])
    err = np.linalg.norm(gen_xy - warped, axis=1)
    assert err.max() < 1.0   # 4-point homography is exact for planar perspective
    # box width ~ local spacing(≈30) * 0.4 * 2 ≈ 24
    widths = [b["width"] for b in boxes]
    assert 10 < np.median(widths) < 40
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_bead_label_service.py::test_generate_grid_boxes_recovers_perspective_grid -v`
Expected: FAIL（`ImportError: cannot import name 'generate_grid_boxes'`）

- [ ] **Step 3: 实现**（追加到 `src/bead_label_service.py` 末尾）

```python
def generate_grid_boxes(corners, rows: int, cols: int) -> list[dict]:
    """4 corner-bead image positions (TL,TR,BR,BL) + dims -> all bead boxes via homography.

    Each box is sized to its local spacing (mean distance to existing neighbors) * 0.4,
    so box size adapts under perspective. Raises cv2.error if the 4 corners are degenerate.
    """
    corners = np.asarray(corners, dtype=np.float32)
    src = np.array([[0, 0], [cols - 1, 0], [cols - 1, rows - 1], [0, rows - 1]],
                   dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, corners)  # maps grid (c,r) -> image (x,y)
    centers = {}
    for r in range(rows):
        for c in range(cols):
            v = H @ np.array([c, r, 1.0])
            centers[(r, c)] = v[:2] / v[2]
    boxes = []
    for r in range(rows):
        for c in range(cols):
            xy = centers[(r, c)]
            dists = []
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nb = centers.get((r + dr, c + dc))
                if nb is not None:
                    dists.append(float(np.linalg.norm(nb - xy)))
            spacing = float(np.mean(dists)) if dists else 10.0
            boxes.append(_box_from_xy(xy, spacing * 0.4, "generated"))
    return boxes
```

（`cv2` 与 `_box_from_xy` 已在该模块顶部/前面可用——Task 3/4 已 `import cv2` 并定义 `_box_from_xy`。）

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_bead_label_service.py -v`
Expected: PASS（含新测试；原有不回归）

- [ ] **Step 5: 提交**

```bash
git add src/bead_label_service.py tests/test_bead_label_service.py
git commit -m "feat(bead_label_service): generate_grid_boxes via 4-corner homography"
```

---

## Task 2: `preview_box_colors`（每框配色预览）

**Files:**
- Modify: `src/bead_label_service.py`（追加函数）
- Test: `tests/test_bead_label_service.py`

- [ ] **Step 1: 写失败测试**（追加到 `tests/test_bead_label_service.py` 末尾）

```python
def test_preview_box_colors_returns_palette_entries():
    from src.bead_label_service import preview_box_colors
    from src.color import ColorMatcher
    img = render_beads(synth_grid_centers(3, 3, spacing=40.0, origin=(40, 40)),
                       img_size=(200, 200), bead_radius=12, body_color=(0, 0, 255))  # BGR 红
    boxes = [{"cx": 40, "cy": 40, "xyxy": [28, 28, 52, 52],
              "width": 24, "height": 24, "source": "generated", "id": 0}]
    out = preview_box_colors(img, boxes, ColorMatcher())
    assert len(out) == 1
    assert {"xy", "name", "rgb"} <= set(out[0].keys())
    assert isinstance(out[0]["name"], str)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_bead_label_service.py::test_preview_box_colors_returns_palette_entries -v`
Expected: FAIL（`ImportError: cannot import name 'preview_box_colors'`）

- [ ] **Step 3: 实现**（追加到 `src/bead_label_service.py` 末尾）

```python
def preview_box_colors(image, boxes, color_matcher) -> list[dict]:
    """Sample each box center's color and match to palette (for eval-preview overlay)."""
    from .bead_grid import _sample_color
    out = []
    for b in boxes:
        color = _sample_color(image, (b["cx"], b["cy"]))
        m = color_matcher.match(color.tolist())
        out.append({"xy": (float(b["cx"]), float(b["cy"])),
                    "name": m["name"], "rgb": m["rgb"]})
    return out
```

（`_sample_color` 是 `src/bead_grid.py` 的模块级函数，返回 RGB uint8 ndarray；`ColorMatcher.match(rgb_list)` 返回含 `name`/`rgb` 的色板条目。局部 import 避免改顶部导入。）

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_bead_label_service.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/bead_label_service.py tests/test_bead_label_service.py
git commit -m "feat(bead_label_service): preview_box_colors palette overlay"
```

---

## Task 3: 重写标注 UI 为角点定位模式

**Files:**
- Rewrite: `src/bead_annotate_ui.py`（整体替换）

- [ ] **Step 1: 整体替换 `src/bead_annotate_ui.py`** 为以下内容：

```python
"""Gradio UI for bead annotation — corner-click grid generation + box edit + export.

Thin UI over src.bead_label_service. User clicks 4 corner beads + picks board size ->
geometry generates all bead boxes -> correct -> export YOLO labels.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import gradio as gr
import numpy as np

from src.bead_label_service import generate_grid_boxes, preview_box_colors, export_yolo
from src.color import ColorMatcher

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHOTOS_DIR = os.path.join(BASE_DIR, "training", "photos")
DATASET_DIR = os.path.join(BASE_DIR, "training", "bead_dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images", "train")
LABELS_DIR = os.path.join(DATASET_DIR, "labels", "train")
BOARD_SIZES_PATH = os.path.join(BASE_DIR, "data", "board_sizes.json")


def _load_presets():
    try:
        with open(BOARD_SIZES_PATH) as f:
            data = json.load(f)
        labels = [f"{p['name']} ({p['rows']}×{p['cols']})" for p in data]
        return labels, data
    except Exception:
        return [], []


PRESET_CHOICES, PRESET_DATA = _load_presets()


def _parse_preset(label):
    for p in PRESET_DATA:
        if f"{p['name']} ({p['rows']}×{p['cols']})" == label:
            return p["rows"], p["cols"]
    return None


_cm = None


def _color_matcher():
    global _cm
    if _cm is None:
        _cm = ColorMatcher()
    return _cm


# box source -> BGR draw color
SRC_BGR = {"generated": (0, 255, 0), "manual": (255, 0, 0)}


def _list_photos():
    if not os.path.isdir(PHOTOS_DIR):
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(f for f in os.listdir(PHOTOS_DIR) if os.path.splitext(f)[1].lower() in exts)


def _stats():
    n = len(os.listdir(LABELS_DIR)) if os.path.isdir(LABELS_DIR) else 0
    return f"数据集: {n} 张已标注 / {len(_list_photos())} 张照片"


def _box_label(b):
    return f"#{b['id']} {b['source']} @({b['cx']},{b['cy']})"


def _choices(state):
    return [_box_label(b) for b in state["boxes"]]


def _new_state(img_bgr=None, name="", rows=29, cols=29):
    return {"img_bgr": img_bgr, "boxes": [], "corners": [], "rows": rows, "cols": cols,
            "name": name, "next_id": 0}


def _stamp_ids(next_id: int, boxes: list[dict]) -> tuple[list[dict], int]:
    out = []
    for b in boxes:
        nb = dict(b)
        nb["id"] = next_id
        next_id += 1
        out.append(nb)
    return out, next_id


def _draw(state, show_color):
    img = state.get("img_bgr")
    if img is None:
        return None
    disp = img.copy()
    if show_color and state["boxes"]:
        for cc in preview_box_colors(img, state["boxes"], _color_matcher()):
            x, y = cc["xy"]
            bgr = [int(v) for v in reversed(cc["rgb"])]
            cv2.circle(disp, (int(x), int(y)), 4, bgr, -1)
    corners = state.get("corners", [])
    if len(corners) >= 2:
        cv2.polylines(disp, [np.array(corners, dtype=np.int32)],
                      len(corners) >= 4, (0, 0, 255), 2)
    for i, (x, y) in enumerate(corners):
        cv2.circle(disp, (int(x), int(y)), 6, (0, 0, 255), -1)
        cv2.putText(disp, str(i + 1), (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    for b in state["boxes"]:
        x1, y1, x2, y2 = b["xyxy"]
        cv2.rectangle(disp, (x1, y1), (x2, y2), SRC_BGR.get(b["source"], (255, 255, 255)), 2)
    return cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)


def h_load(name, state):
    path = os.path.join(PHOTOS_DIR, name or "")
    img = cv2.imread(path) if name else None
    if img is None:
        return None, _new_state(), f"❌ 读不到 {name}", gr.update(), _stats()
    state = _new_state(img, os.path.splitext(name)[0])
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), state, f"✅ 已加载 {name}", gr.update(choices=_choices(state)), _stats()


def h_preset(label):
    rc = _parse_preset(label or "")
    if rc:
        return rc[0], rc[1]
    return gr.update(), gr.update()


def h_reset(state, show_color):
    state = {**state, "corners": [], "boxes": []}
    return _draw(state, show_color), state, gr.update(choices=_choices(state)), "已重置角点与框"


def h_click(evt, state, mode, radius, show_color):
    img = state.get("img_bgr")
    if img is None or evt is None:
        return _draw(state, show_color), state, gr.update(), ""
    x, y = evt.index
    if mode == "角点":
        corners = list(state.get("corners", []))
        if len(corners) < 4:
            corners.append((float(x), float(y)))
            state = {**state, "corners": corners}
        return _draw(state, show_color), state, gr.update(), f"已点 {len(state['corners'])}/4 角豆"
    cx, cy = int(round(x)), int(round(y))
    r = int(radius)
    manual = [{"xyxy": [cx - r, cy - r, cx + r, cy + r],
              "cx": cx, "cy": cy, "width": 2 * r, "height": 2 * r, "source": "manual"}]
    nb, nid = _stamp_ids(state.get("next_id", 0), manual)
    state = {**state, "boxes": state["boxes"] + nb, "next_id": nid}
    return _draw(state, show_color), state, gr.update(choices=_choices(state)), ""


def h_generate(state, rows, cols, show_color):
    corners = state.get("corners", [])
    rows_i, cols_i = int(rows or 0), int(cols or 0)
    if len(corners) != 4:
        return _draw(state, show_color), state, gr.update(), "❌ 请先在「角点」模式点满 4 颗角豆"
    if rows_i < 2 or cols_i < 2:
        return _draw(state, show_color), state, gr.update(), "❌ 板尺寸太小（rows/cols ≥ 2）"
    try:
        boxes = generate_grid_boxes(np.array(corners, dtype=np.float32), rows_i, cols_i)
    except cv2.error:
        return _draw(state, show_color), state, gr.update(), "❌ 4 颗角豆共线，请重置后重新点"
    nb, nid = _stamp_ids(state.get("next_id", 0), boxes)
    state = {**state, "boxes": nb, "next_id": nid, "rows": rows_i, "cols": cols_i}
    return _draw(state, show_color), state, gr.update(choices=_choices(state)), f"✅ 生成 {len(nb)} 颗豆位"


def h_delete(state, sel, show_color):
    sel = sel or []
    keep = [b for b in state["boxes"] if _box_label(b) not in sel]
    state = {**state, "boxes": keep}
    return _draw(state, show_color), state, gr.update(choices=_choices(state)), f"剩余 {len(keep)} 个框"


def h_export(state, name_override):
    img = state.get("img_bgr")
    if img is None:
        return "❌ 未加载照片", _stats()
    if not state["boxes"]:
        return "❌ 无框可导出（先生成网格或加框）", _stats()
    name = (name_override or state.get("name") or "").strip()
    if not name:
        return "❌ 缺导出名", _stats()
    img_path, lbl_path, n = export_yolo(img, state["boxes"], name, IMAGES_DIR, LABELS_DIR)
    return f"✅ 导出 {n} 框 → {lbl_path}", _stats()


def build_ui():
    with gr.Blocks(title="Pintconut 珠子标注") as app:
        gr.Markdown("# 🟡 Pintconut 珠子标注 (角点定位)\n"
                    "加载 → 「角点」模式点 4 颗角豆(左上→右上→右下→左下) → 选板尺寸 → 生成网格 → "
                    "「加框」模式校正 → 导出。框色：🟢生成 / 🔵手动。")
        state = gr.State(_new_state())

        with gr.Row():
            with gr.Column(scale=1):
                photo = gr.Dropdown(choices=_list_photos(), label="照片 (training/photos/)")
                load_btn = gr.Button("📥 加载")
                preset_dd = gr.Dropdown(choices=PRESET_CHOICES, label="板尺寸预设 (选则自动填下方)",
                                        value=PRESET_CHOICES[2] if len(PRESET_CHOICES) > 2 else None)
                with gr.Row():
                    rows_num = gr.Number(value=29, label="rows", precision=0)
                    cols_num = gr.Number(value=29, label="cols", precision=0)
                mode = gr.Radio(["角点", "加框"], value="角点", label="点击模式")
                show_color = gr.Checkbox(value=True, label="叠加匹配色(评估)")
                generate_btn = gr.Button("🎯 生成网格", variant="primary")
                reset_btn = gr.Button("♻️ 重置角点/框")
                box_list = gr.CheckboxGroup(choices=[], label="框列表 (勾选→删除)")
                delete_btn = gr.Button("🗑️ 删除选中")
                radius = gr.Number(value=10, label="加框半径(px, 仅加框模式)")
                name_tb = gr.Textbox(label="导出名 (留空用照片名)")
                export_btn = gr.Button("💾 导出 YOLO", variant="secondary")
            with gr.Column(scale=1):
                canvas = gr.Image(label="标注画布 (角点模式=点角豆 / 加框模式=加框)", type="numpy")
                status = gr.Textbox(label="状态", interactive=False)
                stats = gr.Textbox(label="统计", interactive=False)

        # 先声明组件，再统一接线
        load_btn.click(h_load, [photo, state], [canvas, state, status, box_list, stats])
        preset_dd.change(h_preset, [preset_dd], [rows_num, cols_num])
        generate_btn.click(h_generate, [state, rows_num, cols_num, show_color],
                           [canvas, state, box_list, status])
        reset_btn.click(h_reset, [state, show_color], [canvas, state, box_list, status])
        delete_btn.click(h_delete, [state, box_list, show_color], [canvas, state, box_list, status])
        canvas.select(h_click, [state, mode, radius, show_color], [canvas, state, box_list, status])
        export_btn.click(h_export, [state, name_tb], [status, stats])
        app.load(fn=lambda: gr.update(choices=_list_photos()), outputs=photo)
        app.load(fn=_stats, outputs=stats)
    return app


if __name__ == "__main__":
    build_ui().launch(share=False)
```

- [ ] **Step 2: 服务层测试不回归**

Run: `.venv/bin/python -m pytest tests/test_bead_label_service.py -v`
Expected: PASS（UI 重写不影响服务层）

- [ ] **Step 3: UI 可构建 + 可启动**（子代理无浏览器，验证 import/构造/起服务；交互式评估留给用户）

- 验证 import + 构造：
  Run: `.venv/bin/python -c "import sys; sys.path.insert(0,'.'); from src.bead_annotate_ui import build_ui, PRESET_CHOICES; app=build_ui(); print('build_ui OK:', type(app).__name__, '| presets:', len(PRESET_CHOICES))"`
  Expected: 打印 `build_ui OK: Blocks | presets: 6`，无异常。
- 验证服务器能起（后台起、探测、杀掉）：
  Run: `GRADIO_SERVER_PORT=7862 GRADIO_SERVER_NAME=127.0.0.1 .venv/bin/python src/bead_annotate_ui.py &` → sleep 几秒 → `curl -sf http://127.0.0.1:7862/ -o /dev/null && echo SERVER_OK` → `kill` 进程。
  Expected: `SERVER_OK`，且进程已 kill（不留后台服务）。
- 若 Gradio 6.x API 报错（如 `canvas.select` 的 `evt.index`、`gr.Radio` 值匹配），按实际 API 修正到能 import + 起服务，并在报告说明。`canvas.select` 可能仍报「Expected N arguments, received M」的 cosmetic warning（Gradio 内省未计隐式 `SelectData` 参数），运行时正常，可忽略。

- [ ] **Step 4: 提交**

```bash
git add src/bead_annotate_ui.py
git commit -m "feat(bead_annotate_ui): corner-click grid generation (replace HoughCircles prelabel)"
```

---

## Self-Review

**1. Spec 覆盖**
- §3 流程（加载/点4角/板尺寸/生成/校正/导出）→ Task 3 ✓
- §4 几何（4 角豆→getPerspectiveTransform→全部豆位，局部间距框）→ Task 1 ✓（已验证 0px）
- §5.1 `generate_grid_boxes` / `preview_box_colors` → Task 1/2 ✓
- §5.2 角点模式 + 板尺寸下拉/手填 + 移除 HoughCircles 预标注 + 配色预览改用 boxes → Task 3 ✓
- §5.3 bead_grid 不动 → 计划声明不动 ✓
- §6 错误处理（<4 角、尺寸非法、4 角共线 cv2.error）→ Task 3 `h_generate` ✓
- §7 测试（generate 精确还原 + 数量/source、preview 结构）→ Task 1/2 ✓

**2. 占位符扫描**：无 TBD/TODO；每步含完整代码与命令。

**3. 类型一致性**：box schema（`xyxy/cx/cy/width/height/source/id`）跨 Task 1/2/3 一致；`generate_grid_boxes` 用 `_box_from_xy(.., "generated")`，UI `SRC_BGR` 含 `"generated"`/`"manual"`；state 由 `{"img_bgr","boxes","corners","rows","cols","name","next_id"}` 统一，所有 handler 非变异（`{**state,...}`）；`preview_box_colors` 返回 `{xy,name,rgb}` 与 `_draw` 消费一致。

**范围声明**：本计划 = 角点定位标注（Task 1–3）。训练、`bead_split.py`、CLI 接检测器仍是后续计划（依赖本计划产出的标签）。
